from datetime import timedelta
from pathlib import Path
import shutil
import zipfile

import cartopy.crs as ccrs
import earthaccess
from earthaccess.results import DataGranule
import gdown
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import box
from sklearn.model_selection import train_test_split

QUITE = True
CHIP_SIZE = 256
RNG_SEED = 42


def main():
    print(f'Chipping Data for {MODALITY}')
    hwds_path = Path("hwds")
    modality_path = hwds_path / MODALITY

    data_paths = {
        MODALITY: hwds_path / MODALITY,
        "RAW": modality_path / "RAW",
        "WGS84": modality_path / "WGS84",
        "MERGE": modality_path / "MERGE",
        "CHIPS": modality_path / "CHIPS",
        "CHIPS_TM": modality_path / "CHIPS_TM",
        "PLOTS": modality_path / "PLOTS",
        "SPLITS": modality_path / "SPLITS",
    }

    for p in ("CHIPS", "CHIPS_TM", "SPLITS", "MERGE", "PLOTS", "SPLITS"):
        shutil.rmtree(data_paths[p], ignore_errors=True)

    for p in data_paths.values():
        p.mkdir(parents=True, exist_ok=True)

    gdf = _load_event_database(hwds_path)

    # keepers = [1442, 622, 1079, 628]
    # keepers = [1442, 622]
    # gdf = gdf[gdf["swathID"].isin(keepers)]

    earthaccess.login()

    tm_chips = []

    for i, (swathID, swath) in enumerate(gdf.iterrows(), start=1):
        print(f'Processing Swath {swathID} ({i} / {len(gdf)})')

        results = search_data(swath)

        local_files = earthaccess.download(
            results, local_path=data_paths["RAW"], show_progress=True
        )

        data_tifs = [f for f in local_files if f.name.endswith(".tif")]

        if len(data_tifs) == 0:
            print(f"Skipping: no data for swath {swath['swathID']}")
            continue

        merged_data = _get_merged_swath_data(data_tifs, swath, data_paths)

        if merged_data is None or not is_valid_data(merged_data):
            print("Skipping: not enough valid data")
            continue

        print("Chipping!")
        merged_data = _stack_bands(merged_data, data_bands=STACK_BANDS)
        chips = _chip_data(merged_data, data_paths)

        good_chips = filter_chips(chips)
        print(f"Found {len(good_chips)} good chips")

        for chip in good_chips:
            for band, chip_path in chip.items():
                if band not in ("MASK", "BANDS"):
                    continue

                dest = data_paths["CHIPS_TM"] / chip_path.name
                shutil.copy(chip_path, dest)

        tm_chips += good_chips

    for _, swath in gdf.iterrows():
        swath_id = _make_swath_id(swath["swathID"])

        merged_file = list(data_paths["MERGE"].glob(f"{swath_id}.*BANDS.tif"))
        if len(merged_file) == 0:
            print(f"no chips for {swath_id}")
            continue

        all_chips = list(data_paths["CHIPS"].glob(f"*.{swath_id}.*.tif"))
        good_chips = list(data_paths["CHIPS_TM"].glob(f"*.{swath_id}.*.tif"))

        print(f"plotting {swath_id}")
        _plot_chips(
            merged_file[0], all_chips, good_chips, swath, save_to=data_paths["PLOTS"]
        )

    band_chips = list(data_paths["CHIPS_TM"].glob("*BANDS.tif"))

    create_split_files(band_chips, splits_path=data_paths["SPLITS"])
    means, stds = calculate_stats(chips=band_chips)

    means_str = ", ".join(f"{x:.4f}" for x in means)
    stds_str = ", ".join(f"{x:.4f}" for x in stds)

    stats_str = (
        f"Means {STACK_BANDS}: {means_str}\n"
        f"Stds {STACK_BANDS}: {stds_str}\n"
    )
    (modality_path / 'statistics.txt').write_text(stats_str)
    print(stats_str)


def _load_event_database(data_dir: Path):
    # use 60-swath version
    hwds_google_drive_id = "1h_JIEcrrUF3OSTrmwAKNPa0eUEhPA2Xx"
    drive_url = f"https://drive.google.com/uc?id={hwds_google_drive_id}"

    shp_dir = data_dir / "SHP"
    shp_dir.mkdir(parents=True, exist_ok=True)

    filename = "hwds_v3_20250205_subset_60.zip"

    zip_path = shp_dir / filename

    if not zip_path.exists():
        gdown.download(drive_url, str(zip_path), quiet=False)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(path=shp_dir)

    shp_path = shp_dir / "hwds_v3_20250205_subset_60.shp"
    gdf = gpd.read_file(shp_path)

    gdf["swathDate"] = pd.to_datetime(gdf["swathDate"], format="%Y-%m-%d")
    gdf["ls5hlsDate"] = pd.to_datetime(gdf["ls5hlsDate"], format="%Y-%m-%d")
    gdf["s1Date"] = pd.to_datetime(gdf["s1Date"], format="%Y-%m-%d")

    return _add_buffered(gdf)


def _add_buffered(gdf):
    # make some additional columns that represent buffers after projecting to UTM 15N
    gdf = gdf.to_crs(32615)
    buffered_event = gdf.buffer(3000)
    buffered_event_background = gdf.buffer(10000)
    gdf = gdf.to_crs(4326)

    gdf["buffered_event"] = buffered_event
    gdf["buffered_event_background"] = buffered_event_background
    gdf["buffered_event"] = gdf["buffered_event"].to_crs("EPSG:4326")
    gdf["buffered_event_background"] = gdf["buffered_event_background"].to_crs(
        "EPSG:4326"
    )

    return gdf


def create_split_files(band_chips: list[Path], splits_path: Path) -> None:
    chip_ids = [p.name.removesuffix("BANDS.tif") for p in band_chips]

    the_rest, test = train_test_split(chip_ids, test_size=0.15, random_state=RNG_SEED)
    train, val = train_test_split(the_rest, test_size=0.15, random_state=RNG_SEED)

    splits = {"train": train, "val": val, "test": test}

    for split, chip_ids in splits.items():
        split_path = splits_path / f"{split}.txt"
        split_path.write_text("\n".join(chip_ids))


def calculate_stats(chips: list[Path], n_bands: int = 2) -> tuple:
    mean = np.zeros(n_bands, dtype=np.float64)
    M2 = np.zeros(n_bands, dtype=np.float64)
    count = np.zeros(n_bands, dtype=np.float64)

    for chip in chips:
        with rasterio.open(chip) as src:
            band_data = src.read()
            count, mean, M2 = 0, 0, 0

            _, H, W = band_data.shape

            batch_count = H * W
            batch_mean = band_data.mean(axis=(1, 2))
            batch_var = band_data.var(axis=(1, 2))

            delta = batch_mean - mean
            total_count = count + batch_count

            mean = mean + delta * (batch_count / total_count)
            M2 = (
                M2
                + batch_var * batch_count
                + (delta**2) * count * batch_count / total_count
            )
            count = total_count

    variance = M2 / count
    std = np.sqrt(variance)

    return mean, std


def _plot_chips(
    merged_band_file,
    all_chips,
    good_chips,
    swath,
    save_to: Path | None = None,
    quite=QUITE,
):
    crs_pc = ccrs.PlateCarree()

    with rasterio.open(merged_band_file) as ds:
        bounds = ds.bounds
        full_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        band_data = ds.read()

    img = get_img(band_data)

    # plot BANDS and geom
    fig, ax = plt.subplots(
        1,
        1,
        subplot_kw={"projection": crs_pc},
        figsize=(12, 12),
        layout="constrained",
    )

    swath_geom = swath["geometry"]

    ax.imshow(img, extent=full_extent, origin="upper", transform=crs_pc)
    ax.add_geometries(
        [swath_geom], edgecolor="red", linewidth=2, facecolor="none", crs=crs_pc
    )

    def show_chips(chips, color, linewidth, z):
        for chip in chips:
            with rasterio.open(chip) as ds:
                chip_bounds = ds.bounds
                chip_geom = box(
                    chip_bounds.left,
                    chip_bounds.bottom,
                    chip_bounds.right,
                    chip_bounds.top,
                )

            ax.add_geometries(
                [chip_geom],
                edgecolor=color,
                linewidth=linewidth,
                alpha=1,
                zorder=z,
                facecolor="none",
                crs=crs_pc,
            )

    show_chips(all_chips, "yellow", 1, z=1)
    show_chips(good_chips, "blue", 3, z=2)

    ax.set_extent(full_extent, crs=crs_pc)

    if save_to:
        plt.savefig(
            save_to / f"{merged_band_file.name.removesuffix('BANDS.tif')}.png",
            dpi=300,
            bbox_inches="tight",
        )

    if not quite:
        plt.show()

    plt.close(fig)


def _make_swath_id(swathID):
    return f"{int(swathID):04d}"


def _get_merged_swath_data(data_tifs: list[Path], swath: pd.Series, data_paths: dict) -> dict[str, Path]:
    swathID = f"{int(swath['swathID']):04d}"

    reprojected_tifs = _reproject_files(
        data_tifs, output_path=data_paths["WGS84"], prefix=f"{swathID}."
    )

    merged = {}

    for band in ALL_BANDS:
        band_files = [f for f in reprojected_tifs if band in band_from_filename(f.name)]

        merged_name = make_merge_name(band_files[0].name)
        merged_band_path = _merge(
            band_files, output_file=data_paths["MERGE"] / merged_name
        )

        merged[band] = merged_band_path

    band = ALL_BANDS[0]
    event_tif, mask_tif = _generate_masks(
        merged[band], swath, merged_extension=f"{band}.tif"
    )

    return {
        **merged,
        "EVENT": event_tif,
        "MASK": mask_tif,
    }


def _reproject_files(
    granule_files: list[Path], output_path: Path, prefix: str
) -> list[Path]:

    reprojected_paths = [
        Path(output_path) / f"{prefix}{granule.name}" for granule in granule_files
    ]

    for granule, output_path in zip(granule_files, reprojected_paths):
        if output_path.exists():
            continue

        print(f"reprojecting to wgs84: {output_path.name}")
        _reproject_file(granule, output_path)

    return reprojected_paths


def _reproject_file(local_file: Path, reprojected_file: Path, epsg=4326) -> None:
    # https://rasterio.readthedocs.io/en/stable/topics/reproject.html#reprojecting-a-geotiff-dataset
    with rasterio.open(local_file) as src:
        dst_crs = CRS.from_epsg(epsg)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(reprojected_file, "w", **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                )


def _merge(band_files: list[Path], output_file: Path) -> Path:
    band_datasets = [rasterio.open(rtc_tif) for rtc_tif in band_files]

    try:
        mosaic, out_trans = merge(band_datasets)
        mosaic = np.squeeze(mosaic)

        out_meta = band_datasets[0].meta.copy()

        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[0],
                "width": mosaic.shape[1],
                "transform": out_trans,
                "crs": band_datasets[0].crs,
            }
        )

        with rasterio.open(output_file, "w", **out_meta) as dst:
            dst.write(mosaic, 1)
    finally:
        for ds in band_datasets:
            ds.close()

    return output_file


def _rename(path: Path, extension: str, mask_name: str) -> Path:
    return path.parent / path.name.replace(extension, mask_name)


def _generate_masks(
    merged_path: Path, swath: pd.Series, merged_extension: str
) -> tuple[Path, Path]:

    event_path = _rename(merged_path, merged_extension, "EVENT.tif")
    mask_path = _rename(merged_path, merged_extension, "MASK.tif")

    with rasterio.open(merged_path) as ds:
        profile = ds.profile

        mask_raster = features.rasterize(
            shapes=[[swath["geometry"], 1]],
            fill=0,
            out_shape=ds.shape,
            transform=ds.transform,
        )

        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask_raster, 1)
            print("generated:", mask_path)

        event_mask = features.rasterize(
            shapes=[
                [swath["buffered_event_background"], 3],
                [swath["buffered_event"], 2],
                [swath["geometry"], 1],
            ],
            fill=0,
            out_shape=ds.shape,
            transform=ds.transform,
        )

        with rasterio.open(event_path, "w", **profile) as dst:
            dst.write(event_mask, 1)
            print("generated:", event_path)

    return event_path, mask_path


def _stack_bands(merged: dict[str, Path], data_bands: tuple[str]) -> None:
    with rasterio.open(merged[data_bands[0]]) as src:
        meta = src.meta.copy()

    band = data_bands[0]
    meta.update(count=len(data_bands), dtype=np.float32)
    stacked_file_name = _rename(merged[band], f"{band}.tif", "BANDS.tif")

    with rasterio.open(stacked_file_name, "w", **meta) as dst:
        for idx, band in enumerate(data_bands, start=1):
            with rasterio.open(merged[band]) as src:
                dst.write(src.read(1), idx)

    merged["BANDS"] = stacked_file_name

    return merged


def _chip_data(merged: dict[str, Path], data_paths: dict[str, Path], chip_size=CHIP_SIZE):
    chips = {}

    grid = []
    with rasterio.open(merged["BANDS"]) as ref:
        n_cols = ref.width // chip_size
        n_rows = ref.height // chip_size

        for row in range(n_rows):
            for col in range(n_cols):
                window = Window(col * chip_size, row * chip_size, chip_size, chip_size)
                bounds = ref.window_bounds(window)

                tile_id = f"{row:03d}.{col:03d}"
                chips[tile_id] = {}
                grid.append((tile_id, bounds))

    for chip_layer in CHIP_BANDS:
        layer_path = merged[chip_layer]

        with rasterio.open(layer_path) as src:
            for tile_id, bounds in grid:
                window = src.window(*bounds)
                window = Window(
                    round(window.col_off),
                    round(window.row_off),
                    round(window.width),
                    round(window.height),
                )

                data = src.read(window=window)

                chip_meta = src.meta.copy()
                chip_meta.update(
                    {
                        "width": window.width,
                        "height": window.height,
                        "transform": src.window_transform(window),
                    }
                )

                chip_name = f"{tile_id}.{layer_path.name}"
                chip_path = data_paths["CHIPS"] / chip_name

                with rasterio.open(chip_path, "w", **chip_meta) as dst:
                    dst.write(data)

                chips[tile_id][chip_layer] = chip_path

    return chips


def search_data(swath):
    if MODALITY == 'HLS':
        results = search_hls_data(swath)
    elif MODALITY == 'RTC':
        results = search_rtc_data(swath)

    return results


def is_valid_data(merged_data):
    if MODALITY == 'HLS':
        is_valid = is_valid_hls(merged_data)
    elif MODALITY == 'RTC':
        is_valid = is_valid_rtc(merged_data)

    return is_valid


def filter_chips(chips):
    if MODALITY == 'HLS':
        filtered_chips = filter_hls_chips(chips)
    elif MODALITY == 'RTC':
        filtered_chips = filter_rtc_chips(chips)

    return filtered_chips


def band_from_filename(filename):
    if MODALITY == 'HLS':
        band = band_from_hls_filename(filename)
    elif MODALITY == 'RTC':
        band = band_from_rtc_filename(filename)

    return band


def make_merge_name(filename):
    if MODALITY == 'HLS':
        merge_name = make_merged_hls_name(filename)
    elif MODALITY == 'RTC':
        merge_name = make_merged_rtc_name(filename)

    return merge_name


def get_img(band_data):
    if MODALITY == 'HLS':
        img = get_hls_img(band_data)
    elif MODALITY == 'RTC':
        img = get_rtc_img(band_data)

    return img


# ------------------------- RTC FUNCTIONS --------------------------- #
def search_rtc_data(swath: pd.Series) -> list[DataGranule]:
    start_date = swath["s1Date"]
    final_date = start_date + timedelta(days=1)

    date_range = (start_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))

    results = earthaccess.search_data(
        short_name=["OPERA_L2_RTC-S1_V1"],
        temporal=date_range,
        bounding_box=swath["geometry"].bounds,
    )

    return results


def band_from_rtc_filename(filename):
    # OPERA_L2_RTC-S1_T063-133415-IW2_20170620T001327Z_20250925T045340Z_S1A_30_v1.0_VV.tif
    return filename.split('_')[-1].split('.')[0]


def make_merged_rtc_name(template_filename: str) -> str:
    """
    https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/#naming-convention
    swathID.OPERA_L2_RTC-S1_[BurstID]_[StartDateTime]_[ProductGenerationDateTime] _[Sensor]_[PixelSpacing]_[ProductVersion]_[LayerName].Ext

    Input:   1442.OPERA_L2_RTC-S1_T063-133415-IW2_20170620T001327Z_20250925T045340Z_S1A_30_v1.0_VV.tif
    Returns: 1442.OPERA_L2_RTC-133415-IW2_20170620_S1A_30_v1.0_VV.tif
    """

    # ['1442.OPERA', 'L2', 'RTC-S1', 'T063-133415-IW2', '20170620T001327Z', '20250925T045340Z', 'S1A', '30', 'v1.0', 'VV.tif']
    name_parts = template_filename.split("_")

    name_parts.pop(5)  # Remove Product Generation Time
    name_parts.pop(3)  # Remove Burst ID

    return "_".join(name_parts)


def is_valid_rtc(merged: dict[str, Path]) -> bool:
    with rasterio.open(merged["mask"]) as ds:
        validity_mask = ds.read(1)

    with rasterio.open(merged["EVENT"]) as ds:
        event_mask = ds.read(1)

    is_event_pixel = event_mask == 1
    # https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/#validity-mask
    is_valid_pixel = np.isin(validity_mask, [0, 1])

    total_event_pixels = is_event_pixel.sum()
    valid_event_pixels = (is_event_pixel & is_valid_pixel).sum()

    pct_valid_data = 100.0 * valid_event_pixels / total_event_pixels
    print(f"Percent of the event with valid data: {pct_valid_data:.1f}%")

    return pct_valid_data > 50.0


def filter_rtc_chips(chips: dict[str, dict]) -> list[dict]:
    good_chips = []

    for tile_id, chip in chips.items():
        with rasterio.open(chip["BANDS"]) as ds:
            rtc_data = ds.read()

        with rasterio.open(chip["EVENT"]) as ds:
            event_mask = ds.read(1)

        has_nan_pixels = np.isnan(rtc_data).sum() > 0

        num_pixels = event_mask.size
        num_event_pixels = np.count_nonzero(event_mask > 0)

        pct_pixels_over_event = 100.0 * (num_event_pixels / num_pixels)
        data_overlaps_event = pct_pixels_over_event > 1

        if not has_nan_pixels and data_overlaps_event:
            good_chips.append(chip)

    return good_chips


def normalize_image_array(
    input_array: np.ndarray, vmin: float, vmax: float
) -> np.ndarray:
    input_array = input_array.astype(float)
    scaled_array = (input_array - vmin) / (vmax - vmin)
    scaled_array[np.isnan(input_array)] = 0
    normalized_array = np.round(np.clip(scaled_array, 0, 1) * 255).astype(np.uint8)

    return normalized_array


def get_rtc_img(rtc_data: np.ndarray) -> np.ndarray:
    vv = normalize_image_array(np.sqrt(rtc_data[0]), 0.14, 0.52)
    vh = normalize_image_array(np.sqrt(rtc_data[1]), 0.05, 0.259)

    img = np.stack([vv, vh, vv], axis=-1)

    return img


# ------------------------- HLS FUNCTIONS --------------------------- #

def search_hls_data(swath: pd.Series) -> list[DataGranule]:
    start_date = swath["ls5hlsDate"]
    final_date = start_date + timedelta(days=1)
    date_range = (start_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))

    #  collection_ids = ["C2021957295-LPCLOUD"]  # S2
    collection_ids = ["C2021957657-LPCLOUD", "C2021957295-LPCLOUD"] # S2, L30

    results = earthaccess.search_data(
        concept_id=collection_ids,
        temporal=date_range,
        bounding_box=swath["geometry"].bounds,
        cloud_hosted=True,
    )

    return results


def band_from_hls_filename(filename):
    # 1442.HLS.L30.T15TUG.2017167T165321.v2.0.B06.tif
    parts = filename.split('.')

    sensor, band = parts[2], parts[-2]

    bands = {
        "L30": {
           "B02": "B",
           "B03": "G",
           "B04": "R",
           "B05": "N",
           "B06": "SW1",
           "B07": "SW2",
           "Fmask": "Fmask",
        },
        "S30": {
           "B02": "B",
           "B03": "G",
           "B04": "R",
           "B08": "N",
           "B11": "SW1",
           "B12": "SW2",
           "Fmask": "Fmask",
        }
    }

    try:
        return bands[sensor][band]
    except KeyError:
        return ''


def make_merged_hls_name(template_filename: str) -> str:
    parts = template_filename.split(".")
    parts[4] = parts[4][0:7]
    parts.pop(3)
    parts[-2] = band_from_hls_filename(template_filename)
    f_template_merge = ".".join(parts)
    return f_template_merge


def clear_px_Fmask(Fmask: np.ndarray) -> np.ndarray:
    fmask_clear = np.array([
        0, 4, 16, 20, 32, 36, 48, 52,
        64, 68, 80, 84, 96, 100, 112, 116,
        128, 132, 144, 148, 160, 164, 176, 180,
        192, 196, 208, 212, 224, 228, 240, 244
    ], dtype=Fmask.dtype)

    cloudmask = np.ones_like(Fmask, dtype=np.uint8)
    cloudmask[np.isin(Fmask, fmask_clear)] = 0
    cloudmask[Fmask == 255] = 255

    return cloudmask


def is_valid_hls(merged: dict[str, Path]):
    f_event, f_qc = merged['EVENT'], merged['Fmask']

    with rasterio.open(f_qc) as ds:
        qc = clear_px_Fmask(ds.read(1))

    with rasterio.open(f_event) as ds:
        event_mask = ds.read(1)

    ny, nx = np.shape(qc)
    mask = np.zeros((ny, nx), "uint8")

    ok = np.where((event_mask == 1) & (qc == 0))
    n_cf_event = len(ok[0])
    mask[ok] = 1

    ok = np.where((event_mask == 1) & (qc != 255))
    n_valid_event = len(ok[0])

    ok = np.where((event_mask == 1))
    n_event = len(ok[0])

    pct_cf_event = 0
    if n_valid_event == 0:
        print("No coverage")
    else:
        pct_cf_event = 100.0 * (n_cf_event / n_event)
    print("Percent CF/valid in Event:", pct_cf_event)

    return pct_cf_event > 50


def filter_hls_chips(chips: dict[str, dict]) -> list[dict]:
    good_chips = []

    for tile_id, chip in chips.items():
        with rasterio.open(chip["Fmask"]) as ds:
            qc = clear_px_Fmask(ds.read(1))

        with rasterio.open(chip["EVENT"]) as ds:
            event = ds.read(1)

        ny, nx = qc.shape
        n_px = 1.0 * ny * nx

        # cloud-free pixels (0 clear, 1 cloud, 255 nodata)
        n_cf = len(np.where(qc == 0)[0])
        pct_cf = 100.0 * (n_cf / n_px)

        # event pixels
        n_ev = len(np.where(event > 0)[0])

        # pct of chip in event
        pct_ev = 100.0 * (n_ev / n_px) if n_ev > 0 else 0

        if pct_cf > 95 and pct_ev > 1:
            good_chips.append(chip)

    return good_chips


def bytescale(arr, cmin=0, cmax=1, low=0, high=255):
    # clip the data to be in the range of cmin to cmax
    arr = np.clip(arr, cmin, cmax)
    high = float(high)
    low = float(low)
    cmax = float(cmax)
    cmin = float(cmin)
    m = (high - low) / (cmax - cmin)  # slope
    b = high - (m * cmax)  # intercept
    arr = np.uint8((m * arr) + b)
    return arr


def get_hls_img(hls_data: np.ndarray) -> np.ndarray:
    # B04
    r = bytescale(np.sqrt(np.clip(hls_data[2] / 10000.0, 0, 2)), 0, 0.5)

    # B03
    g = bytescale(np.sqrt(np.clip(hls_data[1] / 10000.0, 0, 2)), 0, 0.5)

    # B02
    b = bytescale(np.sqrt(np.clip(hls_data[0] / 10000.0, 0, 2)), 0, 0.5)

    rgb = np.dstack((r, g, b))
    return rgb


if __name__ == "__main__":
    # HLS Modality Settings
    MODALITY = "HLS"
    ALL_BANDS = ("B", "G", "R", "N", "SW1", "SW2", "Fmask")
    STACK_BANDS = ("B", "G", "R", "N", "SW1", "SW2")
    CHIP_BANDS = ("BANDS", "EVENT", "MASK", "Fmask")
    main()

    # RTC Modality Settings
    MODALITY = "RTC"
    ALL_BANDS = ("VV", "VH", "mask")
    STACK_BANDS = ("VV", "VH")
    CHIP_BANDS = ("BANDS", "EVENT", "MASK")
    main()
