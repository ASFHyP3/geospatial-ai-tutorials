from pathlib import Path
from datetime import timedelta
import shutil

import numpy as np
import cartopy.crs as ccrs
import earthaccess
from earthaccess.results import DataGranule
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.merge import merge
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from shapely.geometry import box
from rasterio.crs import CRS

import event_database


MINIMUM_VALID_DATA_PERCENT = 50.0


def main():
    hwds_path = Path("hwds")
    rtc_path = hwds_path / "RTC"

    data_paths = {
        "RTC": hwds_path / "RTC",
        "RAW": rtc_path / "RAW",
        "WGS84": rtc_path / "WGS84",
        "MERGE": rtc_path / "MERGE",
        "CHIPS": rtc_path / "CHIPS",
        "CHIPS_TM": rtc_path / "CHIPS_TM",
    }

    for p in data_paths.values():
        p.mkdir(parents=True, exist_ok=True)

    gdf = event_database.load(hwds_path)

    keepers = [1442, 622, 1079, 628]
    gdf = gdf[gdf["swathID"].isin(keepers)]

    earthaccess.login()

    # chip_swaths(gdf, data_paths)

    for _, swath in gdf.iterrows():
        swath_id = _make_swath_id(swath["swathID"])

        all_chips = list(data_paths["CHIPS"].glob(f"{swath_id}.*.tif"))
        good_chips = list(data_paths["CHIPS_TM"].glob(f"{swath_id}.*.tif"))

        merged_file = list(data_paths["MERGE"].glob(f"{swath_id}.*BANDS.tif"))[0]

        _plot_chips(merged_file, all_chips, good_chips, swath)


def chip_swaths(gdf, data_paths):
    tm_chips = []
    for _, swath in gdf.iterrows():
        merged_data = _get_data_for_swath(swath, data_paths)

        if merged_data is None or not _is_valid_rtc(merged_data):
            print("Skipping: not enough valid data")
            continue

        print("Chipping!")
        merged_data = _stack_rtc_bands(merged_data, data_bands=("VV", "VH"))
        chips = _chip_rtc_data(merged_data, data_paths)

        good_chips = _filter_chips(chips)
        print(f"Found {len(good_chips)} good chips")

        for chip in good_chips:
            for chip_path in chip.values():
                dest = data_paths["CHIPS_TM"] / chip_path.name
                shutil.copy(chip_path, dest)

        tm_chips += good_chips

    return tm_chips


def _plot_chips(merged_band_file, all_chips, good_chips, swath):
    crs_pc = ccrs.PlateCarree()

    with rasterio.open(merged_band_file) as ds:
        bounds = ds.bounds
        full_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        rtc_data = ds.read()

    def normalize_image_array(
        input_array: np.ndarray, vmin: float, vmax: float
    ) -> np.ndarray:
        input_array = input_array.astype(float)
        scaled_array = (input_array - vmin) / (vmax - vmin)
        scaled_array[np.isnan(input_array)] = 0
        normalized_array = np.round(np.clip(scaled_array, 0, 1) * 255).astype(np.uint8)

        return normalized_array

    vv = normalize_image_array(np.sqrt(rtc_data[0]), 0.14, 0.52)
    vh = normalize_image_array(np.sqrt(rtc_data[1]), 0.05, 0.259)
    img = np.stack([vv, vh, vv], axis=-1)

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
    plt.show()


def _make_swath_id(swathID):
    return f"{int(swathID):04d}"


def _get_data_for_swath(swath: pd.Series, data_paths: dict) -> dict[str, Path] | None:
    swathID = f"{int(swath['swathID']):04d}"

    results = _search_data(swath)
    local_files = earthaccess.download(
        results, local_path=data_paths["RAW"], show_progress=True
    )
    data_tifs = [f for f in local_files if f.name.endswith(".tif")]

    if len(data_tifs) == 0:
        return None

    reprojected_tifs = _reproject_files(
        data_tifs, output_path=data_paths["WGS84"], prefix=f"{swathID}."
    )

    merged = {}
    for band in ("VV", "VH", "mask"):
        band_files = [f for f in reprojected_tifs if band in f.name]

        merged_name = _make_merged_name(band_files[0].name)
        merged_band_path = _merge(
            band_files, output_file=data_paths["MERGE"] / merged_name
        )

        merged[band] = merged_band_path

    event_tif, mask_tif = _generate_masks(
        merged["VV"], swath, merged_extension="VV.tif"
    )

    return {
        **merged,
        "EVENT": event_tif,
        "MASK": mask_tif,
    }


def _is_valid_rtc(merged: dict[str, Path]) -> bool:
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

    return pct_valid_data > MINIMUM_VALID_DATA_PERCENT


def _make_merged_name(template_filename: str) -> str:
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


def _search_data(swath: pd.Series) -> list[DataGranule]:
    start_date = swath["s1Date"]
    final_date = start_date + timedelta(days=1)

    date_range = (start_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))

    results = earthaccess.search_data(
        short_name=["OPERA_L2_RTC-S1_V1"],
        temporal=date_range,
        bounding_box=swath["geometry"].bounds,
    )

    return results


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


def _stack_rtc_bands(merged: dict[str, Path], data_bands: tuple[str]) -> None:
    with rasterio.open(merged[data_bands[0]]) as src:
        meta = src.meta.copy()

    meta.update(count=len(data_bands), dtype=np.float32)
    stacked_file_name = _rename(merged["VV"], "VV.tif", "BANDS.tif")

    with rasterio.open(stacked_file_name, "w", **meta) as dst:
        for idx, band in enumerate(data_bands, start=1):
            with rasterio.open(merged[band]) as src:
                dst.write(src.read(1), idx)

    merged["BANDS"] = stacked_file_name

    return merged


def _chip_rtc_data(merged: dict[str, Path], data_paths: dict[str, Path], chip_size=256):
    chips = {}
    # chips[tile_id] = {band, event, mask}

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

    # BAND (VV, VH), EVENT, MASK
    for chip_layer in ("BANDS", "mask", "EVENT", "MASK"):
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

                chip_name = layer_path.name.replace(
                    f"{chip_layer}.tif", f"{tile_id}.{chip_layer}.tif"
                )
                chip_path = data_paths["CHIPS"] / chip_name

                with rasterio.open(chip_path, "w", **chip_meta) as dst:
                    dst.write(data)

                chips[tile_id][chip_layer] = chip_path

    return chips


def _filter_chips(chips: dict[str, dict]) -> list[dict]:
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


if __name__ == "__main__":
    main()
