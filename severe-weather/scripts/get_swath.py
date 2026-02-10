import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio import features
from rasterio.merge import merge
from rasterio.crs import CRS

import earthaccess


def _search_data(L30, bbox, start_date):
    final_date = start_date + timedelta(days=1)

    date_range = (start_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))

    # search by collection ID (HLS) listed as L30, S30
    collection_ids = ["C2021957295-LPCLOUD"]

    if L30:
        collection_ids = ["C2021957657-LPCLOUD", "C2021957295-LPCLOUD"]

    results = earthaccess.search_data(
        concept_id=collection_ids,
        cloud_hosted=True,
        temporal=date_range,
        bounding_box=bbox,
    )

    return results


def _get_fmask_info(results):
    fmask_links = [
        granule_link
        for granule in results
        for granule_link in granule.data_links()
        if "Fmask" in granule_link
    ]

    links, platforms, dates = [], [], []
    for fmask_link in fmask_links:
        parts = os.path.basename(fmask_link).split(".")

        links.append(fmask_link)
        platforms.append(parts[1])
        dates.append(datetime.strptime(parts[3], "%Y%jT%H%M%S"))

    df = pd.DataFrame.from_dict({"link": links, "date": dates, "platform": platforms})
    df = df.drop_duplicates(subset=["link"])
    df = df.sort_values(by=["date", "platform"], ascending=[True, True])

    return df


def _make_requests_list(fmasks_df, swathDate):
    requests = []

    for _, fmask in fmasks_df.iterrows():
        bands = _get_bands(is_l30="L30" in fmask["link"])

        requests += [
            fmask["link"].replace("Fmask", bands[band]) for band in bands.keys()
        ]

    return requests


def _reproject_files(
    local_files: list[Path], gran_path: Path, swathID: str
) -> list[Path]:
    local_files_wgs84 = []

    for local_file in local_files:
        if not local_file.exists():
            continue

        f_gran_wgs84 = local_file.name.replace("HLS.", f"{swathID}.HLS.")
        f_gran_wgs84 = Path(gran_path) / f_gran_wgs84

        if not os.path.isfile(f_gran_wgs84):
            print(f"reprojecting to wgs84 have: {f_gran_wgs84}")
            _reproject_file(local_file, f_gran_wgs84)

        local_files_wgs84.append(f_gran_wgs84)

    return local_files_wgs84


def _reproject_file(local_file: Path, f_gran_wgs84: Path) -> None:
    with rasterio.open(local_file) as src:
        dst_crs = CRS.from_epsg(4326)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(f_gran_wgs84, "w", **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                )


def _get_bands(is_l30):
    ls_bands = {
        "B": "B02",
        "G": "B03",
        "R": "B04",
        "N": "B05",
        "SW1": "B06",
        "SW2": "B07",
        "Fmask": "Fmask",
    }
    s2_bands = {
        "B": "B02",
        "G": "B03",
        "R": "B04",
        "N": "B08",
        "SW1": "B11",
        "SW2": "B12",
        "Fmask": "Fmask",
    }

    return ls_bands if is_l30 else s2_bands


def _merge_unique_passes(
    local_files_wgs84: list[Path], merge_path: Path, has_new_files: bool
):
    fmask_files = [f for f in local_files_wgs84 if "Fmask" in f.name]
    unique_passes = set(
        (fmask.name.split(".")[2], fmask.name.split(".")[4][0:7])
        for fmask in fmask_files
    )

    print(unique_passes)
    fmasks_merged = []

    for platform, yyyyjjj in unique_passes:
        fmask_files_pass = [
            f
            for f in local_files_wgs84
            if ("Fmask" in f.name and platform in f.name and yyyyjjj in f.name)
        ]

        template = fmask_files_pass[0]
        parts = os.path.basename(template).split(".")
        parts[4] = parts[4][0:7]
        parts.pop(3)
        f_template_merge = ".".join(parts)

        bands = _get_bands(is_l30="L30" in f_template_merge)

        for band in bands.values():
            f_out = merge_path / f_template_merge.replace("Fmask", band)

            if has_new_files and os.path.isfile(f_out):
                print("new files may have been obtained, removing previous merge")
                os.remove(f_out)

            if os.path.isfile(f_out):
                print("available:", f_out)
            else:
                print("generated:", f_out)
                f_out = _merge_band(fmask_files_pass, f_out, band)

            if "Fmask" in str(f_out):
                fmasks_merged.append(f_out)

    return fmasks_merged


def _merge_band(fmask_files_pass, f_out, band):
    to_merge = []

    for f_fmask in fmask_files_pass:
        band_file = str(f_fmask).replace("Fmask", band)
        ds = rasterio.open(band_file)
        to_merge.append(ds)

    mosaic, out_trans = merge(to_merge)
    mosaic = np.squeeze(mosaic)
    out_meta = to_merge[0].meta.copy()

    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[0],
            "width": mosaic.shape[1],
            "transform": out_trans,
            "crs": to_merge[0].crs,
        }
    )

    with rasterio.open(f_out, "w", **out_meta) as dst:
        dst.write(mosaic, 1)

    for ds in to_merge:
        ds.close()

    return f_out


def _generate_masks(fmask_merged: str, row) -> tuple[Path, Path, Path]:
    fmask_merged = Path(fmask_merged)

    def _rename(path: Path, token: str) -> Path:
        return path.with_stem(path.stem.replace("Fmask", token))

    f_event = _rename(fmask_merged, "EVENT")
    f_mask = _rename(fmask_merged, "MASK")
    f_qc = _rename(fmask_merged, "QC")

    with rasterio.open(fmask_merged) as ds:
        profile = ds.profile

        mask_raster = features.rasterize(
            shapes=[[row["geometry"], 1]],
            fill=0,
            out_shape=ds.shape,
            transform=ds.transform,
        )

        with rasterio.open(f_mask, "w", **profile) as dst:
            dst.write(mask_raster, 1)
        print("generated:", f_mask)

        event_mask = features.rasterize(
            shapes=[
                [row["buffered_event_background"], 3],
                [row["buffered_event"], 2],
                [row["geometry"], 1],
            ],
            fill=0,
            out_shape=ds.shape,
            transform=ds.transform,
        )

        with rasterio.open(f_event, "w", **profile) as dst:
            dst.write(event_mask, 1)
        print("generated:", f_event)

        fmask = ds.read(1)
        qc = clear_px_Fmask(fmask)

        with rasterio.open(f_qc, "w", **profile) as dst:
            dst.write(qc, 1)
        print("generated:", f_qc)

    return f_event, f_mask, f_qc


def get_swath(row, hwds_path: Path, L30=False):
    swathID_str = f"{int(row['swathID']):04d}"

    raw_path = hwds_path / "RAW"
    gran_path = hwds_path / "GRANULES"
    merge_path = hwds_path / "MERGE"

    for p in (hwds_path, raw_path, gran_path, merge_path):
        p.mkdir(parents=True, exist_ok=True)

    results = _search_data(
        L30,
        bbox=row["geometry"].bounds,
        start_date=row["swathDate"],
    )

    print(len(results))

    fmask_df = _get_fmask_info(results)
    band_requests = _make_requests_list(fmask_df, row["swathDate"])

    if len(band_requests) == 0:
        print(f"ERROR: can't find data for swath: {swathID_str}")
        return

    print(len(band_requests))
    files_before_download = set(raw_path.glob("*.tif"))

    local_files = earthaccess.download(
        band_requests, local_path=raw_path, show_progress=True
    )

    new_files = set(local_files) - files_before_download
    print(f"total_files: {len(local_files)}, new_files downloaded: {new_files}")

    local_files_wgs84 = _reproject_files(local_files, gran_path, swathID_str)
    print(f"Reprojected files: {len(local_files_wgs84)}")

    has_new_files = len(new_files) > 0
    fmasks_merged = _merge_unique_passes(local_files_wgs84, merge_path, has_new_files)

    fmask_keepers = []

    # generate event, mask, QC layers
    for fmask_merged in fmasks_merged:
        print("template:", fmask_merged)

        bands = _get_bands(is_l30="L30" in str(fmask_merged))

        # Event mask
        f_event, f_mask, f_qc = _generate_masks(fmask_merged, row)

        # assess scene quality
        ds = rasterio.open(f_qc)

        qc = ds.read(1)
        ds = None
        ds = rasterio.open(f_event)
        event_mask = ds.read(1)
        ds = None
        ds = rasterio.open(str(f_mask).replace("Fmask", bands["R"]))
        r = ds.read(1)
        r = np.clip(r / 10000, 0, 2)
        ds = None

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

        if pct_cf_event > 50:
            fmask_keepers.append(fmask_merged)

    return fmask_keepers


def clear_px_Fmask(Fmask):
    # source:
    # https://benmack.github.io/nasa_hls/build/html/tutorials/Working_with_HLS_datasets_and_nasa_hls.html
    # 255 is missing data value
    fmask_clear = np.array(
        [
            0,
            4,
            16,
            20,
            32,
            36,
            48,
            52,
            64,
            68,
            80,
            84,
            96,
            100,
            112,
            116,
            128,
            132,
            144,
            148,
            160,
            164,
            176,
            180,
            192,
            196,
            208,
            212,
            224,
            228,
            240,
            244,
        ]
    )

    ny, nx = np.shape(Fmask)

    # set all pixels to 1, will reset to 0 where clear
    # and use float data type so that we can use -9999 as consistent missing data value
    cloudmask = np.ones((ny, nx), "uint8")

    for i, val in enumerate(fmask_clear):
        ok = np.where(Fmask == val)
        if len(ok[0]) > 0:
            cloudmask[ok] = 0

    missing = np.where(Fmask == 255)
    if len(missing[0]) > 0:
        cloudmask[missing] = 255

    return cloudmask
