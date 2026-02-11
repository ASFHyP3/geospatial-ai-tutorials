from pathlib import Path
from datetime import timedelta

import numpy as np
import earthaccess
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from rasterio.crs import CRS

import event_database


def main():
    gdf = event_database.load()
    gdf = event_database.add_buffered(gdf)

    keepers = [1442, 622, 1079, 628]
    gdf = gdf[gdf["swathID"].isin(keepers)]

    hwds_path = Path("hwds") / 'RTC'
    earthaccess.login()

    raw_path = hwds_path / "RAW"
    wgs84_path = hwds_path / "WGS84"
    merge_path = hwds_path / "MERGE"

    for p in (hwds_path, raw_path, wgs84_path, merge_path):
        p.mkdir(parents=True, exist_ok=True)

    merged = []
    for _, swath in gdf.iterrows():
        swathID = f"{int(swath['swathID']):04d}"

        results = _search_data(swath)
        local_files = earthaccess.download(results, local_path=raw_path, show_progress=True)
        data_tifs = [f for f in local_files if 'VV.tif' in f.name or 'VH.tif' in f.name]

        reprojected_tifs = _reproject_files(data_tifs, output_path=wgs84_path, prefix=f'{swathID}.')

        for pol in ('VV', 'VH'):
            pol_files = [f for f in reprojected_tifs if pol in f.name]

            merged_name = _make_merged_name(pol_files[0].name)
            merged_file_path = _merge(pol_files, output_file=merge_path / merged_name)

            merged.append(merged_file_path)


def _make_merged_name(template_filename: str) -> str:
    """
    https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/#naming-convention
    swathID.OPERA_L2_RTC-S1_[BurstID]_[StartDateTime]_[ProductGenerationDateTime] _[Sensor]_[PixelSpacing]_[ProductVersion]_[LayerName].Ext

    Input:   1442.OPERA_L2_RTC-S1_T063-133415-IW2_20170620T001327Z_20250925T045340Z_S1A_30_v1.0_VV.tif

    Returns: 1442.OPERA_L2_RTC-133415-IW2_20170620_S1A_30_v1.0_VV.tif
    """

    name_parts = template_filename.split('_')

    # ['1442.OPERA', 'L2', 'RTC-S1', 'T063-133415-IW2', '20170620T001327Z', '20250925T045340Z', 'S1A', '30', 'v1.0', 'VV.tif']
    name_parts.pop(3)  # Remove Burst ID
    name_parts.pop(4)  # Remove Product Generation Time

    return '_'.join(name_parts)


def _search_data(swath):
    start_date = swath['s1Date']
    final_date = start_date + timedelta(days=1)

    date_range = (start_date.strftime("%Y-%m-%d"), final_date.strftime("%Y-%m-%d"))

    results = earthaccess.search_data(
        short_name=['OPERA_L2_RTC-S1_V1'],
        temporal=date_range,
        bounding_box=swath["geometry"].bounds,
    )

    return results


def _reproject_files(
    granule_files: list[Path], output_path: Path, prefix: str
) -> list[Path]:

    reprojected_paths = [
        Path(output_path) / f'{prefix}{granule.name}' for granule in granule_files
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


def _merge(rtc_tifs: list[Path], output_file: Path) -> Path:
    band_datasets = [rasterio.open(rtc_tif) for rtc_tif in rtc_tifs]

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

    for ds in band_datasets:
        ds.close()

    return output_file



if __name__ == '__main__':
    main()
