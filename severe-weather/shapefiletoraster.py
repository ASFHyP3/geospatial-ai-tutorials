import pathlib
import requests
import zipfile
import io

import geopandas as gpd
from tqdm import tqdm
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin


CWD = pathlib.Path.cwd()
DATA_DIR = CWD / 'data'


def download_data():
    dl_link = 'https://github.com/jrbell1/hwds_db/raw/refs/heads/main/hwds_v1_2_20250205.zip'
    zip_dir = DATA_DIR / 'hwds_v1_2_20250205'
    shp_path = zip_dir / 'hwds_v1_2_20250205.shp'

    if shp_path.exists():
        return shp_path

    resp = requests.get(dl_link)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        z.extractall(zip_dir)

    return shp_path


def main() -> None:
    out_dir = DATA_DIR / 'hwds-rasters'
    pixel_size = 10  # meters
    padding = 1000   # meters

    out_dir.mkdir(parents=True, exist_ok=True)
    shp_path = download_data()

    gdf = gpd.read_file(shp_path)

    utm_crs = gdf.estimate_utm_crs()
    gdf = gdf.to_crs(utm_crs)

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), miniters=10):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if 'N/A' in row['s1Date']:
            continue

        swath_id = row['swathID']

        minx, miny, maxx, maxy = geom.bounds
        minx -= padding
        miny -= padding
        maxx += padding
        maxy += padding

        width = int(np.ceil((maxx - minx) / pixel_size))
        height = int(np.ceil((maxy - miny) / pixel_size))

        transform = from_origin(minx, maxy, pixel_size, pixel_size)

        rasterized = rasterize(
            [(geom, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8"
        )

        out_path = out_dir / f"swathID_{swath_id}_swathDate_{row['swathDate']}.tif"

        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="uint8",
            crs=gdf.crs,
            transform=transform
        ) as dst:
            dst.write(rasterized, 1)
            dst.update_tags(polygon_id=str(swath_id))


if __name__ == '__main__':
    main()
