from pathlib import Path
import zipfile
import pandas as pd
import geopandas as gpd
import gdown


def load():
    # use 60-swath version
    hwds_google_drive_id = "1h_JIEcrrUF3OSTrmwAKNPa0eUEhPA2Xx"
    drive_url = f"https://drive.google.com/uc?id={hwds_google_drive_id}"

    shp_dir = Path("hwds/SHP")
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

    return gdf


def add_buffered(gdf):
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
