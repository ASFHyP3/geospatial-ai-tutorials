# Shell command for installing cartopy (cannot be used as an import)
# !pip install cartopy

# Standard library
import os
from pathlib import Path
import glob
import zipfile
import logging
import warnings

# Scientific computing
import pandas as pd
from sklearn.model_selection import train_test_split

# Raster & geospatial
import geopandas as gpd

# Misc
import gdown
import earthaccess

import get_swath
import merge_data
import chip_data
import select_chips
import calculate_stats
import plot_chips


# Suppress warnings and rasterio logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("rasterio").setLevel(logging.ERROR)


def main():
    gdf = load_event_database()
    gdf = add_buffered_events_to(gdf)

    keepers = [1442, 622, 1079, 628]
    gdf = gdf[gdf["swathID"].isin(keepers)]

    data_path = Path("hwds")
    earthaccess.login()

    for i, row in gdf.iterrows():
        get_swath.get_swath(row, hwds_path=data_path)

    fmasks_merged = glob.glob(os.path.join("hwds/MERGE", "*Fmask.tif"))

    hls_merged, to_stack = merge_data.merge_data(fmasks_merged)
    chips_dict = chip_data.chip_data(hls_merged)

    chips_df = select_chips.select_chips(chips_dict)

    keep = chips_df["keep"]
    good_chips_df = chips_df.loc[keep]

    print(good_chips_df)

    create_split_files(good_chips_df)
    plot_chips.plot_chips(gdf, chips_df, fmasks_merged)

    means, stds = calculate_stats.calculate_stats(good_chips_df, to_stack)

    print(f"Chip means: {means}")
    print(f"Chip stds: {stds}")


def load_event_database():
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

    return gdf


def add_buffered_events_to(gdf):
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

def create_split_files(good_chips_df):
    # split into train/test (and duplicate test as validation)
    train, test = train_test_split(
        good_chips_df["base"], test_size=0.3, random_state=42
    )
    valid = test.copy()

    # prepare splits folder
    p_splits = "hwds/SPLITS"
    if not os.path.isdir(p_splits):
        print("making:", p_splits)
        os.makedirs(p_splits, exist_ok=True)

    # clean old split files
    files = glob.glob(os.path.join(p_splits, "*.txt"))
    for f in files:
        os.remove(f)

    # write out split files
    split_files = {"train.txt": train, "val.txt": valid, "test.txt": test}

    for fname, split_data in split_files.items():
        f_path = os.path.join(p_splits, fname)
        with open(f_path, "w") as f:
            for item in split_data:
                f.write(f"{item}\n")


if __name__ == "__main__":
    main()
