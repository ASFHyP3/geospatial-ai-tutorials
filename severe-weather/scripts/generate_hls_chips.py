import os
from pathlib import Path
import glob
import logging

from sklearn.model_selection import train_test_split
import earthaccess

import event_database
import get_swath
import chip_data
import select_chips
import calculate_stats
import plot_chips


logging.basicConfig(level=logging.ERROR)
logging.getLogger("rasterio").setLevel(logging.ERROR)


def main():
    gdf = event_database.load()
    gdf = event_database.add_buffered(gdf)

    keepers = [1442, 622, 1079, 628]
    gdf = gdf[gdf["swathID"].isin(keepers)]

    data_path = Path("hwds") / 'HLS'
    earthaccess.login()

    for i, row in gdf.iterrows():
        get_swath.get_swath(row, hwds_path=data_path)

    fmasks_merged = glob.glob(os.path.join("hwds/MERGE", "*Fmask.tif"))

    to_stack = ["B", "G", "R", "N", "SW1", "SW2"]
    chips_dict = chip_data.chip_data(fmasks_merged, to_stack)

    chips_df = select_chips.select_chips(chips_dict)

    keep = chips_df["keep"]
    good_chips_df = chips_df.loc[keep]

    print(good_chips_df)

    create_split_files(good_chips_df)
    plot_chips.plot_chips(gdf, chips_df, fmasks_merged)

    means, stds = calculate_stats.calculate_stats(good_chips_df, to_stack)

    print(f"Chip means: {means}")
    print(f"Chip stds: {stds}")


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
