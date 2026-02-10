import os
import glob
import rasterio
import pandas as pd
import numpy as np


def select_chips(chips_dict):
    # for convenience (wasting disk space though) just copy keepers
    # to a new folder for terramind use (_TM)
    p_chips_keep = "hwds/CHIPS_TM"
    if not os.path.isdir(p_chips_keep):
        print("making:", p_chips_keep)
        os.makedirs(p_chips_keep, exist_ok=True)

    # tidy old files
    files = glob.glob(os.path.join(p_chips_keep, "*.tif"))
    for f in files:
        os.remove(f)

    # show count of chips per key
    for key in chips_dict.keys():
        print(key, len(chips_dict[key]))

    # create DataFrame to track chips
    chips_df = pd.DataFrame.from_dict(chips_dict)
    chips_df["keep"] = False
    chips_df["base"] = ""
    chips_df["root"] = ""
    chips_df["pct_cf"] = 0.0
    chips_df["pct_ev"] = 0.0
    print(chips_df)

    # iterate over chips to assess quality
    for i, row in chips_df.iterrows():
        print(row)

        # read QC, EVENT, and MASK bands
        with rasterio.open(row["QC"]) as ds:
            qc = ds.read(1)
        with rasterio.open(row["EVENT"]) as ds:
            event = ds.read(1)
        with rasterio.open(row["MASK"]) as ds:
            mask = ds.read(1)

        ny, nx = qc.shape
        n_px = 1.0 * ny * nx

        # cloud-free pixels (0 clear, 1 cloud, 255 nodata)
        n_cf = len(np.where(qc == 0)[0])
        pct_cf = 100.0 * (n_cf / n_px)

        # valid pixels (not nodata)
        nv = len(np.where(qc != 255)[0])
        pct_valid = 100.0 * (nv / n_px)

        # event pixels
        n_ev = len(np.where(event > 0)[0])

        # cloud-free AND event pixels
        n_cf_ev = len(np.where((qc == 0) & (event > 0))[0])
        pct_cf_ev = 100.0 * (n_cf_ev / n_ev) if n_ev > 0 else 0

        # pct of chip in event
        pct_ev = 100.0 * (n_ev / n_px) if n_ev > 0 else 0

        # file base and root names
        f_base = ".".join(os.path.basename(row["MASK"]).split(".")[:-2])
        f_root = ".".join(os.path.basename(row["MASK"]).split(".")[:-4])
        print(f_base)
        print(f_root)
        print(pct_cf, pct_ev)

        # update DataFrame
        chips_df.at[i, "base"] = f_base
        chips_df.at[i, "root"] = f_root
        chips_df.at[i, "pct_cf"] = pct_cf
        chips_df.at[i, "pct_ev"] = pct_ev

        # loosened criteria for Colab space
        if pct_cf > 95 and pct_ev > 1:
            chips_df.at[i, "keep"] = True

            # copy MASK and BANDS to _TM
            for f_key in ["MASK", "BANDS"]:
                src_file = row[f_key]
                cmd = ["/usr/bin/cp", src_file, p_chips_keep]
                os.system(" ".join(cmd))

    return chips_df
