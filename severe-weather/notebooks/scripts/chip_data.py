import os
import glob

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


def chip_data(fmasks_merged, to_stack):
    # use fmasks as templates and do stuff
    # keep track of all eventual tiles
    chips_dict = {"Fmask": [], "MASK": [], "EVENT": [], "BANDS": [], "QC": []}

    p_chips = "hwds/CHIPS"
    if not os.path.isdir(p_chips):
        print("making:", p_chips)
        os.makedirs(p_chips, exist_ok=True)
    for fmask_merged in fmasks_merged:
        if "L30" in fmask_merged:
            bands = {
                "B": "B02",
                "G": "B03",
                "R": "B04",
                "N": "B05",
                "SW1": "B06",
                "SW2": "B07",
                "Fmask": "Fmask",
                "EVENT": "EVENT",
                "QC": "QC",
                "MASK": "MASK",
            }
        else:
            bands = {
                "B": "B02",
                "G": "B03",
                "R": "B04",
                "N": "B08",
                "SW1": "B11",
                "SW2": "B12",
                "Fmask": "Fmask",
                "EVENT": "EVENT",
                "QC": "QC",
                "MASK": "MASK",
            }

        hls_merged = {}

        for key in bands.keys():
            hls_merged[key] = fmask_merged.replace("Fmask", bands[key])
            print(hls_merged[key], os.path.isfile(hls_merged[key]))

        # get the bands we need to stack
        to_stack_ds = []
        for t in to_stack:
            f_band = hls_merged[t]
            ds = rasterio.open(f_band)
            to_stack_ds.append(ds)

        # create output file
        f_bands = fmask_merged.replace("Fmask", "BANDS")

        # remove old while debugging
        if os.path.isfile(f_bands):
            os.remove(f_bands)

        if not os.path.isfile(f_bands):
            meta = to_stack_ds[0].meta.copy()
            meta["count"] = len(to_stack)
            meta["dtype"] = np.float32
            with rasterio.open(f_bands, "w", **meta) as dst:
                for tt, band_ds in enumerate(to_stack_ds):
                    img = band_ds.read(1)
                    dst.write(img, tt + 1)
                    band_ds = None

        print(f_bands, os.path.isfile(f_bands))
        hls_merged["BANDS"] = f_bands

        # create chips
        chip_size = 512
        to_tile = {
            "QC": hls_merged["QC"],
            "Fmask": hls_merged["Fmask"],
            "MASK": hls_merged["MASK"],
            "EVENT": hls_merged["EVENT"],
            "BANDS": hls_merged["BANDS"],
        }

        for tt in to_tile.keys():
            print("chips for:", to_tile[tt])

            with rasterio.open(to_tile[tt]) as src:
                meta = src.meta.copy()
                h_tiles = src.width // chip_size
                v_tiles = src.height // chip_size

                for ii in range(v_tiles + 1):
                    for jj in range(h_tiles + 1):
                        x_off = jj * chip_size
                        y_off = ii * chip_size
                        width = min(chip_size, src.width - x_off)
                        height = min(chip_size, src.height - y_off)

                        # skip bad ones
                        if width != chip_size or height != chip_size:
                            print("bad chip, skip:", width, height)
                            continue

                        window = Window(x_off, y_off, width, height)
                        meta.update(
                            {
                                "width": width,
                                "height": height,
                                "transform": src.window_transform(window),
                            }
                        )

                        ii_str = "%3.3d" % ii
                        jj_str = "%3.3d" % jj

                        tile_number = ".".join([ii_str, jj_str])

                        f_chip = os.path.basename(to_tile[tt]).replace(
                            tt + ".tif", tile_number + "." + tt + ".tif"
                        )

                        f_chip = os.path.join(p_chips, f_chip)

                        with rasterio.open(f_chip, "w", **meta) as dst:
                            for bb in range(meta["count"]):
                                data = src.read(bb + 1, window=window)
                                dst.write(data, bb + 1)
                        chips_dict[tt].append(f_chip)

    # for convenience (wasting disk space though) just copy keepers
    # to a new folder for terramind use (_TM)
    p_chips_keep = 'hwds/CHIPS_TM'
    if not os.path.isdir(p_chips_keep):
        print('making:', p_chips_keep)
        os.makedirs(p_chips_keep, exist_ok=True)

    # tidy old files
    files = glob.glob(os.path.join(p_chips_keep, '*.tif'))
    for f in files:
        os.remove(f)

    # show count of chips per key
    for key in chips_dict.keys():
        print(key, len(chips_dict[key]))

    # create DataFrame to track chips
    chips_df = pd.DataFrame.from_dict(chips_dict)
    chips_df['keep'] = False
    chips_df['base'] = ''
    chips_df['root'] = ''
    chips_df['pct_cf'] = 0.0
    chips_df['pct_ev'] = 0.0
    print(chips_df)

    return chips_df
