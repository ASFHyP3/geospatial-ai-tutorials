import glob
import os
import rasterio
import numpy as np


def merge_data(fmasks_merged):
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

        to_stack = ["B", "G", "R", "N", "SW1", "SW2"]
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

    return hls_merged, to_stack
