import os

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import box


def bytescale(arr, cmin=0, cmax=1, low=0, high=255):
    # clip the data to be in the range of cmin to cmax
    arr = np.clip(arr, cmin, cmax)
    high = float(high)
    low = float(low)
    cmax = float(cmax)
    cmin = float(cmin)
    m = (high - low) / (cmax - cmin)  # slope
    b = high - (m * cmax)  # intercept
    arr = np.uint8((m * arr) + b)
    return arr


def plot_chips(gdf, chips_df, fmasks_merged):
    crs_pc = ccrs.PlateCarree()

    for fmask_merged in fmasks_merged:
        # extract swath info
        swathID = os.path.basename(fmask_merged)[0:4]
        int_swathID = int(swathID)
        ok = gdf["swathID"] == int_swathID
        this_gdf = gdf.loc[ok]
        swath_geom = this_gdf["geometry"].values[0]

        # read RGB bands and scale
        ds = rasterio.open(fmask_merged.replace("Fmask", "B04"))
        bounds = ds.bounds
        full_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        r = bytescale(np.sqrt(np.clip(ds.read(1) / 10000.0, 0, 2)), 0, 0.5)
        ds.close()

        ds = rasterio.open(fmask_merged.replace("Fmask", "B03"))
        g = bytescale(np.sqrt(np.clip(ds.read(1) / 10000.0, 0, 2)), 0, 0.5)
        ds.close()

        ds = rasterio.open(fmask_merged.replace("Fmask", "B02"))
        b = bytescale(np.sqrt(np.clip(ds.read(1) / 10000.0, 0, 2)), 0, 0.5)
        ds.close()

        rgb = np.dstack((r, g, b))

        # plot RGB background
        fig, ax = plt.subplots(
            1,
            1,
            subplot_kw={"projection": crs_pc},
            figsize=(12, 12),
            layout="constrained",
        )
        ax.imshow(rgb, extent=full_extent, origin="upper", transform=crs_pc)
        ax.add_geometries(
            [swath_geom], edgecolor="red", linewidth=2, facecolor="none", crs=crs_pc
        )

        # plot chip extents
        base = ".".join(os.path.basename(fmask_merged).split(".")[0:6])
        print(base)
        ok = chips_df["root"] == base
        these_chips = chips_df.loc[ok]
        print("len of these_chips:", len(these_chips))

        if len(these_chips) <= 0:
            print("no chips")

        for _, chip in these_chips.iterrows():
            ds = rasterio.open(chip["MASK"])
            chip_bounds = ds.bounds
            chip_geom = box(
                chip_bounds.left,
                chip_bounds.bottom,
                chip_bounds.right,
                chip_bounds.top,
            )
            ds.close()

            if (chip["pct_cf"] > 95) & (chip["pct_ev"] > 1):
                chip_color, chip_width, chip_z = "green", 3, 2
            else:
                chip_color, chip_width, chip_z = "yellow", 1, 1

            ax.add_geometries(
                [chip_geom],
                edgecolor=chip_color,
                linewidth=chip_width,
                alpha=1,
                zorder=chip_z,
                facecolor="none",
                crs=crs_pc,
            )

        ax.set_extent(full_extent, crs=crs_pc)
        plt.show()
