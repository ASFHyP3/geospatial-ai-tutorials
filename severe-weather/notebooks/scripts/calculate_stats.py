import numpy as np
import rasterio


def calculate_stats(good_chips_df, to_stack):
    # Initialize stats for each band
    band_stats = {}
    for t in to_stack:
        band_stats[t + "_total_sum"] = 0
        band_stats[t + "_total_sq_sum"] = 0
        band_stats[t + "_total_count"] = 0

    # Compute running sums across all good chips
    for f in good_chips_df["BANDS"]:
        with rasterio.open(f) as ds:
            for tt, t in enumerate(to_stack):
                img = ds.read(tt + 1).astype("float32")
                ok = np.where(img > 0)
                if len(ok[0]) > 0:
                    band_stats[t + "_total_sum"] += np.sum(img[ok])
                    band_stats[t + "_total_sq_sum"] += np.sum(img[ok] ** 2)
                    band_stats[t + "_total_count"] += len(ok[0])

    # Compute mean and std for each band
    for t in to_stack:
        total_sum = band_stats[t + "_total_sum"]
        total_sq_sum = band_stats[t + "_total_sq_sum"]
        total_count = band_stats[t + "_total_count"]
        band_stats[t + "_mean"] = total_sum / total_count
        band_stats[t + "_std"] = np.sqrt(
            (total_sq_sum / total_count) - (band_stats[t + "_mean"] ** 2)
        )

    # Print stats
    for key in band_stats.keys():
        print(key, band_stats[key])

    # Collect means and stds
    chip_means = [band_stats[t + "_mean"] for t in to_stack]
    chip_stds = [band_stats[t + "_std"] for t in to_stack]

    return chip_means, chip_stds
