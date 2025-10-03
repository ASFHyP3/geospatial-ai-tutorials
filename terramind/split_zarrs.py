import pathlib
import random

import xarray as xr
import zarr
from tqdm import tqdm


def make_split_files(data_path: pathlib.Path) -> None:
    file_names = [
        f.stem.split('_S1RTC')[0] for f in (data_path / 'S1RTC').rglob('*.tif')
    ]
    random.shuffle(file_names)

    n = len(file_names)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train = file_names[:train_end]
    val = file_names[train_end:val_end]
    test = file_names[val_end:]

    splits_path = data_path / 'splits'
    splits_path.mkdir(parents=True, exist_ok=True)

    (splits_path / 'train_data.txt').write_text("\n".join(train), encoding="utf-8")
    (splits_path / 'valid_data.txt').write_text("\n".join(val), encoding="utf-8")
    (splits_path / 'test_data.txt').write_text("\n".join(test), encoding="utf-8")


def get_zarr_paths_from(data_dir: pathlib.Path) -> list[pathlib.Path]:
    return list(data_dir.rglob("*.zarr.zip"))


def split_zarr(zarr_path: pathlib.Path) -> None:
    zarr_name = zarr_path.name.split(".zarr.zip")[0]
    store = zarr.storage.ZipStore(zarr_path, read_only=True)
    ds = xr.open_zarr(store, mask_and_scale=True)
    print(f"Splitting: {zarr_path}")
    for f in tqdm(ds.sample):
        sample = ds.sel(sample=f, x=slice(0, 255), y=slice(0, 255))

        for t in sample["time"]:
            sample_t = sample.sel(time=t)
            sample_sum = sample_t.sum()
            is_data = "data" in sample_sum

            if is_data and sample_sum.data == 0:
                continue

            to_save = sample_t.drop_vars(["time", "sample"])

            if is_data:
                save_data = to_save["data"].squeeze()
            else:
                save_data = to_save["bands"]

            sample_name = sample_t["sample"].data
            out_path = zarr_path.parent / f"{sample_name}_{zarr_name}.tif"
            save_data.rio.to_raster(out_path)


def split_zarrs_in(data_path: pathlib.Path) -> None:
    zarr_paths = get_zarr_paths_from(data_path)

    for zarr_path in zarr_paths:
        split_zarr(zarr_path)

    make_split_files(data_path)


if __name__ == "__main__":
    split_zarrs_in(pathlib.Path('data'))
