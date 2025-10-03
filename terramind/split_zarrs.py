import pathlib

import xarray as xr
import zarr
from tqdm import tqdm


def get_zarr_paths_from(data_dir: pathlib.Path) -> list[pathlib.Path]:
    return list(data_dir.rglob("*.zarr.zip"))


def split_zarr(zarr_path: pathlib.Path) -> None:
    zarr_name = zarr_path.name.split('.zarr.zip')[0]
    store = zarr.storage.ZipStore(zarr_path, read_only=True)
    ds = xr.open_zarr(store, mask_and_scale=True)

    for f in tqdm(ds.sample):
        sample = ds.sel(sample=f)

        for t in sample['time']:
            sample_t = sample.sel(time=t)
            sample_sum = sample_t.sum()
            is_data = 'data' in sample_sum

            if is_data and sample_t.sum().data == 0:
                continue

            to_save = sample_t.drop_vars(['time', 'sample'])

            if is_data:
                save_data = to_save['data'].squeeze()
            else:
                save_data = to_save['bands']

            sample_name = sample_t['sample'].data
            out_path = zarr_path.parent / f'{sample_name}_{zarr_name}.tif'
            save_data.rio.to_raster(out_path)


if __name__ == '__main__':
    zarr_path = pathlib.Path('data/S1RTC/swathID_1507_swathDate_2020-07-06_S1RTC.zarr.zip')
    data_dir = pathlib.Path('data')
    label_path = pathlib.Path('data/Label/swathID_1507_swathDate_2020-07-06.zarr.zip')

    zarr_paths = get_zarr_paths_from(data_dir)
    for zarr_path in zarr_paths:
        split_zarr(zarr_path)
