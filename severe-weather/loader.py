from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
import zarr
from matplotlib.figure import Figure
from torchgeo.datasets import NonGeoDataset


# https://torchgeo.readthedocs.io/en/latest/tutorials/contribute_non_geo_dataset.html
class SatChipDataset(NonGeoDataset):
    def __init__(self, label_path, s2_path, rtc_path):
        self.label_path = label_path
        self.label_ds = self._load_ds(self.label_path)
        self.s2_path = s2_path
        self.s2_ds = self._load_ds(self.s2_path)
        self.rtc_path = rtc_path
        self.rtc_ds = self._load_ds(self.rtc_path)

    def __len__(self):
        return len(self.label_ds.sample)

    def __getitem__(self, index: int) -> dict[str, Any]:
        slice_range = slice(3, 259)
        sample_data = self.label_ds.isel(sample=index, x=slice_range, y=slice_range).squeeze()
        sample_array = sample_data.bands.data
        sample_id = str(sample_data.sample.data)
        rtc_data = self.rtc_ds.sel(sample=sample_id).isel(x=slice_range, y=slice_range).squeeze()
        rtc_data = self._drop_empty_time_slices(rtc_data)
        rtc_array = rtc_data.squeeze().data.data
        s2_data = self.s2_ds.sel(sample=sample_id).isel(x=slice_range, y=slice_range).squeeze()
        s2_data = self._drop_empty_time_slices(s2_data)
        s2_array = s2_data.squeeze().data.data
        image_array = np.vstack([rtc_array, s2_array])
        return {'mask': sample_array, 'image': image_array}

    def _load_ds(self, label_path: str | Path) -> xr.Dataset:
        """Load a zipped zarr archive"""
        store = zarr.storage.ZipStore(label_path, read_only=True)
        dataset = xr.open_zarr(store)
        return dataset

    def _drop_empty_time_slices(self, ds: xr.Dataset) -> xr.Dataset:
        non_zero_count = ds.isel(band=0).where(ds.isel(band=0) != 0).count(dim=('x', 'y')).data.data
        assert any(non_zero_count > 0), 'All time slices are empty'
        non_zero_indexes = [i for i, count in enumerate(non_zero_count) if count > 0]
        return ds.isel(time=non_zero_indexes)

    def plot(self) -> Figure:
        raise NotImplementedError


if __name__ == '__main__':
    name_prefix = 'swathID_638_swathDate_2020-06-04'
    label_path = f'{name_prefix}.zarr.zip'
    s2_path = f'{name_prefix}_S2L2A.zarr.zip'
    rtc_path = f'{name_prefix}_S1RTC.zarr.zip'
    sc_dataset = SatChipDataset(label_path, s2_path, rtc_path)
    print(len(sc_dataset))
    data = sc_dataset[0]
    breakpoint()
