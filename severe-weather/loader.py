from pathlib import Path
from typing import Any

import kornia.augmentation as K
import numpy as np
import torch
import xarray as xr
import zarr
from matplotlib.figure import Figure
from torch.utils.data import Subset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datamodules.utils import group_shuffle_split
from torchgeo.datasets import NonGeoDataset


# https://torchgeo.readthedocs.io/en/latest/tutorials/contribute_non_geo_dataset.html
class SatChipDataset(NonGeoDataset):
    def __init__(self, label_path, s2_path, rtc_path, split='train'):
        self.label_path = label_path
        label_ds = self._load_ds(self.label_path)
        self.s2_path = s2_path
        s2_ds = self._load_ds(self.s2_path)
        self.rtc_path = rtc_path
        rtc_ds = self._load_ds(self.rtc_path)

        n = len(label_ds.sample)
        train_end = int(0.8 * n)
        if split == 'train':
            split_slice = slice(0, train_end)
        elif split == 'test':
            split_slice = slice(train_end, n)
        else:
            raise ValueError(f'Invalid split: {split}')
        self.label_ds = label_ds.isel(sample=split_slice)
        self.s2_ds = s2_ds.isel(sample=split_slice)
        self.rtc_ds = rtc_ds.isel(sample=split_slice)

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
        image_array = np.transpose(np.vstack([rtc_array, s2_array]), (1, 2, 0))
        return {'mask': sample_array, 'image': image_array}

    def _load_ds(self, label_path: str | Path) -> xr.Dataset:
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


class SatChipDataModule(NonGeoDataModule):
    # TODO: check that these are reasonable
    s1rtc_mean = [-10.93, -17.329]
    s1rtc_std = [4.391, 4.459]
    s2l2a_mean = [
        1390.458,
        1503.317,
        1718.197,
        1853.910,
        2199.100,
        2779.975,
        2987.011,
        3083.234,
        3132.220,
        3162.988,
        2424.884,
        1857.648,
    ]
    s2l2a_std = [
            2106.761,
            2141.107,
            2038.973,
            2134.138,
            2085.321,
            1889.926,
            1820.257,
            1871.918,
            1753.829,
            1797.379,
            1434.261,
            1334.311,
        ]

    def __init__(self, batch_size: int = 8, num_workers: int = 0, **kwargs: Any) -> None:
        super().__init__(SatChipDataset, batch_size, num_workers, **kwargs)
        # you can specify a series of Kornia augmentations that will be
        # applied to a batch of training data in `on_after_batch_transfer` in the NonGeoDataModule base class
        means = torch.Tensor(self.s1rtc_mean + self.s2l2a_mean)
        stds = torch.Tensor(self.s1rtc_std + self.s2l2a_std)
        self.train_aug = K.AugmentationSequential(
            K.Normalize(means, stds),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=None,
            keepdim=True,
        )

        # you can also define specific augmentations for other experiment phases, if not specified
        # self.aug Augmentations will be applied
        self.aug = K.AugmentationSequential(K.Normalize(means, stds), data_keys=None, keepdim=True)
        self.size = 256

    # setup defines how the dataset should be split
    # this could either be predefined from the dataset authors or
    # done in a prescribed way if some or no splits are specified
    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', or 'test'.
        """
        assert stage in ['fit', 'validate', 'test']
        if stage in ['fit', 'validate']:
            dataset = SatChipDataset(split='train', **self.kwargs)
            train_indices, val_indices = group_shuffle_split(range(len(dataset)), test_size=0.2, random_state=0)
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
        if stage in ['test']:
            self.test_dataset = SatChipDataset(split='test', **self.kwargs)


if __name__ == '__main__':
    name_prefix = 'swathID_638_swathDate_2020-06-04'
    label_path = f'{name_prefix}.zarr.zip'
    s2_path = f'{name_prefix}_S2L2A.zarr.zip'
    rtc_path = f'{name_prefix}_S1RTC.zarr.zip'
    sc_dataset = SatChipDataset(label_path, s2_path, rtc_path)
    # Testing dataset
    print(len(sc_dataset))
    data = sc_dataset[0]
    # Testing module
    test_module = SatChipDataModule(batch_size=2, label_path=label_path, s2_path=s2_path, rtc_path=rtc_path)
    test_module.setup('fit')
    test_module.setup('test')
    print(len(test_module.train_dataset))
    print(len(test_module.val_dataset))
    print(len(test_module.test_dataset))
