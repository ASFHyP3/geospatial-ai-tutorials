from pathlib import Path
from collections.abc import Callable
from typing import Any

from albumentations.pytorch import ToTensorV2
import kornia.augmentation as K
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import zarr
from matplotlib.figure import Figure
from torch.utils.data import Subset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datamodules.utils import group_shuffle_split
from torchgeo.datasets import NonGeoDataset


class MultimodalNormalize(Callable):
    def __init__(self, means, stds):
        super().__init__()
        self.means = means
        self.stds = stds

    def __call__(self, batch):
        for m in self.means.keys():
            if m not in batch["image"]:
                continue
            image = batch["image"][m]
            if len(image.shape) == 4:
                # B, C, H, W
                means = torch.tensor(self.means[m], device=image.device).view(1, -1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(1, -1, 1, 1)
            elif len(image.shape) == 2 or len(image.shape) == 1:
                means = torch.tensor(self.means[m], device=image.device)
                stds = torch.tensor(self.stds[m], device=image.device)
            else:
                msg = (
                    f"Expected batch with 4 dimensions (B, C, H, W), sample with 2 dimensions (H, W) "
                    f"or a single channel, but got {len(image.shape)}"
                )
                raise Exception(msg)

            batch["image"][m] = (image - means) / stds
        return batch




# https://torchgeo.readthedocs.io/en/latest/tutorials/contribute_non_geo_dataset.html
class SatChipDataset(NonGeoDataset):
    def __init__(self, label_path, s2_path, rtc_path, transforms=None, split='train'):
        self.transforms = transforms
        self.slice_range = slice(3, 259)

        label_ds = _load_ds(label_path)
        s2_ds = _load_ds(s2_path)
        rtc_ds = _load_ds(rtc_path)

        n = len(label_ds.sample)
        train_end = int(0.8 * n)
        if split == 'train':
            split_slice = slice(0, train_end)
        elif split == 'test':
            split_slice = slice(train_end, n)
        else:
            raise ValueError(f'Invalid split: {split}')

        self.label_ds = label_ds.isel(sample=split_slice)
        samples = self.label_ds.sample

        # TODO: enforce this in satchip as well
        band_order = [
            'COASTAL',
            'BLUE',
            'GREEN',
            'RED',
            'REDEDGE1',
            'REDEDGE2',
            'REDEDGE3',
            'NIR',
            'NIR08',
            'NIR09',
            'SWIR16',
            'SWIR22',
        ]

        # self.s2_ds = s2_ds.isel(sample=split_slice).reindex(band=band_order)
        # self.rtc_ds = rtc_ds.isel(sample=split_slice).reindex(band=['VV', 'VH'])

        self.s2_ds = s2_ds.sel(sample=samples).reindex(sample=samples, band=band_order)
        self.rtc_ds = rtc_ds.sel(sample=samples).reindex(sample=samples, band=['VV', 'VH'])

    def __len__(self):
        return len(self.label_ds.sample)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self.transforms:
            self.transforms = ToTensorV2()

        sample_data = self.label_ds.isel(sample=index, x=self.slice_range, y=self.slice_range).squeeze()
        sample_array = sample_data.bands.data
        sample_id = str(sample_data.sample.data)

        image_output = {
            "S2L2A": self._get_image_array(sample_id, self.s2_ds),
            "S1RTC": self._get_image_array(sample_id, self.rtc_ds),
        }

        mask_output = self.transforms(image=sample_array)['image'][0]
        output = {'mask': mask_output, 'image': image_output}

        return output

    def _get_image_array(self, sample_id, ds):
        data = ds.sel(sample=sample_id).isel(x=self.slice_range, y=self.slice_range).squeeze()
        data = self._drop_empty_time_slices(data)
        array = data.squeeze().data.data

        array = np.transpose(array, (1, 2, 0))

        return self.transforms(image=array)['image']


    def _drop_empty_time_slices(self, ds: xr.Dataset) -> xr.Dataset:
        non_zero_count = ds.isel(band=0).where(ds.isel(band=0) != 0).count(dim=('x', 'y')).data.data
        if any(non_zero_count > 0):
            idx = [i for i, count in enumerate(non_zero_count) if count > 0]
        else:
            idx = [0]

        return ds.isel(time=idx)

    def normalize_image_array(self, input_array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """Function to normalize array values to a byte value between 0 and 255

        Args:
            input_array: The array to normalize.
            vmin: The minimum value to normalize to (mapped to 0).
            vmax: The maximum value to normalize to (mapped to 255).

        Returns:
            The normalized array.
        """
        input_array = input_array.astype(float)
        scaled_array = (input_array - vmin) / (vmax - vmin)
        scaled_array[np.isnan(input_array)] = 0
        normalized_array = np.round(np.clip(scaled_array, 0, 1) * 255).astype(np.uint8)
        return normalized_array

    def plot(self, sample: dict[str, Any], suptitle: str | None = None) -> Figure:
        mask = sample['mask']
        vv = self.normalize_image_array(np.sqrt(sample['image'][:, :, 0]), 0.14, 0.52)
        vh = self.normalize_image_array(np.sqrt(sample['image'][:, :, 1]), 0.05, 0.259)
        red = self.normalize_image_array(sample['image'][:, :, 5], 0, 3000)
        green = self.normalize_image_array(sample['image'][:, :, 4], 0, 3000)
        blue = self.normalize_image_array(sample['image'][:, :, 3], 0, 3000)
        rtc = np.stack([vv, vh, vv], axis=-1)
        rgb = np.stack([red, green, blue], axis=-1)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), layout='compressed')
        ax1.imshow(mask, cmap='gray')
        ax1.set_title('Labels')
        ax1.axis('off')
        ax2.imshow(rtc, cmap='gray')
        ax2.set_title('Sentinel-1 RTC')
        ax2.axis('off')
        ax3.imshow(rgb)
        ax3.set_title('Sentinel-2 RGB')
        ax3.axis('off')
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


def _load_ds(dataset_path: str | Path) -> xr.Dataset:
    store = zarr.storage.ZipStore(dataset_path, read_only=True)
    dataset = xr.open_zarr(store)
    return dataset


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
        means = {
            'S1RTC': torch.Tensor(self.s1rtc_mean),
            'S2L2A': torch.Tensor(self.s2l2a_mean)
        }

        stds = {
            'S1RTC': torch.Tensor(self.s1rtc_std),
            'S2L2A': torch.Tensor(self.s2l2a_std)
        }

        self.aug = MultimodalNormalize(means, stds)

        # you can also define specific augmentations for other experiment phases, if not specified
        # self.aug Augmentations will be applied
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
            self.train_dataset = Subset(dataset, train_indices).dataset
            self.val_dataset = Subset(dataset, val_indices).dataset
        if stage in ['test']:
            self.test_dataset = SatChipDataset(split='test', **self.kwargs)


if __name__ == '__main__':
    data_path = Path('data/zarrs')
    rtc_path = data_path / 'swathID_1507_swathDate_2020-07-06_S1RTC.zarr.zip'
    s2_path = data_path / 'swathID_1507_swathDate_2020-07-06_S2L2A.zarr.zip'
    label_path = data_path / 'swathID_1507_swathDate_2020-07-06.zarr.zip'
    sc_dataset = SatChipDataset(label_path, s2_path, rtc_path)
    # Testing dataset
    print(len(sc_dataset))
    data = sc_dataset[0]
    print(data)
    data = sc_dataset[10]
    f = sc_dataset.plot(data, suptitle='Sample 10')
    plt.show()
    # Testing module
    test_module = SatChipDataModule(batch_size=2, label_path=label_path, s2_path=s2_path, rtc_path=rtc_path)
    test_module.setup('fit')
    test_module.setup('test')
    print(len(test_module.train_dataset))
    print(len(test_module.val_dataset))
    print(len(test_module.test_dataset))
