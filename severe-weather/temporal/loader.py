from collections.abc import Callable
from pathlib import Path
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import zarr
from albumentations.pytorch import ToTensorV2
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
            if m not in batch['image']:
                continue
            image = batch['image'][m]
            if len(image.shape) == 5:
                # B, C, T, H, W
                means = torch.tensor(self.means[m], device=image.device).view(1, -1, 1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(1, -1, 1, 1, 1)
            elif len(image.shape) == 4:
                # B, C, H, W
                means = torch.tensor(self.means[m], device=image.device).view(1, -1, 1, 1)
                stds = torch.tensor(self.stds[m], device=image.device).view(1, -1, 1, 1)
            elif len(image.shape) == 2 or len(image.shape) == 1:
                means = torch.tensor(self.means[m], device=image.device)
                stds = torch.tensor(self.stds[m], device=image.device)
            else:
                msg = (
                    f'Expected batch with 4 dimensions (B, C, H, W), sample with 2 dimensions (H, W) '
                    f'or a single channel, but got {image.shape}'
                )
                raise Exception(msg)

            batch['image'][m] = (image - means) / stds
        return batch


# https://torchgeo.readthedocs.io/en/latest/tutorials/contribute_non_geo_dataset.html
class SatChipTemporalDataset(NonGeoDataset):
    def __init__(self, chip_path, transforms=None, split='train'):
        if not transforms:
            self.transforms = A.Compose([A.CenterCrop(width=256, height=256), ToTensorV2()])
        else:
            self.transforms = transforms
        split_path = chip_path / split
        label_path = split_path / 'LABEL'
        chip_names = ['_'.join(x.name.split('_')[-4:]).split('.')[0] for x in label_path.glob('*.zarr.zip')]
        #  ['465U_866L_3_3', '464U_867L_3_0', '464U_867L_2_0', '465U_865L_0_3']
        modalities = ['S2L2A', 'S1RTC', 'HLS']
        data_dir_paths = [
            p for p in split_path.iterdir() if p.is_dir() and p.name in modalities
        ]

        chip_names = sorted(chip_names)
        self.chip_list = []

        for chip_name in chip_names:
            chip = {'LABEL': next(label_path.glob(f'*{chip_name}.zarr.zip'))}
            for data_dir in data_dir_paths:
                modality = data_dir.name
                data_path = next(data_dir.glob(f'*{chip_name}_{modality}.zarr.zip'))
                chip[modality] = data_path
            self.chip_list.append(chip)

    def __len__(self):
        return len(self.chip_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        chip_paths = self.chip_list[index]
        label = self._get_image_array(chip_paths['LABEL']).squeeze()
        image_output = {
            'S1RTC': self._get_image_array(chip_paths['S1RTC']),
            'S2L2A': self._get_image_array(chip_paths['S2L2A']),
        }
        # B, C, T, H, W
        output = {'mask': label, 'image': image_output}
        return output

    def _get_image_array(self, chip_path: Path) -> torch.Tensor:
        ds = self._load_ds(chip_path)
        crop = slice(0, 256)
        array = ds.bands.isel(x=crop, y=crop, time=slice(0, 4)).values.astype(np.float32)
        array = np.transpose(array, (1, 0, 2, 3))
        tensor = torch.from_numpy(array)
        return tensor

    def _load_ds(self, dataset_path: str | Path) -> xr.Dataset:
        store = zarr.storage.ZipStore(dataset_path, read_only=True)  # type: ignore
        dataset = xr.open_zarr(store)
        return dataset

    def normalize_image_array(self, input_array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """Function to normalize array values to a byte value between 0 and 255

        Args:
            input_array: The array to normalize.
            vmin: The minimum value to normalize to (mapped to 0).
            vmax: The maximum value to normalize to (mapped to 255).

        Returns:
            The normalized array.
        """
        scaled_array = (input_array - vmin) / (vmax - vmin)
        scaled_array[np.isnan(input_array)] = 0
        normalized_array = np.round(np.clip(scaled_array, 0, 1) * 255).astype(np.uint8)
        return normalized_array

    def plot(self, sample: dict[str, Any], suptitle: str | None = None) -> Figure:
        mask = sample['mask']

        vv = self.normalize_image_array(np.sqrt(sample['image']['S1RTC'].numpy()[0, :, :]), 0.14, 0.52)
        vh = self.normalize_image_array(np.sqrt(sample['image']['S1RTC'].numpy()[1, :, :]), 0.05, 0.259)

        red = self.normalize_image_array(sample['image']['S2L2A'].numpy()[3, :, :], 0, 3000)
        green = self.normalize_image_array(sample['image']['S2L2A'].numpy()[2, :, :], 0, 3000)
        blue = self.normalize_image_array(sample['image']['S2L2A'].numpy()[1, :, :], 0, 3000)

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


class SatChipTemporalDataModule(NonGeoDataModule):
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
        super().__init__(SatChipTemporalDataset, batch_size, num_workers, **kwargs)
        means = {'S1RTC': torch.Tensor(self.s1rtc_mean), 'S2L2A': torch.Tensor(self.s2l2a_mean)}
        stds = {'S1RTC': torch.Tensor(self.s1rtc_std), 'S2L2A': torch.Tensor(self.s2l2a_std)}
        self.training_transforms = A.Compose([A.CenterCrop(width=256, height=256), A.D4(), ToTensorV2()])

        # you can also define specific augmentations for other experiment phases, if not specified
        # self.aug Augmentations will be applied
        self.aug = MultimodalNormalize(means, stds)
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

            dataset = SatChipTemporalDataset(split='train', **self.kwargs)
            train_indices, val_indices = group_shuffle_split(range(len(dataset)), test_size=0.2, random_state=0)
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
            print(f'DATASETS: {len(self.train_dataset)}, {len(self.val_dataset)}')
        if stage in ['test']:
            self.test_dataset = SatChipTemporalDataset(split='val', **self.kwargs)


if __name__ == '__main__':
    chip_path = Path.home() / 'Data' / 'severe-weather-temporal' / 'chips'
    dataset = SatChipTemporalDataset(chip_path, split='train')
    # Testing dataset
    print(len(dataset))
    data = dataset[1]
    # Testing module
    test_module = SatChipTemporalDataModule(batch_size=2, chip_path=chip_path)
    test_module.setup('fit')
    test_module.setup('test')
    print(len(test_module.train_dataset))
    print(len(test_module.val_dataset))
    print(len(test_module.test_dataset))
