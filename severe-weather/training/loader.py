from collections.abc import Callable
from pathlib import Path
from typing import Any
from einops import rearrange

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


SATCHIP_MODALITIES = ('S2L2A', 'S1RTC', 'HLS')

# TODO: check that these are reasonable
S1RTC_MEAN = [-10.93, -17.329]
S1RTC_STD = [4.391, 4.459]
S2L2A_MEAN = [
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
S2L2A_STD = [
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
HLS_MEAN = [
    775.229,
    1080.992,
    1228.585,
    2497.202,
    2204.213,
    1610.832,
]
HLS_STD = [
    1281.526,
    1270.029,
    1399.480,
    1368.344,
    1291.676,
    1154.505,
]


class SatChipDataset(NonGeoDataset):
    def __init__(
        self,
        chip_path: Path,
        timesteps: int = 1,
        modalities: tuple[str] | None = None,
        transforms=A.NoOp(),
        split: str = 'train'
    ):
        assert timesteps > 0

        self.transforms = A.Compose([
            transforms,
            ToTensorV2(),
        ])

        split_path = chip_path / split
        label_path = split_path / 'LABEL'

        def make_chip_name(label_path):
            return '_'.join(label_path.name.split('_')[-4:]).split('.')[0]

        chip_names = [make_chip_name(label_path) for label_path in label_path.glob('*.zarr.zip')]
        data_dir_paths = [
            p for p in split_path.iterdir() if p.is_dir() and p.name in SATCHIP_MODALITIES
        ]

        chip_names = sorted(chip_names)

        self.chip_list = self._get_chip_list(chip_names, label_path, data_dir_paths)
        self.timesteps = timesteps

        chip_dir_modalities = [p.name for p in data_dir_paths]
        self.modalities = modalities or chip_dir_modalities

        assert all(m in SATCHIP_MODALITIES for m in self.modalities)
        mods, chip_dir = set(self.modalities), set(chip_dir_modalities)
        assert mods.issubset(chip_dir), f'Missing modalities in chip_path {list(mods - chip_dir)}. Found in chip directory {list(chip_dir)}.'

    def _get_chip_list(self, chip_names, label_path, data_dir_paths):
        chip_list = []

        for chip_name in chip_names:
            chip = {'LABEL': next(label_path.glob(f'*{chip_name}.zarr.zip'))}
            for data_dir in data_dir_paths:
                modality = data_dir.name
                data_path = next(data_dir.glob(f'*{chip_name}_{modality}.zarr.zip'))
                chip[modality] = data_path
            chip_list.append(chip)

        return chip_list

    def __len__(self):
        return len(self.chip_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        chip_paths = self.chip_list[index]

        label_array = self._load_mask(chip_paths['LABEL'])
        label = self.transforms(image=label_array)['image']

        image_output = {}
        for mod in self.modalities:
            array = self._load_image_array(chip_paths[mod])
            transformed = self._apply_transforms(array)

            image_output[mod] = transformed

        output = {'mask': label, 'image': image_output}

        return output

    def _apply_transforms(self, array: np.ndarray):
        print(array.shape)
        if len(array.shape) == 3:
            array = rearrange(array.squeeze(), 'channels height width -> height width channels')
            return self.transforms(image=array)['image']
        elif len(array.shape) == 4:
            timesteps = array.shape[0]
            flatten_temporal = rearrange(array, 'time channels height width -> height width (time channels)')
            transformed = self.transforms(image=flatten_temporal)['image']
            unflattened = rearrange(
                transformed,
                '(time channels) height width -> channels time height width',
                time=timesteps
            )

            return unflattened

    def _load_image_array(self, chip_path: Path) -> torch.Tensor:
        ds = self._load_ds(chip_path)
        array = ds.bands.values.astype(np.float32)

        return array

    def _load_mask(self, mask_path: Path):
        ds = self._load_ds(mask_path)
        array = ds.bands.isel(time=0).values.astype(np.float32)

        return array.squeeze()

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

            breakpoint()
            batch['image'][m] = (image - means) / stds
        return batch


class SatChipDataModule(NonGeoDataModule):
    def __init__(self, batch_size: int = 8, num_workers: int = 0, **kwargs: Any) -> None:
        super().__init__(SatChipDataset, batch_size, num_workers, **kwargs)

        self.training_transforms = A.Compose([A.CenterCrop(width=256, height=256), A.D4()])

        self.aug = MultimodalNormalize(
            means={
                'S1RTC': torch.Tensor(S1RTC_MEAN),
                'S2L2A': torch.Tensor(S2L2A_MEAN),
                'HLS': torch.Tensor(HLS_MEAN),
            },
            stds={
                'S1RTC': torch.Tensor(S1RTC_STD),
                'S2L2A': torch.Tensor(S2L2A_STD),
                'HLS': torch.Tensor(HLS_STD),
            }
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', or 'test'.
        """
        assert stage in ['fit', 'validate', 'test']
        if stage in ['fit', 'validate']:

            dataset = SatChipDataset(split='train', transforms=self.training_transforms, **self.kwargs)
            train_indices, val_indices = group_shuffle_split(range(len(dataset)), test_size=0.2, random_state=0)
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
            print(f'DATASETS: {len(self.train_dataset)}, {len(self.val_dataset)}')
        if stage in ['test']:
            self.test_dataset = SatChipDataset(split='val', **self.kwargs)


if __name__ == '__main__':
    chip_path = Path('chips')
    dataset = SatChipDataset(chip_path, split='train')
    # Testing dataset
    print(len(dataset))
    data = dataset[1]
    f = dataset.plot(data, suptitle='Sample 1')
    plt.show()
    # Testing module
    test_module = SatChipDataModule(batch_size=2, chip_path=chip_path)
    test_module.setup('fit')
    test_module.setup('test')
    print(len(test_module.train_dataset))
    print(len(test_module.val_dataset))
    print(len(test_module.test_dataset))
