from pathlib import Path

import albumentations
import matplotlib.pyplot as plt
import terratorch

dataset_path = Path('')

datamodule = terratorch.datamodules.GenericMultiModalDataModule(
    task="segmentation",
    batch_size=8,
    num_workers=2,
    num_classes=2,
    # Define your input modalities. The names must match the keys in the following dicts.
    modalities=["S2L2A", "S1RTC"],
    rgb_modality="S2L2A",  # Used for plotting. Defaults to the first modality if not provided.
    rgb_indices=[6, 2, 0],  # RGB channel positions in the rgb_modality.

    # Define data paths as dicts using the modality names as keys.
    train_data_root={
        "S2L2A": dataset_path / 'data/S2L2A',
        "S1RTC": dataset_path / 'data/S1RTC',
    },
    train_label_data_root=dataset_path / 'data/Label',
    val_data_root={
        "S2L2A": dataset_path / 'data/S2L2A',
        "S1RTC": dataset_path / 'data/S1RTC',
    },
    val_label_data_root=dataset_path / 'data/Label',
    test_data_root={
        "S2L2A": dataset_path / 'data/S2L2A',
        "S1RTC": dataset_path / 'data/S1RTC',
    },
    test_label_data_root=dataset_path / 'data/Label',

    # Define split files because all samples are saved in the same folder.
    train_split=dataset_path / "splits/train_data.txt",
    val_split=dataset_path / "splits/valid_data.txt",
    test_split=dataset_path / "splits/test_data.txt",

    # Define suffix, again using dicts.
    img_grep={
        "S2L2A": "*_S2L2A.zarr.zip",
        "S1GRD": "*_S1RTC.zarr.zip",
    },
    label_grep="*_Label.zarr.zip",

    # With TerraTorch, you can select a subset of the dataset bands as model inputs by providing dataset_bands (all bands in the data) and output_bands (selected bands). This setting is optional for all modalities and needs to be provided as dicts.
    # Here is an example for with S-1 GRD. You could change the output to ["VV"] to only train on the first band. Note that means and stds must be aligned with the output_bands (equal length of values).
    dataset_bands={
        "S1RTC": ["VV", "VH"]
    },
    output_bands={
        "S1RTC": ["VV", "VH"]
    },

    # Define standardization values. We use the pre-training values here and providing the additional modalities is not a problem, which makes it simple to experiment with different modality combinations. Alternatively, use the dataset statistics that you can generate using `terratorch compute_statistics -c config.yaml` (requires concat_bands: true for this multimodal datamodule).
    means={
        "S2L1C": [2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.560, 3306.481, 1473.847,
                  506.070, 2472.825, 1838.929],
        "S2L2A": [1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975, 2987.011, 3083.234, 3132.220, 3162.988,
                  2424.884, 1857.648],
        "S1GRD": [-12.599, -20.293],
        "S1RTC": [-10.93, -17.329],
        "RGB": [87.271, 80.931, 66.667],
        "DEM": [670.665]
    },
    stds={
        "S2L1C": [1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732, 1727.26, 1024.687,
                  442.165, 1331.411, 1160.419],
        "S2L2A": [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379,
                  1434.261, 1334.311],
        "S1GRD": [5.195, 5.890],
        "S1RTC": [4.391, 4.459],
        "RGB": [58.767, 47.663, 42.631],
        "DEM": [951.272],
    },

    # albumentations supports shared transformations and can handle multimodal inputs.
    train_transform=[
        albumentations.D4(),  # Random flips and rotation
        albumentations.pytorch.transforms.ToTensorV2(),
    ],
    val_transform=None,  # Using ToTensorV2() by default if not provided
    test_transform=None,

    no_label_replace=-1,  # Replace NaN labels. defaults to -1 which is ignored in the loss and metrics.
    no_data_replace=0,  # Replace NaN data
)

datamodule.setup("fit")

val_dataset = datamodule.val_dataset
len(val_dataset)
print(val_dataset)

# plotting a few samples (The code only plots the defined `rgb_modality`)
val_dataset.plot(val_dataset[0])
plt.show()
val_dataset.plot(val_dataset[8])
plt.show()
val_dataset.plot(val_dataset[11])
plt.show()

# checking datasets testing split size
datamodule.setup("test")
test_dataset = datamodule.test_dataset
len(test_dataset)
