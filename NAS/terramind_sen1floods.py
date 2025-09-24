import os
import warnings
from pathlib import Path

import albumentations
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import terratorch
import torch
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.registry import BACKBONE_REGISTRY


warnings.filterwarnings('ignore')
dataset_path = Path('sen1floods11_v1.1')

print('Starting run...')
datamodule = terratorch.datamodules.GenericMultiModalDataModule(
    task='segmentation',
    batch_size=8,
    num_workers=2,
    num_classes=2,
    # Define your input modalities. The names must match the keys in the following dicts.
    modalities=['S2L1C', 'S1GRD'],
    rgb_modality='S2L1C',  # Used for plotting. Defaults to the first modality if not provided.
    rgb_indices=[3, 2, 1],  # RGB channel positions in the rgb_modality.
    # Define data paths as dicts using the modality names as keys.
    train_data_root={
        'S2L1C': dataset_path / 'data/S2L1CHand',
        'S1GRD': dataset_path / 'data/S1GRDHand',
    },
    train_label_data_root=dataset_path / 'data/LabelHand',
    val_data_root={
        'S2L1C': dataset_path / 'data/S2L1CHand',
        'S1GRD': dataset_path / 'data/S1GRDHand',
    },
    val_label_data_root=dataset_path / 'data/LabelHand',
    test_data_root={
        'S2L1C': dataset_path / 'data/S2L1CHand',
        'S1GRD': dataset_path / 'data/S1GRDHand',
    },
    test_label_data_root=dataset_path / 'data/LabelHand',
    # Define split files because all samples are saved in the same folder.
    train_split=dataset_path / 'splits/flood_train_data.txt',
    val_split=dataset_path / 'splits/flood_valid_data.txt',
    test_split=dataset_path / 'splits/flood_test_data.txt',
    # Define suffix, again using dicts.
    img_grep={
        'S2L1C': '*_S2Hand.tif',
        'S1GRD': '*_S1Hand.tif',
    },
    label_grep='*_LabelHand.tif',
    # With TerraTorch, you can select a subset of the dataset bands as model inputs by providing dataset_bands (all bands in the data) and output_bands (selected bands). This setting is optional for all modalities and needs to be provided as dicts.
    # Here is an example for with S-1 GRD. You could change the output to ["VV"] to only train on the first band. Note that means and stds must be aligned with the output_bands (equal length of values).
    dataset_bands={'S1GRD': ['VV', 'VH']},
    output_bands={'S1GRD': ['VV', 'VH']},
    # Define standardization values. We use the pre-training values here and providing the additional modalities is not a problem, which makes it simple to experiment with different modality combinations. Alternatively, use the dataset statistics that you can generate using `terratorch compute_statistics -c config.yaml` (requires concat_bands: true for this multimodal datamodule).
    means={
        'S2L1C': [
            2357.089,
            2137.385,
            2018.788,
            2082.986,
            2295.651,
            2854.537,
            3122.849,
            3040.560,
            3306.481,
            1473.847,
            506.070,
            2472.825,
            1838.929,
        ],
        'S2L2A': [
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
        ],
        'S1GRD': [-12.599, -20.293],
        'S1RTC': [-10.93, -17.329],
        'RGB': [87.271, 80.931, 66.667],
        'DEM': [670.665],
    },
    stds={
        'S2L1C': [
            1624.683,
            1675.806,
            1557.708,
            1833.702,
            1823.738,
            1733.977,
            1732.131,
            1679.732,
            1727.26,
            1024.687,
            442.165,
            1331.411,
            1160.419,
        ],
        'S2L2A': [
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
        ],
        'S1GRD': [5.195, 5.890],
        'S1RTC': [4.391, 4.459],
        'RGB': [58.767, 47.663, 42.631],
        'DEM': [951.272],
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

# Setup train and val datasets
datamodule.setup('fit')

# checking datasets validation split size
val_dataset = datamodule.val_dataset
len(f'Length of validation dataset: {val_dataset}')

# checking datasets testing split size
datamodule.setup('test')
test_dataset = datamodule.test_dataset
len(f'Length of test dataset: {val_dataset}')

# Use the backbone registry to load a PyTorch model for custom pipeline. The pre-trained weights are automatically downloaded with pretrained=True.
model = BACKBONE_REGISTRY.build(
    'terramind_v1_base',
    modalities=['S2L1C', 'S1GRD'],
    pretrained=True,
)

print(model)

pl.seed_everything(64)

# By default, TerraTorch saves the model with the best validation loss. You can overwrite this by defining a custom ModelCheckpoint, e.g., saving the model with the highest validation mIoU.
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='output/terramind_base_sen1floods11/checkpoints/',
    mode='max',
    monitor='val/mIoU',  # Variable to monitor
    filename='best-mIoU',
    save_weights_only=True,
)

# Lightning Trainer
trainer = pl.Trainer(
    accelerator='auto',
    strategy='auto',
    devices=1,  # Deactivate multi-gpu because it often fails in notebooks
    num_nodes=1,
    logger=True,  # Uses TensorBoard by default
    max_epochs=3,  # For demos
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
    default_root_dir='output/terramind_base_sen1floods11/',
)

# Segmentation mask that build the model and handles training and validation steps.
model = terratorch.tasks.SemanticSegmentationTask(
    model_factory='EncoderDecoderFactory',  # Combines a backbone with necks, the decoder, and a head
    model_args={
        # TerraMind backbone
        'backbone': 'terramind_v1_base',  # large version: terramind_v1_large
        'backbone_pretrained': True,
        'backbone_modalities': ['S2L1C', 'S1GRD'],
        # Optionally, define the input bands. This is only needed if you select a subset of the pre-training bands, as explained above.
        # "backbone_bands": {"S1GRD": ["VV"]},
        # Necks
        'necks': [
            {
                'name': 'SelectIndices',
                'indices': [2, 5, 8, 11],  # indices for terramind_v1_base
                # "indices": [5, 11, 17, 23] # indices for terramind_v1_large
            },
            {
                'name': 'ReshapeTokensToImage',
                'remove_cls_token': False,
            },  # TerraMind is trained without CLS token, which neads to be specified.
            {
                'name': 'LearnedInterpolateToPyramidal'
            },  # Some decoders like UNet or UperNet expect hierarchical features. Therefore, we need to learn a upsampling for the intermediate embedding layers when using a ViT like TerraMind.
        ],
        # Decoder
        'decoder': 'UNetDecoder',
        'decoder_channels': [512, 256, 128, 64],
        # Head
        'head_dropout': 0.1,
        'num_classes': 2,
    },
    loss='dice',  # We recommend dice for binary tasks and ce for tasks with multiple classes.
    optimizer='AdamW',
    lr=2e-5,  # The optimal learning rate varies between datasets, we recommend testing different once between 1e-5 and 1e-4. You can perform hyperparameter optimization using terratorch-iterate.
    ignore_index=-1,
    freeze_backbone=True,  # Only used to speed up fine-tuning in this demo, we highly recommend fine-tuning the backbone for the best performance.
    freeze_decoder=False,  # Should be false in most cases as the decoder is randomly initialized.
    plot_on_val=True,  # Plot predictions during validation steps
    class_names=['Others', 'Water'],  # optionally define class names
)

if __name__ == '__main__':
    trainer.fit(model, datamodule=datamodule)
    print('done!')
