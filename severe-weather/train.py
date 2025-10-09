from pathlib import Path
import warnings

import lightning.pytorch as pl
import terratorch
import torch
from terratorch.registry import BACKBONE_REGISTRY
import loader

warnings.filterwarnings('ignore')

print('Starting run...')
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
print('Torch CUDA version:', torch.version.cuda)
print('Compiled with CUDA:', torch.backends.cudnn.is_available())

data_path = Path('data/zarrs')
rtc_path = data_path / 'swathID_1507_swathDate_2020-07-06_S1RTC.zarr.zip'
s2_path = data_path / 'swathID_1507_swathDate_2020-07-06_S2L2A.zarr.zip'
label_path = data_path / 'swathID_1507_swathDate_2020-07-06.zarr.zip'

datamodule = loader.SatChipDataModule(batch_size=8, label_path=label_path, s2_path=s2_path, rtc_path=rtc_path)

# Setup train and val datasets
datamodule.setup('fit')

# checking datasets validation split size
val_dataset = datamodule.val_dataset
print(f'Length of validation dataset: {len(val_dataset)}')

# Use the backbone registry to load a PyTorch model for custom pipeline. The pre-trained weights are automatically downloaded with pretrained=True.
model = BACKBONE_REGISTRY.build(
    'terramind_v1_base',
    modalities=["S2L2A", "S1RTC"],
    pretrained=True,
)

print(model)
pl.seed_everything(64)

# By default, TerraTorch saves the model with the best validation loss. You can overwrite this by defining a custom ModelCheckpoint, e.g., saving the model with the highest validation mIoU.
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath='output/terramind_base_hwds/checkpoints/',
    mode='max',
    monitor='val/mIoU',  # Variable to monitor
    filename='best-mIoU',
    save_weights_only=True,
)

# Lightning Trainer
trainer = pl.Trainer(
    accelerator='auto',
    strategy='auto',
    num_nodes=1,
    logger=True,  # Uses TensorBoard by default
    max_epochs=2,  # For demos
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
    default_root_dir='output/terramind_base_hwds/',
)

# Segmentation mask that build the model and handles training and validation steps.
model = terratorch.tasks.SemanticSegmentationTask(
    model_factory='EncoderDecoderFactory',  # Combines a backbone with necks, the decoder, and a head
    model_args={
        # TerraMind backbone
        'backbone': 'terramind_v1_base',  # large version: terramind_v1_large
        'backbone_pretrained': True,
        'backbone_modalities': ['S2L2A', 'S1RTC'],
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
    plot_on_val=False,  # Plot predictions during validation steps
    class_names=['no-damage', 'damage'],  # optionally define class names
)

print('Starting training...')
trainer.fit(model, datamodule=datamodule)
print('Done!')
