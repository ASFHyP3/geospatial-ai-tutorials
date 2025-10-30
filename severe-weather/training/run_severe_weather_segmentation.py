from pathlib import Path
import warnings

import lightning.pytorch as pl
import terratorch
import loader

warnings.filterwarnings('ignore')

pl.seed_everything(64)

MODALITIES = ['S2L2A', 'S1RTC']
CHIP_PATH = Path('chips')
TIMESTEPS = 1

# Uncomment to run model with temporal data
# CHIP_PATH = Path('chips_timeseries')
# TIMESTEPS = 4


def main():
    datamodule = loader.SatChipDataModule(batch_size=4, timesteps=TIMESTEPS, modalities=MODALITIES, chip_path=CHIP_PATH)

    model = terratorch.tasks.SemanticSegmentationTask(
        model_factory='EncoderDecoderFactory',
        model_args={
            'backbone': 'terramind_v1_base',
            'backbone_pretrained': True,
            'backbone_modalities': MODALITIES,

            # Notebook with documentation on Temporal Wrapper
            # https://github.com/IBM/terratorch/blob/main/examples/notebooks/TemporalWrapper.ipynb

            # Uncomment these 2 lines when running with `chips_timeseries` dataset and use averaging
            # 'backbone_use_temporal': True,
            # 'backbone_temporal_pooling': 'mean',

            # An alternative to averaging temporally
            # 'backbone_use_temporal': True,
            # 'backbone_temporal_pooling': 'diff',
            # 'backbone_temporal_subset_lengths': [1, 2],

            'necks': [
                {
                    'name': 'SelectIndices',
                    'indices': [2, 5, 8, 11],
                },
                {
                    'name': 'ReshapeTokensToImage',
                    'remove_cls_token': False,
                },
                {
                    'name': 'LearnedInterpolateToPyramidal'
                },
            ],
            'decoder': 'UNetDecoder',
            'decoder_channels': [512, 256, 128, 64],
            'head_dropout': 0.1,
            'num_classes': 2,
        },
        loss='dice',
        optimizer='AdamW',
        lr=2e-5,
        ignore_index=-1,
        freeze_backbone=True,
        freeze_decoder=False,
        plot_on_val=False,
        class_names=['no-damage', 'damage'],
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='output/terramind_base_hwds/checkpoints/',
        mode='max',
        monitor='val/mIoU',
        filename='best-mIoU',
        save_weights_only=True,
    )

    trainer = pl.Trainer(
        accelerator='auto',
        strategy='auto',
        num_nodes=1,
        logger=True,
        max_epochs=2,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
        default_root_dir='output/terramind_base_hwds/',
    )

    trainer.fit(model, datamodule=datamodule)
    print('Script complete!')


if __name__ == '__main__':
    main()
