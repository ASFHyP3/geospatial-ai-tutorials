from pathlib import Path
import warnings
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import terratorch
import loader

warnings.filterwarnings('ignore')
# pl.seed_everything(64)

MODALITIES = ['S2L2A', 'S1RTC']
CHIP_PATH = Path.home() / 'Data' / 'chips-large'
# CHIP_PATH = Path('chips')
TIMESTEPS = 1


def main():
    datamodule = loader.SatChipDataModule(
        batch_size=8,
        timesteps=TIMESTEPS,
        modalities=MODALITIES,
        chip_path=CHIP_PATH
    )

    model = terratorch.tasks.SemanticSegmentationTask(
        model_factory='EncoderDecoderFactory',
        model_args={
            'backbone': 'terramind_v1_base',
            'backbone_pretrained': True,
            'backbone_modalities': MODALITIES,
            'necks': [
                {'name': 'SelectIndices', 'indices': [2, 5, 8, 11]},
                {'name': 'ReshapeTokensToImage', 'remove_cls_token': False},
                {'name': 'LearnedInterpolateToPyramidal'}
            ],
            'decoder': 'UNetDecoder',
            'decoder_channels': [512, 256, 128, 64],
            'head_dropout': 0.1,
            'num_classes': 2,
        },
        loss='dice',
        optimizer='AdamW',
        lr=1e-5,
        ignore_index=-1,
        freeze_backbone=False,
        freeze_decoder=False,
        plot_on_val=False,
        class_names=['no-damage', 'damage'],
    )

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir='output/terramind_base_hwds/',
        name='tensorboard_logs',
    )

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='output/terramind_base_hwds/checkpoints/',
        filename='best-epoch{epoch:02d}-mIoU{val/mIoU:.4f}',
        monitor='val/mIoU',
        mode='max',
        save_top_k=1,
        save_last=True,
        save_weights_only=False,
        auto_insert_metric_name=False,
    )

    early_stop = EarlyStopping(
        monitor='val/mIoU',
        patience=10,
        mode='max',
        min_delta=0.001,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator='auto',
        strategy='auto',
        num_nodes=1,
        logger=tb_logger,
        max_epochs=50,
        log_every_n_steps=50,  # Increased from 40
        callbacks=[
            checkpoint_callback,
            early_stop,
            lr_monitor,
            pl.callbacks.RichProgressBar()
        ],
        default_root_dir='output/terramind_base_hwds/',
    )

    trainer.fit(model, datamodule=datamodule)

    print('\n✓ Training complete!')
    print(f'Best model: {checkpoint_callback.best_model_path}')
    print(f'Best mIoU: {checkpoint_callback.best_model_score:.4f}')


if __name__ == '__main__':
    main()
