import warnings

import numpy as np

import matplotlib.pyplot as plt

import torch
import lightning.pytorch as pl
import albumentations

import terratorch
from terratorch.datamodules import GenericNonGeoSegmentationDataModule


means = [0.03528563, 0.00850812]
stds = [0.01947893, 0.00549653]


def plot_sample(sample):
    data = sample["image"].cpu().numpy()
    mask = sample["mask"].cpu().numpy()

    def normalize_image_array(
        input_array: np.ndarray, vmin: float, vmax: float
    ) -> np.ndarray:
        input_array = input_array.astype(float)
        scaled_array = (input_array - vmin) / (vmax - vmin)
        scaled_array[np.isnan(input_array)] = 0
        normalized_array = np.round(np.clip(scaled_array, 0, 1) * 255).astype(np.uint8)

        return normalized_array

    vv = normalize_image_array(np.sqrt(data[0]), 0.14, 0.52)
    vh = normalize_image_array(np.sqrt(data[1]), 0.05, 0.259)
    img = np.stack([vv, vh, vv], axis=-1)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img)
    ax[0].set_title("Image")
    ax[0].axis("off")
    ax[1].imshow(mask, vmin=-1, vmax=1, interpolation="nearest")
    ax[1].set_title("Mask")
    ax[1].axis("off")
    ax[2].imshow(img)
    ax[2].imshow(mask, vmin=-1, vmax=1, interpolation="nearest", alpha=0.5)
    ax[2].set_title("Mask on Image")
    ax[2].axis("off")
    fig.tight_layout()
    plt.show()


def main():
    warnings.filterwarnings("ignore")
    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size=16,
        num_workers=2,
        num_classes=2,
        rgb_indices=[0, 1, 0],
        # TODO: Define the data and label paths
        train_data_root="hwds/CHIPS_TM",
        train_label_data_root="hwds/CHIPS_TM",
        val_data_root="hwds/CHIPS_TM",
        val_label_data_root="hwds/CHIPS_TM",
        test_data_root="hwds/CHIPS_TM",
        test_label_data_root="hwds/CHIPS_TM",
        # TODO: Define the split files
        train_split="hwds/SPLITS/train.txt",
        val_split="hwds/SPLITS/val.txt",
        test_split="hwds/SPLITS/test.txt",
        # TODO: Define suffixs
        img_grep="*.BANDS.tif",
        label_grep="*.MASK.tif",
        # TODO: Update the standardization values. They need to be the same length as the data.
        #  You can define a constant_scale that applies a multiplicator in case the data does not align with the the standardization values.
        # constant_scale=None,
        means=means,
        stds=stds,
        # fyi TerraMind pretraining values (assuming data in range 0-10000)
        # S2L2A means: [1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975, 2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648]
        # S2L2A stds: [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311]
        # albumentations supports shared transformations and can handle multimodal inputs.
        train_transform=[
            albumentations.D4(),  # Random flips and rotation
            albumentations.pytorch.transforms.ToTensorV2(),
        ],
        val_transform=None,  # Fallback to ToTensor
        test_transform=None,
        no_label_replace=-1,  # Replace NaN labels. defaults to -1 which is ignored in the loss and metrics.
        no_data_replace=0,  # Replace NaN data
    )

    # Setup train and val datasets
    datamodule.setup("fit")

    # checking datasets validation split size
    val_dataset = datamodule.val_dataset
    len(val_dataset)

    # Plotting a few samples
    for x in range(5):
        plot_sample(val_dataset[x])

    warnings.filterwarnings("ignore")

    pl.seed_everything(0)

    # By default, TerraTorch saves the model with the best validation loss. You can overwrite this by defining a custom ModelCheckpoint, e.g., saving the model with the highest validation mIoU.
    # TODO Optionally adjust the checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="output/terramind_hwds_base",
        mode="max",
        monitor="val/mIoU",  # Variable to monitor
        filename="best-mIoU",
        save_weights_only=True,
    )

    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        devices=1,  # Deactivate multi-gpu because it often fails in notebooks
        precision="16-mixed",  # Speed up training with half precision, delete for full precision training.
        num_nodes=1,
        logger=True,  # Uses TensorBoard by default
        max_epochs=100,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
        # TODO Define output dir
        default_root_dir="output/terramind_hwds_base",
    )

    # Segmentation mask that build the model and handles training and validation steps.
    model = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",  # Combines a backbone with necks, the decoder, and a head
        model_args={
            # TerraMind backbone
            "backbone": "terramind_v1_base",
            "backbone_pretrained": True,
            # TODO Select the modality. Check the docs for details https://terrastackai.github.io/terratorch/stable/guide/terramind/#subset-of-input-bands
            "backbone_modalities": ["S1RTC"],
            # TODO define the input bands. This is only needed because you need to select a subset of the pre-training bands for Burn Scars.
            #  Check the names in the "List of pre-trained bands" in the docs.
            "backbone_bands": {
                "S1RTC": ['VV', 'VH']
            },
            # Necks
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": [2, 5, 8, 11],  # indices for terramind_v1_base
                    # "indices": [5, 11, 17, 23] # indices for terramind_v1_large
                },
                {
                    "name": "ReshapeTokensToImage",
                    "remove_cls_token": False,
                },  # TerraMind is trained without CLS token, which neads to be specified.
                {
                    "name": "LearnedInterpolateToPyramidal"
                },  # Some decoders like UNet or UperNet expect hierarchical features. Therefore, we need to learn a upsampling for the intermediate embedding layers when using a ViT like TerraMind.
            ],
            # Decoder
            "decoder": "UNetDecoder",
            # "decoder_channels": [256, 128, 64, 32],
            "decoder_channels": [256, 128, 64, 32],
            # Head
            "head_dropout": 0.1,
            "num_classes": 2,
        },
        loss="dice",  # We recommend dice for binary tasks and ce for tasks with multiple classes.
        optimizer="AdamW",
        lr=2e-5,  # The optimal learning rate varies between datasets, we recommend testing different once between 1e-5 and 1e-4. You can perform hyperparameter optimization using terratorch-iterate.
        ignore_index=-1,
        freeze_backbone=False,  # Only used to speed up fine-tuning in this demo, we highly recommend fine-tuning the backbone for the best performance.
        freeze_decoder=False,  # Should be false in most cases as the decoder is randomly initialized.
        plot_on_val=True,  # Plot predictions during validation steps
        class_names=["Others", "Damage"],  # optionally define class names
    )

    # checking datasets testing split size

    datamodule.setup("train")
    train_dataset = datamodule.train_dataset
    print("train_dataset:", len(train_dataset))

    # Training
    trainer.fit(model, datamodule=datamodule)

    # Let's test the fine-tuned model
    best_ckpt_path = "output/terramind_hwds_base/best-mIoU.ckpt"
    trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt_path)

    # Now we can use the model for predictions and plotting
    # from https://github.com/IBM/terramind/blob/main/notebooks%2Fterramind_v1_small_multitemporal_crop.ipynb
    # rest of examples in notebook from
    # https://github.com/IBM/terramind/blob/main/notebooks%2Fterramind_v1_small_burnscars.ipynb
    model = terratorch.tasks.SemanticSegmentationTask.load_from_checkpoint(
        best_ckpt_path,
        model_factory=model.hparams.model_factory,
        model_args=model.hparams.model_args,
    )

    datamodule.setup("test")
    test_dataset = datamodule.test_dataset
    test_loader = datamodule.test_dataloader()
    print("test_dataset:", len(test_dataset))

    with torch.no_grad():
        batch = next(iter(test_loader))
        images = batch["image"]
        images = images.to(model.device)
        masks = batch["mask"].numpy()

        with torch.no_grad():
            outputs = model(images)

        preds = torch.argmax(outputs.output, dim=1).cpu().numpy()

    for i in range(5):
        sample = {
            "image": batch["image"][i].cpu(),
            "mask": batch["mask"][i],
            "prediction": preds[i],
        }
        test_dataset.plot(sample)
        plt.show()

    # Note: This demo only trains for 5 epochs by default, which does not result in good predictions.


if __name__ == "__main__":
    main()
