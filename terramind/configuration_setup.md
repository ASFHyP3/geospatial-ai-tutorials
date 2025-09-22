# Congfiguration YAML Setup
Terratorch requires a configuration `yaml` file when run in the terminal. 
This document will walk through key parameters, describe them, and provide 
default values for the purposes of ASF's foundational model leveraging the TerraMind model. 

### Trainer
[`Trainer` is a Lightning parameter](https://lightning.ai/docs/pytorch/stable/common/trainer.html) that handles loop details.

| *Parameter* | *Description* | *Default*                               |
| :---- | :---- |:----------------------------------------|
| `accelerator` | Device used to run | `auto`                                  |
| `strategy` | Kind of parallelism | `auto`                                  |
| `devices` | Available devices | `auto`                                  |
| `num_nodes` | Number of nodes | `1`                                     |
| `precision` | Kind of precision | `16-mixed`                              |
| `logger/class_path` | [Logging format](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) | `TensorBoardLogger`                     |
| `logger/init_args/save_dir` | File for log |                                         |
| `logger/init_args/name` | Name of log |                                         |
| `max_epochs` | Max number of epochs to train the model | `**[long enough it doesn’t stop early]` |
| `check_val_every_n_epoch` | Frequency to evaluate the model | `1`                                     |
| `enable_checkpointing` | Enables periodically saving model to a file | `True`                                  |
| `default_root_dir` | Directory for model checkpoints |                                         |

### Datamodule
The field data is expected to receive a [generic datamodule](https://ibm.github.io/terratorch/stable/package/generic_datamodules/) or any other datamodule compatible with [Lightning Datamodules](https://lightning.ai/docs/pytorch/stable/data/datamodule.html), as those defined in TerraMind’s [collection of datamodules](https://ibm.github.io/terratorch/stable/package/datamodules/). The class\_path parameter will inform the rest of the required parameters.

| *Parameter* | *Description* | *Default*                                    |
| :---- | :---- |:---------------------------------------------|
| `Class_path` | Datamodule Class (for example, [here](https://ibm.github.io/terratorch/stable/package/generic_datamodules/#terratorch.datamodules.generic_pixel_wise_data_module.GenericNonGeoPixelwiseRegressionDataModule)) | `GenericNonGeoPixelwiseRegressionDataModule` |

The Terramind datamodule 
```
data:
  class_path: GenericNonGeoSegmentationDataModule
  init_args:
    batch_size: 8
    num_workers: 2
    dataset_bands:  # Dataset bands
      - BLUE
      - GREEN
      - RED
      - NIR_NARROW
      - SWIR_1
      - SWIR_2
    rgb_indices:
      - 2
      - 1
      - 0
    train_data_root: hls_burn_scars/data
    val_data_root: hls_burn_scars/data
    test_data_root: hls_burn_scars/data
    train_split: hls_burn_scars/splits/train.txt
    val_split: hls_burn_scars/splits/val.txt
    test_split: hls_burn_scars/splits/test.txt
    img_grep: "*_merged.tif"
    label_grep: "*.mask.tif"
    # Dataset stats
    means:
      -  0.033349706741586264
      -  0.05701185520536176
      -  0.05889748132001316
      -  0.2323245113436119
      -  0.1972854853760658
      -  0.11944914225186566
    stds:
      -  0.02269135568823774
      -  0.026807560223070237
      -  0.04004109844362779
      -  0.07791732423672691
      -  0.08708738838140137
      -  0.07241979477437814
    num_classes: 2
    train_transform:
      - class_path: albumentations.D4
      - class_path: ToTensorV2
    no_data_replace: 0
    no_label_replace: -1
```


### Model
The model field is the configuration for `task + model`.

| *Parameter* | *Description* | *Default*                               |
| :---- | :---- |:----------------------------------------|
| `class_path` | Regression task | [`terratorch.tasks.PixelwiseRegreesionTask`](https://ibm.github.io/terratorch/tasks/) |

. Since we are using the same TerraMind model for all our purposes, parameters here will not change. The following parameters are going to be the defaults. 
```
model:
  class_path: terratorch.tasks.SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: terramind_v1_base  # large version: terramind_v1_large
      backbone_pretrained: true
      backbone_modalities:
        - S2L2A
      backbone_bands: # Select subset of pre-trained bands
        S2L2A:
          - BLUE
          - GREEN
          - RED
          - NIR_NARROW
          - SWIR_1
          - SWIR_2

      necks:
        - name: SelectIndices
          indices: [2, 5, 8, 11]  # base version
#          indices: [5, 11, 17, 23]  # large version
        - name: ReshapeTokensToImage
          remove_cls_token: False
        - name: LearnedInterpolateToPyramidal

      decoder: UNetDecoder
      decoder_channels: [512, 256, 128, 64]

      head_dropout: 0.1
      num_classes: 2
    loss: dice
    ignore_index: -1
    freeze_backbone: false
    freeze_decoder: false
    class_names:
      - Others
      - Burned                       |
```

### Optimizer and Learning Rate Scheduler 
The [Optimizer](https://docs.pytorch.org/docs/stable/optim.html) parameter will implement desired optimization algorithm.
There are a variety of [algorithms](https://docs.pytorch.org/docs/stable/optim.html#algorithms) Pytorch offers, but our focus will be on 
the [AdamW algorithm](https://arxiv.org/pdf/1711.05101). 

| *Parameter*    | *Description*           | *Default*                                                                                          |
|:---------------|:------------------------|:---------------------------------------------------------------------------------------------------|
| `class_path`   | Optimizer Algorithm     | [`AdamW`](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW) |
| `lr`           | Learning Rate           | `1.e-4`                                                                                            |
| `weight_decay` | Weight Decay Coeffcient | `0.1`                                                                                              |

Where **weight decay** (or L2 regularization) is a regularization technique that helps prevent overfitting and 
**learning rate** is a hyperparameter that determines the step size of the optimization algorithm. 

The learning rate scheduler automatically adjusts the learning rate during training. By using the [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html), we are 
instructing the scheduler to [`reduce the learning rate only when improvement stagnates`](https://machinelearningmastery.com/a-gentle-introduction-to-learning-rate-schedulers/).

| *Parameter*  | *Description*                  | *Default*                                                                                                             |
|:-------------|:-------------------------------|:----------------------------------------------------------------------------------------------------------------------|
| `class_path` | Learning Rate Scheduler        | [`ReduceLROnPlateau`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) |
| `monitor`    | Value used to update each step | `val/loss`                                                                                                            |
| `factor`     | factor                         | `0.5`                                                                                                                 |
| `patience`   | patience                       | `5`                                                                                                                   |
