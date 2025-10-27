# Working with SatChip Datasets
SatChip datasets are pre-chipped [TorchGeo](https://github.com/torchgeo/torchgeo?tab=readme-ov-file) datasets specialized for satellite imagery.
In this tutorial, we will discuss the organization of SatChip data, the SatChip Dataset and DataModule, and run a tutorial example with pre-staged data.

### Dataset organization
The data chips are in `zarr` format and prepared using [SatChip](https://github.com/ASFHyP3/satchip), a Python package for preparing satellite images for TorchGeo. Currently supported datasets include:

- `S2L2A`: Sentinel-2 L2A data sourced from the [Sentinel-2 AWS Open Data Archive](https://registry.opendata.aws/sentinel-2/)
- `HLS`: Harmonized Landsat Sentinel-2 data sourced from [LP DAAC's Data Archive](https://www.earthdata.nasa.gov/data/projects/hls)
- `S1RTC`: OPERA Sentinel-1 Radiometric Terrain Corrected (RTC) data from [ASF DAAC's Data Archive](https://www.jpl.nasa.gov/go/opera/products/rtc-product/)
- `HYP3S1RTC`: Sentinel-1 Radiometric Terrain Corrected (RTC) data created using [ASF's HyP3 on-demand platform](https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/)

SatChip prepares data labels and satellite images into 264x264 sample files that follow the TerraMind extension of the MajorTom specification and are saved as `zarr.zip` files. [Here](./generate-chips/generating_chip_data.md) is a tutorial on generating your own chip data.

A pre-staged dataset is available in Google Drive to run this [tutorial](#example-tutorial). These data include a Labeled hail damage, S1RTC, and S2L2A data. The data is also seperated into train and val folders each with their own sets of Label, S1 and S2 data chips.

You can download them in the terminal command line with the following commands.
```bash
mkdir data
cd data
curl -L "https://drive.usercontent.google.com/download?id={12eJsKXlPa4kF1Gt5p887l1J_yX-Nr2nO}&confirm=xxx" -o chips_timeseries.zip
unzip chips_timeseries.zip
cd ..
```

You can load the data in Python using `satchip.util.load_chip`, as demonstrated with the following code snippet.
```python
from satchip import utils
chip = utils.load_chip('data/chips_timeseries/train/S1RTC/swathID_1260_swathDate_2019-06-17_464U_867L_2_0_S1RTC.zarr.zip')
```

The loaded chips will be the following xarray dataset:
```
<xarray.Dataset> Size: 2MB
Dimensions:      (time: 4, band: 2, y: 264, x: 264)
Coordinates:
  * time         (time) datetime64[ns] 32B 2019-06-05T00:54:07 ... 2019-06-24...
  * band         (band) <U2 16B 'VV' 'VH'
  * y            (y) float64 2kB 4.625e+06 4.625e+06 ... 4.622e+06 4.622e+06
  * x            (x) float64 2kB 5.679e+05 5.679e+05 ... 5.705e+05 5.705e+05
    sample       <U13 52B ...
    spatial_ref  int64 8B ...
Data variables:
    bands        (time, band, y, x) float32 2MB ...
    center_lat   float64 8B ...
    center_lon   float64 8B ...
    crs          int64 8B ...
Attributes:
    date_created:     2025-10-21T16:48:58.252747
    satchip_version:  0.2.0
    bounds:           [-104.18344, 41.74808, -104.15138, 41.77162]
```

### Data loader
TorchGeo uses [dataset classes](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#geospatial-datasets) to handle geospatial and non-geospatial data. A `DataSet` stores the data and their corresponding labels, while the `DataModule` wraps around the `DataSet` to access and transform and the `DataLoader` to batch and shuffle the data. The `DataModule` also is reponsible for splitting the data into training, validation and testing splits. The `DataModule` is then utilized by the Pytorch `Trainer` object to train the model.

The `SatChip` dataset class is specifically designed for working with prepared SatChip data. It inherits TerraTorch's [`NonGeoDataset`](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#non-geospatial-datasets) base class. Both the SatChip Dataset and DataModule code is in a [`loaders.py`](training/loaders.py).

There are a number of different dataset classes for loading the data in `chips_timeseries` in different ways. `SatChipDataset` will load is used to load only one timestep. `SatChipTemporalDataset` is used to load datasets with multiple timesteps. There are also `SatChipDataModule ` and `SatChipTemporalDataModule` for loading and training diffrent datasets. For the `chips_timeseries` data we will use `SatChipTemporalDataModule`

Using the `SatChipTemporalDataModule` function, you can initialize the dataset by providing Path object paths to the labeled, S2L2A and S1RTC data.
```python
import loader

datamodule = loader.SatChipTemporalDataModule(batch_size=2, chip_path='data/chips_timeseries')
```
This `DataModule` object can then be passed to the Pytorch `Trainer` object for training.
We recommend using the [`TerraMind`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base) when working with `SatChip` Data. You can check out the suggested model parameters in [`train.py`](train.py).

### Setup
If working on NAS for the first time, check out [this documentation](NAS/initial_setup.md) outlining setting up your NAS environment for the first time. On NAS, the pre-created `terramind` conda environment can be activated with
```
source activate terramind
```
You can verify your environment is correctly setup with `torchgeo --help`.


### Example Tutorial
We will run a trial training using the DataModule with the data downloaded in the [Dataset Organization](#dataset-organization) section. If you have not already, run the commands from that section.

In NAS, we run jobs using the [Portable Batch System (PBS)](https://www.nas.nasa.gov/hecc/support/kb/portable-batch-system-(pbs)-overview_126.html). We submit jobs using the `qsub` command and monitor jobs using the `qstat` command.

You will need to modify the submission script (`training/run_severe_weather_segmentation.sh`) before running a job. Take a look at the submission script by calling `vi run_severe_weather_segmentation.sh`. Lines starting with `# PBS` are configurations for PBS. Lines starting with `##` are comments that describe what each set of lines is doing.  Read through these comments. Next, move to line 36 and change the email to your email. This will allow the NAS system to notify you of your job's status.

While in the training directory you can submit the trail training with the QSub command and submit to the PBS service
```bash
qsub -q gpu_devel run_severe_weather_segmentation.sh
```
This command will print your job ID to the screen. Save the first set of digits somewhere safe - you will need them to query the status of your job.

Note that the `gpu_devel` is a special queue designed for quick prototyping. Each user is only allowed to have one job in this queue at a time, and there are fewer resources available per job. **DO NOT SUBMIT PRODUCTION JOBS TO THIS QUEUE**. Use the `gpu` queue instead.

To monitor that status of your job, you can use `qstat` and you job ID (`qstat XXXXXX`). This will produce an output like:
```
                                                       Req'd    Elap
JobID         User     Queue    Jobname        TSK Nds wallt S wallt Eff
------------- -------- -------- -------------- --- --- ----- - ----- ---
XXXXXX.pbspl4 ffwillia gpu_deve sw-run1        16   1 01:00 R 00:00  0%
```

The most important field is the status field. The values are as follows:

| Code | State        | Meaning                                                                 |
|------|--------------|-------------------------------------------------------------------------|
| Q    | Queued       | Job is in the queue, waiting to be scheduled.                           |
| R    | Running      | Job is currently executing.                                             |
| H    | Held         | Job is held (by user or system) and will not run until released.        |
| W    | Waiting      | Job is waiting for execution time (e.g., scheduled start).              |
| T    | Transiting   | Job is being moved to/from another server.                              |
| S    | Suspended    | Job has been suspended.                                                 |
| E    | Exiting      | Job is finishing; execution is done but cleanup is in progress.         |
| F    | Finished     | Job has completed execution and left the queue (success or failure).    |

Note that `F` means finished - not failed!

If you call `qstat XXXXXX` after your job is finished, you will receive the following message:
```
qstat: XXXXXX.pbspl4.nas.nasa.gov Job has finished, use -x or -H to obtain historical job information
```
As this message states, use the `qstat -fx XXXXXX` instead to query the final results of your job. If this command reports `0` for the `Exit_status` field, congrats - your job completed successfully!
