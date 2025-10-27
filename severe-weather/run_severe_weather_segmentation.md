# Working with SatChip Datasets
SatChip datasets are pre-chipped [TorchGeo](https://github.com/torchgeo/torchgeo?tab=readme-ov-file) datasets specialized for satellite imagery.
In this tutorial, we will discuss the organization of SatChip data, the SatChip Dataset and DataModule, and run a tutorial example with pre-staged data.

### Dataset organization
The data are in `zarr` format and prepared using [SatChip](https://github.com/forrestfwilliams/satchip), a Python package for preparing satellite images for TorchGeo. Currently supported datasets include: 
* S2L2A: Sentinel-2 L2A data sourced from the [Sentinel-2 AWS Open Data Archive](https://registry.opendata.aws/sentinel-2/)
* HLS: Harmonized Landsat Sentinel-2 data sourced from [LP DAAC's Data Archive](https://www.earthdata.nasa.gov/data/projects/hls)
* S1RTC: Sentinel-1 Radiometric Terrain Corrected (RTC) data created using [ASF's HyP3 on-demand platform](https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/)

SatChip prepares data labels and satellite images into 264x264 image arrays that follow the TerraMind extension of the MajorTom specification and are saved as `zarr.zip` files.  [Here](generating_chip_data.md) is a tutorial on generating your own chip data. 

Below is an example of metadata associated with the Sentinel-1 RTC chip `swathID_1260_swathDate_2019-06-17_464U_867L_2_0_S1RTC.zarr.zip`: 
```
<xarray.Dataset> Size: 562kB
Dimensions:      (time: 1, band: 2, y: 264, x: 264)
Coordinates:
  * band         (band) <U2 16B 'VV' 'VH'
    sample       <U13 52B ...
    spatial_ref  int64 8B ...
  * time         (time) datetime64[ns] 8B 2019-06-17T00:54:07
  * x            (x) float64 2kB 5.679e+05 5.679e+05 ... 5.705e+05 5.705e+05
  * y            (y) float64 2kB 4.625e+06 4.625e+06 ... 4.622e+06 4.622e+06
Data variables:
    bands        (time, band, y, x) float32 558kB ...
    center_lat   float64 8B ...
    center_lon   float64 8B ...
    crs          int64 8B ...
Attributes:
    date_created:     2025-10-20T11:50:48.795768
    satchip_version:  0.2.0
    bounds:           [-104.18344, 41.74808, -104.15138, 41.77162]
```
Pre-staged data are available in Google Drive to run this [tutorial](#example-tutorial). These data include Labeled hail damage, S1RTC, and S2L2A data from June 17, 2019 and July 6, 2020. For the purposes of this tutorial, we will use the data from June 2019 to train the model and the data from July 2020 to validate the model. 

You can download a zip file of the tutorial data with the following terminal commands.
```bash
curl -L "https://drive.usercontent.google.com/download?id={1vW-GavIOd3qc49cGkgRBUXRuddtNWUgC}&confirm=xxx" -o chips.zip
unzip chips.zip
```
Once unzipped, you will see the `chips` folder is structured like the following:
```
chips/
├── train/
│   ├── LABEL
│   │   └── swathID_1260_swathDate_2019-06-17*.zarr.zip
│   ├── S1RTC
│   │   └── swathID_1260_swathDate_2019-06-17*S1RTC.zarr.zip
│   └── S2L2A
│       └── swathID_1260_swathDate_2019-06-17*S2L2A.zarr.zip
├── val/
│   ├── LABEL
│   │   └── swathID_649_swathDate_2020-07-06*.zarr.zip
│   ├── S1RTC
│   │   └── swathID_649_swathDate_2020-07-06*S1RTC.zarr.zip
│   └── S2L2A
│       └── swathID_649_swathDate_2020-07-06*S2L2A.zarr.zip
├── create_chips.py
├── swathID_649_swathDate_2020-07-06.tif
└── swathID_1260_swathDate_2020-06-17.tif
```

Note that within the `train` and `val` directories, each data type (`LABEL`, `S1RTC`, and `S2L2A`) has its own subdirectory. This folder also includes  `create_chips.py`, the script that created chips using the attached GeoTIFs. 

When working with a full dataset, you will also include a `test` folder with data to test the model. It is a good rule of thumb to split up your data so that ~80% of the data is used for training, ~10% of the dataset is used for validation, and ~10% of the data is used for testing.  

You can load the data in Python using `satchip.util.load_chip`, as demonstrated with the following code snippet.
```python
from satchip import utils
utils.load_chip('chips/train/S1RTC/swathID_1260_swathDate_2019-06-17_464U_867L_2_0_S1RTC.zarr.zip')
```

### Data loader
TorchGeo uses [dataset classes](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#geospatial-datasets) to handle geospatial and non-geospatial data. A `DataSet`stores the data and their corresponding labels, while the `DataModule` wraps around the `DataSet` to access and transform and the `DataLoader` to batch and shuffle the data. The `DataModule` is then utilized by the Pytorch `Trainer` object to train the model. 

The `SatChip` dataset class is specifically designed for working with prepared SatChip data. It inherits TerraTorch's [`NonGeoDataset`](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#non-geospatial-datasets) base class. Both the SatChip Dataset and DataModule code is in a [`loader.py`](loader.py).

Using the `SatChipDataModule` function, you can initialize the dataset by providing Path object path to the chipped data directory. Note that this data should be organized with the same structure outlined in the [dataset organization secton](#dataset-organization).
```python
import loader

datamodule = loader.SatChipDataModule(chip_path=chip_path, batch_size=2)
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
We will run a trial training using the DataModule with the data downloaded in the [Dataset Organization](#dataset-organization) section. If you have not already, run the following command to download the data. 
```bash
curl -L "https://drive.usercontent.google.com/download?id={1vW-GavIOd3qc49cGkgRBUXRuddtNWUgC}&confirm=xxx" -o chips.zip
unzip chips.zip
```

In NAS, we run jobs using the [Portable Batch System (PBS)](https://www.nas.nasa.gov/hecc/support/kb/portable-batch-system-(pbs)-overview_126.html). We submit jobs using the `qsub` command and monitor jobs using the `qstat` command.

You will need to modify the submission script (`run_severe_weather_segmentation.sh`) before running a job. Take a look at the submission script by calling `vi run_severe_weather_segmentation.sh`. Lines starting with `# PBS` are configurations for PBS. Lines starting with `##` are comments that describe what each set of lines is doing.  Read through these comments. Next, move to line 36 and change the email to your email. This will allow the NAS system to notify you of your job's status. 

You can submit the trail training with the QSub command and submit to the PBS service
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
