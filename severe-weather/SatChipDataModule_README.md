# Working with SatChip Datasets
SatChip datasets are pre-chipped [TorchGeo](https://github.com/torchgeo/torchgeo?tab=readme-ov-file) datasets specialized for satellite imagery.
In this tutorial, we will discuss the organization of SatChip data, the SatChip Dataset and DataModule, and run a tutorial example with pre-staged data.

### Dataset organization
The data are in `zarr` format and prepared using [SatChip](https://github.com/forrestfwilliams/satchip), a Python package for preparing satellite images for TorchGeo. Currently supported datasets include: 
* S2L2A: Sentinel-2 L2A data sourced from the [Sentinel-2 AWS Open Data Archive](https://registry.opendata.aws/sentinel-2/)
* HLS: Harmonized Landsat Sentinel-2 data sourced from [LP DAAC's Data Archive](https://www.earthdata.nasa.gov/data/projects/hls)
* S1RTC: Sentinel-1 Radiometric Terrain Corrected (RTC) data created using [ASF's HyP3 on-demand platform](https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/)

SatChip prepares data labels and satellite images into 264x264 image arrays that follow the TerraMind extension of the MajorTom specification. These files are saved them as `zarr.zip` files and can be loaded in Python with `satchip.utils.load_chip()`. [Here](generating_chip_data.md) is a tutorial on generating your own chip data. 

Below is an example of metadata associated with the Sentinel-1 RTC chip `swathID_638_swathDate_2020-06-04`: 
```
<xarray.Dataset> Size: 587MB
Dimensions:   (sample: 263, platform: 1, time: 4, band: 2, y: 264, x: 264)
Coordinates:
  * band      (band) <U2 16B 'VH' 'VV'
  * platform  (platform) object 8B 'S1RTC'
  * sample    (sample) object 2kB '494U_807L_0_0' ... '507U_805L_1_3'
  * time      (time) datetime64[ns] 32B 2020-06-06T00:46:37 ... 2020-06-13T00...
  * x         (x) int64 2kB 0 1 2 3 4 5 6 7 ... 256 257 258 259 260 261 262 263
  * y         (y) int64 2kB 0 1 2 3 4 5 6 7 ... 256 257 258 259 260 261 262 263
Data variables:
    data      (sample, platform, time, band, y, x) float32 587MB ...
Attributes:
    date_created:     2020-06-04T00:00:00
    satchip_version:  0.3.0
    bounds:           [-103.24111, 44.39625, -101.30495, 45.58643]
```
Pre-staged data are available in Google Drive to run this [tutorial](#example-tutorial). These data includes a Labeled hail damage, S1RTC, and S2L2A data from June 4, 2020.  
You can download them in the terminal command line with the following commands.
```bash
mkdir data
cd data
curl -L "https://drive.usercontent.google.com/download?id={1XgwNgIdAJRvr4sk7K9J_vkxQu2DukFrn}&confirm=xxx" -o swathID_638_swathDate_2020-06-04_S1RTC.zarr.zip
curl -L "https://drive.usercontent.google.com/download?id={1-Gjgn5LVo6pLMViMZZ-gAvk2vXxVpJh2}&confirm=xxx" -o swathID_638_swathDate_2020-06-04_S2L2A.zarr.zip
curl -L "https://drive.usercontent.google.com/download?id={1AtoUCf6Ge5qvLWvTuuhfNfnzXo__xzL7}&confirm=xxx" -o swathID_638_swathDate_2020-06-04.zarr.zip
cd ..
```

### Data loader
TorchGeo uses [dataset classes](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#geospatial-datasets) to handle geospatial and non-geospatial data. A `DataSet`stores the data and their corresponding labels, while the `DataModule` wraps around the `DataSet` to access and transform and the `DataLoader` to batch and shuffle the data. The `DataModule` is then utilized by the Pytorch `Trainer` object to train the model. 

The `SatChip` dataset class is specifically designed for working with prepared SatChip data. It inherits TerraTorch's [`NonGeoDataset`](https://torchgeo.readthedocs.io/en/stable/api/datasets.html#non-geospatial-datasets) base class. Both the SatChip Dataset and DataModule code is in a [`loader.py`](loader.py).

Using the `SatChipDataModule` function, you can initialize the dataset by providing Path object paths to the labeled, S2L2A and S1RTC data.
```python
import loader

datamodule = loader.SatChipDataModule(batch_size=2, label_path=label_path, s2_path=s2_path, rtc_path=rtc_path)
```
This `DataModule` object can then be passed to the Pytorch `Trainer` object for training. 
We recommend using the [`TerraMind`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base) when working with `SatChip` Data. You can check out the suggested model parameters in [`train.py`](train.py).

### Setup
If working on NAS for the first time, check out [this documentation](NAS/initial_setup.md) outlining setting up your NAS environment for the first time. On NAS, the pre-created `terramind` conda environment can be activated with
```
conda activate terramind
```
You can verify your environment is correctly setup with `torchgeo --help`.


### Example Tutorial
We will run a trial training using the DataModule with the data downloaded in the [Dataset Organization](#dataset-organization) section. In NAS, we run jobs using the [Portable Batch System (PBS)](https://www.nas.nasa.gov/hecc/support/kb/portable-batch-system-(pbs)-overview_126.html). We submit jobs using the `qsub` command and monitor jobs using the `qstat` command.

You will need to modify the submission script (`severe_weather_nas_submission_script.sh`) before running a job. Take a look at the submission script by calling `vi severe_weather_nas_submission_script.sh`. Lines starting with `# PBS` are configurations for PBS. Lines starting with `##` are comments that describe what each set of lines is doing.  Read through these comments. Next, move to line 36 and change the email to your email. This will allow the NAS system to notify you of your job's status. 

You can submit the trail training with the QSub command and submit to the PBS service
```bash
qsub -q gpu_devel severe_weather_nas_submission_script.sh
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
