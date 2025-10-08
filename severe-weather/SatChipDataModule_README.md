# Working with SatChip Datasets
SatChip datasets are non-geospatial TorchGeo datasets specialized for satellite imagery. 

### Dataset organization
The data are in `zarr` format and prepared using [SatChip](https://github.com/forrestfwilliams/satchip), a Python package for preparing satellite images for TorchGeo. Currently supported datasets include: 
* S2L2A: Sentinel-2 L2A data sourced from the [Sentinel-2 AWS Open Data Archive](https://registry.opendata.aws/sentinel-2/)
* HLS: Harmonized Landsat Sentinel-2 data sourced from [LP DAAC's Data Archive](https://www.earthdata.nasa.gov/data/projects/hls)
* S1RTC: Sentinel-1 Radiometric Terrain Corrected (RTC) data created using [ASF's HyP3 on-demand platform](https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/)

SatChip prepares data labels and satellite images into 264x264 image arrays that follow the TerraMind extension of the MajorTom specification. These files are saved them as `zarr.zip` files and can be loaded in Python with `satchip.utils.load_chip()`. 

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

### Setup
After downloading this repository, create a new conda environment with the latest version of TerraTorch.
```
conda env create -f severe-weather/environment.yml
conda activate severe-weather
python pip install torchgeo
```
You can verify this setup with `torchgeo --help`. 


### Data loader
Using the `SatChipDataModule` function, you can initialize the dataset by providing Path object paths to the labeled, S2L2A and S1RTC data.
```
SatChipDataModule(batch_size=2, label_path=label_path, s2_path=s2_path, rtc_path=rtc_path)
```

For a tutorial of using this data module with TerraTorch, check out this [notebook](severe-weather-train-example.ipynb). 

