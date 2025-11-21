Download the shapfile polygon with this link

```
curl TODO/hls_hwds_polygon
```

The first step is to generate the label rasters based on the example shapes using `generate_severe_weather_rastters.py`

```bash
    generate_severe_weather_rasters.py 25_hwds_shapefile/hls_hwds_polygons.shp --output-dir label-rasters
```

Then use satchip to turn the label rasters into chips. Below is an example command that chips one of the rasters

```bash
    chiplabel label-rastser/swathID_1046_swathDate_2017-08-02.tif 2017-08-02 --chipdir 25_hwds
```

After the label chips are created it's time to start making the data chips. Well use `satchip chipdata` to make the data chips. An example command would be:

```
    chipdata 25_hwds/LABEL/swathID_1046_swathDate_2017-08-02*.zarr.zip S1RTC --dates 7/27/2017 8/8/2017 --chipdir 25_hwds/S1RTC --imagedir images
```

This creates S1RTC data chips for each label `zarr.zip`
