# Working with Terramind and Terratorch
This document will walk you through the steps to work with TerraMind in Terratorch. 

### Loading in Satchips
When downloading data, make sure to download in the following structure: 

    terrmind    
    ├──data
    │   ├── S2L1C
    │   │   └── *_S2L1C.zarr.zip files
    │   ├── S1RTC
    │   │   └── *_S1RTC.zarr.zip files
    │   └── Label
    │       └── *_Label.zarr.zip files
    └──splits
        ├── test_data.txt 
        ├── train_data.txt
        └── valid_data.txt

The `splits` files should include the base names of the files you wish to include in each dataset (e.g. `swathID_XXX_swathDate_YYYY-MM-DD`). Note that ~80% of data should be used for training, ~10% of data should be used for testing, and ~10% of data should be used for validation. 

