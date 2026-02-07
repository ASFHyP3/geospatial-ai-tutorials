# Shell command for installing cartopy (cannot be used as an import)
# !pip install cartopy

# Standard library
import os
from pathlib import Path
import glob
import zipfile
import logging
import warnings

# Scientific computing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Plotting & visualization
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import box

# Raster & geospatial
import rasterio
from rasterio.windows import Window
import geopandas as gpd

# Machine learning / deep learning
import torch
import lightning.pytorch as pl
import albumentations

# TorchGeo / TerraTorch
import terratorch

# Misc
import gdown
import earthaccess

import get_swath


# Suppress warnings and rasterio logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("rasterio").setLevel(logging.ERROR)


def bytescale(arr, cmin=0, cmax=1, low=0, high=255):
    # clip the data to be in the range of cmin to cmax
    arr = np.clip(arr, cmin, cmax)
    high = float(high)
    low = float(low)
    cmax = float(cmax)
    cmin = float(cmin)
    m = (high - low) / (cmax - cmin)  # slope
    b = high - (m * cmax)             # intercept
    arr = np.uint8((m * arr) + b)
    return arr


def calculate_raster_stats(file_list, min_valid=0):
    """Calculates running mean and std for a list of rasters."""
    total_sum = 0
    total_sq_sum = 0
    total_count = 0

    for file in file_list:
        print(file)
        with rasterio.open(file) as src:
            # Read first band as float32 to avoid overflow
            band = src.read(1).astype('float32')
            ok = np.where(band > min_valid)
            if len(ok[0]) > 0:
                total_sum += np.sum(band[ok])
                total_sq_sum += np.sum(band[ok]**2)
                total_count += len(ok[0])

    mean = total_sum / total_count
    std = np.sqrt((total_sq_sum / total_count) - (mean**2))
    return mean, std


# Helper function for plotting both modalities
def plot_sample(sample):
    data = sample['image'].cpu().numpy()
    mask = sample['mask'].cpu().numpy()

    # Scaling data.
    if data.mean() < 1:
        data = data * 10000
    data = (data.clip(0, 2000) / 2000) * 255
    rgb = data[[2, 1, 0]].astype(np.uint8).transpose(1,2,0)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb)
    ax[0].set_title('Image')
    ax[0].axis('off')
    ax[1].imshow(mask, vmin=-1, vmax=1, interpolation='nearest')
    ax[1].set_title('Mask')
    ax[1].axis('off')
    ax[2].imshow(rgb)
    ax[2].imshow(mask, vmin=-1, vmax=1, interpolation='nearest', alpha=0.5)
    ax[2].set_title('Mask on Image')
    ax[2].axis('off')
    fig.tight_layout()
    plt.show()


def load_event_database():
    # use 60-swath version
    hwds_google_drive_id = '1h_JIEcrrUF3OSTrmwAKNPa0eUEhPA2Xx'
    drive_url = f'https://drive.google.com/uc?id={hwds_google_drive_id}'

    shp_dir = Path('hwds/SHP')
    shp_dir.mkdir(parents=True, exist_ok=True)

    filename = 'hwds_v3_20250205_subset_60.zip'

    zip_path = shp_dir / filename

    if not zip_path.exists():
        gdown.download(drive_url, str(zip_path), quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=shp_dir)

    shp_path = shp_dir / 'hwds_v3_20250205_subset_60.shp'
    gdf = gpd.read_file(shp_path)

    return gdf


def add_buffered_events_to(gdf):
    # make some additional columns that represent buffers after projecting to UTM 15N
    gdf = gdf.to_crs(32615)
    buffered_event = gdf.buffer(3000)
    buffered_event_background = gdf.buffer(10000)
    gdf = gdf.to_crs(4326)

    gdf['buffered_event'] = buffered_event
    gdf['buffered_event_background'] = buffered_event_background
    gdf['buffered_event'] = gdf['buffered_event'].to_crs("EPSG:4326")
    gdf['buffered_event_background'] = gdf['buffered_event_background'].to_crs("EPSG:4326")

    return gdf


def main():
    gdf = load_event_database()

    gdf['swathDate'] = pd.to_datetime(gdf['swathDate'], format='%Y-%m-%d')
    gdf['ls5hlsDate'] = pd.to_datetime(gdf['ls5hlsDate'], format='%Y-%m-%d')

    gdf = add_buffered_events_to(gdf)

    # choose one to work with
    keepers = gdf['swathID'][0:10]
    keepers = [1442, 622, 1079, 628]
    gdf = gdf[gdf['swathID'].isin(keepers)]

    data_path = Path('hwds')

    earthaccess.login()

    for i, row in gdf.iterrows():
        fmasks_merged = get_swath.get_swath(row, hwds_path=data_path)
        return

    # some error in logic above, just brute force get Fmasks from merged
    fmasks_merged = glob.glob(os.path.join('hwds/MERGE', '*Fmask.tif'))
    print(fmasks_merged)

    # use fmasks as templates and do stuff
    # keep track of all eventual tiles
    chips_dict = {'Fmask': [], 'MASK': [], 'EVENT': [], 'BANDS': [], 'QC': []}
    p_chips = 'hwds/CHIPS'
    if not os.path.isdir(p_chips):
        print('making:', p_chips)
        os.makedirs(p_chips, exist_ok=True)

    for fmask_merged in fmasks_merged:
        print(fmask_merged)
        if 'L30' in fmask_merged:
            bands = {
                'B': 'B02', 'G': 'B03', 'R': 'B04', 'N': 'B05',
                'SW1': 'B06', 'SW2': 'B07', 'Fmask': 'Fmask',
                'EVENT': 'EVENT', 'QC': 'QC', 'MASK': 'MASK'
            }
        else:
            bands = {
                'B': 'B02', 'G': 'B03', 'R': 'B04', 'N': 'B08',
                'SW1': 'B11', 'SW2': 'B12', 'Fmask': 'Fmask',
                'EVENT': 'EVENT', 'QC': 'QC', 'MASK': 'MASK'
            }

        to_stack = ['B', 'G', 'R', 'N', 'SW1', 'SW2']
        hls_merged = {}

        for key in bands.keys():
            hls_merged[key] = fmask_merged.replace('Fmask', bands[key])
            print(hls_merged[key], os.path.isfile(hls_merged[key]))

        # get the bands we need to stack
        to_stack_ds = []
        for t in to_stack:
            f_band = hls_merged[t]
            ds = rasterio.open(f_band)
            to_stack_ds.append(ds)

        # create output file
        f_bands = fmask_merged.replace('Fmask', 'BANDS')
        # remove old while debugging
        if os.path.isfile(f_bands):
            os.remove(f_bands)

        if not os.path.isfile(f_bands):
            meta = to_stack_ds[0].meta.copy()
            meta['count'] = len(to_stack)
            meta['dtype'] = np.float32
            with rasterio.open(f_bands, 'w', **meta) as dst:
                for tt, band_ds in enumerate(to_stack_ds):
                    img = band_ds.read(1)
                    dst.write(img, tt + 1)
                    band_ds = None

        print(f_bands, os.path.isfile(f_bands))
        hls_merged['BANDS'] = f_bands

        # create chips
        chip_size = 512
        to_tile = {
            'QC': hls_merged['QC'],
            'Fmask': hls_merged['Fmask'],
            'MASK': hls_merged['MASK'],
            'EVENT': hls_merged['EVENT'],
            'BANDS': hls_merged['BANDS']
        }

        for tt in to_tile.keys():
            print('chips for:', to_tile[tt])
            with rasterio.open(to_tile[tt]) as src:
                meta = src.meta.copy()
                h_tiles = src.width // chip_size
                v_tiles = src.height // chip_size
                for ii in range(v_tiles + 1):
                    for jj in range(h_tiles + 1):
                        x_off = jj * chip_size
                        y_off = ii * chip_size
                        width = min(chip_size, src.width - x_off)
                        height = min(chip_size, src.height - y_off)
                        # skip bad ones
                        if width != chip_size or height != chip_size:
                            print('bad chip, skip:', width, height)
                            continue
                        # create window
                        window = Window(x_off, y_off, width, height)
                        meta.update({
                            'width': width,
                            'height': height,
                            'transform': src.window_transform(window)
                        })
                        # write the tile
                        ii_str = '%3.3d' % ii
                        jj_str = '%3.3d' % jj
                        tile_number = '.'.join([ii_str, jj_str])
                        f_chip = os.path.basename(to_tile[tt]).replace(
                            tt + '.tif', tile_number + '.' + tt + '.tif'
                        )
                        f_chip = os.path.join(p_chips, f_chip)
                        with rasterio.open(f_chip, 'w', **meta) as dst:
                            for bb in range(meta['count']):
                                data = src.read(bb + 1, window=window)
                                dst.write(data, bb + 1)
                        chips_dict[tt].append(f_chip)


    # for convenience (wasting disk space though) just copy keepers
    # to a new folder for terramind use (_TM)
    p_chips_keep = 'hwds/CHIPS_TM'
    if not os.path.isdir(p_chips_keep):
        print('making:', p_chips_keep)
        os.makedirs(p_chips_keep, exist_ok=True)

    # tidy old files
    files = glob.glob(os.path.join(p_chips_keep, '*.tif'))
    for f in files:
        os.remove(f)

    # show count of chips per key
    for key in chips_dict.keys():
        print(key, len(chips_dict[key]))

    # create DataFrame to track chips
    chips_df = pd.DataFrame.from_dict(chips_dict)
    chips_df['keep'] = False
    chips_df['base'] = ''
    chips_df['root'] = ''
    chips_df['pct_cf'] = 0.0
    chips_df['pct_ev'] = 0.0
    print(chips_df)

    # iterate over chips to assess quality
    for i, row in chips_df.iterrows():
        print(row)

        # read QC, EVENT, and MASK bands
        with rasterio.open(row['QC']) as ds:
            qc = ds.read(1)
        with rasterio.open(row['EVENT']) as ds:
            event = ds.read(1)
        with rasterio.open(row['MASK']) as ds:
            mask = ds.read(1)

        ny, nx = qc.shape
        n_px = 1.0 * ny * nx

        # cloud-free pixels (0 clear, 1 cloud, 255 nodata)
        n_cf = len(np.where(qc == 0)[0])
        pct_cf = 100. * (n_cf / n_px)

        # valid pixels (not nodata)
        nv = len(np.where(qc != 255)[0])
        pct_valid = 100. * (nv / n_px)

        # event pixels
        n_ev = len(np.where(event > 0)[0])

        # cloud-free AND event pixels
        n_cf_ev = len(np.where((qc == 0) & (event > 0))[0])
        pct_cf_ev = 100. * (n_cf_ev / n_ev) if n_ev > 0 else 0

        # pct of chip in event
        pct_ev = 100. * (n_ev / n_px) if n_ev > 0 else 0

        # file base and root names
        f_base = '.'.join(os.path.basename(row['MASK']).split('.')[:-2])
        f_root = '.'.join(os.path.basename(row['MASK']).split('.')[:-4])
        print(f_base)
        print(f_root)
        print(pct_cf, pct_ev)

        # update DataFrame
        chips_df.at[i, 'base'] = f_base
        chips_df.at[i, 'root'] = f_root
        chips_df.at[i, 'pct_cf'] = pct_cf
        chips_df.at[i, 'pct_ev'] = pct_ev

        # loosened criteria for Colab space
        if pct_cf > 95 and pct_ev > 1:
            chips_df.at[i, 'keep'] = True

            # copy MASK and BANDS to _TM
            for f_key in ['MASK', 'BANDS']:
                src_file = row[f_key]
                cmd = ['/usr/bin/cp', src_file, p_chips_keep]
                os.system(' '.join(cmd))

    # select only the keeper chips
    keep = chips_df['keep']
    good_chips_df = chips_df.loc[keep]
    print(good_chips_df)

    # split into train/test (and duplicate test as validation)
    train, test = train_test_split(
        good_chips_df['base'],
        test_size=0.3,
        random_state=42
    )
    valid = test.copy()

    # prepare splits folder
    p_splits = 'hwds/SPLITS'
    if not os.path.isdir(p_splits):
        print('making:', p_splits)
        os.makedirs(p_splits, exist_ok=True)

    # clean old split files
    files = glob.glob(os.path.join(p_splits, '*.txt'))
    for f in files:
        os.remove(f)

    # write out split files
    split_files = {
        'train.txt': train,
        'val.txt': valid,
        'test.txt': test
    }

    for fname, split_data in split_files.items():
        f_path = os.path.join(p_splits, fname)
        with open(f_path, 'w') as f:
            for item in split_data:
                f.write(f"{item}\n")

    crs_pc = ccrs.PlateCarree()

    for fmask_merged in fmasks_merged:
        # extract swath info
        swathID = os.path.basename(fmask_merged)[0:4]
        int_swathID = int(swathID)
        ok = gdf['swathID'] == int_swathID
        this_gdf = gdf.loc[ok]
        swath_geom = this_gdf['geometry'].values[0]

        # read RGB bands and scale
        ds = rasterio.open(fmask_merged.replace('Fmask', bands['R']))
        bounds = ds.bounds
        full_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        r = bytescale(np.sqrt(np.clip(ds.read(1) / 10000., 0, 2)), 0, 0.5)
        ds.close()

        ds = rasterio.open(fmask_merged.replace('Fmask', bands['G']))
        g = bytescale(np.sqrt(np.clip(ds.read(1) / 10000., 0, 2)), 0, 0.5)
        ds.close()

        ds = rasterio.open(fmask_merged.replace('Fmask', bands['B']))
        b = bytescale(np.sqrt(np.clip(ds.read(1) / 10000., 0, 2)), 0, 0.5)
        ds.close()

        rgb = np.dstack((r, g, b))

        # plot RGB background
        fig, ax = plt.subplots(
            1, 1,
            subplot_kw={'projection': crs_pc},
            figsize=(12, 12),
            layout='constrained'
        )
        ax.imshow(rgb, extent=full_extent, origin='upper', transform=crs_pc)
        ax.add_geometries([swath_geom], edgecolor='red', linewidth=2,
                          facecolor='none', crs=crs_pc)

        # plot chip extents
        base = '.'.join(os.path.basename(fmask_merged).split('.')[0:6])
        ok = chips_df['root'] == base
        these_chips = chips_df.loc[ok]
        print('len of these_chips:', len(these_chips))

        if len(these_chips) > 0:
            for _, chip in these_chips.iterrows():
                ds = rasterio.open(chip['MASK'])
                chip_bounds = ds.bounds
                chip_geom = box(chip_bounds.left, chip_bounds.bottom,
                                chip_bounds.right, chip_bounds.top)
                ds.close()

                if (chip['pct_cf'] > 95) & (chip['pct_ev'] > 1):
                    chip_color, chip_width, chip_z = 'green', 3, 2
                else:
                    chip_color, chip_width, chip_z = 'yellow', 1, 1

                ax.add_geometries(
                    [chip_geom],
                    edgecolor=chip_color,
                    linewidth=chip_width,
                    alpha=1,
                    zorder=chip_z,
                    facecolor='none',
                    crs=crs_pc
                )
        else:
            print('no chips')

        ax.set_extent(full_extent, crs=crs_pc)
        plt.show()

    # Initialize stats for each band
    band_stats = {}
    for t in to_stack:
        band_stats[t + '_total_sum'] = 0
        band_stats[t + '_total_sq_sum'] = 0
        band_stats[t + '_total_count'] = 0

    # Compute running sums across all good chips
    for f in good_chips_df['BANDS']:
        with rasterio.open(f) as ds:
            for tt, t in enumerate(to_stack):
                img = ds.read(tt + 1).astype('float32')
                ok = np.where(img > 0)
                if len(ok[0]) > 0:
                    band_stats[t + '_total_sum'] += np.sum(img[ok])
                    band_stats[t + '_total_sq_sum'] += np.sum(img[ok] ** 2)
                    band_stats[t + '_total_count'] += len(ok[0])

    # Compute mean and std for each band
    for t in to_stack:
        total_sum = band_stats[t + '_total_sum']
        total_sq_sum = band_stats[t + '_total_sq_sum']
        total_count = band_stats[t + '_total_count']
        band_stats[t + '_mean'] = total_sum / total_count
        band_stats[t + '_std'] = np.sqrt((total_sq_sum / total_count) - (band_stats[t + '_mean'] ** 2))

    # Print stats
    for key in band_stats.keys():
        print(key, band_stats[key])

    # Collect means and stds
    chip_means = [band_stats[t + '_mean'] for t in to_stack]
    chip_stds = [band_stats[t + '_std'] for t in to_stack]

    print("Chip means:", chip_means)
    print("Chip stds:", chip_stds)

    warnings.filterwarnings("ignore")
    datamodule = terratorch.datamodules.GenericNonGeoSegmentationDataModule(
        batch_size=8,
        num_workers=2,
        num_classes=2,
        rgb_indices=[2, 1, 0],

        # TODO: Define the data and label paths
        train_data_root='hwds/CHIPS_TM',
        train_label_data_root='hwds/CHIPS_TM',
        val_data_root='hwds/CHIPS_TM',
        val_label_data_root='hwds/CHIPS_TM',
        test_data_root='hwds/CHIPS_TM',
        test_label_data_root='hwds/CHIPS_TM',

        # TODO: Define the split files
        train_split='hwds/SPLITS/train.txt',
        val_split='hwds/SPLITS/val.txt',
        test_split='hwds/SPLITS/test.txt',

        # TODO: Define suffixs
        img_grep='*.BANDS.tif',
        label_grep='*.MASK.tif',

        # TODO: Update the standardization values. They need to be the same length as the data.
        #  You can define a constant_scale that applies a multiplicator in case the data does not align with the the standardization values.
        # constant_scale=None,
        means=chip_means,
        stds=chip_stds,
        # fyi TerraMind pretraining values (assuming data in range 0-10000)
        # S2L2A means: [1390.458, 1503.317, 1718.197, 1853.910, 2199.100, 2779.975, 2987.011, 3083.234, 3132.220, 3162.988, 2424.884, 1857.648]
        # S2L2A stds: [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926, 1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311]

        # albumentations supports shared transformations and can handle multimodal inputs.
        train_transform=[
            albumentations.D4(), # Random flips and rotation
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
    plot_sample(val_dataset[0])
    plot_sample(val_dataset[1])
    plot_sample(val_dataset[2])

    # checking datasets testing split size
    datamodule.setup("test")
    test_dataset = datamodule.test_dataset
    print('test_dataset:',len(test_dataset))

    datamodule.setup("train")
    train_dataset = datamodule.train_dataset
    print('train_dataset:',len(train_dataset))

    warnings.filterwarnings("ignore")

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    pl.seed_everything(0)

    # By default, TerraTorch saves the model with the best validation loss. You can overwrite this by defining a custom ModelCheckpoint, e.g., saving the model with the highest validation mIoU.
    # TODO Optionally adjust the checkpoint
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='output/terramind_hwds_base',
        mode="max",
        monitor="val/mIoU", # Variable to monitor
        filename="best-mIoU",
        save_weights_only=True,
    )

    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator="auto",
        strategy="auto",
        devices=1, # Deactivate multi-gpu because it often fails in notebooks
        precision="16-mixed",  # Speed up training with half precision, delete for full precision training.
        num_nodes=1,
        logger=True,  # Uses TensorBoard by default
        max_epochs=10, # For demos
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
        # TODO Define output dir
        default_root_dir='output/terramind_hwds_base',
    )

    # Segmentation mask that build the model and handles training and validation steps.
    model = terratorch.tasks.SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",  # Combines a backbone with necks, the decoder, and a head
        model_args={
            # TerraMind backbone
            "backbone": "terramind_v1_base",
            "backbone_pretrained": True,
            # TODO Select the modality. Check the docs for details https://terrastackai.github.io/terratorch/stable/guide/terramind/#subset-of-input-bands
            "backbone_modalities": ["S2L2A"],
            # TODO define the input bands. This is only needed because you need to select a subset of the pre-training bands for Burn Scars.
            #  Check the names in the "List of pre-trained bands" in the docs.
            "backbone_bands": {"S2L2A": ["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]},

            # Necks
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": [2, 5, 8, 11] # indices for terramind_v1_base
                    # "indices": [5, 11, 17, 23] # indices for terramind_v1_large
                },
                {"name": "ReshapeTokensToImage",
                 "remove_cls_token": False},  # TerraMind is trained without CLS token, which neads to be specified.
                {"name": "LearnedInterpolateToPyramidal"}  # Some decoders like UNet or UperNet expect hierarchical features. Therefore, we need to learn a upsampling for the intermediate embedding layers when using a ViT like TerraMind.
            ],

            # Decoder
            "decoder": "UNetDecoder",
            #"decoder_channels": [256, 128, 64, 32],
            "decoder_channels" : [512,256,128,64],

            # Head
            "head_dropout": 0.1,
            "num_classes": 2,
        },

        loss="dice",  # We recommend dice for binary tasks and ce for tasks with multiple classes.
        optimizer="AdamW",
        lr=2e-5,  # The optimal learning rate varies between datasets, we recommend testing different once between 1e-5 and 1e-4. You can perform hyperparameter optimization using terratorch-iterate.
        ignore_index=-1,
        freeze_backbone=True, # Only used to speed up fine-tuning in this demo, we highly recommend fine-tuning the backbone for the best performance.
        freeze_decoder=False,  # Should be false in most cases as the decoder is randomly initialized.
        plot_on_val=True,  # Plot predictions during validation steps
        class_names=["Others", "Damage"]  # optionally define class names
    )

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

    test_loader = datamodule.test_dataloader()
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


if __name__ == '__main__':
    main()
