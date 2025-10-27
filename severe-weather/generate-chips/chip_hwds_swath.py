import argparse
import datetime
import pathlib

from satchip import chip_data, chip_label


DATA_DIR = pathlib.Path.cwd() / 'data'
OUT_DIR = DATA_DIR / 'zarrs'
SCRATCH_DIR = DATA_DIR / 'scratch'


def get_date_from_name(name: str) -> str:
    date_str = name.split('_')[-1].split('.')[0]
    return datetime.datetime.fromisoformat(date_str)


def make_labels(raster_name: str) -> None:
    swath_raster_path = DATA_DIR / 'hwds-rasters' / raster_name

    if swath_raster_path.exists():
        return

    date = get_date_from_name(raster_name)

    chip_label.chip_labels(swath_raster_path, date, OUT_DIR)


def make_chip_data(raster_name: str, platform: str) -> None:
    zar_name = raster_name.split('.')[0]
    label_zarr_path = OUT_DIR / f'{zar_name}.zarr.zip'

    date_start = get_date_from_name(raster_name) - datetime.timedelta(days=28)
    date_end = date_start + datetime.timedelta(days=7)

    chip_data.chip_data(
        label_zarr_path,
        platform,
        date_start,
        date_end,
        strategy='BEST',
        max_cloud_pct=100,
        output_dir=OUT_DIR,
        scratch_dir=SCRATCH_DIR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Process a raster file')
    parser.add_argument('raster_name', type=str, help='Name of raster file')
    args = parser.parse_args()

    raster_name = args.raster_name

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    platform = 'S1RTC'

    make_labels(raster_name)
    make_chip_data(raster_name, platform)

    platform = 'S2L2A'

    make_chip_data(raster_name, platform)


if __name__ == '__main__':
    main()
