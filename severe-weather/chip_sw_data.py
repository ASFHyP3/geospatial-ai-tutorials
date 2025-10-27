import argparse
import datetime

from satchip import chip_label, chip_data
from pathlib import Path


def get_date_from_name(name: str) -> str:
    date_str = name.split('_')[-1].split('.')[0]
    return datetime.datetime.fromisoformat(date_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a raster file')
    parser.add_argument('raster_path', type=Path, help='Path to a label raster')
    args = parser.parse_args()

    raster_path = args.raster_path
    date = get_date_from_name(raster_path.name)

    chip_dir = Path.cwd() / 'chips'

    print(f'label raster: {raster_path}')
    print(f'date: {date}')
    print(f'chip dir: {chip_dir}')

    chip_label.chip_labels(raster_path, date, chip_dir)

    label_path = chip_dir / 'LABEL'
    label_paths = list(label_path.glob('*.zarr.zip'))
    image_dir = Path.cwd() / 'images'
    platform = 'S2L2A'
    # platform = 'S1RTC'

    date_start = date - datetime.timedelta(days=28)
    date_end = date + datetime.timedelta(days=7)
    args = (
        label_paths, platform, date_start, date_end, 'ALL', 100, chip_dir, image_dir
    )

    chip_data.create_chips(*args)
