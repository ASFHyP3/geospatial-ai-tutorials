import os
import rasterio
from rasterio.windows import Window


def chip_data(hls_merged):
    p_chips = "hwds/CHIPS"
    if not os.path.isdir(p_chips):
        print("making:", p_chips)
        os.makedirs(p_chips, exist_ok=True)

    # use fmasks as templates and do stuff
    # keep track of all eventual tiles
    chips_dict = {"Fmask": [], "MASK": [], "EVENT": [], "BANDS": [], "QC": []}

    # create chips
    chip_size = 512
    to_tile = {
        "QC": hls_merged["QC"],
        "Fmask": hls_merged["Fmask"],
        "MASK": hls_merged["MASK"],
        "EVENT": hls_merged["EVENT"],
        "BANDS": hls_merged["BANDS"],
    }

    for tt in to_tile.keys():
        print("chips for:", to_tile[tt])
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
                        print("bad chip, skip:", width, height)
                        continue

                    # create window
                    window = Window(x_off, y_off, width, height)
                    meta.update(
                        {
                            "width": width,
                            "height": height,
                            "transform": src.window_transform(window),
                        }
                    )

                    # write the tile
                    ii_str = "%3.3d" % ii
                    jj_str = "%3.3d" % jj
                    tile_number = ".".join([ii_str, jj_str])
                    f_chip = os.path.basename(to_tile[tt]).replace(
                        tt + ".tif", tile_number + "." + tt + ".tif"
                    )
                    f_chip = os.path.join(p_chips, f_chip)
                    with rasterio.open(f_chip, "w", **meta) as dst:
                        for bb in range(meta["count"]):
                            data = src.read(bb + 1, window=window)
                            dst.write(data, bb + 1)
                    chips_dict[tt].append(f_chip)

    return chips_dict
