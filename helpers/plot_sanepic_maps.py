### To plot all the maps that comes from running sanepic.
# to be used as: `python plot_sanepic_maps.py <base_dir> <nside>`

import numpy as np
import healpy as hp
import argparse
import matplotlib.pyplot as plt

# Construct the argument parser
ap = argparse.ArgumentParser(
    prog="ProgramName",
    description="What the program does",
    epilog="Text at the bottom of help",
)

# Add the arguments to the parser
ap.add_argument("base_dir", help="Path of the base directory")
ap.add_argument("nside", help="nside of the map, just to parse the filename")

# parsing the arguments
args = ap.parse_args()

map_list1 = ["HITS", "POL_HITS", "Q_NVAR", "U_NVAR", "NVAR"]
map_list2 = ["COS2P", "SIN2P"]
map_list3 = ["I", "Q", "U"]


def plot_maps(file_list, attr_list, nrows, ncols, outfname, unit=""):
    attr_idx = 0
    sub = nrows * 100 + ncols * 10 + 1
    plt.figure(figsize=(5 * ncols, 4 * nrows))
    for file in file_list:
        f = open(file, "r")
        healpy_map = np.fromfile(f, dtype=np.float64, count=-1)
        f.close()
        hp.mollview(
            healpy_map,
            norm="hist",
            title=f"Sanepic {attr_list[attr_idx]} map",
            unit=unit,
            sub=sub,
        )
        sub += 1
        attr_idx += 1

    plt.savefig(args.base_dir + "/out_dir/" + outfname + ".png")


# Plotting hit and vars maps
file_list1 = [
    args.base_dir
    + "/out_dir/"
    + "map__"
    + item
    + "output_N"
    + str(args.nside)
    + "_"
    + ".bin"
    for item in map_list1
]

plot_maps(
    file_list1,
    map_list1,
    3,
    2,
    "hitvars",
)


file_list2 = [
    args.base_dir
    + "/out_dir/"
    + "map__"
    + item
    + "_output_N"
    + args.nside
    + "_"
    + ".bin"
    for item in map_list2
]

plot_maps(
    file_list2,
    map_list2,
    1,
    2,
    "polang",
)


file_list3 = [
    args.base_dir
    + "/out_dir/"
    + "map__"
    + item
    + "_output_N"
    + args.nside
    + "_"
    + ".bin_1"
    for item in map_list3
]

plot_maps(file_list3, map_list3, 2, 2, "skymaps", unit="K")
