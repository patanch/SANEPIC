# SANEPIC

## Installation

`mpic++` and `fftw3` are need for the installation. The code can be compiled with the following:

```bash
make ARCH=<target>
```

where `target` comes from the file names `Makefiles/Make.target`.

Example: To compile on Origin cluster, use `make ARCH=OR` after importing necessary modules.

## Running SANEPIC

Since `SANEPIC` requires a bunch of input files to run and it produces some files, it is better if we organize the input files and outputs. For the sake of convenience, consider making the following directory tree anywhere in you system:

```txt
test_dir
├── bolo_info
│   ├── bolometer_info.txt
│   └── iSpf<fknee>_<fsample>_index<index>__<detector_name>
├── out_dir
├── pointings_list
│   ├── pointings_dec.bi
│   ├── pointings_psi.bi
│   └── pointings_ra.bi
├── segments
│   └── segment_list_12_256.bi
└── tod_list
    └── data_<detector_name>output.bi
```

The files in this directory tree are following:

1. **`bolometer_info.txt`**
  
    This file contains the information about the detectors. I don't know what information the first three columns contain, but the fourth column is the detector orientation angle, the fifth is the name of the telescope (LFT, MFT, HFT) and the last column is the detector name. The detector name defined here must be used to replace the `detector_name` in the files below.

2. **`iSpf<fknee>_<fsample>_index<index>__<detector_name>`**

    This binary file contains the noise power spectrum. `fknee`, `fsample` and `index` must be replaced with the $f_{knee}$, $f_{sample}$ and power index respectively of the $1/f$ spectrum.

3. **`pointings_dec.bi`**

    This binary file contains the declination angle of the detector pointing. Note that this file name is generic (not related to a detector). Be careful when you are using multiple detectors for map-making. Same applies to the three files below.

4. **`pointings_psi.bi`**

    This binary file contains the polarization angle associated with the pointing.

5. **`pointings_ra.bi`**

    This binary file contains the right ascension angle of the detector pointing.

6. **`segment_list_<nn>_<seglength>.bi`**

    This binary file contains segmentation information. It tells SANEPIC in how many chunks the TOD and pointing files must be read from the disk and distributed among the MPI processes (To be verified). The product of `nn` and `seglength` must be at least the length of TOD. This file can be produced with `helpers/generate_segfile.c` code. It can be compiled with `gcc generate_segfile.c -o generate_segfile.x` and used as `./generate_segfile.x [filename] [number of segments] [seglength]`.

7. **`data_<detector_name>output.bi`**

    This binary file contains the detector TOD.

To run SANEPIC with the input files, one can use the script `./scripts/run_sanepic.sh` provided in this repository. Before running the script, make sure to update the following variables:

- **`nprocs`**: The number of MPI slots to be used
- **`base_dir`**: The path of the base directory. In this case, it the path that contains the directory tree shown above.
- **`fknee`**: $f_{knee}$ of the $1/f$ spectrum. It must be same as in the file available in the directory tree.
- **`fsamp2`**: $f_{sample}$ of the $1/f$ spectrum. It must be same as in the file available in the directory tree.
- **`indexN`**: Power index of the $1/f$ spectrum. It must be same as in the file available in the directory tree.
- **`NSIDE`**: $N_{nside}$ of the output maps.
- **`fsamp`**: $f_{samp}$ of the TOD.
- **`det_num`**: Number of detectors to be used in map-making.
- **`det_name`**: Name of the detector to be used in map-making. The information of the corresponding detector must be available in the `bolometer_info.txt` file.
- **`pol`**: Set this to 1 if you want to produced the polarization maps. Set it to 0 otherwise.
- **`nn`**: Number of chunks in which TOD must be divided. See `segment_list_<nn>_<seglength>.bi` above.
- **`seglength`**: Length of each TOD chunks. See `segment_list_<nn>_<seglength>.bi` above.
- **`hwp_status`**: Set this to 1 if you want to use HWP modulated signal. Set it to 0 otherwise.
- **`OmegaHWPsamp`**: Doesn't do anything tbh.

Once all the input files have been prepared and parameters are updated in the shell script, one can simply run SANEPIC with

```bash
bash run_sanepic.sh
```


## SANEPIC outputs

SANEPIC produces bunch of outputs files in binary format. With the shell script provided in this repo, the output files are stored in `out_dir`. Typically the following files will be produced:

```txt
map__A_output_N<nside>_.bin_1
map__COS2P_output_N<nside>_.bin
map__HITSoutput_N<nside>_.bin
map__I_output_N<nside>_.bin_1
map__NVARoutput_N<nside>_.bin
map__POL_HITSoutput_N<nside>_.bin
map__Q_NVARoutput_N<nside>_.bin
map__Q_output_N<nside>_.bin_1
map__SIN2P_output_N<nside>_.bin
map__U_NVARoutput_N<nside>_.bin
map__U_output_N<nside>_.bin_1
```

One can use the python script `helpers/plot_sanepic_maps.py` to plot all the output maps at once. The script must be run as

```bash
python plot_sanepic_maps.py <base_dir> <nside>
```
