#!/bin/bash

#################################
##### Setting the arguments #####
#################################

### Number of MPI slots to be used
nprocs=4

### Base directory
# base_dir="/marconi_scratch/userexternal/aanand00/dev_sanepic"
base_dir="/mnt/Data/Projects/uniroma2/coding/dev_sanepic_new/freeze_SANEPIC/tests"

### Output location
out_dir="${base_dir}/out_dir/map_"  # For -O

### TOD location
tod_data_dir="${base_dir}/tod_list/"  # For -F
dataSuffixe="output"  # For -B

### Pointings location
pointing_dir="${base_dir}/pointings_list/pointings_"  #

### ispf location
ispf_dir="${base_dir}/bolo_info"
fknee=0.05  # For -k
fsamp2=9.5 ######## 9.5 or 19.0  # For -k
indexN=1.0  # For -k

### Other params
NSIDE=256
fsamp=19.0  # For -f
det_num=1  # For -d
det_name="LFT6_27_140b"  # For -C
it="_N${NSIDE}_.bin"  # For -e
pol=1  # For -p

### Bolometer file
bolofile="${base_dir}/bolo_info/bolometer_info.txt"  # For -X

### Segments info
segment_dir="${base_dir}/segments" # For -f
nn=12  # For -f and -n
seglength=256  # For -u, -f and -l

### HWP params
hwp_status=0  # For -h
OmegaHWPsamp=1.123456789123  # For -w


###########################
##### Calling SANEPIC #####
###########################

mpirun -n ${nprocs} ../sanepic \
-F $tod_data_dir \
-B $dataSuffixe \
-f ${segment_dir}/segment_list_${nn}_${seglength}.bi \
-d $det_num \
-C $det_name \
-e $dataSuffixe$it \
-h $hwp_status \
-w $OmegaHWPsamp \
-k ${ispf_dir}/iSpf${fknee}_${fsamp2}_index${indexN}__ \
-u $seglength \
-O $out_dir \
-p $pol \
-l $seglength \
-n $nn \
-N $NSIDE \
-X $bolofile \
-Z $pointing_dir \
-i 0

