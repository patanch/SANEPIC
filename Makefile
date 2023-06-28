CCC = mpic++ -g -O0

CC = g++  -g -O0


FITS = -I/group/cmb/litebird/usr/patanch/packages/cfitsio-4.1.0/include/ -L/group/cmb/litebird/usr/patanch/packages/cfitsio/lib/ -lcfitsio

FFTW = -I/sw/packages/fftw/fftw3/include/  -L/sw/packages/fftw/fftw3/lib/ -lfftw3


#You need to load cfitsio library

#DIR_LIB = /group/cmb/litebird/usr/ysakurai/packages/cfitsio/lib/                                                                                                                       

#export LD_LIBRARY_PATH = $(LD_LIBRARY_PATH):$(DIR_LIB) 


mpi : Sanepic.cc
	${CCC} $(FITS) $(FFTW) Sanepic.cc -o Sanepic


clean :
	-rm Sanepic *.o core.*


#mpicxx MainSanepicCorr_mpi.cc todprocess.cc map_making.c -lfftw3 -lcfitsio -L/home/patanch/Libs/libgetdata-20070626/.libs -I/home/patanch/Libs/libgetdata-20070626 -lgetdata 
