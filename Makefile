#usage: make ARCH=arch
#or export ARCH=arch 
# where:
#        arch = OR
#        arch = Marconi

include makefiles/Make.$(ARCH)

default: mpi

mpi : Sanepic.cc
	${CCC} $(FITS) $(FFTW) Sanepic.cc -o Sanepic

clean :
	rm -f *.o core.* Sanepic


#mpicxx MainSanepicCorr_mpi.cc todprocess.cc map_making.c -lfftw3 -lcfitsio -L/home/patanch/Libs/libgetdata-20070626/.libs -I/home/patanch/Libs/libgetdata-20070626 -lgetdata 
