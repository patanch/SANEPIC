# usage: make ARCH=arch
# or export ARCH=arch 
# where:
#        arch = OR
#        arch = Marconi

.PHONY: default mpi clean

default: mpi

mpi : src/sanepic.cc
	$(eval include makefiles/Make.$(ARCH))
	${CCC} src/sanepic.cc -o sanepic $(FITS) $(FFTW)

clean :
	rm -f *.o core.* sanepic


#mpicxx MainSanepicCorr_mpi.cc todprocess.cc map_making.c -lfftw3 -lcfitsio -L/home/patanch/Libs/libgetdata-20070626/.libs -I/home/patanch/Libs/libgetdata-20070626 -lgetdata
