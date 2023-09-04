CCC = mpic++ -g -O2
# CCC = mpiicc -g -O2

CC = g++  -g -O0


FITS = -I${CFITSIO_INCLUDE} -L${CFITSIO_LIB} -lcfitsio

FFTW = -I${FFTW_INCLUDE} -L${FFTW_LIB} -lfftw3


#You need to load cfitsio library

#DIR_LIB = /group/cmb/litebird/usr/ysakurai/packages/cfitsio/lib/                                                                                                                       

#export LD_LIBRARY_PATH = $(LD_LIBRARY_PATH):$(DIR_LIB) 


mpi : src/sanepic.cc
	${CCC} $(FITS) $(FFTW) -D_DEBUG1 src/sanepic.cc -o sanepic


clean :
	-rm sanepic *.o core.*



# # Makefile for Sanepic

# .PHONY: default gcc mpi clean

# default: mpi

# # EXE = sanepic_mpi

# # gcc: CC = gcc
# # gcc: CPPFLAGS += -g -O0
# # gcc: EXE = sanepic_gcc

# mpi: CC = mpic++
# mpi: CPPFLAGS += -g -O0
# mpi: EXE = sanepic_mpi

# LIBS += -lfftw3

# OBJS = src/sanepic.o

# $(EXE): $(OBJS)
# 		@echo ${EXE}
# 		$(CC) $(CPPFLAGS) $(OBJS) $(LIBS) $(OPTIONS) -o $(EXE)

# %.o: src/%.cc
# 		$(CC) -c $(CPPFLAGS) $(OPTIONS) $< -o $@

# EXE_LIST = sanepic_gcc sanepic_mpi

# clean:
# 		@echo $(EXE)
# 		rm -f $(EXE_LIST) $(OBJS)
