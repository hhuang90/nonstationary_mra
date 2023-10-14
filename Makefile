# Choose the available MPI C++ compilier
CXX=mpic++

# OpenMP setting
## Set the header file path for OpenMP
ompIncPath=.
## Set the header file path for OpenMP
ompLibPath=.
## Change the OpenMP library name if what is installed has a different name
ompLib=omp

# Set the header file path for the prequisite libraries
## path for mkl header files
mklIncPath=/opt/intel/mkl/include
## path for dlib header files
dlibIncPath=../library/include
## path for armadillo header files
armaIncPath=.

# Set the library path for the prequisite libraries
## path for mkl library
mklLibPath=/opt/intel/mkl/lib
## path for dlib library
dlibLibPath=../library/lib
## path for armadillo library
armaLibPath=.

# Change the library names if what are installed have different names
## mkl library name
mklLib=mkl_rt
## dlib library name
dlibLib=dlib
## armadillo library name
armaLib=armadillo

# Uncomment "-g" if the debug mode is desired
DEBUG=-g

include=-I./include $(if $(ompIncPath),-I$(ompIncPath),) $(if $(mklIncPath),-I$(mklIncPath),) $(if $(dlibIncPath),-I$(dlibIncPath),) $(if $(armaIncPath),-I$(armaIncPath),)
LIBS=-lm  -l$(ompLib) -l$(mklLib) -l$(dlibLib) -l$(armaLib) $(if $(ompLibPath),-L$(ompLibPath),) $(if $(mklLibPath),-L$(mklLibPath),) $(if $(dlibLibPath),-L$(dlibLibPath),) $(if $(armaLibPath),-L$(armaLibPath),)

# LIBS=-lm  -l$(ompLib) -l$(dlibLib) -l$(armaLib) $(if $(ompLibPath),-L$(ompLibPath),) $(if $(dlibLibPath),-L$(dlibLibPath),) $(if $(armaLibPath),-L$(armaLibPath),) -Wl,-rpath /opt/intel/mkl/lib libmkl_rt.dylib

CXXFLAGS=$(include) -O2 $(DEBUG) -std=c++11 

objDIR=./obj
srcDIR=./src

SRC=$(shell find $(srcDIR) -name '*.cpp')
OBJ=$(patsubst $(srcDIR)/%.cpp,$(objDIR)/%.o,$(SRC))

$(objDIR)/%.o: $(srcDIR)/%.cpp
	$(CXX) -c -o $@ $< $(CXXFLAGS)

MRA: $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(OBJ) MRA
