CXX = g++
MPICXX = mpicxx
NVCC = nvcc
FMLPATH = -I../../fml/src

WARNFLAGS = -Wall -pedantic -Wextra
GPU_WARNFLAGS = -Wall -Wextra

CPPFLAGS = $(FMLPATH)
CXXFLAGS = -O3 -march=native
OMPFLAGS = -fopenmp
NVCCFLAGS = -arch=sm_61

CPU_LDFLAGS = -llapack -lblas
GPU_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml
MPI_LDFLAGS = -lscalapack-openmpi
