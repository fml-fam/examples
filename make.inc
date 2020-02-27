CXX = g++
MPICXX = mpicxx
NVCC = nvcc

CPPFLAGS = -I../../fml/src
CXXFLAGS = -O3 -march=native
OMPFLAGS = -fopenmp
WARNFLAGS = -Wall -pedantic -Wextra

CPU_LDFLAGS = -llapack -lblas
GPU_LDFLAGS = -lcudart -lcublas -lcusolver -lnvidia-ml
MPI_LDFLAGS = -lscalapack-openmpi
