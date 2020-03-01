-include ../make.inc

all: cpu

cpu: ooc_svd.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) ooc_svd.cpp -o cpu $(CPU_LDFLAGS)

gpu: ooc_svd.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler "$(CPPFLAGS) $(CXXFLAGS) $(GPU_WARNFLAGS) $(OMPFLAGS)" ooc_svd.cu -o gpu $(GPU_LDFLAGS) $(CPU_LDFLAGS)

clean:
	rm -rf cpu gpu
