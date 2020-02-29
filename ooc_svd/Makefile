-include ../make.inc

all: cpu

cpu: ooc_svd.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) ooc_svd.cpp -o cpu $(CPU_LDFLAGS)

clean:
	rm -rf cpu
