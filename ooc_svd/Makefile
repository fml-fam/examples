-include ../make.inc

all: ooc_svd

ooc_svd:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) ooc_svd.cpp -o ooc_svd $(CPU_LDFLAGS)

clean:
	rm -rf ooc_svd
