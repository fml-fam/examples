-include ../make.inc

all: decomp

decomp:
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(WARNFLAGS) $(OMPFLAGS) decomp.cpp -o decomp $(MPI_LDFLAGS) $(CPU_LDFLAGS)

clean:
	rm -rf decomp
