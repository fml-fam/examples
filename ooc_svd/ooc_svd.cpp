#include <cstdint>
#include <thread>

#include <cpu/cpumat.hh>
#include <cpu/future/io.hh>
#include <cpu/linalg.hh>

#include "progress.hh"

typedef double REAL;


static inline void read_chunk(const char *filename, int pass, cpumat<REAL> &x)
{
  len_t row_first = pass * x.nrows();
  len_t row_last = (pass+1)*x.nrows() - 1;
  io::read_cpu_chunk(filename, row_first, row_last, x);
}

static inline void process_chunk(cpumat<REAL> &chunk, cpumat<REAL> &tmp, cpumat<REAL> &cp)
{
  linalg::crossprod((REAL) 1, chunk, tmp);
  linalg::add(false, false, (REAL) 1, (REAL) 1, tmp, cp, cp);
}

cpumat<REAL> crossprod_chunk(const char* filename, const uint64_t nrows, const len_t ncols, const len_t chunklen, const bool show_progress=true)
{
  cpumat<REAL> chunk(chunklen, ncols);
  cpumat<REAL> cp(ncols, ncols);
  cpumat<REAL> tmp(ncols, ncols);
  
  const int passes = (int) nrows / chunklen;
  progress bar(passes);
  
  read_chunk(filename, 0, chunk);
  bar.print(show_progress);
  
  for (int pass=1; pass<passes; pass++)
  {
    std::thread process_task(process_chunk, std::ref(chunk), std::ref(tmp), std::ref(cp));
    std::thread read_task(read_chunk, filename, pass, std::ref(chunk));
    
    process_task.join();
    read_task.join();
    
    bar.print(show_progress);
  }
  
  process_chunk(chunk, tmp, cp);
  return cp;
}



int main()
{
  const uint64_t nrows = 1e8;
  const len_t ncols = 3;
  const len_t chunklen = 1e6;
  
  const char *filename = "/tmp/x.mat";
  
  // cpumat<REAL> x(nrows, ncols);
  // x.fill_linspace(1, 2);
  // io::write_cpu(filename, x);
  
  auto cp = crossprod_chunk(filename, nrows, ncols, chunklen);
  cp.info();
  cp.print();
  
  cpuvec<REAL> s;
  cpumat<REAL> vt;
  linalg::eigen_sym(cp, s, vt);
  
  s.info();
  s.print();
  vt.info();
  
  return 0;
}
