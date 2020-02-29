#include <cstdint>
#include <thread>

#include <cpu/cpumat.hh>
#include <cpu/future/io.hh>
#include <cpu/linalg.hh>

#include "progress.hh"

typedef float REAL;


static inline void read_chunk(const char *filename, int pass, cpumat<REAL> &x)
{
  len_t row_first = pass * x.nrows();
  len_t row_last = (pass+1)*x.nrows() - 1;
  io::read_cpu_chunk(filename, row_first, row_last, x);
}

static inline void process_chunk(const cpumat<REAL> &chunk, cpumat<REAL> &cp_chunk, cpumat<REAL> &cp)
{
  linalg::crossprod((REAL) 1, chunk, cp_chunk);
  linalg::add(false, false, (REAL) 1, (REAL) 1, cp_chunk, cp, cp);
}

cpumat<REAL> crossprod_chunk(const char* filename, const uint64_t nrows, const len_t ncols, const len_t chunklen, const bool show_progress=true)
{
  cpumat<REAL> chunk(chunklen, ncols);
  cpumat<REAL> cp_chunk(ncols, ncols);
  cpumat<REAL> cp(ncols, ncols);
  cp.fill_zero();
  
  const int passes = (int) nrows / chunklen;
  progress bar(passes);
  
  read_chunk(filename, 0, chunk);
  bar.print(show_progress);
  
  for (int pass=1; pass<passes; pass++)
  {
    process_chunk(chunk, cp_chunk, cp);
    read_chunk(filename, 0, chunk);
    // std::thread process_task(process_chunk, std::ref(chunk), std::ref(cp_chunk), std::ref(cp));
    // std::thread read_task(read_chunk, filename, pass, std::ref(chunk));
    // 
    // process_task.join();
    // read_task.join();
    
    bar.print(show_progress);
  }
  
  process_chunk(chunk, cp_chunk, cp);
  return cp;
}



int main()
{
  const uint64_t nrows = 1e8;
  const len_t ncols = 3;
  const len_t chunklen = 1e6;
  
  const char *filename = "/tmp/x.mat";
  
  printf("# Generating file %s\n", filename);
  cpumat<REAL> x(nrows, ncols);
  x.fill_linspace(1, 2);
  io::write_cpu(filename, x);
  
  printf("\n# Computing SVD\n");
  auto cp = crossprod_chunk(filename, nrows, ncols, chunklen);
  cp.info();
  cp.print();
  
  cpuvec<REAL> s;
  cpumat<REAL> vt;
  linalg::eigen_sym(cp, s, vt);
  
  s.info();
  s.print();
  vt.info();
  vt.print();
  
  return 0;
}
