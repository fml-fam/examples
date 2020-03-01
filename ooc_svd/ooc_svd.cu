#include <cstdint>
#include <thread>

#include <cpu/cpumat.hh>
#include <cpu/future/io.hh>
#include <cpu/linalg.hh>

#include <gpu/gpuhelpers.hh>
#include <gpu/gpumat.hh>
#include <gpu/gpumat.hh>
#include <gpu/linalg.hh>

#include "progress.hh"

typedef float REAL;


static inline void read_chunk(const char *filename, int pass, cpumat<REAL> &x)
{
  len_t row_first = pass * x.nrows();
  len_t row_last = (pass+1)*x.nrows() - 1;
  io::read_cpu_chunk(filename, row_first, row_last, x);
}

static inline void process_chunk(const cpumat<REAL> &chunk_cpu, gpumat<REAL> &chunk_gpu, gpumat<REAL> &cp_chunk, gpumat<REAL> &cp)
{
  gpuhelpers::cpu2gpu(chunk_cpu, chunk_gpu);
  linalg::crossprod((REAL) 1, chunk_gpu, cp_chunk);
  linalg::add(false, false, (REAL) 1, (REAL) 1, cp_chunk, cp, cp);
}

gpumat<REAL> crossprod_chunk(const char* filename, const uint64_t nrows, const len_t ncols, const len_t chunklen, const bool show_progress=true)
{
  cpumat<REAL> chunk_cpu(chunklen, ncols);
  
  auto c = gpuhelpers::new_card(0);
  gpumat<REAL> chunk_gpu(c, chunklen, ncols);
  gpumat<REAL> cp_chunk(c, ncols, ncols);
  gpumat<REAL> cp(c, ncols, ncols);
  cp.fill_zero();
  
  const int passes = (int) nrows / chunklen;
  progress bar(passes);
  
  read_chunk(filename, 0, chunk_cpu);
  bar.print(show_progress);
  
  for (int pass=1; pass<passes; pass++)
  {
    process_chunk(chunk_cpu, chunk_gpu, cp_chunk, cp);
    read_chunk(filename, pass, chunk_cpu);
    // std::thread process_task(process_chunk, std::ref(chunk_cpu), std::ref(chunk_gpu), std::ref(cp_chunk), std::ref(cp));
    // std::thread read_task(read_chunk, filename, pass, std::ref(chunk_cpu));
    // 
    // process_task.join();
    // read_task.join();
    
    bar.print(show_progress);
  }
  
  process_chunk(chunk_cpu, chunk_gpu, cp_chunk, cp);
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
  
  auto c = cp.get_card();
  gpuvec<REAL> s(c);
  gpumat<REAL> vt(c);
  linalg::eigen_sym(cp, s, vt);
  
  s.info();
  s.print();
  vt.info();
  vt.print();
  
  return 0;
}
