#include <cstdint>

#include <cpu/cpuhelpers.hh>
#include <cpu/cpumat.hh>
#include <cpu/linalg.hh>

#include <mpi/mpimat.hh>

typedef double REAL;


void decomp(mpimat<REAL> &x)
{
  grid g = x.get_grid();
  const len_t m_local = x.nrows_local();
  const len_t n_local = x.ncols_local();
  
  cpumat<REAL> x_local;
  x_local.inherit(x.data_ptr(), m_local, n_local);
  cpumat<REAL> y(m_local, n_local);
  cpumat<REAL> z(m_local, n_local);
  
  for (int k=0; k<g.nprow(); k++)
  {
    if (g.myrow() == k && g.mycol() == k)
      linalg::invert(x_local);
    
    if (g.myrow() == k)
    {
      if (g.mycol() == k)
        g.bcast(m_local, n_local, x_local.data_ptr(), 'R', k, k);
      else
      {
        g.bcast(m_local, n_local, y.data_ptr(), 'R', k, k);
        cpuhelpers::cpu2cpu(x_local, z);
        linalg::matmult(false, false, (REAL) 1.0, y, z, x_local);
      }
    }
    
    if (g.mycol() == k)
    {
      if (g.myrow() == k)
        g.bcast(m_local, n_local, x_local.data_ptr(), 'C', k, k);
      else
      {
        g.bcast(m_local, n_local, y.data_ptr(), 'C', k, k);
        cpuhelpers::cpu2cpu(x_local, z);
        linalg::matmult(false, false, (REAL) 1.0, z, y, x_local);
      }
    }
  }
}



int main()
{
  grid g = grid(PROC_GRID_SQUARE);
  if (g.nprow() != g.npcol())
    throw std::runtime_error("grid must be square");
  
  const uint64_t n = 10;
  mpimat<REAL> x(g, n, n, n/g.nprow(), n/g.npcol());
  x.fill_runif((uint32_t)1234);
  
  x.print();
  decomp(x);
  x.print();
  
  g.exit();
  g.finalize();
  
  return 0;
}
