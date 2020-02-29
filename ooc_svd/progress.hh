#ifndef FML_PROGRESS_H
#define FML_PROGRESS_H
#pragma once


#include <chrono>
#include <cstdio>
#include <cstdint>


class progress
{
  public:
    progress(uint64_t maxiter_);
    void reset();
    void set_max(uint64_t maxiter_);
    void print(const bool should_print);
  
  private:
    std::chrono::high_resolution_clock::time_point query_clock() const;
    
    static const int nchars = 50;
    uint64_t maxiter;
    uint64_t iter;
    std::chrono::high_resolution_clock::time_point start;
};



inline progress::progress(uint64_t maxiter_)
{
  start = query_clock();
  maxiter = maxiter_;
  iter = 0;
}



inline void progress::reset()
{
  start = query_clock();
  iter = 0;
}



inline void progress::set_max(uint64_t maxiter_)
{
  maxiter = maxiter_;
}



inline void progress::print(const bool should_print)
{
  if (!should_print)
    return;
  if (iter > maxiter)
    return;
  
  if (iter > 0)
    putchar('\r');
  
  iter++;
  int n_full = (iter * nchars) / maxiter;
  int n_empty = nchars - n_full;
  
  putchar('[');
  for (int i=0; i<n_full; i++)
    putchar('=');
  for (int i=0; i<n_empty; i++)
    putchar('-');
  putchar(']');
  
  std::chrono::duration<double> dur = query_clock() - start;
  double elapsed = dur.count();
  printf("%10.3fs ", elapsed);
  
  if (iter == maxiter)
    printf("\n\n");
}



inline std::chrono::high_resolution_clock::time_point progress::query_clock() const
{
  return std::chrono::high_resolution_clock::now();
}


#endif
