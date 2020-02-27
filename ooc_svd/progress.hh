#ifndef FML_PROGRESS_H
#define FML_PROGRESS_H
#pragma once


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
    uint64_t maxiter;
    uint64_t iter;
    
    static const int nchars = 60;
};


inline progress::progress(uint64_t maxiter_)
{
  maxiter = maxiter_;
  iter = 0;
}



inline void progress::reset()
{
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
  
  if (iter == maxiter)
    printf("\n\n");
}


#endif
