/*
* Copyright (c) 2023 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the “Software”), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <green/gpu/cugw_qpt.h>

__global__ void validate_info(int *info){
  int idx=blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>0) return;
  if(*info!=0){
    printf("info is: %d\n",*info);
    printf("nonzero info. Aborting application.\n");
    asm("trap;"); // nonzero info = cholesky or LU fails, then all threads should be stopped
  }
}
__global__ void validate_info(int *info, int N){
  int idx=blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>0) return;
  for(int i=0;i<N;++i){
    if(*(info+i)!=0){
      printf("info is: %d\n",*(info+i));
      printf("nonzero info for batched job: %d. Aborting application.\n",i);
      asm("trap;"); // nonzero info = cholesky or LU fails, then all threads should be stopped
    }
  }
}
__global__ void set_up_one_minus_P(cuDoubleComplex *one_minus_P, cuDoubleComplex *P, int naux){
  int  i=blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=naux) return;
  cuDoubleComplex one=make_cuDoubleComplex(1.,0.);
  cuDoubleComplex zero=make_cuDoubleComplex(0.,0.);
  for(int j=0;j<naux;++j){
    one_minus_P[i*naux+j] = cuCsub(zero,P[i*naux+j]);
  }
  one_minus_P[i*naux+i]= cuCadd(one_minus_P[i*naux+i],one);
}
__global__ void set_up_one_minus_P(cuComplex *one_minus_P, cuComplex *P, int naux){
  int  i=blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=naux) return;
  cuComplex one=make_cuComplex(1.,0.);
  cuComplex zero=make_cuComplex(0.,0.);
  for(int j=0;j<naux;++j){
    one_minus_P[i*naux+j] = cuCsubf(zero,P[i*naux+j]);
  }
  one_minus_P[i*naux+i]= cuCaddf(one_minus_P[i*naux+i],one);
}

