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

#include<functional>
#include<algorithm>
#include<vector>
#include <green/gpu/cuda_common.h>

//allocate G and Sigma on the device, and copy G from host to device
void allocate_G_and_Sigma(cuDoubleComplex ** G, cuDoubleComplex **Sigma, std::complex<double> *G_host, int nk, int nao, int nt){
  if(cudaMalloc(G, nk*nt*nao*nao*sizeof(cuDoubleComplex))!=cudaSuccess) throw std::runtime_error("G could not be allocated");
  if(cudaMalloc(Sigma, nk*nt*nao*nao*sizeof(cuDoubleComplex))!=cudaSuccess) throw std::runtime_error("Sigma could not be allocated");

  if(cudaMemcpy(*G,G_host,nk*nt*nao*nao*sizeof(std::complex<double>), cudaMemcpyHostToDevice)!=cudaSuccess) throw std::runtime_error("failure to copy in G");
  cudaMemset(*Sigma, 0, nk*nt*nao*nao*sizeof(cuDoubleComplex));
}

//allocate G, -G and Sigma on the device, and copy G from host to device
template<typename cu_type>
void allocate_G_and_Sigma(cu_type ** G_kstij, cu_type **G_ksmtij, cu_type **Sigma_kstij,
                          const std::complex<double> *G_tskij_host, int nk, int nao, int nt, int ns) {
  using type_map = typename get_type_map<cu_type>::type_map;
  if(cudaMalloc(G_kstij, nk*ns*nt*nao*nao*sizeof(cu_type))!=cudaSuccess) throw std::runtime_error("G could not be allocated");
  if(cudaMalloc(G_ksmtij, nk*ns*nt*nao*nao*sizeof(cu_type))!=cudaSuccess) throw std::runtime_error("-G could not be allocated");
  if(cudaMalloc(Sigma_kstij, nk*ns*nt*nao*nao*sizeof(cu_type))!=cudaSuccess) throw std::runtime_error("Sigma could not be allocated");

  std::size_t naosq = nao*nao;
  for(std::size_t k=0;k<nk;++k){
    for(std::size_t s=0;s<ns;++s) {
      for (std::size_t t=0;t<nt;++t) {
        std::size_t shift_tsk  = (t*ns*nk + s*nk + k) * naosq;
        std::size_t shift_kst  = (k*ns*nt + s*nt + t) * naosq;
        std::vector<typename type_map::cxx_type> G_tmp(naosq);
        std::transform(G_tskij_host + shift_tsk, G_tskij_host + shift_tsk + naosq, G_tmp.data(),
                       [&](const std::complex<double> & in) {return static_cast<typename type_map::cxx_type>(in);});
        if (cudaMemcpy(*G_kstij + shift_kst, G_tmp.data(), naosq*sizeof(typename type_map::cxx_type), cudaMemcpyHostToDevice)!=cudaSuccess)
          throw std::runtime_error("failure to copy in G");
      }
      for (std::size_t t=0;t<nt;++t) {
        std::size_t shift_kst  = (k*ns*nt + s*nt + t) * naosq;
        std::size_t shift_ksmt = (k*ns*nt + s*nt + (nt-1-t)) * naosq;
        if (cudaMemcpy(*G_ksmtij + shift_kst, *G_kstij + shift_ksmt, naosq*sizeof(cu_type),cudaMemcpyDeviceToDevice)!=cudaSuccess)
          throw std::runtime_error("failure to copy G to -G");
      }
    }
  }
  cudaMemset(*Sigma_kstij, 0, nk*ns*nt*nao*nao*sizeof(cu_type));
}

template<typename cu_type>
void allocate_Sigma(cu_type **Sigma_kstij, int nk, int nao, int nt, int ns) {
  if(cudaMalloc(Sigma_kstij, nk*ns*nt*nao*nao*sizeof(cu_type))!=cudaSuccess)
    throw std::runtime_error("Sigma could not be allocated");
  cudaMemset(*Sigma_kstij, 0, nk*ns*nt*nao*nao*sizeof(cu_type));
}

void allocate_density_and_Fock(cuDoubleComplex **Dm_fbz_skij_device, cuDoubleComplex **F_skij_device,
        const std::complex<double> *Dm_fbz_skij_host, int ink, int nk, int nao, int ns) {
  if(cudaMalloc(Dm_fbz_skij_device, ns*nk*nao*nao*sizeof(cuDoubleComplex))!=cudaSuccess) throw std::runtime_error("Dm_fbz could not be allocated");
  if(cudaMalloc(F_skij_device, ns*ink*nao*nao*sizeof(cuDoubleComplex))!=cudaSuccess) throw std::runtime_error("F could not be allocated");

  if(cudaMemcpy(*Dm_fbz_skij_device, Dm_fbz_skij_host, ns*nk*nao*nao*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice)!=cudaSuccess)
    throw std::runtime_error("Could not copy Dm_fbz");

  cudaMemset(*F_skij_device, 0, ns*ink*nao*nao*sizeof(cuDoubleComplex));
}

void allocate_weights(cuDoubleComplex** weights_fbz_device, const std::complex<double> *weights_fbz_host, int ink, int nk) {
  if(cudaMalloc(weights_fbz_device, nk*sizeof(cuDoubleComplex))!=cudaSuccess) throw std::runtime_error("weights_fbz could not be allocated");

  if(cudaMemcpy(*weights_fbz_device, weights_fbz_host, nk*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice)!=cudaSuccess)
    throw std::runtime_error("Could not copy weights_fbz");
}

template<typename cu_type>
void allocate_IR_transformation_matrices(cu_type ** T_tw_fb, cu_type ** T_wt_bf,
                                         const std::complex<double> *T_tw_fb_host, const std::complex<double> *T_wt_bf_host, int nt, int nw_b) {
  using type_map = typename get_type_map<cu_type>::type_map;
  if(cudaMalloc(T_tw_fb, nt*nw_b*sizeof(cu_type))!=cudaSuccess) throw std::runtime_error("T_tw_fb could not be allocated");
  // Ignore tau = 0 and beta in this transformation
  if(cudaMalloc(T_wt_bf, nw_b*(nt-2)*sizeof(cu_type))!=cudaSuccess) throw std::runtime_error("T_wt_bf could not be allocated");
  std::vector<typename type_map::cxx_type> T_tw(nt * nw_b);
  std::vector<typename type_map::cxx_type> T_wt(nw_b * (nt - 2));
  std::transform(T_tw_fb_host, T_tw_fb_host + nt * nw_b, T_tw.data(),
                 [&](const std::complex<double> & in) {return static_cast<typename type_map::cxx_type>(in);});
  std::transform(T_wt_bf_host, T_wt_bf_host + nw_b * (nt - 2), T_wt.data(),
                 [&](const std::complex<double> & in) {return static_cast<typename type_map::cxx_type>(in);});
  if(cudaMemcpy(*T_tw_fb, T_tw.data(), nt * nw_b * sizeof(typename type_map::cxx_type), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("failure to copy in T_tw_fb");
  if(cudaMemcpy(*T_wt_bf, T_wt.data(), nw_b * (nt - 2) * sizeof(typename type_map::cxx_type), cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("failure to copy in T_wt_bf");
}

//copy Sigma from the device to the host after calculation is done.
template<typename cu_type>
void copy_Sigma_from_device_to_host(cu_type *sigma_kstij_device, std::complex<double> *sigma_tskij_host, int nk, int nao, int nt, int ns){
  using type_map = typename get_type_map<cu_type>::type_map;
  std::size_t naosq = nao*nao;
  std::vector<typename type_map::cxx_type> sigma_ij(naosq);
  for(std::size_t k=0;k<nk;++k){
    for(std::size_t s=0;s<ns;++s) {
      for (std::size_t t=0;t<nt;++t) {
        std::size_t shift_tsk  = (t*ns*nk + s*nk + k) * naosq;
        std::size_t shift_kst  = (k*ns*nt + s*nt + t) * naosq;
        if (cudaMemcpy(sigma_ij.data(), sigma_kstij_device + shift_kst, naosq*sizeof(typename type_map::cxx_type), cudaMemcpyDeviceToHost)!=cudaSuccess)
          throw std::runtime_error("failure to copy Sigma to host");
        std::transform(sigma_tskij_host+shift_tsk, sigma_tskij_host+shift_tsk+naosq, sigma_ij.data(),
                       sigma_tskij_host+shift_tsk,
                       [&](const std::complex<double> &A, const typename type_map::cxx_type &B) {return A + static_cast<std::complex<double> >(B);});
      }
    }
  }
}

// explicit instantiation
template void allocate_Sigma(cuComplex**Sigma_kstij, int nk, int nao, int nt, int ns);
template void allocate_Sigma(cuDoubleComplex**Sigma_kstij, int nk, int nao, int nt, int ns);

template void allocate_G_and_Sigma(cuDoubleComplex ** G_kstij, cuDoubleComplex **G_ksmtij, cuDoubleComplex **Sigma_kstij,
                                   const std::complex<double> *G_tskij_host, int nk, int nao, int nt, int ns);
template void allocate_G_and_Sigma(cuComplex ** G_kstij, cuComplex **G_ksmtij, cuComplex **Sigma_kstij,
                                   const std::complex<double> *G_tskij_host, int nk, int nao, int nt, int ns);

template void copy_Sigma_from_device_to_host(cuDoubleComplex *sigma_kstij_device, std::complex<double> *sigma_tskij_host, int nk, int nao, int nt, int ns);
template void copy_Sigma_from_device_to_host(cuComplex *sigma_kstij_device, std::complex<double> *sigma_tskij_host, int nk, int nao, int nt, int ns);


template void allocate_IR_transformation_matrices(cuDoubleComplex ** T_tw_fb, cuDoubleComplex ** T_wt_bf,
                                                  const std::complex<double> *T_tw_fb_host, const std::complex<double> *T_wt_bf_host, int nt, int nw_b);
template void allocate_IR_transformation_matrices(cuComplex ** T_tw_fb, cuComplex ** T_wt_bf,
                                                  const std::complex<double> *T_tw_fb_host, const std::complex<double> *T_wt_bf_host, int nt, int nw_b);

template<typename cu_type>
void copy_2c_Sigma_from_device_to_host(cu_type *sigma_k4tij_device, std::complex<double> *sigma_tskij_host, int nk, int nao, int nt){
  using type_map = typename get_type_map<cu_type>::type_map;
  std::size_t nso = 2*nao;
  std::size_t nsosq = nso*nso;
  std::size_t naosq = nao*nao;
  std::vector<typename type_map::cxx_type> sigma_buffer(naosq);
  for (std::size_t k=0;k<nk;++k) {
    for (std::size_t t=0;t<nt;++t) {
      for (std::size_t ss=0;ss<4;++ss) {
        size_t shift_k4t = (k*4*nt + ss*nt + t) * naosq;
        if (cudaMemcpy(sigma_buffer.data(), sigma_k4tij_device + shift_k4t, naosq*sizeof(typename type_map::cxx_type), cudaMemcpyDeviceToHost) != cudaSuccess)
          throw std::runtime_error("failure to copy Sigma to host");

        size_t i_shift = (ss/2)*nao;
        size_t j_shift = (ss%2)*nao;
        for (std::size_t i=0;i<nao;++i) {
          for (std::size_t j=0;j<nao;++j) {
            size_t shift = t*nk*nsosq + k*nsosq + (i+i_shift)*nso + (j+j_shift);
            sigma_tskij_host[shift] = static_cast<std::complex<double> >(sigma_buffer[i*nao+j]);
          }
        }
      }
    }
  }
}

void copy_2c_Fock_from_device_to_host(cuDoubleComplex *F_3kij_device, std::complex<double> *F_kij_host, int nk, int nao) {
  std::size_t nso = 2*nao;
  std::size_t nsosq = nso*nso;
  std::size_t naosq = nao*nao;
  std::complex<double> F_buffer[naosq];
  for (std::size_t ss=0;ss<3;++ss) {
    for (std::size_t k=0;k<nk;++k) {
      size_t shift_3k = (ss*nk + k) * naosq;
      if (cudaMemcpy(F_buffer, F_3kij_device + shift_3k, naosq*sizeof(std::complex<double>), cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("failure to copy Sigma to host");

      for (std::size_t i=0;i<nao;++i) {
        for (std::size_t j=0;j<nao;++j) {
          if (ss == 0) {
            size_t shift = k*nsosq + i*nso + j;
            F_kij_host[shift] = F_buffer[i*nao+j];
          } else if (ss == 1) {
            size_t shift = k*nsosq + (i+nao)*nso + (j+nao);
            F_kij_host[shift] = F_buffer[i*nao+j];
          } else {
            size_t shift = k*nsosq + (i)*nso + (j+nao);
            F_kij_host[shift] = F_buffer[i*nao+j];
            shift = k*nsosq + (i+nao)*nso + j;
            F_kij_host[shift] = std::conj(F_buffer[j*nao+i]);
          }
        }
      }
    }
  }
}

void copy_Fock_from_device_to_host(cuDoubleComplex *F_skij_device, std::complex<double> *F_host, int nk, int nao, int ns) {
  std::size_t naosq = nao*nao;
  if(cudaMemcpy(F_host, F_skij_device, ns*nk*naosq*sizeof(std::complex<double>), cudaMemcpyDeviceToHost)!=cudaSuccess)
    throw std::runtime_error("failure to copy Fock to host");
}

void Complex_DoubleToFloat(const std::complex<double>* in, std::complex<float>* out, size_t size) {
  for (int i = 0; i < size; ++i) {
    out[i] = static_cast<std::complex<float> >(in[i]);
  }
}

void Complex_FloatToDouble(const std::complex<float>* in, std::complex<double>* out, size_t size) {
  for (int i = 0; i < size; ++i) {
    out[i] = static_cast<std::complex<double> >(in[i]);
  }
}

__global__ void acquire_lock(int *lock){
  while (atomicCAS(lock, 0, 1) != 0)
    ;
}
__global__ void release_lock(int *lock){
  atomicExch(lock, 0);
}

__global__ void set_batch_pointer(cuDoubleComplex** ptrs_to_set, cuDoubleComplex *start, int stride, int N) {
  int idx=blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>0) return;
  for(int i=0;i<N;++i){
    ptrs_to_set[i]=start+stride*i;
  }
}
__global__ void set_batch_pointer(cuComplex** ptrs_to_set, cuComplex *start, int stride, int N) {
  int idx=blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>0) return;
  for(int i=0;i<N;++i){
    ptrs_to_set[i]=start+stride*i;
  }
}

//call this kernel with at least size blocks/threads
__global__ void hermitian_symmetrize(cuDoubleComplex *M, int size) {
  int  idx=blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>=size) return;
  int i=idx;

  M[i*size+i]=make_cuDoubleComplex(cuCreal(M[i*size+i]),0.);
  for(int j=0;j<i;++j) {
    double upper_real=cuCreal(M[i*size+j]);
    double upper_imag=cuCimag(M[i*size+j]);
    double lower_real=cuCreal(M[j*size+i]);
    double lower_imag=cuCimag(M[j*size+i]);
    double real=0.5*(upper_real+lower_real);
    double imag=0.5*(upper_imag-lower_imag);
    M[i*size+j]=make_cuDoubleComplex(real, imag);
    M[j*size+i]=make_cuDoubleComplex(real,-imag);
  }
}
__global__ void hermitian_symmetrize(cuComplex *M, int size) {
  int  idx=blockIdx.x * blockDim.x + threadIdx.x;
  if(idx>=size) return;
  int i=idx;
  M[i*size+i]=make_cuFloatComplex(cuCrealf(M[i*size+i]),0.);
  for(int j=0;j<i;++j){
    float upper_real=cuCrealf(M[i*size+j]);
    float upper_imag=cuCimagf(M[i*size+j]);
    float lower_real=cuCrealf(M[j*size+i]);
    float lower_imag=cuCimagf(M[j*size+i]);
    float real=0.5*(upper_real+lower_real);
    float imag=0.5*(upper_imag-lower_imag);
    M[i*size+j]=make_cuFloatComplex(real, imag);
    M[j*size+i]=make_cuFloatComplex(real,-imag);
  }
}