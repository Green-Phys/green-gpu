/*
 * Copyright (c) 2023 University of Michigan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify,
 * merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <green/gpu/cuhf_utils.h>

__global__ void initialize_array(cuDoubleComplex* array, cuDoubleComplex value, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  array[i] = value;
}

namespace green::gpu {

  cuhf_utils::cuhf_utils(size_t nk, size_t ink, size_t ns, size_t nao, size_t NQ, size_t nkbatch, ztensor<4> dm_fbz, int _myid,
                         int _intranode_rank, int _devCount_per_node) :
      _nk(nk), _ink(ink), _ns(ns), _nao(nao), _NQ(NQ), _nkbatch(nkbatch), _naosq(nao * nao), _NQnaosq(NQ * nao * nao) {
    if (cudaSetDevice(_intranode_rank % _devCount_per_node) != cudaSuccess) throw std::runtime_error("Error in cudaSetDevice1");
    if (cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(_myid) + ": error initializing cublas");

    using cuda_complex = typename cu_type_map<std::complex<double>>::cuda_type;
    allocate_density_and_Fock(&_Dm_fbz_sk2ba, &_F_skij, dm_fbz.data(), _ink, _nk, _nao, _ns);

    if (cudaStreamCreate(&_stream) != cudaSuccess) throw std::runtime_error("main stream creation failed");

    if (cudaMalloc(&_VkbatchQij, nkbatch * _NQnaosq * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating Vkbatch");
    if (cudaMalloc(&_VkbatchaQj_conj, nkbatch * _NQnaosq * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating Vkbatch");

    if (cudaMallocHost(&_V_kQij_buffer, nkbatch * _NQnaosq * sizeof(cxx_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating V on host");

    if (cudaMalloc(&_X_kbatchQij, _nkbatch * _NQnaosq * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating XkbatchQij on device");
    if (cudaMalloc(&_X_kbatchiaQ, _nkbatch * _NQnaosq * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating XkbatchijQ on device");
    if (cudaMalloc(&_Y_kbatchij, _nkbatch * _naosq * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating XkbatchijQ on device");

    if (cudaMalloc(&_weights_fbz, _nk * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating weights_fbz on device");
    cuDoubleComplex nk_inv            = make_cuDoubleComplex(1. / _nk, 0.);
    int             threads_per_block = 512;
    int             blocks_for_id     = _nk / threads_per_block + 1;
    initialize_array<<<blocks_for_id, threads_per_block, 0, _stream>>>(_weights_fbz, nk_inv, _nk);
    if (ns == 3) {
      _X2C = true;
    } else if (ns == 2 or ns == 1) {
      _X2C = false;
    } else {
      throw std::logic_error("Invalid value of \"ns\" in cuhf_utils.");
    }
  }

  cuhf_utils::~cuhf_utils() {
    cudaStreamDestroy(_stream);
    cublasDestroy(_handle);

    cudaFree(_VkbatchQij);
    cudaFree(_VkbatchaQj_conj);
    cudaFree(_X_kbatchQij);
    cudaFree(_X_kbatchiaQ);
    cudaFree(_Y_kbatchij);
    cudaFree(_weights_fbz);
    cudaFree(_Dm_fbz_sk2ba);
    cudaFree(_F_skij);

    cudaFreeHost(_V_kQij_buffer);
  }

  void cuhf_utils::set_up_exchange(cxx_complex* VkbatchQij_host, std::size_t k_pos, std::size_t k2) {
    cudaStreamSynchronize(_stream);
    _k_pos         = k_pos;
    _k2            = k2;
    size_t nk_mult = std::min(_nkbatch, _nk - _k2);
    std::memcpy(_V_kQij_buffer, VkbatchQij_host, nk_mult * _NQnaosq * sizeof(cxx_complex));
    cudaMemcpyAsync(_VkbatchQij, _V_kQij_buffer, nk_mult * _NQnaosq * sizeof(cxx_complex), cudaMemcpyHostToDevice, _stream);

    cublasSetStream(_handle, _stream);
    cuda_complex one  = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    for (size_t kk2 = 0; kk2 < nk_mult; ++kk2) {
      // (k2, Qj, a) -> (k2, a, Qj)*
      GEAM(_handle, CUBLAS_OP_C, CUBLAS_OP_N, _NQ * _nao, _nao, &one, _VkbatchQij + kk2 * _NQnaosq, _nao, &zero,
           _VkbatchaQj_conj + kk2 * _NQnaosq, _NQ * _nao, _VkbatchaQj_conj + kk2 * _NQnaosq, _NQ * _nao);
    }
  }

  void cuhf_utils::add_exchange_to_fock() {
    cudaStreamSynchronize(_stream);
    cuda_complex one       = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex prefactor = (_ns == 1) ? cu_type_map<cxx_complex>::cast(-0.5, 0.) : cu_type_map<cxx_complex>::cast(-1., 0.);
    cuda_complex zero      = cu_type_map<cxx_complex>::cast(0., 0.);
    cublasSetStream(_handle, _stream);

    int nk_mult = std::min(_nkbatch, _nk - _k2);
    for (size_t ss = 0; ss < _ns; ++ss) {
      // X_skQia(k2, Qi, a) = VkbatchQij(k2, Qi, b) * Dm_fbz(s, k2, b, a)
      GEMM_STRIDED_BATCHED(_handle, CUBLAS_OP_N, CUBLAS_OP_N, _nao, _NQ * _nao, _nao, &one,
                           _Dm_fbz_sk2ba + ss * _nk * _naosq + _k2 * _naosq, _nao, _naosq, _VkbatchQij, _nao, _NQnaosq, &zero,
                           _X_kbatchQij, _nao, _NQnaosq, nk_mult);
      // X_kbatchQij(k2, Q, ia) -> (k2, ia, Q)
      for (size_t kk2 = 0; kk2 < nk_mult; ++kk2) {
        GEAM(_handle, CUBLAS_OP_T, CUBLAS_OP_N, _NQ, _naosq, &one, _X_kbatchQij + kk2 * _NQnaosq, _naosq, &zero,
             _X_kbatchiaQ + kk2 * _NQnaosq, _NQ, _X_kbatchiaQ + kk2 * _NQnaosq, _NQ);
      }
      // Y(k2, i, j) = X_kbatchiaQ(k2, i, aQ) * VkbatchaQj_conj(k2, aQ, j)
      GEMM_STRIDED_BATCHED(_handle, CUBLAS_OP_N, CUBLAS_OP_N, _nao, _nao, _NQ * _nao, &one, _VkbatchaQj_conj, _nao, _NQnaosq,
                           _X_kbatchiaQ, _NQ * _nao, _NQnaosq, &zero, _Y_kbatchij, _nao, _naosq, nk_mult);

      // F(s, k, 1, ij) += prefactor * weight_fbz(1, k2) * Y(k2, ij)
      GEMM(_handle, CUBLAS_OP_N, CUBLAS_OP_N, _naosq, 1, nk_mult, &prefactor, _Y_kbatchij, _naosq, _weights_fbz + _k2, nk_mult,
           &one, _F_skij + ss * _ink * _naosq + _k_pos * _naosq, _naosq);
    }
  }

  void cuhf_utils::accumulate_exchange_on_device(std::complex<double>* Vk1k2_Qij, ztensor<4>& V_kbatchQij,
                                                 ztensor<4>& new_Fock, int _nk_batch, integral_reading_type integral_type,
                                                 int devices_rank, int devices_size, const std::vector<size_t>& irre_list,
                                                 hf_reader1& r1, hf_reader2& r2) {
    for (size_t k_reduced_id = devices_rank; k_reduced_id < _ink; k_reduced_id += devices_size) {
      int k = irre_list[k_reduced_id];
      for (size_t k2 = 0; k2 < _nk; k2 += _nk_batch) {
        if (integral_type == as_a_whole) {
          r1(k, k2, Vk1k2_Qij, V_kbatchQij);
        } else {
          r2(k, k2, V_kbatchQij);
        }
        set_up_exchange(V_kbatchQij.data(), k_reduced_id, k2);
        add_exchange_to_fock();
      }
    }
    cudaDeviceSynchronize();
    if (!_X2C) {
      copy_Fock_from_device_to_host(_F_skij, new_Fock.data(), _ink, _nao, _ns);
    } else {
      copy_2c_Fock_from_device_to_host(_F_skij, new_Fock.data(), _ink, _nao);
    }
    cudaDeviceSynchronize();
  }

}  // namespace green::gpu
