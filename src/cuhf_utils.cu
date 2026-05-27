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

  namespace {
    // Common allocation shared by both constructors: V batch buffers, X/Y scratch,
    // pinned host V buffer, and the per-k weights initialized to 1/nk.
    void allocate_common_buffers(cuDoubleComplex** VkbatchQij, cuDoubleComplex** VkbatchaQj_conj,
                                 cuDoubleComplex** X_kbatchQij, cuDoubleComplex** X_kbatchiaQ,
                                 cuDoubleComplex** Y_kbatchij,  cuDoubleComplex** weights_fbz,
                                 std::complex<double>** V_kQij_buffer,
                                 cudaStream_t stream, size_t nk, size_t nkbatch, size_t NQnaosq, size_t naosq) {
      using cuda_complex = cuDoubleComplex;
      if (cudaMalloc(VkbatchQij, nkbatch * NQnaosq * sizeof(cuda_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating Vkbatch");
      if (cudaMalloc(VkbatchaQj_conj, nkbatch * NQnaosq * sizeof(cuda_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating Vkbatch_conj");
      if (cudaMallocHost(V_kQij_buffer, nkbatch * NQnaosq * sizeof(std::complex<double>)) != cudaSuccess)
        throw std::runtime_error("failure allocating V on host");
      if (cudaMalloc(X_kbatchQij, nkbatch * NQnaosq * sizeof(cuda_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating XkbatchQij on device");
      if (cudaMalloc(X_kbatchiaQ, nkbatch * NQnaosq * sizeof(cuda_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating XkbatchiaQ on device");
      if (cudaMalloc(Y_kbatchij, nkbatch * naosq * sizeof(cuda_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating Y_kbatchij on device");
      if (cudaMalloc(weights_fbz, nk * sizeof(cuda_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating weights_fbz on device");
      cuDoubleComplex nk_inv            = make_cuDoubleComplex(1. / nk, 0.);
      int             threads_per_block = 512;
      int             blocks_for_id     = nk / threads_per_block + 1;
      initialize_array<<<blocks_for_id, threads_per_block, 0, stream>>>(*weights_fbz, nk_inv, nk);
    }
  }  // namespace

  cuhf_utils::cuhf_utils(size_t nk, size_t ink, size_t ns, size_t nao, size_t NQ, size_t nkbatch, ztensor<4> dm_fbz, int _myid,
                         int _intranode_rank, int _devCount_per_node) :
      _nao(nao), _nso(nao), _nsosq(0), _NQ(NQ), _naosq(nao * nao), _NQnaosq(NQ * nao * nao),
      _ns(ns), _nk(nk), _ink(ink), _nkbatch(nkbatch) {
    if (cudaSetDevice(_intranode_rank % _devCount_per_node) != cudaSuccess) throw std::runtime_error("Error in cudaSetDevice1");
    if (cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(_myid) + ": error initializing cublas");

    allocate_density_and_Fock(&_Dm_fbz_sk2ba, &_F_skij, dm_fbz.data(), _ink, _nk, _nao, _ns);

    if (cudaStreamCreate(&_stream) != cudaSuccess) throw std::runtime_error("main stream creation failed");

    allocate_common_buffers(&_VkbatchQij, &_VkbatchaQj_conj, &_X_kbatchQij, &_X_kbatchiaQ, &_Y_kbatchij,
                            &_weights_fbz, &_V_kQij_buffer, _stream, _nk, _nkbatch, _NQnaosq, _naosq);

    if (ns == 3) {
      _X2C = true;
    } else if (ns == 2 or ns == 1) {
      _X2C = false;
    } else {
      throw std::logic_error("Invalid value of \"ns\" in cuhf_utils.");
    }
  }

  cuhf_utils::cuhf_utils(size_t nk, size_t ink, size_t nao, size_t NQ, size_t nkbatch,
                         const cuDoubleComplex* dm_fbz_nso_device,
                         int _myid, int _intranode_rank, int _devCount_per_node) :
      _X2C(true), _nao(nao), _nso(2 * nao), _nsosq((2 * nao) * (2 * nao)),
      _NQ(NQ), _naosq(nao * nao), _NQnaosq(NQ * nao * nao),
      _ns(3),  // pseudo-ns: aa, bb, ab spin blocks accumulated separately (ba derived later)
      _nk(nk), _ink(ink), _nkbatch(nkbatch) {
    if (cudaSetDevice(_intranode_rank % _devCount_per_node) != cudaSuccess) throw std::runtime_error("Error in cudaSetDevice1");
    if (cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(_myid) + ": error initializing cublas");

    // X2C path keeps dm_fbz in device-resident (nk, nso, nso) layout. add_exchange_to_fock
    // picks aa/bb/ab sub-views inside this buffer with lda=nso and per-ss offsets.
    if (cudaMalloc(&_Dm_fbz_nso, _nk * _nsosq * sizeof(cuDoubleComplex)) != cudaSuccess)
      throw std::runtime_error("failure allocating Dm_fbz_nso on device");
    if (cudaMemcpy(_Dm_fbz_nso, dm_fbz_nso_device, _nk * _nsosq * sizeof(cuDoubleComplex),
                   cudaMemcpyDeviceToDevice) != cudaSuccess)
      throw std::runtime_error("failure copying Dm_fbz_nso input on device");

    // F_skij still in 3-block (ss, ik, nao, nao) layout — downstream Fock assembly
    // (copy_2c_Fock_from_device_to_host) expects this and derives ba = ab.adjoint().
    if (cudaMalloc(&_F_skij, _ns * _ink * _naosq * sizeof(cuDoubleComplex)) != cudaSuccess)
      throw std::runtime_error("failure allocating F_skij on device");
    cudaMemset(_F_skij, 0, _ns * _ink * _naosq * sizeof(cuDoubleComplex));

    if (cudaStreamCreate(&_stream) != cudaSuccess) throw std::runtime_error("main stream creation failed");

    allocate_common_buffers(&_VkbatchQij, &_VkbatchaQj_conj, &_X_kbatchQij, &_X_kbatchiaQ, &_Y_kbatchij,
                            &_weights_fbz, &_V_kQij_buffer, _stream, _nk, _nkbatch, _NQnaosq, _naosq);
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
    if (_Dm_fbz_sk2ba != nullptr) cudaFree(_Dm_fbz_sk2ba);
    if (_Dm_fbz_nso   != nullptr) cudaFree(_Dm_fbz_nso);
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
    const bool x2c_nso = (_Dm_fbz_nso != nullptr);
    for (size_t ss = 0; ss < _ns; ++ss) {
      // Per-ss view into the dm_fbz buffer.
      //   X2C nso layout: pick aa/bb/ab as nao×nao sub-views of the (k2, nso, nso)
      //   block with lda=nso, stride=nsosq, plus a row/col offset selecting the spin
      //   quadrant. ba (ss=3) is derived later as ab.adjoint().
      //   Legacy 3-block (also non-X2C): contiguous (ss, k2, nao, nao), lda=nao,
      //   stride=naosq.
      cuda_complex* dm_ptr;
      int           dm_lda;
      long long     dm_stride;
      if (x2c_nso) {
        size_t row_off = (ss == 1) ? _nao : 0;       // ss=0 aa→0, ss=1 bb→nao, ss=2 ab→0
        size_t col_off = (ss == 0) ? 0    : _nao;    // ss=0 aa→0, ss=1 bb→nao, ss=2 ab→nao
        dm_ptr    = _Dm_fbz_nso + _k2 * _nsosq + row_off * _nso + col_off;
        dm_lda    = static_cast<int>(_nso);
        dm_stride = static_cast<long long>(_nsosq);
      } else {
        dm_ptr    = _Dm_fbz_sk2ba + ss * _nk * _naosq + _k2 * _naosq;
        dm_lda    = static_cast<int>(_nao);
        dm_stride = static_cast<long long>(_naosq);
      }

      // X_skQia(k2, Qi, a) = VkbatchQij(k2, Qi, b) * Dm_fbz(s, k2, b, a)
      GEMM_STRIDED_BATCHED(_handle, CUBLAS_OP_N, CUBLAS_OP_N, _nao, _NQ * _nao, _nao, &one,
                           dm_ptr, dm_lda, dm_stride, _VkbatchQij, _nao, _NQnaosq, &zero,
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
