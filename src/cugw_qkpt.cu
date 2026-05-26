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

#include <iostream>

#include <green/gpu/cugw_qkpt.h>


namespace green::gpu {
  template<typename prec>
  gw_qkpt<prec>::gw_qkpt(int nao, int naux, int ns, int nt, int nt_batch, cublasHandle_t* handle, cuda_complex* g_ktij,
                         cuda_complex* g_kmtij, cuda_complex* sigma_ktij, int* sigma_k_locks) :
      g_ktij_(g_ktij), g_kmtij_(g_kmtij), sigma_ktij_(sigma_ktij), sigma_k_locks_(sigma_k_locks), nao_(nao), nao2_(nao * nao),
      nao3_(nao2_ * nao), naux_(naux), naux2_(naux * naux), nauxnao_(naux * nao), nauxnao2_(naux * nao * nao), ns_(ns), nt_(nt),
      nt_batch_(nt_batch), ntnaux_(nt * naux), ntnaux2_(nt * naux * naux), ntnao_(nt * nao), ntnao2_(nt * nao2_),
      handle_(handle), cleanup_req_(false) {
    _low_memory_requirement = (g_ktij == nullptr) ? true : false;
    if (cudaStreamCreate(&stream_) != cudaSuccess) throw std::runtime_error("main stream creation failed");

    // interaction matrix and its transpose
    if (cudaMalloc(&V_Qpm_, nauxnao2_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating V on device");
    if (cudaMalloc(&V_pmQ_, nauxnao2_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating V on device");
    // intermediate vars for strided batched multiplies
    if (cudaMalloc(&X1t_tmQ_, nt_batch_ * nao2_ * naux_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating X1 on device");
    if (cudaMalloc(&X2t_Ptm_, nt_batch_ * nao2_ * naux_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating X2 on device");
    // buffer for self-energy and Green's function
    if (cudaMalloc(&sigmak_stij_, ns_ * nt_ * nao2_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating sigma on device");
    if (cudaMalloc(&g_stij_, ns_ * ntnao2_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating g_tij on device");
    if (cudaMalloc(&g_smtij_, ns_ * ntnao2_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating g_mtij on device");

    if (cudaMallocHost(&V_Qpm_buffer_, nauxnao2_ * sizeof(cxx_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating V on host");
    if (_low_memory_requirement) {
      if (cudaMallocHost(&Gk1_stij_buffer_, ns_ * ntnao2_ * sizeof(cxx_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating Gk1_stij on host");
      if (cudaMallocHost(&Gk_smtij_buffer_, ns_ * ntnao2_ * sizeof(cxx_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating Gk_smtij on host");
      if (cudaMallocHost(&Sigmak_stij_buffer_, ns_ * ntnao2_ * sizeof(cxx_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating Sigmak_stij on host");
    }

    if (cudaMalloc(&Pqk0_tQP_local_, nt_batch_ * naux2_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating Pq0");

    // Per-worker scratch for cu_symmetry::transform_k_ao_device (eliminates shared-scratch race)
    size_t scratch_elems = static_cast<size_t>(ns_) * nt_ * nao2_;
    if (cudaMalloc(&transform_input_scratch_, scratch_elems * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating transform_input_scratch on device");
    if (cudaMalloc(&transform_work_scratch_, scratch_elems * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating transform_work_scratch on device");

    cudaEventCreateWithFlags(&data_ready_event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&all_done_event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&transform_done_event_, cudaEventDisableTiming);

    // set memory alias
    V_Qim_ = V_Qpm_;
    V_nPj_ = V_pmQ_;
  }

  template <typename prec>
  gw_qkpt<prec>::~gw_qkpt() {
    cudaStreamDestroy(stream_);
    cudaEventDestroy(data_ready_event_);
    cudaEventDestroy(all_done_event_);
    cudaEventDestroy(transform_done_event_);

    cudaFree(V_Qpm_);
    cudaFree(V_pmQ_);
    cudaFree(X1t_tmQ_);
    cudaFree(X2t_Ptm_);
    cudaFree(Pqk0_tQP_local_);
    cudaFree(g_stij_);
    cudaFree(g_smtij_);
    cudaFree(sigmak_stij_);
    cudaFree(transform_input_scratch_);
    cudaFree(transform_work_scratch_);

    cudaFreeHost(V_Qpm_buffer_);
    if (_low_memory_requirement) {
      cudaFreeHost(Gk1_stij_buffer_);
      cudaFreeHost(Gk_smtij_buffer_);
      cudaFreeHost(Sigmak_stij_buffer_);
    }
    if (require_cleanup()) {
      std::cerr << "gw_qkpt: destroyed with pending self-energy cleanup; sigma contribution will be lost.\n";
    }
  }

  template <typename prec>
  void gw_qkpt<prec>::upload_coulomb_v_first(cxx_complex* V_Qpm_host) {
    std::memcpy(V_Qpm_buffer_, V_Qpm_host, nauxnao2_ * sizeof(cxx_complex));
    cudaMemcpyAsync(V_Qpm_, V_Qpm_buffer_, nauxnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);
    cublasSetStream(*handle_, stream_);
    int      two   = 2;
    scalar_t alpha = -1;
    cuda_complex one  = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    // V_pmQ = V_Qpm^T  (conjugation of V_Qpm applied separately via RSCAL)
    if (GEAM(*handle_, CUBLAS_OP_T, CUBLAS_OP_N, naux_, nao2_, &one, V_Qpm_, nao2_, &zero, V_pmQ_, naux_, V_pmQ_, naux_) !=
        CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("GEAM fails in upload_coulomb_v_first.");
    if (RSCAL(*handle_, nauxnao2_, &alpha, (scalar_t*)V_Qpm_ + 1, two) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("RSCAL fails in upload_coulomb_v_first.");
  }

  template <typename prec>
  void gw_qkpt<prec>::upload_p0_inputs(cxx_complex* Gk1_stij_host, cxx_complex* Gk_smtij_host, cxx_complex* V_Qpm_host, int k,
                                      int k1) {
    cudaStreamSynchronize(stream_);
    k_  = k;
    k1_ = k1;
    upload_coulomb_v_first(V_Qpm_host);
    if (_low_memory_requirement) {
      std::memcpy(Gk1_stij_buffer_, Gk1_stij_host, ns_ * ntnao2_ * sizeof(cxx_complex));
      std::memcpy(Gk_smtij_buffer_, Gk_smtij_host, ns_ * ntnao2_ * sizeof(cxx_complex));
      cudaMemcpyAsync(g_stij_, Gk1_stij_buffer_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);
      cudaMemcpyAsync(g_smtij_, Gk_smtij_buffer_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);
    } else {
      cudaMemcpyAsync(g_stij_, g_ktij_ + k1_ * ns_ * ntnao2_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyDeviceToDevice,
                      stream_);
      cudaMemcpyAsync(g_smtij_, g_kmtij_ + k_ * ns_ * ntnao2_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyDeviceToDevice,
                      stream_);
    }
    cudaEventRecord(data_ready_event_, stream_);
  }

  template <typename prec>
  void gw_qkpt<prec>::upload_p0_coulomb(cxx_complex* V_Qpm_host, int k, int k1) {
    cudaStreamSynchronize(stream_);
    k_  = k;
    k1_ = k1;
    upload_coulomb_v_first(V_Qpm_host);
    cudaEventRecord(data_ready_event_, stream_);
  }

  template <typename prec>
  void gw_qkpt<prec>::upload_coulomb_v_second(cxx_complex* V_Qim_host) {
    std::memcpy(V_Qpm_buffer_, V_Qim_host, nauxnao2_ * sizeof(cxx_complex));
    cudaMemcpyAsync(V_Qim_, V_Qpm_buffer_, nauxnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);
    cublasSetStream(*handle_, stream_);
    cuda_complex one  = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    // V_nPj = V_Qim†
    if (GEAM(*handle_, CUBLAS_OP_C, CUBLAS_OP_N, nauxnao_, nao_, &one, V_Qim_, nao_, &zero, V_nPj_, nauxnao_, V_nPj_, nauxnao_) !=
        CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("GEAM fails in upload_coulomb_v_second.");
  }

  template <typename prec>
  void gw_qkpt<prec>::upload_sigma_inputs(cxx_complex* Gk1_stij_host, cxx_complex* V_Qim_host, int k, int k1) {
    cudaStreamSynchronize(stream_);
    k_  = k;
    k1_ = k1;
    upload_coulomb_v_second(V_Qim_host);
    if (_low_memory_requirement) {
      std::memcpy(Gk1_stij_buffer_, Gk1_stij_host, ns_ * ntnao2_ * sizeof(cxx_complex));
      cudaMemcpyAsync(g_stij_, Gk1_stij_buffer_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);
    } else {
      cudaMemcpyAsync(g_stij_, g_ktij_ + k1_ * ns_ * ntnao2_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyDeviceToDevice,
                      stream_);
    }
    cudaEventRecord(data_ready_event_, stream_);
  }

  template <typename prec>
  void gw_qkpt<prec>::load_Gk1_to_device(cxx_complex* host_ptr, size_t n_elems) {
    // Copy from unpinned source into per-worker pinned buffer, then stage async to device.
    // Safe against concurrent workers because each worker has its own pinned buffer,
    // and cudaStreamSynchronize at the top of upload_p0_coulomb ensures
    // the pinned buffer is not in use from a prior async copy.
    std::memcpy(Gk1_stij_buffer_, host_ptr, n_elems * sizeof(cxx_complex));
    cudaMemcpyAsync(g_stij_, Gk1_stij_buffer_, n_elems * sizeof(cuda_complex),
                    cudaMemcpyHostToDevice, stream_);
  }

  template <typename prec>
  void gw_qkpt<prec>::compute_first_tau_contraction(cuda_complex* Pqk0_tQP, int* Pqk0_tQP_lock) {
    cuda_complex one       = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero      = cu_type_map<cxx_complex>::cast(0., 0.);
    cuda_complex prefactor = (ns_ == 1) ? cu_type_map<cxx_complex>::cast(-2., 0.) : cu_type_map<cxx_complex>::cast(-1., 0.);
    cublasSetStream(*handle_, stream_);
    // Only compute Pq0(t) for t = [0,beta/2] since Pq0(t) = Pq0(beta-t)
    for (int s = 0; s < ns_; ++s) {
      for (int t = 0; t < nt_ / 2; t += nt_batch_) {
        int st      = s * nt_ + t;
        int nt_mult = std::min(nt_batch_, nt_ / 2 - t);
        // X1_t_mQ = G_t_p * V_pmQ; G_tp = G^{k}(-t)_tp
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, nao_ * naux_, nao_, nao_, &one, V_pmQ_, nauxnao_, 0,
                                 g_smtij_ + st * nao2_, nao_, nao2_, &zero, X1t_tmQ_, nauxnao_, nauxnao2_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_first_tau_contraction().");
        }
        // X2_Pt_m = (V_Pt_n)* * G_m_n; G_mn = G^{k1}(t)_{mn}
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_T, CUBLAS_OP_N, nao_, nauxnao_, nao_, &one, g_stij_ + st * nao2_, nao_,
                                 nao2_, V_Qpm_, nao_, 0, &zero, X2t_Ptm_, nao_, nauxnao2_, nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_first_tau_contraction().");
        }
        // Pq0_QP=X2_Ptm Q1_tmQ
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_T, CUBLAS_OP_T, naux_, naux_, nao2_, &prefactor, X2t_Ptm_, nao2_, nauxnao2_,
                                 X1t_tmQ_, naux_, nauxnao2_, &zero, Pqk0_tQP_local_, naux_, naux2_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_first_tau_contraction().");
        }
        write_P0(t, Pqk0_tQP, Pqk0_tQP_lock);
      }
    }
    cudaEventRecord(all_done_event_);
  }

  template <typename prec>
  void gw_qkpt<prec>::write_P0(int t, cuda_complex* Pqk0_tQP, int* Pqk0_tQP_lock) {
    int nt_mult = std::min(nt_batch_, nt_ / 2 - t);
    acquire_lock<<<1, 1, 0, stream_>>>(Pqk0_tQP_lock);
    scalar_t one = 1.;
    if (RAXPY(*handle_, 2 * naux2_ * nt_mult, &one, (scalar_t*)Pqk0_tQP_local_, 1, (scalar_t*)(Pqk0_tQP + t * naux2_), 1) !=
        CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("RAXPY fails on gw_qkpt.write_P0().");
    }
    release_lock<<<1, 1, 0, stream_>>>(Pqk0_tQP_lock);
  }

  template <typename prec>
  void gw_qkpt<prec>::compute_second_tau_contraction(cuda_complex* Pqk_tQP, const cuda_complex* U_q, bool q_conj_after_uq) {
    cuda_complex  one     = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex  zero    = cu_type_map<cxx_complex>::cast(0., 0.);
    cuda_complex  m1      = cu_type_map<cxx_complex>::cast(-1., 0.);
    cuda_complex* Y1t_Qin = X1t_tmQ_;  // name change, reuse memory
    cuda_complex* Y2t_inP = X2t_Ptm_;  // name change, reuse memory
    cublasSetStream(*handle_, stream_);
    for (int s = 0; s < ns_; ++s) {
      for (int t = 0; t < nt_; t += nt_batch_) {
        int st      = s * nt_ + t;
        int nt_mult = std::min(nt_batch_, nt_ - t);
        // GEMM 1: Y1_Qin = V_Qim * G1_mn
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nauxnao_, nao_, &one, g_stij_ + st * nao2_, nao_,
                                 nao2_, V_Qim_, nao_, 0, &zero, Y1t_Qin, nao_, nauxnao2_, nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
        }
        if (U_q != nullptr) {
          // q-space symmetry transform: Y2 = U * P * U† * Y1^T
          // U_q stored row-major as-is. CUBLAS sees U_q^T (col-major).
          // 2a: effective op = OP_N(U_q^T) = U_q^T  →  T1 = U_q^T * Y1
          cublasOperation_t OP_Uq_Left = q_conj_after_uq ? CUBLAS_OP_C : CUBLAS_OP_N;
          if (GEMM_STRIDED_BATCHED(*handle_, OP_Uq_Left, CUBLAS_OP_T, naux_, nao2_, naux_, &one, U_q, naux_, 0,
                                   Y1t_Qin, nao2_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction() [2a].");
          }
          // 2b: T2(Q,in) = P_ibz(Q,Q') * T1(Q',in) → store in Y1 (consumed)
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, naux_, nao2_, naux_, &one, Pqk_tQP + t * naux2_, naux_,
                                   naux2_, Y2t_inP, naux_, nauxnao2_, &zero, Y1t_Qin, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction() [2b].");
          }
          // 2c: effective op = OP_C(U_q^T) = U_q^*  →  Y2 = U_q^* * T2
          cublasOperation_t OP_Uq_Right = q_conj_after_uq ? CUBLAS_OP_N : CUBLAS_OP_C;
          if (GEMM_STRIDED_BATCHED(*handle_, OP_Uq_Right, CUBLAS_OP_N, naux_, nao2_, naux_, &one, U_q, naux_, 0,
                                   Y1t_Qin, naux_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction() [2c].");
          }
        } else {
          // No q-space transform: Y2(Q,in) = P(Q,Q') * Y1^T(Q',in)
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_T, naux_, nao2_, naux_, &one, Pqk_tQP + t * naux2_, naux_,
                                   naux2_, Y1t_Qin, nao2_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
          }
        }
        // GEMM 3: Sigma_ij = -Y2_inP * V_nPj
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nao_, nauxnao_, &m1, V_nPj_, nao_, 0, Y2t_inP,
                                 nauxnao_, nauxnao2_, &zero, sigmak_stij_ + st * nao2_, nao_, nao2_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
        }
      }
    }
    write_sigma(_low_memory_requirement);
    cudaEventRecord(all_done_event_);
  }

  template <typename prec>
  void gw_qkpt<prec>::compute_second_tau_contraction_2C(cuda_complex* Pqk_tQP, const cuda_complex* U_q, bool q_conj_after_uq) {
    cuda_complex  one     = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex  zero    = cu_type_map<cxx_complex>::cast(0., 0.);
    cuda_complex  m1      = cu_type_map<cxx_complex>::cast(-1., 0.);
    cuda_complex* Y1t_Qin = X1t_tmQ_;  // name change, reuse memory
    cuda_complex* Y2t_inP = X2t_Ptm_;  // name change, reuse memory
    cublasSetStream(*handle_, stream_);
    // g_stij = g_stij(aa, bb, ab, ba)
    // Since we are only interested in sigmak_stij(aa, bb, ab), we only loop over ns = 3
    for (int s = 0; s < 3; ++s) {
      for (int t = 0; t < nt_; t += nt_batch_) {
        int st      = s * nt_ + t;
        int nt_mult = std::min(nt_batch_, nt_ - t);
        // Y1_Qin = V_Qim * G1_mn; G1_mn = G^{k1}(t)_mn
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nauxnao_, nao_, &one, g_stij_ + st * nao2_, nao_,
                                 nao2_, V_Qim_, nao_, 0, &zero, Y1t_Qin, nao_, nauxnao2_, nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction_2C().");
        }
        if (U_q != nullptr) {
          // q-space symmetry transform: Y2 = U_q^left * P * U_q^right * Y1
          // U_q stored row-major as-is. CUBLAS sees U_q^T (col-major).
          //   OP_N(U_q^T) = U_q^T      OP_C(U_q^T) = U_q^*
          // Non-TR: Y2 = U_q^T * P * U_q^* * Y1      (Left = OP_N, Right = OP_C)
          // TR    : Y2 = U_q^* * P * U_q^T * Y1      (Left = OP_C, Right = OP_N)
          // Folding the TR conjugation into the U_q OPs (mirroring the scalar
          // path) is mathematically equivalent to applying conj(W) and avoids
          // the post-step RSCAL on Y2, which would also conjugate Y1's
          // contribution — Y1 already carries the correct TR convention from
          // upstream (copy_Gk_2c on the CPU).
          cublasOperation_t OP_Uq_Left  = q_conj_after_uq ? CUBLAS_OP_C : CUBLAS_OP_N;
          cublasOperation_t OP_Uq_Right = q_conj_after_uq ? CUBLAS_OP_N : CUBLAS_OP_C;
          // 2a: T1 = OP_Uq_Left(U_q^T) * Y1
          if (GEMM_STRIDED_BATCHED(*handle_, OP_Uq_Left, CUBLAS_OP_T, naux_, nao2_, naux_, &one, U_q, naux_, 0,
                                   Y1t_Qin, nao2_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction_2C() [2a].");
          }
          // 2b: T2(Q,in) = P_ibz(Q,Q') * T1(Q',in) → store in Y1
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, naux_, nao2_, naux_, &one, Pqk_tQP + t * naux2_, naux_,
                                   naux2_, Y2t_inP, naux_, nauxnao2_, &zero, Y1t_Qin, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction_2C() [2b].");
          }
          // 2c: Y2 = OP_Uq_Right(U_q^T) * T2
          if (GEMM_STRIDED_BATCHED(*handle_, OP_Uq_Right, CUBLAS_OP_N, naux_, nao2_, naux_, &one, U_q, naux_, 0,
                                   Y1t_Qin, naux_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction_2C() [2c].");
          }
        } else {
          // No q-space transform: Y2(Q,in) = P(Q,Q') * Y1^T(Q',in)
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_T, naux_, nao2_, naux_, &one, Pqk_tQP + t * naux2_, naux_,
                                   naux2_, Y1t_Qin, nao2_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction_2C().");
          }
        }
        // Sigma_ij = -Y2_inP * V_nPj
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nao_, nauxnao_, &m1, V_nPj_, nao_, 0, Y2t_inP,
                                 nauxnao_, nauxnao2_, &zero, sigmak_stij_ + st * nao2_, nao_, nao2_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction_2C().");
        }
      }
    }
    write_sigma(true);
    cudaEventRecord(all_done_event_);
  }

  template <typename prec>
  void gw_qkpt<prec>::write_sigma(bool low_memory_mode) {
    // write results. Make sure we have exclusive write access to sigma, then add array sigmak_tij to sigma_ktij
    scalar_t one = 1.;
    if (!low_memory_mode) {
      acquire_lock<<<1, 1, 0, stream_>>>(sigma_k_locks_ + k_);
      if (RAXPY(*handle_, 2 * ns_ * ntnao2_, &one, (scalar_t*)sigmak_stij_, 1, (scalar_t*)(sigma_ktij_ + k_ * ns_ * ntnao2_),
                1) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("RAXPY fails on gw_qkpt.write_sigma().");
      }
      release_lock<<<1, 1, 0, stream_>>>(sigma_k_locks_ + k_);
    } else {
      // Copy sigmak_stij_ back to CPU
      cudaMemcpyAsync(Sigmak_stij_buffer_, sigmak_stij_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyDeviceToHost, stream_);
      cleanup_req_ = true;
    }
  }

  template <typename prec>
  void gw_qkpt<prec>::cleanup(bool low_memory_mode, tensor<std::complex<prec>, 4>& Sigmak_stij_host, ztensor<5>& Sigma_tskij_host, bool x2c) {
    if (require_cleanup()) {
      cudaStreamSynchronize(stream_);
      std::memcpy(Sigmak_stij_host.data(), Sigmak_stij_buffer_, ns_ * ntnao2_ * sizeof(cxx_complex));
      if (!x2c) {
        copy_Sigma(Sigma_tskij_host, Sigmak_stij_host);
      } else {
        copy_Sigma_2c(Sigma_tskij_host, Sigmak_stij_host);
      }
      cleanup_req_ = false;
    }
  }

  template <typename prec>
  void gw_qkpt<prec>::copy_Sigma(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_stij) {
    for (size_t t = 0; t < nt_; ++t) {
      for (size_t s = 0; s < ns_; ++s) {
        matrix(Sigma_tskij_host(t, s, k_red_id_)) += matrix(Sigmak_stij(s, t)).template cast<typename std::complex<double>>();
      }
    }
  }

  template <typename prec>
  void gw_qkpt<prec>::copy_Sigma_2c(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_stij) {
    size_t    nao = Sigmak_stij.shape()[3];
    size_t    nso = 2 * nao;
    for (size_t ss = 0; ss < 3; ++ss) {
      size_t a       = (ss % 2 == 0) ? 0 : 1;
      size_t b       = ((ss + 1) / 2 != 1) ? 0 : 1;
      size_t i_shift = a * nao;
      size_t j_shift = b * nao;
      for (size_t t = 0; t < nt_; ++t) {
        matrix(Sigma_tskij_host(t, 0, k_red_id_)).block(i_shift, j_shift, nao, nao) +=
            matrix(Sigmak_stij(ss, t)).template cast<typename std::complex<double>>();
        if (ss == 2) {
          matrix(Sigma_tskij_host(t, 0, k_red_id_)).block(j_shift, i_shift, nao, nao) +=
              matrix(Sigmak_stij(ss, t)).conjugate().transpose().template cast<typename std::complex<double>>();
        }
      }
    }
  }

  template <typename prec>
  bool gw_qkpt<prec>::is_busy() {
    cudaError_t stream_status = cudaStreamQuery(stream_);
    if (stream_status == cudaSuccess)
      return false;  // not busy;
    else if (stream_status == cudaErrorNotReady)
      return true;  // busy~
    else
      throw std::runtime_error("problem with stream query");
  }

  template class gw_qkpt<float>;
  template class gw_qkpt<double>;

}  // namespace green::gpu
