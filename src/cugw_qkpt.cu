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
  void gw_qkpt<prec>::compute_second_tau_contraction(cuda_complex* Pqk_tQP, const cuda_complex* U, const cuda_complex* U_conj) {
    // CPU-math chain (CUDA / col-major-frame indices on Y):
    //     Y2_out = U† · P · U · Y1ᵀ      then  Σ = -V_nPj · Y2_out
    //
    // CONVENTION KEY:  Y1 is in CUDA / col-major frame (cuBLAS produced it at
    // GEMM 1).  U, U_conj, and Pq are row-major-stored — to use them as math
    // operands in cuBLAS we apply OP_T ("r2c conversion") which lands the math
    // identity of the stored matrix.  The single exception is the OUTERMOST
    // factor U† at GEMM 2c: we want the ADJOINT (X† math) of the stored matrix,
    // which is unreachable via any single OP on row-major bytes alone.  So the
    // caller hands us a buffer whose col-major view is already X† math (for
    // non-TR that's the row-major U_q_conj buffer, col-major view of stored
    // X* = X†; for TR it's the row-major U_q buffer, col-major view of stored
    // X = Xᵀ).  OP_T reads the col-major view, giving the adjoint math we need.
    //
    // GEMM 2a:  Y2_step1 = OP_N(U)      · OP_T(Y1)        →  math:  U   · Y1ᵀ
    // GEMM 2b:  Y1_step2 = OP_N(Pq)     · OP_N(Y2_step1)  →  math:  P   · Y2_step1
    // GEMM 2c:  Y2_step3 = OP_T(U_conj) · OP_N(Y1_step2)  →  math:  U†  · Y1_step2
    //
    // Per-branch operand assignment (decided by the caller):
    //   non-TR:  U      = stored U_q       (OP_N → U_q math)
    //            U_conj = stored U_q_conj  (OP_T → U_q† math)
    //            Pq     = Pqk_tQP_         (OP_N → P math)
    //   TR:      U      = stored U_q_conj  (OP_N → U_q_conj math)
    //            U_conj = stored U_q       (OP_T → U_q_conj† math)
    //            Pq     = Pqk_tQP_conj_    (OP_N → conj(P) math)
    //
    // The kernel itself does not branch on q_need_conj — the caller has baked
    // the TR / non-TR choice into the operand selection.
    //
    // X2C: ns_=4 spinor blocks (aa, bb, ab, ba) are contracted independently;
    // the Hermitian relation Σ_ba = (Σ_ab)† does not hold at a single (k, τ)
    // snapshot under SOC, so all four blocks are computed.  Scalar: ns_∈{1,2}.
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
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction() [Y1].");
        }
        if (U != nullptr) {
          // GEMM 2a: Y2_step1 = U · Y1ᵀ
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_T, naux_, nao2_, naux_, &one, U, naux_, 0,
                                   Y1t_Qin, nao2_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction() [2a].");
          }
          // GEMM 2b: Y1_step2 = P · Y2_step1
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, naux_, nao2_, naux_, &one, Pqk_tQP + t * naux2_, naux_,
                                   naux2_, Y2t_inP, naux_, nauxnao2_, &zero, Y1t_Qin, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction() [2b].");
          }
          // GEMM 2c: Y2_step3 = U† · Y1_step2
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_T, CUBLAS_OP_N, naux_, nao2_, naux_, &one, U_conj, naux_, 0,
                                   Y1t_Qin, naux_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction() [2c].");
          }
        } else {
          // No q-space transform: Y2(Q,in) = P(Q,Q') · Y1ᵀ(Q',in).
          // Unreachable in X2C (caller always supplies U / U_conj) but retained for scalar.
          if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_T, naux_, nao2_, naux_, &one, Pqk_tQP + t * naux2_, naux_,
                                   naux2_, Y1t_Qin, nao2_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                   nt_mult) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction() [no-U].");
          }
        }
        // GEMM 3: Sigma_ij = -Y2_inP · V_nPj
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nao_, nauxnao_, &m1, V_nPj_, nao_, 0, Y2t_inP,
                                 nauxnao_, nauxnao2_, &zero, sigmak_stij_ + st * nao2_, nao_, nao2_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction() [Sigma].");
        }
      }
    }
    write_sigma(_low_memory_requirement);
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
    // Spinor-block layout produced by compute_second_tau_contraction (X2C path), matching
    // copy_Gk_2c (minus_t = false) ordering for the input G:
    //   ss=0 → aa block (0,    0)
    //   ss=1 → bb block (nao,  nao)
    //   ss=2 → ab block (0,    nao)
    //   ss=3 → ba block (nao,  0)
    // CPU eval_selfenergy writes all four independently; the Hermitian relation
    // Σ_ba = (Σ_ab)† is NOT a structural property of GW Σ under SOC at a single
    // (k, τ), so we accumulate each block directly from its own kernel output.
    const size_t nao = Sigmak_stij.shape()[3];
    for (size_t ss = 0; ss < 4; ++ss) {
      const size_t a       = (ss % 2 == 0) ? 0 : 1;             // 0,1,0,1
      const size_t b       = ((ss + 1) / 2 != 1) ? 0 : 1;       // 0,1,1,0
      const size_t i_shift = a * nao;
      const size_t j_shift = b * nao;
      for (size_t t = 0; t < nt_; ++t) {
        matrix(Sigma_tskij_host(t, 0, k_red_id_)).block(i_shift, j_shift, nao, nao) +=
            matrix(Sigmak_stij(ss, t)).template cast<typename std::complex<double>>();
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
