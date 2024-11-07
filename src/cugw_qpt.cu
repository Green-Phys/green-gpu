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
namespace green::gpu {
  template<typename prec>
  gw_qpt<prec>::gw_qpt(int nao, int naux, int nt, int nw_b,
                       const std::complex<double> *T_tw_fb_host, const std::complex<double> *T_wt_bf_host, 
                       LinearSolverType cuda_lin_solver):
      nao_(nao),
      nao2_(nao * nao),
      nao3_(nao2_ * nao),
      naux_(naux),
      naux2_(naux * naux),
      nauxnao_(naux * nao),
      nauxnao2_(naux * nao * nao),
      nt_(nt),
      nw_b_(nw_b),
      ntnaux_(nt * naux),
      ntnaux2_(nt * naux * naux),
      nwnaux_(nw_b * naux),
      nwnaux2_(nw_b * naux * naux),
      ntnao_(nt * nao),
      ntnao2_(nt * nao2_),
      handle_(nullptr),
      solver_handle_(nullptr),
      streams_potrs_(nw_b),
      potrs_ready_event_(nw_b), one_minus_P_ready_event_(nw_b), cuda_lin_solver_(cuda_lin_solver) {
    allocate_IR_transformation_matrices(&T_tw_, &T_wt_, T_tw_fb_host, T_wt_bf_host, nt, nw_b);
  }

  template<typename prec>
  void gw_qpt<prec>::init(cublasHandle_t *handle, cusolverDnHandle_t *solver_handle){
    handle_ = handle;
    solver_handle_ = solver_handle;
    if (cudaStreamCreate(&stream_) != cudaSuccess) throw std::runtime_error("main stream creation failed");
    for (int w = 0; w < nw_b_; ++w) {
      if (cudaStreamCreate(&(streams_potrs_[w])) != cudaSuccess) throw std::runtime_error("potrs stream creation failed");
      cudaEventCreateWithFlags(&one_minus_P_ready_event_[w], cudaEventDisableTiming);
      cudaEventCreateWithFlags(&potrs_ready_event_[w], cudaEventDisableTiming);
    }
    cudaEventCreateWithFlags(&polarization_ready_event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&bare_polarization_ready_event_, cudaEventDisableTiming);
    if(cuda_lin_solver_ == LinearSolverType::Cholesky) {
      cudaEventCreateWithFlags(&Cholesky_decomposition_ready_event_, cudaEventDisableTiming);
    }
    else {
      cudaEventCreateWithFlags(&LU_decomposition_ready_event_, cudaEventDisableTiming);
      cudaEventCreateWithFlags(&getrs_ready_event_, cudaEventDisableTiming);
    }


    // We assume nt_ >= nw_b_ and reuse the same memory for P0 and P in imaginary time and frequency domain
    // This is a safe assumption in IR and Chebyshev representation
    if (nt_ < nw_b_) throw std::runtime_error("Nt is not greater than or equal to Nw_b. Please double check your input!");
    // bare polarization bubble for fixed q
    if (cudaMalloc(&Pqk0_tQP_, ntnaux2_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating Pq0");
    // allocate ntnaux2 elements since Pqk_tQP_conj_ will point to this later
    if (cudaMalloc(&Pqk0_wQP_, ntnaux2_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating Pq0");

    Pqk_tQP_         = Pqk0_tQP_;
    Pqk_wQP_         = Pqk0_wQP_;
    Pqk_tQP_conj_    = Pqk0_wQP_;
    one_minus_P_wPQ_ = Pqk0_tQP_;

    if (cudaMalloc(&one_minus_P_w_ptrs_, nw_b_ * sizeof(cuda_complex*)) != cudaSuccess)
      throw std::runtime_error("failure allocating one_minus_P_w_ptrs_");
    if(cuda_lin_solver_ == LinearSolverType::LU) {
      if (cudaMalloc(&P0_w_ptrs_, nw_b_ * sizeof(cuda_complex*)) != cudaSuccess)
        throw std::runtime_error("failure allocating P0_w_ptrs_");
    }
    if (cudaMalloc(&d_info_, nw_b_ * sizeof(int)) != cudaSuccess)
      throw std::runtime_error("failure allocating info int on device");
    if(cuda_lin_solver_ == LinearSolverType::LU) {
      if (cudaMalloc(&Pivot_, naux_ * nw_b_ * sizeof(int)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc failed to allocate Pivot");
    }

    if (cusolverDnSetStream(*solver_handle_, stream_) != CUSOLVER_STATUS_SUCCESS)
      throw std::runtime_error("cusolver set stream problem");

    // locks so that different threads don't write the results over each other
    cudaMalloc(&Pqk0_tQP_lock_, sizeof(int));
    cudaMemset(Pqk0_tQP_lock_, 0, sizeof(int));
    // For batched potrf/LU
    set_batch_pointer<<<1, 1, 0, stream_>>>(one_minus_P_w_ptrs_, one_minus_P_wPQ_, naux2_, nw_b_);
    // for LU
    if(cuda_lin_solver_ == LinearSolverType::LU) {
      set_batch_pointer<<<1, 1, 0, stream_>>>(P0_w_ptrs_, Pqk0_wQP_, naux2_, nw_b_);
    }

  }

  template <typename prec>
  gw_qpt<prec>::~gw_qpt() {
    cudaStreamDestroy(stream_);
    for (int w = 0; w < nw_b_; ++w) {
      cudaStreamDestroy(streams_potrs_[w]);
      cudaEventDestroy(potrs_ready_event_[w]);
      cudaEventDestroy(one_minus_P_ready_event_[w]);
    }
    cudaEventDestroy(polarization_ready_event_);
    cudaEventDestroy(bare_polarization_ready_event_);
    if(cuda_lin_solver_ == LinearSolverType::LU) {
      cudaEventDestroy(LU_decomposition_ready_event_);
    }
    else {
      cudaEventDestroy(Cholesky_decomposition_ready_event_);
    }

    cudaFree(Pqk0_tQP_);
    cudaFree(Pqk0_tQP_lock_);
    cudaFree(Pqk0_wQP_);

    cudaFree(one_minus_P_w_ptrs_);
    cudaFree(d_info_);
    if(cuda_lin_solver_ == LinearSolverType::LU) {
      cudaFree(P0_w_ptrs_);
      cudaFree(Pivot_);
    }
    cudaFree(T_tw_);
    cudaFree(T_wt_);
  }

  template <typename prec>
  typename gw_qpt<prec>::cuda_complex* gw_qpt<prec>::Pqk0_tQP(cudaEvent_t all_done_event) {
    if (cudaStreamWaitEvent(stream_, all_done_event, 0 /*cudaEventWaitDefault*/))
      throw std::runtime_error("could not wait for data");
    return Pqk0_tQP_;
  }

  template <typename prec>
  typename gw_qpt<prec>::cuda_complex* gw_qpt<prec>::Pqk_tQP(cudaEvent_t all_done_event, cudaStream_t calc_stream,
                                                             int need_minus_q) {
    // make sure the other stream waits until our data is ready (i.e. the equation system solved)
    if (cudaStreamWaitEvent(calc_stream, polarization_ready_event_, 0 /*cudaEventWaitDefault*/))
      throw std::runtime_error("could not wait for data");
    // make sure this stream waits until the other calculation is done
    if (cudaStreamWaitEvent(stream_, all_done_event, 0 /*cudaEventWaitDefault*/))
      throw std::runtime_error("could not wait for data");
    return (!need_minus_q) ? Pqk_tQP_ : Pqk_tQP_conj_;
  }

  template <typename prec>
  int* gw_qpt<prec>::Pqk0_tQP_lock() {
    return Pqk0_tQP_lock_;
  }

  template <typename prec>
  void gw_qpt<prec>::reset_Pqk0() {
    cudaMemsetAsync(Pqk0_tQP_, 0, sizeof(cuda_complex) * nt_ * naux2_, stream_);
  }

  template <typename prec>
  void gw_qpt<prec>::scale_Pq0_tQP(scalar_t scale_factor) {
    size_t size    = nt_ / 2 * naux_ * naux_;
    int    one_int = 1;
    cublasSetStream(*handle_, stream_);
    if (RSCAL(*handle_, 2 * size, &scale_factor, (scalar_t*)Pqk0_tQP_, one_int) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("RSCAL fails on gw_qpt.scale_Pq0_tQP().");
    }
    for (int t = 0; t < nt_ / 2; ++t) {
      cudaMemcpyAsync(Pqk0_tQP_ + (nt_ - t - 1) * naux2_, Pqk0_tQP_ + t * naux2_, naux2_ * sizeof(cuda_complex),
                      cudaMemcpyDeviceToDevice, stream_);
    }
  }

  template <typename prec>
  void gw_qpt<prec>::compute_Pq_chol() {
    int threads_per_block = 512;
    int blocks_for_id     = naux_ / threads_per_block + 1;
    if (_verbose > 2) std::cout << "Running Cholesky solver for (I - P0)P = P0" << std::endl;
    for (int w = 0; w < nw_b_; ++w) {
      cudaStreamWaitEvent(streams_potrs_[w], bare_polarization_ready_event_, 0);
      hermitian_symmetrize<<<blocks_for_id, threads_per_block, 0, streams_potrs_[w]>>>(Pqk0_wQP_ + w * naux2_, naux_);
      set_up_one_minus_P<<<blocks_for_id, threads_per_block, 0, streams_potrs_[w]>>>(one_minus_P_wPQ_ + w * naux2_,
                                                                                     Pqk0_wQP_ + w * naux2_, naux_);
      cudaEventRecord(one_minus_P_ready_event_[w], streams_potrs_[w]);
    }
    for (int w = 0; w < nw_b_; ++w) {
      cudaStreamWaitEvent(stream_, one_minus_P_ready_event_[w], 0 /*cudaEventWaitDefault*/);
    }

    if (cusolverDnSetStream(*solver_handle_, stream_) != CUSOLVER_STATUS_SUCCESS)
      throw std::runtime_error("cusolver set stream problem");
    if (POTRF_BATCHED(*solver_handle_, CUBLAS_FILL_MODE_LOWER, naux_, one_minus_P_w_ptrs_, naux_, d_info_, nw_b_) !=
        CUSOLVER_STATUS_SUCCESS) {
      throw std::runtime_error("Batched Cholesky decomposition fails");
    }
    validate_info<<<1, 1, 0, stream_>>>(d_info_, nw_b_);
    cudaEventRecord(Cholesky_decomposition_ready_event_, stream_);

    for (int w = 0; w < nw_b_; ++w) {
      // Hold streams_potrs_[w] until Cholesky decomposition (potrf) is finished
      if (cudaStreamWaitEvent(streams_potrs_[w], Cholesky_decomposition_ready_event_, 0 /*cudaEventWaitDefault*/))
        throw std::runtime_error("Could not wait for Cholesky decomposition");
      if (cusolverDnSetStream(*solver_handle_, streams_potrs_[w]) != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cusolver set stream problem");
      if (POTRS(*solver_handle_, CUBLAS_FILL_MODE_LOWER, naux_, naux_, one_minus_P_wPQ_ + w * naux2_, naux_,
                Pqk0_wQP_ + w * naux2_, naux_, d_info_ + w)) {
        throw std::runtime_error("Cholesky solver fails");
      }
      validate_info<<<1, 1, 0, streams_potrs_[w]>>>(d_info_ + w);
      cudaEventRecord(potrs_ready_event_[w], streams_potrs_[w]);
    }
    for (int w = 0; w < nw_b_; ++w) {
      cudaStreamWaitEvent(stream_, potrs_ready_event_[w], 0 /*cudaEventWaitDefault*/);
    }
  }

  template <typename prec>
  void gw_qpt<prec>::compute_Pq_lu() {
    int threads_per_block = 512;
    int blocks_for_id     = naux_ / threads_per_block + 1;
    if (_verbose > 2) std::cout << "Running pivoted LU solver for (I - P0)P = P0" << std::endl;
    for (int w = 0; w < nw_b_; ++w) {
      cudaStreamWaitEvent(streams_potrs_[w], bare_polarization_ready_event_, 0);
      hermitian_symmetrize<<<blocks_for_id, threads_per_block, 0, streams_potrs_[w]>>>(Pqk0_wQP_ + w * naux2_, naux_);
      set_up_one_minus_P<<<blocks_for_id, threads_per_block, 0, streams_potrs_[w]>>>(one_minus_P_wPQ_ + w * naux2_,
                                                                                     Pqk0_wQP_ + w * naux2_, naux_);
      cudaEventRecord(one_minus_P_ready_event_[w], streams_potrs_[w]);
    }
    for (int w = 0; w < nw_b_; ++w) {
      cudaStreamWaitEvent(stream_, one_minus_P_ready_event_[w], 0 /*cudaEventWaitDefault*/);
    }

    if (cublasSetStream(*handle_, stream_) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublas set stream problem");

    // get LU decomposition
    if(GETRF_BATCHED(*handle_, naux_, one_minus_P_w_ptrs_, 
              naux_, Pivot_, d_info_, nw_b_) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("CUDA GETRF failed!");
    }
    validate_info<<<1, 1, 0, stream_>>>(d_info_, nw_b_);
    cudaEventRecord(LU_decomposition_ready_event_, stream_);

    if (cudaStreamWaitEvent(stream_, LU_decomposition_ready_event_, 0 /*cudaEventWaitDefault*/))
      throw std::runtime_error("Could not wait for LU decomposition");

    if (cublasSetStream(*handle_, stream_) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cublas set stream problem");

    int host_info; // getrs_batched API requres int on host

    // Apply LU to solve linear system
    if(GETRS_BATCHED(*handle_, CUBLAS_OP_N, naux_, naux_, one_minus_P_w_ptrs_, 
             naux_, Pivot_, P0_w_ptrs_, naux_, &host_info, nw_b_) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("CUDA GETRS failed!");
    }

    if(host_info != 0) 
      throw std::runtime_error("cublas GETRS info = " + std::to_string(host_info));

    cudaEventRecord(getrs_ready_event_, stream_);
    if (cudaStreamWaitEvent(stream_, getrs_ready_event_, 0 /*cudaEventWaitDefault*/))
      throw std::runtime_error("Could not wait for GETRS");

  }

  template <typename prec>
  void gw_qpt<prec>::wait_for_kpts() {
    if (cudaStreamSynchronize(stream_) != cudaSuccess) throw std::runtime_error("could not wait for other streams");
  }

  template <typename prec>
  void gw_qpt<prec>::transform_tw() {
    cublasSetStream(*handle_, stream_);
    cuda_complex one  = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    if (GEMM(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, naux2_, nw_b_, nt_ - 2, &one, Pqk0_tQP_ + naux2_, naux2_, T_wt_, (nt_ - 2),
             &zero, Pqk0_wQP_, naux2_) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("GEMM fails at gw_qpt.transform_tw().");
    }
    cudaEventRecord(bare_polarization_ready_event_, stream_);  // this will allow compute_Pq to start
  }

  template <typename prec>
  void gw_qpt<prec>::transform_wt() {
    cublasSetStream(*handle_, stream_);
    cuda_complex one  = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    if (GEMM(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, naux2_, nt_, nw_b_, &one, Pqk_wQP_, naux2_, T_tw_, nw_b_, &zero, Pqk_tQP_,
             naux2_) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("GEMM fails at gw_qpt.transform_wt().");
    }
    // Compute Pqk_tQP_conj_
    int      two   = 2;
    scalar_t alpha = -1;
    cudaMemcpyAsync(Pqk_tQP_conj_, Pqk_tQP_, ntnaux2_ * sizeof(cuda_complex), cudaMemcpyDeviceToDevice, stream_);
    if (RSCAL(*handle_, ntnaux2_, &alpha, (scalar_t*)Pqk_tQP_conj_ + 1, two) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("RSCAL fails at gw_qpt.transform_wt().");
    }
    cudaEventRecord(polarization_ready_event_, stream_);  // this will allow subsequent loops over k to proceed
  }

  template <typename prec>
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
        throw std::runtime_error("failure allocating Gk_tsij on host");
      if (cudaMallocHost(&Gk_smtij_buffer_, ns_ * ntnao2_ * sizeof(cxx_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating Gk_tsij on host");
      // ! GH:  I think this will interfere with our cudaMemcpyAsync. Should we simply allocate a different array for Sigmak_stij_buffer_?
      // !      The more I think, this here is the real reason why we had to use cudaMemcpy and not the asynchronous version.
      // ! <previously> Sigmak_stij_buffer_ = Gk_smtij_buffer_;
      if (cudaMallocHost(&Sigmak_stij_buffer_, ns_ * ntnao2_ * sizeof(cxx_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating Gk_tsij on host");
    }

    if (cudaMalloc(&Pqk0_tQP_local_, nt_batch_ * naux2_ * sizeof(cuda_complex)) != cudaSuccess)
      throw std::runtime_error("failure allocating Pq0");

    cudaEventCreateWithFlags(&data_ready_event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&all_done_event_, cudaEventDisableTiming);

    // set memory alias
    V_Qim_ = V_Qpm_;
    V_nPj_ = V_pmQ_;
  }

  template <typename prec>
  gw_qkpt<prec>::~gw_qkpt() {
    cudaStreamDestroy(stream_);
    cudaEventDestroy(data_ready_event_);
    cudaEventDestroy(all_done_event_);

    cudaFree(V_Qpm_);
    cudaFree(V_pmQ_);
    cudaFree(X1t_tmQ_);
    cudaFree(X2t_Ptm_);
    cudaFree(Pqk0_tQP_local_);
    cudaFree(g_stij_);
    cudaFree(g_smtij_);
    cudaFree(sigmak_stij_);

    cudaFreeHost(V_Qpm_buffer_);
    if (_low_memory_requirement) {
      cudaFreeHost(Gk1_stij_buffer_);
      cudaFreeHost(Gk_smtij_buffer_);
    }
    if (cleanup_req_ == true) {
      throw std::runtime_error("cleanup of self-energy was not done correctly.");
    }
  }

  template <typename prec>
  void gw_qkpt<prec>::set_up_qkpt_first(cxx_complex* Gk1_stij_host, cxx_complex* Gk_smtij_host, cxx_complex* V_Qpm_host, int k,
                                        bool need_minus_k, int k1, bool need_minus_k1) {
    cudaStreamSynchronize(stream_);  // this should not trigger. But just in case: wait until we're done with all previous calcs
    k_  = k;
    k1_ = k1;
    std::memcpy(V_Qpm_buffer_, V_Qpm_host, nauxnao2_ * sizeof(cxx_complex));
    cudaMemcpyAsync(V_Qpm_, V_Qpm_buffer_, nauxnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);

    // explicit conjugate transpose of V
    int      two   = 2;
    scalar_t alpha = -1;
    cublasSetStream(*handle_, stream_);
    cuda_complex one  = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    // C = alpha*op(A) + beta*op(C)
    if (GEAM(*handle_, CUBLAS_OP_T, CUBLAS_OP_N, naux_, nao2_, &one, V_Qpm_, nao2_, &zero, V_pmQ_, naux_, V_pmQ_, naux_) !=
        CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("GEAM fails on gw_qkpt.set_up_qkpt_first().");
    }
    // there has to be a better way to compute a complex conjugate!!
    if (RSCAL(*handle_, nauxnao2_, &alpha, (scalar_t*)V_Qpm_ + 1, two) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("RSCAL fails on gw_qkpt.set_up_qkpt_first().");
    }

    if (_low_memory_requirement) {
      // How much extra overhead if we do HostToDevice copy instead?
      std::memcpy(Gk1_stij_buffer_, Gk1_stij_host, ns_ * ntnao2_ * sizeof(cxx_complex));
      std::memcpy(Gk_smtij_buffer_, Gk_smtij_host, ns_ * ntnao2_ * sizeof(cxx_complex));
      cudaMemcpyAsync(g_stij_, Gk1_stij_buffer_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);
      cudaMemcpyAsync(g_smtij_, Gk_smtij_buffer_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);
    } else {
      // Prepare proper g_tij and g_mtij. Ugly but leave it for now...
      cudaMemsetAsync(g_stij_, 0, sizeof(cuda_complex) * ns_ * ntnaux2_, stream_);
      cudaMemsetAsync(g_smtij_, 0, sizeof(cuda_complex) * ns_ * ntnaux2_, stream_);
      cudaMemcpyAsync(g_stij_, g_ktij_ + k1_ * ns_ * ntnao2_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyDeviceToDevice,
                      stream_);
      cudaMemcpyAsync(g_smtij_, g_kmtij_ + k_ * ns_ * ntnao2_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyDeviceToDevice,
                      stream_);
    }
    cublasSetStream(*handle_, stream_);
    if (need_minus_k1) {
      if (RSCAL(*handle_, ns_ * ntnao2_, &alpha, (scalar_t*)g_stij_ + 1, two) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("RSCAL fails on gw_qkpt.set_up_qkpt_first().");
      }
    }
    if (need_minus_k) {
      if (RSCAL(*handle_, ns_ * ntnao2_, &alpha, (scalar_t*)g_smtij_ + 1, two) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("RSCAL fails on gw_qkpt.set_up_qkpt_first().");
      }
    }

    // let other streams know that all data is ready for calculation
    cudaEventRecord(data_ready_event_, stream_);
  }

  template <typename prec>
  void gw_qkpt<prec>::set_up_qkpt_second(cxx_complex* Gk1_stij_host, cxx_complex* V_Qim_host, int k, int k1, bool need_minus_k1) {
    cudaStreamSynchronize(stream_);  // this should not trigger. But just in case: wait until we're done with all previous calcs
    k_  = k;
    k1_ = k1;
    std::memcpy(V_Qpm_buffer_, V_Qim_host, nauxnao2_ * sizeof(cxx_complex));
    cudaMemcpyAsync(V_Qim_, V_Qpm_buffer_, nauxnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);

    // explicit conjugate transpose of V
    int      two   = 2;
    scalar_t alpha = -1;
    cublasSetStream(*handle_, stream_);
    cuda_complex one  = cu_type_map<cxx_complex>::cast(1., 0.);
    cuda_complex zero = cu_type_map<cxx_complex>::cast(0., 0.);
    // C = alpha*op(A) + beta*op(C)
    if (GEAM(*handle_, CUBLAS_OP_C, CUBLAS_OP_N, nauxnao_, nao_, &one, V_Qim_, nao_, &zero, V_nPj_, nauxnao_, V_nPj_, nauxnao_) !=
        CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("GEAM fails on gw_qkpt.set_up_qkpt_second().");
    }
    if (_low_memory_requirement) {
      // How much extra overhead if we do HostToDevice copy instead?
      std::memcpy(Gk1_stij_buffer_, Gk1_stij_host, ns_ * ntnao2_ * sizeof(cxx_complex));
      cudaMemcpyAsync(g_stij_, Gk1_stij_buffer_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyHostToDevice, stream_);
    } else {
      // Prepare G^{k1}(t)
      cudaMemcpyAsync(g_stij_, g_ktij_ + k1_ * ns_ * ntnao2_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyDeviceToDevice,
                      stream_);
    }
    if (need_minus_k1) {
      if (RSCAL(*handle_, ns_ * ntnao2_, &alpha, (scalar_t*)g_stij_ + 1, two) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("RSCAL fails on gw_qkpt.set_up_qkpt_second().");
      }
    }

    // let other streams know that all data is ready for calculation*/
    cudaEventRecord(data_ready_event_, stream_);
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
  void gw_qkpt<prec>::compute_second_tau_contraction(cuda_complex* Pqk_tQP) {
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
        // Y1_Qin = V_Qim * G1_mn; G1_mn = G^{k1}(t)_mn
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nauxnao_, nao_, &one, g_stij_ + st * nao2_, nao_,
                                 nao2_, V_Qim_, nao_, 0, &zero, Y1t_Qin, nao_, nauxnao2_, nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
        }
        // Y2_inP = Y1_Qin * Pq_QP
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_T, naux_, nao2_, naux_, &one, Pqk_tQP + t * naux2_, naux_,
                                 naux2_, Y1t_Qin, nao2_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
        }
        // Sigma_ij = Y2_inP V_nPj
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nao_, nauxnao_, &m1, V_nPj_, nao_, 0, Y2t_inP,
                                 nauxnao_, nauxnao2_, &zero, sigmak_stij_ + st * nao2_, nao_, nao2_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
        }
      }
    }
    write_sigma(_low_memory_requirement);
    // GH:  Should we record this event here? Because memcpyAsync is in progress, or does NVIDIA compiler know
    //      that it should consider all_done_event_ after copy is complete?
    cudaEventRecord(all_done_event_);
  }

  template <typename prec>
  void gw_qkpt<prec>::compute_second_tau_contraction_2C(cuda_complex* Pqk_tQP) {
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
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
        }
        // Y2_inP = Y1_Qin * Pq_QP
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_T, naux_, nao2_, naux_, &one, Pqk_tQP + t * naux2_, naux_,
                                 naux2_, Y1t_Qin, nao2_, nauxnao2_, &zero, Y2t_inP, naux_, nauxnao2_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
        }
        // Sigma_ij = Y2_inP V_nPj
        if (GEMM_STRIDED_BATCHED(*handle_, CUBLAS_OP_N, CUBLAS_OP_N, nao_, nao_, nauxnao_, &m1, V_nPj_, nao_, 0, Y2t_inP,
                                 nauxnao_, nauxnao2_, &zero, sigmak_stij_ + st * nao2_, nao_, nao2_,
                                 nt_mult) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("GEMM_STRIDED_BATCHED fails on gw_qkpt.compute_second_tau_contraction().");
        }
      }
    }
    write_sigma(true);
    cudaEventRecord(all_done_event_);
  }

  template <typename prec>
  void gw_qkpt<prec>::write_sigma(bool low_memory_mode) {
    // write results. Make sure we have exclusive write access to sigma, then add array sigmak_tij to sigma_ktij
    // TODO: In my understanding, the lock is only required for RAXPY part now, so we should move them inside the first if condition
    acquire_lock<<<1, 1, 0, stream_>>>(sigma_k_locks_ + k_);
    scalar_t one = 1.;
    if (!low_memory_mode) {
      if (RAXPY(*handle_, 2 * ns_ * ntnao2_, &one, (scalar_t*)sigmak_stij_, 1, (scalar_t*)(sigma_ktij_ + k_ * ns_ * ntnao2_),
                1) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("RAXPY fails on gw_qkpt.write_sigma().");
      }
    } else {
      // Copy sigmak_stij_ asynchronously back to CPU
      cudaMemcpyAsync(Sigmak_stij_buffer_, sigmak_stij_, ns_ * ntnao2_ * sizeof(cuda_complex), cudaMemcpyDeviceToHost, stream_);
      // cudaMemcpyAsync will require a cleanup at later stage. So, we update the cleanup_req_ status to true
      cleanup_req_ = true;
    }
    release_lock<<<1, 1, 0, stream_>>>(sigma_k_locks_ + k_);
  }

  template <typename prec>
  void gw_qkpt<prec>::cleanup(bool low_memory_mode, tensor<std::complex<prec>,4>& Sigmak_stij_host, St_type& Sigma_tskij_shared, bool x2c) {
    if (cleanup_req_) {
      std::memcpy(Sigmak_stij_host.data(), Sigmak_stij_buffer_, ns_ * ntnao2_ * sizeof(cxx_complex));
      if (!x2c) {
        copy_Sigma(Sigma_tskij_shared, Sigmak_stij_host);
      } else {
        copy_Sigma_2c(Sigma_tskij_shared, Sigmak_stij_host);
      }
      cleanup_req_ = false;
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

  template <typename prec>
  void gw_qkpt<prec>::copy_Sigma(St_type& Sigma_tskij_shared, tensor<std::complex<prec>, 4>& Sigmak_stij){
    ztensor<5> Sigma_tskij_host = Sigma_tskij_shared.object();
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, Sigma_tskij_shared.win());
    for (size_t t = 0; t < nt_; ++t) {
      for (size_t s = 0; s < ns_; ++s) {
        matrix(Sigma_tskij_host(t, s, k_red_id_)) += matrix(Sigmak_stij(s, t)).template cast<typename std::complex<double>>();
      }
    }
    MPI_Win_unlock(0, Sigma_tskij_shared.win());
  }

  template <typename prec>
  void gw_qkpt<prec>::copy_Sigma_2c(St_type& Sigma_tskij_shared, tensor<std::complex<prec>, 4>& Sigmak_4tij) {
    ztensor<5> Sigma_tskij_host = Sigma_tskij_shared.object();
    size_t    nao = Sigmak_4tij.shape()[3];
    size_t    nso = 2 * nao;
    MatrixXcf Sigma_ij(nso, nso);
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, Sigma_tskij_shared.win());
    for (size_t ss = 0; ss < 3; ++ss) {
      size_t a       = (ss % 2 == 0) ? 0 : 1;
      size_t b       = ((ss + 1) / 2 != 1) ? 0 : 1;
      size_t i_shift = a * nao;
      size_t j_shift = b * nao;
      for (size_t t = 0; t < nt_; ++t) {
        matrix(Sigma_tskij_host(t, 0, k_red_id_)).block(i_shift, j_shift, nao, nao) +=
            matrix(Sigmak_4tij(ss, t)).template cast<typename std::complex<double>>();
        if (ss == 2) {
          matrix(Sigma_tskij_host(t, 0, k_red_id_)).block(j_shift, i_shift, nao, nao) +=
              matrix(Sigmak_4tij(ss, t)).conjugate().transpose().template cast<typename std::complex<double>>();
        }
      }
    }
    MPI_Win_unlock(0, Sigma_tskij_shared.win());
  }

  template class gw_qpt<float>;
  template class gw_qpt<double>;
  template class gw_qkpt<float>;
  template class gw_qkpt<double>;

}  // namespace green::gpu
