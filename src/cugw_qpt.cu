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

#include <green/gpu/cugw_qpt.h>


namespace green::gpu {
  template<typename prec>
  gw_qpt<prec>::gw_qpt(int nao, int naux, int ns, int nt, int nw_b,
                       const std::complex<double> *T_tw_fb_host, const std::complex<double> *T_wt_bf_host,
                       LinearSolverType cuda_lin_solver):
      nao_(nao),
      nao2_(nao * nao),
      nao3_(nao2_ * nao),
      naux_(naux),
      naux2_(naux * naux),
      nauxnao_(naux * nao),
      nauxnao2_(naux * nao * nao),
      ns_(ns),
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
  void gw_qpt<prec>::dump_Pq0_diagonals_to_text(const std::string& file_path, size_t q_ibz) {
    // Ensure all queued operations on Pqk0_tQP_ are complete before host readback.
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      throw std::runtime_error("Failed to synchronize stream before P0 dump.");
    }

    std::vector<cxx_complex> host_p0(ntnaux2_);
    if (cudaMemcpy(host_p0.data(), Pqk0_tQP_, ntnaux2_ * sizeof(cuda_complex), cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("Failed to copy P0 from device for debug dump.");
    }

    std::ofstream out(file_path);
    if (!out.is_open()) {
      throw std::runtime_error("Failed to open P0 dump file: " + file_path);
    }

    out << "# q_ibz=" << q_ibz << " nt=" << nt_ << " naux=" << naux_ << "\n";
    out << "# columns: q_ibz tau diag_index real imag\n";
    out << std::setprecision(17);
    for (int t = 0; t < nt_; ++t) {
      const size_t t_offset = static_cast<size_t>(t) * static_cast<size_t>(naux2_);
      for (int p = 0; p < naux_; ++p) {
        const size_t diag_idx = t_offset + static_cast<size_t>(p) * static_cast<size_t>(naux_) + static_cast<size_t>(p);
        const cxx_complex val = host_p0[diag_idx];
        out << q_ibz << ' ' << t << ' ' << p << ' ' << val.real() << ' ' << val.imag() << "\n";
      }
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

  template class gw_qpt<float>;
  template class gw_qpt<double>;

}  // namespace green::gpu
