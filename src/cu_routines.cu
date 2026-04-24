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

#include <green/gpu/cu_routines.h>


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

    // Allocate intermediate object
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

  // VkbatchQij_host = V_k(k2~k2+nk_batch, NQ, nao, nao)
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
    /* Exchange diagram: */
    for (size_t k_reduced_id = devices_rank; k_reduced_id < _ink; k_reduced_id += devices_size) {
      int k = irre_list[k_reduced_id];
      for (size_t k2 = 0; k2 < _nk; k2 += _nk_batch) {
        // Given k, k2, read V_k(k2~k2+nk_batch, Q, i, j) on CPU
        if (integral_type == as_a_whole) {
          r1(k, k2, Vk1k2_Qij, V_kbatchQij);
        } else {
          r2(k, k2, V_kbatchQij);
        }
        // Transfer V_k(k2, Q, i, j) to GPU
        set_up_exchange(V_kbatchQij.data(), k_reduced_id, k2);
        add_exchange_to_fock();
      }
    }
    cudaDeviceSynchronize();
    if (!_X2C) {
      // scalar HF
      copy_Fock_from_device_to_host(_F_skij, new_Fock.data(), _ink, _nao, _ns);
    } else {
      // 2c HF
      copy_2c_Fock_from_device_to_host(_F_skij, new_Fock.data(), _ink, _nao);
    }
    cudaDeviceSynchronize();
  }

  template <typename prec>
  cugw_utils<prec>::cugw_utils(int _nts, int _nt_batch, int _nw_b, int _ns, int _nk, int _ink, int _nq, int _inq, int _nqkpt, int _NQ, int _nao,
                               const cu_symmetry_data& sym_data, ztensor_view<5>& G_tskij_host,
                               bool low_device_memory, const MatrixXcd& Ttn_FB,
                               const MatrixXcd& Tnt_BF, LinearSolverType cuda_lin_solver, int _myid, int _intranode_rank,
                               int _devCount_per_node) :
      _low_device_memory(low_device_memory), qkpts(_nqkpt), G_tskij_host_(G_tskij_host), V_Qpm(_NQ, _nao, _nao), V_Qim(_NQ, _nao, _nao),
      Gk1_stij(_ns, _nts, _nao, _nao), Gk_smtij(_ns, _nts, _nao, _nao),
      qpt(_nao, _NQ, _ns, _nts, _nw_b, Ttn_FB.data(), Tnt_BF.data(), cuda_lin_solver), _qkpt_cublas_handles(_nqkpt) {
    if (cudaSetDevice(_intranode_rank % _devCount_per_node) != cudaSuccess) throw std::runtime_error("Error in cudaSetDevice2");
    if (cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(_myid) + ": error initializing cublas");
    /*#ifdef ENABLE_TENSOR_CORE
          ///EXPERIMENTAL: Set the math mode to allow cuBLAS to use Tensor Cores:
          cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    #endif*/
    if (cusolverDnCreate(&_solver_handle) != CUSOLVER_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(_myid) + ": cusolver init problem");

    // initialize and transfer device green's function, self-energy and IR matrices
    _X2C = _ns == 4;
    if (_X2C and !_low_device_memory)
      throw std::logic_error("cugw_utils for 2C Hamiltonian in high_device_memory mode is not implemented.");
    if (!_low_device_memory) {
      allocate_G_and_Sigma(&g_kstij_device, &g_ksmtij_device, &sigma_kstij_device, G_tskij_host.data(), _ink, _nao, _nts, _ns);
    } else {
      sigma_kstij_device = nullptr;
      g_kstij_device     = nullptr;
      g_ksmtij_device    = nullptr;
    }

    // locks so that different threads don't write the results over each other
    cudaMalloc(&sigma_k_locks, _ink * sizeof(int));
    cudaMemset(sigma_k_locks, 0, _ink * sizeof(int));

    // IBZ upload infrastructure: dedicated stream + GPU-side fencing for G(k_ibz,-tau)
    if (_low_device_memory && !_X2C) {
      ibz_g_elems_ = static_cast<size_t>(_ns) * _nts * _nao * _nao;
      if (cudaStreamCreate(&ibz_upload_stream_) != cudaSuccess)
        throw std::runtime_error("failure creating ibz_upload_stream");
      cudaEventCreateWithFlags(&ibz_upload_ready_event_, cudaEventDisableTiming);
      if (cudaMallocHost(&ibz_pinned_buffer_, ibz_g_elems_ * sizeof(cxx_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating ibz_pinned_buffer");
      if (cudaMalloc(&ibz_g_device_, ibz_g_elems_ * sizeof(cuda_complex)) != cudaSuccess)
        throw std::runtime_error("failure allocating ibz_g_device");
    }

    qpt.init(&_handle, &_solver_handle);
    // Each process gets one cuda runner for qpoints
    for (int i = 0; i < _nqkpt; ++i) {
      if (cublasCreate(&_qkpt_cublas_handles[i]) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Rank " + std::to_string(_myid) + ": error initializing cublas");
      // initialize qkpt workers
      qkpts[i] = new gw_qkpt<prec>(_nao, _NQ, _ns, _nts, _nt_batch, &_qkpt_cublas_handles[i],
                                   g_kstij_device, g_ksmtij_device, sigma_kstij_device, sigma_k_locks);
    }

    // Upload symmetry maps and pre-built transforms to device.
    _cu_symmetry.initialize(sym_data, _nao, _NQ, _nts, _ns);

  }

  template <typename prec>
  void cugw_utils<prec>::accumulate_gw_selfenergy_on_device(int _nts, int _ns, int _nk, int _ink, int _nq, int _inq,
                                                            int _nao, std::complex<double>* Vk1k2_Qij,
                                                            ztensor<5>& Sigma_tskij_host,
                                                            int _devices_rank, int _devices_size,
                                                            bool low_device_memory, int verbose, gw_reader0_callback<prec>& r0,
                                                            gw_reader1_callback<prec>& r1,
                                                            gw_reader2_callback<prec>& r2) {
    // this is the main GW loop
    if (!_devices_rank && verbose > 0) std::cout << "GW main loop" << std::endl;
    qpt.verbose() = verbose;

    // Outer loop over q-points in the irreducible bosonic BZ (q-mesh)
    for (size_t q_ibz_idx = _devices_rank; q_ibz_idx < _inq; q_ibz_idx += _devices_size) {
      if (verbose > 2) std::cout << "q = " << q_ibz_idx << std::endl;
      size_t q_ir = _cu_symmetry.q_reduced_to_full(q_ibz_idx);  // canonical full BZ index of this irreducible q
      qpt.reset_Pqk0();
      prev_epoch_events_.clear();
      // P0 build phase: iterate over fermionic IBZ, for each k_ibz iterate over its star to build P0(q_ir)
      for (size_t k_ibz_id = 0; k_ibz_id < _ink; ++k_ibz_id) {
        // Scalar low-memory: load G(k_ibz,-tau) once per IBZ point, upload to device via dedicated stream.
        if (_low_device_memory && !_X2C) {
          r0(static_cast<int>(k_ibz_id), Gk_smtij);
          // GPU-side fence: ibz_upload_stream waits for all workers that consumed prior ibz buffer
          for (auto& ev : prev_epoch_events_) {
            cudaStreamWaitEvent(ibz_upload_stream_, ev, 0);
          }
          prev_epoch_events_.clear();
          // CPU-side fence: ensure prior iteration's DMA from ibz_pinned_buffer_ is complete
          // before we overwrite it with the new k_ibz data
          if (k_ibz_id > 0) {
            cudaEventSynchronize(ibz_upload_ready_event_);
          }
          // Upload G(k_ibz,-tau) to shared device buffer via pinned staging
          std::memcpy(ibz_pinned_buffer_, Gk_smtij.data(), ibz_g_elems_ * sizeof(cxx_complex));
          cudaMemcpyAsync(ibz_g_device_, ibz_pinned_buffer_, ibz_g_elems_ * sizeof(cuda_complex),
                          cudaMemcpyHostToDevice, ibz_upload_stream_);
          cudaEventRecord(ibz_upload_ready_event_, ibz_upload_stream_);
        }

        // Iterate directly over k-star without grouping
        for (auto k_full : _cu_symmetry.k_star(k_ibz_id)) {
          gw_qkpt<prec>* qkpt = obtain_idle_qkpt(qkpts);
          size_t k1_full = _cu_symmetry.k1_from_k2q(k_full, q_ir);
          std::array<size_t, 4> k_vector = {k_full, 0, q_ir, k1_full};

          size_t k_reduced_id  = k_ibz_id;
          size_t k1_reduced_id = _cu_symmetry.k_full_to_reduced(k1_full);
          bool need_minus_k  = false;
          bool need_minus_k1 = (_cu_symmetry.k_reduced_to_full(k1_reduced_id) != k1_full);
          r1(k_full, k1_full, k_reduced_id, k1_reduced_id, k_vector, V_Qpm, Vk1k2_Qij, Gk1_stij, need_minus_k, need_minus_k1);

          if (_low_device_memory && !_X2C) {
            // Low-memory scalar: use ibz G uploaded on dedicated stream
            qkpt->set_up_qkpt_first_coulomb_only(V_Qpm.data(), k_reduced_id, k1_reduced_id);
            // Load G(k1) from host via worker's pinned staging buffer (race-safe)
            qkpt->load_Gk1_to_device(Gk1_stij.data(), ibz_g_elems_);
            // Wait for ibz G upload, then rotate both G matrices on worker stream
            cudaStreamWaitEvent(qkpt->stream(), ibz_upload_ready_event_, 0);
            _cu_symmetry.transform_k_ao_device(qkpt->handle(), qkpt->stream(), qkpt->g_smtij_device(),
                                               k_full, qkpt->g_smtij_device(),
                                               Gk_smtij.shape()[1], Gk_smtij.shape()[0],
                                               ibz_g_device_,
                                               qkpt->transform_input_scratch(), qkpt->transform_work_scratch());
            // ibz_g_device_ is no longer needed after this transform; record event so next k_ibz
            // upload can proceed without waiting for the full P0 computation to complete.
            cudaEventRecord(qkpt->transform_done_event(), qkpt->stream());
            prev_epoch_events_.push_back(qkpt->transform_done_event());
            _cu_symmetry.transform_k_ao_device(qkpt->handle(), qkpt->stream(), qkpt->g_stij_device(),
                                               k1_full, qkpt->g_stij_device(),
                                               Gk_smtij.shape()[1], Gk_smtij.shape()[0],
                                               nullptr,
                                               qkpt->transform_input_scratch(), qkpt->transform_work_scratch());
          } else if (_low_device_memory && _X2C) {
            // X2C low-memory
            qkpt->set_up_qkpt_first(Gk1_stij.data(), Gk_smtij.data(), V_Qpm.data(), k_reduced_id, k1_reduced_id);
          } else {
            // High-memory scalar
            qkpt->set_up_qkpt_first(nullptr, nullptr, V_Qpm.data(), k_reduced_id, k1_reduced_id);
            prepare_first_contraction_highmem_scalar(qkpt, k_full, k1_full);
          }
          qkpt->compute_first_tau_contraction(qpt.Pqk0_tQP(qkpt->all_done_event()), qpt.Pqk0_tQP_lock());
        }
      }

      // Compute Polarization Pq from irreducible polarization Pqk0
      qpt.wait_for_kpts();
      qpt.scale_Pq0_tQP(1. / _nk);
      qpt.transform_tw();
      qpt.compute_Pq();
      qpt.transform_wt();

      // Accumulate Sigma(k) for k in the fermionic IBZ, iterating over the full
      // star of q_ibz_idx in the q-mesh full BZ. For each degenerate q_deg, the
      // U_q transform (spatial rotation + TR) maps P(q_ir) -> P(q_deg).
      for (size_t k_reduced_id = 0; k_reduced_id < _ink; ++k_reduced_id) {
        size_t k = _cu_symmetry.k_reduced_to_full(k_reduced_id);

        // Iterate directly over q-star without grouping
        for (auto q_deg_signed : _cu_symmetry.q_star(q_ibz_idx)) {
          size_t q_deg         = static_cast<size_t>(q_deg_signed);
          size_t k1            = _cu_symmetry.k2_from_k1q(k, q_deg);
          size_t k1_reduced_id = _cu_symmetry.k_full_to_reduced(k1);
          bool   need_minus_k1 = _cu_symmetry.k_reduced_to_full(k1_reduced_id) != k1;
          std::array<size_t, 4> k_vector = {k, q_deg, 0, k1};

          r2(k, k1, k1_reduced_id, k_vector, V_Qim, Vk1k2_Qij, Gk1_stij, need_minus_k1);

          gw_qkpt<prec>* qkpt = obtain_idle_qkpt_for_sigma(qkpts, _low_device_memory, Sigmak_stij, Sigma_tskij_host, _X2C);
          qkpt->set_k_red_id(k_reduced_id);
          bool q_need_conj = (_cu_symmetry.q_tr_conj(q_deg) != 0);

          if (_low_device_memory) {
            // Low-memory (X2C or scalar)
            qkpt->set_up_qkpt_second(Gk1_stij.data(), V_Qim.data(), k_reduced_id, k1_reduced_id);
            if (_X2C) {
              const auto* U_q = reinterpret_cast<const cuda_complex*>(_cu_symmetry.q_p0_transform_d(q_deg));
              qkpt->compute_second_tau_contraction_2C(
                  qpt.Pqk_tQP(qkpt->all_done_event(), qkpt->stream(), 0),
                  U_q, q_need_conj);
            } else {
              // Rotate G(k1_ibz) -> G(k1_full) before Sigma contraction
              _cu_symmetry.transform_k_ao_device(qkpt->handle(), qkpt->stream(), qkpt->g_stij_device(),
                                                 k1, qkpt->g_stij_device(), _nts, _ns,
                                                 nullptr,
                                                 qkpt->transform_input_scratch(), qkpt->transform_work_scratch());
              const auto* U_q = std::is_same_v<prec, double>
                  ? reinterpret_cast<const cuda_complex*>(_cu_symmetry.q_p0_transform_d(q_deg))
                  : reinterpret_cast<const cuda_complex*>(_cu_symmetry.q_p0_transform_f(q_deg));
              qkpt->compute_second_tau_contraction(
                  qpt.Pqk_tQP(qkpt->all_done_event(), qkpt->stream(), 0),
                  U_q, q_need_conj);
            }
          } else {
            // High-memory scalar
            qkpt->set_up_qkpt_second(nullptr, V_Qim.data(), k_reduced_id, k1_reduced_id);
            _cu_symmetry.transform_k_ao_device(qkpt->handle(), qkpt->stream(), qkpt->g_stij_device(),
                                               k1, qkpt->g_stij_device(), _nts, _ns,
                                               nullptr,
                                               qkpt->transform_input_scratch(), qkpt->transform_work_scratch());
            const auto* U_q = std::is_same_v<prec, double>
                ? reinterpret_cast<const cuda_complex*>(_cu_symmetry.q_p0_transform_d(q_deg))
                : reinterpret_cast<const cuda_complex*>(_cu_symmetry.q_p0_transform_f(q_deg));
            qkpt->compute_second_tau_contraction(
                qpt.Pqk_tQP(qkpt->all_done_event(), qkpt->stream(), 0),
                U_q, q_need_conj);
          }
        }
      }

      // qpt P/P0 buffers are reused on the next q iteration. Ensure no qkpt stream
      // still consumes the current q's polarization before re-entering P0 build.
      wait_and_clean_qkpts(qkpts, _low_device_memory, Sigmak_stij, Sigma_tskij_host, _X2C);
    }
    cudaDeviceSynchronize();
    if (!_low_device_memory and !_X2C) {
      copy_Sigma_from_device_to_host(sigma_kstij_device, Sigma_tskij_host.data(), _ink, _nao, _nts, _ns);
    }
  }

  template <typename prec>
  void cugw_utils<prec>::prepare_first_contraction_highmem_scalar(gw_qkpt<prec>* qkpt, size_t k_full, size_t k1_full) {
    _cu_symmetry.transform_k_ao_device(qkpt->handle(), qkpt->stream(), qkpt->g_smtij_device(),
                                       k_full, qkpt->g_smtij_device(), Gk_smtij.shape()[1], Gk_smtij.shape()[0],
                                       nullptr,
                                       qkpt->transform_input_scratch(), qkpt->transform_work_scratch());
    _cu_symmetry.transform_k_ao_device(qkpt->handle(), qkpt->stream(), qkpt->g_stij_device(),
                                       k1_full, qkpt->g_stij_device(), Gk_smtij.shape()[1], Gk_smtij.shape()[0],
                                       nullptr,
                                       qkpt->transform_input_scratch(), qkpt->transform_work_scratch());
  }

  template <typename prec>
  void cugw_utils<prec>::copy_Sigma(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_stij, int k, int nts,
                                    int ns) {
    for (size_t t = 0; t < nts; ++t) {
      for (size_t s = 0; s < ns; ++s) {
        matrix(Sigma_tskij_host(t, s, k)) += matrix(Sigmak_stij(s, t)).template cast<typename std::complex<double>>();
      }
    }
  }

  template <typename prec>
  void cugw_utils<prec>::copy_Sigma_2c(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_4tij, int k, int nts) {
    size_t    nao = Sigmak_4tij.shape()[3];
    size_t    nso = 2 * nao;
    MatrixXcf Sigma_ij(nso, nso);
    for (size_t ss = 0; ss < 3; ++ss) {
      size_t a       = (ss % 2 == 0) ? 0 : 1;
      size_t b       = ((ss + 1) / 2 != 1) ? 0 : 1;
      size_t i_shift = a * nao;
      size_t j_shift = b * nao;
      for (size_t t = 0; t < nts; ++t) {
        matrix(Sigma_tskij_host(t, 0, k)).block(i_shift, j_shift, nao, nao) +=
            matrix(Sigmak_4tij(ss, t)).template cast<typename std::complex<double>>();
        if (ss == 2) {
          matrix(Sigma_tskij_host(t, 0, k)).block(j_shift, i_shift, nao, nao) +=
              matrix(Sigmak_4tij(ss, t)).conjugate().transpose().template cast<typename std::complex<double>>();
        }
      }
    }
  }

  template <typename prec>
  cugw_utils<prec>::~cugw_utils() {
    for (int i = 0; i < qkpts.size(); ++i) {
      delete qkpts[i];
    }
    if (cublasDestroy(_handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublas error destroying handle");
    if (cusolverDnDestroy(_solver_handle) != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("culapck error destroying handle");
    for (int i = 0; i < qkpts.size(); ++i) {
      if (cublasDestroy(_qkpt_cublas_handles[i]) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublas error destroying handle");
    }
    if (!_low_device_memory) cudaFree(g_kstij_device);
    if (!_low_device_memory) cudaFree(g_ksmtij_device);
    cudaFree(sigma_kstij_device);
    cudaFree(sigma_k_locks);

    // IBZ upload infrastructure cleanup
    if (ibz_g_device_ != nullptr) cudaFree(ibz_g_device_);
    if (ibz_pinned_buffer_ != nullptr) cudaFreeHost(ibz_pinned_buffer_);
    if (ibz_upload_stream_ != nullptr) {
      cudaEventDestroy(ibz_upload_ready_event_);
      cudaStreamDestroy(ibz_upload_stream_);
    }
  }

  template class cugw_utils<float>;
  template class cugw_utils<double>;

}  // namespace green::gpu
