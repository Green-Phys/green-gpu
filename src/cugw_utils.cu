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

#include <green/gpu/cugw_utils.h>

namespace green::gpu {

  template <typename prec>
  cugw_utils<prec>::cugw_utils(int _nts, int _nt_batch, int _nw_b, int _ns, int _nk, int _ink, int _nq, int _inq, int _nqkpt,
                               int _NQ, int _nao, const cu_symmetry_data& sym_data, ztensor_view<5>& G_tskij_host,
                               bool low_device_memory, const MatrixXcd& Ttn_FB, const MatrixXcd& Tnt_BF,
                               LinearSolverType cuda_lin_solver, int _myid, int _intranode_rank, int _devCount_per_node) :
      _low_device_memory(low_device_memory), qkpts(_nqkpt), G_tskij_host_(G_tskij_host), V_Qpm(_NQ, _nao, _nao),
      V_Qim(_NQ, _nao, _nao), Gk1_stij(_ns, _nts, _nao, _nao), Gk_smtij(_ns, _nts, _nao, _nao),
      qpt(_nao, _NQ, _ns, _nts, _nw_b, Ttn_FB.data(), Tnt_BF.data(), cuda_lin_solver), _qkpt_cublas_handles(_nqkpt) {
    if (cudaSetDevice(_intranode_rank % _devCount_per_node) != cudaSuccess) throw std::runtime_error("Error in cudaSetDevice2");
    if (cublasCreate(&_handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(_myid) + ": error initializing cublas");
    if (cusolverDnCreate(&_solver_handle) != CUSOLVER_STATUS_SUCCESS)
      throw std::runtime_error("Rank " + std::to_string(_myid) + ": cusolver init problem");

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
    for (int i = 0; i < _nqkpt; ++i) {
      if (cublasCreate(&_qkpt_cublas_handles[i]) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Rank " + std::to_string(_myid) + ": error initializing cublas");
      qkpts[i] = new gw_qkpt<prec>(_nao, _NQ, _ns, _nts, _nt_batch, &_qkpt_cublas_handles[i],
                                   g_kstij_device, g_ksmtij_device, sigma_kstij_device, sigma_k_locks);
    }

    _cu_symmetry.initialize(sym_data, _nao, _NQ, _nts, _ns);
  }

  template <typename prec>
  cugw_utils<prec>::~cugw_utils() {
    for (int i = 0; i < qkpts.size(); ++i) {
      delete qkpts[i];
    }
    if (cublasDestroy(_handle) != CUBLAS_STATUS_SUCCESS)
      std::cerr << "cugw_utils: cublasDestroy failed for main handle.\n";
    if (cusolverDnDestroy(_solver_handle) != CUSOLVER_STATUS_SUCCESS)
      std::cerr << "cugw_utils: cusolverDnDestroy failed.\n";
    for (int i = 0; i < qkpts.size(); ++i) {
      if (cublasDestroy(_qkpt_cublas_handles[i]) != CUBLAS_STATUS_SUCCESS)
        std::cerr << "cugw_utils: cublasDestroy failed for qkpt handle " << i << ".\n";
    }
    if (!_low_device_memory) cudaFree(g_kstij_device);
    if (!_low_device_memory) cudaFree(g_ksmtij_device);
    cudaFree(sigma_kstij_device);
    cudaFree(sigma_k_locks);

    if (ibz_g_device_ != nullptr) cudaFree(ibz_g_device_);
    if (ibz_pinned_buffer_ != nullptr) cudaFreeHost(ibz_pinned_buffer_);
    if (ibz_upload_stream_ != nullptr) {
      cudaEventDestroy(ibz_upload_ready_event_);
      cudaStreamDestroy(ibz_upload_stream_);
    }
  }

  template <typename prec>
  void cugw_utils<prec>::accumulate_gw_selfenergy_on_device(int _nts, int _ns, int _nk, int _ink, int _nq, int _inq,
                                                            int _nao, std::complex<double>* Vk1k2_Qij,
                                                            ztensor<5>& Sigma_tskij_host,
                                                            int _devices_rank, int _devices_size,
                                                            bool low_device_memory, int verbose,
                                                            gw_reader0_callback<prec>& r0,
                                                            gw_reader1_callback<prec>& r1,
                                                            gw_reader2_callback<prec>& r2) {
    if (!_devices_rank && verbose > 0) std::cout << "GW main loop" << std::endl;
    qpt.verbose() = verbose;

    for (size_t q_ibz_idx = _devices_rank; q_ibz_idx < _inq; q_ibz_idx += _devices_size) {
      if (verbose > 2) std::cout << "q = " << q_ibz_idx << std::endl;
      size_t q_ir = _cu_symmetry.q_reduced_to_full(q_ibz_idx);  // canonical full-BZ representative of this IBZ q
      qpt.reset_Pqk0();
      prev_epoch_events_.clear();

      // --- P0 build: accumulate bare polarization over fermionic IBZ k-stars ---
      for (size_t k_ibz_id = 0; k_ibz_id < _ink; ++k_ibz_id) {
        if (_low_device_memory && !_X2C) {
          r0(static_cast<int>(k_ibz_id), Gk_smtij);
          upload_ibz_g(k_ibz_id);
        }

        for (auto k_full : _cu_symmetry.k_star(k_ibz_id)) {
          gw_qkpt<prec>* qkpt = obtain_idle_qkpt(qkpts);
          size_t k1_full = _cu_symmetry.k1_from_k2q(k_full, q_ir);
          std::array<size_t, 4> k_vector = {k_full, 0, q_ir, k1_full};

          size_t k_reduced_id  = k_ibz_id;
          size_t k1_reduced_id = _cu_symmetry.k_full_to_reduced(k1_full);
          bool need_minus_k  = false;
          bool need_minus_k1 = false;
          r1(k_full, k1_full, k_reduced_id, k1_reduced_id, k_vector, V_Qpm, Vk1k2_Qij, Gk1_stij, need_minus_k, need_minus_k1);

          if (_low_device_memory && !_X2C) {
            qkpt->upload_p0_coulomb(V_Qpm.data(), k_reduced_id, k1_reduced_id);
            prepare_first_contraction_lowmem_scalar(qkpt, k_full, k1_full);
          } else if (_low_device_memory && _X2C) {
            r0(static_cast<int>(k_full), Gk_smtij);
            qkpt->upload_p0_inputs(Gk1_stij.data(), Gk_smtij.data(), V_Qpm.data(), k_reduced_id, k1_reduced_id);
          } else {
            qkpt->upload_p0_inputs(nullptr, nullptr, V_Qpm.data(), k_reduced_id, k1_reduced_id);
            prepare_first_contraction_highmem_scalar(qkpt, k_full, k1_full);
          }

          qkpt->compute_first_tau_contraction(qpt.Pqk0_tQP(qkpt->all_done_event()), qpt.Pqk0_tQP_lock());
        }
      }

      // --- Solve BSE: P0 -> P (dressed polarization) ---
      qpt.wait_for_kpts();
      qpt.scale_Pq0_tQP(1. / _nk);
      qpt.transform_tw();
      qpt.compute_Pq();
      qpt.transform_wt();

      // --- Sigma accumulation: contract P with G over q-star ---
      for (size_t k_reduced_id = 0; k_reduced_id < _ink; ++k_reduced_id) {
        size_t k = _cu_symmetry.k_reduced_to_full(k_reduced_id);

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
            qkpt->upload_sigma_inputs(Gk1_stij.data(), V_Qim.data(), k_reduced_id, k1_reduced_id);
          } else {
            qkpt->upload_sigma_inputs(nullptr, V_Qim.data(), k_reduced_id, k1_reduced_id);
          }
          // Scalar path: rotate G(k1_ibz) → G(k1_full) on the qkpt's g_stij buffer.
          // X2C path: G rotation is handled separately upstream, so skip the AO transform here.
          // TODO: implement the X2C (nso×nso) G rotation on the GPU so this branch can go away
          //       and the AO transform becomes unconditional.
          if (!_X2C) {
            _cu_symmetry.transform_k_ao_device(qkpt->handle(), qkpt->stream(), qkpt->g_stij_device(),
                                               k1, qkpt->g_stij_device(),
                                               Gk1_stij.shape()[1], Gk1_stij.shape()[0],
                                               nullptr,
                                               qkpt->transform_input_scratch(), qkpt->transform_work_scratch());
          }
          accumulate_sigma(qkpt, q_deg, q_need_conj);
        }
      }

      wait_and_clean_qkpts(qkpts, _low_device_memory, Sigmak_stij, Sigma_tskij_host, _X2C);
    }
    cudaDeviceSynchronize();
    // --- High-memory path: copy accumulated Sigma from device to host ---
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
  void cugw_utils<prec>::prepare_first_contraction_lowmem_scalar(gw_qkpt<prec>* qkpt, size_t k_full, size_t k1_full) {
    // Load G(k1_ibz,tau) from host via per-worker pinned buffer (race-safe)
    qkpt->load_Gk1_to_device(Gk1_stij.data(), ibz_g_elems_);
    // Worker stream waits for IBZ G(k_ibz,-tau) upload to land on device
    cudaStreamWaitEvent(qkpt->stream(), ibz_upload_ready_event_, 0);
    // Rotate G(k_ibz,-tau) -> G(k_full,-tau) in-place using shared IBZ device buffer as source
    _cu_symmetry.transform_k_ao_device(qkpt->handle(), qkpt->stream(), qkpt->g_smtij_device(),
                                       k_full, qkpt->g_smtij_device(),
                                       Gk_smtij.shape()[1], Gk_smtij.shape()[0],
                                       ibz_g_device_,
                                       qkpt->transform_input_scratch(), qkpt->transform_work_scratch());
    // ibz_g_device_ is free after this transform; record event so next IBZ upload need not
    // wait for the full P0 computation to finish.
    cudaEventRecord(qkpt->transform_done_event(), qkpt->stream());
    prev_epoch_events_.push_back(qkpt->transform_done_event());
    // Rotate G(k1_ibz,tau) -> G(k1_full,tau) in-place (no shared source needed)
    _cu_symmetry.transform_k_ao_device(qkpt->handle(), qkpt->stream(), qkpt->g_stij_device(),
                                       k1_full, qkpt->g_stij_device(),
                                       Gk_smtij.shape()[1], Gk_smtij.shape()[0],
                                       nullptr,
                                       qkpt->transform_input_scratch(), qkpt->transform_work_scratch());
  }

  template <typename prec>
  void cugw_utils<prec>::accumulate_sigma(gw_qkpt<prec>* qkpt, size_t q_deg, bool q_need_conj) {
    // q-space transform + second-tau contraction.  Kernel chain (CPU math):
    //     Y2_out = U_conj · P · U · Y1ᵀ            then  Σ = -V_nPj · Y2_out
    //
    // Per the row/col-major convention derived in compute_second_tau_contraction,
    // operand and Pq selection per TR branch is:
    //
    //   non-TR (q_need_conj=false):  P_qdeg = U_q · P · U_q†
    //       Pq    = Pqk_tQP_         (P math)
    //       U     = U_q stored
    //       U_conj = U_q_conj stored
    //
    //   TR    (q_need_conj=true ):   P_qdeg = U_q_conj · conj(P) · U_q_conj†
    //       Pq    = Pqk_tQP_conj_    (conj(P) math)
    //       U     = U_q_conj stored
    //       U_conj = U_q stored
    //
    // Both U_q and U_q_conj are precomputed at cu_symmetry::initialize.
    // For the scalar path the caller has already rotated G(k1_ibz) → G(k1_full)
    // on qkpt->g_stij_device() before invoking this helper; X2C handles G
    // rotation separately.
    const cuda_complex* U_q      = _cu_symmetry.q_p0_transform(q_deg);
    const cuda_complex* U_q_conj = _cu_symmetry.q_p0_transform_conj(q_deg);
    const auto* U      = q_need_conj ? U_q_conj : U_q;
    const auto* U_conj = q_need_conj ? U_q      : U_q_conj;
    qkpt->compute_second_tau_contraction(
        qpt.Pqk_tQP(qkpt->all_done_event(), qkpt->stream(), q_need_conj),
        U, U_conj);
  }

  template <typename prec>
  void cugw_utils<prec>::upload_ibz_g(size_t k_ibz_id) {
    // GPU-side fence: ibz_upload_stream waits for all workers that consumed the prior ibz buffer
    for (auto& ev : prev_epoch_events_) {
      cudaStreamWaitEvent(ibz_upload_stream_, ev, 0);
    }
    prev_epoch_events_.clear();
    // CPU-side fence: prior iteration's DMA from ibz_pinned_buffer_ must be complete
    // before we overwrite it with the new k_ibz data
    if (k_ibz_id > 0) {
      cudaEventSynchronize(ibz_upload_ready_event_);
    }
    std::memcpy(ibz_pinned_buffer_, Gk_smtij.data(), ibz_g_elems_ * sizeof(cxx_complex));
    cudaMemcpyAsync(ibz_g_device_, ibz_pinned_buffer_, ibz_g_elems_ * sizeof(cuda_complex),
                    cudaMemcpyHostToDevice, ibz_upload_stream_);
    cudaEventRecord(ibz_upload_ready_event_, ibz_upload_stream_);
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
  void cugw_utils<prec>::copy_Sigma_2c(ztensor<5>& Sigma_tskij_host, tensor<std::complex<prec>, 4>& Sigmak_4tij, int k,
                                       int nts) {
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

  template class cugw_utils<float>;
  template class cugw_utils<double>;

}  // namespace green::gpu
