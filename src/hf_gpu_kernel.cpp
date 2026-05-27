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

#include <green/gpu/cu_symmetry.h>
#include <green/gpu/cuhf_utils.h>
#include <green/gpu/hf_gpu_kernel.h>

namespace green::gpu {
  // Defined in gw_gpu_kernel.cpp. Forward-declared here so we don't have to pull
  // <green/gpu/gw_gpu_kernel.h> (with its grids/transformer dependencies) into the
  // HF translation unit.
  cu_symmetry_data make_cu_symmetry_data(const symmetry::brillouin_zone_utils& bz,
                                         int nao, int naux,
                                         bool build_k_ao, bool build_q_p0);

  void hf_gpu_kernel::HF_check_devices_free_space() {
    std::cout << std::setprecision(4) << std::boolalpha;
    // check devices' free space and determine nkbatch
    std::size_t hf_utils_size = cuhf_utils::size_divided_by_kbatch(_nao, _NQ);
    std::size_t available_memory;
    std::size_t total_memory;
    cudaMemGetInfo(&available_memory, &total_memory);
    _nk_batch = std::min(int(available_memory / hf_utils_size), int(_ink));
    _nk_batch = std::min(int(_nk_batch), 16);
    if (!_devices_rank) {
      std::cout << "Available memory: " << available_memory / (1024 * 1024. * 1024.) << " GB "
                << " of total: " << total_memory / (1024 * 1024. * 1024.) << " GB" << std::endl;
      std::cout << "Will take nkbatch = " << _nk_batch << "." << std::endl;
      std::cout << "Size of hf_utils per GPU: " << (_nk_batch * hf_utils_size) / (1024 * 1024. * 1024.) << " GB " << std::endl;
      std::cout << "Additional CPU memory per node in cuHF solver: "
                << (_devCount_per_node * _nk_batch * _NQ * _nao * _nao) * sizeof(std::complex<double>) / (1024. * 1024. * 1024)
                << " GB" << std::endl;
    }
    if (_nk_batch == 0) throw std::runtime_error("Not enough gpu memory for cuda HF.");
    std::cout << std::setprecision(15);
  }

  void hf_gpu_kernel::read_exchange_VkQij(int k, int k2, std::complex<double>* Vk1k2_Qij, ztensor<4>& V_kbatchQij) {
    ztensor<3> V(_NQ, _nao, _nao);
    size_t     nk_mult = std::min(_nk_batch, _nk - k2);
    for (size_t ki = 0; ki < nk_mult; ++ki) {
      _coul_int->symmetrize(Vk1k2_Qij, V, k, k2 + ki);
      memcpy(V_kbatchQij.data() + ki * _NQnaosq, V.data(), _NQnaosq * sizeof(std::complex<double>));
    }
  }
  void hf_gpu_kernel::read_exchange_VkQij(int k, int k2, ztensor<4>& V_kbatchQij) {
    ztensor<3> V(_NQ, _nao, _nao);
    size_t     nk_mult = std::min(_nk_batch, _nk - k2);
    for (size_t ki = 0; ki < nk_mult; ++ki) {
      _coul_int->read_integrals(k, k2 + ki);
      _coul_int->symmetrize(V, k, k2 + ki);
      memcpy(V_kbatchQij.data() + ki * _NQnaosq, V.data(), _NQnaosq * sizeof(std::complex<double>));
    }
  }

  ztensor<4> hf_gpu_kernel::solve(const ztensor<4>& dm) {
    statistics.start("Total");
    statistics.start("Initialization");
    ztensor<4> new_Fock(_ns, _ink, _nso, _nso);
    new_Fock.set_zero();
    setup_MPI_structure();
    _coul_int = new df_integral_t(_path, _nao, _nk, _NQ, _bz_utils);
    MPI_Barrier(utils::context().global);
    set_shared_Coulomb();
    statistics.end();
    update_integrals(_coul_int, statistics);
    // Only those processes assigned with a device will be involved in HF self-energy calculation
    if (_devices_comm != MPI_COMM_NULL) {
      statistics.start("Exchange diagram");
      compute_exchange_selfenergy(new_Fock, dm);
      statistics.end();
    }

    statistics.start("Direct diagram");
    compute_direct_selfenergy(new_Fock, dm);
    statistics.end();

    statistics.start("Add Ewald");
    add_Ewald(new_Fock, dm, _S_k, _madelung);
    statistics.end();

    statistics.start("Fock reduce");
    utils::allreduce(MPI_IN_PLACE, new_Fock.data(), new_Fock.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, utils::context().global);
    statistics.end();
    statistics.end();
    statistics.print(utils::context().global);

    clean_MPI_structure();
    clean_shared_Coulomb();
    delete _coul_int;
    MPI_Barrier(utils::context().global);
    return new_Fock;
  }

  void scalar_hf_gpu_kernel::compute_exchange_selfenergy(ztensor<4>& new_Fock, const ztensor<4>& dm) {
    statistics.start("Initialization");
    ztensor<4> dm_fbz(_ns, _nk, _nao, _nao);
    get_dm_fbz(dm_fbz, dm);
    // Also determines _nk_batch
    HF_check_devices_free_space();
    // Each process gets one cuda runner hf_utils
    cuhf_utils hf_utils(_nk, _ink, _ns, _nao, _NQ, _nk_batch, dm_fbz, utils::context().global_rank, utils::context().node_rank,
                        _devCount_per_node);

    statistics.end();

    MPI_Barrier(_devices_comm);

    // FIXME Potential to be too large in memory
    ztensor<4> V_kbatchQij(_nk_batch, _NQ, _nao, _nao);

    // callback for shared-memory integrals
    hf_reader1 r1 = [&](int k, int k2, std::complex<double>* Vq, ztensor<4>& Vq_batch) {
      statistics.start("Read");
      read_exchange_VkQij(k, k2, Vq, Vq_batch);
      statistics.end();
    };
    // callback for local integrals
    hf_reader2 r2 = [&](int k, int k2, ztensor<4>& Vq_batch) {
      statistics.start("Read");
      read_exchange_VkQij(k, k2, Vq_batch);
      statistics.end();
    };
    statistics.start("Exchange loop");
    hf_utils.accumulate_exchange_on_device(_Vk1k2_Qij, V_kbatchQij, new_Fock, _nk_batch, _coul_int_reading_type,
                         _devices_rank, _devices_size, _bz_utils.k_symmetry().reduced_to_full(), r1, r2);
    statistics.end();
  }

  void scalar_hf_gpu_kernel::compute_direct_selfenergy(ztensor<4>& F, const ztensor<4>& dm) {
    // TODO or NOTE: It looks like we are building the Hartree term on single CPU, with no MPI whatsoever
    // I see - we build the Hartree bubble on all the cpu procs through full sum, and only then use MPI for _ink * _ns
    // to update the Fock. This can be fixed later.
    if (utils::context().global_rank < _ink * _ns) {
      int hf_nprocs = (utils::context().global_size > _ink * _ns) ? _ink * _ns : utils::context().global_size;

      // Direct diagram
      MatrixXcd  X1(_nao, _nao);
      ztensor<3> v(_NQ, _nao, _nao);
      ztensor<2> upper_Coul(_NQ, 1);
      MMatrixXcd X1m(X1.data(), _nao * _nao, 1);
      MMatrixXcd vm(v.data(), _NQ, _nao * _nao);
      for (int ikps = 0; ikps < _nk * _ns; ++ikps) {
        int is    = ikps % _ns;
        int ikp   = ikps / _ns;
        if (_coul_int_reading_type == as_a_whole) {
          _coul_int->symmetrize(_Vk1k2_Qij, v, ikp, ikp);
        } else {
          _coul_int->read_integrals(ikp, ikp);
          _coul_int->symmetrize(v, ikp, ikp);
        }

        X1 = _bz_utils.k_symmetry().value_AO(dm(is), ikp).transpose();
        // (Q, 1) = (Q, ab) * (ab, 1)
        matrix(upper_Coul) += vm * X1m;
      }
      upper_Coul /= double(_nk);

      for (int ii = utils::context().global_rank; ii < _ink * _ns; ii += hf_nprocs) {
        int is   = ii / _ink;
        int ik   = ii % _ink;
        int k_ir = _bz_utils.k_symmetry().full_point(ik);
        if (_coul_int_reading_type == as_a_whole) {
          _coul_int->symmetrize((std::complex<double>*)_Vk1k2_Qij, v, k_ir, k_ir);
        } else {
          _coul_int->read_integrals(k_ir, k_ir);
          _coul_int->symmetrize(v, k_ir, k_ir);
        }

        MMatrixXcd Fm(F.data() + is * _ink * _nao * _nao + ik * _nao * _nao, 1, _nao * _nao);
        // (1, ij) = (1, Q) * (Q, ij)
        Fm += matrix(upper_Coul).transpose() * vm;
      }
    }
  }

  void scalar_hf_gpu_kernel::add_Ewald(ztensor<4>& new_Fock, const ztensor<4>& dm, const ztensor<4>& S, double madelung) {
    if (utils::context().global_rank < _ink * _ns) {
      double prefactor = (_ns == 2) ? 1.0 : 0.5;
      size_t hf_nprocs = (utils::context().global_size > _ink * _ns) ? _ink * _ns : utils::context().global_size;
      for (size_t ii = utils::context().global_rank; ii < _ns * _ink; ii += hf_nprocs) {
        size_t      is = ii / _ink;
        size_t      ik = ii % _ink;
        CMMatrixXcd dmm(dm.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        CMMatrixXcd Sm(S.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        MMatrixXcd  Fm(new_Fock.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        Fm -= prefactor * madelung * Sm * dmm * Sm;
      }
    }
  }

  void scalar_hf_gpu_kernel::get_dm_fbz(ztensor<4>& dm_fbz, const ztensor<4>& dm) {
    size_t nknaosq  = _nk * _naosq;
    size_t inknaosq = _ink * _naosq;
    for (int s = 0; s < _ns; ++s) {
      dm_fbz(s) << _bz_utils.ibz_to_full(dm(s));
    }
  }

  void scalar_hf_gpu_kernel::complexity_estimation() {
    // Direct diagram
    double flop_count_direct =
        _ink * _ns * matmul_cost(1, _NQ, _naosq) + matmul_cost(1, _NQ, _ink) + _ink * _ns * matmul_cost(1, _naosq, _NQ);
    // Exchange diagram
    double flop_count_exchange = _ink * _ns * _nk * (matmul_cost(_NQ * _nao, _nao, _nao) + matmul_cost(_nao, _nao, _NQ * _nao)) +
                                 _ink * _ns * matmul_cost(1, _naosq, _nk);
    _hf_total_flops = flop_count_direct + flop_count_exchange;

    if (!utils::context().global_rank && _verbose > 1) {
      std::cout << "############ Total HF Operations per Iteration ############" << std::endl;
      std::cout << "Total:         " << _hf_total_flops << std::endl;
      std::cout << "Matmul (Direct diagram):  " << flop_count_direct << std::endl;
      std::cout << "Matmul (Exchange diagram): " << flop_count_exchange << std::endl;
      std::cout << "###########################################################" << std::endl;
    }
  }

  void x2c_hf_gpu_kernel::compute_exchange_selfenergy(ztensor<4> &new_Fock, const ztensor<4> &dm) {
    statistics.start("Initialization");

    // Build dm_fbz on device using the merged transform_k_ao_device pipeline:
    //   dm_fbz(k_full) = U_k · dm_ibz(reduced_point(k_full)) · U_k†  (· conj if TR).
    // The nso×nso U from the input file already encodes the σ_y spinor mixing for
    // TR-related k, and the post-step RSCAL conjugates the full nso×nso result.
    // Result lives on device in (nk, nso, nso) row-major layout; cuhf_utils picks
    // per-spin-block (aa, bb, ab) sub-views with lda=nso at GEMM time.
    cu_symmetry           sym;
    cu_symmetry_data      sym_data = make_cu_symmetry_data(_bz_utils, _nao, /*naux=*/0,
                                                            /*build_k_ao=*/true, /*build_q_p0=*/false);
    sym.initialize(sym_data, _nao, _nso, /*naux=*/0, /*nts=*/1, /*ns=*/1);

    const size_t     nsosq    = static_cast<size_t>(_nso) * _nso;
    cuDoubleComplex* dm_ibz_d = nullptr;
    cuDoubleComplex* dm_fbz_d = nullptr;
    if (cudaMalloc(&dm_ibz_d, _ink * nsosq * sizeof(cuDoubleComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate dm_ibz_d in x2c_hf_gpu_kernel::compute_exchange_selfenergy");
    if (cudaMalloc(&dm_fbz_d, _nk * nsosq * sizeof(cuDoubleComplex)) != cudaSuccess)
      throw std::runtime_error("Failed to allocate dm_fbz_d in x2c_hf_gpu_kernel::compute_exchange_selfenergy");

    // dm has shape (ns=1, ink, nso, nso); the s=0 slice contiguously occupies the first ink*nsosq elements.
    if (cudaMemcpy(dm_ibz_d, dm.data(), _ink * nsosq * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("Failed to upload dm_ibz to device");

    cublasHandle_t local_handle;
    if (cublasCreate(&local_handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Failed to create cublas handle for HF X2C transform");
    cudaStream_t local_stream;
    if (cudaStreamCreate(&local_stream) != cudaSuccess)
      throw std::runtime_error("Failed to create cuda stream for HF X2C transform");

    for (size_t k_full = 0; k_full < _nk; ++k_full) {
      size_t k_ibz = sym.k_full_to_reduced(k_full);
      sym.transform_k_ao_device(local_handle, local_stream,
                                dm_ibz_d + k_ibz  * nsosq, k_full,
                                dm_fbz_d + k_full * nsosq,
                                /*nts=*/1, /*ns=*/1,
                                /*ibz_in_device=*/nullptr,
                                /*input_scratch=*/nullptr,
                                /*work_scratch=*/nullptr);
    }
    cudaStreamSynchronize(local_stream);
    cudaStreamDestroy(local_stream);
    cublasDestroy(local_handle);
    cudaFree(dm_ibz_d);

    HF_check_devices_free_space();
    // Each NxN AO block of the 2-component exchange potential is evaluated individually
    // using pseudo-ns=3 (aa, bb, ab); ba is derived as ab.adjoint() in copy_2c_Fock_from_device_to_host.
    cuhf_utils hf_utils(_nk, _ink, _nao, _NQ, _nk_batch, dm_fbz_d,
                        utils::context().global_rank, utils::context().node_rank, _devCount_per_node);
    cudaFree(dm_fbz_d);  // cuhf_utils owns its own copy now
    statistics.end();
    MPI_Barrier(_devices_comm);

    // FIXME Potential to be too large in memory
    ztensor<4> V_kbatchQij(_nk_batch, _NQ, _nao, _nao);
    hf_reader1 r1 = [&](int k, int k2, std::complex<double>* Vq, ztensor<4> & Vq_batch) { statistics.start("Read"); read_exchange_VkQij(k, k2, Vq, Vq_batch);statistics.end();};
    hf_reader2 r2 = [&](int k, int k2, ztensor<4> & Vq_batch) { statistics.start("Read"); read_exchange_VkQij(k, k2, Vq_batch);statistics.end();};
    statistics.start("Exchange loop");
    hf_utils.accumulate_exchange_on_device(_Vk1k2_Qij, V_kbatchQij, new_Fock, _nk_batch, _coul_int_reading_type,
                         _devices_rank, _devices_size, _bz_utils.k_symmetry().reduced_to_full(), r1, r2);
    statistics.end();
  }

  void x2c_hf_gpu_kernel::compute_direct_selfenergy(ztensor<4> &new_Fock, const ztensor<4> &dm) {
    if (utils::context().global_rank < _ink) {
      int direct_nprocs = (utils::context().global_size > _ink)? _ink : utils::context().global_size;

      ztensor<3> v(_NQ, _nao, _nao);
      MMatrixXcd vm(v.data(), _NQ, _nao * _nao);

      MatrixXcd X1(_nao, _nao);
      MMatrixXcd X1m(X1.data(), _nao * _nao, 1);

      // Loop over full BZ to correctly account for TR-related k-points.
      // value_AO applies TR spin-flip (conj + aa<->bb swap) for anti-unitary star members,
      // matching hf_x2c_cpu_kernel::solve.
      ztensor<2> upper_Coul(_NQ, 1);
      for (int ikp = 0; ikp < _nk; ++ikp) {
        if (_coul_int_reading_type == as_a_whole) {
          _coul_int->symmetrize(_Vk1k2_Qij, v, ikp, ikp);
        } else {
          _coul_int->read_integrals(ikp, ikp);
          _coul_int->symmetrize(v, ikp, ikp);
        }

        // Get full-BZ dm via symmetry rotation (TR spin-flip for anti-unitary star members)
        MatrixXcd dm_so = _bz_utils.k_symmetry().value_AO(dm(0), ikp);
        // Sum of alpha-alpha and beta-beta spin block
        X1 = (dm_so.block(0, 0, _nao, _nao) + dm_so.block(_nao, _nao, _nao, _nao)).transpose();
        // (Q, 1) = (Q, ab) * (ab, 1)
        matrix(upper_Coul) += vm * X1m;
      }
      upper_Coul /= double(_nk);

      MatrixXcd Fm(1, _nao * _nao);
      MMatrixXcd Fmm(Fm.data(), _nao, _nao);
      for (int ik = utils::context().global_rank; ik < _ink; ik += direct_nprocs) {
        int k_ir = _bz_utils.k_symmetry().full_point(ik);

        if (_coul_int_reading_type == as_a_whole) {
          _coul_int->symmetrize(_Vk1k2_Qij, v, k_ir, k_ir);
        } else {
          _coul_int->read_integrals(k_ir, k_ir);
          _coul_int->symmetrize(v, k_ir, k_ir);
        }

        Fm = matrix(upper_Coul).transpose() * vm;
        MMatrixXcd Fm_nso(new_Fock.data() + ik*_nso*_nso, _nso, _nso);
        Fm_nso.block(0, 0, _nao, _nao) += Fmm;
        Fm_nso.block(_nao, _nao, _nao, _nao) += Fmm;
      }
    }
  }

  void x2c_hf_gpu_kernel::add_Ewald(ztensor<4>& new_Fock, const ztensor<4>& dm, const ztensor<4>& S, double madelung) {
    if (utils::context().global_rank < _ink * _ns) {
      int direct_nprocs = (utils::context().global_size > _ink) ? _ink : utils::context().global_size;
      // The Madelung uses the AO overlap S_aa (= S_bb) on both sides for all blocks —
      // this is S_AO (spin-independent), not the spinor off-diagonal S_ab which is zero.
      // s=0 (aa), s=1 (bb), s=2 (ab); ba is set as adjoint of the ab contribution.
      MatrixXcd buffer(_nao, _nao);
      for (size_t iks = utils::context().global_rank; iks < 3 * _ink; iks += direct_nprocs) {
        size_t      ik  = iks / 3;
        size_t      s   = iks % 3;
        size_t      a   = s % 2;             // s=0→0 (alpha), s=1→1 (beta), s=2→0 (alpha)
        size_t      b   = (s > 0) ? 1 : 0;  // s=0→0, s=1→1, s=2→1
        MMatrixXcd  Fm_nso(new_Fock.data() + ik * _nso * _nso, _nso, _nso);
        CMMatrixXcd Sm_nso(S.data() + ik * _nso * _nso, _nso, _nso);
        CMMatrixXcd dm_nso(dm.data() + ik * _nso * _nso, _nso, _nso);
        MatrixXcd   S_aa = Sm_nso.block(0, 0, _nao, _nao);  // AO overlap; same for all spin blocks
        buffer = madelung * S_aa * dm_nso.block(a * _nao, b * _nao, _nao, _nao).eval() * S_aa;
        Fm_nso.block(a * _nao, b * _nao, _nao, _nao) -= buffer;
        if (a != b) {  // s=2 (ab): beta-alpha is adjoint of alpha-beta contribution
          Fm_nso.block(_nao, 0, _nao, _nao) -= buffer.conjugate().transpose();
        }
      }
    }
  }

  void x2c_hf_gpu_kernel::get_dm_fbz(ztensor<4>& dm_fbz, const ztensor<4>& dm) {
    // Apply G(k) = U_k * G(ik) * U_k† via value_AO (handles both space group
    // rotation and time-reversal conjugation via k_sym_transform_ao).
    for (int k = 0; k < _nk; ++k) {
      MatrixXcd dm_k = _bz_utils.k_symmetry().value_AO(dm(0), k);
      matrix(dm_fbz(0, k)) = dm_k.block(0,    0,    _nao, _nao);  // alpha-alpha
      matrix(dm_fbz(1, k)) = dm_k.block(_nao, _nao, _nao, _nao);  // beta-beta
      matrix(dm_fbz(2, k)) = dm_k.block(0,    _nao, _nao, _nao);  // alpha-beta
    }
  }
}
