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
#include <green/gpu/hf_gpu_kernel.h>

namespace green::gpu {
  void hf_gpu_kernel::HF_complexity_estimation() {
    // Direct diagram
    double flop_count_direct =
        _ink * _ns * matmul_cost(1, _NQ, _naosq) + matmul_cost(1, _NQ, _ink) + _ink * _ns * matmul_cost(1, _naosq, _NQ);
    // Exchange diagram
    double flop_count_exchange = _ink * _ns * _nk * (matmul_cost(_NQ * _nao, _nao, _nao) + matmul_cost(_nao, _nao, _NQ * _nao)) +
                                 _ink * _ns * matmul_cost(1, _naosq, _nk);
    _hf_total_flops = flop_count_direct + flop_count_exchange;

    if (!utils::context.global_rank && _verbose > 1) {
      std::cout << "############ Total HF Operations per Iteration ############" << std::endl;
      std::cout << "Total:         " << _hf_total_flops << std::endl;
      std::cout << "Matmul (Direct diagram):  " << flop_count_direct << std::endl;
      std::cout << "Matmul (Exchange diagram): " << flop_count_exchange << std::endl;
      std::cout << "###########################################################" << std::endl;
    }
  }

  ztensor<4> hf_gpu_kernel::solve(const ztensor<4>& dm) {
    statistics.start("Total");
    statistics.start("Initialization");
    ztensor<4> new_Fock(_ns, _ink, _nao, _nao);
    new_Fock.set_zero();
    setup_MPI_structure();
    _coul_int = new df_integral_t(_path, _nao, _nk, _NQ, _bz_utils);
    MPI_Barrier(utils::context.global);
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
    utils::allreduce(MPI_IN_PLACE, new_Fock.data(), new_Fock.size(), MPI_C_DOUBLE_COMPLEX, MPI_SUM, utils::context.global);
    statistics.end();
    statistics.end();
    statistics.print(utils::context.global);

    clean_MPI_structure();
    clean_shared_Coulomb();
    delete _coul_int;
    MPI_Barrier(utils::context.global);
    return new_Fock;
  }

  void hf_gpu_kernel::compute_exchange_selfenergy(ztensor<4>& new_Fock, const ztensor<4>& dm) {
    statistics.start("Initialization");
    ztensor<4> dm_fbz(_ns, _nk, _nao, _nao);
    get_dm_fbz(dm_fbz, dm);
    // Also determines _nk_batch
    HF_check_devices_free_space();
    // Each process gets one cuda runner hf_utils
    cuhf_utils hf_utils(_nk, _ink, _ns, _nao, _NQ, _nk_batch, dm_fbz, utils::context.global_rank, utils::context.node_rank,
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
    hf_utils.solve(_Vk1k2_Qij, V_kbatchQij, new_Fock, _nk_batch, _coul_int_reading_type, _devices_rank, _devices_size,
                   _bz_utils.symmetry().reduced_to_full(), r1, r2);
    statistics.end();
  }

  void hf_gpu_kernel::compute_direct_selfenergy(ztensor<4>& F, const ztensor<4>& dm) {
    if (utils::context.global_rank < _ink * _ns) {
      int hf_nprocs = (utils::context.global_size > _ink * _ns) ? _ink * _ns : utils::context.global_size;

      // Direct diagram
      MatrixXcd  X1(_nao, _nao);
      ztensor<3> v(_NQ, _nao, _nao);
      ztensor<2> upper_Coul(_NQ, 1);
      MMatrixXcd X1m(X1.data(), _nao * _nao, 1);
      MMatrixXcd vm(v.data(), _NQ, _nao * _nao);
      for (int ikps = 0; ikps < _ink * _ns; ++ikps) {
        int is    = ikps % _ns;
        int ikp   = ikps / _ns;
        int kp_ir = _bz_utils.symmetry().full_point(ikp);
        if (_coul_int_reading_type == as_a_whole) {
          _coul_int->symmetrize(_Vk1k2_Qij, v, kp_ir, kp_ir);
        } else {
          _coul_int->read_integrals(kp_ir, kp_ir);
          _coul_int->symmetrize(v, kp_ir, kp_ir);
        }

        X1 = CMMatrixXcd(dm.data() + is * _ink * _nao * _nao + ikp * _nao * _nao, _nao, _nao);
        X1 = X1.transpose().eval();
        // (Q, 1) = (Q, ab) * (ab, 1)
        matrix(upper_Coul) += _bz_utils.symmetry().weight()[kp_ir] * vm * X1m;
      }
      upper_Coul /= double(_nk);

      for (int ii = utils::context.global_rank; ii < _ink * _ns; ii += hf_nprocs) {
        int is   = ii / _ink;
        int ik   = ii % _ink;
        int k_ir = _bz_utils.symmetry().full_point(ik);
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

  void hf_gpu_kernel::add_Ewald(ztensor<4>& new_Fock, const ztensor<4>& dm, const ztensor<4>& S, double madelung) {
    if (utils::context.global_rank < _ink * _ns) {
      double prefactor = (_ns == 2) ? 1.0 : 0.5;
      size_t hf_nprocs = (utils::context.global_size > _ink * _ns) ? _ink * _ns : utils::context.global_size;
      for (size_t ii = utils::context.global_rank; ii < _ns * _ink; ii += hf_nprocs) {
        size_t      is = ii / _ink;
        size_t      ik = ii % _ink;
        CMMatrixXcd dmm(dm.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        CMMatrixXcd Sm(S.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        MMatrixXcd  Fm(new_Fock.data() + is * _ink * _nao * _nao + ik * _nao * _nao, _nao, _nao);
        Fm -= prefactor * madelung * Sm * dmm * Sm;
      }
    }
  }

  void hf_gpu_kernel::get_dm_fbz(ztensor<4>& dm_fbz, const ztensor<4>& dm) {
    size_t nknaosq  = _nk * _naosq;
    size_t inknaosq = _ink * _naosq;
    for (int s = 0; s < _ns; ++s) {
      for (int k = 0; k < _nk; ++k) {
        int         k_pos         = _bz_utils.symmetry().full_to_reduced()[k];
        size_t      shift_sk_full = s * nknaosq + k * _naosq;
        size_t      shift_sk      = s * inknaosq + k_pos * _naosq;
        MMatrixXcd  dmm_fbz(dm_fbz.data() + shift_sk_full, _nao, _nao);
        CMMatrixXcd dmm(dm.data() + shift_sk, _nao, _nao);
        if (_bz_utils.symmetry().reduced_to_full()[k_pos] == k) {
          dmm_fbz = dmm;
        } else {
          dmm_fbz = dmm.conjugate();
        }
      }
    }
  }
}
