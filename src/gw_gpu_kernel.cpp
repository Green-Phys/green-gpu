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
#include <green/gpu/cuda_common.h>
#include <green/gpu/gw_gpu_kernel.h>
#include <green/gpu/df_integral_t.h>

namespace green::gpu {
    void gw_gpu_kernel::GW_complexity_estimation() {
      // Calculate the complexity of GW
      double NQsq=(double)_NQ*_NQ;
      //first set of matmuls
      double flop_count_firstmatmul= _ink*_nk*_ns*_nts/2.*(
		      matmul_cost(_nao*_NQ, _nao, _nao) //X1_t_mQ = G_t_p * V_pmQ;
		      +matmul_cost(_NQ*_nao, _nao, _nao)//X2_Pt_m = (V_Pt_n)* * G_m_n;
		      +matmul_cost(_NQ, _naosq, _NQ)    //Pq0_QP=X2_Ptm Q1_tmQ
		      );
      double flop_count_fourier=_ink*(matmul_cost(NQsq, _nts, _nw_b)+matmul_cost(NQsq, _nw_b, _nts)); //Fourier transform forward and back
      double flop_count_solver=2./3.*_ink*_nw_b*(NQsq*_NQ+NQsq); //approximate LU and backsubst cost (note we are doing cholesky which is cheaper
      //secondset of matmuls
      double flop_count_secondmatmul=_ink*_nk*_ns*_nts*(
		      matmul_cost(_NQ*_nao, _nao, _nao) //Y1_Qin = V_Qim * G1_mn;
		      +matmul_cost(_naosq, _NQ, _NQ)    //Y2_inP = Y1_Qin * Pq_QP
		      +matmul_cost(_nao, _NQ*_nao, _nao)//Sigma_ij = Y2_inP V_nPj
		      );
      _flop_count= flop_count_firstmatmul+flop_count_fourier+flop_count_solver+flop_count_secondmatmul;

      if (!utils::context.global_rank && _verbose > 1) {
        std::cout << "############ Total GW Operations per Iteration ############" << std::endl;
        std::cout << "Total:         " << _flop_count << std::endl;
        std::cout << "First matmul:  " << flop_count_firstmatmul << std::endl;
        std::cout << "Fourier:       " << flop_count_fourier << std::endl;
        std::cout << "Solver:        " << flop_count_solver << std::endl;
        std::cout << "Second matmul: " << flop_count_secondmatmul << std::endl;
        std::cout << "###########################################################" << std::endl;
      }
    }

    void gw_gpu_kernel::solve(G_type& g, St_type& sigma_tau) {
      MPI_Datatype dt_matrix = utils::create_matrix_datatype<std::complex<double>>(_nso*_nso);
      MPI_Op matrix_sum_op = utils::create_matrix_operation<std::complex<double>>();
      statistics.start("total");
      statistics.start("Initialization");
      sigma_tau.fence();
      if (!utils::context.node_rank) sigma_tau.object().set_zero();
      sigma_tau.fence();
      setup_MPI_structure();
      _coul_int = new df_integral_t(_path, _nao, _nk, _NQ, _bz_utils);
      MPI_Barrier(utils::context.global);
      set_shared_Coulomb();
      statistics.end();
      update_integrals(_coul_int, statistics);
      // Only those processes assigned with a device will be involved in GW self-energy calculation
      if (_devices_comm != MPI_COMM_NULL) {
        gw_innerloop(g, sigma_tau);
      }
      MPI_Barrier(utils::context.global);
      sigma_tau.fence();
      if (!utils::context.node_rank) {
        if (_devices_comm != MPI_COMM_NULL) statistics.start("selfenergy_reduce");
        utils::allreduce(MPI_IN_PLACE, sigma_tau.object().data(), sigma_tau.object().size()/(_nso*_nso), dt_matrix, matrix_sum_op, utils::context.internode_comm);
        sigma_tau.object() /= (_nk);
        if (_devices_comm != MPI_COMM_NULL) statistics.end();
      }
      sigma_tau.fence();
      MPI_Barrier(utils::context.global);
      statistics.end();
      statistics.print(utils::context.global);

      clean_MPI_structure();
      clean_shared_Coulomb();
      delete _coul_int;
      MPI_Barrier(utils::context.global);
      MPI_Type_free(&dt_matrix);
      MPI_Op_free(&matrix_sum_op);
    }

    void gw_gpu_kernel::gw_innerloop(G_type& g, St_type& sigma_tau) {
      if (!_sp) {
        compute_gw_selfenergy<double>(g, sigma_tau);
      } else {
        compute_gw_selfenergy<float>(g, sigma_tau);
      }
    }

    template<typename prec>
    void gw_gpu_kernel::compute_gw_selfenergy(G_type& g, St_type& sigma_tau) {
      // check devices' free space and space requirements
      GW_check_devices_free_space();
      statistics.start("Initialization");
      cugw_utils<prec> cugw(_nts, _nt_batch, _nw_b, _ns, _nk, _ink, _nqkpt, _NQ, _nao, g.object(), _low_device_memory,
                            _ft.Ttn_FB(), _ft.Tnt_BF(), _cuda_lin_solver, utils::context.global_rank, utils::context.node_rank,
                            _devCount_per_node);
      statistics.end();
      // As we move evaluation into a GPU
      irre_pos_callback irre_pos = [&](size_t k) -> size_t {return _bz_utils.symmetry().reduced_to_full()[k];};
      mom_cons_callback mom_cons = [&](const std::array<size_t, 3> &k123) -> const std::array<size_t, 4> {return _bz_utils.momentum_conservation(k123);};
      gw_reader1_callback<prec> r1 = [&](int k, int k1, int k_reduced_id, int k1_reduced_id, const std::array<size_t, 4>& k_vector,
                                         tensor<std::complex<prec>,3>& V_Qpm, std::complex<double> *Vk1k2_Qij,
                                         tensor<std::complex<prec>,4>&Gk_smtij, tensor<std::complex<prec>,4>&Gk1_stij,
                                         bool need_minus_k, bool need_minus_k1) {
        statistics.start("read");
        int q = k_vector[2];
        if (_coul_int_reading_type == chunks) {
          read_next(k_vector);
          _coul_int->symmetrize(V_Qpm, k, k1);
        } else {
          _coul_int->symmetrize(Vk1k2_Qij, V_Qpm, k, k1);
        }
        if (_low_device_memory) {
          copy_Gk(g.object(), Gk_smtij, k_reduced_id, true);
          copy_Gk(g.object(), Gk1_stij, k1_reduced_id, false);
        }
        statistics.end();
      };
      gw_reader2_callback<prec> r2 = [&](int k, int k1, int k1_reduced_id, const std::array<size_t, 4>& k_vector,
                                        tensor<std::complex<prec>,3>& V_Qim, std::complex<double> *Vk1k2_Qij,
                                        tensor<std::complex<prec>,4>&Gk1_stij,
                                        bool need_minus_k1) {
        statistics.start("read");
        int q = k_vector[1];
        if (_coul_int_reading_type == chunks) {
          read_next(k_vector);
          _coul_int->symmetrize(V_Qim, k, k1);
        } else {
          _coul_int->symmetrize(Vk1k2_Qij, V_Qim, k, k1);
        }
        if (_low_device_memory) {
          copy_Gk(g.object(), Gk1_stij, k1_reduced_id, false);
        }
        statistics.end();
      };

      // Since all process in _devices_comm will write to the self-energy simultaneously,
      // instaed of adding locks in cugw.solve(), we allocate private _Sigma_tskij_local_host
      // and do MPIAllreduce on CPU later on. Since the number of processes with a GPU is very
      // limited, the additional memory overhead is fairly limited.
      ztensor<5> Sigma_tskij_host_local(_nts, _ns, _ink, _nao, _nao);
      statistics.start("Solve cuGW");
      cugw.solve(_nts, _ns, _nk, _ink, _nao, _bz_utils.symmetry().reduced_to_full(), _bz_utils.symmetry().full_to_reduced(),
                 _Vk1k2_Qij, Sigma_tskij_host_local, _devices_rank, _devices_size, _low_device_memory, _verbose,
                 irre_pos, mom_cons, r1, r2);
      statistics.end();
      statistics.start("Update Host Self-energy");
      // Copy back to Sigma_tskij_local_host
      MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, sigma_tau.win());
      sigma_tau.object() += Sigma_tskij_host_local;
      MPI_Win_unlock(0, sigma_tau.win());
      statistics.end();
    }
    // Explicit instatiations
    template void gw_gpu_kernel::compute_gw_selfenergy<float>(G_type& g, St_type& sigma_tau);
    template void gw_gpu_kernel::compute_gw_selfenergy<double>(G_type& g, St_type& sigma_tau);

    void gw_gpu_kernel::GW_check_devices_free_space() {
      // check devices' free space and space requirements
      auto prec = std::cout.precision();
      auto flags = std::cout.flags();
      std::cout << std::setprecision(4) << std::boolalpha;
      if (!_devices_rank && _verbose > 1) std::cout << "Economical gpu memory mode: " << _low_device_memory << std::endl;
      std::size_t qpt_size = (!_sp)? gw_qpt<double>::size(_nao, _NQ, _nts, _nw_b) : gw_qpt<float>::size(_nao, _NQ, _nts, _nw_b);
      std::size_t qkpt_size = (!_sp)? gw_qkpt<double>::size(_nao, _NQ, _nts, _nt_batch, _ns) : gw_qkpt<float>::size(_nao, _NQ, _nts, _nt_batch, _ns);
      if (!_devices_rank && _verbose > 1) std::cout << "size of tau batch: " << _nt_batch << std::endl;
      if (!_devices_rank && _verbose > 1) std::cout << "size per qpt: " << qpt_size / (1024 * 1024. * 1024.) << " GB " << std::endl;
      std::size_t available_memory;
      std::size_t total_memory;
      cudaMemGetInfo(&available_memory, &total_memory);
      _nqkpt=std::min(int((available_memory*0.8-qpt_size)/qkpt_size), 16);
      if (!_devices_rank && _verbose > 1) {
        std::cout << "size per qkpt: " << qkpt_size / (1024 * 1024. * 1024.) << " GB " << std::endl;
        std::cout << "available memory: " << available_memory / (1024 * 1024. * 1024.) << " GB " << " of total: "
                  << total_memory / (1024 * 1024. * 1024.) << " GB. " << std::endl;
        std::cout << "can create: " << _nqkpt << " qkpts in parallel" << std::endl;
      }
      if(_nqkpt==0) throw std::runtime_error("not enough memory to create qkpt. Please reduce nt_batch");
      if(_nqkpt==1 && !utils::context.global_rank) std::cerr<<"WARNING: ONLY ONE QKPT CREATED. LIKELY CODE WILL BE SLOW. REDUCE NT_BATCH"<<std::endl;
      // restore std::cout state
      std::cout << std::setprecision(prec);
      std::cout.flags(flags);
    }

    void gw_gpu_kernel::copy_Gk(const ztensor<5> &G_tskij_host, ztensor<4> &Gk_stij, int k, bool minus_t) {
      for (size_t t = 0; t < _nts; ++t) {
        for (size_t s = 0; s < _ns; ++s) {
          size_t shift_st = (s * _nts + t) * _naosq;
          size_t shift_tsk = (((minus_t)? (_nts - 1 - t) : t) * _ns * _ink + s * _ink + k) * _naosq;

          std::memcpy(Gk_stij.data() + shift_st, G_tskij_host.data() + shift_tsk,
                      _naosq * sizeof(std::complex<double>));
        }
      }
    }

    void gw_gpu_kernel::copy_Gk(const ztensor<5> &G_tskij_host, ctensor<4> &Gk_stij, int k, bool minus_t) {
      for (size_t t = 0; t < _nts; ++t) {
        for (size_t s = 0; s < _ns; ++s) {
          size_t shift_st = (s * _nts + t) * _naosq;
          size_t shift_tsk = (((minus_t)? (_nts - 1 - t) : t) * _ns * _ink + s * _ink + k) * _naosq;

          std::complex<float> G_tmp[_naosq];
          Complex_DoubleToFloat(G_tskij_host.data() + shift_tsk, G_tmp, _naosq);
          std::memcpy(Gk_stij.data() + shift_st, G_tmp, _naosq * sizeof(std::complex<float>));
        }
      }
    }

  void gw_gpu_kernel::read_next(const std::array<size_t, 4> &k) {
    // k = (k1, 0, q, k1+q) or (k1, q, 0, k1-q)
    size_t k1 = k[0];
    size_t k1q = k[3];
    _coul_int->read_integrals(k1, k1q);
  }

} // namespace mbpt
