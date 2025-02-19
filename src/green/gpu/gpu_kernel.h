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

#ifndef GREEN_GPU_KERNEL_H
#define GREEN_GPU_KERNEL_H

#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>
#include <green/integrals/df_integral_t.h>

#include "common_defs.h"
#include "cuda_check.h"

namespace green::gpu {
  using green::integrals::df_integral_t;
  using green::integrals::integral_reading_type;
  class gpu_kernel {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;

  public:
    gpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const bz_utils_t& bz_utils) :
        _coul_int(nullptr), _nk(bz_utils.nk()), _ink(bz_utils.ink()), _nao(nao), _nso(nso), _ns(ns), _NQ(NQ), _bz_utils(bz_utils),
        _naosq(nao * nao), _nao3(nao * nao * nao), _NQnaosq(NQ * nao * nao), _nk_batch(0), _devices_comm(MPI_COMM_NULL),
        _devices_rank(0), _devices_size(0), _shared_win(MPI_WIN_NULL), _devCount_total(0), _devCount_per_node(0),
        _low_device_memory(p["cuda_low_gpu_memory"]), _verbose(p["verbose"]), _Vk1k2_Qij(nullptr) {
      _verbose_ints = (!utils::context.internode_rank) ? _verbose : 0;
      check_for_cuda(utils::context.global, utils::context.global_rank, _devCount_per_node, _verbose);
      if (p["cuda_low_cpu_memory"].as<bool>()) {
        _coul_int_reading_type = green::integrals::chunks;
      } else {
        _coul_int_reading_type = green::integrals::as_a_whole;
      }
    }
    virtual ~gpu_kernel() { clean_shared_Coulomb(); }

  protected:
    /**
     * \brief Setup device MPI communicator
     */
    void setup_MPI_structure();
    /**
     * \brief Release device MPI communicator
     */
    void clean_MPI_structure();

    /**
     * \brief Allocate shared-memory area for Coloumb integrals if integrals will be read as a whole.
     */
    inline void set_shared_Coulomb() {
      if (_coul_int_reading_type == green::integrals::as_a_whole) {
        statistics.start("Read");
        // Allocate Coulomb integrals in double precision and cast them to single precision whenever needed
        allocate_shared_Coulomb(&_Vk1k2_Qij);
        statistics.end();
      } else {
        if (!utils::context.global_rank && _verbose > 0) std::cout << "Will read Coulomb integrals from chunks." << std::endl;
      }
      MPI_Barrier(utils::context.global);
    }

    /**
     * \brief Release memory and MPI shared window if Coloumb integrals stored in a shared-memory area
     */
    inline void clean_shared_Coulomb() {
      if (_coul_int_reading_type == green::integrals::as_a_whole && _shared_win != MPI_WIN_NULL) {
        MPI_Win_free(&_shared_win);
      }
    }

    /**
     * Read the whole Coulomb integral into a shared memory are
     */
    void update_integrals(df_integral_t* coul_int, utils::timing& statistics) const {
      if (_coul_int_reading_type == green::integrals::as_a_whole) {
        statistics.start("read whole integral");
        MPI_Win_fence(0, _shared_win);
        coul_int->read_entire(_Vk1k2_Qij, utils::context.node_rank, utils::context.node_size);
        MPI_Win_fence(0, _shared_win);
        statistics.end();
      }
    }

    /**
     * Allocate the entire Coulomb integral to an MPI shared-memory area. Collective behavior among node_comm
     */
    template <typename prec>
    void allocate_shared_Coulomb(std::complex<prec>** Vk1k2_Qij) {
      size_t   number_elements    = _bz_utils.symmetry().num_kpair_stored() * _NQ * _naosq;
      MPI_Aint shared_buffer_size = number_elements * sizeof(std::complex<prec>);
      if (!utils::context.global_rank && _verbose > 0) {
        std::cout << std::setprecision(4);
        std::cout << "Reading the entire Coulomb integrals at once. Estimated memory requirement per node = "
                  << (double)shared_buffer_size / 1024 / 1024 / 1024 << " GB." << std::endl;
        std::cout << std::setprecision(15);
      }
      // Collective operations among node_comm
      utils::setup_mpi_shared_memory(Vk1k2_Qij, shared_buffer_size, _shared_win, utils::context.node_comm,
                                     utils::context.node_rank);
    }

  protected:
    df_integral_t*        _coul_int;

    size_t                _nk;
    size_t                _ink;
    size_t                _nao;
    size_t                _nso;
    size_t                _ns;
    size_t                _NQ;

    const bz_utils_t&     _bz_utils;

    const size_t          _naosq;
    const size_t          _nao3;
    const size_t          _NQnaosq;
    size_t                _nk_batch;

    MPI_Comm              _devices_comm;
    int                   _devices_rank;
    int                   _devices_size;
    MPI_Win               _shared_win;

    int                   _devCount_total;
    int                   _devCount_per_node;
    integral_reading_type _coul_int_reading_type;
    bool                  _low_device_memory;
    int                   _verbose;
    int                   _verbose_ints;

    std::complex<double>* _Vk1k2_Qij;
    utils::timing         statistics;
  };

}  // namespace green::gpu

#endif  // GREEN_GPU_KERNEL_H
