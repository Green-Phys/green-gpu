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

#ifndef GREEN_GPU_HF_GPU_KERNEL_H
#define GREEN_GPU_HF_GPU_KERNEL_H

#include <green/params/params.h>
#include <green/symmetry/symmetry.h>
#include <mpi.h>

#include "cuda_check.h"
#include "cuda_common.h"
#include "df_integral_t.h"
#include "green/gpu/gpu_kernel.h"
#include "green/utils/mpi_shared.h"
#include "green/utils/mpi_utils.h"
#include "green/utils/timing.h"

namespace green::gpu {
  /**
   * @brief cuda HF kernel class
   */
  class hf_gpu_kernel : public gpu_kernel {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;

  public:
    hf_gpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, double madelung,
                  const bz_utils_t& bz_utils, const ztensor<4>& S_k, int verbose = 1) :
        gpu_kernel(p, nao, nso, ns, NQ, bz_utils), _madelung(madelung), _S_k(S_k), _path(p["dfintegral_hf_file"]) {
      if (verbose) HF_complexity_estimation();
    }
    ~            hf_gpu_kernel() override = default;
    ztensor<4>   solve(const ztensor<4>& dm);
    void         get_iter(int& iter) {}

    virtual void set_shared_Coulomb() {
      if (_coul_int_reading_type == as_a_whole) {
        hf_statistics.start("Read");
        // Always read Coulomb integrals in double precision and cast them to single precision whenever needed
        read_entire_Coulomb_integrals(&_Vk1k2_Qij);
        hf_statistics.end();
      } else {
        if (!utils::context.global_rank) std::cout << "Will read Coulomb integrals from chunks." << std::endl;
      }
      MPI_Barrier(utils::context.global);
    }
    /**
     * Determine whether anything will be executed in single precision
     * @param run_sp - [INPUT] Whether to switch to single precision
     */
    virtual void single_prec_run(bool run_sp) { std::cerr << "Single precision is not implemented for cuda HF." << std::endl; }

    virtual const std::string name() const { return "cuda Hartree-Fock"; }

  protected:
    void              HF_check_devices_free_space();

    void              read_exchange_VkQij(int k, int k2, std::complex<double>* Vk1k2_Qij, ztensor<4>& V_kbatchQij);
    void              read_exchange_VkQij(int k, int k2, ztensor<4>& V_kbatchQij);

    double            _madelung;
    const ztensor<4>& _S_k;
    std::string       _path;
    utils::timing     hf_statistics;

  private:
    void   HF_complexity_estimation();

    void   get_dm_fbz(ztensor<4>& dm_fbz, const ztensor<4>& dm);

    void   compute_exchange_selfenergy(ztensor<4>& new_Fock, const ztensor<4>& dm);
    void   compute_direct_selfenergy(ztensor<4>& F, const ztensor<4>& dm);
    void   add_Ewald(ztensor<4>& new_Fock, const ztensor<4>& dm, const ztensor<4>& S, double madelung);

    double _hf_total_flops{};
  };

}  // namespace green::gpu

#endif  // GREEN_GPU_HF_GPU_KERNEL_H