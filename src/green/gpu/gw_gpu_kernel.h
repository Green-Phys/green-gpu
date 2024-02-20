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

#ifndef GREEN_GPU_GW_GPU_KERNEL_H
#define GREEN_GPU_GW_GPU_KERNEL_H

#include <green/grids/transformer_t.h>
#include <green/params/params.h>
#include <green/symmetry/symmetry.h>
#include <green/utils/mpi_shared.h>
#include <green/utils/mpi_utils.h>
#include <green/utils/timing.h>

#include <array>

#include "common_defs.h"
#include "df_integral_t.h"
#include "gpu_kernel.h"

namespace green::gpu {
  /**
   * @brief cuda GW Solver class performs self-energy calculation by means of GW approximation using density fitting
   */
  class gw_gpu_kernel : public gpu_kernel {
  public:
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
    using G_type     = utils::shared_object<ztensor<5>>;
    using St_type    = utils::shared_object<ztensor<5>>;

    /**
     * Class constructor
     *
     * @param comm     -- global communicator
     * @param p        -- simulation parameters
     * @param ft       -- Fourier transformer between imaginary time and frequency axis
     * @param Gk       -- Lattice Green's function (tau, ns, nk, nao, nao)
     * @param Sigma    -- Lattice self-energy (tau, ns, nk, nao, nao)
     * @param bz_utils -- Brillouin zone utilities
     */
    gw_gpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const grids::transformer_t& ft,
                  const bz_utils_t& bz_utils, int verbose = 1) :
        gpu_kernel(p, nao, nso, ns, NQ, bz_utils), _beta(p["BETA"]), _nts(ft.sd().repn_fermi().nts()),
        _nts_b(ft.sd().repn_bose().nts()), _ni(ft.sd().repn_fermi().ni()), _ni_b(ft.sd().repn_bose().ni()),
        _nw(ft.sd().repn_fermi().nw()), _nw_b(ft.sd().repn_bose().nw()), _sp(p["P_sp"].as<bool>() && p["Sigma_sp"].as<bool>()),
        _ft(ft), _path(p["dfintegral_file"])
    //_naosq(p.nao*p.nao), _nao3(p.nao*p.nao*p.nao),
    {
      // Check if nts is an even number since we will take the advantage of Pq0(beta-t) = Pq0(t) later
      if (_nts % 2 != 0) throw std::runtime_error("Number of tau points should be even number");

      if (verbose > 0) {
        GW_complexity_estimation();
      }
      init_events();
    }

    void     solve(G_type& g, St_type& sigma_tau);

    virtual ~gw_gpu_kernel() = default;

    void     gw_innerloop(G_type& g, St_type& sigma_tau);

  protected:
    void GW_check_devices_free_space();

    /*
     * Read a chunk of Coulomb integral with given (k[0], k[3]) k-pair
     */
    void read_next(const std::array<size_t, 4>& k);

  private:
    void GW_complexity_estimation();

    template <typename prec>
    void compute_gw_selfenergy(G_type& g, St_type& sigma_tau);

    void copy_Gk(const ztensor<5>& _G_tskij_host, tensor<std::complex<double>, 4>& Gk_stij, int k, bool minus_t);
    void copy_Gk(const ztensor<5>& _G_tskij_host, tensor<std::complex<float>, 4>& Gk_stij, int k, bool minus_t);

    void init_events() {
      gw_statistics.add("Initialization");
      gw_statistics.add("GW_loop");
      gw_statistics.add("read");
      gw_statistics.add("selfenergy_reduce");
      gw_statistics.add("total");
    }

  protected:
    double                      _beta;
    size_t                      _nts;
    size_t                      _nts_b;
    size_t                      _ni;
    size_t                      _ni_b;
    size_t                      _nw;
    size_t                      _nw_b;

    const grids::transformer_t& _ft;

    size_t                      _nt_batch;

    const std::string           _path;
    bool                        _sp;

    int                         _nqkpt;

    double                      _flop_count;

    utils::timing               gw_statistics;
  };

}  // namespace green::gpu

#endif  // GREEN_GPU_GW_GPU_KERNEL_H
