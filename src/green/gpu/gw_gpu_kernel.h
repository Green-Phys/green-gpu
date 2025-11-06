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
     * \brief Initialize GW GPU kernel
     *
     * \param p  -- simulation parameters
     * \param nao -- number of orbitals
     * \param nso -- number of spin-orbitals
     * \param ns -- number of spins
     * \param NQ -- auxiliary basis size
     * \param ft -- Fourier transformer between imaginary time and frequency axis
     * \param bz_utils -- Brillouin zone utilities
     * \param verbose -- print verbose information
     */
    gw_gpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const grids::transformer_t& ft,
                  const bz_utils_t& bz_utils, LinearSolverType cuda_lin_solver, int verbose = 1) :
        gpu_kernel(p, nao, nso, ns, NQ, bz_utils), _beta(p["BETA"]), _nts(ft.sd().repn_fermi().nts()),
        _nts_b(ft.sd().repn_bose().nts()), _ni(ft.sd().repn_fermi().ni()), _ni_b(ft.sd().repn_bose().ni()),
        _nw(ft.sd().repn_fermi().nw()), _nw_b(ft.sd().repn_bose().nw()), _sp(p["P_sp"].as<bool>() && p["Sigma_sp"].as<bool>()),
        _ft(ft), _nt_batch(p["nt_batch"]), _path(p["dfintegral_file"]), _cuda_lin_solver(cuda_lin_solver) {
      // Check if nts is an even number since we will take the advantage of Pq0(beta-t) = Pq0(t) later
      if (_nts % 2 != 0) throw std::runtime_error("Number of tau points should be even number");
    }

    /**
     * \brief For a given Green's function compute GW approximation for the Self-energy
     * \param g Green's function obejct
     * \param sigma_tau Dynamical part of the Self-energy
     */
    void solve(G_type& g, St_type& sigma_tau);

    ~gw_gpu_kernel() override = default;

  protected:
    virtual void gw_innerloop(G_type& g, St_type& sigma_tau) = 0;
    void GW_check_devices_free_space();

    /**
     * \brief Read a chunk of Coulomb integral with given (k[0], k[3]) k-pair
     */
    void read_next(const std::array<size_t, 4>& k);

    /**
     * \brief calculate effective floating points operations per second reached on GPU.
     * This is not representative of the GPU capabilities, but instead, includes read/write overheads.
     */
    void flops_achieved();

    /**
     * \brief print the effective FLOPs achieved for the iteration.
     * 
     */
    void print_effective_flops();

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

    int                         _nqkpt{};

    double                      _flop_count{};
    double                      _eff_flops{};
    LinearSolverType            _cuda_lin_solver;
  };

  class scalar_gw_gpu_kernel : public gw_gpu_kernel {
  public:
    /**
     * \brief Initialize GW GPU kernel
     *
     * \param p  -- simulation parameters
     * \param nao -- number of orbitals
     * \param nso -- number of spin-orbitals
     * \param ns -- number of spins
     * \param NQ -- auxiliary basis size
     * \param ft -- Fourier transformer between imaginary time and frequency axis
     * \param bz_utils -- Brillouin zone utilities
     * \param verbose -- print verbose information
     */
    scalar_gw_gpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const grids::transformer_t& ft,
                  const bz_utils_t& bz_utils, LinearSolverType cuda_lin_solver, int verbose = 1) : gw_gpu_kernel(p, nao, nso, ns, NQ, ft, bz_utils, cuda_lin_solver, verbose) {
      if (verbose > 0) {
        complexity_estimation();
      }
    }

    ~scalar_gw_gpu_kernel() override = default;

  protected:
    void gw_innerloop(G_type& g, St_type& sigma_tau) override;
  private:
    /**
     * @brief Calculate and print complexity estimation for each device
     */
    void complexity_estimation();

    template <typename prec>
    void compute_gw_selfenergy(G_type& g, St_type& sigma_tau);

    void copy_Gk(const ztensor<5>& G_tskij_host, tensor<std::complex<double>, 4>& Gk_stij, int k, bool minus_t);
    void copy_Gk(const ztensor<5>& G_tskij_host, tensor<std::complex<float>, 4>& Gk_stij, int k, bool minus_t);
  };

  class x2c_gw_gpu_kernel : public gw_gpu_kernel {
  public:
    /**
     * \brief Initialize GW GPU kernel
     *
     * \param p  -- simulation parameters
     * \param nao -- number of orbitals
     * \param nso -- number of spin-orbitals
     * \param ns -- number of spins
     * \param NQ -- auxiliary basis size
     * \param ft -- Fourier transformer between imaginary time and frequency axis
     * \param bz_utils -- Brillouin zone utilities
     * \param verbose -- print verbose information
     */
    x2c_gw_gpu_kernel(const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const grids::transformer_t& ft,
                  const bz_utils_t& bz_utils, LinearSolverType cuda_lin_solver, int verbose = 1) : gw_gpu_kernel(p, nao, nso, ns, NQ, ft, bz_utils, cuda_lin_solver, verbose) {
      if (!_low_device_memory && !utils::context.global_rank && _verbose > 2) std::cout<<"X2C GW force using low device memory implementation"<<std::endl;
      _low_device_memory = true;
      if (verbose > 0) {
        complexity_estimation();
      }
    }

    ~x2c_gw_gpu_kernel() override = default;
  protected:
    void gw_innerloop(G_type& g, St_type& sigma_tau) override;
  private:
    /**
     * @brief Calculate and print complexity estimation for each device
     */
    void complexity_estimation();

    template<typename prec>
    void compute_2c_gw_selfenergy(G_type& g, St_type& sigma_tau);

    void copy_Gk_2c(const ztensor<5> &G_tskij_host, tensor<std::complex<double>,4> &Gk_4tij, int k, bool need_minus_k, bool minus_t);
    void copy_Gk_2c(const ztensor<5> &G_tskij_host, tensor<std::complex<float>,4> &Gk_4tij, int k, bool need_minus_k, bool minus_t);
  };

}  // namespace green::gpu

#endif  // GREEN_GPU_GW_GPU_KERNEL_H
