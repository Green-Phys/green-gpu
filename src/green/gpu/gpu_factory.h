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

#ifndef GREEN_GPU_FACTORY_H
#define GREEN_GPU_FACTORY_H

#include <green/gpu/gw_gpu_kernel.h>
#include <green/gpu/hf_gpu_kernel.h>

namespace green::gpu {
  using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
  using x_type     = ztensor<4>;
  using G_type     = utils::shared_object<ztensor<5>>;

  /**
   * \brief Create a custom HF kernel extension that uses GPU back-end
   *
   * \param X2C Using X2C spin-orbit coupling
   * \param p simulation parameters
   * \param nao number of orbitals
   * \param nso number of spin orbitals
   * \param ns number of spins
   * \param NQ auxiliary basis size
   * \param madelung madelung constant
   * \param bz_utils Brillouin zone utilities
   * \param S_k overlap matrix
   * \return pair of shared pointer to kernel and callback function for kernel evaluation
   */
  inline std::tuple<std::shared_ptr<void>, std::function<x_type(const x_type&)>> custom_hf_kernel(
      bool X2C, const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, double madelung,
      const bz_utils_t& bz_utils, const ztensor<4>& S_k) {
    if(X2C) {
      std::shared_ptr<void> kernel(new x2c_hf_gpu_kernel(p, nao, nso, ns, NQ, madelung, bz_utils, S_k));
      std::function         callback = [kernel](const x_type& dm) -> x_type {
        return static_cast<x2c_hf_gpu_kernel*>(kernel.get())->solve(dm);
      };
      return std::tuple{kernel, callback};
    }
    std::shared_ptr<void> kernel(new scalar_hf_gpu_kernel(p, nao, nso, ns, NQ, madelung, bz_utils, S_k));
    std::function         callback = [kernel](const x_type& dm) -> x_type {
      return static_cast<scalar_hf_gpu_kernel*>(kernel.get())->solve(dm);
    };
    return std::tuple{kernel, callback};
  }

  /**
   * \brief Create a custom GW kernel extension that uses GPU back-end
   *
   * \param X2C Using X2C spin-orbit coupling
   * \param p simulation parameters
   * \param nao number of orbitals
   * \param nso number of spin-orbitals
   * \param ns number of spins
   * \param NQ auziliary basis size
   * \param ft time/frequency fourier trasform utilities
   * \param bz_utils Brillouin zone utilities
   * \param S_k overlap martix
   * \return pair of shared pointer to kernel and callback function for kernel evaluation
   */
  inline std::tuple<std::shared_ptr<void>, std::function<void(G_type&, G_type&)>> custom_gw_kernel(
      bool X2C, const params::params& p, size_t nao, size_t nso, size_t ns, size_t NQ, const grids::transformer_t& ft,
      const bz_utils_t& bz_utils, const ztensor<4>& S_k) {
    if (X2C) {
      std::shared_ptr<void> kernel(new x2c_gw_gpu_kernel(p, nao, nso, ns, NQ, ft, bz_utils, p["cuda_linear_solver"], p["verbose"]));
      std::function callback = [kernel](G_type& g, G_type& s) { static_cast<x2c_gw_gpu_kernel*>(kernel.get())->solve(g, s); };
      return std::tuple{kernel, callback};
    }
    std::shared_ptr<void> kernel(new scalar_gw_gpu_kernel(p, nao, nso, ns, NQ, ft, bz_utils, p["cuda_linear_solver"], p["verbose"]));
    std::function callback = [kernel](G_type& g, G_type& s) { static_cast<scalar_gw_gpu_kernel*>(kernel.get())->solve(g, s); };
    return std::tuple{kernel, callback};
  }

  /**
   * \brief Add GPU-kernel specific parameters
   * \param p simulation parameters object
   */
  inline void custom_kernel_parameters(params::params& p) {
    p.define<int>("verbose", "Print verbose output.", 0);
    p.define<LinearSolverType>("cuda_linear_solver", "Type of linear solver for Bethe-Salpeter equation (LU or Cholesky).",
                               LinearSolverType::LU);
    p.define<bool>("cuda_low_gpu_memory", "GPU Device has small amount of memory");
    p.define<bool>("cuda_low_cpu_memory", "Host has small amount of memory, we will read Coulomb integrals in chunks");
    p.define<size_t>("nt_batch", "Size of tau batch in cuda GW solver", 1);
  }

}  // namespace green::mbpt

#endif  // GREEN_GPU_FACTORY_H
