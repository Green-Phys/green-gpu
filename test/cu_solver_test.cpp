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

#include <green/gpu/gpu_factory.h>
#include <green/grids.h>

#include "catch2/matchers/catch_matchers.hpp"
#include "tensor_test.h"


#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>


void solve_hf(const std::string& input, const std::string& int_hf, const std::string& data) {
  auto        p           = green::params::params("DESCR");
  std::string input_file  = TEST_PATH + input;
  std::string df_int_path = TEST_PATH + int_hf;
  std::string test_file   = TEST_PATH + data;
  std::string grid_file   = GRID_PATH + "/ir/1e4.h5"s;
  std::string args =
      "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type SIGMA_DAMPING --damping 0.8 --input_file=" + input_file +
      " --BETA 100 --grid_file=" + grid_file + " --dfintegral_hf_file=" + df_int_path +
      " --cuda_low_gpu_memory false --cuda_low_cpu_memory false";
  green::grids::define_parameters(p);
  green::mbpt::custom_kernel_parameters(p);
  green::symmetry::define_parameters(p);
  p.define<std::string>("dfintegral_hf_file", "Path to Hartree-Fock integrals");
  p.define<std::string>("dfintegral_file", "Path to integrals for high orfer theories");
  p.parse(args);
  green::symmetry::brillouin_zone_utils bz(p);
  size_t                                nao, nso, NQ, ns, nk, ink, nts;
  double                                madelung;
  green::gpu::ztensor<4>                tmp;
  {
    green::h5pp::archive ar(input_file);
    ar["params/nao"] >> nao;
    ar["params/nso"] >> nso;
    ar["params/NQ"] >> NQ;
    ar["params/ns"] >> ns;
    ar["params/nk"] >> nk;
    ar["grid/ink"] >> ink;
    ar["HF/madelung"] >> madelung;
    green::gpu::dtensor<5> S_k;
    ar["HF/S-k"] >> S_k;
    ar.close();
    tmp.resize(ns, nk, nso, nso);
    tmp << S_k.view<std::complex<double>>().reshape(ns, nk, nso, nso);
  }
  {
    green::h5pp::archive ar(grid_file);
    ar["fermi/metadata/ncoeff"] >> nts;
    ar.close();
    nts += 2;
  }
  auto G_shared    = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared    = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto Sigma1      = green::gpu::ztensor<4>(ns, ink, nso, nso);
  auto Sigma1_test = green::gpu::ztensor<4>(ns, ink, nso, nso);
  auto Sk          = green::gpu::ztensor<4>(ns, ink, nso, nso);
  for (int is = 0; is < ns; ++is) Sk(is) << bz.full_to_ibz(tmp(is));
  {
    green::h5pp::archive ar(test_file, "r");
    G_shared.fence();
    if (!green::utils::context.node_rank) ar["G_tau"] >> G_shared.object();
    G_shared.fence();
    ar["result/Sigma1"] >> Sigma1_test;
    ar.close();
  }
  double prefactor = (ns == 2 or nao != nso) ? -1.0 : -2.0;
  green::gpu::ztensor<4> dm(ns, nk, nso, nso);
  dm << G_shared.object()(G_shared.object().shape()[0] - 1);
  auto [kernel, solver] = green::mbpt::custom_hf_kernel(nso != nao, p, nao, nso, ns, NQ, madelung, bz, Sk);
  Sigma1 << solver(dm);
  Sigma1 *= prefactor;
  REQUIRE_THAT(Sigma1, IsCloseTo(Sigma1_test));
}


void solve_gw(const std::string& input, const std::string& int_f, const std::string& data, const std::string& lin) {
  auto        p           = green::params::params("DESCR");
  std::string input_file  = TEST_PATH + input;
  std::string df_int_path = TEST_PATH + int_f;
  std::string test_file   = TEST_PATH + data;
  std::string grid_file   = GRID_PATH + "/ir/1e4.h5"s;
  std::string args =
      "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type SIGMA_DAMPING --damping 0.8 --input_file=" + input_file +
      " --BETA 100 --grid_file=" + grid_file + " --dfintegral_file=" + df_int_path +
      " --cuda_low_gpu_memory false --cuda_low_cpu_memory false --cuda_linear_solver=" + lin;
  green::grids::define_parameters(p);
  green::symmetry::define_parameters(p);
  green::mbpt::custom_kernel_parameters(p);
  p.define<std::string>("dfintegral_hf_file", "Path to Hartree-Fock integrals");
  p.define<std::string>("dfintegral_file", "Path to integrals for high orfer theories");
  p.define<bool>("P_sp", "Compute polarization in single precision", false);
  p.define<bool>("Sigma_sp", "Compute self-energy in single precision", false);
  p.parse(args);
  green::symmetry::brillouin_zone_utils bz(p);
  green::grids::transformer_t           ft(p);
  size_t                                NQ, nao, nso, ns, nk, ink, nts;
  green::gpu::ztensor<4>                tmp;
  {
    green::h5pp::archive ar(input_file);
    ar["params/nso"] >> nso;
    ar["params/nao"] >> nao;
    ar["params/NQ"] >> NQ;
    ar["params/ns"] >> ns;
    ar["params/nk"] >> nk;
    ar["grid/ink"] >> ink;
    green::gpu::dtensor<5> S_k;
    ar["HF/S-k"] >> S_k;
    ar.close();
    tmp.resize(ns, nk, nso, nso);
    tmp << S_k.view<std::complex<double>>().reshape(ns, nk, nso, nso);
  }
  {
    green::h5pp::archive ar(grid_file);
    ar["fermi/metadata/ncoeff"] >> nts;
    ar.close();
    nts += 2;
  }
  auto G_shared     = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared     = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto S_shared_tst = green::utils::shared_object(green::gpu::ztensor<5>(nullptr, nts, ns, ink, nso, nso));
  auto Sigma1       = green::gpu::ztensor<4>(ns, ink, nso, nso);
  auto Sk           = green::gpu::ztensor<4>(ns, ink, nso, nso);
  for (int is = 0; is < ns; ++is) Sk(is) << bz.full_to_ibz(tmp(is));
  {
    green::h5pp::archive ar(test_file, "r");
    G_shared.fence();
    if (!green::utils::context.node_rank) ar["G_tau"] >> G_shared.object();
    G_shared.fence();
    S_shared_tst.fence();
    if (!green::utils::context.node_rank) ar["result/Sigma_tau"] >> S_shared_tst.object();
    S_shared_tst.fence();
    ar.close();
  }

  auto [kernel, solver] = green::mbpt::custom_gw_kernel(nso != nao, p, nao, nso, ns, NQ, ft, bz, Sk);
  //green::gpu::scalar_gw_gpu_kernel solver(p, nao, nso, ns, NQ, ft, bz, p["cuda_linear_solver"], 10);
  solver(G_shared, S_shared);
  REQUIRE_THAT(S_shared.object(), IsCloseTo(S_shared_tst.object(), 1e-6));
}

TEST_CASE("GPU Solver") {
  SECTION("GW_LU") {
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "LU");
  }
  SECTION("GW_Cholesky") {
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5", "Cholesky");
  }
  SECTION("GW_X2C") {
    solve_gw("/GW_X2C/input.h5", "/GW_X2C/df_hf_int", "/GW_X2C/data.h5", "LU");
  }

  SECTION("HF") {
    solve_hf("/HF/input.h5", "/HF/df_hf_int", "/HF/data.h5");
  }
  SECTION("HF_X2C") {
    solve_hf("/HF_X2C/input.h5", "/HF_X2C/df_hf_int", "/HF_X2C/data.h5");
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
