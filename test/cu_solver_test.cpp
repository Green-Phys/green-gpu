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

void solve_gw(const std::string& input, const std::string& int_f, const std::string& data) {
  auto        p           = green::params::params("DESCR");
  std::string input_file  = TEST_PATH + input;
  std::string df_int_path = TEST_PATH + int_f;
  std::string test_file   = TEST_PATH + data;
  std::string grid_file   = GRID_PATH + "/ir/1e4.h5"s;
  std::string args =
      "test --restart 0 --itermax 1 --E_thr 1e-13 --mixing_type SIGMA_DAMPING --damping 0.8 --input_file=" + input_file +
      " --BETA 100 --grid_file=" + grid_file + " --dfintegral_file=" + df_int_path +
      " --cuda_low_gpu_memory false --cuda_low_cpu_memory false";
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
  green::gpu::ztensor<4>                 tmp;
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

  green::gpu::gw_gpu_kernel solver(p, nao, nso, ns, NQ, ft, bz, 10);
  solver.solve(G_shared, S_shared);
  REQUIRE_THAT(S_shared.object(), IsCloseTo(S_shared_tst.object(), 1e-6));
}

TEST_CASE("GPU Solver") {
  SECTION("GW") {
    solve_gw("/GW/input.h5", "/GW/df_int", "/GW/data.h5");
  }

}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
