#ifndef CU_SYMMETRY_TEST_H
#define CU_SYMMETRY_TEST_H

#include <green/gpu/cu_symmetry.h>
#include <complex>

namespace green::gpu {
  // Returns the max absolute error of the GPU symmetry transform vs the reference Fock matrix.
  // prec selects the cu_symmetry<prec> instantiation under test; tolerance at the call site
  // should scale with prec (double: ~1e-4 on DFT Fock data; float: ~1e-2).
  template <class prec>
  double test_symmetry_transform_roundtrip(
      const cu_symmetry_data& sym_data,
      const std::complex<double>* Fock_fbz,  // [ns, nk, nao, nao]
      int ns, int nk, int nao);
}

#endif
