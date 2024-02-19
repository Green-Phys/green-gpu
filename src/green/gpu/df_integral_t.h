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

#ifndef GREEN_GPU_DFINTEGRAL_H
#define GREEN_GPU_DFINTEGRAL_H


#include <hdf5.h>
#include <hdf5_hl.h>

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include "base_integral.h"
#include <green/symmetry/symmetry.h>

#include "df_integral_types_e.h"

namespace green::gpu {

/**
 * @brief Integral class read Density fitted 3-center integrals from a HDF5 file, given by the path argument
 */
  class df_integral_t : public base_integral {
    using bz_utils_t = symmetry::brillouin_zone_utils<symmetry::inv_symm_op>;
// prefixes for hdf5
    const std::string rval_ = "VQ";
    const std::string ival_ = "ImVQ";
    const std::string corr_val_ = "EW";
    const std::string corr_bar_val_ = "EW_bar";
  public:
    using MatrixXcd = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXcf = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXd  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    df_integral_t(const std::string & path, int nao, int nk, int NQ, const bz_utils_t & bz_utils, integral_reading_type reading_type = chunks) :
                  base_integral(path, nk), _vij_Q(1, NQ, nao, nao),
                  _k0(-1), _current_chunk(-1), _chunk_size(0), _bz_utils(bz_utils), _reading_type(reading_type) {
      hid_t file = H5Fopen((path + "/meta.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      if (H5LTread_dataset_long(file,"chunk_size",&_chunk_size) < 0)
        throw std::logic_error("Fails on reading chunk_size.");
      H5Fclose(file);
      _vij_Q.resize(_chunk_size, NQ, nao, nao);
    }


    virtual ~df_integral_t() {}

    /**
     * Read next part of the interaction integral from
     * @param k1
     * @param k2
     */
    void read_integrals(size_t k1, size_t k2) {
      assert(k1>=0);
      assert(k2>=0);
      // Find corresponding index for k-pair (k1,k2). Only k-pair with k1 > k2 will be stored.
      size_t idx = (k1 >= k2)? k1*(k1+1)/2 + k2 : k2*(k2+1)/2 + k1; // k-pair = (k1, k2) or (k2, k1)
      // Corresponding symmetry-related k-pair
      if (_bz_utils.symmetry().conj_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().conj_kpair_list()[idx];
      } else if (_bz_utils.symmetry().trans_kpair_list()[idx] != idx){
        idx = _bz_utils.symmetry().trans_kpair_list()[idx];
      }
      long idx_red = _bz_utils.symmetry().irre_pos_kpair(idx);
      if( (idx_red/_chunk_size) == _current_chunk) return; // we have data cached

      _current_chunk = idx_red/_chunk_size;

      size_t c_id = _current_chunk * _chunk_size;
      read_a_chunk(c_id, _vij_Q);
    }
    /**
     * Read the entire 3-indx Coulomb intergrals without MPI-parallel.
     * @param type - flaot/double
     * @param Vk1k2_Qij - A pointer to 3-index Coulomb integrals
     */
    template<typename type>
    void read_entire(std::complex<type> *Vk1k2_Qij) {
      const int NQ = _vij_Q.shape()[1];
      const int nao = _vij_Q.shape()[2];
      size_t num_kpair_stored = _bz_utils.symmetry().num_kpair_stored();
      size_t last_chunk_id = (num_kpair_stored/_chunk_size)*_chunk_size;
      ztensor<4> V_Qij_chunk(_chunk_size, NQ, nao, nao);
      for (std::size_t c_id = 0; c_id < num_kpair_stored; c_id+=_chunk_size) {
        size_t shift = c_id*NQ*nao*nao;
        size_t element_counts = (c_id!=last_chunk_id)? _chunk_size*NQ*nao*nao : (num_kpair_stored-c_id)*NQ*nao*nao;

        read_a_chunk(c_id, V_Qij_chunk);

        Complex_DoubleToType(V_Qij_chunk.data(), Vk1k2_Qij+shift, element_counts);
      }
    }

    /**
     * Read the entire 3-index Coulomb integrals collaboratively within a node.
     * @param type - flaot/double
     * @param Vk1k2_Qij - A pointer to 3-index Coulomb integrals
     * @param intranode_rank - local rank in the intranode communicator
     * @param processes_per_node - size of the intranode communicator
     */
    template<typename type>
    void read_entire(std::complex<type> *Vk1k2_Qij, int intranode_rank, int processes_per_node) {
      const int NQ = _vij_Q.shape()[1];
      const int nao = _vij_Q.shape()[2];
      size_t num_kpair_stored = _bz_utils.symmetry().num_kpair_stored();
      size_t number_of_chunks = (num_kpair_stored % _chunk_size == 0)? num_kpair_stored/_chunk_size : num_kpair_stored/_chunk_size + 1;
      size_t last_chunk_id = (num_kpair_stored/_chunk_size)*_chunk_size;
      // Will deal with leftovers later
      size_t chunks_per_process = number_of_chunks/processes_per_node;
      size_t c_id_rank_shift = intranode_rank*chunks_per_process;

      for (std::size_t c = 0; c < chunks_per_process; c++) {
        size_t c_id = (c_id_rank_shift + c) * _chunk_size;
        size_t shift = c_id*NQ*nao*nao;
        size_t element_counts = (c_id!=last_chunk_id)? _chunk_size*NQ*nao*nao : (num_kpair_stored-c_id)*NQ*nao*nao;

        read_a_chunk(c_id, _vij_Q);

        Complex_DoubleToType(_vij_Q.data(), Vk1k2_Qij+shift, element_counts);
      }

      // Deal with the remaining chunks
      size_t chunks_leftover = number_of_chunks - processes_per_node * chunks_per_process;
      // Each process deal with one leftover chunk. This will not be triggered if chunks_leftover = 0.
      if (intranode_rank < chunks_leftover) {
        size_t c_id = (processes_per_node * chunks_per_process + intranode_rank) * _chunk_size;
        size_t element_counts = (c_id!=last_chunk_id)? _chunk_size*NQ*nao*nao : (num_kpair_stored-c_id)*NQ*nao*nao;
        size_t shift = c_id*NQ*nao*nao;
        read_a_chunk(c_id, _vij_Q);
        Complex_DoubleToType(_vij_Q.data(), Vk1k2_Qij+shift, element_counts);
      }
    }

    void read_a_chunk(size_t c_id, ztensor<4> &V_buffer) {
      std::string fname = base_path() + "/" + rval_ + "_" + std::to_string(c_id) + ".h5";
      hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

      if (H5LTread_dataset_double(file, ("/" + std::to_string(c_id)).c_str(), reinterpret_cast<double *>(V_buffer.data())) < 0) {
        throw std::runtime_error("failure reading VQij with chunk id = " + std::to_string(c_id));
      }
      H5Fclose(file);
    }

    void Complex_DoubleToType(const std::complex<double>* in, std::complex<double>* out, size_t size) {
      memcpy(out, in, size*sizeof(std::complex<double>));
    }

    void Complex_DoubleToType(const std::complex<double>* in, std::complex<float>* out, size_t size) {
      for (int i = 0; i < size; ++i) {
        out[i] = static_cast<std::complex<float> >(in[i]);
      }
    }

    /**
     * read next part of the G=0 correction to interaction integral for the specific k-point
     * @param file - file to be used
     * @param k - k-point
     */
    void read_correction(int k) {
      _v0ij_Q.reshape(_vij_Q.shape()[1], _vij_Q.shape()[2], _vij_Q.shape()[3]);
      _v_bar_ij_Q.reshape(_vij_Q.shape()[1], _vij_Q.shape()[2], _vij_Q.shape()[3]);
      // avoid unnecessary reading
      if (k == _k0) {
        // we have data cached
        return;
      }
      _k0 = k;
      std::string inner = std::to_string(_current_chunk * _chunk_size);
      std::string fname = base_path() + "/" + corr_val_ + ".h5";
      hid_t file = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

      // Construct integral dataset name
      std::string dsetnum = corr_val_ + "/" + std::to_string(k);
      // read data
      if (H5LTread_dataset_double(file, dsetnum.c_str(), reinterpret_cast<double *>(_v0ij_Q.data())))
        throw std::logic_error("Fails on reading ewald corrections.");
      // Construct integral dataset name
      dsetnum = corr_bar_val_ + "/" + std::to_string(k);
      // read data
      if (H5LTread_dataset_double(file, dsetnum.c_str(), reinterpret_cast<double *>(_v_bar_ij_Q.data())) < 0)
        throw std::logic_error("Fails on reading ewald corrections.");;
      H5Fclose(file);
    };

    /**
     * Determine the type of symmetries for the integral based on the current k-points
     *
     * @param k1 incomming k-point
     * @param k2 outgoing k-point
     * @return A pair of sign and type of applied symmetry
     */
    std::pair<int, integral_symmetry_type_e> v_type(size_t k1, size_t k2) {
      size_t idx = (k1 >= k2)? k1*(k1+1)/2 + k2 : k2*(k2+1)/2 + k1; // k-pair = (k1, k2) or (k2, k1)
      // determine sign
      int sign = (k1 >= k2)? 1 : -1;
      // determine applied symmetry type
      // by default no symmetries applied
      integral_symmetry_type_e symmetry_type = direct;
      if (_bz_utils.symmetry().conj_kpair_list()[idx] != idx) {
        symmetry_type = conjugated;
      } else if (_bz_utils.symmetry().trans_kpair_list()[idx] != idx) {
        symmetry_type = transposed;
      }
      return std::make_pair(sign, symmetry_type);
    }

    /**
     * Extract V(Q, i, j) with given (k1, k2) from chunks of integrals (_vij_Q)
     * @tparam prec
     * @param vij_Q_k1k2
     * @param k1
     * @param k2
     */
    template<typename prec>
    void symmetrize(tensor<prec, 3> &vij_Q_k1k2, const size_t k1, const size_t k2) {
      int k1k2_wrap = wrap(k1, k2);
      std::pair<int, integral_symmetry_type_e> vtype = v_type(k1, k2);
      int NQ = _vij_Q.shape()[1];
      if (vtype.first < 0) {
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(vij_Q_k1k2(Q)) = matrix(_vij_Q(k1k2_wrap, Q)).transpose().conjugate().cast<prec>();
        }
      } else {
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(vij_Q_k1k2(Q)) = matrix(_vij_Q(k1k2_wrap, Q)).cast<prec>();
        }
      }
      if (vtype.second == conjugated) { // conjugate
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(vij_Q_k1k2(Q)) = matrix(vij_Q_k1k2(Q)).conjugate();
        }
      } else if (vtype.second == transposed) { // transpose
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(vij_Q_k1k2(Q)) = matrix(vij_Q_k1k2(Q)).transpose().eval();
        }
      }
    }

    /**
     * Extract V(Q, i, j) with given (k1, k2) in precision "prec" from the entire integrals (Vk1k2_Qij)
     * @param Vk1k2_Qij
     * @param V
     * @param k1
     * @param k2
     */
    template<typename prec>
    void symmetrize(std::complex<double> *Vk1k2_Qij, tensor<prec, 3> &V, const int k1, const int k2) {
      int k1k2_wrap = wrap(k1, k2, as_a_whole);
      std::pair<int, integral_symmetry_type_e> vtype = v_type(k1, k2);
      size_t NQ    = V.shape()[0];
      size_t nao   = V.shape()[1];
      size_t shift = k1k2_wrap*NQ*nao*nao;
      size_t element_counts_V  = NQ*nao*nao;
      ztensor<3> V_double_buffer(NQ, nao, nao);
      memcpy(V_double_buffer.data(), Vk1k2_Qij+shift, element_counts_V*sizeof(std::complex<double>));
      if (vtype.first < 0) {
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(V(Q)) = matrix(V_double_buffer(Q)).transpose().conjugate().eval().cast<prec>();
        }
      } else {
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(V(Q)) = matrix(V_double_buffer(Q)).cast<prec>();
        }
      }
      if (vtype.second == conjugated) { // conjugate
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(V(Q)) = matrix(V(Q)).conjugate();
        }
      } else if (vtype.second == transposed) { // transpose
        for (int Q = 0; Q < NQ; ++Q) {
          matrix(V(Q)) = matrix(V(Q)).transpose().eval();
        }
      }
    }

    const ztensor<4> &vij_Q() const {
      return _vij_Q;
    }
    const ztensor<3> &v0ij_Q() const {
      return _v0ij_Q;
    }
    const ztensor<3> &v_bar_ij_Q() const {
      return _v_bar_ij_Q;
    }

    int wrap(int k1, int k2, integral_reading_type read_type = chunks) {
      size_t idx = (k1 >= k2)? k1*(k1+1)/2 + k2 : k2*(k2+1)/2 + k1; // k-pair = (k1, k2) or (k2, k1)
      // determine type
      if (_bz_utils.symmetry().conj_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().conj_kpair_list()[idx];
      } else if (_bz_utils.symmetry().trans_kpair_list()[idx] != idx) {
        idx = _bz_utils.symmetry().trans_kpair_list()[idx];
      }
      int idx_red = _bz_utils.symmetry().irre_pos_kpair(idx);
      return (read_type==chunks)? idx_red % _chunk_size : idx_red;
    }

    void reset() {
      _current_chunk = -1;
      _k0 = -1;
    }

  private:
    // Coulomb integrals stored in density fitting format
    ztensor<4> _vij_Q;
    // G=0 correction to coulomb integral stored in density fitting format
    ztensor<3> _v0ij_Q;
    ztensor<3> _v_bar_ij_Q;

    bool _exch;
    // current leading index
    int _k0;
    long _current_chunk;
    long _chunk_size;
    const bz_utils_t &_bz_utils;

    integral_reading_type _reading_type;
  };

}

#endif //GREEN_GPU_DFINTEGRAL_H
