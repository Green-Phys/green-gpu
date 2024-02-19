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

#include <green/gpu/gpu_kernel.h>

namespace green::gpu {

  void gpu_kernel::setup_MPI_structure() {
    _devCount_total = (utils::context.node_rank < _devCount_per_node) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &_devCount_total, 1, MPI_INT, MPI_SUM, utils::context.global);
    if (!utils::context.global_rank)
      std::cout << "Your host has " << _devCount_per_node << " devices/node and we'll use " << _devCount_total
                << " devices in total." << std::endl;
    if (_devCount_total > _ink && !utils::context.global_rank) {
      std::cerr << "***Warining***: The maximum number of GPUs to parallel would be " << _ink << " for cuGW and " << _ink
                << " for cuHF. Extra resources would simply be idle." << std::endl;
    }

    utils::setup_devices_communicator(utils::context.global, utils::context.global_rank, utils::context.node_rank, _devCount_per_node, _devCount_total, _devices_comm, _devices_rank,
                               _devices_size);
  }

  void gpu_kernel::clean_MPI_structure() {
    /*if (_coul_int_reading_type == as_a_whole) {
      if (MPI_Win_free(&_shared_win)!=MPI_SUCCESS) throw std::runtime_error("Fail destroying mpi shared memory");
    }*/
    if (_devices_comm != MPI_COMM_NULL) {
      if (MPI_Comm_free(&_devices_comm) != MPI_SUCCESS) throw std::runtime_error("Fail releasing device communicator");
    }
  }

}  // namespace green::gpu