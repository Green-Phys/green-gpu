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

#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<stdexcept>

#include<mpi.h>

void check_for_cuda(MPI_Comm global_comm, int global_rank, int &devCount_per_node) {
  if(cudaGetDeviceCount(&devCount_per_node)!= cudaSuccess)
    throw std::runtime_error("error counting cuda devices, typically that means no cuda devices available.");
  if(devCount_per_node==0) throw std::runtime_error("you're starting this code with cuda support but no device is available");
  if (!global_rank) {
    for (int i = 0; i < devCount_per_node; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      std::cout<<"Device Number: " << i << std::endl;
      std::cout<<"  Device name: " << prop.name << std::endl;
      std::cout<<"  Peak Memory Bandwidth (GB/s): "<<2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6<<std::endl;
      std::cout<<"  Compute Capability: "<<prop.major<<"."<<prop.minor<<std::endl;
      std::cout<<"  total global mem: "<<prop.totalGlobalMem/(1024.*1024*1024)<<" GB"<<std::endl;
    }
  }
  MPI_Barrier(global_comm);
}

