// -----------------------------------------------------------------
// cudaNumericSort - Sort numeric values using CUDA 
// Copyright (C) 2024  Gabriele Bonacini
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
// -----------------------------------------------------------------

#pragma once

#include <iostream>

namespace cudasorts {

__global__ void sortNumericCudaEvenCheck(auto* toSort, size_t halfrows, uint32_t* actions){

   unsigned int cidx { blockIdx.x * blockDim.x + threadIdx.x },
                cblk { blockDim.x * gridDim.x };

   for(size_t idx{cidx}; idx < halfrows; idx += cblk){
           size_t realIdx { idx * 2 };
           if(toSort[realIdx] > toSort[realIdx + 1]){
               auto  tmp { toSort[realIdx + 1] };
               toSort[realIdx + 1] = toSort[realIdx];
               toSort[realIdx] = tmp;
               atomicAdd(actions, 1);
           }
   }
}

__global__ void sortNumericCudaOddCheck(auto*  toSort, size_t halfrows, uint32_t* actions){

   unsigned int cidx { blockIdx.x * blockDim.x + threadIdx.x },
                cblk { blockDim.x * gridDim.x };

   for(size_t idx{cidx}; idx < halfrows - 1; idx += cblk){
           size_t realIdx { idx * 2 + 1 };
           if(toSort[realIdx] > toSort[realIdx + 1]){
               auto  tmp { toSort[realIdx + 1] };
               toSort[realIdx + 1] = toSort[realIdx];
               toSort[realIdx] = tmp;
               atomicAdd(actions, 1);
           }
   }
}

__host__ size_t sortNumericCudaCheck(auto*  toSort, size_t rows, size_t blks=256){
    uint32_t  *movedOdd   {nullptr},
              *movedEven  {nullptr};

    if(cudaMallocManaged(&movedOdd, sizeof(bool)) != cudaSuccess){
        std::cerr << "Error: allocating unified memory  (loops)\n";
        exit(EXIT_FAILURE);
    }

    if(cudaMallocManaged(&movedEven, sizeof(bool)) != cudaSuccess){
        std::cerr << "Error: allocating unified memory  (loops)\n";
        exit(EXIT_FAILURE);
    }

    size_t       j         { 0 },
                 halfRows  { rows / 2};

    const size_t BLOCKS { blks },
                 DIM    { (rows + BLOCKS - 1) / BLOCKS };

    for( ; j < rows; j++){
       *movedOdd  = 0;
       *movedEven = 0;

       sortNumericCudaEvenCheck<<<DIM, BLOCKS>>>(toSort, halfRows, movedEven);
       sortNumericCudaOddCheck<<<DIM, BLOCKS>>>(toSort, halfRows, movedOdd);
       cudaDeviceSynchronize(); 

       if( *movedOdd == 0  && *movedEven == 0 ) break;
    }

    return j;
}

__global__ void sortNumericCudaEven(auto* toSort, size_t halfrows){

   unsigned int cidx { blockIdx.x * blockDim.x + threadIdx.x },
                cblk { blockDim.x * gridDim.x };

   for(size_t idx{cidx}; idx < halfrows; idx += cblk){
           size_t realIdx { idx * 2 };
           if(toSort[realIdx] > toSort[realIdx + 1]){
               auto  tmp { toSort[realIdx + 1] };
               toSort[realIdx + 1] = toSort[realIdx];
               toSort[realIdx] = tmp;
           }
   }
}

__global__ void sortNumericCudaOdd(auto*  toSort, size_t halfrows){

   unsigned int cidx { blockIdx.x * blockDim.x + threadIdx.x },
                cblk { blockDim.x * gridDim.x };

   for(size_t idx{cidx}; idx < halfrows - 1; idx += cblk){
           size_t realIdx { idx * 2 + 1 };
           if(toSort[realIdx] > toSort[realIdx + 1]){
               auto  tmp { toSort[realIdx + 1] };
               toSort[realIdx + 1] = toSort[realIdx];
               toSort[realIdx] = tmp;
           }
   }
}

__host__ size_t sortNumericCuda(auto*  toSort, size_t rows, size_t blks=256){
    size_t       j         { 0 },
                 halfRows  { rows / 2};

    const size_t BLOCKS { blks },
                 DIM    { (rows + BLOCKS - 1) / BLOCKS };

    for( ; j < rows; j++){
       sortNumericCudaEven<<<DIM, BLOCKS>>>(toSort, halfRows);
       sortNumericCudaOdd<<<DIM, BLOCKS>>>(toSort, halfRows);
       cudaDeviceSynchronize(); 
    }

    return j;
}

__host__ size_t sortNumericHybrCuda(auto*  toSort, size_t rows, unsigned char trigger, size_t blks=256){
    size_t         j         { 0 },
                   halfRows  { rows / 2};

    const size_t   BLOCKS { blks },
                   DIM    { (rows + BLOCKS - 1) / BLOCKS };

    unsigned char  triggerPercent { trigger > static_cast<unsigned char>(90) ? static_cast<unsigned char>(90) : trigger };

    uint32_t  *movedOdd   {nullptr},
              *movedEven  {nullptr};

    if(cudaMallocManaged(&movedOdd, sizeof(bool)) != cudaSuccess){
        std::cerr << "Error: allocating unified memory  (loops)\n";
        exit(EXIT_FAILURE);
    }

    if(cudaMallocManaged(&movedEven, sizeof(bool)) != cudaSuccess){
        std::cerr << "Error: allocating unified memory  (loops)\n";
        exit(EXIT_FAILURE);
    }

    for( ; j < rows; j++){
       if( j*100/rows < triggerPercent ){
          sortNumericCudaEven<<<DIM, BLOCKS>>>(toSort, halfRows);
          sortNumericCudaOdd<<<DIM, BLOCKS>>>(toSort, halfRows);
          cudaDeviceSynchronize(); 
       }else{
          *movedOdd    = 0;
          *movedEven   = 0;
          sortNumericCudaEvenCheck<<<DIM, BLOCKS>>>(toSort, halfRows, movedEven);
          sortNumericCudaOddCheck<<<DIM, BLOCKS>>>(toSort, halfRows, movedOdd);
          cudaDeviceSynchronize(); 

          if( *movedOdd == 0  && *movedEven == 0 ) break;
       }
    }

    return j;
}

} // End Namespace
