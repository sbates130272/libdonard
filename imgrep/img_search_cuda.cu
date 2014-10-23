////////////////////////////////////////////////////////////////////////
//
// Copyright 2014 PMC-Sierra, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License. You may
// obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0 Unless required by
// applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for
// the specific language governing permissions and limitations under the
// License.
//
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
//
//   Author: Logan Gunthorpe
//
//   Date:   Oct 23 2014
//
//   Description:
//     Image Search CUDA Routines
//
////////////////////////////////////////////////////////////////////////

#include "img_search_cuda.h"

__global__ void cmplx_mult_and_scale(complex_cuda_px *x, complex_cuda_px *y,
                                     image_px divconst)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    image_px a,b,c,d;
    a = x[i].x;
    b = x[i].y;
    c = y[i].x;
    d = y[i].y;

    x[i].x = (a*c - b*d) / divconst;
    x[i].y = (b*c + a*d) / divconst;
}

cudaError_t img_search_cuda_multiply(complex_cuda_px *x, complex_cuda_px *y,
                                     size_t bufsize, image_px divconst,
                                     cudaStream_t stream)
{
    dim3 block_size, grid_size;

    block_size.x = 1024;
    while (bufsize & (block_size.x - 1))
        block_size.x >>= 1;

    grid_size.x = bufsize / block_size.x;

    cmplx_mult_and_scale<<<grid_size, block_size, 0, stream>>>
        (x, y, divconst);

    return cudaPeekAtLastError();
}
