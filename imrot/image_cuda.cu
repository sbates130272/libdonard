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
//     Image CUDA Routines
//
////////////////////////////////////////////////////////////////////////

#include "image_cuda.h"
#include "error.h"

#include <assert.h>

__global__ void rot180(image_px *src, size_t w, size_t h,
                       size_t bufwidth)
{
    extern __shared__ image_px block1[];

    image_px *block2 = &block1[blockDim.x];

    int tid = threadIdx.x;
    int x1 = tid + blockIdx.x * blockDim.x;
    int y1 = blockIdx.y * blockDim.y;

    int x2 = w - x1;
    int y2 = h - y1;

    block1[tid] = src[y1*bufwidth + x1];
    block2[tid] = src[y2*bufwidth + x2];

    __syncthreads();

    if (x2 < 0 || y2 < 0)
        return;

    if (y1 == y2 && x1 > w /2)
        return;

    src[y1*bufwidth + x1] = block2[tid];
    src[y2*bufwidth + x2] = block1[tid];
}

cudaError_t image_rot180_cuda(struct image *img)
{
    size_t threads_per_block = 1024;
    while (threads_per_block > img->width)
        threads_per_block -= 32;

    dim3 block_size, grid_size;
    block_size.x = threads_per_block;
    grid_size.x = (img->width + threads_per_block - 1) / threads_per_block;
    grid_size.y = (img->height+1)/2;

    size_t shared_mem = threads_per_block * 2 * sizeof(image_px);

    rot180<<<grid_size, block_size, shared_mem, image_stream(img)>>>
        (img->buf, img->width-1, img->height-1, img->width);

    return cudaPeekAtLastError();
}
