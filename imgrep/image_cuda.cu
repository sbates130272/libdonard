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

__global__ void reshape_array(image_px *src, size_t src_w, size_t src_h,
                              image_px *dst)
{
    extern __shared__ image_px sdata[];
    int tid = threadIdx.x;
    int x = tid + blockIdx.x * blockDim.x;
    int y = blockIdx.y * blockDim.y;

    int dst_w = gridDim.x * blockDim.x;

    if (x >= src_w || y >= src_h)
        sdata[tid] = 0;
    else
        sdata[tid] = src[y*src_w + x];

    __syncthreads();

    dst[y*dst_w + x] = sdata[tid];
}

image_px *image_cuda_reshape(struct image *img, size_t width, size_t height)
{
    image_px *newbuf;

    if (cudaMalloc(&newbuf, width * height * sizeof(*newbuf)) != cudaSuccess)
        return NULL;

    size_t threads_per_block = 512;
    if (threads_per_block > width)
        threads_per_block = width;

    dim3 block_size, grid_size;
    block_size.x = threads_per_block;
    grid_size.x = width / threads_per_block;
    grid_size.y = height;

    assert((grid_size.x * threads_per_block) == width);

    size_t shared_mem = threads_per_block * sizeof(image_px);

    reshape_array<<<grid_size, block_size, shared_mem, image_stream(img)>>>
        (img->buf, img->bufwidth, img->bufheight, newbuf);

    if (cudaPeekAtLastError() != cudaSuccess) {
        errno = ECUDA_ERROR;
        cudaFree(newbuf);
        return NULL;
    }

    return newbuf;
}

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
        (img->buf, img->width-1, img->height-1, img->bufwidth);

    return cudaPeekAtLastError();
}


__device__ __forceinline__ int max_comp(image_px a, image_px b)
{
    return a > b;
}

__device__ __forceinline__ int min_comp(image_px a, image_px b)
{
    return a < b;
}

typedef int (*comp_func)(image_px, image_px);

template <comp_func comp>
__device__ __forceinline__
void reduce(image_px *sdata, unsigned int *sidxs,
            unsigned int tid, unsigned int other_tid)
{
    if (comp(sdata[other_tid], sdata[tid])) {
        sdata[tid] = sdata[other_tid];
        sidxs[tid] = sidxs[other_tid];
    }
}

template <unsigned int block_size, comp_func comp>
__global__ void extrema(image_px *idata, image_px *odata, unsigned int *indexes)
{
    __shared__ unsigned int sidxs[block_size*2];
    __shared__ image_px sdata[block_size*2];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(block_size*2) + tid;

    sdata[tid] = idata[i];
    sidxs[tid] = i;

    image_px other = idata[i+block_size];
    if (comp(other, sdata[tid])) {
        sdata[tid] = other;
        sidxs[tid] = i + block_size;
    }

    __syncthreads();

    if (block_size >= 1024 && tid < 512) {
        reduce<comp>(sdata, sidxs, tid, tid+512);
        __syncthreads();
    }

    if (block_size >= 512 && tid < 256) {
        reduce<comp>(sdata, sidxs, tid, tid+256);
        __syncthreads();
    }

    if (block_size >= 256 && tid < 128) {
        reduce<comp>(sdata, sidxs, tid, tid+128);
        __syncthreads();
    }

    if (block_size >= 128 && tid < 64) {
        reduce<comp>(sdata, sidxs, tid, tid+64);
        __syncthreads();
    }

    if (tid < 32) {
        if (block_size >= 64) reduce<comp>(sdata, sidxs, tid, tid+32);
        if (block_size >= 32) reduce<comp>(sdata, sidxs, tid, tid+16);
        if (block_size >= 16) reduce<comp>(sdata, sidxs, tid, tid+8);
        if (block_size >=  8) reduce<comp>(sdata, sidxs, tid, tid+4);
        if (block_size >=  4) reduce<comp>(sdata, sidxs, tid, tid+2);
        if (block_size >=  2) reduce<comp>(sdata, sidxs, tid, tid+1);
    }

    if (tid == 0) {
        odata[blockIdx.x] = sdata[0];
        indexes[blockIdx.x] = sidxs[0];
    }
}

template <comp_func comp>
static image_px extrema_step(image_px *buf, size_t elements, unsigned int *idx,
                             cudaStream_t stream)
{
    image_px ret;

    dim3 block_size, grid_size;
    block_size.x = 1024;

    while (elements & (block_size.x*2-1))
        block_size.x >>= 1;

    grid_size.x = elements / (block_size.x * 2);

    image_px *results;
    unsigned int *indexes;

    if (cudaMalloc(&results, grid_size.x * sizeof(*results)) != cudaSuccess) {
        errno = ECUDA_ERROR;
        return nan("");
    }

    if (cudaMalloc(&indexes, grid_size.x * sizeof(*indexes)) != cudaSuccess) {
        cudaFree(results);
        errno = ECUDA_ERROR;
        return nan("");
    }

    #define call(x) extrema<x, comp><<<grid_size, block_size, 0, stream>>> \
                              (buf, results, indexes);

    switch (block_size.x) {
    case 1024: call(1024); break;
    case 512:  call(512);  break;
    case 256:  call(256);  break;
    case 128:  call(128);  break;
    case 64:   call(64);   break;
    case 32:   call(32);   break;
    case 16:   call(16);   break;
    case 8:    call(8);    break;
    case 4:    call(4);    break;
    case 2:    call(2);    break;
    case 1:    call(1);    break;
    default:
        assert(0);
    }

    if (cudaPeekAtLastError() != cudaSuccess) {
        errno = ECUDA_ERROR;
        return nan("");
    }

    if (grid_size.x == 1) {
        cudaMemcpyAsync(idx, &indexes[0], sizeof(*idx), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(&ret, &results[0], sizeof(ret), cudaMemcpyDeviceToHost,
                        stream);
    } else {
        unsigned int idx_ret;
        ret = extrema_step<comp>(results, grid_size.x, &idx_ret, stream);

        if (isnan(ret))
            goto free_and_exit;

        cudaMemcpyAsync(idx, &indexes[idx_ret], sizeof(*idx), cudaMemcpyDeviceToHost,
                        stream);
    }

free_and_exit:
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        ret = nan("");

    cudaFree(indexes);
    cudaFree(results);

    return ret;
}

image_px image_cuda_max(const struct image *img, size_t *x, size_t *y)
{
    unsigned int loc = 0;
    size_t els = img->bufwidth * img->bufheight;

    image_px ret = extrema_step<max_comp>(img->buf, els, &loc,
                                          image_stream(img));

    if (y != NULL) *y = loc / img->bufwidth;
    if (x != NULL) *x = loc % img->bufwidth;

    return ret;
}

image_px image_cuda_min(const struct image *img, size_t *x, size_t *y)
{
    unsigned int loc = 0;
    size_t els = img->bufwidth * img->bufheight;

    image_px ret = extrema_step<min_comp>(img->buf, els, &loc,
                                          image_stream(img));

    if (y != NULL) *y = loc / img->bufwidth;
    if (x != NULL) *x = loc % img->bufwidth;

    return ret;
}

__global__ void load_bytes(unsigned char *src, size_t src_w, size_t src_h,
                           image_px *dst)
{
    extern __shared__ image_px sdata[];
    int tid = threadIdx.x;
    int x = tid + blockIdx.x * blockDim.x;
    int y = blockIdx.y * blockDim.y;

    int dst_w = gridDim.x * blockDim.x;

    if (x >= src_w || y >= src_h)
        sdata[tid] = 0;
    else
        sdata[tid] = src[y*src_w + x] / 255.0;

    __syncthreads();

    dst[y*dst_w + x] = sdata[tid];
}


int image_cuda_load_bytes(struct image *img, void *src_data,
                          size_t width, size_t height)
{
    size_t threads_per_block = 512;
    if (threads_per_block > img->bufwidth)
        threads_per_block = img->bufwidth;

    dim3 block_size, grid_size;
    block_size.x = threads_per_block;
    grid_size.x = img->bufwidth / threads_per_block;
    grid_size.y = img->bufheight;

    assert((grid_size.x * threads_per_block) == img->bufwidth);

    size_t shared_mem = threads_per_block * sizeof(image_px);

    load_bytes<<<grid_size, block_size, shared_mem, image_stream(img)>>>
        ((unsigned char *)src_data, width, height, img->buf);

    if (cudaPeekAtLastError() != cudaSuccess) {
        errno = ECUDA_ERROR;
        return -errno;
    }

    return 0;
}
