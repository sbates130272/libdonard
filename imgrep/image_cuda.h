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


#ifndef __IMGREP_IMAGE_CUDA_H__
#define __IMGREP_IMAGE_CUDA_H__

#include "image.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

image_px *image_cuda_reshape(struct image *img, size_t width, size_t height);
cudaError_t image_rot180_cuda(struct image *img);
image_px image_cuda_max(const struct image *img, size_t *x, size_t *y);
image_px image_cuda_min(const struct image *img, size_t *x, size_t *y);
int image_cuda_load_bytes(struct image *img, void *src_data,
                          size_t width, size_t height);

#ifdef __cplusplus
}
#endif



#endif
