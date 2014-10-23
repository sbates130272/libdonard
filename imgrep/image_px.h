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
//     Image Pixel Type and Abstraction
//
////////////////////////////////////////////////////////////////////////


#ifndef __IMGREP_IMAGE_PX_H__
#define __IMGREP_IMAGE_PX_H__

#include <cufft.h>

typedef float image_px;
typedef cufftComplex complex_cuda_px;
#define IMAGE_PX_STORAGE  FloatPixel
#define FFTW  FFTW_MANGLE_FLOAT

#define cufftExecC2R_px   cufftExecC2R
#define cufftExecR2C_px   cufftExecR2C


#endif
