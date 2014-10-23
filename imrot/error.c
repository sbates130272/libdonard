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
//     Common error handling code.
//
////////////////////////////////////////////////////////////////////////

#include "error.h"

#include <cuda_runtime.h>
#include <cufft.h>

#include <string.h>
#include <stdio.h>

#define ERRMSG(idx, message)  [idx & ~IMGREP_ERROR_MASK] = message

const char *errlist[] = {
    ERRMSG(EBADIMAGE, "Invalid image file!"),
    ERRMSG(ETRANSFORMFAIL, "Failed transforming image to greyscale!"),
    ERRMSG(ENOPLAN_ERROR, "No plan generated for this image size!"),
    ERRMSG(ELOC_MISMATCH, "Image location mismatch!"),
    ERRMSG(EBAD_TIFF_IMAGE, "Error parsing TIFF image!"),
    ERRMSG(EUNSUPPORTED_TIFF, "Unsuported TIFF format."),
};

static const char *cufft_error(int errnum)
{
    switch (errnum & ~CUFFT_ERROR_MASK) {
    case CUFFT_SUCCESS: return "cufft: No Error.";
    case CUFFT_INVALID_PLAN: return "cufft: Invalid Plan Handle.";
    case CUFFT_ALLOC_FAILED: return "cufft: Could not allocate memory.";
    case CUFFT_INVALID_TYPE: return "cufft: Invalid Type.";
    case CUFFT_INVALID_VALUE: return "cufft: Invalid parameter.";
    case CUFFT_INTERNAL_ERROR: return "cufft: Internal Error.";
    case CUFFT_EXEC_FAILED: return "cufft: Failed to execute on the GPU.";
    case CUFFT_SETUP_FAILED: return "cuftt The CUFFT library failed to initialize.";
    case CUFFT_INVALID_SIZE: return "cufft: User specified an invalid transform size.";
    case CUFFT_UNALIGNED_DATA: return "cufft: Unaligned Data.";
    default: return "Unknown Error.";
    }
}

const char *error_strerror(int errnum)
{
    if (!(errnum & IMGREP_ERROR_MASK))
        return strerror(errnum);

    if (errnum == ECUDA_ERROR)
        return cudaGetErrorString(cudaGetLastError());

    if ((errnum & CUFFT_ERROR_MASK) != IMGREP_ERROR_MASK)
        return cufft_error(errnum);

    int idx = errnum & ~IMGREP_ERROR_MASK;

    if (idx > sizeof(errlist) / sizeof(*errlist) || errlist[idx] == NULL)
        return "Unknown Error";

    return errlist[idx];
}

void error_perror(const char *s)
{
    if (!(errno & IMGREP_ERROR_MASK)) {
        perror(s);
        return;
    }

    fprintf(stderr, "%s: %s\n", s, error_strerror(errno));
}
