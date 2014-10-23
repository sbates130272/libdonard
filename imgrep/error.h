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


#ifndef __IMGREP_ERROR_H__
#define __IMGREP_ERROR_H__


#include <errno.h>

enum {
    IMGREP_ERROR_MASK = 0x800000,
    EBADIMAGE         = 0x800001,
    ETRANSFORMFAIL    = 0x800002,
    ECUDA_ERROR       = 0x800003,
    ENOPLAN_ERROR     = 0x800004,
    ELOC_MISMATCH     = 0x800005,
    CUFFT_ERROR_MASK  = 0x840000,
    EBAD_TIFF_IMAGE   = 0x800006,
    EUNSUPPORTED_TIFF = 0x800007,
};

#ifdef __cplusplus
extern "C" {
#endif

const char *error_strerror(int errnum);
void error_perror(const char *s);

#ifdef __cplusplus
}
#endif


#endif
