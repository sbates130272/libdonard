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
//     Image Search Routines
//
////////////////////////////////////////////////////////////////////////

#ifndef __IMGREP_IMG_SEARCH_H__
#define __IMGREP_IMG_SEARCH_H__

#include "image.h"

#include <stdlib.h>

struct img_search_res {
    size_t x, y, w, h;
    image_px confidence;
};

#ifdef __cplusplus
extern "C" {
#endif

int img_search_init(int plan_effort, int verbosity);
void img_search_deinit(void);

int img_search_set_needle(struct image *needle);

int img_search(struct image *haystack, struct img_search_res *res);

int img_search_convolve(struct image *img, struct image *kernel,
                        struct image *result, int flags);

#ifdef __cplusplus
}
#endif


#endif
