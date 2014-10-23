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
//     Image Searching Thread
//
////////////////////////////////////////////////////////////////////////


#ifndef __IMROT_ROTTHRD_H__
#define __IMROT_ROTTHRD_H__

#include "image.h"

#include <libdonard/fifo.h>

struct rotthrd;

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ROTTHREAD_NO_RDMA = 1,
};

struct rotthrd *rotthrd_start(struct fifo *input, int num_threads, int flags);

void rotthrd_join(struct rotthrd *rt, int print_cpu_time);
void rotthrd_free(struct rotthrd *rt);

#ifdef __cplusplus
}
#endif

#endif
