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
//     Image Loading Thread
//
////////////////////////////////////////////////////////////////////////


#ifndef __IMGREP_LOADTHRD_H__
#define __IMGREP_LOADTHRD_H__

#include <libdonard/fifo.h>

struct loadthrd;

#ifdef __cplusplus
extern "C" {
#endif

enum {
    LOADTHREAD_ONE_LOAD = 1,
    LOADTHREAD_CUDA = 2,
    LOADTHREAD_NO_RDMA = 4,
};

struct loadthrd *loadthrd_start(struct fifo *input, struct fifo *output,
                                int num_threads, int flags);

void loadthrd_join(struct loadthrd *lt, int print_cpu_time);

#ifdef __cplusplus
}
#endif

#endif
