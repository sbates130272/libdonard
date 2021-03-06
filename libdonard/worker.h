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
//     Common Worker Thread Code
//
////////////////////////////////////////////////////////////////////////


#ifndef __LIBDONARD_WORKER_H__
#define __LIBDONARD_WORKER_H__

#include <pthread.h>

struct worker {
    int num_threads;
    pthread_t *threads;
    struct rusage *rusage;
};

#ifdef __cplusplus
extern "C" {
#endif

int worker_start(struct worker *w, int num_threads,
                 void *(*start_routine) (void *));
void worker_join(struct worker *w, int print_cputime);
int worker_finish_thread(struct worker *);

#ifdef __cplusplus
}
#endif


#endif
