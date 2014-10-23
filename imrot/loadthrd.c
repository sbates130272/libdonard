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

#include "loadthrd.h"
#include "image.h"
#include "error.h"

#include <libdonard/worker.h>
#include <libdonard/macro.h>

struct loadthrd {
    struct worker worker;
    struct fifo *input;
    struct fifo *output;
    int flags;
};

static void *load_thread(void *arg)
{
    struct loadthrd *lt = container_of(arg, struct loadthrd, worker);
    char *fname;

    int open_flags = 0;
    if (lt->flags & LOADTHREAD_NO_RDMA)
        open_flags |= IMAGE_FLAG_NO_RDMA;

    struct image *img = NULL;
    while ((fname = fifo_pop(lt->input)) != NULL) {

        if (!(lt->flags & LOADTHREAD_ONE_LOAD) || img == NULL) {
            if (lt->flags & LOADTHREAD_CUDA)
                img = image_open_cuda(fname, 0, 0, open_flags);
            else
                img = image_open_local(fname, 0, 0);
        }

        if (img == NULL) {
            error_perror(fname);
            free(fname);
            continue;
        }

        free(fname);

        if (lt->flags & LOADTHREAD_ONE_LOAD)
            image_ref(img);

        fifo_push(lt->output, img);
    }

    if (worker_finish_thread(&lt->worker))
        fifo_close(lt->output);

    return NULL;
}

struct loadthrd *loadthrd_start(struct fifo *input, struct fifo *output,
                                int num_threads, int flags)
{
    struct loadthrd *lt = malloc(sizeof(*lt));
    if (lt == NULL)
        return NULL;

    lt->input = input;
    lt->output = output;
    lt->flags = flags;

    if (!worker_start(&lt->worker, num_threads, load_thread))
        return lt;

    free(lt);
    return NULL;
}

void loadthrd_join(struct loadthrd *lt, int print_cputime)
{
    if (print_cputime)
        fprintf(stderr, "Load Thread CPU Time:\n");

    worker_join(&lt->worker, print_cputime);

    free(lt);
}
