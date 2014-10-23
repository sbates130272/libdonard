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

#include "rotthrd.h"
#include "image.h"
#include "error.h"

#include <libdonard/worker.h>
#include <libdonard/macro.h>

#include <string.h>
#include <stdio.h>

struct rotthrd {
    struct worker worker;
    struct fifo *input;
    int flags;
};

static void *rot_thread(void *arg)
{
    struct rotthrd *rt = container_of(arg, struct rotthrd, worker);
    struct image *img;

    int save_flags = 0;
    if (rt->flags & ROTTHREAD_NO_RDMA)
        save_flags |= IMAGE_FLAG_NO_RDMA;

    while ((img = fifo_pop(rt->input)) != NULL) {
        if (image_rot180(img)) {
            error_perror(img->filename);
        } else {
            image_save(img, save_flags);
        }

        image_free(img);
    }

    worker_finish_thread(&rt->worker);

    return NULL;
}

struct rotthrd *rotthrd_start(struct fifo *input, int num_threads, int flags)
{
    struct rotthrd *rt = malloc(sizeof(*rt));
    if (rt == NULL)
        return NULL;

    rt->input = input;
    rt->flags = flags;

    if (!worker_start(&rt->worker, num_threads, rot_thread))
        return rt;

    free(rt);
    return NULL;
}

void rotthrd_join(struct rotthrd *rt, int print_cputime)
{
    if (print_cputime)
        fprintf(stderr, "Rotation Thread CPU Time:\n");

    worker_join(&rt->worker, print_cputime);
    free(rt);
}
