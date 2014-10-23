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

#include "searchthrd.h"
#include "image.h"
#include "img_search.h"
#include "error.h"

#include <libdonard/worker.h>
#include <libdonard/macro.h>

#include <string.h>
#include <stdio.h>

struct searchthrd {
    struct worker worker;
    struct image *needle;
    struct fifo *input;
    struct fifo *output;
};

static void *search_thread(void *arg)
{
    struct searchthrd *st = container_of(arg, struct searchthrd, worker);
    struct image *img;

    while ((img = fifo_pop(st->input)) != NULL) {
        struct image *source = NULL;
        struct searchthrd_result *r = malloc(sizeof(*r));
        memset(r, 0, sizeof(*r));
        if (r == NULL) {
            error_perror(img->filename);
            image_free(img);
            continue;
        }

        r->width = img->width;
        r->height = img->height;
        r->filename = img->filename;
        r->filesize = img->filesize;
        img->filename = NULL;

        if (img_search(img, &r->res)) {
            error_perror(r->filename);
        } else {
            fifo_push(st->output, r);
        }

        image_free(img);
        if (source != NULL)
            image_free(source);
    }

    if (worker_finish_thread(&st->worker))
        fifo_close(st->output);

    return NULL;
}

struct searchthrd *searchthrd_start(struct image *needle,
                                    struct fifo *input, struct fifo *output,
                                    int num_threads)
{
    struct searchthrd *st = malloc(sizeof(*st));
    if (st == NULL)
        return NULL;

    st->input = input;
    st->output = output;
    st->needle = needle;

    if (!worker_start(&st->worker, num_threads, search_thread))
        return st;

    free(st);
    return NULL;
}

void searchthrd_join(struct searchthrd *st, int print_cputime)
{
    if (print_cputime)
        fprintf(stderr, "Search Thread CPU Time:\n");

    worker_join(&st->worker, print_cputime);
    free(st);
}

void searchthrd_free_result(struct searchthrd_result *res)
{
    if (res->filename)
        free((void *) res->filename);
    free(res);
}
