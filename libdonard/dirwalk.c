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
//     Directory Walking Thread
//
////////////////////////////////////////////////////////////////////////

#include "dirwalk.h"

#include <fnmatch.h>
#include <fts.h>
#include <pthread.h>
#include <errno.h>

#include <stdio.h>
#include <string.h>

struct dirwalk_thrd {
    char * const * argv;
    int argc;
    const char * const * filters;
    struct fifo *output;
    int options;
};

static void process_file(struct dirwalk_thrd *dt, const char *fpath)
{
    const char * const *f;

    for (f = dt->filters; *f; f++) {
        if (fnmatch(*f, fpath, FNM_CASEFOLD) == 0)
            break;
    }

    if (*f == NULL)
        return;

    char *fpath2 = malloc(strlen(fpath) + 1);
    if (fpath2 == NULL) {
        perror(fpath);
        return;
    }

    strcpy(fpath2, fpath);
    fifo_push(dt->output, fpath2);
}

static int compare(const FTSENT **a, const FTSENT **b)
{
    int a_is_dir = (*a)->fts_info & (FTS_D | FTS_DP);
    int b_is_dir = (*b)->fts_info & (FTS_D | FTS_DP);

    if (a_is_dir && !b_is_dir)
        return 1;

    if (!a_is_dir && b_is_dir)
        return -1;

    return strcmp((*a)->fts_name, (*b)->fts_name);
}

static void *dirwalk_thread(void *arg)
{
    struct dirwalk_thrd *dt = arg;

    char *dot[] = {".", NULL};
    char * const *paths = dt->argc > 0 ? dt->argv : dot;

    int options = FTS_LOGICAL | FTS_NOCHDIR | FTS_NOSTAT;

    if (dt->options & DIRWALK_ONE_FILE_SYSTEM)
        options |= FTS_XDEV;

    FTSENT *node;
    FTS *tree = fts_open(paths, options, compare);
    if (tree == NULL) {
        perror("fts_open");
        goto exit_thread;
    }

    while ((node = fts_read(tree)) != NULL) {
        if (node->fts_info & FTS_ERR && node->fts_errno) {
            fprintf(stderr, "%s: %s\n", node->fts_accpath,
                    strerror(node->fts_errno));
            continue;
        }

        if (!(node->fts_info & FTS_F))
            continue;

        process_file(dt, node->fts_accpath);
    }

    if (errno)
        perror("fts_read");

    if (fts_close(tree))
        perror("fts_close");

exit_thread:
    fifo_close(dt->output);
    free(dt);
    return NULL;
}


int dirwalk(char * const argv[], int argc,
            const char * const filters[], struct fifo *output,
            int options)
{
    struct dirwalk_thrd *dt = malloc(sizeof(*dt));
    if (dt == NULL)
        return -errno;

    dt->argv = argv;
    dt->argc = argc;
    dt->filters = filters;
    dt->output = output;
    dt->options = options;

    pthread_t thrd;
    if (pthread_create(&thrd, NULL, dirwalk_thread, dt) != 0)
        goto free_and_exit;

    pthread_detach(thrd);

    return 0;

free_and_exit:
    free(dt);
    return -errno;

}
