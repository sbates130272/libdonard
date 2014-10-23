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
//     Test throughput for copying files to GPU.
//
////////////////////////////////////////////////////////////////////////

#include "version.h"

#include <libdonard/filemap.h>
#include <libdonard/worker.h>
#include <libdonard/macro.h>
#include <libdonard/utils.h>
#include <libdonard/fifo.h>
#include <libdonard/dirwalk.h>
#include <libdonard/perfstats.h>

#include <argconfig/argconfig.h>
#include <argconfig/report.h>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>

#include <stdio.h>

const char program_desc[] =
    "Test the speed of transfering data from a file to the GPU.";

struct config {
    unsigned copies;
    unsigned threads;
    unsigned pbuf_size_mb;
    int no_direct_dma;
    int show_version;
};

static const struct config defaults = {
    .threads = 1,
    .pbuf_size_mb = 64,
};

static const struct argconfig_commandline_options command_line_options[] = {
    {"b",             "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"bufsize",       "NUM", CFG_POSITIVE, &defaults.pbuf_size_mb, required_argument,
            "pin buffer size (in MB)"
    },
    {"D",               "", CFG_NONE, &defaults.no_direct_dma, no_argument, NULL},
    {"no-direct_dma",   "", CFG_NONE, &defaults.no_direct_dma, no_argument,
            "don't use direct dma transfers from the NVMe devcie to the GPU"},
    {"t",             "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"threads",       "NUM", CFG_POSITIVE, &defaults.threads, required_argument,
            "number of threads"
    },
    {"V",               "", CFG_NONE, &defaults.show_version, no_argument, NULL},
    {"version",         "", CFG_NONE, &defaults.show_version, no_argument,
            "print the version and exit"},
    {0}
};

struct loadthrd {
    struct worker worker;
    struct fifo *input;
    int no_direct_dma;
    unsigned warned;
    size_t bytes;
};

static void *load_thread(void *arg)
{
    struct loadthrd *lt = container_of(arg, struct loadthrd, worker);
    char *fname;
    size_t bytes = 0;

    while ((fname = fifo_pop(lt->input)) != NULL) {
        struct filemap *fm;

        if (lt->no_direct_dma)
            fm = filemap_open_cuda(fname);
        else
            fm = filemap_open_cuda_nvme(fname);

        if (fm == NULL) {
            perror("ERROR: Could not create filemap.");
            break;
        }

        if (fm->map_error && !lt->warned) {
            lt->warned = 1;
            fprintf(stderr, "WARNING: Not using direct DMA transfer: %s\n",
                    filemap_map_error_string(fm->map_error));
        }

        bytes += fm->length;
        filemap_free(fm);
    }

    __sync_add_and_fetch(&lt->bytes, bytes);

    worker_finish_thread(&lt->worker);

    return NULL;
}

static void print_cpu_time(void)
{
    struct rusage u;

    getrusage(RUSAGE_SELF, &u);

    fprintf(stderr, "Total CPU Time: %.1fs user, %.1fs system\n",
            utils_timeval_to_secs(&u.ru_utime),
            utils_timeval_to_secs(&u.ru_stime));
}

const char *filters[] = {
    "*",
    NULL
};

int main(int argc, char *argv[])
{
    struct config cfg;
    struct loadthrd lt;
    int ret = 0;

    argconfig_append_usage("[FILE|DIR, ...]");

    int args = argconfig_parse(argc, argv, program_desc, command_line_options,
                               &defaults, &cfg, sizeof(cfg));
    argv[args+1] = NULL;

    if (cfg.show_version) {
        printf("Donard nvme2gpu_read version %s\n", VERSION);
        return 0;
    }

    if (pinpool_init(cfg.threads, cfg.pbuf_size_mb*1024*1024)) {
        perror("Could not initialize pin pool");
        return -1;
    }

    perfstats_init();

    lt.bytes = 0;
    lt.warned = 0;
    lt.no_direct_dma = cfg.no_direct_dma;

    lt.input = fifo_new(16);
    if (lt.input == NULL) {
        perror("Could not create fifo");
        ret = -1;
        goto deinit_and_return;
    }

    if (dirwalk(&argv[1], args, filters, lt.input, 0)) {
        perror("Could not start dirwalk thread");
        ret = -1;
        goto free_fifo_and_return;
    }

    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    perfstats_enable();

    if (worker_start(&lt.worker, cfg.threads, load_thread)) {
        perror("Could not start threads");
        ret = -1;
        goto free_fifo_and_return;
    }

    worker_join(&lt.worker, 0);
    perfstats_disable();

    struct timeval end_time;
    gettimeofday(&end_time, NULL);

    print_cpu_time();
    perfstats_print();

    fprintf(stderr, "\nCopied ");
    report_transfer_rate(stderr, &start_time, &end_time, lt.bytes);
    fprintf(stderr, "\n");

free_fifo_and_return:
    fifo_free(lt.input);

deinit_and_return:
    perfstats_deinit();
    pinpool_deinit();

    return ret;
}
