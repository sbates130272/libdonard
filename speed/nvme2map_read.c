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

#include <libdonard/nvme_dev.h>
#include <libdonard/worker.h>
#include <libdonard/macro.h>
#include <libdonard/fifo.h>
#include <libdonard/utils.h>
#include <libdonard/dirwalk.h>
#include <libdonard/perfstats.h>

#include <argconfig/argconfig.h>
#include <argconfig/report.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>

#include <stdio.h>

const char program_desc[] =
    "Test the speed of transfering data from a file to mapped memory.";

struct config {
    unsigned copies;
    unsigned threads;
    unsigned buf_size_mb;
    unsigned bufsize;
    int no_direct_dma;
    int show_version;
    char *mmap_file;
    unsigned mmap_offset;
};

static const struct config defaults = {
    .threads = 1,
    .buf_size_mb = 64,
    .mmap_file = "/dev/mtramon1",
};

static const struct argconfig_commandline_options command_line_options[] = {
    {"b",             "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"bufsize",       "NUM", CFG_POSITIVE, &defaults.buf_size_mb, required_argument,
            "pin buffer size (in MB)"
    },
    {"D",               "", CFG_NONE, &defaults.no_direct_dma, no_argument, NULL},
    {"no-direct_dma",   "", CFG_NONE, &defaults.no_direct_dma, no_argument,
            "don't use direct dma transfers to copy the file"},
    {"m",             "FILE", CFG_STRING, NULL, required_argument, NULL},
    {"mmap",          "FILE", CFG_STRING, &defaults.mmap_file, required_argument,
            "use a buffer mmaped from the specified file"
    },
    {"o",             "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"offset",        "NUM", CFG_POSITIVE, &defaults.mmap_offset, required_argument,
            "offset within the mmaped buffer"
    },
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
    int idx;
    struct worker worker;
    struct fifo *input;
    int no_direct_dma;
    void *buf;
    size_t bufsize;
    size_t bytes;
};

static int read_file(const char *fname, void *buf, size_t bufsize)
{
    int fd = open(fname, O_RDONLY);
    if (fd < 0)
        return -1;

    int ret = read(fd, buf, bufsize);

    close(fd);
    return ret;
}

static void *load_thread(void *arg)
{
    struct loadthrd *lt = container_of(arg, struct loadthrd, worker);
    char *fname;
    size_t bytes = 0;

    int idx = __sync_fetch_and_add(&lt->bytes, 1);
    unsigned char *buf = lt->buf;
    buf += idx*lt->bufsize;

    while ((fname = fifo_pop(lt->input)) != NULL) {
        int ret;

        if (lt->no_direct_dma)
            ret = read_file(fname, buf, lt->bufsize);
        else
            ret = nvme_dev_read_file(fname, buf, lt->bufsize);

        if (ret < 0) {
            fprintf(stderr, "ERROR: Could not read file %s: %s\n", fname,
                    strerror(errno));
            break;
        }

        bytes += ret;
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

struct mmap_buf {
    int fd;
    void *buf;
    size_t len;
};

static struct mmap_buf *open_mmap_buf(struct config *cfg)
{
    struct mmap_buf *mbuf = malloc(sizeof(*mbuf));
    if (mbuf == NULL)
        return NULL;

    mbuf->len = cfg->bufsize * cfg->threads;
    mbuf->fd = open(cfg->mmap_file, O_RDWR);
    if (mbuf->fd < 0)
        goto free_and_exit;

    mbuf->buf = mmap(NULL, mbuf->len, PROT_READ | PROT_WRITE, MAP_SHARED,
                     mbuf->fd, cfg->mmap_offset);

    if (mbuf->buf != MAP_FAILED)
        return mbuf;

    close(mbuf->fd);

free_and_exit:
    free(mbuf);
    return NULL;
}

static void close_mmap_buf(struct mmap_buf *mbuf)
{
    munmap(mbuf->buf, mbuf->len);
    close(mbuf->fd);
    free(mbuf);
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
        printf("Donard nvme2map_read version %s\n", VERSION);
        return 0;
    }

    perfstats_init();

    cfg.bufsize = cfg.buf_size_mb * 1024 * 1024;

    struct mmap_buf *mbuf = open_mmap_buf(&cfg);
    if (mbuf == NULL) {
        fprintf(stderr, "Unable to mmap %s: %s\n", cfg.mmap_file,
                strerror(errno));
        return -1;
    }

    lt.idx = 0;
    lt.bytes = 0;
    lt.no_direct_dma = cfg.no_direct_dma;
    lt.buf = mbuf->buf;
    lt.bufsize = cfg.bufsize;

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

    close_mmap_buf(mbuf);

    perfstats_deinit();

    return ret;
}
