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
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>

#include <cuda_runtime.h>

#include <stdio.h>

#define FALLOC_FL_NO_HIDE_STALE  0x4

const char program_desc[] =
    "Test the speed of transfering data from a file from the GPU.";

struct config {
    unsigned write_count;
    unsigned threads;
    unsigned pbuf_size_mb;
    char *output_dir;
    int no_direct_dma;
    int show_version;
};

static const struct config defaults = {
    .write_count = 32,
    .threads = 1,
    .pbuf_size_mb = 32,
    .output_dir = "/mnt/princeton/speed_write_test",
};

static const struct argconfig_commandline_options command_line_options[] = {
    {"c",             "NUM", CFG_POSITIVE, &defaults.write_count, required_argument, NULL},
    {"count",         "NUM", CFG_POSITIVE, &defaults.write_count, required_argument,
            "number of files per thread to write data to"
    },
    {"b",             "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"bufsize",       "NUM", CFG_POSITIVE, &defaults.pbuf_size_mb, required_argument,
            "pin buffer size (in MB)"
    },
    {"D",               "", CFG_NONE, &defaults.no_direct_dma, no_argument, NULL},
    {"no-direct_dma",   "", CFG_NONE, &defaults.no_direct_dma, no_argument,
            "don't use direct dma transfers from the NVMe devcie to the GPU"},
    {"o",             "DIR", CFG_STRING, &defaults.output_dir, required_argument, NULL},
    {"output-dir",    "DIR", CFG_STRING, &defaults.output_dir, required_argument,
            "path to save output files"},
    {"t",             "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"threads",       "NUM", CFG_POSITIVE, &defaults.threads, required_argument,
            "number of threads"
    },
    {"V",               "", CFG_NONE, &defaults.show_version, no_argument, NULL},
    {"version",         "", CFG_NONE, &defaults.show_version, no_argument,
            "print the version and exit"},
    {0}
};

struct savethrd {
    struct worker worker;
    struct fifo *input;
    int no_direct_dma;
    int write_count;
    const char *output_dir;
    size_t bytes;
};

static void init_pattern_pinbuf(struct pin_buf *buf, unsigned int tid)
{
    int *tmp = malloc(buf->bufsize);
    if (tmp == NULL)
        return;

    for (int i = 0; i < buf->bufsize / sizeof(*tmp); i++) {
        tmp[i] = i + tid;
    }

    cudaMemcpy(buf->address, tmp, buf->bufsize, cudaMemcpyHostToDevice);
    free(tmp);
}

static int write_zeros(int fd, size_t length)
{
    const unsigned char buf[4096] = {0};
    while (length) {
        size_t towrite = length;

        if (towrite > sizeof(buf))
            towrite = sizeof(buf);

        ssize_t ret = write(fd, buf, towrite);
        if (ret < 0)
            return -1;

        length -= ret;
    }

    return 0;
}

static void write_no_dma(struct pin_buf *buf, int fd)
{
    unsigned char *cpubuf = malloc(buf->bufsize);
    if (cpubuf == NULL)
        return;

    cudaMemcpy(cpubuf, buf->address, buf->bufsize, cudaMemcpyDeviceToHost);
    write(fd, cpubuf, buf->bufsize);
    fsync(fd);

    free(cpubuf);
}

static void *save_thread(void *arg)
{
    struct savethrd *st = container_of(arg, struct savethrd, worker);
    size_t bytes = 0;

    struct pin_buf *buf = pinpool_alloc();
    unsigned int tid = pthread_self();

    struct filemap fmap = {
        .data = buf->address,
        .length = buf->bufsize,
        .pinbuf = buf,
        .type = FILEMAP_TYPE_CUDA,
    };

    init_pattern_pinbuf(buf, tid);

    for (int i = 0; i < st->write_count; i++) {
        char fn[PATH_MAX];
        sprintf(fn, "%s/%u-%05d.dat", st->output_dir, tid, i);

        int fd = open(fn, O_CREAT | O_WRONLY | O_TRUNC, 0666);
        if (fd < 0) {
            fprintf(stderr, "Error opening file '%s': %s\n", fn,
                    strerror(errno));
            break;
        }

        if (fallocate(fd, FALLOC_FL_NO_HIDE_STALE, 0, fmap.length)) {
            perror("Could not fallocate the file, writing zeros instead: ");
            write_zeros(fd, fmap.length);
        }

        if (st->no_direct_dma || filemap_write_cuda_nvme(&fmap, fd)) {
            write_no_dma(buf, fd);
        }

        bytes += buf->bufsize;

        close(fd);
    }

    pinpool_free(buf);

    __sync_add_and_fetch(&st->bytes, bytes);

    worker_finish_thread(&st->worker);

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

static void delete_output_dir_files(const char *dir)
{
    char cmd[MAX_INPUT];
    sprintf(cmd, "rm -rf %s/*.dat", dir);
    system(cmd);
}

int main(int argc, char *argv[])
{
    struct config cfg;
    struct savethrd st;
    int ret = 0;

    argconfig_append_usage("[FILE|DIR, ...]");

    int args = argconfig_parse(argc, argv, program_desc, command_line_options,
                               &defaults, &cfg, sizeof(cfg));
    argv[args+1] = NULL;

    if (cfg.show_version) {
        printf("Donard nvme2gpu_write version %s\n", VERSION);
        return 0;
    }

    umask(0);
    delete_output_dir_files(cfg.output_dir);

    if (pinpool_init(cfg.threads, cfg.pbuf_size_mb*1024*1024)) {
        perror("Could not initialize pin pool");
        return -1;
    }

    perfstats_init();

    st.bytes = 0;
    st.no_direct_dma = cfg.no_direct_dma;
    st.write_count = cfg.write_count;
    st.output_dir = cfg.output_dir;


    struct timeval start_time;
    gettimeofday(&start_time, NULL);

    perfstats_enable();

    if (worker_start(&st.worker, cfg.threads, save_thread)) {
        perror("Could not start threads");
        ret = -1;
        goto deinit_and_return;
    }

    worker_join(&st.worker, 0);
    perfstats_disable();

    struct timeval end_time;
    gettimeofday(&end_time, NULL);

    print_cpu_time();
    perfstats_print();

    fprintf(stderr, "\nCopied ");
    report_transfer_rate(stderr, &start_time, &end_time, st.bytes);
    fprintf(stderr, "\n");

deinit_and_return:
    perfstats_deinit();
    pinpool_deinit();

    return ret;
}
