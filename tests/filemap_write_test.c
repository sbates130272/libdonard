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
//     File map write test, just copies a file through the GPU
//
////////////////////////////////////////////////////////////////////////


#include <libdonard/filemap.h>
#include <libdonard/utils.h>

#include <argconfig/argconfig.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <linux/limits.h>

#include <string.h>
#include <stdio.h>

#include <cuda_runtime.h>

#define FALLOC_FL_NO_HIDE_STALE  0x4

const char program_desc[] =
    "Perform a write from GPU memory to NVMe SSD.";

struct config {
    char *src;
    char *dest;
    int no_write;
    int zeros;
    int quiet;
};

static const struct config defaults = {
    .src = "/mnt/princeton/random_test2",
    .dest = "/mnt/princeton/random_test2.write_test",
    .zeros = 0,
    .no_write = 0,
    .quiet = 0,
};

static const struct argconfig_commandline_options command_line_options[] = {
    {"s",               "IMAGE", CFG_STRING, NULL, required_argument, NULL},
    {"source",          "IMAGE", CFG_STRING, &defaults.src, required_argument,
          "image to use as the source for the copy"},
    {"d",             "IMAGE", CFG_STRING, NULL, required_argument, NULL},
    {"dest",          "IMAGE", CFG_STRING, &defaults.dest, required_argument,
          "filename for destination of copy"},
    {"z",            "", CFG_NONE, &defaults.zeros, no_argument, NULL},
    {"zeros",        "", CFG_NONE, &defaults.zeros, no_argument,
            "fill the destination with zeros before the GPU copy"},
    {"w",               "", CFG_NONE, &defaults.no_write, no_argument, NULL},
    {"no-write",        "", CFG_NONE, &defaults.no_write, no_argument,
            "do not perform the GPU write itself"},
    {"q",               "", CFG_NONE, &defaults.no_write, no_argument, NULL},
    {"quiet",        "", CFG_NONE, &defaults.no_write, no_argument,
            "run in quiet mode"},
    {"h",                     "", CFG_NONE, NULL, no_argument, NULL},
    {"help",                  "", CFG_NONE, NULL, no_argument, NULL},
    {"-help",                 "", CFG_NONE, NULL, no_argument,
            "display this help and exit"},
    {0}
};


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

int main(int argc, char *argv[])
{
    int fd_dst;
    int ret = 0;

    struct config cfg;

    argconfig_parse(argc, argv, program_desc, command_line_options,
                    &defaults, &cfg, sizeof(cfg));


    if (pinpool_init(1, 64*1024*1024)) {
        perror("Could not initialize pin pool");
        return -1;
    }

    if (!cfg.quiet)
        fprintf(stderr, "Copying file '%s' to '%s'.\n", cfg.src, cfg.dest);

    int fd = open(cfg.src, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error opening file '%s': %s\n", cfg.src, strerror(errno));
        ret = -2;
        goto leave;
    }

    struct filemap * src = filemap_alloc_cuda_nvme(fd, cfg.src);
    if (!src) {
        perror("Could not create nvme file map");
        ret = -4;
        goto leave;
    }

    if (src->map_error) {
        fprintf(stderr, "Could not allocate nvme file map: %s\n",
                filemap_map_error_string(src->map_error));

        filemap_free(src);
        ret = -5;
        goto leave;
    }

    close(fd);


    umask(0);
    fd_dst = open(cfg.dest, O_CREAT | O_WRONLY | O_TRUNC, 0666);
    if (fd_dst < 0) {
        fprintf(stderr, "Error opening file '%s': %s\n", cfg.dest,
                strerror(errno));
        ret = -3;
        goto leave;
    }

    if ((ret = fallocate(fd_dst, FALLOC_FL_NO_HIDE_STALE, 0, src->length))) {
        perror("Could not fallocate the file");
        fprintf(stderr, "Writing zeros instead.\n");
        cfg.zeros = 1;
    }

    if (cfg.zeros)
    {
        if (write_zeros(fd_dst, src->length)) {
            perror("Could not zero the file");
            ret = -7;
            goto leave;
        }
    }

    if (!cfg.no_write) {
        if (filemap_write_cuda_nvme(src, fd_dst)) {
            perror("GPU Write failed");
            ret = -8;
            goto leave;
        }
    }

    close(fd_dst);

    if (utils_cmp(cfg.src,cfg.dest)) {
        if (!cfg.quiet)
            fprintf(stderr, "Compare FAILED!\n");
        ret = -9;
        goto leave;
    }
    else
        if (!cfg.quiet)
            fprintf(stderr, "Compare PASSED!\n");

leave:

    pinpool_deinit();

    return ret;
}
