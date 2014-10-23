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
//     GPU Image Rotation with rdma
//
////////////////////////////////////////////////////////////////////////

#include "error.h"
#include "loadthrd.h"
#include "rotthrd.h"
#include "image.h"
#include "version.h"

#include <libdonard/fifo.h>
#include <libdonard/dirwalk.h>
#include <libdonard/utils.h>
#include <libdonard/perfstats.h>

#include <argconfig/argconfig.h>
#include <argconfig/suffix.h>

#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

#include <stdio.h>

const char program_desc[] =
    "Rotate images using the GPU and rDMA";

struct config {
    int use_cuda;
    int load_threads;
    int rot_threads;
    int show_version;
    int discard_mode;
    int verbosity;
    int pbuf_size_mb;
    int no_rdma;
    int one_file_system;
};

static const struct config defaults = {
    .load_threads = 4,
    .pbuf_size_mb = 32,
};

static const struct argconfig_commandline_options command_line_options[] = {
    {"b",             "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"bufsize",       "NUM", CFG_POSITIVE, &defaults.pbuf_size_mb, required_argument,
            "pin buffer size (in MB)"
    },
    {"c",               "", CFG_NONE, &defaults.use_cuda, no_argument, NULL},
    {"cuda",            "", CFG_NONE, &defaults.use_cuda, no_argument,
            "use cuda"},
    {"D",               "", CFG_NONE, &defaults.discard_mode, no_argument, NULL},
    {"discard",         "", CFG_NONE, &defaults.discard_mode, no_argument,
            "load images and discard the data without rotating "
            "(for performance testing)"},
    {"L",               "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"load-threads",    "NUM", CFG_POSITIVE, &defaults.load_threads, required_argument,
            "number of image loading threads to use"},
    {"R",               "", CFG_NONE, &defaults.no_rdma, no_argument, NULL},
    {"no-rdma",         "", CFG_NONE, &defaults.no_rdma, no_argument,
            "disable using RDMA for transfering data to the GPU"},
    {"S",               "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"rot-threads",     "NUM", CFG_POSITIVE, &defaults.rot_threads, required_argument,
            "number of image rotation threads to use"},
    {"v",               "", CFG_INCREMENT, NULL, no_argument, NULL},
    {"verbose",         "", CFG_INCREMENT, &defaults.verbosity, no_argument,
            "be verbose"},
    {"x",               "", CFG_NONE, &defaults.one_file_system, no_argument, NULL},
    {"one-file-system", "", CFG_NONE, &defaults.one_file_system, no_argument,
            "don't cross filesystem boundaries"},
    {"h",                     "", CFG_NONE, NULL, no_argument, NULL},
    {"help",                  "", CFG_NONE, NULL, no_argument, NULL},
    {"-help",                 "", CFG_NONE, NULL, no_argument,
            "display this help and exit"},
    {"V",               "", CFG_NONE, &defaults.show_version, no_argument, NULL},
    {"version", "", CFG_NONE, &defaults.show_version, no_argument,
            "print the version and exit"},

    {0}
};

const char *filters_tiff[] = {
    "*.tif",
    "*.tiff",
    NULL
};

struct stats {
    double   elapsed_time;
    double   pixels;
    double   filebytes;
    unsigned files;
};

static void print_stats(struct stats *s)
{
    fprintf(stderr, "Processed %d files.\n", s->files);

    double speed = s->pixels / s->elapsed_time;

    const char *p_suffix = suffix_si_get(&s->pixels);
    const char *s_suffix = suffix_si_get(&speed);

    fprintf(stderr, "%6.2f%spixels in %.1fs    %6.2f%spixels/s\n",
            s->pixels, p_suffix, s->elapsed_time, speed, s_suffix);


    double data_speed = s->filebytes / s->elapsed_time;
    const char *b_suffix = suffix_si_get(&s->filebytes);
    const char *d_suffix = suffix_si_get(&data_speed);

    fprintf(stderr, "%6.2f%sB      in %.1fs    %6.2f%sB/s\n",
            s->filebytes, b_suffix, s->elapsed_time, data_speed, d_suffix);
}

static void print_cpu_time(void)
{
    struct rusage u;

    getrusage(RUSAGE_SELF, &u);

    fprintf(stderr, "Total CPU Time: %.1fs user, %.1fs system\n",
            utils_timeval_to_secs(&u.ru_utime),
            utils_timeval_to_secs(&u.ru_stime));
}

static void discard_images(struct fifo *image_fifo, struct config *cfg,
                                                      struct stats *s)
{
    struct image *img;
    while ((img = fifo_pop(image_fifo)) != NULL) {
        s->pixels += img->width * img->height;
        s->filebytes += img->filesize;
        s->files++;

        image_free(img);
    }
}


static void copy_images(struct fifo *image_fifo, struct fifo *rot_fifo,
                        struct config *cfg,
                        struct stats *s)
{
    struct image *img;
    while ((img = fifo_pop(image_fifo)) != NULL) {
        s->pixels += img->width * img->height;
        s->filebytes += img->filesize;
        s->files++;

        fifo_push(rot_fifo, img);
    }

    fifo_close(rot_fifo);
}

int main(int argc, char *argv[])
{
    struct config cfg;
    struct stats stats = {0};

    argconfig_append_usage("[IMAGE|DIR, ...]");

    int args = argconfig_parse(argc, argv, program_desc, command_line_options,
                               &defaults, &cfg, sizeof(cfg));
    argv[args+1] = NULL;

    if (cfg.show_version) {
        printf("Donard imrot version %s\n", VERSION);
        return 0;
    }

    if (cfg.rot_threads == 0) {
        if (cfg.use_cuda)
            cfg.rot_threads = 1;
        else
            cfg.rot_threads = get_nprocs() - cfg.load_threads;
    }

    if (cfg.rot_threads < 1) cfg.rot_threads = 1;

    fprintf(stderr, "Load Threads: %d\n", cfg.load_threads);
    fprintf(stderr, "Rotation Threads: %d\n", cfg.rot_threads);
    fprintf(stderr, "Mode: %s\n\n", cfg.use_cuda ? (cfg.no_rdma ? "CUDA" : "CUDA+RDMA") : "CPU");

    if (cfg.use_cuda) {
        if (pinpool_init(PINPOOL_MAX_BUFS, cfg.pbuf_size_mb*1024*1024)) {
            perror("Could not initialize pin pool");
            return -1;
        }
    }

    perfstats_init();
    image_init();

    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    perfstats_enable();

    struct fifo *filename_fifo = fifo_new(8);
    if (filename_fifo == NULL) {
        error_perror("Could not create fifo");
        goto deinit;
    }

    int options = 0;
    if (cfg.one_file_system)
        options |= DIRWALK_ONE_FILE_SYSTEM;

    if (dirwalk(&argv[1], args, filters_tiff, filename_fifo, options)) {
        perror("Could not start dirwalk thread");
        return -errno;
    }

    struct fifo *image_fifo = fifo_new(4);
    if (image_fifo == NULL) {
        error_perror("Could not create fifo");
        goto free_filename_fifo;
    }

    struct fifo *rot_fifo = fifo_new(4);
    if (image_fifo == NULL) {
        error_perror("Could not create fifo");
        goto free_image_fifo;
    }

    int loadthrd_flags = 0;
    int rotthrd_flags = 0;
    if (cfg.use_cuda)
        loadthrd_flags |= LOADTHREAD_CUDA;
    if (cfg.no_rdma) {
        loadthrd_flags |= LOADTHREAD_NO_RDMA;
        rotthrd_flags |= ROTTHREAD_NO_RDMA;
    }


    struct loadthrd *lt = loadthrd_start(filename_fifo, image_fifo,
                                         cfg.load_threads, loadthrd_flags);
    if (lt == NULL) {
        error_perror("Could not start load threads");
        goto free_rot_fifo;
    }

    struct rotthrd *rt = NULL;
    if (!cfg.discard_mode) {
        rt = rotthrd_start(rot_fifo, cfg.rot_threads, rotthrd_flags);
        if (rt == NULL) {
            error_perror("Could not start rotation threads");
            goto free_rot_fifo;
        }
        copy_images(image_fifo, rot_fifo, &cfg, &stats);
    } else {
        discard_images(image_fifo, &cfg, &stats);
    }

    loadthrd_join(lt, cfg.verbosity >= 1);
    if (rt != NULL)
        rotthrd_join(rt, cfg.verbosity >= 1);

    perfstats_disable();
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    stats.elapsed_time = utils_timeval_to_secs(&end_time) -
        utils_timeval_to_secs(&start_time);

    print_stats(&stats);
    perfstats_print();

    if (cfg.verbosity >= 1) {
        fprintf(stderr, "\n\nImage Fifo Max: %d\n", fifo_max_fill(image_fifo));
        print_cpu_time();
    }

free_rot_fifo:
    fifo_free(rot_fifo);

free_image_fifo:
    fifo_free(image_fifo);

free_filename_fifo:
    fifo_free(filename_fifo);

deinit:
    image_deinit();
    perfstats_deinit();

    if (cfg.use_cuda)
        pinpool_deinit();

    return -errno;
}
