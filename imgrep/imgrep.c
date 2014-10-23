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
//     Image Decoding Routines
//
////////////////////////////////////////////////////////////////////////

#include "error.h"
#include "loadthrd.h"
#include "searchthrd.h"
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
    "Search a haystack of images for a needle image.";


struct config {
    char *needle;
    unsigned load_threads;
    unsigned search_threads;
    int one_file_system;
    double threshold;
    int show_version;
    int use_cuda;
    int use_tiff;
    int plan_effort;
    int print_not_found;
    int quiet;
    int verbosity;
    int discard_mode;
    int one_load_mode;
    int pbuf_size_mb;
    int no_rdma;
};

static const struct config defaults = {
    .needle = DATAROOTDIR "/imgrep/pmclogo.png",
    .load_threads = 4,
    .threshold = 250,
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
            "load images and discard the data without searching "
            "(for performance testing)"},
    {"L",               "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"load-threads",    "NUM", CFG_POSITIVE, &defaults.load_threads, required_argument,
            "number of image loading threads to use"},
    {"n",               "IMAGE", CFG_STRING, NULL, required_argument, NULL},
    {"needle",          "IMAGE", CFG_STRING, &defaults.needle, required_argument,
          "image to search for"},
    {"N",               "", CFG_NONE, &defaults.print_not_found, no_argument, NULL},
    {"not-found",       "", CFG_NONE, &defaults.print_not_found, no_argument,
            "print images that the needle was not found in"},
    {"O",               "", CFG_NONE, &defaults.one_load_mode, no_argument, NULL},
    {"one-load",         "", CFG_NONE, &defaults.one_load_mode, no_argument,
            "load only one image per thread and reuse it multiple times"
            "(for performance testing)"},
    {"P",               "NUM", CFG_INT, NULL, required_argument, NULL},
    {"plan-effort",     "NUM", CFG_INT, &defaults.plan_effort, required_argument,
            "effort used to create the fftw plans"},
    {"q",             "", CFG_INCREMENT, NULL, no_argument, NULL},
    {"quiet",         "", CFG_INCREMENT, &defaults.quiet, no_argument,
            "be quiet"},
    {"R",               "", CFG_NONE, &defaults.no_rdma, no_argument, NULL},
    {"no-rdma",         "", CFG_NONE, &defaults.no_rdma, no_argument,
            "disable using RDMA for transfering data to the GPU"},
    {"S",               "NUM", CFG_POSITIVE, NULL, required_argument, NULL},
    {"search-threads",  "NUM", CFG_POSITIVE, &defaults.search_threads, required_argument,
            "number of image searching threads to use"},
    {"t",               "DOUBLE", CFG_DOUBLE, NULL, required_argument, NULL},
    {"threshold",       "DOUBLE", CFG_DOUBLE, &defaults.threshold, required_argument,
            "confidence threshold before reporting"},
    {"T",               "", CFG_NONE, &defaults.use_tiff, no_argument, NULL},
    {"tiff",            "", CFG_NONE, &defaults.use_tiff, no_argument,
            "grep through tiff images instead of jpegs"},
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

const char *filters_jpeg[] = {
    "*.jpg",
    "*.jpeg",
    NULL
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

static void print_results(struct fifo *result_fifo, struct config *cfg,
                          struct stats *s)
{
    struct searchthrd_result *r;
    while ((r = fifo_pop(result_fifo)) != NULL) {
        if (!cfg->quiet && r->res.confidence > cfg->threshold && !cfg->one_load_mode) {
            printf("%s:  %5zd+%-3zd %5zd+%-3zd    (%.2f)\n", r->filename,
                   r->res.x, r->res.w, r->res.y, r->res.h,
                   r->res.confidence);
            fflush(stdout);
        } else if (!cfg->quiet && cfg->print_not_found >= 1) {
            printf("%s: Not found\n", r->filename);
        }

        s->pixels += r->width * r->height;
        s->filebytes += r->filesize;
        s->files++;

        searchthrd_free_result(r);
    }
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

int main(int argc, char *argv[])
{
    struct config cfg;
    struct stats stats = {0};

    argconfig_append_usage("[IMAGE|DIR, ...]");

    int args = argconfig_parse(argc, argv, program_desc, command_line_options,
                               &defaults, &cfg, sizeof(cfg));
    argv[args+1] = NULL;

    if (cfg.show_version) {
        printf("Donard imgrep version %s\n", VERSION);
        return 0;
    }

    if (cfg.search_threads == 0) {
        if (cfg.use_cuda)
            cfg.search_threads = 1;
        else
            cfg.search_threads = get_nprocs() - cfg.load_threads;
    }

    if (cfg.search_threads < 1) cfg.search_threads = 1;

    fprintf(stderr, "Load Threads: %d\n", cfg.load_threads);
    fprintf(stderr, "Search Threads: %d\n", cfg.search_threads);
    fprintf(stderr, "Image: %s\n", cfg.use_tiff ? "TIFF" : "JPEG");
    fprintf(stderr, "Mode: %s\n\n", cfg.use_cuda ? (cfg.no_rdma ? "CUDA" : "CUDA+RDMA") : "CPU");

    if (cfg.use_cuda) {
        if (pinpool_init(PINPOOL_MAX_BUFS, cfg.pbuf_size_mb*1024*1024)) {
            perror("Could not initialize pin pool");
            return -1;
        }
    }

    if (img_search_init(cfg.plan_effort, cfg.verbosity)) {
        error_perror("Could not create image search plans");
        return 1;
    }

    perfstats_init();
    image_init();

    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    perfstats_enable();

    struct image *needle = image_open_local(cfg.needle, 0, 0);
    if (needle == NULL) {
        fprintf(stderr, "Could not open needle '%s': %s\n", cfg.needle,
                error_strerror(errno));
        goto deinit;
    }

    if (cfg.use_cuda)
        image_moveto(needle, IMAGE_CUDA);

    if (img_search_set_needle(needle)) {
        error_perror("Could not preprocess needle");
        goto free_needle;
    }

    struct fifo *filename_fifo = fifo_new(8);
    if (filename_fifo == NULL) {
        error_perror("Could not create fifo");
        goto free_needle;
    }

    int options = 0;
    if (cfg.one_file_system)
        options |= DIRWALK_ONE_FILE_SYSTEM;

    const char **filters;
    if (cfg.use_tiff)
        filters = filters_tiff;
    else
        filters = filters_jpeg;

    if (dirwalk(&argv[1], args, filters, filename_fifo, options)) {
        perror("Could not start dirwalk thread");
        return -errno;
    }

    struct fifo *image_fifo = fifo_new(4);
    if (image_fifo == NULL) {
        error_perror("Could not create fifo");
        goto free_filename_fifo;
    }

    struct fifo *result_fifo = fifo_new(4);
    if (image_fifo == NULL) {
        error_perror("Could not create fifo");
        goto free_image_fifo;
    }

    int loadthrd_flags = 0;
    if (cfg.one_load_mode)
        loadthrd_flags |= LOADTHREAD_ONE_LOAD;
    if (cfg.use_cuda)
        loadthrd_flags |= LOADTHREAD_CUDA;
    if (cfg.no_rdma)
        loadthrd_flags |= LOADTHREAD_NO_RDMA;


    struct loadthrd *lt = loadthrd_start(filename_fifo, image_fifo,
                                         cfg.load_threads, loadthrd_flags);
    if (lt == NULL) {
        error_perror("Could not start load threads");
        goto free_result_fifo;
    }

    struct searchthrd *st = NULL;
    if (!cfg.discard_mode) {
        st = searchthrd_start(needle, image_fifo, result_fifo,
                              cfg.search_threads);
        if (st == NULL) {
            error_perror("Could not start search threads");
            goto free_result_fifo;
        }

        print_results(result_fifo, &cfg, &stats);
    } else {
        discard_images(image_fifo, &cfg, &stats);
    }


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

    loadthrd_join(lt, cfg.verbosity >= 1);
    if (st != NULL)
        searchthrd_join(st, cfg.verbosity >= 1);

free_result_fifo:
    fifo_free(result_fifo);

free_image_fifo:
    fifo_free(image_fifo);

free_filename_fifo:
    fifo_free(filename_fifo);

free_needle:
    image_free(needle);

deinit:
    image_deinit();
    perfstats_deinit();
    img_search_deinit();

    if (cfg.use_cuda)
        pinpool_deinit();

    return -errno;
}
