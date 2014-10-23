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
//     Image Search Routines
//
////////////////////////////////////////////////////////////////////////


#include "img_search.h"
#include "img_search_cuda.h"
#include "image.h"
#include "error.h"

#include <fftw3.h>
#include <math.h>
#include <pthread.h>

#define MIN_WIDTH   128
#define MIN_HEIGHT  128
#define MAX_WIDTH   4096
#define MAX_HEIGHT  4096

#include <stdio.h>

const char *fft_wisdom_file = DATAROOTDIR "/imgrep/fftw_wisdom.dat";

struct fft_plan;
struct fft_plan {
    size_t width, height;

    FFTW(plan) fwd_local, rev_local;

    pthread_mutex_t cuda_mutex;

    cufftHandle fwd_cuda, rev_cuda;

    enum image_loc needle_loc;
    void *needle_fft;

    struct fft_plan *next_width;
    struct fft_plan *next_height;
};

static struct fft_plan *all_fft_plans;
static struct image *current_needle;

static struct fft_plan* create_plan(size_t w, size_t h, unsigned flags,
                                    unsigned verbosity)
{
    if (verbosity >= 2)
        fprintf(stderr, "INFO: Creating plan %zdx%zd\n", w, h);

    cufftResult res;
    struct fft_plan *plan = NULL;

    struct image *img = image_new_local(w, h);
    if (img == NULL)
        return NULL;

    FFTW(complex) *fft_data = FFTW(alloc_complex)(h * (w/2+1));
    if (fft_data == NULL)
        goto free_img;

    plan = malloc(sizeof(*plan));
    if (plan == NULL)
        goto free_fft_data;

    plan->width = w;
    plan->height = h;
    plan->next_height = NULL;
    plan->next_width = NULL;
    plan->needle_fft = NULL;
    pthread_mutex_init(&plan->cuda_mutex, NULL);

    plan->fwd_local = FFTW(plan_dft_r2c_2d)(h, w, img->buf, fft_data, flags);
    if (plan->fwd_local == NULL)
        goto free_plan;

    plan->rev_local = FFTW(plan_dft_c2r_2d)(h, w, fft_data, img->buf, flags);
    if (plan->rev_local == NULL)
        goto free_fwd_local;

    res = cufftPlan2d(&plan->fwd_cuda, h, w, CUFFT_R2C);
    if (res != CUFFT_SUCCESS) {
        errno = CUFFT_ERROR_MASK | res;
        goto free_rev_local;
    }

    res = cufftSetCompatibilityMode(plan->fwd_cuda, CUFFT_COMPATIBILITY_NATIVE);
    if (res != CUFFT_SUCCESS) {
        errno = CUFFT_ERROR_MASK | res;
        goto free_fwd_cuda;
    }

    res = cufftPlan2d(&plan->rev_cuda, h, w, CUFFT_C2R);
    if (res != CUFFT_SUCCESS) {
        errno = CUFFT_ERROR_MASK | res;
        goto free_fwd_cuda;
    }

    res = cufftSetCompatibilityMode(plan->rev_cuda, CUFFT_COMPATIBILITY_NATIVE);
    if (res != CUFFT_SUCCESS) {
        errno = CUFFT_ERROR_MASK | res;
        goto free_rev_cuda;
    }

    goto free_fft_data;

free_rev_cuda:
    cufftDestroy(plan->rev_cuda);

free_fwd_cuda:
    cufftDestroy(plan->fwd_cuda);

free_rev_local:
    FFTW(destroy_plan)(plan->rev_local);

free_fwd_local:
    FFTW(destroy_plan)(plan->fwd_local);

free_plan:
    free(plan);
    plan = NULL;

free_fft_data:
    FFTW(free)(fft_data);

free_img:
    image_free(img);
    return plan;
}

static void free_needle_fft(struct fft_plan *p)
{
    if (p->needle_fft != NULL) {
        if (p->needle_loc == IMAGE_LOCAL)
            FFTW(free)(p->needle_fft);
        else if (p->needle_loc == IMAGE_CUDA)
            cudaFree(p->needle_fft);
    }
}

static void free_plan(struct fft_plan *p)
{
    if (p->next_height != NULL)
        free_plan(p->next_height);
    if (p->next_width != NULL)
        free_plan(p->next_width);

    cufftDestroy(p->rev_cuda);
    cufftDestroy(p->fwd_cuda);
    FFTW(destroy_plan)(p->rev_local);
    FFTW(destroy_plan)(p->fwd_local);

    free_needle_fft(p);

    free(p);
}

static struct fft_plan *find_plan(struct image *img)
{
    struct fft_plan *p;

    for (p = all_fft_plans; p != NULL; p = p->next_width)
        if (p->width == img->bufwidth)
            break;

    if (p == NULL) return p;

    for (; p != NULL; p = p->next_height)
        if (p->height == img->bufheight)
            break;

    return p;
}

int img_search_init(int effort, int verbosity)
{
    errno = 0;
    struct fft_plan *cur_plan = NULL;

    unsigned flags;

    if (effort >= 3)      flags = FFTW_EXHAUSTIVE;
    else if (effort == 2) flags = FFTW_PATIENT;
    else if (effort == 1) flags = FFTW_MEASURE;
    else                  flags = FFTW_ESTIMATE;

    if (effort >= 0 &&
        FFTW(import_wisdom_from_filename)(fft_wisdom_file) &&
        verbosity)
    {
        fprintf(stderr, "INFO: Imported FFTW wisdom from: %s\n",
                fft_wisdom_file);
    }

    size_t w = MIN_WIDTH;
    while (w <= MAX_WIDTH) {
        size_t h = MIN_HEIGHT;
        struct fft_plan * p = create_plan(w, h, flags, verbosity);
        if (p == NULL)
            goto free_and_exit;

        if (cur_plan == NULL)
            all_fft_plans = p;
        else
            cur_plan->next_width = p;

        cur_plan = p;

        struct fft_plan *dim2_plan = NULL;
        while (h <= MAX_HEIGHT) {
            struct fft_plan * q = create_plan(w, h, flags, verbosity);
            if (q == NULL)
                goto free_and_exit;

            if (dim2_plan == NULL)
                cur_plan->next_height = q;
            else
                dim2_plan->next_height = q;

            dim2_plan = q;
            h <<= 1;
        }

        w <<= 1;
    }

    if (FFTW(export_wisdom_to_filename)(fft_wisdom_file) && verbosity)
        fprintf(stderr, "INFO: Exported FFTW wisdom to: %s\n",
                fft_wisdom_file);

    return 0;

free_and_exit:
    free_plan(all_fft_plans);
    return -errno;

}

void img_search_deinit(void)
{
    free_plan(all_fft_plans);
}

static void multiply_local(FFTW(complex) *x, FFTW(complex) *y,
                           size_t bufsize, image_px divconst)
{
    for (size_t i = 0; i < bufsize; i++) {
        image_px a, b, c, d;

        a = x[i][0];
        b = x[i][1];
        c = y[i][0];
        d = y[i][1];

        x[i][0] = (a*c - b*d) / divconst;
        x[i][1] = (b*c + a*d) / divconst;
    }
}

static int convolve_local(struct image *img, struct image *kernel,
                          struct image *result, int flags)
{
    int h = img->bufheight;
    int w = img->bufwidth;

    struct fft_plan *plan = find_plan(img);
    if (plan == NULL) {
        errno = ENOPLAN_ERROR;
        return -errno;
    }

    size_t bufsize = h * (w/2+1);
    FFTW(complex) *img_fft_data = FFTW(alloc_complex)(bufsize);
    if (img_fft_data == NULL)
        return -errno;

    FFTW(complex) *ker_fft_data;
    if (kernel != NULL) {
        ker_fft_data = FFTW(alloc_complex)(bufsize);
        if (ker_fft_data == NULL) {
            FFTW(free)(img_fft_data);
            return -errno;
        }
    } else {
        ker_fft_data = plan->needle_fft;
        if (img->loc != plan->needle_loc) {
            FFTW(free)(img_fft_data);
            errno = ELOC_MISMATCH;
            return -errno;
        }
    }

    FFTW(execute_dft_r2c)(plan->fwd_local, img->buf, img_fft_data);

    if (kernel != NULL)
        FFTW(execute_dft_r2c)(plan->fwd_local, kernel->buf, ker_fft_data);

    multiply_local(img_fft_data, ker_fft_data, bufsize, h*w);

    FFTW(execute_dft_c2r)(plan->rev_local, img_fft_data, result->buf);

    if (kernel != NULL)
        FFTW(free)(ker_fft_data);

    FFTW(free)(img_fft_data);
    return 0;
}

static int convolve_cuda(struct image *img, struct image *kernel,
                         struct image *result, int flags)
{
    int h = img->bufheight;
    int w = img->bufwidth;

    errno = 0;

    cufftResult res;
    complex_cuda_px *img_fft_data, *ker_fft_data;

    struct fft_plan *plan = find_plan(img);
    if (plan == NULL) {
        errno = ENOPLAN_ERROR;
        return -errno;
    }

    size_t bufsize = h * (w/2+1);
    if (cudaMalloc((void **) &img_fft_data, bufsize * sizeof(*img_fft_data))
        != cudaSuccess)
    {
        errno = ECUDA_ERROR;
        return -errno;
    }

    if (kernel != NULL) {
        if (cudaMalloc((void **) &ker_fft_data, bufsize*sizeof(*img_fft_data))
            != cudaSuccess)
        {
            errno = ECUDA_ERROR;
            goto free_img_data;
        }
    } else {
        ker_fft_data = plan->needle_fft;
        if (img->loc != plan->needle_loc) {
            errno = ELOC_MISMATCH;
            goto free_img_data;
        }
    }

    //I believe set Stream is not thread safe.
    pthread_mutex_lock(&plan->cuda_mutex);
    {

        res = cufftSetStream(plan->fwd_cuda, image_stream(img));
        if (res != CUFFT_SUCCESS) {
            errno = CUFFT_ERROR_MASK | res;
            goto free_ker_data;
        }

        res = cufftExecR2C(plan->fwd_cuda, img->buf, img_fft_data);
        if (res != CUFFT_SUCCESS) {
            errno = CUFFT_ERROR_MASK | res;
            goto free_ker_data;
        }

        if (kernel != NULL) {
            res = cufftExecR2C(plan->fwd_cuda, kernel->buf, ker_fft_data);
            if (res != CUFFT_SUCCESS) {
                errno = CUFFT_ERROR_MASK | res;
                goto free_ker_data;
            }
        }
    }
    pthread_mutex_unlock(&plan->cuda_mutex);

    if (img_search_cuda_multiply(img_fft_data, ker_fft_data,
                                 bufsize, h*w, image_stream(img))
        != cudaSuccess)
    {
        errno = ECUDA_ERROR;
        goto free_ker_data;
    }

    pthread_mutex_lock(&plan->cuda_mutex);
    {
        res = cufftSetStream(plan->rev_cuda, image_stream(img));
        if (res != CUFFT_SUCCESS) {
            errno = CUFFT_ERROR_MASK | res;
            goto free_ker_data;
        }

        res = cufftExecC2R(plan->rev_cuda, img_fft_data, result->buf);
        if (res != CUFFT_SUCCESS) {
            errno = CUFFT_ERROR_MASK | res;
            goto free_ker_data;
        }
    }
    pthread_mutex_unlock(&plan->cuda_mutex);


free_ker_data:
    if (kernel != NULL)
        cudaFree(ker_fft_data);

free_img_data:
    cudaFree(img_fft_data);

    return -errno;
}

int img_search_convolve(struct image *img, struct image *kernel,
                        struct image *result, int flags)
{
    if (kernel != NULL)
        image_moveto(kernel, img->loc);

    if (img->loc != result->loc)
        return ELOC_MISMATCH;

    if (img->loc == IMAGE_LOCAL)
        return convolve_local(img, kernel, result, flags);
    else if (img->loc == IMAGE_CUDA)
        return convolve_cuda(img, kernel, result, flags);

    errno = EINVAL;
    return -errno;
}

static int edge_detect(struct image *needle)
{
    struct image *edge_kernel = image_new_local(needle->bufwidth,
                                                needle->bufheight);
    if (edge_kernel == NULL)
        return -errno;

    IMAGE_PX(edge_kernel, p);

    for (size_t r = 0; r < 3; r++)
        for (size_t c = 0; c < 3; c++)
            p[r][c] = -1.0 / 8.0;
    p[1][1] = 1.0;

    int ret = img_search_convolve(needle, edge_kernel, needle, 0);

    image_free(edge_kernel);
    return ret;
}

static int generate_needle_fft_local(struct image *needle,
                                     struct fft_plan *plan)
{
    size_t bufsize = plan->height * (plan->width/2+1);
    FFTW(complex) *fft_data = FFTW(alloc_complex)(bufsize);
    if (fft_data == NULL)
        return -errno;

    FFTW(execute_dft_r2c)(plan->fwd_local, needle->buf, fft_data);

    plan->needle_fft = fft_data;

    return 0;
}

static int generate_needle_fft_cuda(struct image *needle,
                                    struct fft_plan *plan)
{
    complex_cuda_px *fft_data;
    size_t bufsize = plan->height * (plan->width/2+1);
    if (cudaMalloc((void **) &fft_data, bufsize * sizeof(*fft_data))
        != cudaSuccess)
    {
        errno = ECUDA_ERROR;
        return -errno;
    }

    cufftResult res;
    res = cufftExecR2C(plan->fwd_cuda, needle->buf, fft_data);
    if (res != CUFFT_SUCCESS) {
        errno = CUFFT_ERROR_MASK | res;
        cudaFree(fft_data);
        return -errno;
    }

    plan->needle_fft = fft_data;

    return 0;
}

static int generate_needle_fft(struct image *needle, struct fft_plan *plan)
{
    if (plan->next_width) {
        int ret = generate_needle_fft(needle, plan->next_width);
        if (ret)
            return -errno;
    }

    if (plan->next_height) {
        int ret = generate_needle_fft(needle, plan->next_height);
        if (ret)
            return -errno;
    }

    free_needle_fft(plan);
    plan->needle_loc = needle->loc;

    struct image *n2 = image_reshape_clone(needle, plan->width,
                                           plan->height);
    if (n2 == NULL)
        return -errno;

    int ret;
    if (needle->loc == IMAGE_LOCAL)
        ret = generate_needle_fft_local(n2, plan);
    else
        ret = generate_needle_fft_cuda(n2, plan);

    image_free(n2);
    return ret;
}

int img_search_set_needle(struct image *needle)
{
    int ret;

    current_needle = needle;

    struct image *n = image_clone(needle);

    if ((ret = image_rot180(n)))
        goto free_and_exit;

    if ((ret = edge_detect(n)))
        goto free_and_exit;

    if ((ret = generate_needle_fft(n, all_fft_plans)))
        goto free_and_exit;


free_and_exit:
    image_free(n);
    return ret;
}

int img_search(struct image *haystack, struct img_search_res *res)
{
    struct image *resimg = image_znew_dup(haystack);
    if (resimg < 0)
        return -errno;

    int ret = img_search_convolve(haystack, NULL, resimg, 0);
    if (ret) {
        image_free(resimg);
        return ret;
    }

    resimg->height = haystack->height;
    resimg->width = haystack->width;

    //if (resimg->loc == IMAGE_LOCAL)
    //    image_save(resimg, "res.jpg");

    res->confidence = image_max(resimg, &res->x, &res->y);

    image_free(resimg);

    if (isnan(res->confidence)) {
        errno = ECUDA_ERROR;
        return -errno;
    }

    res->w = current_needle->width;
    res->h = current_needle->height;
    res->x -= current_needle->width - 1;
    res->y -= current_needle->height;

    if (res->x < 0 || res->y < 0 ||
        res->x > haystack->width-current_needle->width ||
        res->y > haystack->height-current_needle->height)
    {
        res->confidence = -1.0;
    }

    return 0;
}
