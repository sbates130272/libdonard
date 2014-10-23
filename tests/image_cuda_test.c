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


#include "test_util.h"

#include <imgrep/image.h>
#include <imgrep/error.h>

#include <time.h>

static void print_diffs(struct image *a, struct image *b)
{
    IMAGE_PX(a, ap);
    IMAGE_PX(b, bp);

    for (size_t y = 0; y < a->bufheight; y++)
            for (size_t x = 0; x < a->bufwidth; x++)
            if (ap[y][x] != bp[y][x])
                printf(" %5zd %5zd %10f %10f\n", x, y, ap[y][x], bp[y][x]);
}

static int test_reshape_clone(struct image *img)
{
    int ret = 0;

    struct image *cuda_a = image_copyto(img, IMAGE_CUDA);
    if (cuda_a == NULL)
        return 1;

    struct image *cuda_b = image_reshape_clone(cuda_a, 2048, 2048);
    if (cuda_b == NULL) {
        ret = 2;
        goto free_cuda_a;
    }

    struct image *img_b = image_reshape_clone(img, 2048, 2048);
    if (img_b == NULL) {
        ret = 3;
        goto free_cuda_b;
    }

    if (image_moveto(cuda_b, IMAGE_LOCAL)) {
        ret = 4;
        goto free_img_b;
    }

    if (image_compare(img_b, cuda_b)) {
        image_save_full(img_b, "reshape_clone.jpg");
        image_save_full(cuda_b, "reshape_clone_cuda.jpg");
        ret = 5;
    }

free_img_b:
    image_free(img_b);

free_cuda_b:
    image_free(cuda_b);

free_cuda_a:
    image_free(cuda_a);
    return ret;
}

static int test_reshape(struct image *img)
{
    int ret = 0;

    struct image *cuda_a = image_copyto(img, IMAGE_CUDA);
    if (cuda_a == NULL)
        return 6;

    if (image_reshape(cuda_a, 2048, 2048))
        return 7;

    struct image *img_b = image_reshape_clone(img, 2048, 2048);
    if (img_b == NULL) {
        ret = 8;
        goto free_cuda_a;
    }

    if (image_moveto(cuda_a, IMAGE_LOCAL)) {
        ret = 9;
        goto free_img_b;
    }

    if (image_compare(img_b, cuda_a)) {
        image_save_full(img_b, "reshape.jpg");
        image_save_full(cuda_a, "reshape_cuda.jpg");
        ret = 10;
    }

free_img_b:
    image_free(img_b);

free_cuda_a:
    image_free(cuda_a);
    return ret;
}

static int test_rot180(struct image *img)
{
    int ret = 0;

    struct image *cimg = image_copyto(img, IMAGE_CUDA);
    if (cimg == NULL)
        return 11;

    struct image *rimg = image_clone(img);
    if (rimg == NULL) {
        ret = 12;
        goto free_cimg;
    }

    if (image_rot180(cimg)) {
        error_perror("Blah");
        ret = 13;
        goto free_rimg;
    }

    if (image_rot180(rimg)) {
        error_perror("Blah2");
        ret = 13;
        goto free_rimg;
    }

    if (image_moveto(cimg, IMAGE_LOCAL)) {
        ret = 12;
        goto free_rimg;
    }

    if (image_compare(cimg, rimg)) {
        print_diffs(cimg, rimg);
        image_save_full(cimg, "rotate_cuda.jpg");
        image_save_full(rimg, "rotate.jpg");
        ret = 14;
    }

free_rimg:
    image_free(rimg);

free_cimg:
    image_free(cimg);
    return ret;
}

static int test_max(int w, int h)
{
    int ret = 0;
    struct image *img = image_new_local(w, h);
    if (img == NULL)
        return 15;

    IMAGE_PX(img, p);
    image_px i = 0;
    for (int y = 0; y < img->bufwidth; y++)
        for (int x = 0; x < img->bufwidth; x++)
            p[y][x] = i++;

    struct image *cimg = image_copyto(img, IMAGE_CUDA);
    if (cimg == NULL) {
        free(img);
        return 16;
    }

    size_t cuda_x, cuda_y;
    image_px cuda_max = image_max(cimg, &cuda_x, &cuda_y);

    size_t x, y;
    image_px max = image_max(img, &x, &y);

    if (x != cuda_x || y != cuda_y)
        ret = 17;

    if (cuda_max != max)
        ret = 18;

    image_free(cimg);
    image_free(img);
    return ret;
}

static int test_min(void)
{
    int ret = 0;
    struct image *img = image_new_local(4096, 4096);
    if (img == NULL)
        return 19;

    IMAGE_PX(img, p);
    image_px i = 0;
    for (int y = 0; y < img->bufwidth; y++)
        for (int x = 0; x < img->bufwidth; x++)
            p[y][x] = -(i++);

    struct image *cimg = image_copyto(img, IMAGE_CUDA);
    if (cimg == NULL) {
        image_free(img);
        return 20;
    }

    size_t cuda_x, cuda_y;
    image_px cuda_min = image_min(cimg, &cuda_x, &cuda_y);

    size_t x, y;
    image_px min = image_min(img, &x, &y);

    if (x != cuda_x || y != cuda_y)
        ret = 21;

    if (cuda_min != min)
        ret = 22;

    image_free(cimg);
    image_free(img);
    return ret;
}

static int test_max_rand(void)
{
    int ret = 0;
    struct image *img = image_new_cuda(1024, 1024);
    if (img == NULL)
        return 23;

    image_px val = rand();
    size_t x = rand() & 1023;
    size_t y = rand() & 1023;

    image_set_pixel(img, x, y, val);

    size_t x2, y2;
    image_px max = image_max(img, &x2, &y2);


    if (x != x2 || y != y2)
        ret = 24;

    if (max != val)
        ret = 25;

    image_free(img);
    return ret;
}

static int test_min_rand(void)
{
    int ret = 0;
    struct image *img = image_new_cuda(1024, 1024);
    if (img == NULL)
        return 23;

    image_px val = -rand();
    size_t x = rand() & 1023;
    size_t y = rand() & 1023;

    image_set_pixel(img, x, y, val);

    size_t x2, y2;
    image_px min = image_min(img, &x2, &y2);

    if (x != x2 || y != y2)
        ret = 26;

    if (min != val)
        ret = 27;

    image_free(img);
    return ret;
}

static int test_load_bytes(void)
{
    int ret = 0;
    const int H = 2000;
    const int W = 3000;
    unsigned char data[H][W];

    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            data[r][c] = r*W+c;

    unsigned char *cdata;
    if (cudaMalloc((void **) &cdata, sizeof(data)) != cudaSuccess) {
        errno = ECUDA_ERROR;
        error_perror("Could not allocate CUDA memory");
        return 28;
    }

    if (cudaMemcpy(cdata, data, sizeof(data),
                   cudaMemcpyHostToDevice) != cudaSuccess)
    {
        errno = ECUDA_ERROR;
        error_perror("Could not copy data to CUDA memory");
        return 29;
    }

    struct image *limg = image_znew_local(4096, 2048);
    if (limg == NULL) {
        error_perror("Could not allocate local image");
        return 30;
    }

    struct image *cimg = image_znew_cuda(4096, 2048);
    if (limg == NULL) {
        error_perror("Could not allocate cuda image");
        ret = 31;
        goto free_limg;
    }

    if ((ret = image_load_bytes(limg, data, W, H))) {
        error_perror("Local load bytes failed");
        goto free_cimg;
    }

    if ((ret = image_load_bytes(cimg, cdata, W, H))) {
        error_perror("CUDA load bytes failed");
        goto free_cimg;
    }

    if (image_moveto(cimg, IMAGE_LOCAL)) {
        error_perror("Could not move CUDA image to host");
        ret = 32;
        goto free_cimg;
    }


    if (image_compare(cimg, limg)) {
        fprintf(stderr, "Loaded images don't match!\n");
        ret = 33;
    }

    image_save_full(limg, "load2.jpg");

free_cimg:
    image_free(cimg);

free_limg:
    image_free(limg);
    return ret;
}

int main(int argc, char *argv[])
{
    int ret = 0;
    const char *img_path;
    char buf1[PATH_MAX];

    if (argc == 1) {
        img_path = test_util_find_img("pmclogo.png", buf1);
    } else if (argc != 2) {
        fprintf(stderr, "USAGE: %s FILE.\n", argv[0]);
        exit(-1);
    } else {
        img_path = argv[1];
    }

    srand(time(NULL));

    image_init();
    struct image *img = image_open_local(img_path, 0, 0);
    if (img == NULL) {
        fprintf(stderr, "Could not open image '%s': %s\n", img_path,
                error_strerror(errno));
        goto cleanup_and_exit;
    }


    if ((ret = test_reshape_clone(img))) {
        goto free_and_exit;
    }

    if ((ret = test_reshape(img))) {
        goto free_and_exit;
    }

    if ((ret = test_rot180(img))) {
        goto free_and_exit;
    }

    if ((ret = test_max(4096, 4096))) {
        goto free_and_exit;
    }

    if ((ret = test_max(256, 512))) {
        goto free_and_exit;
    }

    if ((ret = test_min())) {
        goto free_and_exit;
    }

    if ((ret = test_max_rand())) {
        goto free_and_exit;
    }

    if ((ret = test_min_rand())) {
        goto free_and_exit;
    }

    if ((ret = test_load_bytes())) {
        goto free_and_exit;
    }

free_and_exit:
    image_free(img);

cleanup_and_exit:
    image_deinit();

    return ret;
}
