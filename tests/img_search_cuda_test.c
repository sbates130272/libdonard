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
#include <imgrep/img_search.h>
#include <imgrep/error.h>

#include <time.h>

static void print_diffs(struct image *a, struct image *b)
{
    IMAGE_PX(a, ap);
    IMAGE_PX(b, bp);

    for (size_t y = 0; y < a->bufheight; y++)
        for (size_t x = 0; x < a->bufwidth; x++)
            if (fabs(ap[y][x] - bp[y][x]) > 1e-5)
                printf(" %5zd %5zd %10f %10f\n", x, y, ap[y][x], bp[y][x]);
}

struct image *create_kernel(struct image *img)
{
    struct image *sharp = image_new_local(img->bufwidth, img->bufheight);

    IMAGE_PX(sharp, p);
    p[0][1] = -1;
    p[1][0] = -1;
    p[1][1] = 5;
    p[1][2] = -1;
    p[2][1] = -1;

    return sharp;
}

int test_convolve(struct image *img, struct image *ker)
{
    int ret = 0;

    struct image *local = image_clone(img);
    if (local == NULL)
        return 1;

    struct image *cuda = image_copyto(img, IMAGE_CUDA);
    if (cuda == NULL) {
        ret = 2;
        goto free_local;
    }

    if (img_search_convolve(local, ker, local, 0)) {
        ret = 2;
        goto free_cuda;
    }

    if (img_search_convolve(cuda, ker, cuda, 0)) {
        error_perror("Could not perform cuda convolve");
        ret = 4;
        goto free_cuda;
    }

    image_moveto(cuda, IMAGE_LOCAL);

    image_save_full(local, "convolve_local.jpg");
    image_save_full(cuda, "convolve_cuda.jpg");

    if (image_compare(local, cuda) != 0) {
        print_diffs(local, cuda);
        ret = 5;
        goto free_cuda;
    }

free_cuda:
    image_free(cuda);
free_local:
    image_free(local);

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

    if (img_search_init(0, 1)) {
        error_perror("Could not create image search plans");
        return 1;
    }

    image_init();
    struct image *img = image_open_local(img_path, 0, 0);
    if (img == NULL) {
        fprintf(stderr, "Could not open image '%s': %s\n", img_path,
                error_strerror(errno));
        goto cleanup_and_exit;
    }

    struct image *ker = create_kernel(img);
    if (ker == NULL) {
        error_perror("Could not create kernel");
        goto free_img_and_exit;
    }

    if ((ret = test_convolve(img, ker))) {
        goto free_and_exit;
    }

free_and_exit:
    image_free(ker);

free_img_and_exit:
    image_free(img);

cleanup_and_exit:
    image_deinit();
    img_search_deinit();

    return ret;
}
