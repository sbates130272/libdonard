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

#include <libdonard/filemap.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <string.h>
#include <stdio.h>
#include <math.h>

static struct image * test_local_image(int fd, const char *fname)
{
    struct filemap *map = filemap_alloc_local(fd, fname);

    if (!map) {
        error_perror("Could not create file map");
        return NULL;
    }

    struct image *img = image_load_local(map, 1024, 1024);
    if (!img) {
        error_perror("Could not load image");
        return NULL;
    }

    IMAGE_PX(img, pixels);

    for (size_t r = 0; r < 10; r++) {
        for (size_t c = 0; c < 10; c++) {
            printf("%3d ", (int) (pixels[r][c]*255));
        }
        printf("\n");
    }

    filemap_free(map);

    return img;
}

static int test_reshape(struct image *img)
{
    struct image *img2 = image_clone(img);
    if (img2 == NULL) {
        error_perror("Could not clone image");
        return -1;
    }

    int ret = 0;
    if ((ret = image_reshape(img2, 512, 512))) {
        image_free(img2);
        return ret;
    }

    /*
    IMAGE_PX(img, p1);
    IMAGE_PX(img2, p2);

    for (size_t r = 0; r < img2->bufheight; r++) {
        for (size_t c = 0; c < img2->bufwidth; c++) {
            if (p1[r][c] != p2[r][c]) {
                ret = -2;
                goto free_and_return;
            }
        }
    }


free_and_return:*/
    image_free(img2);
    return ret;
}

static int test_rot180(struct image *img)
{
    int ret = 0;
    struct image *img2 = image_clone(img);
    if (img2 == NULL) {
        error_perror("Could not clone image");
        return -1;
    }

    if (image_rot180(img2))
        return 2;
    //image_save(img2, "rot180.jpg");
    if (image_rot180(img2))
        return 2;

    if (image_compare(img2, img)) {
        fprintf(stderr, "Image rotation failure!");
        ret = -1;
    }

    image_free(img2);
    return ret;
}

static int test_move(struct image *img)
{
    int ret = 0;

    struct image *a = image_copyto(img, IMAGE_CUDA);
    if (a == NULL) {
        error_perror("Could not copy image to CUDA");
        return -1;
    }

    struct image *b = image_clone(img);
    if (b == NULL) {
        error_perror("Could not clone image");
        ret = -1;
        goto free_a;
    }

    struct image *c = image_copyto(a, IMAGE_LOCAL);
    if (c == NULL) {
        error_perror("Could not copy image to host");
        ret = -1;
        goto free_b;
    }

    if (image_moveto(b, IMAGE_CUDA)) {
        error_perror("Could not move image to CUDA");
        ret = -1;
        goto free_c;
    }

    if (image_moveto(a, IMAGE_LOCAL)) {
        error_perror("Could not move image to host");
        ret = -1;
        goto free_c;
    }

    if (image_moveto(b, IMAGE_LOCAL)) {
        error_perror("Could not move image to host");
        ret = -1;
        goto free_c;
    }

    if (image_compare(img, a) != 0) {
        fprintf(stderr, "Image 'a' mismatch!");
        ret = -1;
        goto free_c;
    }

    if (image_compare(img, b) != 0) {
        fprintf(stderr, "Image 'b' mismatch!");
        ret = -1;
        goto free_c;
    }

    if (image_compare(img, c) != 0) {
        fprintf(stderr, "Image 'c' mismatch!");
        ret = -1;
        goto free_c;
    }


free_c:
    image_free(c);

free_b:
    image_free(b);

free_a:
    image_free(a);
    return ret;
}

static int test_load_bytes(void)
{
    const int H = 90;
    const int W = 100;
    unsigned char data[H][W];

    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            data[r][c] = r*W+c;

    struct image *img = image_znew_local(128, 128);
    if (img == NULL) {
        error_perror("Could not allocate image");
        return -1;
    }

    int ret = 0;
    IMAGE_PX(img, px);

    if ((ret = image_load_bytes(img, data, W, H))) {
        error_perror("Error loading bytes");
        ret = -1;
        goto free_and_return;
    }

    for (int r = 0; r < img->bufheight; r++) {
        for (int c = 0; c < img->bufwidth; c++)
        {
            if (r >= H || c >= W)
                ret |= px[r][c] != 0.0;
            else
                ret |= fabs((px[r][c] * 255.0) - data[r][c]) > 0.001;

        }
    }

    image_save(img, "load.jpg");

free_and_return:
    image_free(img);
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

    image_init();

    int fd = open(img_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error opening file '%s': %s\n", argv[1],
                error_strerror(errno));
        goto cleanup_and_exit;
    }

    struct image *img;
    if ((img = test_local_image(fd, argv[1])) == NULL) {
        ret = -1;
        goto cleanup_and_exit;
    }

    if ((ret = test_reshape(img)))
        goto free_and_exit;

    if ((ret = test_rot180(img)))
        goto free_and_exit;

    if ((ret = test_move(img)))
        goto free_and_exit;

    if ((ret = test_load_bytes()))
        goto free_and_exit;

free_and_exit:
    image_free(img);

cleanup_and_exit:
    image_deinit();

    return ret;
}
