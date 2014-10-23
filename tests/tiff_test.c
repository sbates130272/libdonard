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
//     Test tiff parsing code
//
////////////////////////////////////////////////////////////////////////

#include "test_util.h"

#include <libdonard/tiff.h>
#include <imgrep/error.h>
#include <imgrep/image.h>

#include <stdio.h>
#include <limits.h>

static void print_data(struct tiff_tag *tag)
{
    unsigned values[tag->count];

    switch(tag->type) {
    case TIFF_Byte:
        for (int i = 0; i < tag->count; i++)
            values[i] = tag->data.byte[i];
        break;
    case TIFF_ASCII:
        printf("%s\n", tag->data.ascii);
        return;
    case TIFF_Word:
        for (int i = 0; i < tag->count; i++)
            values[i] = tag->data.word[i];
        break;
    case TIFF_DWord:
        for (int i = 0; i < tag->count; i++)
            values[i] = tag->data.dword[i];
        break;
    case TIFF_Rational:
        printf("(rational data)\n");
        return;
    default:
        return;
    }

    if (tag->count == 1) {
        printf("%d\n", values[0]);
        return;
    }

    printf("(");
    for (int i = 0; i < tag->count-1; i++)
        printf("%d, ", values[i]);
    printf("%d)\n", values[tag->count-1]);
}

static int test_image_open(const char *filename)
{
    int ret = 0;

    struct image *magick = image_open_local_magick(filename, 0, 0);
    if (magick == NULL) {
        error_perror("Unable to open TIFF with Magick Library");
        return 1;
    }

    struct image *tiff = image_open_local(filename, 0, 0);
    if (tiff == NULL) {
        error_perror("Unable to open TIFF with native parser");
        ret = 2;
        goto free_magick;
    }

    if (image_compare(magick, tiff)) {
        ret = 3;
        goto free_tiff;
    }

    struct image *cuda = image_open_cuda(filename, 0, 0, 0);
    if (cuda == NULL) {
        error_perror("Unable to open TIFF with on CUDA with native parser");
        ret = 4;
        goto free_tiff;
    }

    image_moveto(cuda, IMAGE_LOCAL);

    if (image_compare(magick, cuda)) {
        ret = 5;
    }

    image_free(cuda);


    cuda = image_open_cuda(filename, 0, 0, IMAGE_FLAG_NO_RDMA);
    if (cuda == NULL) {
        error_perror("Unable to open TIFF with on CUDA with native parser (No RDMA)");
        ret = 6;
        goto free_tiff;
    }

    image_moveto(cuda, IMAGE_LOCAL);

    if (image_compare(magick, cuda)) {
        ret = 7;
    }

    image_free(cuda);

free_tiff:
    image_free(tiff);
free_magick:
    image_free(magick);

    return ret;
}


int main(int argc, char *argv[])
{
    const char *img_path;
    char buf[PATH_MAX];

    if (argc == 1) {
        img_path = test_util_find_img("test.tiff", buf);
    } else if (argc != 2) {
        fprintf(stderr, "usage: %s [FILE]\n", argv[0]);
        return -1;
    } else {
        img_path = argv[1];
    }

    struct tiff_file *f = tiff_open(img_path);
    if (f == NULL) {
        fprintf(stderr, "Could not open tiff file '%s': %s\n",
                img_path, error_strerror(errno));
        return -1;
    }

    struct tiff_tag *tag;
    errno = 0;

    while((tag = tiff_read_tag(f)) != NULL) {
        printf("  %30s : ", tiff_tag_name(tag->id));

        print_data(tag);

        tiff_free_tag(tag);
    }

    tiff_close(f);

    if (errno) {
        perror("Error parsing TIFF file");
        return -2;
    }

    if (pinpool_init(1, 32*1024*1024)) {
        perror("Could not initialize pin pool");
        return -1;
    }

    int ret = test_image_open(img_path);

    pinpool_deinit();

    return ret;
}
