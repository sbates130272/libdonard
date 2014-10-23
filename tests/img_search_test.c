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
//     Image Search Test
//
////////////////////////////////////////////////////////////////////////

#include "test_util.h"

#include <imgrep/image.h>
#include <imgrep/img_search.h>
#include <imgrep/error.h>
#include <libdonard/filemap.h>

#include <string.h>
#include <stdio.h>
#include <math.h>

static int search_test(struct image *haystack, int compare)
{
    struct img_search_res result;
    if (img_search(haystack, &result)) {
        error_perror("Search Failed");
        return -1;
    }

    printf("%s %zd %zd %f\n",
           haystack->loc == IMAGE_LOCAL ? "Local" : "CUDA",
           result.x, result.y,
           result.confidence);

    if (compare) {
        if (result.x != 385 || result.y != 1253)
            return 1;

        if (result.confidence < 300)
            return 2;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    int ret = 0;

    char buf1[PATH_MAX], buf2[PATH_MAX];
    const char *needle_path, *haystack_path;;

    if (argc == 1) {
        needle_path = test_util_find_img("pmclogo.png", buf1);
        haystack_path = test_util_find_img("test_img.jpg", buf2);
    } else if (argc != 3) {
        fprintf(stderr, "USAGE: %s NEEDLE HAYSTACK.\n", argv[0]);
        exit(-1);
    } else {
        needle_path = argv[1];
        haystack_path = argv[2];
    }

    if (img_search_init(0, 1)) {
        error_perror("Could not create image search plans");
        return 1;
    }

    image_init();

    struct image *needle = image_open_local(needle_path, 0, 0);
    if (needle == NULL) {
        fprintf(stderr, "Could not open needle image '%s': %s\n",
                argv[1], error_strerror(errno));
        ret = -1;
        goto cleanup_and_exit;
    }

    struct image *haystack = image_open_local(haystack_path, 0, 0);
    if (haystack == NULL) {
        fprintf(stderr, "Could not open haystack image '%s': %s\n",
                argv[2], error_strerror(errno));
        image_free(needle);
        ret = -1;
        goto cleanup_and_exit;
    }

    image_save_full(needle, "needle.jpg");

    if (img_search_set_needle(needle)) {
        error_perror("Could not set needle");
        return -1;
    }

    image_save_full(needle, "needle_edge.jpg");

    ret = search_test(haystack, argc==1);

    image_moveto(needle, IMAGE_CUDA);

    if (img_search_set_needle(needle)) {
        error_perror("Could not set needle");
        return -1;
    }

    image_moveto(haystack, IMAGE_CUDA);

    ret |= search_test(haystack, argc==1);

    image_free(haystack);
    image_free(needle);

cleanup_and_exit:
    image_deinit();
    img_search_deinit();

    return ret;
}
