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

#ifndef __IMGREP_IMAGE_H__
#define __IMGREP_IMAGE_H__

#include "image_px.h"

#include <libdonard/filemap.h>

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include <cuda_runtime.h>

enum image_loc {
    IMAGE_LOCAL,
    IMAGE_CUDA,
};

struct stream;

struct image {
    size_t width, height;
    size_t bufwidth, bufheight;
    size_t filesize;
    const char *filename;
    enum image_loc loc;
    image_px *buf;
    struct stream *stream;

    pthread_mutex_t mutex;
    int refs;
};

#define IMAGE_PX(img, pixels)   image_px (*pixels)[img->bufwidth] = \
    (void *) img->buf


#ifdef __cplusplus
extern "C" {
#endif

void image_init(void);
void image_deinit(void);

void image_ref(struct image *img);
void image_free(struct image *img);

struct image *image_new_local(size_t width, size_t height);
struct image *image_new_cuda(size_t width, size_t height);
struct image *image_new_dup(struct image *img);

//Non-zeroed version of image_new
struct image *image_znew_local(size_t width, size_t height);
struct image *image_znew_cuda(size_t width, size_t height);
struct image *image_znew_dup(struct image *img);

enum {
    IMAGE_FLAG_NO_RDMA = 1,
};

struct image *image_load_local(struct filemap *fmap, size_t bufwidth,
                               size_t bufheight);
struct image *image_open_local(const char *filename, size_t bufwidth,
                               size_t bufheight);
struct image *image_open_local_magick(const char *filename, size_t bufwidth,
                                      size_t bufheight);
struct image *image_open_cuda(const char *filename, size_t bufwidth,
                              size_t bufheight, int flags);

int image_save(const struct image *img, const char *fname);
int image_save_full(struct image *img, const char *fname);

void image_fprint(FILE *f, const struct image *img);
#define image_print(img)  image_fprint(stdout, img)

struct image * image_clone(const struct image *img);
int image_moveto(struct image *img, enum image_loc newloc);
struct image *image_copyto(const struct image *img, enum image_loc newloc);

int image_reshape(struct image *img, size_t bufwidth, size_t bufheight);
struct image *image_reshape_clone(struct image *img, size_t bufwidth,
                                  size_t bufheight);
int image_rot180(struct image *img);

image_px image_max(const struct image *img, size_t *x, size_t *y);
image_px image_min(const struct image *img, size_t *x, size_t *y);

int image_compare(struct image *a, struct image *b);

void image_sync(const struct image *img);
cudaStream_t image_stream(const struct image *img);

void image_set_pixel(struct image *img, size_t x, size_t y, image_px value);

void image_set_stream(struct image *img, struct image *stream_img);

int image_load_bytes(struct image *img, void *bytes, size_t width,
                      size_t height);

#ifdef __cplusplus
}
#endif

#endif
