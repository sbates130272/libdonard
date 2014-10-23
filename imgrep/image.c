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

#include "image.h"
#include "image_cuda.h"
#include "error.h"

#include <libdonard/tiff.h>

#include <wand/magick_wand.h>
#include <fftw3.h>

#include <string.h>

PixelWand *bg_colour;

void image_init(void)
{
    MagickWandGenesis();

    bg_colour = NewPixelWand();
    PixelSetColor(bg_colour, "black");
}

void image_deinit(void)
{
    DestroyPixelWand(bg_colour);

    MagickWandTerminus();
}

struct stream {
    cudaStream_t stream;
    int refs;
};

static struct stream *stream_create(void)
{
    struct stream *ret = malloc(sizeof(*ret));

    if (ret == NULL)
        return NULL;

    ret->refs = 1;
    if (cudaStreamCreate(&ret->stream) != cudaSuccess) {
        errno = ECUDA_ERROR;
        free(ret);
        return NULL;
    }

    return ret;
}

static void stream_ref(struct stream *s)
{
    if (s == NULL) return;

    s->refs++;
}

static void stream_free(struct stream *s)
{
    if (s == NULL) return;

    s->refs--;
    if (s->refs != 0) return;

    cudaStreamDestroy(s->stream);
    free(s);
}

static cudaStream_t stream_get(struct stream *s)
{
    return (s != NULL) ? s->stream : NULL;
}

static int next_highest_pow2(int v)
{
    if (v < 128) return 128;

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

const static size_t bufsize(const struct image *img)
{
    return img->bufwidth * img->bufheight * sizeof(*img->buf);
}

void *alloc_buf_size(struct image *img, size_t width, size_t height)
{
    void *ret;

    switch(img->loc) {
    case IMAGE_LOCAL:
        return FFTW(alloc_real)(width * height);
    case IMAGE_CUDA:
        if (cudaMalloc(&ret, width * height * sizeof(*img->buf)) != cudaSuccess) {
            errno = ECUDA_ERROR;
            return NULL;
        }
        return ret;
    }

    return NULL;
}


void *alloc_buf(struct image *img)
{
    return alloc_buf_size(img, img->bufwidth, img->bufheight);
}

static void free_buf(image_px *buf, enum image_loc loc)
{
    switch(loc) {
    case IMAGE_LOCAL:
        FFTW(free)(buf);
        break;
    case IMAGE_CUDA:
        cudaFree(buf);
        break;
    }
}

static struct image *alloc_image(void)
{
    struct image *img = malloc(sizeof(*img));
    if (img == NULL)
        return NULL;

    img->refs = 1;
    img->stream = NULL;
    img->filename = NULL;
    img->filesize = 0;

    if (pthread_mutex_init(&img->mutex, NULL))
        goto error_exit;

    return img;

error_exit:
    free(img);
    return NULL;
}

void image_ref(struct image *img)
{
    pthread_mutex_lock(&img->mutex);
    img->refs++;
    pthread_mutex_unlock(&img->mutex);
}

void image_free(struct image *img)
{
    int refs;
    pthread_mutex_lock(&img->mutex);
    img->refs--;
    refs = img->refs;
    pthread_mutex_unlock(&img->mutex);

    if (refs != 0)
        return;

    image_sync(img);

    free_buf(img->buf, img->loc);

    if (img->filename != NULL)
        free((void *) img->filename);

    stream_free(img->stream);

    free(img);
}

struct image *image_znew_local(size_t width, size_t height)
{
    struct image *img = alloc_image();
    if (img == NULL)
        return NULL;

    img->width = img->bufwidth = width;
    img->height = img->bufheight = height;
    img->loc = IMAGE_LOCAL;

    img->buf = alloc_buf(img);
    if (img->buf == NULL)
        goto free_img_and_exit;

    return img;

free_img_and_exit:
    free(img);
    return NULL;
}

struct image *image_znew_cuda(size_t width, size_t height)
{
    struct image *img = alloc_image();
    if (img == NULL)
        return NULL;

    img->stream = stream_create();
    img->width = img->bufwidth = width;
    img->height = img->bufheight = height;
    img->loc = IMAGE_CUDA;

    img->buf = alloc_buf(img);
    if (img->buf == NULL)
        goto free_img_and_exit;

    return img;

free_img_and_exit:
    free(img);
    return NULL;
}

struct image *image_new_local(size_t width, size_t height)
{
    struct image *img = image_znew_local(width, height);
    if (img == NULL)
        return img;

    memset(img->buf, 0, bufsize(img));

    return img;
}

struct image *image_new_cuda(size_t width, size_t height)
{
    struct image *img = image_znew_cuda(width, height);
    if (img == NULL)
        return img;

    cudaMemsetAsync(img->buf, 0, bufsize(img), image_stream(img));

    return img;
}

struct image *image_new_dup(struct image *img)
{
    struct image *ret;

    if (img->loc == IMAGE_LOCAL) {
        ret = image_new_local(img->bufwidth, img->bufheight);
    } else if (img->loc == IMAGE_CUDA) {
        ret = image_new_cuda(img->bufwidth, img->bufheight);
    } else {
        errno = EINVAL;
        return NULL;
    }

    if (ret == NULL)
        return ret;

    image_set_stream(ret, img);

    return ret;
}

struct image *image_znew_dup(struct image *img)
{
    struct image *ret;

    if (img->loc == IMAGE_LOCAL) {
        ret = image_znew_local(img->bufwidth, img->bufheight);
    } else if (img->loc == IMAGE_CUDA) {
        ret = image_znew_cuda(img->bufwidth, img->bufheight);
    } else {
        errno = EINVAL;
        return NULL;
    }

    if (ret == NULL)
        return ret;

    image_set_stream(ret, img);

    return ret;
}


static void copy_filename(struct image *img, const char *fname)
{
    if (fname == NULL) {
        img->filename = NULL;
        return;
    }

    img->filename = malloc(strlen(fname)+1);
    if (img->filename == NULL)
        return;

    strcpy((char *) img->filename, fname);
}

static MagickWand *merge_transparent_pixels(MagickWand *mw)
{
    MagickSetImageBackgroundColor(mw, bg_colour);
    MagickWand *oldmw = mw;
    mw = MagickMergeImageLayers(mw, FlattenLayer);
    DestroyMagickWand(oldmw);
    if (mw == NULL) {
        errno = EBADIMAGE;
        return NULL;
    }

    return mw;
}

struct image *image_load_local(struct filemap *fmap, size_t bufwidth,
                               size_t bufheight)
{
    if (fmap->type != FILEMAP_TYPE_LOCAL) {
        errno = EINVAL;
        return NULL;
    }

    MagickWand *mw = NewMagickWand();
    if (mw == NULL)
        return NULL;

    if (MagickReadImageBlob(mw, fmap->data, fmap->length) == MagickFalse) {
        errno = EBADIMAGE;
        goto free_mw_and_exit;
    }

    if (MagickGetImageAlphaChannel(mw)) {
        mw = merge_transparent_pixels(mw);
        if (mw == NULL) return NULL;
    }

    if (MagickTransformImageColorspace(mw, GRAYColorspace) == MagickFalse) {
        errno = ETRANSFORMFAIL;
        goto free_mw_and_exit;
    }

    struct image *img = alloc_image();
    if (img == NULL)
        goto free_mw_and_exit;

    img->width = MagickGetImageWidth(mw);
    img->height = MagickGetImageHeight(mw);
    img->bufwidth = bufwidth ? bufwidth : next_highest_pow2(img->width);
    img->bufheight = bufheight ? bufheight : next_highest_pow2(img->height);
    img->filesize = fmap->length;
    img->loc = IMAGE_LOCAL;
    copy_filename(img, fmap->filename);

    img->buf = alloc_buf(img);
    if (img->buf == NULL)
        goto free_img_and_exit;

    MagickSetImageVirtualPixelMethod(mw, BlackVirtualPixelMethod);

    if (MagickExportImagePixels(mw, 0, 0, img->bufwidth,
                                img->bufheight,
                                "I", IMAGE_PX_STORAGE,
                                img->buf) == MagickFalse)
        goto free_buf_and_exit;


    DestroyMagickWand(mw);
    return img;

free_buf_and_exit:
    free_buf(img->buf, img->loc);

free_img_and_exit:
    free(img);

free_mw_and_exit:

    DestroyMagickWand(mw);
    return NULL;
}

static int parse_tiff_header(const char *filename, size_t *width,
                             size_t *height, size_t *offset, size_t *count)
{
    struct tiff_file *f = tiff_open(filename);
    if (f == NULL)
        return -1;

    *width = *height = *offset = *count = 0;
    errno = 0;
    struct tiff_tag *tag;
    while((tag = tiff_read_tag(f)) != NULL) {
        int value;
        if (tag->type == TIFF_Byte) {
            value = tag->data.byte[0];
        } else if (tag->type == TIFF_Word) {
            value = tag->data.word[0];
        } else if (tag->type == TIFF_DWord) {
            value = tag->data.dword[0];
        } else {
            tiff_free_tag(tag);
            continue;
        }

        int unsupported = 0;
        switch (tag->id) {
        case TIFF_ImageWidth:      *width = value;            break;
        case TIFF_ImageLength:     *height = value;           break;
        case TIFF_BitsPerSample:   unsupported = value != 8;  break;
        case TIFF_SampleFormat:    unsupported = value != 1;  break;
        case TIFF_Compression:     unsupported = value != 1;  break;
        case TIFF_StripByteCounts: *count = value;            break;
        case TIFF_StripOffsets:
            *offset = value;
            unsupported = tag->count != 1;
            break;
        default:
            break;
        };


        tiff_free_tag(tag);

        if (unsupported) {
            errno = EUNSUPPORTED_TIFF;
            break;
        }
    }

    if (*count != *width * *height)
        errno = EUNSUPPORTED_TIFF;

    tiff_close(f);
    if (errno)
        return -1;

    return 0;
}


static struct image *open_tiff(const char *filename, size_t bufwidth,
                               size_t bufheight, enum image_loc loc,
                               int flags)
{
    size_t width, height, offset, count;
    if (parse_tiff_header(filename, &width, &height, &offset, &count))
        return NULL;

    struct filemap *fmap;
    switch (loc) {
    case IMAGE_LOCAL: fmap = filemap_open_local(filename);     break;
    case IMAGE_CUDA:
        if (flags & IMAGE_FLAG_NO_RDMA)
            fmap = filemap_open_cuda(filename);
        else
            fmap = filemap_open_cuda_nvme(filename);
        break;
    default: return NULL;
    }

    if (fmap == NULL)
        return NULL;

    struct image *img = alloc_image();
    if (img == NULL)
        goto free_map_and_exit;

    img->width = width;
    img->height = height;
    img->bufwidth = bufwidth ? bufwidth : next_highest_pow2(width);
    img->bufheight = bufheight ? bufheight : next_highest_pow2(height);
    img->filesize = fmap->length;
    img->loc = loc;
    copy_filename(img, fmap->filename);

    img->buf = alloc_buf(img);
    if (img->buf == NULL)
        goto free_img_and_exit;

    unsigned char *cdata = fmap->data;
    cdata += offset;

    if (image_load_bytes(img, cdata, width, height))
        goto free_buf_and_exit;

    filemap_free(fmap);

    return img;


free_buf_and_exit:
    free_buf(img->buf, img->loc);

free_img_and_exit:
    free(img);

free_map_and_exit:
    filemap_free(fmap);

    return NULL;
}

static int has_tiff_extension(const char *filename)
{
    const char *extn = strrchr(filename, '.');
    if (extn == NULL)
        return 0;

    if (strcasecmp(extn, ".tiff") == 0)
        return 1;

    if (strcasecmp(extn, ".tif") == 0)
        return 1;

    return 0;
}

struct image *image_open_local_magick(const char *filename, size_t bufwidth,
                                      size_t bufheight)
{
    struct filemap *map = filemap_open_local(filename);
    if (map == NULL)
        return NULL;

    struct image *img = image_load_local(map, bufwidth, bufheight);
    filemap_free(map);
    return img;
}

struct image *image_open_local(const char *filename, size_t bufwidth,
                               size_t bufheight)
{
    if (has_tiff_extension(filename))
        return open_tiff(filename, bufwidth, bufheight, IMAGE_LOCAL, 0);

    return image_open_local_magick(filename, bufwidth, bufheight);
}

struct image *image_open_cuda(const char *filename, size_t bufwidth,
                              size_t bufheight, int flags)
{
    if (has_tiff_extension(filename))
        return open_tiff(filename, bufwidth, bufheight, IMAGE_CUDA, flags);

    struct image *img = image_open_local_magick(filename, bufwidth, bufheight);
    if (img == NULL)
        return NULL;

    image_moveto(img, IMAGE_CUDA);

    return img;
}

int image_save(const struct image *img, const char *fname)
{
    int ret = 0;

    image_sync(img);

    image_px mx = image_max(img, NULL, NULL);
    image_px mn = image_min(img, NULL, NULL);

    MagickWand *mw = NewMagickWand();
    if (mw == NULL)
        return -errno;

    PixelWand *bg = NewPixelWand();
    if (bg == NULL)
        goto free_and_exit;

    if (MagickNewImage(mw, img->width, img->height, bg) == MagickFalse) {
        ret = -1;
        DestroyPixelWand(bg);
        goto free_and_exit;
    }

    DestroyPixelWand(bg);

    if (MagickSetImageColorspace(mw, GRAYColorspace) == MagickFalse) {
        ret = -2;
        goto free_and_exit;
    }

    struct image *img2 = image_clone(img);
    if (img2 == NULL) {
        ret = -errno;
        goto free_and_exit;
    }

    if (mx > 1.0 || mn < 0.0) {
        for (size_t i = 0; i < img2->bufwidth * img2->bufheight; i++) {
            img2->buf[i] += -mn;
            img2->buf[i] /= (mx - mn);
        }
    }

    if (MagickImportImagePixels(mw, 0, 0, img2->bufwidth, img2->bufheight,
                                "I", IMAGE_PX_STORAGE,
                                img2->buf) == MagickFalse)
    {
        ret = -3;
        goto free_image_and_exit;
    }

    if (MagickWriteImage(mw, fname) == MagickFalse)
        ret = -4;

free_image_and_exit:
    image_free(img2);

free_and_exit:
    DestroyMagickWand(mw);
    return ret;
}

int image_save_full(struct image *img, const char *fname)
{
    size_t w = img->width;
    size_t h = img->height;

    img->width = img->bufwidth;
    img->height = img->bufheight;

    int ret = image_save(img, fname);

    img->width = w;
    img->height = h;

    return ret;
}

static void print_row(FILE *f, const struct image *img, int r)
{
    IMAGE_PX(img, p);

    fprintf(f, r == 0 ? "[" : " ");
    fprintf(f, "[ ");
    for (int c = 0; c < 3; c++)
        fprintf(f, " %8g", p[r][c]);

    fprintf(f, " ..., ");

    for (int c = img->width-3; c < img->width; c++)
        fprintf(f, " %6g", p[r][c]);
    fprintf(f, "]");
}

void image_fprint(FILE *f, const struct image *img)
{
    for (size_t r = 0; r < 3; r++) {
        print_row(f, img, r);
        fprintf(f, "\n");
    }

    fprintf(f, " ..., \n");

    for (size_t r = img->height-3; r < img->height; r++) {
        print_row(f, img, r);
        if (r == img->height-1)
            fprintf(f, "]");
        fprintf(f, "\n");
    }
}

static image_px *reshape(struct image *img, size_t bufwidth, size_t bufheight)
{
    if (img->loc == IMAGE_CUDA)
        return image_cuda_reshape(img, bufwidth, bufheight);

    image_px (*newbuf)[bufwidth] = alloc_buf_size(img, bufwidth, bufheight);
    if (newbuf == NULL)
        return NULL;

    IMAGE_PX(img, pixels);

    size_t h = img->bufheight > bufheight ? bufheight : img->bufheight;
    size_t w = img->bufwidth > bufwidth ? bufwidth : img->bufwidth;

    size_t r;
    for (r = 0; r < h; r++) {
        size_t c;
        for (c = 0; c < w; c++)
            newbuf[r][c] = pixels[r][c];

        for (; c < bufwidth; c++)
            newbuf[r][c] = 0;
    }

    for (; r < bufheight; r++)
        for (size_t c = 0; c < bufwidth; c++)
            newbuf[r][c] = 0;

    return (image_px *) newbuf;
}

int image_reshape(struct image *img, size_t bufwidth, size_t bufheight)
{
    image_px *newbuf = reshape(img, bufwidth, bufheight);

    if (newbuf == NULL)
        return -errno;

    image_sync(img);
    free_buf(img->buf, img->loc);
    img->buf = newbuf;
    img->bufheight = bufheight;
    img->bufwidth = bufwidth;

    return 0;
}

static void copy_struct(struct image *dst, const struct image *src)
{
    dst->width = src->width;
    dst->height = src->height;
    dst->bufwidth = src->bufwidth;
    dst->bufheight = src->bufheight;
    dst->filesize = src->filesize;
    copy_filename(dst, src->filename);
    dst->loc = src->loc;

    stream_ref(src->stream);
    dst->stream = src->stream;
}

struct image *image_reshape_clone(struct image *img, size_t bufwidth,
                                  size_t bufheight)
{
    struct image *clone = alloc_image();
    if (clone == NULL)
        return NULL;

    image_px *newbuf = reshape(img, bufwidth, bufheight);
    if (newbuf == NULL) {
        free(clone);
        return NULL;
    }

    copy_struct(clone, img);
    clone->buf = newbuf;
    clone->bufheight = bufheight;
    clone->bufwidth = bufwidth;

    return clone;
}

int image_rot180(struct image *img)
{
    if (img->loc == IMAGE_CUDA) {
        if (image_rot180_cuda(img) != cudaSuccess) {
            errno = ECUDA_ERROR;
            return -errno;
        }

        return 0;
    }

    IMAGE_PX(img, pixels);

    size_t h = img->height-1;
    size_t w = img->width-1;

    for (size_t r = 0; r < img->height; r++) {
        for (size_t c = 0; c < img->width/2; c++) {
            float tmp;
            tmp = pixels[r][c];
            pixels[r][c] = pixels[h-r][w-c];
            pixels[h-r][w-c] = tmp;
        }
    }

    size_t i = img->width/2;
    if (i * 2 == img->width)
        return 0 ;

    for (size_t r = 0; r < img->height/2; r++) {
        float tmp;
        tmp = pixels[r][i];
        pixels[r][i] = pixels[h-r][i];
        pixels[h-r][i] = tmp;
    }

    return 0;
}

image_px image_max(const struct image *img, size_t *x, size_t *y)
{
    if (img->loc == IMAGE_CUDA)
        return image_cuda_max(img, x, y);

    image_px ret = img->buf[0];
    size_t loc = 0;

    for (size_t i = 1; i < img->bufwidth * img->bufheight; i++) {
        if (img->buf[i] > ret) {
            ret = img->buf[i];
            loc = i;
        }
    }

    if (y != NULL) *y = loc / img->bufwidth;
    if (x != NULL) *x = loc % img->bufwidth;

    return ret;
}

image_px image_min(const struct image *img, size_t *x, size_t *y)
{
    if (img->loc == IMAGE_CUDA)
        return image_cuda_min(img, x, y);

    image_px ret = img->buf[0];
    size_t loc = 0;

    for (size_t i = 1; i < img->bufwidth * img->bufheight; i++) {
        if (img->buf[i] < ret) {
            ret = img->buf[i];
            loc = i;
        }
    }

    if (y != NULL) *y = loc / img->bufwidth;
    if (x != NULL) *x = loc % img->bufwidth;

    return ret;
}

static int cuda_memcpy(const struct image *src, image_px *destbuf,
                       enum image_loc destloc, struct stream *s)
{
    enum cudaMemcpyKind kind;
    if (src->loc == IMAGE_LOCAL && destloc == IMAGE_LOCAL)
        kind = cudaMemcpyHostToHost;
    else if (src->loc == IMAGE_LOCAL && destloc == IMAGE_CUDA)
        kind = cudaMemcpyHostToDevice;
    else if (src->loc == IMAGE_CUDA && destloc == IMAGE_LOCAL)
        kind = cudaMemcpyDeviceToHost;
    else
        kind = cudaMemcpyDeviceToDevice;

    if (cudaMemcpyAsync(destbuf, src->buf, bufsize(src), kind,
                        stream_get(s)) != cudaSuccess)
    {
        errno = ECUDA_ERROR;
        return -errno;
    }

    return 0;
}

int image_moveto(struct image *img, enum image_loc newloc)
{
    if (img->loc == newloc)
        return 0;

    enum image_loc oldloc = img->loc;
    img->loc = newloc;
    image_px *newbuf = alloc_buf(img);
    img->loc = oldloc;

    if (newbuf == NULL)
        return -errno;

    image_sync(img);

    if (img->stream == NULL && newloc == IMAGE_CUDA)
        img->stream = stream_create();

    if (cuda_memcpy(img, newbuf, newloc, 0)) {
        free_buf(newbuf, newloc);
        return -errno;
    }

    image_sync(img);

    free_buf(img->buf, img->loc);

    img->buf = newbuf;
    img->loc = newloc;

    return 0;
}

struct image *image_copyto(const struct image *img, enum image_loc newloc)
{
    struct image *clone = alloc_image();
    if (clone == NULL)
        return NULL;

    copy_struct(clone, img);
    clone->loc = newloc;

    clone->buf = alloc_buf(clone);
    if (clone->buf == NULL)
        goto free_exit;

    if (clone->stream == NULL && newloc == IMAGE_CUDA)
        clone->stream = stream_create();

    if (cuda_memcpy(img, clone->buf, clone->loc, clone->stream))
        goto free_buf_exit;

    return clone;

free_buf_exit:
    free_buf(clone->buf, clone->loc);

free_exit:
    free(clone);
    return NULL;
}

struct image *image_clone(const struct image *img)
{
    return image_copyto(img, img->loc);
}

int image_compare(struct image *a, struct image *b)
{
    image_sync(a);
    image_sync(b);

    if (a->loc != IMAGE_LOCAL || b->loc != IMAGE_LOCAL)
        return 1;

    if (a->width != b->width || a->height != b->height)
        return 2;

    if (a->bufwidth != b->bufwidth || a->bufheight != b->bufheight)
        return 3;

    IMAGE_PX(a, ap);
    IMAGE_PX(b, bp);
    for (size_t y = 0; y < a->bufheight; y++)
        for (size_t x = 0; x < a->bufwidth; x++)
            if (fabs(ap[y][x] - bp[y][x]) > 1e-5)
                return 4;

    return 0;
}

void image_sync(const struct image *img)
{
    if (img->stream == NULL)
        return;

    cudaStreamSynchronize(img->stream->stream);
}

cudaStream_t image_stream(const struct image *img)
{
    if (img->stream == NULL)
        return NULL;

    return img->stream->stream;
}

void image_set_pixel(struct image *img, size_t x, size_t y, image_px value)
{
    if (img->loc == IMAGE_LOCAL) {
        IMAGE_PX(img, p);
        p[y][x] = value;
    } else if (img->loc == IMAGE_CUDA) {
        IMAGE_PX(img, p);
        cudaMemcpyAsync(&p[y][x], &value, sizeof(value),
                        cudaMemcpyHostToDevice, image_stream(img));
    }
}

void image_set_stream(struct image *img, struct image *stream_img)
{
    stream_free(img->stream);
    stream_ref(stream_img->stream);
    img->stream = stream_img->stream;
}

int image_load_bytes(struct image *img, void *bytes, size_t width,
                     size_t height)
{
    img->width = width;
    img->height = height;

    if (width > img->bufwidth || height > img->bufheight) {
        errno = EINVAL;
        return -EINVAL;
    }

    if (img->loc == IMAGE_CUDA)
        return image_cuda_load_bytes(img, bytes, width, height);

    unsigned char (*src)[width] = bytes;
    IMAGE_PX(img, dest);

    size_t r;
    for (r = 0; r < height; r++) {
        size_t c;
        for (c = 0; c < width; c++)
            dest[r][c] = src[r][c] / 255.0;

        for (; c < img->bufwidth; c++)
            dest[r][c] = 0;
    }

    for (; r < img->bufheight; r++)
        for (size_t c = 0; c < img->bufwidth; c++)
            dest[r][c] = 0;

    return 0;
}
