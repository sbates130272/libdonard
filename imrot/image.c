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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>


#include <string.h>

void image_init(void)
{
}

void image_deinit(void)
{
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

const static size_t bufsize(const struct image *img)
{
    return img->width * img->height * sizeof(*img->buf);
}

void *alloc_buf_size(struct image *img, size_t width, size_t height)
{
    void *ret;

    switch(img->loc) {
    case IMAGE_LOCAL:
        return malloc(width * height * sizeof(*img->buf));
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
    return alloc_buf_size(img, img->width, img->height);
}

static void free_buf(image_px *buf, enum image_loc loc)
{
    switch(loc) {
    case IMAGE_LOCAL:
        free(buf);
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
    img->fmap = NULL;
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

    if (img->fmap) {
        filemap_free(img->fmap);
    } else {
        free_buf(img->buf, img->loc);
    }

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

    img->width = width;
    img->height = height;
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
    img->width = width;
    img->height = height;
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
        ret = image_new_local(img->width, img->height);
    } else if (img->loc == IMAGE_CUDA) {
        ret = image_new_cuda(img->width, img->height);
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
        ret = image_znew_local(img->width, img->height);
    } else if (img->loc == IMAGE_CUDA) {
        ret = image_znew_cuda(img->width, img->height);
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
    img->filesize = fmap->length;
    img->loc = loc;
    copy_filename(img, fmap->filename);

    unsigned char *cdata = fmap->data;
    cdata += offset;

    if (loc == IMAGE_CUDA) {
        img->fmap = fmap;
        img->buf = cdata;
    } else {
        img->buf = alloc_buf(img);
        if (img->buf == NULL)
            goto free_map_and_exit;
        memcpy(img->buf, cdata, bufsize(img));
        filemap_free(fmap);
    }

    return img;

free_map_and_exit:
    filemap_free(fmap);

    return NULL;
}

struct image *image_open_local(const char *filename, size_t bufwidth,
                               size_t bufheight)
{
    return open_tiff(filename, bufwidth, bufheight, IMAGE_LOCAL, 0);
}

struct image *image_open_cuda(const char *filename, size_t bufwidth,
                              size_t bufheight, int flags)
{
    return open_tiff(filename, bufwidth, bufheight, IMAGE_CUDA, flags);
}

static void copy_struct(struct image *dst, const struct image *src)
{
    dst->width = src->width;
    dst->height = src->height;
    dst->filesize = src->filesize;
    copy_filename(dst, src->filename);
    dst->loc = src->loc;

    stream_ref(src->stream);
    dst->stream = src->stream;
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

    if (img->fmap) {
        filemap_free(img->fmap);
    } else {
        free_buf(img->buf, img->loc);
    }

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

void image_set_stream(struct image *img, struct image *stream_img)
{
    stream_free(img->stream);
    stream_ref(stream_img->stream);
    img->stream = stream_img->stream;
}

int image_save(struct image *img, int flags)
{
    int fd = open(img->filename, O_WRONLY);
    if (fd < 0)
        return -1;

    if (img->loc == IMAGE_CUDA && !(flags & IMAGE_FLAG_NO_RDMA)) {
        if (filemap_write_cuda_nvme(img->fmap, fd) == 0)
            return 0;
    }

    size_t width, height, offset, count;
    if (parse_tiff_header(img->filename, &width, &height, &offset, &count))
        return -1;

    image_px *src = img->buf;

    if (img->loc == IMAGE_CUDA) {
        src = malloc(bufsize(img));
        cudaMemcpy(src, img->buf, bufsize(img), cudaMemcpyDeviceToHost);
    }

    lseek(fd, offset, SEEK_SET);
    write(fd, src, bufsize(img));

    //In order to be fair with the donard version force syncing the file to disk
    fsync(fd);

    close(fd);

    if (img->loc == IMAGE_CUDA)
        free(src);

    return 0;
}
