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
//     NVME device wrapper
//
////////////////////////////////////////////////////////////////////////

#include "nvme_dev.h"

#include <nvme_donard/nvme_donard.h>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <sys/time.h>

static int find_dev(dev_t dev)
{
    static struct dev_cache {
        dev_t dev;
        int fd;
    } cache[] = {[16] = {.dev=-1}};

    for (struct dev_cache *c = cache; c->dev != -1; c++)
        if (c->dev == dev)
            return c->fd;

    const char *dir = "/dev";
    DIR *dp = opendir(dir);
    if (!dp)
        return -errno;

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        if (entry->d_type != DT_UNKNOWN  && entry->d_type != DT_BLK)
            continue;

        struct stat64 st;
        if (fstatat(dirfd(dp), entry->d_name, &st, 0))
            continue;

        if (!S_ISBLK(st.st_mode))
            continue;

        if (st.st_rdev != dev)
            continue;


        int ret = openat(dirfd(dp), entry->d_name, O_RDONLY);

        for (struct dev_cache *c = cache; c->dev != -1; c++) {
            if (c->dev == 0) {
                c->dev = dev;
                c->fd = ret;
            }
        }

        closedir(dp);
        return ret;
    }

    errno = ENOENT;
    closedir(dp);
    return -1;
}

int nvme_dev_find(dev_t dev)
{
    int devfd = find_dev(dev);
    if (devfd < 0)
        return -EPERM;

    if (ioctl(devfd, NVME_IOCTL_ID, 0) < 0)
        return -ENXIO;

    return devfd;
}

int nvme_dev_get_sector_list(int fd, struct stat64 *st,
                             struct nvme_dev_sector **slist_p,
                             size_t max_size)
{
    int blk_size = st->st_blksize / 512;
    size_t size = st->st_size;
    if (max_size && size > max_size)
        size = max_size;

    unsigned long num_blocks = (size + st->st_blksize - 1) / st->st_blksize;

    int list_count = 1;

    struct nvme_dev_sector *slist;
    *slist_p = slist = malloc(num_blocks * sizeof(*slist));
    if (slist == NULL)
        return -1;

    for (int i = 0; i < num_blocks; i++) {
        unsigned long blknum = i;

        if (ioctl(fd, FIBMAP, &blknum) < 0) {
            free(slist);
            return -1;
        }

        //Seems we can't transfer more than 65536 LBAs at once so
        // in that case we split it into multiple transfers
        if (i != 0 && blknum * blk_size == slist->slba + slist->count &&
            slist->count + blk_size <= 65536) {
            slist->count += blk_size;
            continue;
        }

        if (i != 0) {
            slist++;
            list_count++;
        }

        slist->slba = blknum * blk_size;
        slist->count = blk_size;
    }

    return list_count;
}

int nvme_dev_read(int devfd, int slba, int nblocks, void *dest)
{
    char meta[nblocks*16];

    struct nvme_user_io iocmd = {
        .opcode = nvme_cmd_read,
        .slba = slba,
        .nblocks = nblocks-1,
        .addr = (__u64) dest,
        .metadata = (__u64) meta,
    };

    return ioctl(devfd, NVME_IOCTL_SUBMIT_IO, &iocmd);
}

int nvme_dev_write(int devfd, int slba, int nblocks, const void *src)
{
    char meta[nblocks*16];

    struct nvme_user_io iocmd = {
        .opcode = nvme_cmd_write,
        .slba = slba,
        .nblocks = nblocks-1,
        .addr = (__u64) src,
        .metadata = (__u64) meta,
    };

    return ioctl(devfd, NVME_IOCTL_SUBMIT_IO, &iocmd);
}

int nvme_dev_gpu_read(int devfd, int slba, int nblocks,
                      const struct pin_buf *dest,
                      unsigned long offset)
{
    struct nvme_gpu_io iocmd = {
        .opcode = nvme_cmd_read,
        .slba = slba,
        .nblocks = nblocks-1,
        .gpu_mem_handle = dest->handle,
        .gpu_mem_offset = offset,
    };

    return ioctl(devfd, NVME_IOCTL_SUBMIT_GPU_IO, &iocmd);
}

int nvme_dev_gpu_write(int devfd, int slba, int nblocks,
                       const struct pin_buf *src,
                       unsigned long offset)
{

    struct nvme_gpu_io iocmd = {
        .opcode = nvme_cmd_write,
        .slba = slba,
        .nblocks = nblocks-1,
        .gpu_mem_handle = src->handle,
        .gpu_mem_offset = offset,
    };

    return ioctl(devfd, NVME_IOCTL_SUBMIT_GPU_IO, &iocmd);
}

int nvme_dev_read_fd(int fd, void *buf, size_t bufsize)
{
    struct stat64 st;
    if (fstat(fd, &st))
        return -1;

    int devfd = nvme_dev_find(st.st_dev);
    if (devfd < 0)
        return -1;

    struct nvme_dev_sector *slist;
    int sector_count = nvme_dev_get_sector_list(fd, &st, &slist, bufsize);
    if (sector_count < 0) {
        errno = EPERM;
        return -1;
    }

    size_t txfr_size = st.st_size + st.st_blksize - 1;
    if (txfr_size > bufsize)
        txfr_size = bufsize;
    unsigned long num_blocks = txfr_size / 512;

    size_t bytes = 0;
    unsigned char *dest = buf;
    for (int i = 0; i < sector_count; i++) {
        unsigned count = slist[i].count;
        if (count > num_blocks)
            count = num_blocks;

        if (!count) break;

        if (nvme_dev_read(devfd, slist[i].slba, count, dest))
            goto free_and_exit;

        dest += slist[i].count * 512;
        bytes += slist[i].count * 512;
        num_blocks -= slist[i].count;
    }

    free(slist);
    return bytes;

free_and_exit:
    free(slist);
    return -1;
}

int nvme_dev_read_file(const char *fname, void *buf, size_t bufsize)
{
    int fd = open(fname, O_RDONLY);
    if (fd < 0)
        return -1;

    int ret = nvme_dev_read_fd(fd, buf, bufsize);

    close(fd);
    return ret;
}

int nvme_dev_write_fd(int fd, const void *buf, size_t bufsize)
{
    struct stat64 st;
    if (fstat(fd, &st))
        return -1;

    int devfd = nvme_dev_find(st.st_dev);
    if (devfd < 0)
        return -1;

    if (posix_fadvise(fd, 0, st.st_size, POSIX_FADV_DONTNEED))
        return -1;

    struct nvme_dev_sector *slist;
    int sector_count = nvme_dev_get_sector_list(fd, &st, &slist, bufsize);
    if (sector_count < 0)
        return -1;

    const unsigned char *cbuf = buf;
    unsigned long num_blocks = bufsize / 512;
    int wrote = 0;

    for (int i = 0; i < sector_count; i++) {
        int count = slist[i].count;
        if (count <= num_blocks)
            count = num_blocks;

        if (!count) break;

        if (nvme_dev_write(devfd, slist[i].slba, slist[i].count, cbuf))
            goto error_exit;

        num_blocks -= slist[i].count;
        wrote += slist[i].count * 512;
        cbuf += slist[i].count * 512;
    }

    //Updated modification and access times
    futimes(fd, NULL);

    free(slist);
    return wrote;

error_exit:
    free(slist);
    return -1;
}

int nvme_dev_write_file(const char *fname, const void *buf, size_t bufsize)
{
    int fd = open(fname, O_RDONLY);
    if (fd < 0)
        return -1;

    int ret = nvme_dev_write_fd(fd, buf, bufsize);

    close(fd);
    return ret;
}
