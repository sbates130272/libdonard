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
//     File mapper test code.
//
////////////////////////////////////////////////////////////////////////


#include <libdonard/filemap.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#include <string.h>
#include <stdio.h>

#include <cuda_runtime.h>

static int test_local_maps(int fd, const char *fname)
{
    int ret = 0;
    struct filemap * nvme_map = filemap_alloc_local_nvme(fd, fname);

    if (!nvme_map) {
        perror("Could not create nvme file map");
        return -1;
    }

    if (nvme_map->map_error) {
        fprintf(stderr, "Could not allocate nvme file map: %s\n",
                filemap_map_error_string(nvme_map->map_error));

        filemap_free(nvme_map);
        return -1;
    }

    struct filemap * mmap_map = filemap_alloc_local(fd, fname);
    if (!mmap_map) {
        perror("Could not create mmap file map");
        filemap_free(nvme_map);
        return -1;
    }

    if (mmap_map->length != nvme_map->length) {
        fprintf(stderr, "Map lengths do not match!\n");
        ret = -1;
        goto free_and_return;
    }

    if (memcmp(mmap_map->data, nvme_map->data, nvme_map->length) == 0) {
        printf("Success: Local maps match. (%zd)\n", nvme_map->length);
    } else {
        printf("Failed: Local maps don't match!\n");
        ret = -1;
    }

free_and_return:
    filemap_free(mmap_map);
    filemap_free(nvme_map);
    return ret;
}

static int test_cuda_map(int fd, const char *fname)
{
    int ret = 0;
    struct filemap *cuda_map = filemap_alloc_cuda(fd, fname);

    if (!cuda_map) {
        perror("Could not create cuda file map");
        return -1;
    }

    if (cuda_map->map_error) {
        fprintf(stderr, "Could not allocate cuda file map: %s\n",
                filemap_map_error_string(cuda_map->map_error));

        filemap_free(cuda_map);
        return -1;
    }

    struct filemap * mmap_map = filemap_alloc_local(fd, fname);
    if (!mmap_map) {
        perror("Could not create mmap file map");
        filemap_free(cuda_map);
        return -1;
    }

    unsigned char *hostbuf = malloc(cuda_map->length);
    cudaMemcpy(hostbuf, cuda_map->data, cuda_map->length,
               cudaMemcpyDeviceToHost);

    if (mmap_map->length != cuda_map->length) {
        fprintf(stderr, "Map lengths do not match!\n");
        ret = -1;
        goto free_and_return;
    }

    if (memcmp(mmap_map->data, hostbuf, cuda_map->length) == 0) {
        printf("Success: CUDA map matches. (%zd)\n", cuda_map->length);
    } else {
        printf("Failed: CUDA map matches!\n");
        ret = -1;
    }

free_and_return:
    free(hostbuf);
    filemap_free(mmap_map);
    filemap_free(cuda_map);
    return ret;
}

static int test_cuda_nvme_map(int fd, const char *fname)
{
    int ret = 0;
    struct filemap * nvme_map = filemap_alloc_cuda_nvme(fd, fname);

    if (!nvme_map) {
        perror("Could not create nvme file map");
        return -1;
    }

    if (nvme_map->map_error) {
        fprintf(stderr, "Could not allocate nvme file map: %s\n",
                filemap_map_error_string(nvme_map->map_error));

        filemap_free(nvme_map);
        return -1;
    }

    struct filemap * mmap_map = filemap_alloc_local(fd, fname);
    if (!mmap_map) {
        perror("Could not create mmap file map");
        filemap_free(nvme_map);
        return -1;
    }

    unsigned char *hostbuf = malloc(nvme_map->length);
    cudaMemcpy(hostbuf, nvme_map->data, nvme_map->length,
               cudaMemcpyDeviceToHost);

    if (mmap_map->length != nvme_map->length) {
        fprintf(stderr, "Map lengths do not match!\n");
        ret = -1;
        goto free_and_return;
    }

    if (memcmp(mmap_map->data, hostbuf, nvme_map->length) == 0) {
        printf("Success: NVME CUDA map matches. (%zd)\n", nvme_map->length);
    } else {
        printf("Failed: NVME CUDA map matches!\n");
        ret = -1;
    }

free_and_return:
    free(hostbuf);
    filemap_free(mmap_map);
    filemap_free(nvme_map);
    return ret;
}


int main(int argc, char *argv[])
{
    const char *fname;
    if (argc != 2) {
        fname = "/mnt/princeton/random_test3";
        if (access(fname, F_OK)) {
            fprintf(stderr, "File does not exist: %s\n", fname);
            return -1;
        }
    } else {
        fname = argv[1];
    }

    if (pinpool_init(1, 64*1024*1024)) {
        perror("Could not initialize pin pool");
        return -1;
    }

    int fd = open(fname, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error opening file '%s': %s\n", argv[1], strerror(errno));
        return -1;
    }

    int ret = 0;
    if ((ret = test_local_maps(fd, argv[1])))
        goto leave;

    if ((ret = test_cuda_map(fd, argv[1])))
        goto leave;

    if ((ret = test_cuda_nvme_map(fd, argv[1])))
        goto leave;

    pinpool_deinit();

    if (pinpool_init(1, 1*1024*1024)) {
        perror("Could not initialize pin pool");
        return -1;
    }

    printf("Testing large file support with 1MB pin buffer size\n");

    if ((ret = test_cuda_nvme_map(fd, argv[1])))
        goto leave;



leave:

    pinpool_deinit();

    return ret;
}
