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

#ifndef P2MTR_NVME_DEV_H
#define P2MTR_NVME_DEV_H

#include "pinpool.h"

#include <sys/stat.h>
#include <stdlib.h>

int nvme_dev_find(dev_t dev);

struct nvme_dev_sector {
    unsigned long slba;
    unsigned long count;
};

int nvme_dev_get_sector_list(int fd, struct stat64 *st,
                             struct nvme_dev_sector **slist,
                             size_t max_size);

int nvme_dev_read(int devfd, int slba, int nblocks, void *dest);
int nvme_dev_write(int devfd, int slba, int nblocks, const void *src);
int nvme_dev_gpu_read(int devfd, int slba, int nblocks,
                      const struct pin_buf *dest,
                      unsigned long offset);
int nvme_dev_gpu_write(int devfd, int slba, int nblocks,
                       const struct pin_buf *src,
                       unsigned long offset);

int nvme_dev_read_fd(int fd, void *buf, size_t bufsize);
int nvme_dev_read_file(const char *fname, void *buf, size_t bufsize);

int nvme_dev_write_fd(int fd, const void *buf, size_t bufsize);
int nvme_dev_write_file(const char *fname, const void *buf, size_t bufsize);

#endif
