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
//     NVME Dev test
//
////////////////////////////////////////////////////////////////////////

#include <libdonard/nvme_dev.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

#include <string.h>
#include <stdio.h>

static int test_read(const char *fname)
{
    struct stat st;
    stat(fname, &st);

    size_t bufsize = st.st_size + 4096;

    void *buf1 = malloc(bufsize);
    if (!buf1) return -1;

    void *buf2 = malloc(bufsize);
    if (!buf2) {
        free(buf1);
        return -1;
    }

    int nvme_read = nvme_dev_read_file(fname, buf1, bufsize);

    int fd = open(fname, O_RDONLY);
    int reg_read = read(fd, buf2, bufsize);
    close(fd);

    int matches;
    if (reg_read != nvme_read)
        matches = -1;
    else
        matches = memcmp(buf1, buf2, nvme_read);

    free(buf1);
    free(buf2);

    return matches;
}

static int test_write(const char *fname)
{
    const size_t count = 1 << 20;
    unsigned int *data = malloc(count * sizeof(*data));

    for (int i = 0; i < count; i++)
        data[i] = 0xAABB0000 | i;

    int wrote = nvme_dev_write_file(fname, data, count * sizeof(*data));

    if (wrote < 0)
        return -1;

    int fd = open(fname, O_RDONLY);
    void *data2 = malloc(wrote);
    read(fd, data2, wrote);
    close(fd);

    return memcmp(data, data2, wrote);
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


    int ret = 0;
    if ((ret = test_read(fname)))
        goto leave;
    if ((ret = test_write("/mnt/princeton/random_test2.write_test")))
        goto leave;



leave:
    return ret;
}
