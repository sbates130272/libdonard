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
//     Mmap Test Code
//
////////////////////////////////////////////////////////////////////////

#include <libdonard/pinpool.h>

#include <cuda_runtime.h>

#include <string.h>
#include <stdio.h>
#include <stdint.h>

int main(int argc, char *argv[])
{
    int ret = 0;

    if (pinpool_init(1, 1*1024*1024)) {
        perror("Could not initialize pin pool");
        return -1;
    }

    struct pin_buf *p = pinpool_alloc();
    if (p == NULL) {
        perror("Could not allocate pinned buffer");
        ret = -2;
        goto leave;
    }

    uint32_t testdata[128*1204];
    for (int i = 0; i < sizeof(testdata)/sizeof(*testdata); i++)
        testdata[i] = 2*i;

    if (cudaMemcpy(p->address, testdata, sizeof(testdata),
                   cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "Cuda Memcpy Failed!\n");
        ret = -3;
        goto leave;
    }

    uint32_t *buf = pinpool_mmap(p);
    if (buf == NULL) {
        perror("Could not mmap buffer");
        ret = -4;
        goto leave;
    }

    for (int i = 0; i < sizeof(testdata)/sizeof(*testdata); i++) {
        if (buf[i] != 2*i) {
            printf("Mismatch at %d during read!\n", i);
            ret = -5;
            goto unmap;
        }

        buf[i] = 3*i + 1;
    }

    if (cudaMemcpy(testdata, p->address, sizeof(testdata),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        fprintf(stderr, "Cuda Memcpy Failed!\n");
        ret = -6;
        goto leave;
    }

    for (int i = 0; i < sizeof(testdata)/sizeof(*testdata); i++) {
        if (testdata[i] != 3*i + 1) {
            printf("Mismatch at %d during write!\n", i);
            ret = -7;
            goto unmap;
        }
    }


unmap:
    pinpool_munmap(p);


leave:
    pinpool_deinit();

    return ret;
}
