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
//     Hold allocated pin poll memory for experimentation
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

    while(1);

    pinpool_deinit();

    return ret;
}
