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
//     Test utilities
//
////////////////////////////////////////////////////////////////////////

#include "test_util.h"
#include <stdio.h>

static int file_exists(const char * filename)
{
    FILE *f;
    if ((f = fopen(filename, "r"))) {
        fclose(f);
        return 1;
    }

    return 0;
}

const char *test_util_find_img(const char *fname, char *buf)
{
    sprintf(buf, "data/%s", fname);
    if (file_exists(buf)) return buf;

    sprintf(buf, "../data/%s", fname);
    if (file_exists(buf)) return buf;

    sprintf(buf, "../../data/%s", fname);
    if (file_exists(buf)) return buf;

    sprintf(buf, "../../../data/%s", fname);
    if (file_exists(buf)) return buf;

    return fname;
}
