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
//     Miscelaneo Utility Functions
//
////////////////////////////////////////////////////////////////////////


#include "utils.h"
#include "version.h"

double utils_timeval_to_secs(struct timeval *t)
{
    return  t->tv_sec + t->tv_usec / 1e6;
}

int utils_cmp(char *fname1, char *fname2)
{

    FILE *f1 = fopen(fname1, "r");
    if (!f1) {  perror(fname1); exit(EXIT_FAILURE);  };
    FILE *f2 = fopen(fname2, "r");
    if (!f2) {  perror(fname2); exit(EXIT_FAILURE);  };
    int ret=0, c1, c2;

    while (ret==0){
        c1 = getc(f1); c2 = getc(f2);
        if (c1 != c2) ret=1;
        if (c1==EOF || c2==EOF)
            break;
    }

    fclose (f1);
    fclose (f2);

    return ret;
}

const char *utils_libdonard_version(void)
{
    return VERSION;
}
