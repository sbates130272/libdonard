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
//     Fifo Test
//
////////////////////////////////////////////////////////////////////////


#include <libdonard/dirwalk.h>

#include <unistd.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    struct fifo *fifo = fifo_new(8);
    if (fifo == NULL) {
        perror("Could not create fifo");
        return -1;
    }

    const char *filters[] = {
        "*.jpg",
        "*.jpeg",
        NULL
    };

    if (dirwalk(&argv[1], argc-1, filters, fifo, 0))
        perror("Could not start dirwalk thread");


    char *fpath;
    while ((fpath = fifo_pop(fifo)) != NULL) {
        printf("%s\n", fpath);
        free(fpath);
    }


    fifo_free(fifo);

    usleep(50000);

    return 0;
}
