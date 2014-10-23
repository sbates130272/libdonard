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
//     Directory Walking Thread
//
////////////////////////////////////////////////////////////////////////


#ifndef __LIBDONARD_DIRWALK_H__
#define __LIBDONARD_DIRWALK_H__

#include "fifo.h"

enum {
    DIRWALK_ONE_FILE_SYSTEM = 1,
};

#ifdef __cplusplus
extern "C" {
#endif

int dirwalk(char * const argv[], int argc,
            const char * const filters[], struct fifo *output,
            int options);

#ifdef __cplusplus
}
#endif

#endif
