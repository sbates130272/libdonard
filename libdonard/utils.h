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


#ifndef __LIBDONARD_UTILS_H__
#define __LIBDONARD_UTILS_H__

#include <stdlib.h>
#include <stdio.h>
#include <error.h>
#include <sys/time.h>



#ifdef __cplusplus
extern "C" {
#endif

double utils_timeval_to_secs(struct timeval *t);
int utils_cmp(char *fname1, char *fname2);
const char *utils_libdonard_version(void);

#ifdef __cplusplus
}
#endif


#endif
