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
//     Capture Perf Stats
//
////////////////////////////////////////////////////////////////////////


#ifndef __LIBDONARD_PERFSTATS_H__
#define __LIBDONARD_PERFSTATS_H__

#ifndef HAVE_LINUX_PERF_EVENT_H

static inline void perfstats_init(void) {}
static inline void perfstats_deinit(void) {}
static inline void perfstats_enable(void) {}
static inline void perfstats_disable(void) {}
static inline void perfstats_print(void) {}

#else

void perfstats_init(void);
void perfstats_deinit(void);
void perfstats_enable(void);
void perfstats_disable(void);
void perfstats_print(void);


#endif


#endif
