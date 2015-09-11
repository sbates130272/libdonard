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


#ifdef HAVE_LINUX_PERF_EVENT_H

#include "perfstats.h"

#include <linux/perf_event.h>
#include <linux/unistd.h>

#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>

#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>

enum {
    COUNTER_PAGE_FAULTS,
};

static struct perf_event_attr attrs[] = {
    { .type = PERF_TYPE_SOFTWARE, .config = PERF_COUNT_SW_PAGE_FAULTS},
};

#define STAT_COUNT (sizeof(attrs) / sizeof(*attrs))

static int fds[STAT_COUNT];

static inline int
sys_perf_event_open(struct perf_event_attr *attr,
                    pid_t pid, int cpu, int group_fd,
                    unsigned long flags)
{
    attr->size = sizeof(*attr);
    return syscall(__NR_perf_event_open, attr, pid, cpu,
                   group_fd, flags);
}

void perfstats_init(void)
{
    int pid = getpid();

    for (int i = 0; i < STAT_COUNT; i++) {
        attrs[i].inherit = 1;
        attrs[i].disabled = 1;
        attrs[i].enable_on_exec = 0;
        fds[i] = sys_perf_event_open(&attrs[i], pid, -1, -1, 0);
    }
}

void perfstats_deinit(void)
{
    for (int i = 0; i < STAT_COUNT; i++) {
        close(fds[i]);
        fds[i] = -1;
    }
}


void perfstats_enable(void)
{
    for (int i = 0; i < STAT_COUNT; i++) {
        if (fds[i] <= 0)
            continue;

        ioctl(fds[i], PERF_EVENT_IOC_ENABLE);
    }
}

void perfstats_disable(void)
{
    for (int i = 0; i < STAT_COUNT; i++) {
        if (fds[i] <= 0)
            continue;

        ioctl(fds[i], PERF_EVENT_IOC_DISABLE);
    }
}

static uint64_t readcounter(int i)
{
    uint64_t ret;
    if (read(fds[i], &ret, sizeof(ret)) != sizeof(ret))
        return -1;

    return ret;
}

void perfstats_print(void)
{
    if (fds[COUNTER_PAGE_FAULTS] > 0)
        printf("Page Faults: %" PRIu64 "\n",
               readcounter(COUNTER_PAGE_FAULTS));
}










#endif
