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


#include <libdonard/fifo.h>

#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>

static int random_fd;
static struct fifo *fifo;

struct producer {
    int start, end;
};

void *producer(void *arg)
{
    struct producer *p = arg;
    printf("Producer started: %d to %d\n", p->start, p->end);

    struct timespec t = {0};

    for (int i = p->start; i < p->end; i++){
        read(random_fd, &t.tv_nsec, sizeof(t.tv_nsec));
        t.tv_nsec &= 0xFFFFF;
        nanosleep(&t, NULL);

        int *ii = malloc(sizeof(*ii));
        *ii = i;
        fifo_push(fifo, ii);
    }

    printf("Producer finished: %d to %d\n", p->start, p->end);
    return NULL;
}

void *consumer(void *arg)
{
    int id = (intptr_t) arg;

    printf("Consumer Started: %d\n", id);

    int sum = 0;
    struct timespec t = {0};

    while (1) {
        int *ii = fifo_pop(fifo);
        if (ii == NULL)
            break;

        sum += *ii;
        free(ii);

        read(random_fd, &t.tv_nsec, sizeof(t.tv_nsec));
        t.tv_nsec &= 0xFFFFFF;
        nanosleep(&t, NULL);
    }

    printf("Consumer finished: %d\n", id);
    return (void *)(intptr_t) sum;
}

int calc_sum(struct producer *p, int count)
{
    int ret = 0;

    for (int i = 0; i < count; i++)
        for (int j = p[i].start; j < p[i].end; j++)
            ret += j;

    return ret;
}


int main(int argc, char *argv[])
{
    random_fd = open("/dev/urandom", O_RDONLY);
    if (random_fd < 0) {
        perror("Could not open /dev/urandom");
        return random_fd;
    }

    fifo = fifo_new(16);
    if (!fifo) {
        perror("Could not create fifo");
        close(random_fd);
        return -1;
    }

    struct producer ps[] = {
        {.start=5, .end=60},
        {.start=50, .end=125},
        {.start=1000, .end=1250},
        {.start=1500, .end=1545},
        {.start=2000, .end=2200},
        {.start=3000, .end=4000},
    };

    pthread_t p_thrds[sizeof(ps) / sizeof(*ps)];

    for (int i = 0; i < sizeof(ps) / sizeof(*ps); i++)
        pthread_create(&p_thrds[i], NULL, producer, &ps[i]);

    pthread_t c_thrds[20];

    for (int i = 0; i < sizeof(c_thrds) / sizeof(*c_thrds); i++)
        pthread_create(&c_thrds[i], NULL, consumer, (void *)(intptr_t) i);

    int actual_sum = calc_sum(ps, sizeof(ps) / sizeof(*ps));

    for (int i = 0; i < sizeof(ps) / sizeof(*ps); i++)
        pthread_join(p_thrds[i], NULL);

    fifo_close(fifo);

    int thread_sum = 0;

    for (int i = 0; i < sizeof(c_thrds) / sizeof(*c_thrds); i++) {
        void *retval;
        pthread_join(c_thrds[i], &retval);
        thread_sum += (intptr_t)retval;
    }

    printf("Thread Sum: %d\n", thread_sum);
    printf("Actual Sum: %d\n", actual_sum);

    fifo_free(fifo);
    close(random_fd);

    return 0;
}
