/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
extern "C"
{
#include <libdonard/filemap.h>
}
#include <errno.h>
#include <cuda_runtime.h>

void Bench(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	Bench( argc, argv);
}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file> <num_of_elements>\n", argv[0]);

}

void Bench( int argc, char** argv) 
{

    struct timeval time_start, time_end, total_start, total_end;
    unsigned long filesize;
    char *input_f;
    FILE *fp;
    int input_time,input_time_wo_init = 0;
    float *h_elements;
    int i;
	if(argc < 2){
	Usage(argc, argv);
	exit(0);
	}
	
	input_f = argv[1];
//	num_of_nodes = atoi(argv[2]);

    	gettimeofday(&time_start, NULL);
    	if(pinpool_init(4, 32*1024*1024))
    	{
            fprintf(stderr,"Could not initialize pin pool");
            exit(1);
        }

    	gettimeofday(&time_end, NULL);
	input_time_wo_init = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	printf("pinpool_init %d\n",input_time_wo_init);

	float* d_elements = NULL;
    int fd = open(input_f, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error opening file '%s': %s\n", argv[1], strerror(errno));
        return;
    }

	struct filemap *fm = filemap_alloc_cuda_nvme_mt(fd, input_f);
	
    	gettimeofday(&time_end, NULL);
	input_time = ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	printf("HGProfile: File to GPU %d Bandwidth %lf (%lf wo init) MB/Sec\n",input_time, (double)fm->length/(input_time*1.024), (double)fm->length/((input_time-input_time_wo_init)*1.024));
    	free(fm);
    	pinpool_deinit();
}
