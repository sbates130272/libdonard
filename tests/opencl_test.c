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
//     Quick OpenCL Test
//
////////////////////////////////////////////////////////////////////////

#include <CL/opencl.h>

#include <stdio.h>
#include <errno.h>
#include <string.h>

#define check_error(x, text) ({\
    cl_int ret = x; \
    if (ret != CL_SUCCESS) { \
        if (text[0] != 0) \
           fprintf(stderr, "Error %s: %d\n", text, ret); \
        return -1; \
    } \
})

#define array_size(x) (sizeof(x) / sizeof(*x))

static cl_int list_devices(cl_platform_id platform)
{
    cl_device_id devices[10];
    cl_uint num_devices;

    check_error(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                               array_size(devices), devices,
                               &num_devices),
                "getting device list");


    for (int i = 0; i < num_devices; i++) {
        char vendor[80];
        char dev_name[80];
        cl_device_type type;

        check_error(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(vendor),
                                    vendor, NULL), "getting device vendor name");

        check_error(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(dev_name),
                                    dev_name, NULL), "getting device name");

        check_error(clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type),
                                    &type, NULL), "getting device type");

        const char *type_str = "UNKNOWN";
        switch (type) {
        case CL_DEVICE_TYPE_CPU:         type_str = "CPU";     break;
        case CL_DEVICE_TYPE_GPU:         type_str = "GPU";     break;
        case CL_DEVICE_TYPE_ACCELERATOR: type_str = "ACCEL";   break;
        case CL_DEVICE_TYPE_DEFAULT:     type_str = "DEFAULT"; break;
        }

        printf("      %d - %s - %s (%s)\n", i+1, vendor, dev_name, type_str);
    }

    return CL_SUCCESS;
}

static cl_int list_platforms(void)
{
    cl_platform_id platforms[10];
    cl_uint num_platforms;

    check_error(clGetPlatformIDs(array_size(platforms), platforms,
                                 &num_platforms),
                "getting platform ids");


    printf("Platforms:\n");
    for (int i = 0; i < num_platforms; i++) {
        char vendor[80];
        char name[80];
        char version[80];

        check_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor),
                                    vendor, NULL), "getting platform vendor name");

        check_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name),
                                      name, NULL), "getting platform name");

        check_error(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version),
                                      version, NULL), "getting platform version");

        printf("  %d - %s %s %s\n", i+1, vendor, name, version);
        check_error(list_devices(platforms[i]), "");
    }

    return CL_SUCCESS;
}

static cl_int find_first_gpu(cl_device_id *dev)
{
    if (dev == NULL)
        return -1;

    cl_platform_id platforms[10];
    cl_uint num_platforms;

    check_error(clGetPlatformIDs(array_size(platforms), platforms,
                                 &num_platforms),
                "getting platform ids");

    for (int i = 0; i < num_platforms; i++) {

        cl_device_id devices[10];
        cl_uint num_devices;
        cl_int ret = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU,
                                    array_size(devices), devices,
                                    &num_devices);
        if (ret != CL_SUCCESS)
            continue;

        if (num_devices >= 1) {
            *dev = devices[0];
            return 0;
        }
    }

    return -1;
}

static void pfn_notify(const char *errinfo, const void *private_info,
                       size_t cb, void *user_data)
{
    fprintf(stderr, "PFN Notify Error: %s\n", errinfo);
}



cl_program load_program(cl_context ctx, cl_device_id gpu, cl_int *errcode_ret)
{
    extern char _binary____tests_opencl_test_kernel_cl_start;
    extern char _binary____tests_opencl_test_kernel_cl_size;

    const char *strings[] = {&_binary____tests_opencl_test_kernel_cl_start};
    size_t lengths[] = {(size_t) &_binary____tests_opencl_test_kernel_cl_size};

    cl_program pgm =  clCreateProgramWithSource(ctx, array_size(strings),
                                                strings, lengths,
                                                errcode_ret);
    if (*errcode_ret != CL_SUCCESS)
        return NULL;

    *errcode_ret = clBuildProgram(pgm, 1, &gpu, "-Werror", NULL, NULL);

    if (*errcode_ret != CL_SUCCESS) {
        size_t logsize;
        *errcode_ret = clGetProgramBuildInfo(pgm, gpu, CL_PROGRAM_BUILD_LOG, 0,
                                             NULL, &logsize);
        if (*errcode_ret != CL_SUCCESS)
            return NULL;

        char buildlog[logsize+1];

        *errcode_ret = clGetProgramBuildInfo(pgm, gpu, CL_PROGRAM_BUILD_LOG,
                                             sizeof(buildlog),
                                             buildlog, &logsize);
        if (*errcode_ret != CL_SUCCESS)
            return NULL;

        fprintf(stderr, "\n\n--Build Log:\n%s\n", buildlog);

    }

    return pgm;
}

static cl_int square_array(cl_context ctx, cl_command_queue queue, cl_kernel kern,
                           float *a_h, int N)
{
    cl_int errcode;

    cl_mem a_d = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(*a_h) * N, a_h, &errcode);
    if (errcode != CL_SUCCESS)
        return errcode;

    errcode = clSetKernelArg(kern, 0, sizeof(cl_mem), (void *) &a_d);
    if (errcode != CL_SUCCESS)
        return errcode;

    size_t global_work_size = N;
    cl_event event;
    errcode = clEnqueueNDRangeKernel(queue, kern, 1, NULL, &global_work_size,
                                     NULL, 0, NULL, &event);
    if (errcode != CL_SUCCESS)
        return errcode;

    errcode = clEnqueueReadBuffer(queue, a_d, CL_TRUE, 0, sizeof(*a_h) * N,
                                  (void *) a_h, 1, &event, NULL);

    clReleaseEvent(event);

    return errcode;
}

static int run_square_array(cl_context ctx, cl_command_queue queue,
                        cl_kernel square_kernel)
{
    int ret = 0;

    const int N = 10;
    float a_h[N];
    for (int i = 0; i < N; i++)
        a_h[i] = i;

    ret = square_array(ctx, queue, square_kernel, a_h, N);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Error: Square Array Failed, %d\n", ret);
        return -1;
    }

    for (int i=0; i<N; i++) {
        printf("%d %f\n", i, a_h[i]);
        if (abs(a_h[i] - i*i) > 0.001)
            ret = -2;
    }

    return ret;
}

int main(int argc, char **argv)
{
    int ret;

    check_error(list_platforms(), "");
    printf("\n\n");

    cl_device_id gpu;
    check_error(find_first_gpu(&gpu), "could not find GPU device!");

    cl_int errcode_ret;
    cl_context ctx = clCreateContext(NULL, 1, &gpu, pfn_notify, NULL,
                                     &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        fprintf(stderr, "Error: could not create context: %d\n", errcode_ret);
        return -1;
    }

    cl_command_queue queue = clCreateCommandQueue(ctx, gpu, 0, &errcode_ret);
    check_error(errcode_ret, "could not create command queue");
    if (errcode_ret != CL_SUCCESS) {
        fprintf(stderr, "Error: could not create context: %d\n", errcode_ret);
        ret = -1;
        goto release_context;
    }

    cl_program pgm = load_program(ctx, gpu, &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        fprintf(stderr, "Error: could not create program: %d\n", errcode_ret);
        ret = -1;
        goto release_queue;
    }

    cl_kernel square_kernel = clCreateKernel(pgm, "square_array", &errcode_ret);
    if (errcode_ret != CL_SUCCESS) {
        fprintf(stderr, "Error: could not create program: %d\n", errcode_ret);
        ret = -1;
        goto release_program;
    }

    ret = run_square_array(ctx, queue, square_kernel);

    clReleaseKernel(square_kernel);

release_program:
    clReleaseProgram(pgm);

release_queue:
    clReleaseCommandQueue(queue);

release_context:
    clReleaseContext(ctx);

    return ret;
}
