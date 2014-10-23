
__kernel void square_array(__global float *a)
{
    unsigned int idx = get_global_id(0);
    a[idx] = a[idx] * a[idx];
}
