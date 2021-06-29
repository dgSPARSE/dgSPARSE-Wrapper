#ifndef SPMM_H
#define SPMM_H

extern "C"
{
    void spmm_cuda(
        int m, int k,
        int *rowptr,
        int *colind,
        float *values,
        float *denseB,
        float *denseC);

    void spmm_cuda_no_edge_value(
        int m, int k,
        int *rowptr,
        int *colind,
        float *values,
        float *denseB,
        float *denseC);
}
#endif //SPMM_H