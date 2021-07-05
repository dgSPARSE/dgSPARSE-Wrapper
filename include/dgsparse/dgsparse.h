#ifndef SPMM_H
#define SPMM_H

extern "C"
{
    void spmm_cuda(
        int m, 
        int k,
        int *rowptr,
        int *colind,
        float *values,
        float *dense,
        float *out);

    void spmm_cuda_no_edge_value(
        int m, 
        int k,
        int *rowptr,
        int *colind,
        float *values,
        float *dense,
        float *out);
    void sddmm_cuda_coo(
        int k,
        int nnz,
        int *rowind,
        int *colind,
        float *D1,
        float *D2,
        float *out);

    void sddmm_cuda_csr(
        int m,
        int k,
        int nnz,
        int *rowptr,
        int *colind,
        float *D1,
        float *D2,
        float *out);

    void edge_softmax_cuda(
        int mrows,
        int head,
        int *rowptr,
        float *values,
        float *softmax);

}
#endif //SPMM_H