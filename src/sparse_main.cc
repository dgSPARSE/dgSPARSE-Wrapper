#include <string.h>

#include "cusparse_v2.h"
#include "symbol_helper_cusparse.h"
#include "dgsparse.h"

#pragma GCC diagnostic ignored "-Wpointer-arith"
#define DGSPARSE
using namespace sparse_wrapper;

// cusparseStatus_t CUSPARSEAPI
// cusparseCreate(cusparseHandle_t* handle) {
// #if DG_SPARSE_ACC
//     LOAD_SPARSE_SYMBOL_FOR_ONCE(DGSPARSE_LIB, cusparseCreate);
//     LOG(TRACE, "Enter %s()", __FUNCTION__);
//     return _real_sym(handle);
// #else
//     LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreate);
//     LOG(TRACE, "Enter %s()", __FUNCTION__);
//     return _real_sym(handle);
// #endif
// }

// cusparseStatus_t CUSPARSEAPI
// cusparseDestroy(cusparseHandle_t handle) {
//     LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroy);
//     LOG(TRACE, "Enter %s()", __FUNCTION__);
//     return _real_sym(handle);
// }

cusparseStatus_t CUSPARSEAPI
cusparseCreate(cusparseHandle_t *handle)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreate);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroy(cusparseHandle_t handle)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroy);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle);
}

cusparseStatus_t CUSPARSEAPI
cusparseGetVersion(cusparseHandle_t handle,
                   int *version)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetVersion);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     version);
}

cusparseStatus_t CUSPARSEAPI
cusparseGetProperty(libraryPropertyType type,
                    int *value)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetProperty);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(type, value);
}

const char *CUSPARSEAPI
cusparseGetErrorName(cusparseStatus_t status)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetErrorName);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(status);
}

const char *CUSPARSEAPI
cusparseGetErrorString(cusparseStatus_t status)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetErrorString);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(status);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetStream(cusparseHandle_t handle,
                  cudaStream_t streamId)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSetStream);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     streamId);
}

cusparseStatus_t CUSPARSEAPI
cusparseGetStream(cusparseHandle_t handle,
                  cudaStream_t *streamId)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetStream);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     streamId);
}

cusparseStatus_t CUSPARSEAPI
cusparseGetPointerMode(cusparseHandle_t handle,
                       cusparsePointerMode_t *mode)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetPointerMode);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mode);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetPointerMode(cusparseHandle_t handle,
                       cusparsePointerMode_t mode)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSetPointerMode);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mode);
}

//##############################################################################
//# HELPER ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateMatDescr(cusparseMatDescr_t *descrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateMatDescr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyMatDescr(cusparseMatDescr_t descrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyMatDescr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseCopyMatDescr(cusparseMatDescr_t dest,
                     const cusparseMatDescr_t src)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCopyMatDescr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dest,
                     src);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetMatType(cusparseMatDescr_t descrA,
                   cusparseMatrixType_t type)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSetMatType);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA,
                     type);
}

cusparseMatrixType_t CUSPARSEAPI
cusparseGetMatType(const cusparseMatDescr_t descrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetMatType);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetMatFillMode(cusparseMatDescr_t descrA,
                       cusparseFillMode_t fillMode)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSetMatFillMode);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA,
                     fillMode);
}

cusparseFillMode_t CUSPARSEAPI
cusparseGetMatFillMode(const cusparseMatDescr_t descrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetMatFillMode);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetMatDiagType(cusparseMatDescr_t descrA,
                       cusparseDiagType_t diagType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSetMatDiagType);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA,
                     diagType);
}

cusparseDiagType_t CUSPARSEAPI
cusparseGetMatDiagType(const cusparseMatDescr_t descrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetMatDiagType);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetMatIndexBase(cusparseMatDescr_t descrA,
                        cusparseIndexBase_t base)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSetMatIndexBase);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA,
                     base);
}

cusparseIndexBase_t CUSPARSEAPI
cusparseGetMatIndexBase(const cusparseMatDescr_t descrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetMatIndexBase);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrsv2Info(csrsv2Info_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateCsrsv2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrsv2Info(csrsv2Info_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyCsrsv2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsric02Info(csric02Info_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateCsric02Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsric02Info(csric02Info_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyCsric02Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateBsric02Info(bsric02Info_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateBsric02Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsric02Info(bsric02Info_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyBsric02Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrilu02Info(csrilu02Info_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateCsrilu02Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrilu02Info(csrilu02Info_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyCsrilu02Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateBsrilu02Info(bsrilu02Info_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateBsrilu02Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsrilu02Info(bsrilu02Info_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyBsrilu02Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateBsrsv2Info(bsrsv2Info_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateBsrsv2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsrsv2Info(bsrsv2Info_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyBsrsv2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateBsrsm2Info(bsrsm2Info_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateBsrsm2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyBsrsm2Info(bsrsm2Info_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyBsrsm2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsru2csrInfo(csru2csrInfo_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateCsru2csrInfo);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsru2csrInfo(csru2csrInfo_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyCsru2csrInfo);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateColorInfo(cusparseColorInfo_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateColorInfo);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyColorInfo(cusparseColorInfo_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyColorInfo);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseSetColorAlgs(cusparseColorInfo_t info,
                     cusparseColorAlg_t alg)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSetColorAlgs);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info, alg);
}

cusparseStatus_t CUSPARSEAPI
cusparseGetColorAlgs(cusparseColorInfo_t info,
                     cusparseColorAlg_t *alg)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGetColorAlgs);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info, alg);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreatePruneInfo(pruneInfo_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreatePruneInfo);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyPruneInfo(pruneInfo_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyPruneInfo);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

//##############################################################################
//# SPARSE LEVEL 1 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSaxpyi(cusparseHandle_t handle,
               int nnz,
               const float *alpha,
               const float *xVal,
               const int *xInd,
               float *y,
               cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSaxpyi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     alpha,
                     xVal,
                     xInd,
                     y,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseDaxpyi(cusparseHandle_t handle,
               int nnz,
               const double *alpha,
               const double *xVal,
               const int *xInd,
               double *y,
               cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDaxpyi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     alpha,
                     xVal,
                     xInd,
                     y,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseCaxpyi(cusparseHandle_t handle,
               int nnz,
               const cuComplex *alpha,
               const cuComplex *xVal,
               const int *xInd,
               cuComplex *y,
               cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCaxpyi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     alpha,
                     xVal,
                     xInd,
                     y,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseZaxpyi(cusparseHandle_t handle,
               int nnz,
               const cuDoubleComplex *alpha,
               const cuDoubleComplex *xVal,
               const int *xInd,
               cuDoubleComplex *y,
               cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZaxpyi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     alpha,
                     xVal,
                     xInd,
                     y,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgthr(cusparseHandle_t handle,
              int nnz,
              const float *y,
              float *xVal,
              const int *xInd,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgthr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     y,
                     xVal,
                     xInd,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgthr(cusparseHandle_t handle,
              int nnz,
              const double *y,
              double *xVal,
              const int *xInd,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgthr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     y,
                     xVal,
                     xInd,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgthr(cusparseHandle_t handle,
              int nnz,
              const cuComplex *y,
              cuComplex *xVal,
              const int *xInd,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgthr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     y,
                     xVal,
                     xInd,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgthr(cusparseHandle_t handle,
              int nnz,
              const cuDoubleComplex *y,
              cuDoubleComplex *xVal,
              const int *xInd,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgthr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     y,
                     xVal,
                     xInd,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgthrz(cusparseHandle_t handle,
               int nnz,
               float *y,
               float *xVal,
               const int *xInd,
               cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgthrz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     y,
                     xVal,
                     xInd,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgthrz(cusparseHandle_t handle,
               int nnz,
               double *y,
               double *xVal,
               const int *xInd,
               cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgthrz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     y,
                     xVal,
                     xInd,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgthrz(cusparseHandle_t handle,
               int nnz,
               cuComplex *y,
               cuComplex *xVal,
               const int *xInd,
               cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgthrz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     y,
                     xVal,
                     xInd,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgthrz(cusparseHandle_t handle,
               int nnz,
               cuDoubleComplex *y,
               cuDoubleComplex *xVal,
               const int *xInd,
               cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgthrz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     y,
                     xVal,
                     xInd,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseSsctr(cusparseHandle_t handle,
              int nnz,
              const float *xVal,
              const int *xInd,
              float *y,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSsctr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     xVal,
                     xInd,
                     y,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseDsctr(cusparseHandle_t handle,
              int nnz,
              const double *xVal,
              const int *xInd,
              double *y,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDsctr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     xVal,
                     xInd,
                     y,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseCsctr(cusparseHandle_t handle,
              int nnz,
              const cuComplex *xVal,
              const int *xInd,
              cuComplex *y,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCsctr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     xVal,
                     xInd,
                     y,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseZsctr(cusparseHandle_t handle,
              int nnz,
              const cuDoubleComplex *xVal,
              const int *xInd,
              cuDoubleComplex *y,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZsctr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     xVal,
                     xInd,
                     y,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseSroti(cusparseHandle_t handle,
              int nnz,
              float *xVal,
              const int *xInd,
              float *y,
              const float *c,
              const float *s,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSroti);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     xVal,
                     xInd,
                     y,
                     c,
                     s,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseDroti(cusparseHandle_t handle,
              int nnz,
              double *xVal,
              const int *xInd,
              double *y,
              const double *c,
              const double *s,
              cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDroti);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     nnz,
                     xVal,
                     xInd,
                     y,
                     c,
                     s,
                     idxBase);
}

//##############################################################################
//# SPARSE LEVEL 2 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSgemvi(cusparseHandle_t handle,
               cusparseOperation_t transA,
               int m,
               int n,
               const float *alpha,
               const float *A,
               int lda,
               int nnz,
               const float *xVal,
               const int *xInd,
               const float *beta,
               float *y,
               cusparseIndexBase_t idxBase,
               void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgemvi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     n,
                     alpha,
                     A,
                     lda,
                     nnz,
                     xVal,
                     xInd,
                     beta,
                     y,
                     idxBase,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgemvi_bufferSize(cusparseHandle_t handle,
                          cusparseOperation_t transA,
                          int m,
                          int n,
                          int nnz,
                          int *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgemvi_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     n,
                     nnz,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgemvi(cusparseHandle_t handle,
               cusparseOperation_t transA,
               int m,
               int n,
               const double *alpha,
               const double *A,
               int lda,
               int nnz,
               const double *xVal,
               const int *xInd,
               const double *beta,
               double *y,
               cusparseIndexBase_t idxBase,
               void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgemvi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     n,
                     alpha,
                     A,
                     lda,
                     nnz,
                     xVal,
                     xInd,
                     beta,
                     y,
                     idxBase,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgemvi_bufferSize(cusparseHandle_t handle,
                          cusparseOperation_t transA,
                          int m,
                          int n,
                          int nnz,
                          int *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgemvi_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     n,
                     nnz,
                     pBufferSize);
}
cusparseStatus_t CUSPARSEAPI
cusparseCgemvi(cusparseHandle_t handle,
               cusparseOperation_t transA,
               int m,
               int n,
               const cuComplex *alpha,
               const cuComplex *A,
               int lda,
               int nnz,
               const cuComplex *xVal,
               const int *xInd,
               const cuComplex *beta,
               cuComplex *y,
               cusparseIndexBase_t idxBase,
               void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgemvi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     n,
                     alpha,
                     A,
                     lda,
                     nnz,
                     xVal,
                     xInd,
                     beta,
                     y,
                     idxBase,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgemvi_bufferSize(cusparseHandle_t handle,
                          cusparseOperation_t transA,
                          int m,
                          int n,
                          int nnz,
                          int *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgemvi_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     n,
                     nnz,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgemvi(cusparseHandle_t handle,
               cusparseOperation_t transA,
               int m,
               int n,
               const cuDoubleComplex *alpha,
               const cuDoubleComplex *A,
               int lda,
               int nnz,
               const cuDoubleComplex *xVal,
               const int *xInd,
               const cuDoubleComplex *beta,
               cuDoubleComplex *y,
               cusparseIndexBase_t idxBase,
               void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgemvi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     n,
                     alpha,
                     A,
                     lda,
                     nnz,
                     xVal,
                     xInd,
                     beta,
                     y,
                     idxBase,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgemvi_bufferSize(cusparseHandle_t handle,
                          cusparseOperation_t transA,
                          int m,
                          int n,
                          int nnz,
                          int *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgemvi_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     n,
                     nnz,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCsrmvEx_bufferSize(cusparseHandle_t handle,
                           cusparseAlgMode_t alg,
                           cusparseOperation_t transA,
                           int m,
                           int n,
                           int nnz,
                           const void *alpha,
                           cudaDataType alphatype,
                           const cusparseMatDescr_t descrA,
                           const void *csrValA,
                           cudaDataType csrValAtype,
                           const int *csrRowPtrA,
                           const int *csrColIndA,
                           const void *x,
                           cudaDataType xtype,
                           const void *beta,
                           cudaDataType betatype,
                           void *y,
                           cudaDataType ytype,
                           cudaDataType executiontype,
                           size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCsrmvEx_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     alg,
                     transA,
                     m,
                     n,
                     nnz,
                     alpha,
                     alphatype,
                     descrA,
                     csrValA,
                     csrValAtype,
                     csrRowPtrA,
                     csrColIndA,
                     x,
                     xtype,
                     beta,
                     betatype,
                     y,
                     ytype,
                     executiontype,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCsrmvEx(cusparseHandle_t handle,
                cusparseAlgMode_t alg,
                cusparseOperation_t transA,
                int m,
                int n,
                int nnz,
                const void *alpha,
                cudaDataType alphatype,
                const cusparseMatDescr_t descrA,
                const void *csrValA,
                cudaDataType csrValAtype,
                const int *csrRowPtrA,
                const int *csrColIndA,
                const void *x,
                cudaDataType xtype,
                const void *beta,
                cudaDataType betatype,
                void *y,
                cudaDataType ytype,
                cudaDataType executiontype,
                void *buffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCsrmvEx);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     alg,
                     transA,
                     m,
                     n,
                     nnz,
                     alpha,
                     alphatype,
                     descrA,
                     csrValA,
                     csrValAtype,
                     csrRowPtrA,
                     csrColIndA,
                     x,
                     xtype,
                     beta,
                     betatype,
                     y,
                     ytype,
                     executiontype,
                     buffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrmv(cusparseHandle_t handle,
               cusparseDirection_t dirA,
               cusparseOperation_t transA,
               int mb,
               int nb,
               int nnzb,
               const float *alpha,
               const cusparseMatDescr_t descrA,
               const float *bsrSortedValA,
               const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA,
               int blockDim,
               const float *x,
               const float *beta,
               float *y)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrmv);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     x,
                     beta,
                     y);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrmv(cusparseHandle_t handle,
               cusparseDirection_t dirA,
               cusparseOperation_t transA,
               int mb,
               int nb,
               int nnzb,
               const double *alpha,
               const cusparseMatDescr_t descrA,
               const double *bsrSortedValA,
               const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA,
               int blockDim,
               const double *x,
               const double *beta,
               double *y)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrmv);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     x,
                     beta,
                     y);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrmv(cusparseHandle_t handle,
               cusparseDirection_t dirA,
               cusparseOperation_t transA,
               int mb,
               int nb,
               int nnzb,
               const cuComplex *alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex *bsrSortedValA,
               const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA,
               int blockDim,
               const cuComplex *x,
               const cuComplex *beta,
               cuComplex *y)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrmv);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     x,
                     beta,
                     y);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrmv(cusparseHandle_t handle,
               cusparseDirection_t dirA,
               cusparseOperation_t transA,
               int mb,
               int nb,
               int nnzb,
               const cuDoubleComplex *alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex *bsrSortedValA,
               const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA,
               int blockDim,
               const cuDoubleComplex *x,
               const cuDoubleComplex *beta,
               cuDoubleComplex *y)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrmv);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     x,
                     beta,
                     y);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrxmv(cusparseHandle_t handle,
                cusparseDirection_t dirA,
                cusparseOperation_t transA,
                int sizeOfMask,
                int mb,
                int nb,
                int nnzb,
                const float *alpha,
                const cusparseMatDescr_t descrA,
                const float *bsrSortedValA,
                const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA,
                const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA,
                int blockDim,
                const float *x,
                const float *beta,
                float *y)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrxmv);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     sizeOfMask,
                     mb,
                     nb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedMaskPtrA,
                     bsrSortedRowPtrA,
                     bsrSortedEndPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     x,
                     beta,
                     y);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrxmv(cusparseHandle_t handle,
                cusparseDirection_t dirA,
                cusparseOperation_t transA,
                int sizeOfMask,
                int mb,
                int nb,
                int nnzb,
                const double *alpha,
                const cusparseMatDescr_t descrA,
                const double *bsrSortedValA,
                const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA,
                const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA,
                int blockDim,
                const double *x,
                const double *beta,
                double *y)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrxmv);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     sizeOfMask,
                     mb,
                     nb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedMaskPtrA,
                     bsrSortedRowPtrA,
                     bsrSortedEndPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     x,
                     beta,
                     y);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrxmv(cusparseHandle_t handle,
                cusparseDirection_t dirA,
                cusparseOperation_t transA,
                int sizeOfMask,
                int mb,
                int nb,
                int nnzb,
                const cuComplex *alpha,
                const cusparseMatDescr_t descrA,
                const cuComplex *bsrSortedValA,
                const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA,
                const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA,
                int blockDim,
                const cuComplex *x,
                const cuComplex *beta,
                cuComplex *y)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrxmv);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     sizeOfMask,
                     mb,
                     nb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedMaskPtrA,
                     bsrSortedRowPtrA,
                     bsrSortedEndPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     x,
                     beta,
                     y);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrxmv(cusparseHandle_t handle,
                cusparseDirection_t dirA,
                cusparseOperation_t transA,
                int sizeOfMask,
                int mb,
                int nb,
                int nnzb,
                const cuDoubleComplex *alpha,
                const cusparseMatDescr_t descrA,
                const cuDoubleComplex *bsrSortedValA,
                const int *bsrSortedMaskPtrA,
                const int *bsrSortedRowPtrA,
                const int *bsrSortedEndPtrA,
                const int *bsrSortedColIndA,
                int blockDim,
                const cuDoubleComplex *x,
                const cuDoubleComplex *beta,
                cuDoubleComplex *y)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrxmv);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     sizeOfMask,
                     mb,
                     nb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedMaskPtrA,
                     bsrSortedRowPtrA,
                     bsrSortedEndPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     x,
                     beta,
                     y);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsrsv2_zeroPivot(cusparseHandle_t handle,
                          csrsv2Info_t info,
                          int *position)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsrsv2_zeroPivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     position);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_bufferSize(cusparseHandle_t handle,
                           cusparseOperation_t transA,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           float *csrSortedValA,
                           const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA,
                           csrsv2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrsv2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_bufferSize(cusparseHandle_t handle,
                           cusparseOperation_t transA,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           double *csrSortedValA,
                           const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA,
                           csrsv2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrsv2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_bufferSize(cusparseHandle_t handle,
                           cusparseOperation_t transA,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           cuComplex *csrSortedValA,
                           const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA,
                           csrsv2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrsv2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_bufferSize(cusparseHandle_t handle,
                           cusparseOperation_t transA,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex *csrSortedValA,
                           const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA,
                           csrsv2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrsv2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseOperation_t transA,
                              int m,
                              int nnz,
                              const cusparseMatDescr_t descrA,
                              float *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              csrsv2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseOperation_t transA,
                              int m,
                              int nnz,
                              const cusparseMatDescr_t descrA,
                              double *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              csrsv2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseOperation_t transA,
                              int m,
                              int nnz,
                              const cusparseMatDescr_t descrA,
                              cuComplex *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              csrsv2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseOperation_t transA,
                              int m,
                              int nnz,
                              const cusparseMatDescr_t descrA,
                              cuDoubleComplex *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              csrsv2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_analysis(cusparseHandle_t handle,
                         cusparseOperation_t transA,
                         int m,
                         int nnz,
                         const cusparseMatDescr_t descrA,
                         const float *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         csrsv2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrsv2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_analysis(cusparseHandle_t handle,
                         cusparseOperation_t transA,
                         int m,
                         int nnz,
                         const cusparseMatDescr_t descrA,
                         const double *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         csrsv2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrsv2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_analysis(cusparseHandle_t handle,
                         cusparseOperation_t transA,
                         int m,
                         int nnz,
                         const cusparseMatDescr_t descrA,
                         const cuComplex *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         csrsv2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrsv2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_analysis(cusparseHandle_t handle,
                         cusparseOperation_t transA,
                         int m,
                         int nnz,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         csrsv2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrsv2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrsv2_solve(cusparseHandle_t handle,
                      cusparseOperation_t transA,
                      int m,
                      int nnz,
                      const float *alpha,
                      const cusparseMatDescr_t descrA,
                      const float *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      csrsv2Info_t info,
                      const float *f,
                      float *x,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrsv2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     f,
                     x,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsv2_solve(cusparseHandle_t handle,
                      cusparseOperation_t transA,
                      int m,
                      int nnz,
                      const double *alpha,
                      const cusparseMatDescr_t descrA,
                      const double *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      csrsv2Info_t info,
                      const double *f,
                      double *x,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrsv2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     f,
                     x,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsv2_solve(cusparseHandle_t handle,
                      cusparseOperation_t transA,
                      int m,
                      int nnz,
                      const cuComplex *alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      csrsv2Info_t info,
                      const cuComplex *f,
                      cuComplex *x,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrsv2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     f,
                     x,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsv2_solve(cusparseHandle_t handle,
                      cusparseOperation_t transA,
                      int m,
                      int nnz,
                      const cuDoubleComplex *alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      csrsv2Info_t info,
                      const cuDoubleComplex *f,
                      cuDoubleComplex *x,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrsv2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     transA,
                     m,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     f,
                     x,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseXbsrsv2_zeroPivot(cusparseHandle_t handle,
                          bsrsv2Info_t info,
                          int *position)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXbsrsv2_zeroPivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     position);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_bufferSize(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           cusparseOperation_t transA,
                           int mb,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           float *bsrSortedValA,
                           const int *bsrSortedRowPtrA,
                           const int *bsrSortedColIndA,
                           int blockDim,
                           bsrsv2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrsv2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_bufferSize(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           cusparseOperation_t transA,
                           int mb,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           double *bsrSortedValA,
                           const int *bsrSortedRowPtrA,
                           const int *bsrSortedColIndA,
                           int blockDim,
                           bsrsv2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrsv2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_bufferSize(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           cusparseOperation_t transA,
                           int mb,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           cuComplex *bsrSortedValA,
                           const int *bsrSortedRowPtrA,
                           const int *bsrSortedColIndA,
                           int blockDim,
                           bsrsv2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrsv2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_bufferSize(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           cusparseOperation_t transA,
                           int mb,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex *bsrSortedValA,
                           const int *bsrSortedRowPtrA,
                           const int *bsrSortedColIndA,
                           int blockDim,
                           bsrsv2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrsv2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              cusparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const cusparseMatDescr_t descrA,
                              float *bsrSortedValA,
                              const int *bsrSortedRowPtrA,
                              const int *bsrSortedColIndA,
                              int blockSize,
                              bsrsv2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              cusparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const cusparseMatDescr_t descrA,
                              double *bsrSortedValA,
                              const int *bsrSortedRowPtrA,
                              const int *bsrSortedColIndA,
                              int blockSize,
                              bsrsv2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              cusparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const cusparseMatDescr_t descrA,
                              cuComplex *bsrSortedValA,
                              const int *bsrSortedRowPtrA,
                              const int *bsrSortedColIndA,
                              int blockSize,
                              bsrsv2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              cusparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const cusparseMatDescr_t descrA,
                              cuDoubleComplex *bsrSortedValA,
                              const int *bsrSortedRowPtrA,
                              const int *bsrSortedColIndA,
                              int blockSize,
                              bsrsv2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_analysis(cusparseHandle_t handle,
                         cusparseDirection_t dirA,
                         cusparseOperation_t transA,
                         int mb,
                         int nnzb,
                         const cusparseMatDescr_t descrA,
                         const float *bsrSortedValA,
                         const int *bsrSortedRowPtrA,
                         const int *bsrSortedColIndA,
                         int blockDim,
                         bsrsv2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrsv2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_analysis(cusparseHandle_t handle,
                         cusparseDirection_t dirA,
                         cusparseOperation_t transA,
                         int mb,
                         int nnzb,
                         const cusparseMatDescr_t descrA,
                         const double *bsrSortedValA,
                         const int *bsrSortedRowPtrA,
                         const int *bsrSortedColIndA,
                         int blockDim,
                         bsrsv2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrsv2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_analysis(cusparseHandle_t handle,
                         cusparseDirection_t dirA,
                         cusparseOperation_t transA,
                         int mb,
                         int nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuComplex *bsrSortedValA,
                         const int *bsrSortedRowPtrA,
                         const int *bsrSortedColIndA,
                         int blockDim,
                         bsrsv2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrsv2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_analysis(cusparseHandle_t handle,
                         cusparseDirection_t dirA,
                         cusparseOperation_t transA,
                         int mb,
                         int nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex *bsrSortedValA,
                         const int *bsrSortedRowPtrA,
                         const int *bsrSortedColIndA,
                         int blockDim,
                         bsrsv2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrsv2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsv2_solve(cusparseHandle_t handle,
                      cusparseDirection_t dirA,
                      cusparseOperation_t transA,
                      int mb,
                      int nnzb,
                      const float *alpha,
                      const cusparseMatDescr_t descrA,
                      const float *bsrSortedValA,
                      const int *bsrSortedRowPtrA,
                      const int *bsrSortedColIndA,
                      int blockDim,
                      bsrsv2Info_t info,
                      const float *f,
                      float *x,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrsv2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     f,
                     x,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsv2_solve(cusparseHandle_t handle,
                      cusparseDirection_t dirA,
                      cusparseOperation_t transA,
                      int mb,
                      int nnzb,
                      const double *alpha,
                      const cusparseMatDescr_t descrA,
                      const double *bsrSortedValA,
                      const int *bsrSortedRowPtrA,
                      const int *bsrSortedColIndA,
                      int blockDim,
                      bsrsv2Info_t info,
                      const double *f,
                      double *x,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrsv2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     f,
                     x,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsv2_solve(cusparseHandle_t handle,
                      cusparseDirection_t dirA,
                      cusparseOperation_t transA,
                      int mb,
                      int nnzb,
                      const cuComplex *alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex *bsrSortedValA,
                      const int *bsrSortedRowPtrA,
                      const int *bsrSortedColIndA,
                      int blockDim,
                      bsrsv2Info_t info,
                      const cuComplex *f,
                      cuComplex *x,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrsv2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     f,
                     x,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsv2_solve(cusparseHandle_t handle,
                      cusparseDirection_t dirA,
                      cusparseOperation_t transA,
                      int mb,
                      int nnzb,
                      const cuDoubleComplex *alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex *bsrSortedValA,
                      const int *bsrSortedRowPtrA,
                      const int *bsrSortedColIndA,
                      int blockDim,
                      bsrsv2Info_t info,
                      const cuDoubleComplex *f,
                      cuDoubleComplex *x,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrsv2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     mb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     info,
                     f,
                     x,
                     policy,
                     pBuffer);
}

//##############################################################################
//# SPARSE LEVEL 3 ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSbsrmm(cusparseHandle_t handle,
               cusparseDirection_t dirA,
               cusparseOperation_t transA,
               cusparseOperation_t transB,
               int mb,
               int n,
               int kb,
               int nnzb,
               const float *alpha,
               const cusparseMatDescr_t descrA,
               const float *bsrSortedValA,
               const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA,
               const int blockSize,
               const float *B,
               const int ldb,
               const float *beta,
               float *C,
               int ldc)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrmm);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transB,
                     mb,
                     n,
                     kb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockSize,
                     B,
                     ldb,
                     beta,
                     C,
                     ldc);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrmm(cusparseHandle_t handle,
               cusparseDirection_t dirA,
               cusparseOperation_t transA,
               cusparseOperation_t transB,
               int mb,
               int n,
               int kb,
               int nnzb,
               const double *alpha,
               const cusparseMatDescr_t descrA,
               const double *bsrSortedValA,
               const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA,
               const int blockSize,
               const double *B,
               const int ldb,
               const double *beta,
               double *C,
               int ldc)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrmm);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transB,
                     mb,
                     n,
                     kb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockSize,
                     B,
                     ldb,
                     beta,
                     C,
                     ldc);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrmm(cusparseHandle_t handle,
               cusparseDirection_t dirA,
               cusparseOperation_t transA,
               cusparseOperation_t transB,
               int mb,
               int n,
               int kb,
               int nnzb,
               const cuComplex *alpha,
               const cusparseMatDescr_t descrA,
               const cuComplex *bsrSortedValA,
               const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA,
               const int blockSize,
               const cuComplex *B,
               const int ldb,
               const cuComplex *beta,
               cuComplex *C,
               int ldc)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrmm);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transB,
                     mb,
                     n,
                     kb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockSize,
                     B,
                     ldb,
                     beta,
                     C,
                     ldc);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrmm(cusparseHandle_t handle,
               cusparseDirection_t dirA,
               cusparseOperation_t transA,
               cusparseOperation_t transB,
               int mb,
               int n,
               int kb,
               int nnzb,
               const cuDoubleComplex *alpha,
               const cusparseMatDescr_t descrA,
               const cuDoubleComplex *bsrSortedValA,
               const int *bsrSortedRowPtrA,
               const int *bsrSortedColIndA,
               const int blockSize,
               const cuDoubleComplex *B,
               const int ldb,
               const cuDoubleComplex *beta,
               cuDoubleComplex *C,
               int ldc)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrmm);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transB,
                     mb,
                     n,
                     kb,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockSize,
                     B,
                     ldb,
                     beta,
                     C,
                     ldc);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgemmi(cusparseHandle_t handle,
               int m,
               int n,
               int k,
               int nnz,
               const float *alpha,
               const float *A,
               int lda,
               const float *cscValB,
               const int *cscColPtrB,
               const int *cscRowIndB,
               const float *beta,
               float *C,
               int ldc)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgemmi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     nnz,
                     alpha,
                     A,
                     lda,
                     cscValB,
                     cscColPtrB,
                     cscRowIndB,
                     beta,
                     C,
                     ldc);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgemmi(cusparseHandle_t handle,
               int m,
               int n,
               int k,
               int nnz,
               const double *alpha,
               const double *A,
               int lda,
               const double *cscValB,
               const int *cscColPtrB,
               const int *cscRowIndB,
               const double *beta,
               double *C,
               int ldc)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgemmi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     nnz,
                     alpha,
                     A,
                     lda,
                     cscValB,
                     cscColPtrB,
                     cscRowIndB,
                     beta,
                     C,
                     ldc);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgemmi(cusparseHandle_t handle,
               int m,
               int n,
               int k,
               int nnz,
               const cuComplex *alpha,
               const cuComplex *A,
               int lda,
               const cuComplex *cscValB,
               const int *cscColPtrB,
               const int *cscRowIndB,
               const cuComplex *beta,
               cuComplex *C,
               int ldc)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgemmi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     nnz,
                     alpha,
                     A,
                     lda,
                     cscValB,
                     cscColPtrB,
                     cscRowIndB,
                     beta,
                     C,
                     ldc);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgemmi(cusparseHandle_t handle,
               int m,
               int n,
               int k,
               int nnz,
               const cuDoubleComplex *alpha,
               const cuDoubleComplex *A,
               int lda,
               const cuDoubleComplex *cscValB,
               const int *cscColPtrB,
               const int *cscRowIndB,
               const cuDoubleComplex *beta,
               cuDoubleComplex *C,
               int ldc)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgemmi);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     nnz,
                     alpha,
                     A,
                     lda,
                     cscValB,
                     cscColPtrB,
                     cscRowIndB,
                     beta,
                     C,
                     ldc);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrsm2Info(csrsm2Info_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateCsrsm2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrsm2Info(csrsm2Info_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyCsrsm2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsrsm2_zeroPivot(cusparseHandle_t handle,
                          csrsm2Info_t info,
                          int *position)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsrsm2_zeroPivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     position);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrsm2_bufferSizeExt(cusparseHandle_t handle,
                              int algo,
                              cusparseOperation_t transA,
                              cusparseOperation_t transB,
                              int m,
                              int nrhs,
                              int nnz,
                              const float *alpha,
                              const cusparseMatDescr_t descrA,
                              const float *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              const float *B,
                              int ldb,
                              csrsm2Info_t info,
                              cusparseSolvePolicy_t policy,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrsm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm2_bufferSizeExt(cusparseHandle_t handle,
                              int algo,
                              cusparseOperation_t transA,
                              cusparseOperation_t transB,
                              int m,
                              int nrhs,
                              int nnz,
                              const double *alpha,
                              const cusparseMatDescr_t descrA,
                              const double *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              const double *B,
                              int ldb,
                              csrsm2Info_t info,
                              cusparseSolvePolicy_t policy,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrsm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm2_bufferSizeExt(cusparseHandle_t handle,
                              int algo,
                              cusparseOperation_t transA,
                              cusparseOperation_t transB,
                              int m,
                              int nrhs,
                              int nnz,
                              const cuComplex *alpha,
                              const cusparseMatDescr_t descrA,
                              const cuComplex *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              const cuComplex *B,
                              int ldb,
                              csrsm2Info_t info,
                              cusparseSolvePolicy_t policy,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrsm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm2_bufferSizeExt(cusparseHandle_t handle,
                              int algo,
                              cusparseOperation_t transA,
                              cusparseOperation_t transB,
                              int m,
                              int nrhs,
                              int nnz,
                              const cuDoubleComplex *alpha,
                              const cusparseMatDescr_t descrA,
                              const cuDoubleComplex *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              const cuDoubleComplex *B,
                              int ldb,
                              csrsm2Info_t info,
                              cusparseSolvePolicy_t policy,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrsm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrsm2_analysis(cusparseHandle_t handle,
                         int algo,
                         cusparseOperation_t transA,
                         cusparseOperation_t transB,
                         int m,
                         int nrhs,
                         int nnz,
                         const float *alpha,
                         const cusparseMatDescr_t descrA,
                         const float *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         const float *B,
                         int ldb,
                         csrsm2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrsm2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm2_analysis(cusparseHandle_t handle,
                         int algo,
                         cusparseOperation_t transA,
                         cusparseOperation_t transB,
                         int m,
                         int nrhs,
                         int nnz,
                         const double *alpha,
                         const cusparseMatDescr_t descrA,
                         const double *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         const double *B,
                         int ldb,
                         csrsm2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrsm2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm2_analysis(cusparseHandle_t handle,
                         int algo,
                         cusparseOperation_t transA,
                         cusparseOperation_t transB,
                         int m,
                         int nrhs,
                         int nnz,
                         const cuComplex *alpha,
                         const cusparseMatDescr_t descrA,
                         const cuComplex *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         const cuComplex *B,
                         int ldb,
                         csrsm2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrsm2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm2_analysis(cusparseHandle_t handle,
                         int algo,
                         cusparseOperation_t transA,
                         cusparseOperation_t transB,
                         int m,
                         int nrhs,
                         int nnz,
                         const cuDoubleComplex *alpha,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         const cuDoubleComplex *B,
                         int ldb,
                         csrsm2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrsm2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrsm2_solve(cusparseHandle_t handle,
                      int algo,
                      cusparseOperation_t transA,
                      cusparseOperation_t transB,
                      int m,
                      int nrhs,
                      int nnz,
                      const float *alpha,
                      const cusparseMatDescr_t descrA,
                      const float *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      float *B,
                      int ldb,
                      csrsm2Info_t info,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrsm2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrsm2_solve(cusparseHandle_t handle,
                      int algo,
                      cusparseOperation_t transA,
                      cusparseOperation_t transB,
                      int m,
                      int nrhs,
                      int nnz,
                      const double *alpha,
                      const cusparseMatDescr_t descrA,
                      const double *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      double *B,
                      int ldb,
                      csrsm2Info_t info,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrsm2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrsm2_solve(cusparseHandle_t handle,
                      int algo,
                      cusparseOperation_t transA,
                      cusparseOperation_t transB,
                      int m,
                      int nrhs,
                      int nnz,
                      const cuComplex *alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      cuComplex *B,
                      int ldb,
                      csrsm2Info_t info,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrsm2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrsm2_solve(cusparseHandle_t handle,
                      int algo,
                      cusparseOperation_t transA,
                      cusparseOperation_t transB,
                      int m,
                      int nrhs,
                      int nnz,
                      const cuDoubleComplex *alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      cuDoubleComplex *B,
                      int ldb,
                      csrsm2Info_t info,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrsm2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     transA,
                     transB,
                     m,
                     nrhs,
                     nnz,
                     alpha,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     B,
                     ldb,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseXbsrsm2_zeroPivot(cusparseHandle_t handle,
                          bsrsm2Info_t info,
                          int *position)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXbsrsm2_zeroPivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     position);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_bufferSize(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           cusparseOperation_t transA,
                           cusparseOperation_t transXY,
                           int mb,
                           int n,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           float *bsrSortedVal,
                           const int *bsrSortedRowPtr,
                           const int *bsrSortedColInd,
                           int blockSize,
                           bsrsm2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrsm2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_bufferSize(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           cusparseOperation_t transA,
                           cusparseOperation_t transXY,
                           int mb,
                           int n,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           double *bsrSortedVal,
                           const int *bsrSortedRowPtr,
                           const int *bsrSortedColInd,
                           int blockSize,
                           bsrsm2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrsm2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_bufferSize(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           cusparseOperation_t transA,
                           cusparseOperation_t transXY,
                           int mb,
                           int n,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           cuComplex *bsrSortedVal,
                           const int *bsrSortedRowPtr,
                           const int *bsrSortedColInd,
                           int blockSize,
                           bsrsm2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrsm2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_bufferSize(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           cusparseOperation_t transA,
                           cusparseOperation_t transXY,
                           int mb,
                           int n,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex *bsrSortedVal,
                           const int *bsrSortedRowPtr,
                           const int *bsrSortedColInd,
                           int blockSize,
                           bsrsm2Info_t info,
                           int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrsm2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              cusparseOperation_t transA,
                              cusparseOperation_t transB,
                              int mb,
                              int n,
                              int nnzb,
                              const cusparseMatDescr_t descrA,
                              float *bsrSortedVal,
                              const int *bsrSortedRowPtr,
                              const int *bsrSortedColInd,
                              int blockSize,
                              bsrsm2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrsm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transB,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              cusparseOperation_t transA,
                              cusparseOperation_t transB,
                              int mb,
                              int n,
                              int nnzb,
                              const cusparseMatDescr_t descrA,
                              double *bsrSortedVal,
                              const int *bsrSortedRowPtr,
                              const int *bsrSortedColInd,
                              int blockSize,
                              bsrsm2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrsm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transB,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              cusparseOperation_t transA,
                              cusparseOperation_t transB,
                              int mb,
                              int n,
                              int nnzb,
                              const cusparseMatDescr_t descrA,
                              cuComplex *bsrSortedVal,
                              const int *bsrSortedRowPtr,
                              const int *bsrSortedColInd,
                              int blockSize,
                              bsrsm2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrsm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transB,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_bufferSizeExt(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              cusparseOperation_t transA,
                              cusparseOperation_t transB,
                              int mb,
                              int n,
                              int nnzb,
                              const cusparseMatDescr_t descrA,
                              cuDoubleComplex *bsrSortedVal,
                              const int *bsrSortedRowPtr,
                              const int *bsrSortedColInd,
                              int blockSize,
                              bsrsm2Info_t info,
                              size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrsm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transB,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_analysis(cusparseHandle_t handle,
                         cusparseDirection_t dirA,
                         cusparseOperation_t transA,
                         cusparseOperation_t transXY,
                         int mb,
                         int n,
                         int nnzb,
                         const cusparseMatDescr_t descrA,
                         const float *bsrSortedVal,
                         const int *bsrSortedRowPtr,
                         const int *bsrSortedColInd,
                         int blockSize,
                         bsrsm2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrsm2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_analysis(cusparseHandle_t handle,
                         cusparseDirection_t dirA,
                         cusparseOperation_t transA,
                         cusparseOperation_t transXY,
                         int mb,
                         int n,
                         int nnzb,
                         const cusparseMatDescr_t descrA,
                         const double *bsrSortedVal,
                         const int *bsrSortedRowPtr,
                         const int *bsrSortedColInd,
                         int blockSize,
                         bsrsm2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrsm2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_analysis(cusparseHandle_t handle,
                         cusparseDirection_t dirA,
                         cusparseOperation_t transA,
                         cusparseOperation_t transXY,
                         int mb,
                         int n,
                         int nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuComplex *bsrSortedVal,
                         const int *bsrSortedRowPtr,
                         const int *bsrSortedColInd,
                         int blockSize,
                         bsrsm2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrsm2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_analysis(cusparseHandle_t handle,
                         cusparseDirection_t dirA,
                         cusparseOperation_t transA,
                         cusparseOperation_t transXY,
                         int mb,
                         int n,
                         int nnzb,
                         const cusparseMatDescr_t descrA,
                         const cuDoubleComplex *bsrSortedVal,
                         const int *bsrSortedRowPtr,
                         const int *bsrSortedColInd,
                         int blockSize,
                         bsrsm2Info_t info,
                         cusparseSolvePolicy_t policy,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrsm2_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrsm2_solve(cusparseHandle_t handle,
                      cusparseDirection_t dirA,
                      cusparseOperation_t transA,
                      cusparseOperation_t transXY,
                      int mb,
                      int n,
                      int nnzb,
                      const float *alpha,
                      const cusparseMatDescr_t descrA,
                      const float *bsrSortedVal,
                      const int *bsrSortedRowPtr,
                      const int *bsrSortedColInd,
                      int blockSize,
                      bsrsm2Info_t info,
                      const float *B,
                      int ldb,
                      float *X,
                      int ldx,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrsm2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     B,
                     ldb,
                     X,
                     ldx,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrsm2_solve(cusparseHandle_t handle,
                      cusparseDirection_t dirA,
                      cusparseOperation_t transA,
                      cusparseOperation_t transXY,
                      int mb,
                      int n,
                      int nnzb,
                      const double *alpha,
                      const cusparseMatDescr_t descrA,
                      const double *bsrSortedVal,
                      const int *bsrSortedRowPtr,
                      const int *bsrSortedColInd,
                      int blockSize,
                      bsrsm2Info_t info,
                      const double *B,
                      int ldb,
                      double *X,
                      int ldx,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrsm2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     B,
                     ldb,
                     X,
                     ldx,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrsm2_solve(cusparseHandle_t handle,
                      cusparseDirection_t dirA,
                      cusparseOperation_t transA,
                      cusparseOperation_t transXY,
                      int mb,
                      int n,
                      int nnzb,
                      const cuComplex *alpha,
                      const cusparseMatDescr_t descrA,
                      const cuComplex *bsrSortedVal,
                      const int *bsrSortedRowPtr,
                      const int *bsrSortedColInd,
                      int blockSize,
                      bsrsm2Info_t info,
                      const cuComplex *B,
                      int ldb,
                      cuComplex *X,
                      int ldx,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrsm2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     B,
                     ldb,
                     X,
                     ldx,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrsm2_solve(cusparseHandle_t handle,
                      cusparseDirection_t dirA,
                      cusparseOperation_t transA,
                      cusparseOperation_t transXY,
                      int mb,
                      int n,
                      int nnzb,
                      const cuDoubleComplex *alpha,
                      const cusparseMatDescr_t descrA,
                      const cuDoubleComplex *bsrSortedVal,
                      const int *bsrSortedRowPtr,
                      const int *bsrSortedColInd,
                      int blockSize,
                      bsrsm2Info_t info,
                      const cuDoubleComplex *B,
                      int ldb,
                      cuDoubleComplex *X,
                      int ldx,
                      cusparseSolvePolicy_t policy,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrsm2_solve);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     transA,
                     transXY,
                     mb,
                     n,
                     nnzb,
                     alpha,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     B,
                     ldb,
                     X,
                     ldx,
                     policy,
                     pBuffer);
}

//##############################################################################
//# PRECONDITIONERS
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t info,
                               int enable_boost,
                               double *tol,
                               float *boost_val)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrilu02_numericBoost);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     enable_boost,
                     tol,
                     boost_val);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t info,
                               int enable_boost,
                               double *tol,
                               double *boost_val)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrilu02_numericBoost);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     enable_boost,
                     tol,
                     boost_val);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t info,
                               int enable_boost,
                               double *tol,
                               cuComplex *boost_val)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrilu02_numericBoost);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     enable_boost,
                     tol,
                     boost_val);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_numericBoost(cusparseHandle_t handle,
                               csrilu02Info_t info,
                               int enable_boost,
                               double *tol,
                               cuDoubleComplex *boost_val)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrilu02_numericBoost);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     enable_boost,
                     tol,
                     boost_val);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle,
                            csrilu02Info_t info,
                            int *position)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsrilu02_zeroPivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     position);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_bufferSize(cusparseHandle_t handle,
                             int m,
                             int nnz,
                             const cusparseMatDescr_t descrA,
                             float *csrSortedValA,
                             const int *csrSortedRowPtrA,
                             const int *csrSortedColIndA,
                             csrilu02Info_t info,
                             int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrilu02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_bufferSize(cusparseHandle_t handle,
                             int m,
                             int nnz,
                             const cusparseMatDescr_t descrA,
                             double *csrSortedValA,
                             const int *csrSortedRowPtrA,
                             const int *csrSortedColIndA,
                             csrilu02Info_t info,
                             int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrilu02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_bufferSize(cusparseHandle_t handle,
                             int m,
                             int nnz,
                             const cusparseMatDescr_t descrA,
                             cuComplex *csrSortedValA,
                             const int *csrSortedRowPtrA,
                             const int *csrSortedColIndA,
                             csrilu02Info_t info,
                             int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrilu02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_bufferSize(cusparseHandle_t handle,
                             int m,
                             int nnz,
                             const cusparseMatDescr_t descrA,
                             cuDoubleComplex *csrSortedValA,
                             const int *csrSortedRowPtrA,
                             const int *csrSortedColIndA,
                             csrilu02Info_t info,
                             int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrilu02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int nnz,
                                const cusparseMatDescr_t descrA,
                                float *csrSortedVal,
                                const int *csrSortedRowPtr,
                                const int *csrSortedColInd,
                                csrilu02Info_t info,
                                size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrilu02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedVal,
                     csrSortedRowPtr,
                     csrSortedColInd,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int nnz,
                                const cusparseMatDescr_t descrA,
                                double *csrSortedVal,
                                const int *csrSortedRowPtr,
                                const int *csrSortedColInd,
                                csrilu02Info_t info,
                                size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrilu02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedVal,
                     csrSortedRowPtr,
                     csrSortedColInd,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int nnz,
                                const cusparseMatDescr_t descrA,
                                cuComplex *csrSortedVal,
                                const int *csrSortedRowPtr,
                                const int *csrSortedColInd,
                                csrilu02Info_t info,
                                size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrilu02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedVal,
                     csrSortedRowPtr,
                     csrSortedColInd,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int nnz,
                                const cusparseMatDescr_t descrA,
                                cuDoubleComplex *csrSortedVal,
                                const int *csrSortedRowPtr,
                                const int *csrSortedColInd,
                                csrilu02Info_t info,
                                size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrilu02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedVal,
                     csrSortedRowPtr,
                     csrSortedColInd,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02_analysis(cusparseHandle_t handle,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           const float *csrSortedValA,
                           const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA,
                           csrilu02Info_t info,
                           cusparseSolvePolicy_t policy,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrilu02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02_analysis(cusparseHandle_t handle,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           const double *csrSortedValA,
                           const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA,
                           csrilu02Info_t info,
                           cusparseSolvePolicy_t policy,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrilu02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02_analysis(cusparseHandle_t handle,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           const cuComplex *csrSortedValA,
                           const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA,
                           csrilu02Info_t info,
                           cusparseSolvePolicy_t policy,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrilu02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02_analysis(cusparseHandle_t handle,
                           int m,
                           int nnz,
                           const cusparseMatDescr_t descrA,
                           const cuDoubleComplex *csrSortedValA,
                           const int *csrSortedRowPtrA,
                           const int *csrSortedColIndA,
                           csrilu02Info_t info,
                           cusparseSolvePolicy_t policy,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrilu02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrilu02(cusparseHandle_t handle,
                  int m,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  float *csrSortedValA_valM,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  csrilu02Info_t info,
                  cusparseSolvePolicy_t policy,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrilu02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA_valM,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrilu02(cusparseHandle_t handle,
                  int m,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  double *csrSortedValA_valM,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  csrilu02Info_t info,
                  cusparseSolvePolicy_t policy,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrilu02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA_valM,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrilu02(cusparseHandle_t handle,
                  int m,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  cuComplex *csrSortedValA_valM,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  csrilu02Info_t info,
                  cusparseSolvePolicy_t policy,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrilu02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA_valM,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrilu02(cusparseHandle_t handle,
                  int m,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex *csrSortedValA_valM,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  csrilu02Info_t info,
                  cusparseSolvePolicy_t policy,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrilu02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA_valM,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t info,
                               int enable_boost,
                               double *tol,
                               float *boost_val)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrilu02_numericBoost);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     enable_boost,
                     tol,
                     boost_val);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t info,
                               int enable_boost,
                               double *tol,
                               double *boost_val)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrilu02_numericBoost);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     enable_boost,
                     tol,
                     boost_val);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t info,
                               int enable_boost,
                               double *tol,
                               cuComplex *boost_val)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrilu02_numericBoost);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     enable_boost,
                     tol,
                     boost_val);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_numericBoost(cusparseHandle_t handle,
                               bsrilu02Info_t info,
                               int enable_boost,
                               double *tol,
                               cuDoubleComplex *boost_val)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrilu02_numericBoost);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     enable_boost,
                     tol,
                     boost_val);
}

cusparseStatus_t CUSPARSEAPI
cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle,
                            bsrilu02Info_t info,
                            int *position)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXbsrilu02_zeroPivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     position);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_bufferSize(cusparseHandle_t handle,
                             cusparseDirection_t dirA,
                             int mb,
                             int nnzb,
                             const cusparseMatDescr_t descrA,
                             float *bsrSortedVal,
                             const int *bsrSortedRowPtr,
                             const int *bsrSortedColInd,
                             int blockDim,
                             bsrilu02Info_t info,
                             int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrilu02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_bufferSize(cusparseHandle_t handle,
                             cusparseDirection_t dirA,
                             int mb,
                             int nnzb,
                             const cusparseMatDescr_t descrA,
                             double *bsrSortedVal,
                             const int *bsrSortedRowPtr,
                             const int *bsrSortedColInd,
                             int blockDim,
                             bsrilu02Info_t info,
                             int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrilu02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_bufferSize(cusparseHandle_t handle,
                             cusparseDirection_t dirA,
                             int mb,
                             int nnzb,
                             const cusparseMatDescr_t descrA,
                             cuComplex *bsrSortedVal,
                             const int *bsrSortedRowPtr,
                             const int *bsrSortedColInd,
                             int blockDim,
                             bsrilu02Info_t info,
                             int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrilu02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_bufferSize(cusparseHandle_t handle,
                             cusparseDirection_t dirA,
                             int mb,
                             int nnzb,
                             const cusparseMatDescr_t descrA,
                             cuDoubleComplex *bsrSortedVal,
                             const int *bsrSortedRowPtr,
                             const int *bsrSortedColInd,
                             int blockDim,
                             bsrilu02Info_t info,
                             int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrilu02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                cusparseDirection_t dirA,
                                int mb,
                                int nnzb,
                                const cusparseMatDescr_t descrA,
                                float *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd,
                                int blockSize,
                                bsrilu02Info_t info,
                                size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrilu02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                cusparseDirection_t dirA,
                                int mb,
                                int nnzb,
                                const cusparseMatDescr_t descrA,
                                double *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd,
                                int blockSize,
                                bsrilu02Info_t info,
                                size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrilu02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                cusparseDirection_t dirA,
                                int mb,
                                int nnzb,
                                const cusparseMatDescr_t descrA,
                                cuComplex *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd,
                                int blockSize,
                                bsrilu02Info_t info,
                                size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrilu02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_bufferSizeExt(cusparseHandle_t handle,
                                cusparseDirection_t dirA,
                                int mb,
                                int nnzb,
                                const cusparseMatDescr_t descrA,
                                cuDoubleComplex *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd,
                                int blockSize,
                                bsrilu02Info_t info,
                                size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrilu02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02_analysis(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           int mb,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           float *bsrSortedVal,
                           const int *bsrSortedRowPtr,
                           const int *bsrSortedColInd,
                           int blockDim,
                           bsrilu02Info_t info,
                           cusparseSolvePolicy_t policy,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrilu02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02_analysis(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           int mb,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           double *bsrSortedVal,
                           const int *bsrSortedRowPtr,
                           const int *bsrSortedColInd,
                           int blockDim,
                           bsrilu02Info_t info,
                           cusparseSolvePolicy_t policy,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrilu02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02_analysis(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           int mb,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           cuComplex *bsrSortedVal,
                           const int *bsrSortedRowPtr,
                           const int *bsrSortedColInd,
                           int blockDim,
                           bsrilu02Info_t info,
                           cusparseSolvePolicy_t policy,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrilu02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02_analysis(cusparseHandle_t handle,
                           cusparseDirection_t dirA,
                           int mb,
                           int nnzb,
                           const cusparseMatDescr_t descrA,
                           cuDoubleComplex *bsrSortedVal,
                           const int *bsrSortedRowPtr,
                           const int *bsrSortedColInd,
                           int blockDim,
                           bsrilu02Info_t info,
                           cusparseSolvePolicy_t policy,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrilu02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsrilu02(cusparseHandle_t handle,
                  cusparseDirection_t dirA,
                  int mb,
                  int nnzb,
                  const cusparseMatDescr_t descrA,
                  float *bsrSortedVal,
                  const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd,
                  int blockDim,
                  bsrilu02Info_t info,
                  cusparseSolvePolicy_t policy,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsrilu02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsrilu02(cusparseHandle_t handle,
                  cusparseDirection_t dirA,
                  int mb,
                  int nnzb,
                  const cusparseMatDescr_t descrA,
                  double *bsrSortedVal,
                  const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd,
                  int blockDim,
                  bsrilu02Info_t info,
                  cusparseSolvePolicy_t policy,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsrilu02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsrilu02(cusparseHandle_t handle,
                  cusparseDirection_t dirA,
                  int mb,
                  int nnzb,
                  const cusparseMatDescr_t descrA,
                  cuComplex *bsrSortedVal,
                  const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd,
                  int blockDim,
                  bsrilu02Info_t info,
                  cusparseSolvePolicy_t policy,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsrilu02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsrilu02(cusparseHandle_t handle,
                  cusparseDirection_t dirA,
                  int mb,
                  int nnzb,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex *bsrSortedVal,
                  const int *bsrSortedRowPtr,
                  const int *bsrSortedColInd,
                  int blockDim,
                  bsrilu02Info_t info,
                  cusparseSolvePolicy_t policy,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsrilu02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsric02_zeroPivot(cusparseHandle_t handle,
                           csric02Info_t info,
                           int *position)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsric02_zeroPivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     position);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsric02_bufferSize(cusparseHandle_t handle,
                            int m,
                            int nnz,
                            const cusparseMatDescr_t descrA,
                            float *csrSortedValA,
                            const int *csrSortedRowPtrA,
                            const int *csrSortedColIndA,
                            csric02Info_t info,
                            int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsric02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsric02_bufferSize(cusparseHandle_t handle,
                            int m,
                            int nnz,
                            const cusparseMatDescr_t descrA,
                            double *csrSortedValA,
                            const int *csrSortedRowPtrA,
                            const int *csrSortedColIndA,
                            csric02Info_t info,
                            int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsric02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsric02_bufferSize(cusparseHandle_t handle,
                            int m,
                            int nnz,
                            const cusparseMatDescr_t descrA,
                            cuComplex *csrSortedValA,
                            const int *csrSortedRowPtrA,
                            const int *csrSortedColIndA,
                            csric02Info_t info,
                            int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsric02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsric02_bufferSize(cusparseHandle_t handle,
                            int m,
                            int nnz,
                            const cusparseMatDescr_t descrA,
                            cuDoubleComplex *csrSortedValA,
                            const int *csrSortedRowPtrA,
                            const int *csrSortedColIndA,
                            csric02Info_t info,
                            int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsric02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsric02_bufferSizeExt(cusparseHandle_t handle,
                               int m,
                               int nnz,
                               const cusparseMatDescr_t descrA,
                               float *csrSortedVal,
                               const int *csrSortedRowPtr,
                               const int *csrSortedColInd,
                               csric02Info_t info,
                               size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsric02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedVal,
                     csrSortedRowPtr,
                     csrSortedColInd,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsric02_bufferSizeExt(cusparseHandle_t handle,
                               int m,
                               int nnz,
                               const cusparseMatDescr_t descrA,
                               double *csrSortedVal,
                               const int *csrSortedRowPtr,
                               const int *csrSortedColInd,
                               csric02Info_t info,
                               size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsric02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedVal,
                     csrSortedRowPtr,
                     csrSortedColInd,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsric02_bufferSizeExt(cusparseHandle_t handle,
                               int m,
                               int nnz,
                               const cusparseMatDescr_t descrA,
                               cuComplex *csrSortedVal,
                               const int *csrSortedRowPtr,
                               const int *csrSortedColInd,
                               csric02Info_t info,
                               size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsric02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedVal,
                     csrSortedRowPtr,
                     csrSortedColInd,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsric02_bufferSizeExt(cusparseHandle_t handle,
                               int m,
                               int nnz,
                               const cusparseMatDescr_t descrA,
                               cuDoubleComplex *csrSortedVal,
                               const int *csrSortedRowPtr,
                               const int *csrSortedColInd,
                               csric02Info_t info,
                               size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsric02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedVal,
                     csrSortedRowPtr,
                     csrSortedColInd,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsric02_analysis(cusparseHandle_t handle,
                          int m,
                          int nnz,
                          const cusparseMatDescr_t descrA,
                          const float *csrSortedValA,
                          const int *csrSortedRowPtrA,
                          const int *csrSortedColIndA,
                          csric02Info_t info,
                          cusparseSolvePolicy_t policy,
                          void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsric02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsric02_analysis(cusparseHandle_t handle,
                          int m,
                          int nnz,
                          const cusparseMatDescr_t descrA,
                          const double *csrSortedValA,
                          const int *csrSortedRowPtrA,
                          const int *csrSortedColIndA,
                          csric02Info_t info,
                          cusparseSolvePolicy_t policy,
                          void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsric02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsric02_analysis(cusparseHandle_t handle,
                          int m,
                          int nnz,
                          const cusparseMatDescr_t descrA,
                          const cuComplex *csrSortedValA,
                          const int *csrSortedRowPtrA,
                          const int *csrSortedColIndA,
                          csric02Info_t info,
                          cusparseSolvePolicy_t policy,
                          void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsric02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsric02_analysis(cusparseHandle_t handle,
                          int m,
                          int nnz,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex *csrSortedValA,
                          const int *csrSortedRowPtrA,
                          const int *csrSortedColIndA,
                          csric02Info_t info,
                          cusparseSolvePolicy_t policy,
                          void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsric02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsric02(cusparseHandle_t handle,
                 int m,
                 int nnz,
                 const cusparseMatDescr_t descrA,
                 float *csrSortedValA_valM,
                 const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA,
                 csric02Info_t info,
                 cusparseSolvePolicy_t policy,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsric02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA_valM,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsric02(cusparseHandle_t handle,
                 int m,
                 int nnz,
                 const cusparseMatDescr_t descrA,
                 double *csrSortedValA_valM,
                 const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA,
                 csric02Info_t info,
                 cusparseSolvePolicy_t policy,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsric02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA_valM,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsric02(cusparseHandle_t handle,
                 int m,
                 int nnz,
                 const cusparseMatDescr_t descrA,
                 cuComplex *csrSortedValA_valM,
                 const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA,
                 csric02Info_t info,
                 cusparseSolvePolicy_t policy,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsric02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA_valM,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsric02(cusparseHandle_t handle,
                 int m,
                 int nnz,
                 const cusparseMatDescr_t descrA,
                 cuDoubleComplex *csrSortedValA_valM,
                 const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA,
                 csric02Info_t info,
                 cusparseSolvePolicy_t policy,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsric02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA_valM,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseXbsric02_zeroPivot(cusparseHandle_t handle,
                           bsric02Info_t info,
                           int *position)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXbsric02_zeroPivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     info,
                     position);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsric02_bufferSize(cusparseHandle_t handle,
                            cusparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const cusparseMatDescr_t descrA,
                            float *bsrSortedVal,
                            const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd,
                            int blockDim,
                            bsric02Info_t info,
                            int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsric02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsric02_bufferSize(cusparseHandle_t handle,
                            cusparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const cusparseMatDescr_t descrA,
                            double *bsrSortedVal,
                            const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd,
                            int blockDim,
                            bsric02Info_t info,
                            int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsric02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsric02_bufferSize(cusparseHandle_t handle,
                            cusparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const cusparseMatDescr_t descrA,
                            cuComplex *bsrSortedVal,
                            const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd,
                            int blockDim,
                            bsric02Info_t info,
                            int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsric02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsric02_bufferSize(cusparseHandle_t handle,
                            cusparseDirection_t dirA,
                            int mb,
                            int nnzb,
                            const cusparseMatDescr_t descrA,
                            cuDoubleComplex *bsrSortedVal,
                            const int *bsrSortedRowPtr,
                            const int *bsrSortedColInd,
                            int blockDim,
                            bsric02Info_t info,
                            int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsric02_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsric02_bufferSizeExt(cusparseHandle_t handle,
                               cusparseDirection_t dirA,
                               int mb,
                               int nnzb,
                               const cusparseMatDescr_t descrA,
                               float *bsrSortedVal,
                               const int *bsrSortedRowPtr,
                               const int *bsrSortedColInd,
                               int blockSize,
                               bsric02Info_t info,
                               size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsric02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsric02_bufferSizeExt(cusparseHandle_t handle,
                               cusparseDirection_t dirA,
                               int mb,
                               int nnzb,
                               const cusparseMatDescr_t descrA,
                               double *bsrSortedVal,
                               const int *bsrSortedRowPtr,
                               const int *bsrSortedColInd,
                               int blockSize,
                               bsric02Info_t info,
                               size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsric02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsric02_bufferSizeExt(cusparseHandle_t handle,
                               cusparseDirection_t dirA,
                               int mb,
                               int nnzb,
                               const cusparseMatDescr_t descrA,
                               cuComplex *bsrSortedVal,
                               const int *bsrSortedRowPtr,
                               const int *bsrSortedColInd,
                               int blockSize,
                               bsric02Info_t info,
                               size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsric02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsric02_bufferSizeExt(cusparseHandle_t handle,
                               cusparseDirection_t dirA,
                               int mb,
                               int nnzb,
                               const cusparseMatDescr_t descrA,
                               cuDoubleComplex *bsrSortedVal,
                               const int *bsrSortedRowPtr,
                               const int *bsrSortedColInd,
                               int blockSize,
                               bsric02Info_t info,
                               size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsric02_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockSize,
                     info,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsric02_analysis(cusparseHandle_t handle,
                          cusparseDirection_t dirA,
                          int mb,
                          int nnzb,
                          const cusparseMatDescr_t descrA,
                          const float *bsrSortedVal,
                          const int *bsrSortedRowPtr,
                          const int *bsrSortedColInd,
                          int blockDim,
                          bsric02Info_t info,
                          cusparseSolvePolicy_t policy,
                          void *pInputBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsric02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pInputBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsric02_analysis(cusparseHandle_t handle,
                          cusparseDirection_t dirA,
                          int mb,
                          int nnzb,
                          const cusparseMatDescr_t descrA,
                          const double *bsrSortedVal,
                          const int *bsrSortedRowPtr,
                          const int *bsrSortedColInd,
                          int blockDim,
                          bsric02Info_t info,
                          cusparseSolvePolicy_t policy,
                          void *pInputBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsric02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pInputBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsric02_analysis(cusparseHandle_t handle,
                          cusparseDirection_t dirA,
                          int mb,
                          int nnzb,
                          const cusparseMatDescr_t descrA,
                          const cuComplex *bsrSortedVal,
                          const int *bsrSortedRowPtr,
                          const int *bsrSortedColInd,
                          int blockDim,
                          bsric02Info_t info,
                          cusparseSolvePolicy_t policy,
                          void *pInputBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsric02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pInputBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsric02_analysis(cusparseHandle_t handle,
                          cusparseDirection_t dirA,
                          int mb,
                          int nnzb,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex *bsrSortedVal,
                          const int *bsrSortedRowPtr,
                          const int *bsrSortedColInd,
                          int blockDim,
                          bsric02Info_t info,
                          cusparseSolvePolicy_t policy,
                          void *pInputBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsric02_analysis);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pInputBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsric02(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int mb,
                 int nnzb,
                 const cusparseMatDescr_t descrA,
                 float *bsrSortedVal,
                 const int *bsrSortedRowPtr,
                 const int *bsrSortedColInd,
                 int blockDim,
                 bsric02Info_t info,
                 cusparseSolvePolicy_t policy,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsric02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsric02(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int mb,
                 int nnzb,
                 const cusparseMatDescr_t descrA,
                 double *bsrSortedVal,
                 const int *bsrSortedRowPtr,
                 const int *bsrSortedColInd,
                 int blockDim,
                 bsric02Info_t info,
                 cusparseSolvePolicy_t policy,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsric02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsric02(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int mb,
                 int nnzb,
                 const cusparseMatDescr_t descrA,
                 cuComplex *bsrSortedVal,
                 const int *bsrSortedRowPtr,
                 const int *
                     bsrSortedColInd,
                 int blockDim,
                 bsric02Info_t info,
                 cusparseSolvePolicy_t policy,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsric02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsric02(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int mb,
                 int nnzb,
                 const cusparseMatDescr_t descrA,
                 cuDoubleComplex *bsrSortedVal,
                 const int *bsrSortedRowPtr,
                 const int *bsrSortedColInd,
                 int blockDim,
                 bsric02Info_t info,
                 cusparseSolvePolicy_t policy,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsric02);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nnzb,
                     descrA,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     blockDim,
                     info,
                     policy,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int m,
                             int n,
                             const float *dl,
                             const float *d,
                             const float *du,
                             const float *B,
                             int ldb,
                             size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgtsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int m,
                             int n,
                             const double *dl,
                             const double *d,
                             const double *du,
                             const double *B,
                             int ldb,
                             size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgtsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int m,
                             int n,
                             const cuComplex *dl,
                             const cuComplex *d,
                             const cuComplex *du,
                             const cuComplex *B,
                             int ldb,
                             size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgtsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2_bufferSizeExt(cusparseHandle_t handle,
                             int m,
                             int n,
                             const cuDoubleComplex *dl,
                             const cuDoubleComplex *d,
                             const cuDoubleComplex *du,
                             const cuDoubleComplex *B,
                             int ldb,
                             size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgtsv2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2(cusparseHandle_t handle,
               int m,
               int n,
               const float *dl,
               const float *d,
               const float *du,
               float *B,
               int ldb,
               void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgtsv2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2(cusparseHandle_t handle,
               int m,
               int n,
               const double *dl,
               const double *d,
               const double *du,
               double *B,
               int ldb,
               void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgtsv2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2(cusparseHandle_t handle,
               int m,
               int n,
               const cuComplex *dl,
               const cuComplex *d,
               const cuComplex *du,
               cuComplex *B,
               int ldb,
               void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgtsv2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2(cusparseHandle_t handle,
               int m,
               int n,
               const cuDoubleComplex *dl,
               const cuDoubleComplex *d,
               const cuDoubleComplex *du,
               cuDoubleComplex *B,
               int ldb,
               void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgtsv2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int m,
                                     int n,
                                     const float *dl,
                                     const float *d,
                                     const float *du,
                                     const float *B,
                                     int ldb,
                                     size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgtsv2_nopivot_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int m,
                                     int n,
                                     const double *dl,
                                     const double *d,
                                     const double *du,
                                     const double *B,
                                     int ldb,
                                     size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgtsv2_nopivot_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int m,
                                     int n,
                                     const cuComplex *dl,
                                     const cuComplex *d,
                                     const cuComplex *du,
                                     const cuComplex *B,
                                     int ldb,
                                     size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgtsv2_nopivot_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2_nopivot_bufferSizeExt(cusparseHandle_t handle,
                                     int m,
                                     int n,
                                     const cuDoubleComplex *dl,
                                     const cuDoubleComplex *d,
                                     const cuDoubleComplex *du,
                                     const cuDoubleComplex *B,
                                     int ldb,
                                     size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgtsv2_nopivot_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2_nopivot(cusparseHandle_t handle,
                       int m,
                       int n,
                       const float *dl,
                       const float *d,
                       const float *du,
                       float *B,
                       int ldb,
                       void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgtsv2_nopivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2_nopivot(cusparseHandle_t handle,
                       int m,
                       int n,
                       const double *dl,
                       const double *d,
                       const double *du,
                       double *B,
                       int ldb,
                       void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgtsv2_nopivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2_nopivot(cusparseHandle_t handle,
                       int m,
                       int n,
                       const cuComplex *dl,
                       const cuComplex *d,
                       const cuComplex *du,
                       cuComplex *B,
                       int ldb,
                       void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgtsv2_nopivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2_nopivot(cusparseHandle_t handle,
                       int m,
                       int n,
                       const cuDoubleComplex *dl,
                       const cuDoubleComplex *d,
                       const cuDoubleComplex *du,
                       cuDoubleComplex *B,
                       int ldb,
                       void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgtsv2_nopivot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     dl,
                     d,
                     du,
                     B,
                     ldb,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int m,
                                         const float *dl,
                                         const float *d,
                                         const float *du,
                                         const float *x,
                                         int batchCount,
                                         int batchStride,
                                         size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgtsv2StridedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     batchStride,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int m,
                                         const double *dl,
                                         const double *d,
                                         const double *du,
                                         const double *x,
                                         int batchCount,
                                         int batchStride,
                                         size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgtsv2StridedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     batchStride,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int m,
                                         const cuComplex *dl,
                                         const cuComplex *d,
                                         const cuComplex *du,
                                         const cuComplex *x,
                                         int batchCount,
                                         int batchStride,
                                         size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgtsv2StridedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     batchStride,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2StridedBatch_bufferSizeExt(cusparseHandle_t handle,
                                         int m,
                                         const cuDoubleComplex *dl,
                                         const cuDoubleComplex *d,
                                         const cuDoubleComplex *du,
                                         const cuDoubleComplex *x,
                                         int batchCount,
                                         int batchStride,
                                         size_t *bufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgtsv2StridedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     batchStride,
                     bufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgtsv2StridedBatch(cusparseHandle_t handle,
                           int m,
                           const float *dl,
                           const float *d,
                           const float *du,
                           float *x,
                           int batchCount,
                           int batchStride,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgtsv2StridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     batchStride,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgtsv2StridedBatch(cusparseHandle_t handle,
                           int m,
                           const double *dl,
                           const double *d,
                           const double *du,
                           double *x,
                           int batchCount,
                           int batchStride,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgtsv2StridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     batchStride,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgtsv2StridedBatch(cusparseHandle_t handle,
                           int m,
                           const cuComplex *dl,
                           const cuComplex *d,
                           const cuComplex *du,
                           cuComplex *x,
                           int batchCount,
                           int batchStride,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgtsv2StridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     batchStride,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgtsv2StridedBatch(cusparseHandle_t handle,
                           int m,
                           const cuDoubleComplex *dl,
                           const cuDoubleComplex *d,
                           const cuDoubleComplex *du,
                           cuDoubleComplex *x,
                           int batchCount,
                           int batchStride,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgtsv2StridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     batchStride,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int algo,
                                            int m,
                                            const float *dl,
                                            const float *d,
                                            const float *du,
                                            const float *x,
                                            int batchCount,
                                            size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgtsvInterleavedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int algo,
                                            int m,
                                            const double *dl,
                                            const double *d,
                                            const double *du,
                                            const double *x,
                                            int batchCount,
                                            size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgtsvInterleavedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int algo,
                                            int m,
                                            const cuComplex *dl,
                                            const cuComplex *d,
                                            const cuComplex *du,
                                            const cuComplex *x,
                                            int batchCount,
                                            size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgtsvInterleavedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgtsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int algo,
                                            int m,
                                            const cuDoubleComplex *dl,
                                            const cuDoubleComplex *d,
                                            const cuDoubleComplex *du,
                                            const cuDoubleComplex *x,
                                            int batchCount,
                                            size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgtsvInterleavedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgtsvInterleavedBatch(cusparseHandle_t handle,
                              int algo,
                              int m,
                              float *dl,
                              float *d,
                              float *du,
                              float *x,
                              int batchCount,
                              void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgtsvInterleavedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgtsvInterleavedBatch(cusparseHandle_t handle,
                              int algo,
                              int m,
                              double *dl,
                              double *d,
                              double *du,
                              double *x,
                              int batchCount,
                              void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgtsvInterleavedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgtsvInterleavedBatch(cusparseHandle_t handle,
                              int algo,
                              int m,
                              cuComplex *dl,
                              cuComplex *d,
                              cuComplex *du,
                              cuComplex *x,
                              int batchCount,
                              void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgtsvInterleavedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgtsvInterleavedBatch(cusparseHandle_t handle,
                              int algo,
                              int m,
                              cuDoubleComplex *dl,
                              cuDoubleComplex *d,
                              cuDoubleComplex *du,
                              cuDoubleComplex *x,
                              int batchCount,
                              void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgtsvInterleavedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     dl,
                     d,
                     du,
                     x,
                     batchCount,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int algo,
                                            int m,
                                            const float *ds,
                                            const float *dl,
                                            const float *d,
                                            const float *du,
                                            const float *dw,
                                            const float *x,
                                            int batchCount,
                                            size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgpsvInterleavedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     ds,
                     dl,
                     d,
                     du,
                     dw,
                     x,
                     batchCount,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int algo,
                                            int m,
                                            const double *ds,
                                            const double *dl,
                                            const double *d,
                                            const double *du,
                                            const double *dw,
                                            const double *x,
                                            int batchCount,
                                            size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgpsvInterleavedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     ds,
                     dl,
                     d,
                     du,
                     dw,
                     x,
                     batchCount,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int algo,
                                            int m,
                                            const cuComplex *ds,
                                            const cuComplex *dl,
                                            const cuComplex *d,
                                            const cuComplex *du,
                                            const cuComplex *dw,
                                            const cuComplex *x,
                                            int batchCount,
                                            size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgpsvInterleavedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     ds,
                     dl,
                     d,
                     du,
                     dw,
                     x,
                     batchCount,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgpsvInterleavedBatch_bufferSizeExt(cusparseHandle_t handle,
                                            int algo,
                                            int m,
                                            const cuDoubleComplex *ds,
                                            const cuDoubleComplex *dl,
                                            const cuDoubleComplex *d,
                                            const cuDoubleComplex *du,
                                            const cuDoubleComplex *dw,
                                            const cuDoubleComplex *x,
                                            int batchCount,
                                            size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgpsvInterleavedBatch_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     ds,
                     dl,
                     d,
                     du,
                     dw,
                     x,
                     batchCount,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgpsvInterleavedBatch(cusparseHandle_t handle,
                              int algo,
                              int m,
                              float *ds,
                              float *dl,
                              float *d,
                              float *du,
                              float *dw,
                              float *x,
                              int batchCount,
                              void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgpsvInterleavedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     ds,
                     dl,
                     d,
                     du,
                     dw,
                     x,
                     batchCount,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgpsvInterleavedBatch(cusparseHandle_t handle,
                              int algo,
                              int m,
                              double *ds,
                              double *dl,
                              double *d,
                              double *du,
                              double *dw,
                              double *x,
                              int batchCount,
                              void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgpsvInterleavedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     ds,
                     dl,
                     d,
                     du,
                     dw,
                     x,
                     batchCount,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgpsvInterleavedBatch(cusparseHandle_t handle,
                              int algo,
                              int m,
                              cuComplex *ds,
                              cuComplex *dl,
                              cuComplex *d,
                              cuComplex *du,
                              cuComplex *dw,
                              cuComplex *x,
                              int batchCount,
                              void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgpsvInterleavedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     ds,
                     dl,
                     d,
                     du,
                     dw,
                     x,
                     batchCount,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgpsvInterleavedBatch(cusparseHandle_t handle,
                              int algo,
                              int m,
                              cuDoubleComplex *ds,
                              cuDoubleComplex *dl,
                              cuDoubleComplex *d,
                              cuDoubleComplex *du,
                              cuDoubleComplex *dw,
                              cuDoubleComplex *x,
                              int batchCount,
                              void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgpsvInterleavedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     algo,
                     m,
                     ds,
                     dl,
                     d,
                     du,
                     dw,
                     x,
                     batchCount,
                     pBuffer);
}

//##############################################################################
//# EXTRA ROUTINES
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsrgemm2Info(csrgemm2Info_t *info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateCsrgemm2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyCsrgemm2Info(csrgemm2Info_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyCsrgemm2Info);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(info);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrgemm2_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const float *alpha,
                                const cusparseMatDescr_t descrA,
                                int nnzA,
                                const int *csrSortedRowPtrA,
                                const int *csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int nnzB,
                                const int *csrSortedRowPtrB,
                                const int *csrSortedColIndB,
                                const float *beta,
                                const cusparseMatDescr_t descrD,
                                int nnzD,
                                const int *csrSortedRowPtrD,
                                const int *csrSortedColIndD,
                                csrgemm2Info_t info,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrgemm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     beta,
                     descrD,
                     nnzD,
                     csrSortedRowPtrD,
                     csrSortedColIndD,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const double *alpha,
                                const cusparseMatDescr_t descrA,
                                int nnzA,
                                const int *csrSortedRowPtrA,
                                const int *csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int nnzB,
                                const int *csrSortedRowPtrB,
                                const int *csrSortedColIndB,
                                const double *beta,
                                const cusparseMatDescr_t descrD,
                                int nnzD,
                                const int *csrSortedRowPtrD,
                                const int *csrSortedColIndD,
                                csrgemm2Info_t info,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrgemm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     beta,
                     descrD,
                     nnzD,
                     csrSortedRowPtrD,
                     csrSortedColIndD,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrgemm2_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const cuComplex *alpha,
                                const cusparseMatDescr_t descrA,
                                int nnzA,
                                const int *csrSortedRowPtrA,
                                const int *csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int nnzB,
                                const int *csrSortedRowPtrB,
                                const int *csrSortedColIndB,
                                const cuComplex *beta,
                                const cusparseMatDescr_t descrD,
                                int nnzD,
                                const int *csrSortedRowPtrD,
                                const int *csrSortedColIndD,
                                csrgemm2Info_t info,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrgemm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     beta,
                     descrD,
                     nnzD,
                     csrSortedRowPtrD,
                     csrSortedColIndD,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrgemm2_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                int k,
                                const cuDoubleComplex *alpha,
                                const cusparseMatDescr_t descrA,
                                int nnzA,
                                const int *csrSortedRowPtrA,
                                const int *csrSortedColIndA,
                                const cusparseMatDescr_t descrB,
                                int nnzB,
                                const int *csrSortedRowPtrB,
                                const int *csrSortedColIndB,
                                const cuDoubleComplex *beta,
                                const cusparseMatDescr_t descrD,
                                int nnzD,
                                const int *csrSortedRowPtrD,
                                const int *csrSortedColIndD,
                                csrgemm2Info_t info,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrgemm2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     beta,
                     descrD,
                     nnzD,
                     csrSortedRowPtrD,
                     csrSortedColIndD,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsrgemm2Nnz(cusparseHandle_t handle,
                     int m,
                     int n,
                     int k,
                     const cusparseMatDescr_t descrA,
                     int nnzA,
                     const int *csrSortedRowPtrA,
                     const int *csrSortedColIndA,
                     const cusparseMatDescr_t descrB,
                     int nnzB,
                     const int *csrSortedRowPtrB,
                     const int *csrSortedColIndB,
                     const cusparseMatDescr_t descrD,
                     int nnzD,
                     const int *csrSortedRowPtrD,
                     const int *csrSortedColIndD,
                     const cusparseMatDescr_t descrC,
                     int *csrSortedRowPtrC,
                     int *nnzTotalDevHostPtr,
                     const csrgemm2Info_t info,
                     void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsrgemm2Nnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     descrA,
                     nnzA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     descrD,
                     nnzD,
                     csrSortedRowPtrD,
                     csrSortedColIndD,
                     descrC,
                     csrSortedRowPtrC,
                     nnzTotalDevHostPtr,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrgemm2(cusparseHandle_t handle,
                  int m,
                  int n,
                  int k,
                  const float *alpha,
                  const cusparseMatDescr_t descrA,
                  int nnzA,
                  const float *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int nnzB,
                  const float *csrSortedValB,
                  const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB,
                  const float *beta,
                  const cusparseMatDescr_t descrD,
                  int nnzD,
                  const float *csrSortedValD,
                  const int *csrSortedRowPtrD,
                  const int *csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  float *csrSortedValC,
                  const int *csrSortedRowPtrC,
                  int *csrSortedColIndC,
                  const csrgemm2Info_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrgemm2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     beta,
                     descrD,
                     nnzD,
                     csrSortedValD,
                     csrSortedRowPtrD,
                     csrSortedColIndD,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrgemm2(cusparseHandle_t handle,
                  int m,
                  int n,
                  int k,
                  const double *alpha,
                  const cusparseMatDescr_t descrA,
                  int nnzA,
                  const double *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int nnzB,
                  const double *csrSortedValB,
                  const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB,
                  const double *beta,
                  const cusparseMatDescr_t descrD,
                  int nnzD,
                  const double *csrSortedValD,
                  const int *csrSortedRowPtrD,
                  const int *csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  double *csrSortedValC,
                  const int *csrSortedRowPtrC,
                  int *csrSortedColIndC,
                  const csrgemm2Info_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrgemm2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     beta,
                     descrD,
                     nnzD,
                     csrSortedValD,
                     csrSortedRowPtrD,
                     csrSortedColIndD,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrgemm2(cusparseHandle_t handle,
                  int m,
                  int n,
                  int k,
                  const cuComplex *alpha,
                  const cusparseMatDescr_t descrA,
                  int nnzA,
                  const cuComplex *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int nnzB,
                  const cuComplex *csrSortedValB,
                  const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB,
                  const cuComplex *beta,
                  const cusparseMatDescr_t descrD,
                  int nnzD,
                  const cuComplex *csrSortedValD,
                  const int *csrSortedRowPtrD,
                  const int *csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  cuComplex *csrSortedValC,
                  const int *csrSortedRowPtrC,
                  int *csrSortedColIndC,
                  const csrgemm2Info_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrgemm2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     beta,
                     descrD,
                     nnzD,
                     csrSortedValD,
                     csrSortedRowPtrD,
                     csrSortedColIndD,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrgemm2(cusparseHandle_t handle,
                  int m,
                  int n,
                  int k,
                  const cuDoubleComplex *alpha,
                  const cusparseMatDescr_t descrA,
                  int nnzA,
                  const cuDoubleComplex *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const cusparseMatDescr_t descrB,
                  int nnzB,
                  const cuDoubleComplex *csrSortedValB,
                  const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB,
                  const cuDoubleComplex *beta,
                  const cusparseMatDescr_t descrD,
                  int nnzD,
                  const cuDoubleComplex *csrSortedValD,
                  const int *csrSortedRowPtrD,
                  const int *csrSortedColIndD,
                  const cusparseMatDescr_t descrC,
                  cuDoubleComplex *csrSortedValC,
                  const int *csrSortedRowPtrC,
                  int *csrSortedColIndC,
                  const csrgemm2Info_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrgemm2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     k,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     beta,
                     descrD,
                     nnzD,
                     csrSortedValD,
                     csrSortedRowPtrD,
                     csrSortedColIndD,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrgeam2_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                const float *alpha,
                                const cusparseMatDescr_t descrA,
                                int nnzA,
                                const float *csrSortedValA,
                                const int *csrSortedRowPtrA,
                                const int *csrSortedColIndA,
                                const float *beta,
                                const cusparseMatDescr_t descrB,
                                int nnzB,
                                const float *csrSortedValB,
                                const int *csrSortedRowPtrB,
                                const int *csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const float *csrSortedValC,
                                const int *csrSortedRowPtrC,
                                const int *csrSortedColIndC,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrgeam2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     beta,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrgeam2_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                const double *alpha,
                                const cusparseMatDescr_t descrA,
                                int nnzA,
                                const double *csrSortedValA,
                                const int *csrSortedRowPtrA,
                                const int *csrSortedColIndA,
                                const double *beta,
                                const cusparseMatDescr_t descrB,
                                int nnzB,
                                const double *csrSortedValB,
                                const int *csrSortedRowPtrB,
                                const int *csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const double *csrSortedValC,
                                const int *csrSortedRowPtrC,
                                const int *csrSortedColIndC,
                                size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI
cusparseCcsrgeam2_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                const cuComplex *alpha,
                                const cusparseMatDescr_t descrA,
                                int nnzA,
                                const cuComplex *csrSortedValA,
                                const int *csrSortedRowPtrA,
                                const int *csrSortedColIndA,
                                const cuComplex *beta,
                                const cusparseMatDescr_t descrB,
                                int nnzB,
                                const cuComplex *csrSortedValB,
                                const int *csrSortedRowPtrB,
                                const int *csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const cuComplex *csrSortedValC,
                                const int *csrSortedRowPtrC,
                                const int *csrSortedColIndC,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrgeam2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     beta,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrgeam2_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                const cuDoubleComplex *alpha,
                                const cusparseMatDescr_t descrA,
                                int nnzA,
                                const cuDoubleComplex *csrSortedValA,
                                const int *csrSortedRowPtrA,
                                const int *csrSortedColIndA,
                                const cuDoubleComplex *beta,
                                const cusparseMatDescr_t descrB,
                                int nnzB,
                                const cuDoubleComplex *csrSortedValB,
                                const int *csrSortedRowPtrB,
                                const int *csrSortedColIndB,
                                const cusparseMatDescr_t descrC,
                                const cuDoubleComplex *csrSortedValC,
                                const int *csrSortedRowPtrC,
                                const int *csrSortedColIndC,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrgeam2_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     beta,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsrgeam2Nnz(cusparseHandle_t handle,
                     int m,
                     int n,
                     const cusparseMatDescr_t descrA,
                     int nnzA,
                     const int *csrSortedRowPtrA,
                     const int *csrSortedColIndA,
                     const cusparseMatDescr_t descrB,
                     int nnzB,
                     const int *csrSortedRowPtrB,
                     const int *csrSortedColIndB,
                     const cusparseMatDescr_t descrC,
                     int *csrSortedRowPtrC,
                     int *nnzTotalDevHostPtr,
                     void *workspace)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsrgeam2Nnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     nnzA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrB,
                     nnzB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     descrC,
                     csrSortedRowPtrC,
                     nnzTotalDevHostPtr,
                     workspace);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsrgeam2(cusparseHandle_t handle,
                  int m,
                  int n,
                  const float *alpha,
                  const cusparseMatDescr_t descrA,
                  int nnzA,
                  const float *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const float *beta,
                  const cusparseMatDescr_t descrB,
                  int nnzB,
                  const float *csrSortedValB,
                  const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  float *csrSortedValC,
                  int *csrSortedRowPtrC,
                  int *csrSortedColIndC,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrgeam2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     beta,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrgeam2(cusparseHandle_t handle,
                  int m,
                  int n,
                  const double *alpha,
                  const cusparseMatDescr_t descrA,
                  int nnzA,
                  const double *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const double *beta,
                  const cusparseMatDescr_t descrB,
                  int nnzB,
                  const double *csrSortedValB,
                  const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  double *csrSortedValC,
                  int *csrSortedRowPtrC,
                  int *csrSortedColIndC,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrgeam2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     beta,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrgeam2(cusparseHandle_t handle,
                  int m,
                  int n,
                  const cuComplex *alpha,
                  const cusparseMatDescr_t descrA,
                  int nnzA,
                  const cuComplex *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const cuComplex *beta,
                  const cusparseMatDescr_t descrB,
                  int nnzB,
                  const cuComplex *csrSortedValB,
                  const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  cuComplex *csrSortedValC,
                  int *csrSortedRowPtrC,
                  int *csrSortedColIndC,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrgeam2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     beta,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrgeam2(cusparseHandle_t handle,
                  int m,
                  int n,
                  const cuDoubleComplex *alpha,
                  const cusparseMatDescr_t descrA,
                  int nnzA,
                  const cuDoubleComplex *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const cuDoubleComplex *beta,
                  const cusparseMatDescr_t descrB,
                  int nnzB,
                  const cuDoubleComplex *csrSortedValB,
                  const int *csrSortedRowPtrB,
                  const int *csrSortedColIndB,
                  const cusparseMatDescr_t descrC,
                  cuDoubleComplex *csrSortedValC,
                  int *csrSortedRowPtrC,
                  int *csrSortedColIndC,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrgeam2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     alpha,
                     descrA,
                     nnzA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     beta,
                     descrB,
                     nnzB,
                     csrSortedValB,
                     csrSortedRowPtrB,
                     csrSortedColIndB,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}

//##############################################################################
//# SPARSE MATRIX REORDERING
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseScsrcolor(cusparseHandle_t handle,
                  int m,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  const float *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const float *fractionToColor,
                  int *ncolors,
                  int *coloring,
                  int *reordering,
                  const cusparseColorInfo_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsrcolor);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     fractionToColor,
                     ncolors,
                     coloring,
                     reordering,
                     info);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsrcolor(cusparseHandle_t handle,
                  int m,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  const double *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const double *fractionToColor,
                  int *ncolors,
                  int *coloring,
                  int *reordering,
                  const cusparseColorInfo_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsrcolor);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     fractionToColor,
                     ncolors,
                     coloring,
                     reordering,
                     info);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsrcolor(cusparseHandle_t handle,
                  int m,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  const cuComplex *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const float *fractionToColor,
                  int *ncolors,
                  int *coloring,
                  int *reordering,
                  const cusparseColorInfo_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsrcolor);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     fractionToColor,
                     ncolors,
                     coloring,
                     reordering,
                     info);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsrcolor(cusparseHandle_t handle,
                  int m,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  const cuDoubleComplex *csrSortedValA,
                  const int *csrSortedRowPtrA,
                  const int *csrSortedColIndA,
                  const double *fractionToColor,
                  int *ncolors,
                  int *coloring,
                  int *reordering,
                  const cusparseColorInfo_t info)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsrcolor);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     nnz,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     fractionToColor,
                     ncolors,
                     coloring,
                     reordering,
                     info);
}

//##############################################################################
//# SPARSE FORMAT CONVERSION
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSnnz(cusparseHandle_t handle,
             cusparseDirection_t dirA,
             int m,
             int n,
             const cusparseMatDescr_t descrA,
             const float *A,
             int lda,
             int *nnzPerRowCol,
             int *nnzTotalDevHostPtr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSnnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerRowCol,
                     nnzTotalDevHostPtr);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnnz(cusparseHandle_t handle,
             cusparseDirection_t dirA,
             int m,
             int n,
             const cusparseMatDescr_t descrA,
             const double *A,
             int lda,
             int *nnzPerRowCol,
             int *nnzTotalDevHostPtr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerRowCol,
                     nnzTotalDevHostPtr);
}

cusparseStatus_t CUSPARSEAPI
cusparseCnnz(cusparseHandle_t handle,
             cusparseDirection_t dirA,
             int m,
             int n,
             const cusparseMatDescr_t descrA,
             const cuComplex *A,
             int lda,
             int *nnzPerRowCol,
             int *nnzTotalDevHostPtr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCnnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerRowCol,
                     nnzTotalDevHostPtr);
}

cusparseStatus_t CUSPARSEAPI
cusparseZnnz(cusparseHandle_t handle,
             cusparseDirection_t dirA,
             int m,
             int n,
             const cusparseMatDescr_t descrA,
             const cuDoubleComplex *A,
             int lda,
             int *nnzPerRowCol,
             int *nnzTotalDevHostPtr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZnnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerRowCol,
                     nnzTotalDevHostPtr);
}

//##############################################################################
//# SPARSE FORMAT CONVERSION #
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSnnz_compress(cusparseHandle_t handle,
                      int m,
                      const cusparseMatDescr_t descr,
                      const float *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      int *nnzPerRow,
                      int *nnzC,
                      float tol)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSnnz_compress);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     descr,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     nnzPerRow,
                     nnzC,
                     tol);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnnz_compress(cusparseHandle_t handle,
                      int m,
                      const cusparseMatDescr_t descr,
                      const double *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      int *nnzPerRow,
                      int *nnzC,
                      double tol)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnnz_compress);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     descr,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     nnzPerRow,
                     nnzC,
                     tol);
}

cusparseStatus_t CUSPARSEAPI
cusparseCnnz_compress(cusparseHandle_t handle,
                      int m,
                      const cusparseMatDescr_t descr,
                      const cuComplex *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      int *nnzPerRow,
                      int *nnzC,
                      cuComplex tol)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCnnz_compress);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     descr,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     nnzPerRow,
                     nnzC,
                     tol);
}

cusparseStatus_t CUSPARSEAPI
cusparseZnnz_compress(cusparseHandle_t handle,
                      int m,
                      const cusparseMatDescr_t descr,
                      const cuDoubleComplex *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      int *nnzPerRow,
                      int *nnzC,
                      cuDoubleComplex tol)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZnnz_compress);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     descr,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     nnzPerRow,
                     nnzC,
                     tol);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsr2csr_compress(cusparseHandle_t handle,
                          int m,
                          int n,
                          const cusparseMatDescr_t descrA,
                          const float *csrSortedValA,
                          const int *csrSortedColIndA,
                          const int *csrSortedRowPtrA,
                          int nnzA,
                          const int *nnzPerRow,
                          float *csrSortedValC,
                          int *csrSortedColIndC,
                          int *csrSortedRowPtrC,
                          float tol)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsr2csr_compress);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedColIndA,
                     csrSortedRowPtrA,
                     nnzA,
                     nnzPerRow,
                     csrSortedValC,
                     csrSortedColIndC,
                     csrSortedRowPtrC,
                     tol);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2csr_compress(cusparseHandle_t handle,
                          int m,
                          int n,
                          const cusparseMatDescr_t descrA,
                          const double *csrSortedValA,
                          const int *csrSortedColIndA,
                          const int *csrSortedRowPtrA,
                          int nnzA,
                          const int *nnzPerRow,
                          double *csrSortedValC,
                          int *csrSortedColIndC,
                          int *csrSortedRowPtrC,
                          double tol)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsr2csr_compress);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedColIndA,
                     csrSortedRowPtrA,
                     nnzA,
                     nnzPerRow,
                     csrSortedValC,
                     csrSortedColIndC,
                     csrSortedRowPtrC,
                     tol);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2csr_compress(cusparseHandle_t handle,
                          int m,
                          int n,
                          const cusparseMatDescr_t descrA,
                          const cuComplex *csrSortedValA,
                          const int *csrSortedColIndA,
                          const int *csrSortedRowPtrA,
                          int nnzA,
                          const int *nnzPerRow,
                          cuComplex *csrSortedValC,
                          int *csrSortedColIndC,
                          int *csrSortedRowPtrC,
                          cuComplex tol)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsr2csr_compress);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedColIndA,
                     csrSortedRowPtrA,
                     nnzA,
                     nnzPerRow,
                     csrSortedValC,
                     csrSortedColIndC,
                     csrSortedRowPtrC,
                     tol);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2csr_compress(cusparseHandle_t handle,
                          int m,
                          int n,
                          const cusparseMatDescr_t descrA,
                          const cuDoubleComplex *csrSortedValA,
                          const int *csrSortedColIndA,
                          const int *csrSortedRowPtrA,
                          int nnzA,
                          const int *nnzPerRow,
                          cuDoubleComplex *csrSortedValC,
                          int *csrSortedColIndC,
                          int *csrSortedRowPtrC,
                          cuDoubleComplex tol)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsr2csr_compress);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedColIndA,
                     csrSortedRowPtrA,
                     nnzA,
                     nnzPerRow,
                     csrSortedValC,
                     csrSortedColIndC,
                     csrSortedRowPtrC,
                     tol);
}

cusparseStatus_t CUSPARSEAPI
cusparseSdense2csr(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const float *A,
                   int lda,
                   const int *nnzPerRow,
                   float *csrSortedValA,
                   int *csrSortedRowPtrA,
                   int *csrSortedColIndA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSdense2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerRow,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA);
}

cusparseStatus_t CUSPARSEAPI
cusparseDdense2csr(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const double *A,
                   int lda,
                   const int *nnzPerRow,
                   double *csrSortedValA,
                   int *csrSortedRowPtrA,
                   int *csrSortedColIndA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDdense2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerRow,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA);
}

cusparseStatus_t CUSPARSEAPI
cusparseCdense2csr(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex *A,
                   int lda,
                   const int *nnzPerRow,
                   cuComplex *csrSortedValA,
                   int *csrSortedRowPtrA,
                   int *csrSortedColIndA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCdense2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerRow,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA);
}

cusparseStatus_t CUSPARSEAPI
cusparseZdense2csr(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex *A,
                   int lda,
                   const int *nnzPerRow,
                   cuDoubleComplex *csrSortedValA,
                   int *csrSortedRowPtrA,
                   int *csrSortedColIndA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZdense2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerRow,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsr2dense(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const float *csrSortedValA,
                   const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA,
                   float *A,
                   int lda)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsr2dense);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     A,
                     lda);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2dense(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const double *csrSortedValA,
                   const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA,
                   double *A,
                   int lda)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsr2dense);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     A,
                     lda);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2dense(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex *csrSortedValA,
                   const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA,
                   cuComplex *A,
                   int lda)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsr2dense);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     A,
                     lda);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2dense(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex *csrSortedValA,
                   const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA,
                   cuDoubleComplex *A,
                   int lda)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsr2dense);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     A,
                     lda);
}

cusparseStatus_t CUSPARSEAPI
cusparseSdense2csc(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const float *A,
                   int lda,
                   const int *nnzPerCol,
                   float *cscSortedValA,
                   int *cscSortedRowIndA,
                   int *cscSortedColPtrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSdense2csc);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerCol,
                     cscSortedValA,
                     cscSortedRowIndA,
                     cscSortedColPtrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseDdense2csc(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const double *A,
                   int lda,
                   const int *nnzPerCol,
                   double *cscSortedValA,
                   int *cscSortedRowIndA,
                   int *cscSortedColPtrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDdense2csc);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerCol,
                     cscSortedValA,
                     cscSortedRowIndA,
                     cscSortedColPtrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseCdense2csc(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex *A,
                   int lda,
                   const int *nnzPerCol,
                   cuComplex *cscSortedValA,
                   int *cscSortedRowIndA,
                   int *cscSortedColPtrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCdense2csc);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerCol,
                     cscSortedValA,
                     cscSortedRowIndA,
                     cscSortedColPtrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseZdense2csc(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex *A,
                   int lda,
                   const int *nnzPerCol,
                   cuDoubleComplex *cscSortedValA,
                   int *cscSortedRowIndA,
                   int *cscSortedColPtrA)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZdense2csc);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     A,
                     lda,
                     nnzPerCol,
                     cscSortedValA,
                     cscSortedRowIndA,
                     cscSortedColPtrA);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsc2dense(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const float *cscSortedValA,
                   const int *cscSortedRowIndA,
                   const int *cscSortedColPtrA,
                   float *A,
                   int lda)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsc2dense);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     cscSortedValA,
                     cscSortedRowIndA,
                     cscSortedColPtrA,
                     A,
                     lda);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsc2dense(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const double *cscSortedValA,
                   const int *cscSortedRowIndA,
                   const int *cscSortedColPtrA,
                   double *A,
                   int lda)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsc2dense);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     cscSortedValA,
                     cscSortedRowIndA,
                     cscSortedColPtrA,
                     A,
                     lda);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsc2dense(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex *cscSortedValA,
                   const int *cscSortedRowIndA,
                   const int *cscSortedColPtrA,
                   cuComplex *A,
                   int lda)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsc2dense);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     cscSortedValA,
                     cscSortedRowIndA,
                     cscSortedColPtrA,
                     A,
                     lda);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsc2dense(cusparseHandle_t handle,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex *cscSortedValA,
                   const int *cscSortedRowIndA,
                   const int *cscSortedColPtrA,
                   cuDoubleComplex *A,
                   int lda)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsc2dense);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     descrA,
                     cscSortedValA,
                     cscSortedRowIndA,
                     cscSortedColPtrA,
                     A,
                     lda);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcoo2csr(cusparseHandle_t handle,
                 const int *cooRowInd,
                 int nnz,
                 int m,
                 int *csrSortedRowPtr,
                 cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcoo2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     cooRowInd,
                     nnz,
                     m,
                     csrSortedRowPtr,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsr2coo(cusparseHandle_t handle,
                 const int *csrSortedRowPtr,
                 int nnz,
                 int m,
                 int *cooRowInd,
                 cusparseIndexBase_t idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsr2coo);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     cooRowInd,
                     nnz,
                     m,
                     cooRowInd,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsr2bsrNnz(cusparseHandle_t handle,
                    cusparseDirection_t dirA,
                    int m,
                    int n,
                    const cusparseMatDescr_t descrA,
                    const int *csrSortedRowPtrA,
                    const int *csrSortedColIndA,
                    int blockDim,
                    const cusparseMatDescr_t descrC,
                    int *bsrSortedRowPtrC,
                    int *nnzTotalDevHostPtr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsr2bsrNnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     blockDim,
                     descrC,
                     bsrSortedRowPtrC,
                     nnzTotalDevHostPtr);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsr2bsr(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int m,
                 int n,
                 const cusparseMatDescr_t descrA,
                 const float *csrSortedValA,
                 const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA,
                 int blockDim,
                 const cusparseMatDescr_t descrC,
                 float *bsrSortedValC,
                 int *bsrSortedRowPtrC,
                 int *bsrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsr2bsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     blockDim,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2bsr(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int m,
                 int n,
                 const cusparseMatDescr_t descrA,
                 const double *csrSortedValA,
                 const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA,
                 int blockDim,
                 const cusparseMatDescr_t descrC,
                 double *bsrSortedValC,
                 int *bsrSortedRowPtrC,
                 int *bsrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsr2bsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     blockDim,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2bsr(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int m,
                 int n,
                 const cusparseMatDescr_t descrA,
                 const cuComplex *csrSortedValA,
                 const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA,
                 int blockDim,
                 const cusparseMatDescr_t descrC,
                 cuComplex *bsrSortedValC,
                 int *bsrSortedRowPtrC,
                 int *bsrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsr2bsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     blockDim,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2bsr(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int m,
                 int n,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex *csrSortedValA,
                 const int *csrSortedRowPtrA,
                 const int *csrSortedColIndA,
                 int blockDim,
                 const cusparseMatDescr_t descrC,
                 cuDoubleComplex *bsrSortedValC,
                 int *bsrSortedRowPtrC,
                 int *bsrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsr2bsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     blockDim,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseSbsr2csr(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int mb,
                 int nb,
                 const cusparseMatDescr_t descrA,
                 const float *bsrSortedValA,
                 const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA,
                 int blockDim,
                 const cusparseMatDescr_t descrC,
                 float *csrSortedValC,
                 int *csrSortedRowPtrC,
                 int *csrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSbsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseDbsr2csr(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int mb,
                 int nb,
                 const cusparseMatDescr_t descrA,
                 const double *bsrSortedValA,
                 const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA,
                 int blockDim,
                 const cusparseMatDescr_t descrC,
                 double *csrSortedValC,
                 int *csrSortedRowPtrC,
                 int *csrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDbsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseCbsr2csr(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int mb,
                 int nb,
                 const cusparseMatDescr_t descrA,
                 const cuComplex *bsrSortedValA,
                 const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA,
                 int blockDim,
                 const cusparseMatDescr_t descrC,
                 cuComplex *csrSortedValC,
                 int *csrSortedRowPtrC,
                 int *csrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCbsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseZbsr2csr(cusparseHandle_t handle,
                 cusparseDirection_t dirA,
                 int mb,
                 int nb,
                 const cusparseMatDescr_t descrA,
                 const cuDoubleComplex *bsrSortedValA,
                 const int *bsrSortedRowPtrA,
                 const int *bsrSortedColIndA,
                 int blockDim,
                 const cusparseMatDescr_t descrC,
                 cuDoubleComplex *csrSortedValC,
                 int *csrSortedRowPtrC,
                 int *csrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZbsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     blockDim,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int mb,
                                int nb,
                                int nnzb,
                                const float *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd,
                                int rowBlockDim,
                                int colBlockDim,
                                int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgebsr2gebsc_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int mb,
                                int nb,
                                int nnzb,
                                const double *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd,
                                int rowBlockDim,
                                int colBlockDim,
                                int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgebsr2gebsc_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int mb,
                                int nb,
                                int nnzb,
                                const cuComplex *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd,
                                int rowBlockDim,
                                int colBlockDim,
                                int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgebsr2gebsc_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsc_bufferSize(cusparseHandle_t handle,
                                int mb,
                                int nb,
                                int nnzb,
                                const cuDoubleComplex *bsrSortedVal,
                                const int *bsrSortedRowPtr,
                                const int *bsrSortedColInd,
                                int rowBlockDim,
                                int colBlockDim,
                                int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgebsr2gebsc_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int mb,
                                   int nb,
                                   int nnzb,
                                   const float *bsrSortedVal,
                                   const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd,
                                   int rowBlockDim,
                                   int colBlockDim,
                                   size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgebsr2gebsc_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int mb,
                                   int nb,
                                   int nnzb,
                                   const double *bsrSortedVal,
                                   const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd,
                                   int rowBlockDim,
                                   int colBlockDim,
                                   size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgebsr2gebsc_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int mb,
                                   int nb,
                                   int nnzb,
                                   const cuComplex *bsrSortedVal,
                                   const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd,
                                   int rowBlockDim,
                                   int colBlockDim,
                                   size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgebsr2gebsc_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsc_bufferSizeExt(cusparseHandle_t handle,
                                   int mb,
                                   int nb,
                                   int nnzb,
                                   const cuDoubleComplex *bsrSortedVal,
                                   const int *bsrSortedRowPtr,
                                   const int *bsrSortedColInd,
                                   int rowBlockDim,
                                   int colBlockDim,
                                   size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgebsr2gebsc_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsc(cusparseHandle_t handle,
                     int mb,
                     int nb,
                     int nnzb,
                     const float *bsrSortedVal,
                     const int *bsrSortedRowPtr,
                     const int *bsrSortedColInd,
                     int rowBlockDim,
                     int colBlockDim,
                     float *bscVal,
                     int *bscRowInd,
                     int *bscColPtr,
                     cusparseAction_t copyValues,
                     cusparseIndexBase_t idxBase,
                     void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgebsr2gebsc);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     bscVal,
                     bscRowInd,
                     bscColPtr,
                     copyValues,
                     idxBase,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsc(cusparseHandle_t handle,
                     int mb,
                     int nb,
                     int nnzb,
                     const double *bsrSortedVal,
                     const int *bsrSortedRowPtr,
                     const int *bsrSortedColInd,
                     int rowBlockDim,
                     int colBlockDim,
                     double *bscVal,
                     int *bscRowInd,
                     int *bscColPtr,
                     cusparseAction_t copyValues,
                     cusparseIndexBase_t idxBase,
                     void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgebsr2gebsc);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     bscVal,
                     bscRowInd,
                     bscColPtr,
                     copyValues,
                     idxBase,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsc(cusparseHandle_t handle,
                     int mb,
                     int nb,
                     int nnzb,
                     const cuComplex *bsrSortedVal,
                     const int *bsrSortedRowPtr,
                     const int *bsrSortedColInd,
                     int rowBlockDim,
                     int colBlockDim,
                     cuComplex *bscVal,
                     int *bscRowInd,
                     int *bscColPtr,
                     cusparseAction_t copyValues,
                     cusparseIndexBase_t idxBase,
                     void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgebsr2gebsc);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     bscVal,
                     bscRowInd,
                     bscColPtr,
                     copyValues,
                     idxBase,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsc(cusparseHandle_t handle,
                     int mb,
                     int nb,
                     int nnzb,
                     const cuDoubleComplex *bsrSortedVal,
                     const int *bsrSortedRowPtr,
                     const int *bsrSortedColInd,
                     int rowBlockDim,
                     int colBlockDim,
                     cuDoubleComplex *bscVal,
                     int *bscRowInd,
                     int *bscColPtr,
                     cusparseAction_t copyValues,
                     cusparseIndexBase_t idxBase,
                     void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgebsr2gebsc);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     mb,
                     nb,
                     nnzb,
                     bsrSortedVal,
                     bsrSortedRowPtr,
                     bsrSortedColInd,
                     rowBlockDim,
                     colBlockDim,
                     bscVal,
                     bscRowInd,
                     bscColPtr,
                     copyValues,
                     idxBase,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseXgebsr2csr(cusparseHandle_t handle,
                   cusparseDirection_t dirA,
                   int mb,
                   int nb,
                   const cusparseMatDescr_t descrA,
                   const int *bsrSortedRowPtrA,
                   const int *bsrSortedColIndA,
                   int rowBlockDim,
                   int colBlockDim,
                   const cusparseMatDescr_t descrC,
                   int *csrSortedRowPtrC,
                   int *csrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXgebsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     descrA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     descrC,
                     csrSortedRowPtrC,
                     csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2csr(cusparseHandle_t handle,
                   cusparseDirection_t dirA,
                   int mb,
                   int nb,
                   const cusparseMatDescr_t descrA,
                   const float *bsrSortedValA,
                   const int *bsrSortedRowPtrA,
                   const int *bsrSortedColIndA,
                   int rowBlockDim,
                   int colBlockDim,
                   const cusparseMatDescr_t descrC,
                   float *csrSortedValC,
                   int *csrSortedRowPtrC,
                   int *csrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgebsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2csr(cusparseHandle_t handle,
                   cusparseDirection_t dirA,
                   int mb,
                   int nb,
                   const cusparseMatDescr_t descrA,
                   const double *bsrSortedValA,
                   const int *bsrSortedRowPtrA,
                   const int *bsrSortedColIndA,
                   int rowBlockDim,
                   int colBlockDim,
                   const cusparseMatDescr_t descrC,
                   double *csrSortedValC,
                   int *csrSortedRowPtrC,
                   int *csrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgebsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2csr(cusparseHandle_t handle,
                   cusparseDirection_t dirA,
                   int mb,
                   int nb,
                   const cusparseMatDescr_t descrA,
                   const cuComplex *bsrSortedValA,
                   const int *bsrSortedRowPtrA,
                   const int *bsrSortedColIndA,
                   int rowBlockDim,
                   int colBlockDim,
                   const cusparseMatDescr_t descrC,
                   cuComplex *csrSortedValC,
                   int *csrSortedRowPtrC,
                   int *csrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgebsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2csr(cusparseHandle_t handle,
                   cusparseDirection_t dirA,
                   int mb,
                   int nb,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex *bsrSortedValA,
                   const int *bsrSortedRowPtrA,
                   const int *bsrSortedColIndA,
                   int rowBlockDim,
                   int colBlockDim,
                   const cusparseMatDescr_t descrC,
                   cuDoubleComplex *csrSortedValC,
                   int *csrSortedRowPtrC,
                   int *csrSortedColIndC)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgebsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsr2gebsr_bufferSize(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              int m,
                              int n,
                              const cusparseMatDescr_t descrA,
                              const float *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              int rowBlockDim,
                              int colBlockDim,
                              int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsr2gebsr_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2gebsr_bufferSize(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              int m,
                              int n,
                              const cusparseMatDescr_t descrA,
                              const double *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              int rowBlockDim,
                              int colBlockDim,
                              int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsr2gebsr_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2gebsr_bufferSize(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              int m,
                              int n,
                              const cusparseMatDescr_t descrA,
                              const cuComplex *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              int rowBlockDim,
                              int colBlockDim,
                              int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsr2gebsr_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2gebsr_bufferSize(cusparseHandle_t handle,
                              cusparseDirection_t dirA,
                              int m,
                              int n,
                              const cusparseMatDescr_t descrA,
                              const cuDoubleComplex *csrSortedValA,
                              const int *csrSortedRowPtrA,
                              const int *csrSortedColIndA,
                              int rowBlockDim,
                              int colBlockDim,
                              int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsr2gebsr_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                 cusparseDirection_t dirA,
                                 int m,
                                 int n,
                                 const cusparseMatDescr_t descrA,
                                 const float *csrSortedValA,
                                 const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA,
                                 int rowBlockDim,
                                 int colBlockDim,
                                 size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsr2gebsr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                 cusparseDirection_t dirA,
                                 int m,
                                 int n,
                                 const cusparseMatDescr_t descrA,
                                 const double *csrSortedValA,
                                 const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA,
                                 int rowBlockDim,
                                 int colBlockDim,
                                 size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsr2gebsr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                 cusparseDirection_t dirA,
                                 int m,
                                 int n,
                                 const cusparseMatDescr_t descrA,
                                 const cuComplex *csrSortedValA,
                                 const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA,
                                 int rowBlockDim,
                                 int colBlockDim,
                                 size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsr2gebsr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                 cusparseDirection_t dirA,
                                 int m,
                                 int n,
                                 const cusparseMatDescr_t descrA,
                                 const cuDoubleComplex *csrSortedValA,
                                 const int *csrSortedRowPtrA,
                                 const int *csrSortedColIndA,
                                 int rowBlockDim,
                                 int colBlockDim,
                                 size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsr2gebsr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     rowBlockDim,
                     colBlockDim,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsr2gebsrNnz(cusparseHandle_t handle,
                      cusparseDirection_t dirA,
                      int m,
                      int n,
                      const cusparseMatDescr_t descrA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      const cusparseMatDescr_t descrC,
                      int *bsrSortedRowPtrC,
                      int rowBlockDim,
                      int colBlockDim,
                      int *nnzTotalDevHostPtr,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsr2gebsrNnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrC,
                     bsrSortedRowPtrC,
                     rowBlockDim,
                     colBlockDim,
                     nnzTotalDevHostPtr,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsr2gebsr(cusparseHandle_t handle,
                   cusparseDirection_t dirA,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const float *csrSortedValA,
                   const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   float *bsrSortedValC,
                   int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC,
                   int rowBlockDim,
                   int colBlockDim,
                   void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsr2gebsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC,
                     rowBlockDim,
                     colBlockDim,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2gebsr(cusparseHandle_t handle,
                   cusparseDirection_t dirA,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const double *csrSortedValA,
                   const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   double *bsrSortedValC,
                   int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC,
                   int rowBlockDim,
                   int colBlockDim,
                   void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsr2gebsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC,
                     rowBlockDim,
                     colBlockDim,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2gebsr(cusparseHandle_t handle,
                   cusparseDirection_t dirA,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuComplex *csrSortedValA,
                   const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   cuComplex *bsrSortedValC,
                   int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC,
                   int rowBlockDim,
                   int colBlockDim,
                   void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsr2gebsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC,
                     rowBlockDim,
                     colBlockDim,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2gebsr(cusparseHandle_t handle,
                   cusparseDirection_t dirA,
                   int m,
                   int n,
                   const cusparseMatDescr_t descrA,
                   const cuDoubleComplex *csrSortedValA,
                   const int *csrSortedRowPtrA,
                   const int *csrSortedColIndA,
                   const cusparseMatDescr_t descrC,
                   cuDoubleComplex *bsrSortedValC,
                   int *bsrSortedRowPtrC,
                   int *bsrSortedColIndC,
                   int rowBlockDim,
                   int colBlockDim,
                   void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsr2gebsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     m,
                     n,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC,
                     rowBlockDim,
                     colBlockDim,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsr_bufferSize(cusparseHandle_t handle,
                                cusparseDirection_t dirA,
                                int mb,
                                int nb,
                                int nnzb,
                                const cusparseMatDescr_t descrA,
                                const float *bsrSortedValA,
                                const int *bsrSortedRowPtrA,
                                const int *bsrSortedColIndA,
                                int rowBlockDimA,
                                int colBlockDimA,
                                int rowBlockDimC,
                                int colBlockDimC,
                                int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgebsr2gebsr_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     rowBlockDimC,
                     colBlockDimC,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsr_bufferSize(cusparseHandle_t handle,
                                cusparseDirection_t dirA,
                                int mb,
                                int nb,
                                int nnzb,
                                const cusparseMatDescr_t descrA,
                                const double *bsrSortedValA,
                                const int *bsrSortedRowPtrA,
                                const int *bsrSortedColIndA,
                                int rowBlockDimA,
                                int colBlockDimA,
                                int rowBlockDimC,
                                int colBlockDimC,
                                int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgebsr2gebsr_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     rowBlockDimC,
                     colBlockDimC,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsr_bufferSize(cusparseHandle_t handle,
                                cusparseDirection_t dirA,
                                int mb,
                                int nb,
                                int nnzb,
                                const cusparseMatDescr_t descrA,
                                const cuComplex *bsrSortedValA,
                                const int *bsrSortedRowPtrA,
                                const int *bsrSortedColIndA,
                                int rowBlockDimA,
                                int colBlockDimA,
                                int rowBlockDimC,
                                int colBlockDimC,
                                int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgebsr2gebsr_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     rowBlockDimC,
                     colBlockDimC,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsr_bufferSize(cusparseHandle_t handle,
                                cusparseDirection_t dirA,
                                int mb,
                                int nb,
                                int nnzb,
                                const cusparseMatDescr_t descrA,
                                const cuDoubleComplex *bsrSortedValA,
                                const int *bsrSortedRowPtrA,
                                const int *bsrSortedColIndA,
                                int rowBlockDimA,
                                int colBlockDimA,
                                int rowBlockDimC,
                                int colBlockDimC,
                                int *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgebsr2gebsr_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     rowBlockDimC,
                     colBlockDimC,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                   cusparseDirection_t dirA,
                                   int mb,
                                   int nb,
                                   int nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const float *bsrSortedValA,
                                   const int *bsrSortedRowPtrA,
                                   const int *bsrSortedColIndA,
                                   int rowBlockDimA,
                                   int colBlockDimA,
                                   int rowBlockDimC,
                                   int colBlockDimC,
                                   size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgebsr2gebsr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     rowBlockDimC,
                     colBlockDimC,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                   cusparseDirection_t dirA,
                                   int mb,
                                   int nb,
                                   int nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const double *bsrSortedValA,
                                   const int *bsrSortedRowPtrA,
                                   const int *bsrSortedColIndA,
                                   int rowBlockDimA,
                                   int colBlockDimA,
                                   int rowBlockDimC,
                                   int colBlockDimC,
                                   size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgebsr2gebsr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     rowBlockDimC,
                     colBlockDimC,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                   cusparseDirection_t dirA,
                                   int mb,
                                   int nb,
                                   int nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const cuComplex *bsrSortedValA,
                                   const int *bsrSortedRowPtrA,
                                   const int *bsrSortedColIndA,
                                   int rowBlockDimA,
                                   int colBlockDimA,
                                   int rowBlockDimC,
                                   int colBlockDimC,
                                   size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgebsr2gebsr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     rowBlockDimC,
                     colBlockDimC,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsr_bufferSizeExt(cusparseHandle_t handle,
                                   cusparseDirection_t dirA,
                                   int mb,
                                   int nb,
                                   int nnzb,
                                   const cusparseMatDescr_t descrA,
                                   const cuDoubleComplex *bsrSortedValA,
                                   const int *bsrSortedRowPtrA,
                                   const int *bsrSortedColIndA,
                                   int rowBlockDimA,
                                   int colBlockDimA,
                                   int rowBlockDimC,
                                   int colBlockDimC,
                                   size_t *pBufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgebsr2gebsr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     rowBlockDimC,
                     colBlockDimC,
                     pBufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseXgebsr2gebsrNnz(cusparseHandle_t handle,
                        cusparseDirection_t dirA,
                        int mb,
                        int nb,
                        int nnzb,
                        const cusparseMatDescr_t descrA,
                        const int *bsrSortedRowPtrA,
                        const int *bsrSortedColIndA,
                        int rowBlockDimA,
                        int colBlockDimA,
                        const cusparseMatDescr_t descrC,
                        int *bsrSortedRowPtrC,
                        int rowBlockDimC,
                        int colBlockDimC,
                        int *nnzTotalDevHostPtr,
                        void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXgebsr2gebsrNnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     descrC,
                     bsrSortedRowPtrC,
                     rowBlockDimC,
                     colBlockDimC,
                     nnzTotalDevHostPtr,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSgebsr2gebsr(cusparseHandle_t handle,
                     cusparseDirection_t dirA,
                     int mb,
                     int nb,
                     int nnzb,
                     const cusparseMatDescr_t descrA,
                     const float *bsrSortedValA,
                     const int *bsrSortedRowPtrA,
                     const int *bsrSortedColIndA,
                     int rowBlockDimA,
                     int colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     float *bsrSortedValC,
                     int *bsrSortedRowPtrC,
                     int *bsrSortedColIndC,
                     int rowBlockDimC,
                     int colBlockDimC,
                     void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSgebsr2gebsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC,
                     rowBlockDimC,
                     colBlockDimC,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDgebsr2gebsr(cusparseHandle_t handle,
                     cusparseDirection_t dirA,
                     int mb,
                     int nb,
                     int nnzb,
                     const cusparseMatDescr_t descrA,
                     const double *bsrSortedValA,
                     const int *bsrSortedRowPtrA,
                     const int *bsrSortedColIndA,
                     int rowBlockDimA,
                     int colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     double *bsrSortedValC,
                     int *bsrSortedRowPtrC,
                     int *bsrSortedColIndC,
                     int rowBlockDimC,
                     int colBlockDimC,
                     void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDgebsr2gebsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC,
                     rowBlockDimC,
                     colBlockDimC,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCgebsr2gebsr(cusparseHandle_t handle,
                     cusparseDirection_t dirA,
                     int mb,
                     int nb,
                     int nnzb,
                     const cusparseMatDescr_t descrA,
                     const cuComplex *bsrSortedValA,
                     const int *bsrSortedRowPtrA,
                     const int *bsrSortedColIndA,
                     int rowBlockDimA,
                     int colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     cuComplex *bsrSortedValC,
                     int *bsrSortedRowPtrC,
                     int *bsrSortedColIndC,
                     int rowBlockDimC,
                     int colBlockDimC,
                     void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCgebsr2gebsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC,
                     rowBlockDimC,
                     colBlockDimC,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZgebsr2gebsr(cusparseHandle_t handle,
                     cusparseDirection_t dirA,
                     int mb,
                     int nb,
                     int nnzb,
                     const cusparseMatDescr_t descrA,
                     const cuDoubleComplex *bsrSortedValA,
                     const int *bsrSortedRowPtrA,
                     const int *bsrSortedColIndA,
                     int rowBlockDimA,
                     int colBlockDimA,
                     const cusparseMatDescr_t descrC,
                     cuDoubleComplex *bsrSortedValC,
                     int *bsrSortedRowPtrC,
                     int *bsrSortedColIndC,
                     int rowBlockDimC,
                     int colBlockDimC,
                     void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZgebsr2gebsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     dirA,
                     mb,
                     nb,
                     nnzb,
                     descrA,
                     bsrSortedValA,
                     bsrSortedRowPtrA,
                     bsrSortedColIndA,
                     rowBlockDimA,
                     colBlockDimA,
                     descrC,
                     bsrSortedValC,
                     bsrSortedRowPtrC,
                     bsrSortedColIndC,
                     rowBlockDimC,
                     colBlockDimC,
                     pBuffer);
}

//##############################################################################
//# SPARSE MATRIX SORTING
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateIdentityPermutation(cusparseHandle_t handle,
                                  int n,
                                  int *p)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateIdentityPermutation);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     n,
                     p);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle,
                               int m,
                               int n,
                               int nnz,
                               const int *cooRowsA,
                               const int *cooColsA,
                               size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcoosort_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     cooRowsA,
                     cooColsA,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcoosortByRow(cusparseHandle_t handle,
                      int m,
                      int n,
                      int nnz,
                      int *cooRowsA,
                      int *cooColsA,
                      int *P,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcoosortByRow);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     cooRowsA,
                     cooColsA,
                     P,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcoosortByColumn(cusparseHandle_t handle,
                         int m,
                         int n,
                         int nnz,
                         int *cooRowsA,
                         int *cooColsA,
                         int *P,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcoosortByColumn);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     cooRowsA,
                     cooColsA,
                     P,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle,
                               int m,
                               int n,
                               int nnz,
                               const int *csrRowPtrA,
                               const int *csrColIndA,
                               size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsrsort_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     csrRowPtrA,
                     csrColIndA,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcsrsort(cusparseHandle_t handle,
                 int m,
                 int n,
                 int nnz,
                 const cusparseMatDescr_t descrA,
                 const int *csrRowPtrA,
                 int *csrColIndA,
                 int *P,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcsrsort);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     csrRowPtrA,
                     csrColIndA,
                     P,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle,
                               int m,
                               int n,
                               int nnz,
                               const int *cscColPtrA,
                               const int *cscRowIndA,
                               size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcscsort_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     cscColPtrA,
                     cscRowIndA,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseXcscsort(cusparseHandle_t handle,
                 int m,
                 int n,
                 int nnz,
                 const cusparseMatDescr_t descrA,
                 const int *cscColPtrA,
                 int *cscRowIndA,
                 int *P,
                 void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseXcscsort);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     cscColPtrA,
                     cscRowIndA,
                     P,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                int nnz,
                                float *csrVal,
                                const int *csrRowPtr,
                                int *csrColInd,
                                csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsru2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                int nnz,
                                double *csrVal,
                                const int *csrRowPtr,
                                int *csrColInd,
                                csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsru2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                int nnz,
                                cuComplex *csrVal,
                                const int *csrRowPtr,
                                int *csrColInd,
                                csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsru2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsru2csr_bufferSizeExt(cusparseHandle_t handle,
                                int m,
                                int n,
                                int nnz,
                                cuDoubleComplex *csrVal,
                                const int *csrRowPtr,
                                int *csrColInd,
                                csru2csrInfo_t info,
                                size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsru2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsru2csr(cusparseHandle_t handle,
                  int m,
                  int n,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  float *csrVal,
                  const int *csrRowPtr,
                  int *csrColInd,
                  csru2csrInfo_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsru2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsru2csr(cusparseHandle_t handle,
                  int m,
                  int n,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  double *csrVal,
                  const int *csrRowPtr,
                  int *csrColInd,
                  csru2csrInfo_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsru2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsru2csr(cusparseHandle_t handle,
                  int m,
                  int n,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  cuComplex *csrVal,
                  const int *csrRowPtr,
                  int *csrColInd,
                  csru2csrInfo_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsru2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsru2csr(cusparseHandle_t handle,
                  int m,
                  int n,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex *csrVal,
                  const int *csrRowPtr,
                  int *csrColInd,
                  csru2csrInfo_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsru2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseScsr2csru(cusparseHandle_t handle,
                  int m,
                  int n,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  float *csrVal,
                  const int *csrRowPtr,
                  int *csrColInd,
                  csru2csrInfo_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScsr2csru);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDcsr2csru(cusparseHandle_t handle,
                  int m,
                  int n,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  double *csrVal,
                  const int *csrRowPtr,
                  int *csrColInd,
                  csru2csrInfo_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDcsr2csru);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCcsr2csru(cusparseHandle_t handle,
                  int m,
                  int n,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  cuComplex *csrVal,
                  const int *csrRowPtr,
                  int *csrColInd,
                  csru2csrInfo_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCcsr2csru);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseZcsr2csru(cusparseHandle_t handle,
                  int m,
                  int n,
                  int nnz,
                  const cusparseMatDescr_t descrA,
                  cuDoubleComplex *csrVal,
                  const int *csrRowPtr,
                  int *csrColInd,
                  csru2csrInfo_t info,
                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseZcsr2csru);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     descrA,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     info,
                     pBuffer);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csr_bufferSizeExt(cusparseHandle_t handle,
                                      int m,
                                      int n,
                                      const __half *A,
                                      int lda,
                                      const __half *threshold,
                                      const cusparseMatDescr_t descrC,
                                      const __half *csrSortedValC,
                                      const int *csrSortedRowPtrC,
                                      const int *csrSortedColIndC,
                                      size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneDense2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBufferSizeInBytes);
}
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csr_bufferSizeExt(cusparseHandle_t handle,
                                      int m,
                                      int n,
                                      const float *A,
                                      int lda,
                                      const float *threshold,
                                      const cusparseMatDescr_t descrC,
                                      const float *csrSortedValC,
                                      const int *csrSortedRowPtrC,
                                      const int *csrSortedColIndC,
                                      size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneDense2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csr_bufferSizeExt(cusparseHandle_t handle,
                                      int m,
                                      int n,
                                      const double *A,
                                      int lda,
                                      const double *threshold,
                                      const cusparseMatDescr_t descrC,
                                      const double *csrSortedValC,
                                      const int *csrSortedRowPtrC,
                                      const int *csrSortedColIndC,
                                      size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneDense2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBufferSizeInBytes);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrNnz(cusparseHandle_t handle,
                           int m,
                           int n,
                           const __half *A,
                           int lda,
                           const __half *threshold,
                           const cusparseMatDescr_t descrC,
                           int *csrRowPtrC,
                           int *nnzTotalDevHostPtr,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneDense2csrNnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     threshold,
                     descrC,
                     csrRowPtrC,
                     nnzTotalDevHostPtr,
                     pBuffer);
}
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrNnz(cusparseHandle_t handle,
                           int m,
                           int n,
                           const float *A,
                           int lda,
                           const float *threshold,
                           const cusparseMatDescr_t descrC,
                           int *csrRowPtrC,
                           int *nnzTotalDevHostPtr,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneDense2csrNnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     threshold,
                     descrC,
                     csrRowPtrC,
                     nnzTotalDevHostPtr,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrNnz(cusparseHandle_t handle,
                           int m,
                           int n,
                           const double *A,
                           int lda,
                           const double *threshold,
                           const cusparseMatDescr_t descrC,
                           int *csrSortedRowPtrC,
                           int *nnzTotalDevHostPtr,
                           void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneDense2csrNnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     threshold,
                     descrC,
                     csrSortedRowPtrC,
                     nnzTotalDevHostPtr,
                     pBuffer);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csr(cusparseHandle_t handle,
                        int m,
                        int n,
                        const __half *A,
                        int lda,
                        const __half *threshold,
                        const cusparseMatDescr_t descrC,
                        __half *csrSortedValC,
                        const int *csrSortedRowPtrC,
                        int *csrSortedColIndC,
                        void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneDense2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csr(cusparseHandle_t handle,
                        int m,
                        int n,
                        const float *A,
                        int lda,
                        const float *threshold,
                        const cusparseMatDescr_t descrC,
                        float *csrSortedValC,
                        const int *csrSortedRowPtrC,
                        int *csrSortedColIndC,
                        void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneDense2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csr(cusparseHandle_t handle,
                        int m,
                        int n,
                        const double *A,
                        int lda,
                        const double *threshold,
                        const cusparseMatDescr_t descrC,
                        double *csrSortedValC,
                        const int *csrSortedRowPtrC,
                        int *csrSortedColIndC,
                        void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneDense2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csr_bufferSizeExt(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const __half *csrSortedValA,
                                    const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA,
                                    const __half *threshold,
                                    const cusparseMatDescr_t descrC,
                                    const __half *csrSortedValC,
                                    const int *csrSortedRowPtrC,
                                    const int *csrSortedColIndC,
                                    size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneCsr2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBufferSizeInBytes);
}
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csr_bufferSizeExt(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const float *csrSortedValA,
                                    const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA,
                                    const float *threshold,
                                    const cusparseMatDescr_t descrC,
                                    const float *csrSortedValC,
                                    const int *csrSortedRowPtrC,
                                    const int *csrSortedColIndC,
                                    size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneCsr2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csr_bufferSizeExt(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnzA,
                                    const cusparseMatDescr_t descrA,
                                    const double *csrSortedValA,
                                    const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA,
                                    const double *threshold,
                                    const cusparseMatDescr_t descrC,
                                    const double *csrSortedValC,
                                    const int *csrSortedRowPtrC,
                                    const int *csrSortedColIndC,
                                    size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneCsr2csr_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBufferSizeInBytes);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrNnz(cusparseHandle_t handle,
                         int m,
                         int n,
                         int nnzA,
                         const cusparseMatDescr_t descrA,
                         const __half *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         const __half *threshold,
                         const cusparseMatDescr_t descrC,
                         int *csrSortedRowPtrC,
                         int *nnzTotalDevHostPtr,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneCsr2csrNnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     threshold,
                     descrC,
                     csrSortedRowPtrC,
                     nnzTotalDevHostPtr,
                     pBuffer);
}
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrNnz(cusparseHandle_t handle,
                         int m,
                         int n,
                         int nnzA,
                         const cusparseMatDescr_t descrA,
                         const float *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         const float *threshold,
                         const cusparseMatDescr_t descrC,
                         int *csrSortedRowPtrC,
                         int *nnzTotalDevHostPtr,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneCsr2csrNnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     threshold,
                     descrC,
                     csrSortedRowPtrC,
                     nnzTotalDevHostPtr,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrNnz(cusparseHandle_t handle,
                         int m,
                         int n,
                         int nnzA,
                         const cusparseMatDescr_t descrA,
                         const double *csrSortedValA,
                         const int *csrSortedRowPtrA,
                         const int *csrSortedColIndA,
                         const double *threshold,
                         const cusparseMatDescr_t descrC,
                         int *csrSortedRowPtrC,
                         int *nnzTotalDevHostPtr,
                         void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneCsr2csrNnz);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     threshold,
                     descrC,
                     csrSortedRowPtrC,
                     nnzTotalDevHostPtr,
                     pBuffer);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csr(cusparseHandle_t handle,
                      int m,
                      int n,
                      int nnzA,
                      const cusparseMatDescr_t descrA,
                      const __half *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      const __half *threshold,
                      const cusparseMatDescr_t descrC,
                      __half *csrSortedValC,
                      const int *csrSortedRowPtrC,
                      int *csrSortedColIndC,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneCsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csr(cusparseHandle_t handle,
                      int m,
                      int n,
                      int nnzA,
                      const cusparseMatDescr_t descrA,
                      const float *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      const float *threshold,
                      const cusparseMatDescr_t descrC,
                      float *csrSortedValC,
                      const int *csrSortedRowPtrC,
                      int *csrSortedColIndC,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneCsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csr(cusparseHandle_t handle,
                      int m,
                      int n,
                      int nnzA,
                      const cusparseMatDescr_t descrA,
                      const double *csrSortedValA,
                      const int *csrSortedRowPtrA,
                      const int *csrSortedColIndA,
                      const double *threshold,
                      const cusparseMatDescr_t descrC,
                      double *csrSortedValC,
                      const int *csrSortedRowPtrC,
                      int *csrSortedColIndC,
                      void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneCsr2csr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     threshold,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     pBuffer);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    float percentage,
    const cusparseMatDescr_t descrC,
    const __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneDense2csrByPercentage_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBufferSizeInBytes);
}
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    float percentage,
    const cusparseMatDescr_t descrC,
    const float *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneDense2csrByPercentage_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    float percentage,
    const cusparseMatDescr_t descrC,
    const double *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneDense2csrByPercentage_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBufferSizeInBytes);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    const __half *A,
    int lda,
    float percentage,
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    pruneInfo_t info,
    void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneDense2csrNnzByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     percentage,
                     descrC,
                     csrRowPtrC,
                     nnzTotalDevHostPtr,
                     info,
                     pBuffer);
}
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *A,
    int lda,
    float percentage,
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    pruneInfo_t info,
    void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneDense2csrNnzByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     percentage,
                     descrC,
                     csrRowPtrC,
                     nnzTotalDevHostPtr,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *A,
    int lda,
    float percentage,
    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    pruneInfo_t info,
    void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneDense2csrNnzByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     percentage,
                     descrC,
                     csrRowPtrC,
                     nnzTotalDevHostPtr,
                     info,
                     pBuffer);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneDense2csrByPercentage(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const __half *A,
                                    int lda,
                                    float percentage,
                                    const cusparseMatDescr_t descrC,
                                    __half *csrSortedValC,
                                    const int *csrSortedRowPtrC,
                                    int *csrSortedColIndC,
                                    pruneInfo_t info,
                                    void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneDense2csrByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}
#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneDense2csrByPercentage(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const float *A,
                                    int lda,
                                    float percentage,
                                    const cusparseMatDescr_t descrC,
                                    float *csrSortedValC,
                                    const int *csrSortedRowPtrC,
                                    int *csrSortedColIndC,
                                    pruneInfo_t info,
                                    void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneDense2csrByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneDense2csrByPercentage(cusparseHandle_t handle,
                                    int m,
                                    int n,
                                    const double *A,
                                    int lda,
                                    float percentage,
                                    const cusparseMatDescr_t descrC,
                                    double *csrSortedValC,
                                    const int *csrSortedRowPtrC,
                                    int *csrSortedColIndC,
                                    pruneInfo_t info,
                                    void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneDense2csrByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     A,
                     lda,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}

#if defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const __half *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage,
    const cusparseMatDescr_t descrC,
    const __half *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneCsr2csrByPercentage_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBufferSizeInBytes);
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage,
    const cusparseMatDescr_t descrC,
    const float *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneCsr2csrByPercentage_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBufferSizeInBytes);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrByPercentage_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage,
    const cusparseMatDescr_t descrC,
    const double *csrSortedValC,
    const int *csrSortedRowPtrC,
    const int *csrSortedColIndC,
    pruneInfo_t info,
    size_t *pBufferSizeInBytes)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneCsr2csrByPercentage_bufferSizeExt);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBufferSizeInBytes);
}

#if defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const __half *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage,
    const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr,
    pruneInfo_t info,
    void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneCsr2csrNnzByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     percentage,
                     descrC,
                     csrSortedRowPtrC,
                     nnzTotalDevHostPtr,
                     info,
                     pBuffer);
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const float *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage,
    const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr,
    pruneInfo_t info,
    void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneCsr2csrNnzByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     percentage,
                     descrC,
                     csrSortedRowPtrC,
                     nnzTotalDevHostPtr,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrNnzByPercentage(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const double *csrSortedValA,
    const int *csrSortedRowPtrA,
    const int *csrSortedColIndA,
    float percentage,
    const cusparseMatDescr_t descrC,
    int *csrSortedRowPtrC,
    int *nnzTotalDevHostPtr,
    pruneInfo_t info,
    void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneCsr2csrNnzByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     percentage,
                     descrC,
                     csrSortedRowPtrC,
                     nnzTotalDevHostPtr,
                     info,
                     pBuffer);
}

#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI
cusparseHpruneCsr2csrByPercentage(cusparseHandle_t handle,
                                  int m,
                                  int n,
                                  int nnzA,
                                  const cusparseMatDescr_t descrA,
                                  const __half *csrSortedValA,
                                  const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA,
                                  float percentage, /* between 0 to 100 */
                                  const cusparseMatDescr_t descrC,
                                  __half *csrSortedValC,
                                  const int *csrSortedRowPtrC,
                                  int *csrSortedColIndC,
                                  pruneInfo_t info,
                                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseHpruneCsr2csrByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}

#endif // defined(__cplusplus)

cusparseStatus_t CUSPARSEAPI
cusparseSpruneCsr2csrByPercentage(cusparseHandle_t handle,
                                  int m,
                                  int n,
                                  int nnzA,
                                  const cusparseMatDescr_t descrA,
                                  const float *csrSortedValA,
                                  const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA,
                                  float percentage,
                                  const cusparseMatDescr_t descrC,
                                  float *csrSortedValC,
                                  const int *csrSortedRowPtrC,
                                  int *csrSortedColIndC,
                                  pruneInfo_t info,
                                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpruneCsr2csrByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseDpruneCsr2csrByPercentage(cusparseHandle_t handle,
                                  int m,
                                  int n,
                                  int nnzA,
                                  const cusparseMatDescr_t descrA,
                                  const double *csrSortedValA,
                                  const int *csrSortedRowPtrA,
                                  const int *csrSortedColIndA,
                                  float percentage,
                                  const cusparseMatDescr_t descrC,
                                  double *csrSortedValC,
                                  const int *csrSortedRowPtrC,
                                  int *csrSortedColIndC,
                                  pruneInfo_t info,
                                  void *pBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDpruneCsr2csrByPercentage);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnzA,
                     descrA,
                     csrSortedValA,
                     csrSortedRowPtrA,
                     csrSortedColIndA,
                     percentage,
                     descrC,
                     csrSortedValC,
                     csrSortedRowPtrC,
                     csrSortedColIndC,
                     info,
                     pBuffer);
}

//##############################################################################
//# CSR2CSC
//##############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCsr2cscEx2(cusparseHandle_t handle,
                   int m,
                   int n,
                   int nnz,
                   const void *csrVal,
                   const int *csrRowPtr,
                   const int *csrColInd,
                   void *cscVal,
                   int *cscColPtr,
                   int *cscRowInd,
                   cudaDataType valType,
                   cusparseAction_t copyValues,
                   cusparseIndexBase_t idxBase,
                   cusparseCsr2CscAlg_t alg,
                   void *buffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCsr2cscEx2);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     cscVal,
                     cscColPtr,
                     cscRowInd,
                     valType,
                     copyValues,
                     idxBase,
                     alg,
                     buffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle,
                              int m,
                              int n,
                              int nnz,
                              const void *csrVal,
                              const int *csrRowPtr,
                              const int *csrColInd,
                              void *cscVal,
                              int *cscColPtr,
                              int *cscRowInd,
                              cudaDataType valType,
                              cusparseAction_t copyValues,
                              cusparseIndexBase_t idxBase,
                              cusparseCsr2CscAlg_t alg,
                              size_t *bufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCsr2cscEx2_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     m,
                     n,
                     nnz,
                     csrVal,
                     csrRowPtr,
                     csrColInd,
                     cscVal,
                     cscColPtr,
                     cscRowInd,
                     valType,
                     copyValues,
                     idxBase,
                     alg,
                     bufferSize);
}

// #############################################################################
// # SPARSE VECTOR DESCRIPTOR
// #############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateSpVec(cusparseSpVecDescr_t *spVecDescr,
                    int64_t size,
                    int64_t nnz,
                    void *indices,
                    void *values,
                    cusparseIndexType_t idxType,
                    cusparseIndexBase_t idxBase,
                    cudaDataType valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateSpVec);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spVecDescr,
                     size,
                     nnz,
                     indices,
                     values,
                     idxType,
                     idxBase,
                     valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroySpVec);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spVecDescr);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr,
                 int64_t *size,
                 int64_t *nnz,
                 void **indices,
                 void **values,
                 cusparseIndexType_t *idxType,
                 cusparseIndexBase_t *idxBase,
                 cudaDataType *valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpVecGet);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spVecDescr,
                     size,
                     nnz,
                     indices,
                     values,
                     idxType,
                     idxBase,
                     valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr,
                          cusparseIndexBase_t *idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpVecGetIndexBase);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spVecDescr,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr,
                       void **values)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpVecGetValues);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spVecDescr,
                     values);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr,
                       void *values)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpVecSetValues);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spVecDescr,
                     values);
}

// #############################################################################
// # DENSE VECTOR DESCRIPTOR
// #############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateDnVec(cusparseDnVecDescr_t *dnVecDescr,
                    int64_t size,
                    void *values,
                    cudaDataType valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateDnVec);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnVecDescr,
                     size,
                     values,
                     valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyDnVec);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnVecDescr);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr,
                 int64_t *size,
                 void **values,
                 cudaDataType *valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnVecGet);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnVecDescr,
                     size,
                     values,
                     valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr,
                       void **values)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnVecGetValues);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnVecDescr,
                     values);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr,
                       void *values)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnVecSetValues);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnVecDescr,
                     values);
}

// #############################################################################
// # SPARSE MATRIX DESCRIPTOR
// #############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroySpMat);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetFormat(cusparseSpMatDescr_t spMatDescr,
                       cusparseFormat_t *format)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMatGetFormat);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     format);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetIndexBase(cusparseSpMatDescr_t spMatDescr,
                          cusparseIndexBase_t *idxBase)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMatGetIndexBase);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     idxBase);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr,
                       void **values)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMatGetValues);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     values);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr,
                       void *values)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMatSetValues);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     values);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr,
                     int64_t *rows,
                     int64_t *cols,
                     int64_t *nnz)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMatGetSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     rows,
                     cols,
                     nnz);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                             int batchCount)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMatSetStridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     batchCount);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMatGetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                             int *batchCount)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMatGetStridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     batchCount);
}

cusparseStatus_t CUSPARSEAPI
cusparseCooSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                           int batchCount,
                           int64_t batchStride)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCooSetStridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     batchCount,
                     batchStride);
}

cusparseStatus_t CUSPARSEAPI
cusparseCsrSetStridedBatch(cusparseSpMatDescr_t spMatDescr,
                           int batchCount,
                           int64_t offsetsBatchStride,
                           int64_t columnsValuesBatchStride)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCsrSetStridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     batchCount,
                     offsetsBatchStride,
                     columnsValuesBatchStride);
}

//------------------------------------------------------------------------------
// ### CSR ###

cusparseStatus_t CUSPARSEAPI
cusparseCreateCsr(cusparseSpMatDescr_t *spMatDescr,
                  int64_t rows,
                  int64_t cols,
                  int64_t nnz,
                  void *csrRowOffsets,
                  void *csrColInd,
                  void *csrValues,
                  cusparseIndexType_t csrRowOffsetsType,
                  cusparseIndexType_t csrColIndType,
                  cusparseIndexBase_t idxBase,
                  cudaDataType valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateCsr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     rows,
                     cols,
                     nnz,
                     csrRowOffsets,
                     csrColInd,
                     csrValues,
                     csrRowOffsetsType,
                     csrColIndType,
                     idxBase,
                     valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseCsrGet(cusparseSpMatDescr_t spMatDescr,
               int64_t *rows,
               int64_t *cols,
               int64_t *nnz,
               void **csrRowOffsets,
               void **csrColInd,
               void **csrValues,
               cusparseIndexType_t *csrRowOffsetsType,
               cusparseIndexType_t *csrColIndType,
               cusparseIndexBase_t *idxBase,
               cudaDataType *valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCsrGet);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     rows,
                     cols,
                     nnz,
                     csrRowOffsets,
                     csrColInd,
                     csrValues,
                     csrRowOffsetsType,
                     csrColIndType,
                     idxBase,
                     valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr,
                       void *csrRowOffsets,
                       void *csrColInd,
                       void *csrValues)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCsrSetPointers);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     csrRowOffsets,
                     csrColInd,
                     csrValues);
}

//------------------------------------------------------------------------------
// ### COO ###

cusparseStatus_t CUSPARSEAPI
cusparseCreateCoo(cusparseSpMatDescr_t *spMatDescr,
                  int64_t rows,
                  int64_t cols,
                  int64_t nnz,
                  void *cooRowInd,
                  void *cooColInd,
                  void *cooValues,
                  cusparseIndexType_t cooIdxType,
                  cusparseIndexBase_t idxBase,
                  cudaDataType valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateCoo);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     rows,
                     cols,
                     nnz,
                     cooRowInd,
                     cooColInd,
                     cooValues,
                     cooIdxType,
                     idxBase,
                     valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseCreateCooAoS(cusparseSpMatDescr_t *spMatDescr,
                     int64_t rows,
                     int64_t cols,
                     int64_t nnz,
                     void *cooInd,
                     void *cooValues,
                     cusparseIndexType_t cooIdxType,
                     cusparseIndexBase_t idxBase,
                     cudaDataType valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateCooAoS);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     rows,
                     cols,
                     nnz,
                     cooInd,
                     cooValues,
                     cooIdxType,
                     idxBase,
                     valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseCooGet(cusparseSpMatDescr_t spMatDescr,
               int64_t *rows,
               int64_t *cols,
               int64_t *nnz,
               void **cooRowInd, // COO row indices
               void **cooColInd, // COO column indices
               void **cooValues, // COO values
               cusparseIndexType_t *idxType,
               cusparseIndexBase_t *idxBase,
               cudaDataType *valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCooGet);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     rows,
                     cols,
                     nnz,
                     cooRowInd,
                     cooColInd,
                     cooValues,
                     idxType,
                     idxBase,
                     valueType);
}

cusparseStatus_t CUSPARSEAPI
cusparseCooAoSGet(cusparseSpMatDescr_t spMatDescr,
                  int64_t *rows,
                  int64_t *cols,
                  int64_t *nnz,
                  void **cooInd,    // COO indices
                  void **cooValues, // COO values
                  cusparseIndexType_t *idxType,
                  cusparseIndexBase_t *idxBase,
                  cudaDataType *valueType)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCooAoSGet);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(spMatDescr,
                     rows,
                     cols,
                     nnz,
                     cooInd,
                     cooValues,
                     idxType,
                     idxBase,
                     valueType);
}

// #############################################################################
// # DENSE MATRIX DESCRIPTOR
// #############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseCreateDnMat(cusparseDnMatDescr_t *dnMatDescr,
                    int64_t rows,
                    int64_t cols,
                    int64_t ld,
                    void *values,
                    cudaDataType valueType,
                    cusparseOrder_t order)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseCreateDnMat);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnMatDescr,
                     rows,
                     cols,
                     ld,
                     values,
                     valueType,
                     order);
}

cusparseStatus_t CUSPARSEAPI
cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDestroyDnMat);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnMatDescr);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr,
                 int64_t *rows,
                 int64_t *cols,
                 int64_t *ld,
                 void **values,
                 cudaDataType *type,
                 cusparseOrder_t *order)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnMatGet);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnMatDescr,
                     rows,
                     cols,
                     ld,
                     values,
                     type,
                     order);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr,
                       void **values)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnMatGetValues);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnMatDescr,
                     values);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr,
                       void *values)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnMatSetValues);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnMatDescr,
                     values);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr,
                             int batchCount,
                             int64_t batchStride)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnMatSetStridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnMatDescr,
                     batchCount,
                     batchStride);
}

cusparseStatus_t CUSPARSEAPI
cusparseDnMatGetStridedBatch(cusparseDnMatDescr_t dnMatDescr,
                             int *batchCount,
                             int64_t *batchStride)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseDnMatGetStridedBatch);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(dnMatDescr,
                     batchCount,
                     batchStride);
}

// #############################################################################
// # VECTOR-VECTOR OPERATIONS
// #############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseAxpby(cusparseHandle_t handle,
              const void *alpha,
              cusparseSpVecDescr_t vecX,
              const void *beta,
              cusparseDnVecDescr_t vecY)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseAxpby);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     alpha,
                     vecX,
                     beta,
                     vecY);
}

cusparseStatus_t CUSPARSEAPI
cusparseGather(cusparseHandle_t handle,
               cusparseDnVecDescr_t vecY,
               cusparseSpVecDescr_t vecX)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseGather);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     vecY,
                     vecX);
}

cusparseStatus_t CUSPARSEAPI
cusparseScatter(cusparseHandle_t handle,
                cusparseSpVecDescr_t vecX,
                cusparseDnVecDescr_t vecY)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseScatter);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     vecX,
                     vecY);
}

cusparseStatus_t CUSPARSEAPI
cusparseRot(cusparseHandle_t handle,
            const void *c_coeff,
            const void *s_coeff,
            cusparseSpVecDescr_t vecX,
            cusparseDnVecDescr_t vecY)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseRot);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     c_coeff,
                     s_coeff,
                     vecX,
                     vecY);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpVV_bufferSize(cusparseHandle_t handle,
                        cusparseOperation_t opX,
                        cusparseSpVecDescr_t vecX,
                        cusparseDnVecDescr_t vecY,
                        const void *result,
                        cudaDataType computeType,
                        size_t *bufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpVV_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opX,
                     vecX,
                     vecY,
                     result,
                     computeType,
                     bufferSize);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpVV(cusparseHandle_t handle,
             cusparseOperation_t opX,
             cusparseSpVecDescr_t vecX,
             cusparseDnVecDescr_t vecY,
             void *result,
             cudaDataType computeType,
             void *externalBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpVV);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opX,
                     vecX,
                     vecY,
                     result,
                     computeType,
                     externalBuffer);
}

// #############################################################################
// # SPARSE MATRIX-VECTOR MULTIPLICATION
// #############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSpMV(cusparseHandle_t handle,
             cusparseOperation_t opA,
             const void *alpha,
             cusparseSpMatDescr_t matA,
             cusparseDnVecDescr_t vecX,
             const void *beta,
             cusparseDnVecDescr_t vecY,
             cudaDataType computeType,
             cusparseSpMVAlg_t alg,
             void *externalBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMV);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opA,
                     alpha,
                     matA,
                     vecX,
                     beta,
                     vecY,
                     computeType,
                     alg,
                     externalBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpMV_bufferSize(cusparseHandle_t handle,
                        cusparseOperation_t opA,
                        const void *alpha,
                        cusparseSpMatDescr_t matA,
                        cusparseDnVecDescr_t vecX,
                        const void *beta,
                        cusparseDnVecDescr_t vecY,
                        cudaDataType computeType,
                        cusparseSpMVAlg_t alg,
                        size_t *bufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMV_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opA,
                     alpha,
                     matA,
                     vecX,
                     beta,
                     vecY,
                     computeType,
                     alg,
                     bufferSize);
}

// #############################################################################
// # SPARSE MATRIX-MATRIX MULTIPLICATION
// #############################################################################
#ifndef DGSPARSE
cusparseStatus_t CUSPARSEAPI
cusparseSpMM(cusparseHandle_t handle,
             cusparseOperation_t opA,
             cusparseOperation_t opB,
             const void *alpha,
             cusparseSpMatDescr_t matA,
             cusparseDnMatDescr_t matB,
             const void *beta,
             cusparseDnMatDescr_t matC,
             cudaDataType computeType,
             cusparseSpMMAlg_t alg,
             void *externalBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMM);
    // LOG(INFO, "Enter %s()", __FUNCTION__);
    return _real_sym(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer);
}
#endif

#ifdef DGSPARSE
cusparseStatus_t CUSPARSEAPI
cusparseSpMM(cusparseHandle_t handle,
             cusparseOperation_t opA,
             cusparseOperation_t opB,
             const void *alpha,
             cusparseSpMatDescr_t matA,
             cusparseDnMatDescr_t matB,
             const void *beta,
             cusparseDnMatDescr_t matC,
             cudaDataType computeType,
             cusparseSpMMAlg_t alg,
             void *externalBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(DGSPARSE_LIB, spmm_cuda);
    // LOG(INFO, "Enter %s() our dgsparse", __FUNCTION__);
    int64_t rows;
    int64_t cols;
    int64_t Brows;
    int64_t Bcols;
    int64_t ld;
    int64_t nnz;
    void *csrRowOffsets = nullptr;
    void *csrColInd = nullptr;
    void *csrValues = nullptr;
    void *matBValues = nullptr;
    void *matCValues = nullptr;
    cusparseIndexType_t csrRowOffsetsType;
    cusparseIndexType_t csrColIndType;
    cusparseIndexBase_t idxBase;
    cudaDataType valueType;
    cudaDataType BType;
    cusparseOrder_t order;
    cusparseCsrGet(matA, &rows, &cols, &nnz, &csrRowOffsets, &csrColInd,
                   &csrValues, &csrRowOffsetsType, &csrColIndType, &idxBase, &valueType);
    cusparseDnMatGet(matB,
                     &Brows,
                     &Bcols,
                     &ld,
                     &matBValues,
                     &BType,
                     &order);
    cusparseDnMatGetValues(matC, &matCValues);
    _real_sym((int)rows, (int)Bcols, reinterpret_cast<int *>(csrRowOffsets), reinterpret_cast<int *>(csrColInd),
              reinterpret_cast<float *>(csrValues), reinterpret_cast<float *>(matBValues), reinterpret_cast<float *>(matCValues));
    return CUSPARSE_STATUS_SUCCESS;
}
#endif

cusparseStatus_t CUSPARSEAPI
cusparseSpMM_bufferSize(cusparseHandle_t handle,
                        cusparseOperation_t opA,
                        cusparseOperation_t opB,
                        const void *alpha,
                        cusparseSpMatDescr_t matA,
                        cusparseDnMatDescr_t matB,
                        const void *beta,
                        cusparseDnMatDescr_t matC,
                        cudaDataType computeType,
                        cusparseSpMMAlg_t alg,
                        size_t *bufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpMM_bufferSize);
    // LOG(INFO, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opA,
                     opB,
                     alpha,
                     matA,
                     matB,
                     beta,
                     matC,
                     computeType,
                     alg,
                     bufferSize);
    // return CUSPARSE_STATUS_SUCCESS;
}

// #############################################################################
// # SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM)
// #############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t *descr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpGEMM_createDescr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descr);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpGEMM_destroyDescr);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(descr);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpGEMM_workEstimation(cusparseHandle_t handle,
                              cusparseOperation_t opA,
                              cusparseOperation_t opB,
                              const void *alpha,
                              cusparseSpMatDescr_t matA,
                              cusparseSpMatDescr_t matB,
                              const void *beta,
                              cusparseSpMatDescr_t matC,
                              cudaDataType computeType,
                              cusparseSpGEMMAlg_t alg,
                              cusparseSpGEMMDescr_t spgemmDescr,
                              size_t *bufferSize1,
                              void *externalBuffer1)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpGEMM_workEstimation);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opA,
                     opB,
                     alpha,
                     matA,
                     matB,
                     beta,
                     matC,
                     computeType,
                     alg,
                     spgemmDescr,
                     bufferSize1,
                     externalBuffer1);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpGEMM_compute(cusparseHandle_t handle,
                       cusparseOperation_t opA,
                       cusparseOperation_t opB,
                       const void *alpha,
                       cusparseSpMatDescr_t matA,
                       cusparseSpMatDescr_t matB,
                       const void *beta,
                       cusparseSpMatDescr_t matC,
                       cudaDataType computeType,
                       cusparseSpGEMMAlg_t alg,
                       cusparseSpGEMMDescr_t spgemmDescr,
                       size_t *bufferSize2,
                       void *externalBuffer2)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpGEMM_compute);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opA,
                     opB,
                     alpha,
                     matA,
                     matB,
                     beta,
                     matC,
                     computeType,
                     alg,
                     spgemmDescr,
                     bufferSize2,
                     externalBuffer2);
}

cusparseStatus_t CUSPARSEAPI
cusparseSpGEMM_copy(cusparseHandle_t handle,
                    cusparseOperation_t opA,
                    cusparseOperation_t opB,
                    const void *alpha,
                    cusparseSpMatDescr_t matA,
                    cusparseSpMatDescr_t matB,
                    const void *beta,
                    cusparseSpMatDescr_t matC,
                    cudaDataType computeType,
                    cusparseSpGEMMAlg_t alg,
                    cusparseSpGEMMDescr_t spgemmDescr)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseSpGEMM_copy);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opA,
                     opB,
                     alpha,
                     matA,
                     matB,
                     beta,
                     matC,
                     computeType,
                     alg,
                     spgemmDescr);
}

// #############################################################################
// # GENERAL MATRIX-MATRIX PATTERN-CONSTRAINED MULTIPLICATION
// #############################################################################

cusparseStatus_t CUSPARSEAPI
cusparseConstrainedGeMM(cusparseHandle_t handle,
                        cusparseOperation_t opA,
                        cusparseOperation_t opB,
                        const void *alpha,
                        cusparseDnMatDescr_t matA,
                        cusparseDnMatDescr_t matB,
                        const void *beta,
                        cusparseSpMatDescr_t matC,
                        cudaDataType computeType,
                        void *externalBuffer)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseConstrainedGeMM);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opA,
                     opB,
                     alpha,
                     matA,
                     matB,
                     beta,
                     matC,
                     computeType,
                     externalBuffer);
}

cusparseStatus_t CUSPARSEAPI
cusparseConstrainedGeMM_bufferSize(cusparseHandle_t handle,
                                   cusparseOperation_t opA,
                                   cusparseOperation_t opB,
                                   const void *alpha,
                                   cusparseDnMatDescr_t matA,
                                   cusparseDnMatDescr_t matB,
                                   const void *beta,
                                   cusparseSpMatDescr_t matC,
                                   cudaDataType computeType,
                                   size_t *bufferSize)
{
    LOAD_SPARSE_SYMBOL_FOR_ONCE(CUSPARSE_LIB, cusparseConstrainedGeMM_bufferSize);
    LOG(TRACE, "Enter %s()", __FUNCTION__);
    return _real_sym(handle,
                     opA,
                     opB,
                     alpha,
                     matA,
                     matB,
                     beta,
                     matC,
                     computeType,
                     bufferSize);
}