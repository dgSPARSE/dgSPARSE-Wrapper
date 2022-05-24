#ifndef _SYMBOL_HELPER_CUSPARSE_H_
#define _SYMBOL_HELPER_CUSPARSE_H_

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <stdlib.h>

#include "common.h"
#include "logger.h"

#define DIE_IF(condition, args...)                                             \
  do {                                                                         \
    if (condition) {                                                           \
      printf(args);                                                            \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace sparse_wrapper {

enum SparseLibraryEnum {
  CUSPARSE_LIB_11_0,
  CUSPARSE_LIB_11_1,
  CUSPARSE_LIB_11_2,
  CUSPARSE_LIB_11_3,
  DGSPARSE_LIB, // Deep Graph Sparse Library
  SPARSE_LIBRARY_COUNT
};

constexpr const char *sparse_libray_path[SPARSE_LIBRARY_COUNT] = {
    "/usr/local/cuda-11.0/lib64/libcusparse.so.real", // this should be
                                                      // the path to
                                                      // the real
                                                      // cuSPARSE
                                                      // library
    "/usr/local/cuda-11.1/lib64/libcusparse.so.real", // this should be
                                                      // the path to
                                                      // the real
                                                      // cuSPARSE
                                                      // library
    "/usr/local/cuda-11.2/lib64/libcusparse.so.real", // this should be
                                                      // the path to
                                                      // the real
                                                      // cuSPARSE
                                                      // library
    "/usr/local/cuda-11.3/lib64/libcusparse.so.real", // this should be
                                                      // the path to
                                                      // the real
                                                      // cuSPARSE
                                                      // library
    "/usr/local/dgsparse/lib64/dgsparse.so" // this should be the path to the
                                            // dgSPARSE library
};

extern void *__sparse_handle__[SPARSE_LIBRARY_COUNT];

#define GENERATE_SPARSE_LIBRARY_LOADER(LIB_TYPE)                               \
  inline void *LoadSparseLibrary_##LIB_TYPE() {                                \
    static void *_handle =                                                     \
        dlopen(sparse_libray_path[LIB_TYPE], RTLD_LAZY | RTLD_GLOBAL);         \
    DIE_IF(!_handle, "Failed to load library %s, error message: %s",           \
           sparse_libray_path[LIB_TYPE], dlerror());                           \
    if (!__sparse_handle__[LIB_TYPE]) {                                        \
      __sparse_handle__[LIB_TYPE] = _handle;                                   \
    }                                                                          \
    return _handle;                                                            \
  }

#define LOAD_SPARSE_SYMBOL_FOR_ONCE(LIB_TYPE, SYMBOL)                          \
  static auto _real_sym = reinterpret_cast<decltype(&SYMBOL)>(                 \
      dlsym(runtime::LoadSparseLibrary_##LIB_TYPE(), #SYMBOL));

namespace runtime {

GENERATE_SPARSE_LIBRARY_LOADER(CUSPARSE_LIB_11_0);
GENERATE_SPARSE_LIBRARY_LOADER(CUSPARSE_LIB_11_1);
GENERATE_SPARSE_LIBRARY_LOADER(CUSPARSE_LIB_11_2);
GENERATE_SPARSE_LIBRARY_LOADER(CUSPARSE_LIB_11_3);
GENERATE_SPARSE_LIBRARY_LOADER(DGSPARSE_LIB);

inline void LibraryCleanup() {
  for (int i = 0; i < SPARSE_LIBRARY_COUNT; ++i) {
    if (__sparse_handle__[i]) {
      LOG(INFO, "dlclose on library type %d", i);
      dlclose(__sparse_handle__[i]);
    }
  }
  exit(EXIT_SUCCESS);
}

} // namespace runtime

} // namespace sparse_wrapper

#endif // _SYMBOL_HELPER_H_