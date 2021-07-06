#ifndef _RUNTIME_INFO_H_
#define _RUNTIME_INFO_H_

#include <new>

namespace sparse_wrapper {
namespace runtime {

// singleton
class GlobalExecutionContext {
   public:
    // write global context here

    bool lightgbm_initialized;

    GlobalExecutionContext() {
    }
    ~GlobalExecutionContext() {
    }
    GlobalExecutionContext(const GlobalExecutionContext& copyme) = delete;
    GlobalExecutionContext& operator=(const GlobalExecutionContext& rhs) = delete;
};

struct GlobalExecutionContextInterface {
    class GlobalExecutionContext* pimpl = nullptr;
    GlobalExecutionContextInterface() {
        pimpl = new (std::nothrow) GlobalExecutionContext{};
    }
    ~GlobalExecutionContextInterface() {
    }
};

inline GlobalExecutionContext* GetGlobalContext() {
    static GlobalExecutionContextInterface global_context_interface{};
    return global_context_interface.pimpl;
}

}  // namespace runtime
}  // namespace sparse_wrapper

#define GLOBAL_CONTEXT() GetGlobalContext()

#endif