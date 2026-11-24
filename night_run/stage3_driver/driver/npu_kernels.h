// npu_kernels.h - internal interface between the runtime and the kernel library.
#ifndef NPU_KERNELS_H
#define NPU_KERNELS_H

#include "npu_runtime.h"

namespace npu {

struct BellmanArgs {
    const int32_t* R;      // n*A, Q16.16
    const int32_t* C;      // n*A, Q16.16
    const int32_t* D;      // n*A, validated to [0, n) by the runtime
    const int32_t* Vin;    // n, Q16.16, never aliases Vout (runtime copies if needed)
    int32_t*       Vout;   // n, Q16.16
    int n;
    int A;
};

struct Backend {
    const char* name;
    int maxN;              // 0 = unlimited
    int maxA;              // 0 = unlimited
    npuError_t (*bellman)(const BellmanArgs& args, uint64_t* cycles);
};

const Backend* refBackend();
#ifdef NPU_HAVE_SIM
const Backend* simBackend();
#endif

}  // namespace npu

#endif  // NPU_KERNELS_H
