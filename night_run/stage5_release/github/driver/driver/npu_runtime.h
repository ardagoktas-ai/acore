/*
 * npu_runtime.h - public C API of my own CUDA-style runtime for my NPU (Driver v2.0).
 *
 * The runtime exposes devices (one per backend), device memory, host<->device copies
 * and one kernel: a Bellman backup over Q16.16 buffers with exactly the chip contract
 *
 *   Vout[i] = max_a ( R[i*A+a] + ((CONT[i*A+a] * Vin[DST[i*A+a]]) >>> 16) )
 *
 * where the product is a signed 64-bit product, >>> is an arithmetic (floor) shift,
 * the shifted product is truncated to 32 bits, and the sum wraps in 32 bits.
 *
 * Execution is synchronous: every launch has finished when the call returns.
 * npuSynchronize() exists for API compatibility with stream-based code.
 * The runtime is not thread-safe: one host thread should drive it at a time.
 */
#ifndef NPU_RUNTIME_H
#define NPU_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(NPU_BUILD_DLL)
#    define NPU_API __declspec(dllexport)
#  else
#    define NPU_API __declspec(dllimport)
#  endif
#else
#  define NPU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum npuError {
    npuSuccess                   = 0,
    npuErrorNotInitialized       = 1,
    npuErrorInvalidValue         = 2,
    npuErrorMemoryAllocation     = 3,
    npuErrorInvalidDevicePointer = 4,
    npuErrorInvalidDevice        = 5,
    npuErrorUnsupportedSize      = 6,
    npuErrorBackend              = 7
} npuError_t;

/* Initialise the runtime (idempotent). flags is reserved and must be 0. */
NPU_API npuError_t  npuInit(unsigned int flags);

/* Number of devices; device 0 is always the "ref" backend. */
NPU_API npuError_t  npuGetDeviceCount(int* count);
NPU_API npuError_t  npuSetDevice(int device);
NPU_API npuError_t  npuGetDevice(int* device);
/* Backend name of a device ("ref", "sim"), or NULL for an invalid index. */
NPU_API const char* npuGetDeviceName(int device);

/* Device memory. Pointers are opaque handles; never dereference them on the host. */
NPU_API npuError_t  npuMalloc(void** devPtr, size_t bytes);
NPU_API npuError_t  npuFree(void* devPtr);
NPU_API npuError_t  npuMemcpyHtoD(void* dstDev, const void* srcHost, size_t bytes);
NPU_API npuError_t  npuMemcpyDtoH(void* dstHost, const void* srcDev, size_t bytes);

/*
 * One Bellman backup on the current device.
 *   dR, dCont : int32 Q16.16, n*A elements, row-major (index i*A+a)
 *   dDst      : int32 state indices, n*A elements, each in [0, n)
 *   dVin      : int32 Q16.16, n elements
 *   dVout     : int32 Q16.16, n elements (may alias dVin; Jacobi semantics are kept)
 */
NPU_API npuError_t  npuLaunchBellman(const void* dR, const void* dCont, const void* dDst,
                                     const void* dVin, void* dVout, int n, int A);

NPU_API npuError_t  npuSynchronize(void);

/*
 * Chip cycles of the last launch. "sim" reports cycles counted on the RTL;
 * "ref" reports the analytical single-backup model of the chip (n + 9, see ISA.md)
 * when n <= 64 and A <= 8, and 0 for sizes the chip cannot hold.
 */
NPU_API npuError_t  npuGetLastCycles(uint64_t* cycles);

/* Last error recorded by any call (not reset by successful calls). */
NPU_API npuError_t  npuGetLastError(void);
NPU_API const char* npuGetErrorString(npuError_t err);

#ifdef __cplusplus
}
#endif

#endif /* NPU_RUNTIME_H */
