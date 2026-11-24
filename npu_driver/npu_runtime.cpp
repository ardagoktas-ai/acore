// npu_runtime.cpp - device registry, device memory, copies and kernel dispatch.
#ifndef NPU_BUILD_DLL
#define NPU_BUILD_DLL
#endif
#include "npu_runtime.h"
#include "npu_kernels.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace {

bool g_init = false;
std::vector<const npu::Backend*> g_devices;
int g_device = 0;
std::map<uintptr_t, size_t> g_allocs;   // base address -> size in bytes
npuError_t g_lastError = npuSuccess;
uint64_t g_lastCycles = 0;

npuError_t fail(npuError_t e) {
    g_lastError = e;
    return e;
}

void ensureInit() {
    if (g_init) return;
    g_devices.clear();
    g_devices.push_back(npu::refBackend());
#ifdef NPU_HAVE_SIM
    g_devices.push_back(npu::simBackend());
#endif
    g_device = 0;
    g_init = true;
}

// True when [p, p+bytes) lies inside one live allocation.
bool inAllocation(const void* p, size_t bytes) {
    if (!p) return false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    std::map<uintptr_t, size_t>::const_iterator it = g_allocs.upper_bound(a);
    if (it == g_allocs.begin()) return false;
    --it;
    const uintptr_t base = it->first;
    const size_t size = it->second;
    if (a < base) return false;
    const size_t off = static_cast<size_t>(a - base);
    return off <= size && bytes <= size - off;
}

bool overlaps(const void* a, size_t na, const void* b, size_t nb) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a);
    const uintptr_t y = reinterpret_cast<uintptr_t>(b);
    return x < y + nb && y < x + na;
}

}  // namespace

extern "C" {

npuError_t npuInit(unsigned int flags) {
    if (flags != 0) return fail(npuErrorInvalidValue);
    ensureInit();
    return npuSuccess;
}

npuError_t npuGetDeviceCount(int* count) {
    if (!count) return fail(npuErrorInvalidValue);
    ensureInit();
    *count = static_cast<int>(g_devices.size());
    return npuSuccess;
}

npuError_t npuSetDevice(int device) {
    ensureInit();
    if (device < 0 || device >= static_cast<int>(g_devices.size()))
        return fail(npuErrorInvalidDevice);
    g_device = device;
    return npuSuccess;
}

npuError_t npuGetDevice(int* device) {
    if (!device) return fail(npuErrorInvalidValue);
    ensureInit();
    *device = g_device;
    return npuSuccess;
}

const char* npuGetDeviceName(int device) {
    ensureInit();
    if (device < 0 || device >= static_cast<int>(g_devices.size())) return 0;
    return g_devices[device]->name;
}

npuError_t npuMalloc(void** devPtr, size_t bytes) {
    if (!devPtr || bytes == 0) return fail(npuErrorInvalidValue);
    ensureInit();
    void* p = std::malloc(bytes);
    if (!p) return fail(npuErrorMemoryAllocation);
    g_allocs[reinterpret_cast<uintptr_t>(p)] = bytes;
    *devPtr = p;
    return npuSuccess;
}

npuError_t npuFree(void* devPtr) {
    if (!devPtr) return npuSuccess;
    std::map<uintptr_t, size_t>::iterator it = g_allocs.find(reinterpret_cast<uintptr_t>(devPtr));
    if (it == g_allocs.end()) return fail(npuErrorInvalidDevicePointer);
    g_allocs.erase(it);
    std::free(devPtr);
    return npuSuccess;
}

npuError_t npuMemcpyHtoD(void* dstDev, const void* srcHost, size_t bytes) {
    if (bytes == 0) return npuSuccess;
    if (!srcHost) return fail(npuErrorInvalidValue);
    if (!inAllocation(dstDev, bytes)) return fail(npuErrorInvalidDevicePointer);
    std::memcpy(dstDev, srcHost, bytes);
    return npuSuccess;
}

npuError_t npuMemcpyDtoH(void* dstHost, const void* srcDev, size_t bytes) {
    if (bytes == 0) return npuSuccess;
    if (!dstHost) return fail(npuErrorInvalidValue);
    if (!inAllocation(srcDev, bytes)) return fail(npuErrorInvalidDevicePointer);
    std::memcpy(dstHost, srcDev, bytes);
    return npuSuccess;
}

npuError_t npuLaunchBellman(const void* dR, const void* dCont, const void* dDst,
                            const void* dVin, void* dVout, int n, int A) {
    ensureInit();
    if (n < 1 || A < 1) return fail(npuErrorInvalidValue);
    const size_t nA = static_cast<size_t>(n) * static_cast<size_t>(A);
    if (nA > static_cast<size_t>(0x7fffffff) / sizeof(int32_t)) return fail(npuErrorUnsupportedSize);
    const size_t matBytes = nA * sizeof(int32_t);
    const size_t vecBytes = static_cast<size_t>(n) * sizeof(int32_t);

    if (!inAllocation(dR, matBytes) || !inAllocation(dCont, matBytes) ||
        !inAllocation(dDst, matBytes) || !inAllocation(dVin, vecBytes) ||
        !inAllocation(dVout, vecBytes))
        return fail(npuErrorInvalidDevicePointer);

    const npu::Backend* b = g_devices[g_device];
    if ((b->maxN && n > b->maxN) || (b->maxA && A > b->maxA))
        return fail(npuErrorUnsupportedSize);

    const int32_t* D = static_cast<const int32_t*>(dDst);
    for (size_t k = 0; k < nA; ++k)
        if (D[k] < 0 || D[k] >= n) return fail(npuErrorInvalidValue);

    // Keep Jacobi semantics when the output overlaps the input vector.
    std::vector<int32_t> vinCopy;
    const int32_t* vin = static_cast<const int32_t*>(dVin);
    if (overlaps(dVin, vecBytes, dVout, vecBytes)) {
        vinCopy.assign(vin, vin + n);
        vin = &vinCopy[0];
    }

    npu::BellmanArgs args;
    args.R = static_cast<const int32_t*>(dR);
    args.C = static_cast<const int32_t*>(dCont);
    args.D = D;
    args.Vin = vin;
    args.Vout = static_cast<int32_t*>(dVout);
    args.n = n;
    args.A = A;

    uint64_t cyc = 0;
    const npuError_t e = b->bellman(args, &cyc);
    if (e != npuSuccess) return fail(e);
    g_lastCycles = cyc;
    return npuSuccess;
}

npuError_t npuSynchronize(void) {
    ensureInit();
    return npuSuccess;   // all launches are synchronous
}

npuError_t npuGetLastCycles(uint64_t* cycles) {
    if (!cycles) return fail(npuErrorInvalidValue);
    *cycles = g_lastCycles;
    return npuSuccess;
}

npuError_t npuGetLastError(void) { return g_lastError; }

const char* npuGetErrorString(npuError_t err) {
    switch (err) {
    case npuSuccess:                   return "no error";
    case npuErrorNotInitialized:       return "runtime not initialized";
    case npuErrorInvalidValue:         return "invalid value (bad size, null pointer or DST out of range)";
    case npuErrorMemoryAllocation:     return "device memory allocation failed";
    case npuErrorInvalidDevicePointer: return "invalid device pointer or copy out of bounds";
    case npuErrorInvalidDevice:        return "invalid device index";
    case npuErrorUnsupportedSize:      return "problem size not supported by this backend";
    case npuErrorBackend:              return "backend failure";
    default:                           return "unknown error";
    }
}

}  // extern "C"
