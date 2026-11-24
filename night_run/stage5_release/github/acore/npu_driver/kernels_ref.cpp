// kernels_ref.cpp - "ref" backend: optimised C++ Bellman backup, bit-exact with the
// chip's Q16.16 semantics (signed 64-bit product, floor shift by 16, truncation to
// 32 bits, 32-bit wrapping add, signed max over actions).
#ifndef NPU_BUILD_DLL
#define NPU_BUILD_DLL
#endif
#include "npu_kernels.h"

namespace {

// Floor shift written without relying on implementation-defined behaviour of >>
// on negative values: for p < 0, floor(p / 2^16) == ~((~p) >> 16).
inline int32_t mulShift16(int32_t c, int32_t v) {
    const int64_t p = static_cast<int64_t>(c) * static_cast<int64_t>(v);
    const int64_t s = (p >= 0) ? (p >> 16) : ~((~p) >> 16);
    return static_cast<int32_t>(static_cast<uint32_t>(static_cast<uint64_t>(s)));
}

inline int32_t wrapAdd(int32_t a, int32_t b) {
    return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
}

npuError_t refBellman(const npu::BellmanArgs& g, uint64_t* cycles) {
    const int n = g.n;
    const int A = g.A;
    const int32_t* __restrict R = g.R;
    const int32_t* __restrict C = g.C;
    const int32_t* __restrict D = g.D;
    const int32_t* __restrict Vin = g.Vin;
    int32_t* __restrict Vout = g.Vout;

    // Single pass, no allocation: one row of A actions per state.
    for (int i = 0; i < n; ++i) {
        const int base = i * A;
        int32_t best = wrapAdd(R[base], mulShift16(C[base], Vin[D[base]]));
        for (int a = 1; a < A; ++a) {
            const int k = base + a;
            const int32_t q = wrapAdd(R[k], mulShift16(C[k], Vin[D[k]]));
            if (q > best) best = q;
        }
        Vout[i] = best;
    }

    // Analytical single-backup cycle model of Chip v2.0 (CFG, SWEEP, HALT): n + 9.
    if (cycles) *cycles = (n <= 64 && A <= 8) ? static_cast<uint64_t>(n) + 9u : 0u;
    return npuSuccess;
}

const npu::Backend kRef = {"ref", 0, 0, refBellman};

}  // namespace

namespace npu {
const Backend* refBackend() { return &kRef; }
}  // namespace npu
