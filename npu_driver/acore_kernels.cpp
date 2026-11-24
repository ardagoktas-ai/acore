// acore_kernels.cpp - float64 kernels for Acore's planner. Every arithmetic step
// mirrors the numpy code it replaces (same operations, same order, no fused
// multiply-add), so the agent's decisions stay bit-identical.
#ifndef NPU_BUILD_DLL
#define NPU_BUILD_DLL
#endif
#include "npu_runtime.h"

#include <cmath>
#include <cstring>
#include <vector>

extern "C" {

/*
 * Jacobi value iteration, matching Acore's numpy loop:
 *   V_new[i] = max_a q(i, a); stop when max_i |V_new[i] - V[i]| < tol.
 * Without INFO:  q = R + CONT * V[DST]
 * With INFO:     q = INFO != 0 ? (R + INFO) + (CHAIN ? CONT * V[DST] : CONT * Vbase[DST])
 *                              : R + CONT * V[DST]
 * V holds the warm start on entry and the result on exit.
 * Returns the number of sweeps performed.
 */
NPU_API int acoreValueIteration(const double* R, const double* C, const int64_t* D,
                                const double* INFO, const unsigned char* CHAIN,
                                const double* Vbase, int n, int A, double* V,
                                int maxIter, double tol) {
    if (n < 1 || A < 1) return 0;
    const size_t nA = static_cast<size_t>(n) * static_cast<size_t>(A);
    std::vector<double> bc;
    if (INFO) {
        bc.resize(nA);
        for (size_t k = 0; k < nA; ++k) {
            const double prod = C[k] * Vbase[D[k]];
            bc[k] = prod;
        }
    }
    std::vector<double> cur(V, V + n), nxt(static_cast<size_t>(n));
    for (int it = 0; it < maxIter; ++it) {
        double delta = 0.0;
        for (int i = 0; i < n; ++i) {
            const size_t base = static_cast<size_t>(i) * static_cast<size_t>(A);
            double best = 0.0;
            for (int a = 0; a < A; ++a) {
                const size_t k = base + static_cast<size_t>(a);
                double q;
                if (INFO && INFO[k] != 0.0) {
                    double w;
                    if (CHAIN[k]) {
                        const double prod = C[k] * cur[D[k]];
                        w = prod;
                    } else {
                        w = bc[k];
                    }
                    const double s = R[k] + INFO[k];
                    q = s + w;
                } else {
                    const double prod = C[k] * cur[D[k]];
                    q = R[k] + prod;
                }
                if (a == 0 || q > best) best = q;
            }
            nxt[i] = best;
            const double d = std::fabs(best - cur[i]);
            if (d > delta) delta = d;
        }
        if (delta < tol) {
            std::memcpy(V, &nxt[0], sizeof(double) * static_cast<size_t>(n));
            return it + 1;
        }
        cur.swap(nxt);
    }
    std::memcpy(V, &cur[0], sizeof(double) * static_cast<size_t>(n));
    return maxIter;
}

}  // extern "C"
