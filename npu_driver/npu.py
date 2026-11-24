"""Python binding for my NPU runtime (ctypes + numpy only).

Public API:
    available_backends() -> list of str
    bellman_backup(R, CONT, DST, V, backend="ref") -> numpy float64 array (n,)
"""
import ctypes
import os
import sys

import numpy as np

__all__ = ["available_backends", "bellman_backup"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD = os.path.join(_HERE, "build")
_LIB = None
_Q = 65536.0
_I32_MIN = -2147483648
_I32_MAX = 2147483647


def _lib_path():
    names = ["npu_runtime.dll", "libnpu_runtime.so", "libnpu_runtime.dylib"]
    for name in names:
        p = os.path.join(_BUILD, name)
        if os.path.exists(p):
            return p
    raise RuntimeError("NPU runtime not built; run: python %s"
                       % os.path.join(_HERE, "build.py"))


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ctypes.CDLL(_lib_path())
    c_int, c_void_p, c_size_t = ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
    sig = {
        "npuInit": ([ctypes.c_uint], c_int),
        "npuGetDeviceCount": ([ctypes.POINTER(c_int)], c_int),
        "npuSetDevice": ([c_int], c_int),
        "npuGetDeviceName": ([c_int], ctypes.c_char_p),
        "npuMalloc": ([ctypes.POINTER(c_void_p), c_size_t], c_int),
        "npuFree": ([c_void_p], c_int),
        "npuMemcpyHtoD": ([c_void_p, c_void_p, c_size_t], c_int),
        "npuMemcpyDtoH": ([c_void_p, c_void_p, c_size_t], c_int),
        "npuLaunchBellman": ([c_void_p] * 5 + [c_int, c_int], c_int),
        "npuSynchronize": ([], c_int),
        "npuGetLastCycles": ([ctypes.POINTER(ctypes.c_uint64)], c_int),
        "npuGetErrorString": ([c_int], ctypes.c_char_p),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = res
    _LIB = lib
    _check(lib.npuInit(0), "npuInit")
    return lib


def _check(code, what):
    if code != 0:
        msg = _LIB.npuGetErrorString(code) if _LIB is not None else b"?"
        raise RuntimeError("%s failed: %s (code %d)" % (what, msg.decode(), code))


def available_backends():
    """Names of the backends compiled into the runtime ("ref" is always first)."""
    lib = _load()
    count = ctypes.c_int(0)
    _check(lib.npuGetDeviceCount(ctypes.byref(count)), "npuGetDeviceCount")
    return [lib.npuGetDeviceName(i).decode() for i in range(count.value)]


def _to_q16(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("inputs must be finite")
    q = np.clip(np.rint(x * _Q), _I32_MIN, _I32_MAX)
    return np.ascontiguousarray(q.astype(np.int32))


def last_cycles():
    """Chip cycles reported for the last launch (see npuGetLastCycles)."""
    lib = _load()
    c = ctypes.c_uint64(0)
    _check(lib.npuGetLastCycles(ctypes.byref(c)), "npuGetLastCycles")
    return int(c.value)


def bellman_backup(R, CONT, DST, V, backend="ref"):
    """One Bellman backup in Q16.16 on the chosen backend; returns float64 (n,)."""
    lib = _load()
    R = np.asarray(R)
    CONT = np.asarray(CONT)
    DST = np.asarray(DST)
    V = np.asarray(V)
    if R.ndim != 2:
        raise ValueError("R must have shape (n, A)")
    n, A = R.shape
    if CONT.shape != (n, A) or DST.shape != (n, A) or V.shape != (n,):
        raise ValueError("shapes must be R,CONT,DST: (n, A) and V: (n,)")
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if A == 0:
        raise ValueError("A must be >= 1")
    if not np.all(np.equal(np.mod(DST, 1), 0)):
        raise ValueError("DST must contain integers")
    if DST.min() < 0 or DST.max() >= n:
        raise ValueError("DST entries must be in [0, n)")

    names = available_backends()
    if backend not in names:
        raise ValueError("backend %r not available (have %s)" % (backend, names))
    _check(lib.npuSetDevice(names.index(backend)), "npuSetDevice")

    hR = _to_q16(R).reshape(-1)
    hC = _to_q16(CONT).reshape(-1)
    hD = np.ascontiguousarray(DST.astype(np.int32)).reshape(-1)
    hV = _to_q16(V)
    hOut = np.zeros(n, dtype=np.int32)

    mat_bytes = hR.nbytes
    vec_bytes = hV.nbytes
    ptrs = [ctypes.c_void_p() for _ in range(5)]
    try:
        for p, size in zip(ptrs, [mat_bytes] * 3 + [vec_bytes] * 2):
            _check(lib.npuMalloc(ctypes.byref(p), size), "npuMalloc")
        for p, h in zip(ptrs[:4], [hR, hC, hD, hV]):
            _check(lib.npuMemcpyHtoD(p, h.ctypes.data_as(ctypes.c_void_p), h.nbytes),
                   "npuMemcpyHtoD")
        _check(lib.npuLaunchBellman(ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4],
                                    int(n), int(A)), "npuLaunchBellman")
        _check(lib.npuSynchronize(), "npuSynchronize")
        _check(lib.npuMemcpyDtoH(hOut.ctypes.data_as(ctypes.c_void_p), ptrs[4],
                                 hOut.nbytes), "npuMemcpyDtoH")
    finally:
        for p in ptrs:
            if p.value:
                lib.npuFree(p)
    return hOut.astype(np.float64) / _Q


_ACORE_FN = None


def _acore_fn():
    global _ACORE_FN
    if _ACORE_FN is None:
        lib = _load()
        fn = lib.acoreValueIteration
        c_void_p = ctypes.c_void_p
        fn.argtypes = [c_void_p] * 6 + [ctypes.c_int, ctypes.c_int, c_void_p,
                                        ctypes.c_int, ctypes.c_double]
        fn.restype = ctypes.c_int
        _ACORE_FN = fn
    return _ACORE_FN


def acore_value_iteration(R, CONT, DST, V0, max_iter=2000, tol=1e-3,
                          INFO=None, CHAIN=None, V_base=None):
    """Float64 Jacobi value iteration (Acore's planner); returns a new array (n,).
    R, CONT: float64 (n, A); DST: int64 (n, A); CHAIN: uint8 (n, A)."""
    fn = _acore_fn()
    R = np.ascontiguousarray(R, dtype=np.float64)
    CONT = np.ascontiguousarray(CONT, dtype=np.float64)
    DST = np.ascontiguousarray(DST, dtype=np.int64)
    n, A = R.shape
    V = np.array(V0, dtype=np.float64, copy=True)
    if INFO is not None:
        INFO = np.ascontiguousarray(INFO, dtype=np.float64)
        CHAIN = np.ascontiguousarray(CHAIN, dtype=np.uint8)
        V_base = np.ascontiguousarray(V_base, dtype=np.float64)
        p_info, p_chain, p_vb = INFO.ctypes.data, CHAIN.ctypes.data, V_base.ctypes.data
    else:
        p_info = p_chain = p_vb = None
    fn(R.ctypes.data, CONT.ctypes.data, DST.ctypes.data, p_info, p_chain, p_vb,
       int(n), int(A), V.ctypes.data, int(max_iter), float(tol))
    return V


if __name__ == "__main__":
    print("backends:", available_backends())
    sys.exit(0)
