# Driver v2.0 - my CUDA-style software stack for my NPU

Status: the "ref" backend is implemented and passes my contract test. The optional Verilator "sim" backend (`driver/kernels_sim.cpp` plus the Verilator step in `build.py`) is written but has not been built yet, because my current machine has no Verilator ([TODO: build and test the sim backend on a machine with Verilator]). Measured accuracy and timings of the contract test: [TODO: record contract test results].

## Architecture

    Acore (Python)                       ACORE_BACKEND=npu
        |
    driver/npu.py        ctypes binding: float64 <-> Q16.16, buffer management
        |
    driver/npu_runtime.h  extern "C" CUDA-like API (npuInit, npuMalloc, npuLaunchBellman, ...)
    driver/npu_runtime.cpp device registry, device-memory table, bounds checks, dispatch
        |
    kernel library (driver/npu_kernels.h interface)
        +-- "ref": driver/kernels_ref.cpp  optimised C++, bit-exact with the chip, any n and A
        +-- "sim": driver/kernels_sim.cpp  Verilated npu_top, n <= 64, A <= 8 (untested)

Each backend is one device: device 0 is always "ref"; "sim" becomes device 1 when built.

### Numeric contract (identical on every backend)

    Vout[i] = max_a ( R[i*A+a] + ((CONT[i*A+a] * Vin[DST[i*A+a]]) >>> 16) )

Signed 64-bit product, arithmetic (floor) shift, truncation to 32 bits, 32-bit wrapping sum, signed max. The ref kernel implements the floor shift without relying on implementation-defined `>>` of negative numbers. Conversion in `npu.py`: `q = clip(rint(x * 65536), int32)`, result `/ 65536`.

### Design decisions

- Synchronous execution: every launch finishes before returning; `npuSynchronize` is kept so code written against a stream model ports unchanged.
- Device memory on the ref backend is host memory tracked in a table, so every copy and launch is bounds-checked like a real device.
- `DST` is validated to `[0, n)` in the runtime (the chip only uses 6 bits and treats out-of-range indices as undefined behaviour).
- If `dVout` overlaps `dVin`, the runtime copies `Vin` first so the Jacobi semantics of the chip are kept.
- The runtime is single-threaded (no locks), which keeps it portable to MinGW builds without a threading library.

## API reference (`driver/npu_runtime.h`)

| Function | Meaning |
|----------|---------|
| `npuInit(flags)` | Initialise (idempotent); `flags` must be 0. Other calls also initialise lazily. |
| `npuGetDeviceCount(int*)` | Number of backends compiled in. |
| `npuSetDevice(int)` / `npuGetDevice(int*)` | Select / query the current backend. |
| `npuGetDeviceName(int)` | `"ref"` or `"sim"`; NULL for invalid index. |
| `npuMalloc(void**, bytes)` / `npuFree(void*)` | Device buffers (opaque handles). |
| `npuMemcpyHtoD(dst, src, bytes)` / `npuMemcpyDtoH(dst, src, bytes)` | Copies with bounds checks. |
| `npuLaunchBellman(dR, dCont, dDst, dVin, dVout, n, A)` | One backup on the current device. |
| `npuSynchronize()` | No-op (launches are synchronous). |
| `npuGetLastCycles(uint64_t*)` | Chip cycles of the last launch (see below). |
| `npuGetLastError()` / `npuGetErrorString(err)` | Error reporting. |

Error codes: `npuSuccess`, `npuErrorNotInitialized`, `npuErrorInvalidValue`, `npuErrorMemoryAllocation`, `npuErrorInvalidDevicePointer`, `npuErrorInvalidDevice`, `npuErrorUnsupportedSize`, `npuErrorBackend`.

Cycles: "ref" returns the analytical single-backup model of Chip v2.0, `n + 9` cycles from `start` to `done`, for `n <= 64, A <= 8` and 0 otherwise. This model matches the Icarus runs of the chip testbench (n=1,A=4: 10; n=7,A=4: 16; n=33,A=4: 42; n=64,A=8: 73; n=5,A=1: 14). "sim" will return the cycles counted on the Verilated RTL.

### Python (`driver/npu.py`)

    import npu
    npu.available_backends()                       # e.g. ["ref"]
    Vnew = npu.bellman_backup(R, CONT, DST, V)     # float64, shape (n,)
    Vnew = npu.bellman_backup(R, CONT, DST, V, backend="sim")

`R`, `CONT`: float (n, A); `DST`: int (n, A) in `[0, n)`; `V`: float (n,). An extra helper `last_cycles()` returns `npuGetLastCycles`.

## "sim" backend (written, not yet built)

Built only when `verilator` is on PATH and `../../stage2_chip/rtl` exists. `build.py` runs `verilator --cc -Wno-fatal --top-module npu_top` into `driver/build/sim_obj`, locates the Verilator runtime via `verilator --getenv VERILATOR_ROOT`, and compiles the generated model, `verilated.cpp` (and `verilated_threads.cpp` when present) and `kernels_sim.cpp` with `-DNPU_HAVE_SIM -std=c++17`. Sim-enabled variants are tried first with GNU-style compilers; if Verilation or compilation fails, the build falls back to the ref-only variants, so the optional backend can never break the required one.

Per launch the harness drives the chip exactly as `stage2_chip/docs/ISA.md` and `tb_npu.v` do: synchronous reset (3 cycles), host writes of R/CONT/DST at `0x000/0x200/0x400 + i*8 + a`, V at `0x600 + i`, the program `CFG n,A; SWEEP; HALT` at `0x800`, a one-cycle `start` pulse, clocking until `done` while counting cycles the same way the testbench does, then a registered read of V through `host_raddr`. Inputs change while the clock is low and are sampled on the next rising edge. Limits: n <= 64, A <= 8 (`npuErrorUnsupportedSize` otherwise); a run longer than 100000 cycles returns `npuErrorBackend`.

Design decisions: one Verilated instance is kept for the lifetime of the library and reset on every launch, which avoids rebuilding the model per call. `sc_time_stamp()` is defined in the harness because Verilator 4.x requires it. The MSVC (`cl`) path is ref-only for now: [TODO: sim build with cl.exe].

## Acore integration plan (`ACORE_BACKEND=npu`)

1. Acore reads `ACORE_BACKEND` at start-up; `npu` selects this driver, anything else keeps the current numpy/Python path. [TODO: identify the exact value-iteration function in Acore that performs the backup.]
2. The backup call is replaced by `npu.bellman_backup(R, CONT, DST, V, backend=os.environ.get("ACORE_NPU_DEVICE", "ref"))` inside the existing loop; the `1e-3` convergence test stays on the host because the chip has no max-|delta| reduction yet.
3. Next optimisation: keep R, CONT and DST resident on the device across sweeps (allocate once, launch per sweep, ping-pong two V buffers) instead of copying per call. This needs a small persistent-context API in `npu.py`. [TODO]
4. Acceptance: Acore's test suite (25/25 in Acore v5.0) must still pass with `ACORE_BACKEND=npu`, since Q16.16 changes results slightly. [TODO: measure]

Context: Acore v5.0 did not reach its goal (best 0.8196 s against a 0.7684 s baseline, goal under 0.5 s). Whether this driver helps is unmeasured: [TODO: time Acore with ACORE_BACKEND=npu].

## Build

    python driver/build.py

Tries `cl` (Windows), `g++`, `clang++`, `c++`, with flag variants (on MinGW: static libgcc/libstdc++ so the DLL loads without extra DLLs). A variant is kept only if it compiles and a fresh Python process can load it and call `npuGetDeviceCount`. Output: `driver/build/npu_runtime.dll` (Windows), `libnpu_runtime.so` (Linux), `libnpu_runtime.dylib` (macOS). Nothing is downloaded. Note: the compiler must target the same architecture (32/64-bit) as the Python interpreter.
