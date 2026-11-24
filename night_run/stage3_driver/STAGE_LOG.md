# Stage 3 log

Stop: all checks pass

## Final verifier report
```
[PASS] file driver/*.cpp
[PASS] file driver/npu_runtime.h
[PASS] file driver/build.py
[PASS] file driver/npu.py
[PASS] file docs/DRIVER.md
[PASS] syntax build.py
[PASS] syntax npu.py
[PASS] C++ build - 0.5 s
[PASS] backend 'ref' (C++ kernel library) - {"max_abs_error": 0.014062233901086074, "timed_n": 512, "us_per_call_a4": 52.12, "ok": true}
[NA] backend 'sim' (Verilator co-sim of my chip) - optional: needs verilator and a VERIFIED chip stage
```

## Turns
- #1: KEPT score 1.00 (9 pass/0 fail) files ['docs/DRIVER.md', 'driver/build.py', 'driver/kernels_ref.cpp', 'driver/npu.py', 'driver/npu_kernels.h', 'driver/npu_runtime.cpp'] | Create the first complete, buildable version of the driver stack (header, runtime and a bit-exact "ref" kernel library, a build script with compiler fallback and a load smoke test, a ctypes binding, a
- #2: KEPT score 1.00 (9 pass/0 fail) files ['docs/DRIVER.md', 'driver/build.py', 'driver/kernels_sim.cpp'] | Implement the optional Verilator "sim" backend (`kernels_sim.cpp`, which drives `npu_top` through the ISA host protocol) and let `build.py` try a sim-enabled build first when Verilator and the chip RT
