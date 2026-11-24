# Stage 2 log

Stop: all checks pass

## Final verifier report
```
[PASS] file rtl/*.v
[PASS] file tb/tb_npu.v
[PASS] file docs/ISA.md
[PASS] RTL free of simulation-only system tasks
[PASS] module npu_top defined in rtl/
[NA] Verilator lint (rtl) - optional: verilator not installed
[PASS] Icarus compile
[PASS] sim case n=1 A=4 - bit-exact, 10 cycles
[PASS] sim case n=7 A=4 - bit-exact, 16 cycles
[PASS] sim case n=33 A=4 - bit-exact, 42 cycles
[PASS] sim case n=64 A=8 - bit-exact, 73 cycles
[PASS] sim case n=5 A=1 - bit-exact, 14 cycles
```

## Turns
- #1: KEPT score 1.00 (11 pass/0 fail) files ['docs/CHIP.md', 'docs/ISA.md', 'openlane/config.json', 'rtl/npu_top.v', 'tb/tb_npu.v'] | Create the first complete, verifiable version of the NPU: RTL (`npu_top` with its own ISA and a 5-stage gather/MAC/max pipeline), the testbench, ISA and CHIP docs, and an untested OpenLane config.
- #2: KEPT score 1.00 (11 pass/0 fail) files ['docs/CHIP.md', 'docs/ISA.md', 'rtl/npu_top.v', 'tb/tb_npu.v'] | Cut SWEEP from N*A+6 to N+6 cycles with 8 parallel action lanes (per-lane R/CONT/DST banks, 8-port V gather, 8 multipliers, masked max tree), using a stride-8 host memory map, and update the testbench
