# Architecture

This document explains how Acore, the NPU, the driver, the world model and Acore Studio fit together. Numbers come only from `FACTS.json`. Anything not yet measured is marked [TODO].

## 1. Acore - the reasoning core

Acore is symbolic and has no weights. It models small "micro-universes" (grid worlds with states, actions, rewards and terminal states) and plans in them with value iteration. The inner operation of value iteration is the Bellman backup:

    V'[i] = max_a ( R[i*A+a] + CONT[i*A+a] * V[DST[i*A+a]] )

Status: VERIFIED.

- The test suite passes (25/25), and the goal of under 0.5 s was reached after 26 iterations.
- The runtime is 0.474 s against a 0.781 s baseline (speedup 1.65), with 7 accepted changes.
- Acore runs with the native driver (`acore/npu_driver/`). When the driver is not available, the pure-Python fallback passes 25/25 (100.0%) in 0.560s.
- All timings come from the suite's internal timer, take the fastest of several runs, and were measured on my own CPU.

## 2. The NPU - my own chip and ISA

The NPU executes exactly this backup in fixed point (Q16.16). Its ISA is minimal and built for this one job:

- `CFG` sets the number of states and actions.
- `SWEEP` runs one full backup over all states. All actions of a state run in parallel lanes. It then swaps the two V banks (Jacobi style).
- `LOOP` / `DJNZ` repeat sweeps; `HALT` raises `done`.

The host works through a memory-mapped port:

1. It writes R, CONT, DST, V and the program.
2. It pulses `start`.
3. It waits for `done` and reads V back.

Full specification: `chip/docs/ISA.md`.

Status: VERIFIED in Icarus Verilog. Measured cycles for one backup: n=1,A=4: 10; n=7,A=4: 16; n=33,A=4: 42; n=64,A=8: 73; n=5,A=1: 14.

Open: the convergence test still runs on the host. [TODO: max-|delta| reduction in hardware]

## 3. The driver - my CUDA-style software stack

```
Acore  -->  npu_driver/npu.py  (ctypes, float64 <-> Q16.16)
                 |
           npu_runtime (C API: npuInit, npuMalloc,
                 |      npuMemcpyHtoD, npuLaunchBellman, ...)
                 |
     +-----------+-----------+
     |                       |
"ref" kernel             "sim" kernel
C++, bit-exact           Verilated npu_top
with the chip            [TODO: not built yet]
```

Every backend is a device, like GPUs in CUDA. All backends share one numeric contract, so code written against "ref" today runs on the RTL simulation, and later on silicon, without changes. `acore/npu_driver/` is the copy of the driver that Acore links against; `driver/` is the standalone stage.

Status: VERIFIED for "ref", and re-verified after Acore was linked to the driver. Max abs error is 0.014062233901086074; one call takes 54.18 microseconds (timed n=512, A=4).

## 4. Acore-WM - the world model

Acore-WM is a small neural network trained next to Acore. From a history of (grid observation, action) it predicts three things for Acore's micro-universes: the next grid, the reward and the done flag. Training data is generated on the fly by `training/envs.py`. The model sees only synthetic data, and nothing is downloaded.

Status: VERIFIED as code. The CPU smoke run took loss from 7.0234 to 6.2045. The real GPU run has not happened yet: [TODO: final loss and cost].

## 5. Acore Studio - the interface

Acore Studio is an interactive research interface (`studio/`). It is started with `python studio/server.py` from inside the `studio/` folder. A browser UI sends natural-language questions to the server. A rule-based language layer turns each question into a query on Acore's own learned model and phrases the answer. No LLM is involved.

Status: VERIFIED, contract checks 19/19 passed.

## 6. How the parts connect

1. Acore owns the reasoning: it defines the micro-universe and runs value iteration.
2. The backup inside value iteration goes through the driver to a device: "ref" now, the NPU later.
3. Acore Studio exposes Acore's learned model to me as a person, without a neural language model in between.
4. Acore-WM learns the dynamics of the same micro-universes. The goal is that Acore can eventually plan in worlds whose rules it was not given. [TODO: integration design]
5. Each layer is designed for the layer above it: the ISA exists because of the backup, and the driver exists because of the ISA.

Design decision: I keep the hardware narrow (one operation done well) and keep generality in software. That keeps the chip small enough for a solo, fabless effort.
