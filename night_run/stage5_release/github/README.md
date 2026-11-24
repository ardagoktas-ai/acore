# Acore - a vertically integrated reasoning stack, built from first principles

I am a 16-year-old solo founder working in stealth mode from Aksaray, Turkey. This repository holds my whole stack:

- my own reasoning core (Acore),
- my own NPU with its own ISA,
- my own CUDA-style driver stack,
- a small neural world model that I train next to Acore,
- Acore Studio, an interactive interface to Acore.

## Vision

Most AI progress today comes from brute-force GPU clusters. I am taking a different route. I am building a **vertically integrated** stack, designed from **first principles** and built **fabless**:

- **Acore**: my own symbolic reasoning core. It models small "micro-universes" and plans in them with value iteration.
- **NPU + ISA**: my own chip, designed around the one operation Acore spends its time in: the Bellman backup.
- **Driver**: my own CUDA-style runtime and kernel library. Software talks to my chip the way CUDA code talks to a GPU.
- **Acore-WM**: a small neural world model that learns the dynamics of the same micro-universes.
- **Acore Studio**: a research interface where I can ask Acore questions and inspect what it has learned.

This is **not** AGI, and I do not claim it is. It is my path toward it: understand every layer, own every layer, and only scale what is proven to work.

Previous work: Lossrun AI, a 157M-parameter hybrid Mamba-2 + attention model that I trained in 6 days and deployed live.

## Architecture

```
 +--------------------------------------------------------------+
 |  Acore Studio (browser UI + python server, no LLM)           |
 |        | questions                 ^ answers from Acore's    |
 |        v                           | own learned model       |
 |  Acore (Python)   symbolic reasoning, value iteration        |
 |        |                                    ^                |
 |        | Bellman backups                    | predicted      |
 |        v                                    | dynamics       |
 |  npu_driver/npu.py (ctypes, float64<->Q16.16)|  (planned)    |
 |        |   pure-Python fallback if absent   |                |
 |        v                              Acore-WM (PyTorch)     |
 |  npu_runtime  (CUDA-style C API)      world model trained    |
 |        |                              on synthetic           |
 |   +----+---------------------+        micro-universes        |
 |   |                          |                               |
 |  "ref" kernel (C++)     "sim" kernel (Verilated RTL)         |
 |  bit-exact with chip    [TODO: not built yet]                |
 |                              |                               |
 |                              v                               |
 |                   NPU RTL (Verilog, own ISA)                 |
 |                   CFG / SWEEP / LOOP / DJNZ / HALT           |
 +--------------------------------------------------------------+
```

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

Every status and number below is copied from `FACTS.json`. Nothing is rounded up.

| Stage | Status | Evidence |
|-------|--------|----------|
| Acore v5.0 | VERIFIED | Goal reached after 26 iterations. Tests 25/25. Runtime **0.474 s**, against a 0.781 s baseline (speedup 1.65); the goal was under 0.5 s. 7 accepted changes. Acore uses the native driver. The pure-Python fallback passes 25/25 (100.0%) in 0.560s. Timing comes from the suite's internal timer, takes the fastest of several runs, and was measured on my own CPU. |
| Chip v2.0 | VERIFIED | All checks pass in Icarus Verilog. Cycles from `start` to `done` for one backup: n=1,A=4: 10; n=7,A=4: 16; n=33,A=4: 42; n=64,A=8: 73; n=5,A=1: 14. |
| Driver v2.0 | VERIFIED | All checks pass, re-verified after Acore was linked to the driver. The "ref" backend has a max abs error of 0.014062233901086074 (Q16.16 vs float) and takes 54.18 microseconds per call (timed n=512, A=4). The "sim" backend is written but not built yet (no Verilator on my machine). |
| AI v1.5 (Acore-WM training code) | VERIFIED | All checks pass. The CPU smoke run took loss from 7.0234 to 6.2045. The real GPU run has not happened yet. |
| Release files | VERIFIED | All checks pass (this README, docs, model card, publish scripts). |
| Pitch texts | VERIFIED | All checks pass. |
| Acore Studio | VERIFIED | All checks pass; contract checks 19/19 passed. |

## Quick start

Requirements: Python with numpy, a C++ compiler, Icarus Verilog, and PyTorch (for training only). Verilator is optional.

```
# Acore test suite (uses the native driver if built, else pure Python)
python acore/npu_driver/build.py
python acore/test_v5_0.py        # [TODO: confirm the exact test command]

# Chip testbench (Icarus Verilog)
iverilog -o tb_npu chip/tb/tb_npu.v chip/rtl/npu_top.v
vvp tb_npu

# Driver ("ref" backend)
python driver/driver/build.py

# Acore Studio
cd studio
python studio/server.py

# World model: local checks on CPU
cd training
pip install -r requirements.txt
python envs.py
python train.py --smoke
```

GPU training instructions: [training/README_TRAINING.md](training/README_TRAINING.md).

## Acore Studio

**What it is:** an interactive research interface for Acore. I type natural-language questions about a micro-universe. A rule-based language layer answers them from Acore's own learned model. There is no LLM anywhere in the loop, so every answer traces back to what Acore actually learned. Status: VERIFIED (contract checks 19/19 passed).

**How to run it:**

```
cd studio
python studio/server.py
```

Then open the address the server prints in a browser. [TODO: document the default port]

Files:

- `studio/studio/server.py`: the local server.
- `studio/studio/acore_studio.py`: the bridge to Acore and the rule-based language layer.
- `studio/studio/worlds.py`: the example micro-universes.
- `studio/studio/static/index.html`: the UI.

Documentation: `studio/docs/STUDIO.md`.

## Repository layout

```
acore/      Acore v5.0 reasoning core, test suite, linked native driver (npu_driver/)
chip/       NPU RTL (rtl/), testbench (tb/), ISA and chip docs (docs/), OpenLane config
driver/     CUDA-style runtime, kernels (ref, sim), Python binding, build script, docs
training/   Acore-WM world model: environments, training script, GPU benchmark
studio/     Acore Studio: server, language layer, example worlds, web UI, docs
docs/       ARCHITECTURE.md
FACTS.json  the single source of truth for every number in this README
```

## Roadmap

1. Driver: keep R, CONT and DST resident on the device across sweeps instead of copying them per call. [TODO]
2. Driver: build and test the "sim" backend on a machine with Verilator. [TODO]
3. Chip: add a hardware convergence check (max-|delta| reduction and a conditional branch). [TODO]
4. Chip: run the OpenLane flow toward a tape-out-ready layout. [TODO]
5. Acore-WM: do the real GPU training run, then publish the weights. [TODO]
6. Acore Studio + Acore-WM: let Acore plan in worlds whose rules it was not given. [TODO: integration design]

## License

[TODO: choose a license]

## Contact

[TODO: my full name and contact]
