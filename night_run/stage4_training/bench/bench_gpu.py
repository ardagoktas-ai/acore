"""
Bellman backup benchmark (PyTorch) for my chip comparison.

Kernel (the same backup my chip and driver compute, dense form):
    Q[a, s] = R[a, s] + gamma * sum_s' P[a, s, s'] * V[s']
    V'[s]   = max_a Q[a, s]

Assumption recorded here: P is a dense [A, n, n] row-stochastic tensor, R is
[A, n], V is [n]. One "backup" = one full sweep producing V' from V.

Runs on whatever CUDA GPU is present (A100/H100) for several n and A and
writes bench_results.csv. All numbers come from the run itself; nothing is
pre-filled. CPU runs are refused unless --allow-cpu is given (for testing the
script only; CPU numbers are not GPU results).

Usage:
    python bench/bench_gpu.py                       # default sweep, GPU
    python bench/bench_gpu.py --allow-cpu --quick   # script test on CPU
"""

import argparse
import csv
import os
import platform
import time

import torch

# n and A pairs measured by my chip simulation, plus larger sizes for the GPU.
CHIP_POINTS = [(1, 4), (7, 4), (33, 4), (64, 8), (5, 1)]
GPU_NS = [128, 512, 1024, 2048, 4096]
GPU_AS = [1, 4, 8]


def make_problem(n, A, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed + 31 * n + A)
    P = torch.rand(A, n, n, generator=g, dtype=torch.float64)
    P /= P.sum(-1, keepdim=True)
    R = torch.randn(A, n, generator=g, dtype=torch.float64)
    V = torch.randn(n, generator=g, dtype=torch.float64)
    return P, R, V, (P.to(device, dtype), R.to(device, dtype), V.to(device, dtype))


def backup(P, R, V, gamma):
    return (R + gamma * torch.matmul(P, V)).amax(dim=0)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_backup(P, R, V, gamma, device, iters, warmup):
    for _ in range(warmup):
        backup(P, R, V, gamma)
    sync(device)
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            backup(P, R, V, gamma)
        end.record()
        sync(device)
        return start.elapsed_time(end) * 1000.0 / iters      # microseconds
    t0 = time.perf_counter()
    for _ in range(iters):
        backup(P, R, V, gamma)
    return (time.perf_counter() - t0) * 1e6 / iters


def main():
    p = argparse.ArgumentParser(description="Bellman backup GPU benchmark")
    p.add_argument("--out", default="bench_results.csv")
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--dtypes", default="float32,bfloat16")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--quick", action="store_true", help="chip points only, few iterations")
    a = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False      # exact fp32 unless asked otherwise
        dev_name = torch.cuda.get_device_name(0)
    elif a.allow_cpu:
        device, dev_name = torch.device("cpu"), platform.processor() or "cpu"
        print("WARNING: no CUDA device; CPU numbers are only a script test")
    else:
        raise SystemExit("no CUDA GPU found (use --allow-cpu to test the script on CPU)")

    points = list(CHIP_POINTS)
    if not a.quick:
        points += [(n, A) for n in GPU_NS for A in GPU_AS]
    iters, warmup = (10, 2) if a.quick else (a.iters, a.warmup)
    dtypes = [getattr(torch, d) for d in a.dtypes.split(",")]

    rows = []
    for n, A in points:
        P64, R64, V64, _ = make_problem(n, A, "cpu", torch.float64)
        ref = backup(P64, R64, V64, a.gamma)
        for dt in dtypes:
            _, _, _, (P, R, V) = make_problem(n, A, device, dt)
            err = (backup(P, R, V, a.gamma).double().cpu() - ref).abs().max().item()
            usec = time_backup(P, R, V, a.gamma, device, iters, warmup)
            flops = 2.0 * A * n * n + 2.0 * A * n             # matvec + scale/add (+max ignored)
            gflops = flops / (usec * 1e-6) / 1e9
            row = {"device": dev_name, "torch": torch.__version__, "dtype": str(dt).split(".")[-1],
                   "n": n, "A": A, "iters": iters, "usec_per_backup": f"{usec:.3f}",
                   "gflops": f"{gflops:.3f}", "max_abs_error_vs_fp64": f"{err:.3e}"}
            rows.append(row)
            print(f"n={n:5d} A={A} {row['dtype']:9s} {row['usec_per_backup']:>10s} usec  "
                  f"{row['gflops']:>9s} GFLOP/s  err {row['max_abs_error_vs_fp64']}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
