#!/usr/bin/env python3
"""
yonetici.py - One-shot autonomous NIGHT RUN for my Acore project.

Start it once, walk away, read NIGHT_REPORT.md in the morning.

Stages (run in order, each with its own budget share, time limit and
INDEPENDENT verification done by this script, not by the model):
  1  Acore v5.0   evolve acore_v1_9.py: keep 25/25, push runtime below 0.5 s
  2  Chip v2.0    Verilog NPU + own ISA for Acore's heaviest loop (Bellman backup),
                  checked bit-exactly against my own Python golden model
  3  Driver v2.0  CUDA-style C++ runtime + kernel library + Python binding,
                  checked by my own contract test (+ Verilator co-sim if available)
  4  AI v1.5      H100 training script (bf16, torch.compile, hard time/cost caps)
                  with a local CPU smoke test
  5  Release      GitHub + Hugging Face repo files, READMEs, PRIVATE publish scripts
  6  Pitch        founder video script, X DM and email draft for Cory Levy

Safety:
  - Only network access: the Anthropic API. Nothing is pushed, sent, or rented.
  - Hard dollar cap across all runs (ledger), per-stage budget shares, worst-case
    cost check before every call, circuit breaker on repeated identical errors.
  - Every change is rolled back if it lowers the verification score.
  - Stages already VERIFIED are skipped when the script is started again.

Usage:
  python yonetici.py --check        (no API calls: tools, files, baseline test)
  python yonetici.py --night        (the full chain)
"""

import argparse
import ctypes
import datetime as dt
import fnmatch
import hashlib
import io
import json
import os
import platform
import pstats
import py_compile
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------
# Fixed model settings (only this model may touch my code)
# --------------------------------------------------------------------------
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
MODEL = "claude-opus-5-5"
EFFORT = "medium"

# USD per million tokens for claude-opus-5-5 (5-minute cache TTL).
PRICE_INPUT = 4.00
PRICE_OUTPUT = 20.00  # thinking tokens are billed as output
PRICE_CACHE_WRITE = 5.00
PRICE_CACHE_READ = 0.20

WORK_DIR_NAME = ".yonetici"
OUT_DIR_NAME = "night_run"

# Budget share and wall-clock limit (hours) per stage. Unspent money rolls forward.
STAGES = {
    1: {"name": "Acore v5.0", "dir": "stage1_acore", "weight": 0.35, "hours": 3.5},
    2: {"name": "Chip v2.0", "dir": "stage2_chip", "weight": 0.23, "hours": 2.0},
    3: {"name": "Driver v2.0", "dir": "stage3_driver", "weight": 0.18, "hours": 1.5},
    4: {"name": "AI v1.5", "dir": "stage4_training", "weight": 0.10, "hours": 1.0},
    5: {"name": "Release files", "dir": "stage5_release", "weight": 0.08, "hours": 0.6},
    6: {"name": "Pitch texts", "dir": "stage6_pitch", "weight": 0.06, "hours": 0.4},
    7: {"name": "Acore Studio", "dir": "stage7_studio", "weight": 0.25, "hours": 2.0},
}

# --finish: complete what the first night left open, then refresh release + pitch.
FINISH_ORDER = [1, 7, 5, 6]
FINISH_WEIGHTS = {1: 0.52, 7: 0.30, 5: 0.10, 6: 0.08}

FOUNDER_FACTS = {
    "name_note": "[TODO: my full name]",
    "age": 16,
    "location": "Aksaray, Turkey",
    "setup": "solo founder, stealth mode",
    "previous_work": "Lossrun AI: a 157M-parameter hybrid Mamba-2 + attention model that I "
                     "trained in 6 days and deployed live",
    "vision": "a vertically integrated stack designed from first principles: my own reasoning "
              "core (Acore), my own NPU and ISA, and my own CUDA-style software stack, built "
              "fabless instead of with brute-force GPU clusters",
    "audience": "Cory Levy, founder of Z Fellows",
}


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, msg=""):
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode(), flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_text(path, limit=None):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if limit and len(text) > limit:
        text = text[:limit] + f"\n...(truncated, {len(text) - limit} more characters)..."
    return text


def write_text(path, text):
    # Write atomically so a crash never leaves a half-written file.
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return default


def save_json(path, data):
    write_text(path, json.dumps(data, indent=2, default=str))


def run_cmd(cmd, cwd, timeout, env_extra=None):
    """Run a command; return (output, exit_code, seconds, timed_out)."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"  # never let a stale .pyc fake a result
    if env_extra:
        env.update(env_extra)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout, shell=isinstance(cmd, str),
        )
        out = proc.stdout.decode("utf-8", errors="replace")
        return out, proc.returncode, time.perf_counter() - start, False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", errors="replace")
        return out, None, float(timeout), True
    except OSError as e:
        return f"Could not start command: {e}", None, 0.0, False


def _norm(line):
    return re.sub(r"0x[0-9a-fA-F]+|\d+(\.\d+)?", "N", line.strip())


def log_excerpt(output, limit):
    """Send only the useful part of a log: failures, tracebacks, and the tail."""
    lines = output.splitlines()
    keep = set()
    key = re.compile(r"(?i)(fail|error|traceback|assert|expected|exception|mismatch)")
    for i, ln in enumerate(lines):
        if key.search(ln):
            for j in range(max(0, i - 2), min(len(lines), i + 6)):
                keep.add(j)
    for j in range(max(0, len(lines) - 25), len(lines)):
        keep.add(j)
    parts, last = [], -2
    for j in sorted(keep):
        if j != last + 1:
            parts.append("...")
        parts.append(lines[j][:400])
        last = j
    text = "\n".join(parts)
    if len(text) > limit:
        text = "...(truncated)...\n" + text[-limit:]
    return text


def keep_awake(log):
    """Ask the OS not to sleep while this process runs (best effort)."""
    system = platform.system()
    try:
        if system == "Windows":
            # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            log("Keep-awake: Windows sleep blocked while this script runs.")
        elif system == "Darwin" and shutil.which("caffeinate"):
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
            log("Keep-awake: caffeinate started.")
        else:
            log("Keep-awake: not available here. Disable sleep in the OS power settings.")
    except Exception as e:  # noqa: BLE001
        log(f"Keep-awake failed ({e}). Disable sleep in the OS power settings.")


def detect_tools():
    tools = {name: shutil.which(name) for name in
             ("iverilog", "vvp", "verilator", "g++", "clang++", "cl")}
    tools["cxx"] = tools["g++"] or tools["clang++"] or tools["cl"]
    out, code, _, _ = run_cmd([sys.executable, "-c", "import torch; print(torch.__version__)"],
                              os.getcwd(), 120)
    tools["torch"] = out.strip().splitlines()[-1] if code == 0 and out.strip() else None
    out, code, _, _ = run_cmd([sys.executable, "-c", "import numpy; print(numpy.__version__)"],
                              os.getcwd(), 60)
    tools["numpy"] = out.strip() if code == 0 else None
    return tools


# --------------------------------------------------------------------------
# Edit parsing and application
# --------------------------------------------------------------------------
class EditError(Exception):
    pass


def parse_edits(text):
    """Parse SEARCH/REPLACE blocks with a line-based state machine."""
    edits, state, search, replace = [], None, [], []
    for ln in text.splitlines():
        tag = ln.strip()
        if tag == "<<<<<<< SEARCH":
            state, search, replace = "search", [], []
        elif tag == "=======" and state == "search":
            state = "replace"
        elif tag == ">>>>>>> REPLACE" and state == "replace":
            edits.append(("\n".join(search), "\n".join(replace)))
            state = None
        elif state == "search":
            search.append(ln)
        elif state == "replace":
            replace.append(ln)
    if state is not None:
        raise EditError("Unterminated SEARCH/REPLACE block (output was probably cut off).")
    return edits


def apply_one(src, search, replace):
    if not search.strip():
        raise EditError("Empty SEARCH section.")
    if replace == "" and src.count(search + "\n") == 1:
        search += "\n"  # a pure deletion should not leave a blank line behind
    n = src.count(search)
    if n == 1:
        return src.replace(search, replace, 1)
    if n > 1:
        raise EditError(f"SEARCH text matches {n} places; it must be unique:\n{search[:300]}")
    # Fallback: tolerate trailing-whitespace differences, still requiring uniqueness.
    src_lines = src.split("\n")
    pat = [ln.rstrip() for ln in search.split("\n")]
    hits = [i for i in range(len(src_lines) - len(pat) + 1)
            if [ln.rstrip() for ln in src_lines[i:i + len(pat)]] == pat]
    if len(hits) != 1:
        raise EditError(f"SEARCH text not found exactly once:\n{search[:300]}")
    i = hits[0]
    return "\n".join(src_lines[:i] + replace.split("\n") + src_lines[i + len(pat):])


def apply_edits(src, edits):
    for search, replace in edits:
        src = apply_one(src, search, replace)
    return src


def parse_file_ops(text):
    """FILE (full content), EDIT (SEARCH/REPLACE) and DELETE blocks."""
    ops, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        m = re.match(r"^=== (FILE|EDIT|DELETE): (.+?) ===$", lines[i].strip())
        if m:
            kind, path = m.group(1), m.group(2).strip()
            if kind == "DELETE":
                ops.append(("delete", path, None))
                i += 1
                continue
            end = "=== END FILE ===" if kind == "FILE" else "=== END EDIT ==="
            body, i = [], i + 1
            while i < len(lines) and lines[i].strip() != end:
                body.append(lines[i])
                i += 1
            if i >= len(lines):
                raise EditError(f"Unterminated {kind} block for {path} (output cut off?).")
            if kind == "FILE" and body and body[0].strip().startswith("```") \
                    and body[-1].strip() == "```":
                body = body[1:-1]  # tolerate a wrapping code fence
            ops.append((kind.lower(), path, "\n".join(body)))
        i += 1
    return ops


# --------------------------------------------------------------------------
# Anthropic API and budget
# --------------------------------------------------------------------------
def call_claude(api_key, system_blocks, user_text, max_tokens, log):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "output_config": {"effort": EFFORT},
        "system": system_blocks,
        "messages": [{"role": "user", "content": user_text}],
    }
    data = json.dumps(body).encode("utf-8")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }
    delay = 10
    for attempt in range(1, 7):
        req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = "".join(b.get("text", "") for b in payload.get("content", [])
                           if b.get("type") == "text")
            return text, payload.get("usage", {}), payload.get("stop_reason")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504, 529):
                log(f"API busy (HTTP {e.code}), retry {attempt}/6 in {delay}s")
            else:
                raise RuntimeError(f"API error HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log(f"Network problem ({e}), retry {attempt}/6 in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 300)
    raise RuntimeError("API unreachable after 6 attempts.")


def cost_of(usage):
    return (usage.get("input_tokens", 0) * PRICE_INPUT
            + usage.get("output_tokens", 0) * PRICE_OUTPUT
            + usage.get("cache_creation_input_tokens", 0) * PRICE_CACHE_WRITE
            + usage.get("cache_read_input_tokens", 0) * PRICE_CACHE_READ) / 1e6


def worst_case_cost(system_blocks, user_text, max_tokens):
    chars = sum(len(b["text"]) for b in system_blocks) + len(user_text)
    return ((chars / 3.0) * PRICE_CACHE_WRITE + max_tokens * PRICE_OUTPUT) / 1e6


class Budget:
    """Dollar ledger shared by every run of this script."""

    def __init__(self, path, cap, run_limit=None):
        self.path, self.cap, self.run_limit = path, cap, run_limit
        self.ledger = load_json(path, {"total_spent_usd": 0.0, "runs": []})
        self.night = 0.0
        self.tokens = {"cache_read": 0, "cache_write": 0, "input": 0, "output": 0}

    @property
    def total(self):
        return self.ledger["total_spent_usd"]

    def left(self):
        left = self.cap - self.total
        if self.run_limit is not None:
            left = min(left, self.run_limit - self.night)
        return left

    def charge(self, usage):
        c = cost_of(usage)
        self.ledger["total_spent_usd"] += c
        self.night += c
        self.tokens["cache_read"] += usage.get("cache_read_input_tokens", 0)
        self.tokens["cache_write"] += usage.get("cache_creation_input_tokens", 0)
        self.tokens["input"] += usage.get("input_tokens", 0)
        self.tokens["output"] += usage.get("output_tokens", 0)
        save_json(self.path, self.ledger)
        return c


class StageGate:
    """Per-stage money and time limits, checked before every API call."""

    def __init__(self, night, budget_usd, hours):
        self.night, self.budget_usd = night, budget_usd
        self.deadline = time.time() + hours * 3600
        self.spent = 0.0

    def blocked(self, system, user, max_tokens):
        worst = worst_case_cost(system, user, max_tokens)
        if self.spent + worst > self.budget_usd:
            return f"stage budget ${self.budget_usd:.2f} would be exceeded"
        if self.night.budget.total + worst > self.night.budget.cap:
            return f"global cap ${self.night.budget.cap:.2f} would be exceeded"
        if worst > self.night.budget.left():
            return "this run's spending limit would be exceeded"
        if time.time() > self.deadline:
            return "stage time limit reached"
        if self.night.deadline and time.time() > self.night.deadline:
            return "night deadline reached"
        return None

    def ask(self, system, user, max_tokens):
        n = self.night
        text, usage, stop = n.call(n.api_key, system, user, max_tokens, n.log)
        c = n.budget.charge(usage)
        self.spent += c
        n.log(f"API ${c:.4f} | stage ${self.spent:.2f}/{self.budget_usd:.2f} | night "
              f"${n.budget.night:.2f} | cache read {usage.get('cache_read_input_tokens', 0)} "
              f"write {usage.get('cache_creation_input_tokens', 0)} | out "
              f"{usage.get('output_tokens', 0)}")
        return text, stop


# --------------------------------------------------------------------------
# STAGE 1: Acore evolution (speed, keep every test passing)
# --------------------------------------------------------------------------
class TestResult:
    def __init__(self):
        self.passed, self.total, self.score = 0, 0, 0.0
        self.seconds = float("inf")
        self.exit_code, self.timed_out = None, False
        self.output, self.score_source = "", "none"

    def short(self):
        t = "TIMEOUT" if self.timed_out else f"{self.seconds:.3f}s"
        return f"{self.passed}/{self.total} ({self.score * 100:.1f}%) in {t}"


def parse_score(output, exit_code, custom_regex):
    if custom_regex:
        hits = re.findall(custom_regex, output)
        if hits:
            p, t = hits[-1][:2]
            return int(p), int(t), "custom-regex"
    hits = re.findall(r"(?i)(\d+)\s*/\s*(\d+)\s*(?:tests?\s*)?pass", output)
    if hits:
        p, t = hits[-1]
        return int(p), int(t), "pattern"
    return (1, 1, "exit-code") if exit_code == 0 else (0, 1, "exit-code")


def run_tests(cfg, cwd, tests_file, expected_total=None, repeats=None, env_extra=None):
    """Run the suite several times; score = worst run, time = fastest run."""
    res = TestResult()
    times = []
    for i in range(repeats or cfg.repeats):
        out, code, elapsed, timed_out = run_cmd([sys.executable, tests_file], cwd,
                                                cfg.test_timeout, env_extra)
        passed, total, src = parse_score(out, code, cfg.score_regex)
        if expected_total and total < expected_total:
            total = expected_total  # missing tests count as failures
        if cfg.time_regex:
            m = re.findall(cfg.time_regex, out)
            if m:
                elapsed = float(m[-1] if isinstance(m[-1], str) else m[-1][0]) * cfg.time_scale
        if i == 0 or passed / total < res.score:
            res.passed, res.total, res.score_source = passed, total, src
            res.score = passed / total
            res.exit_code, res.output = code, out
        times.append(elapsed)
        if timed_out:
            res.timed_out = True
            break
        if code not in (0, None) and passed == 0:
            break
    res.seconds = min(times)
    return res


def error_signature(res, best_output=""):
    if res.timed_out:
        return "TIMEOUT"
    lines = [ln.strip() for ln in res.output.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit)\b", ln):
            return _norm(ln)[:160]
    known = {_norm(ln) for ln in best_output.splitlines()}
    fail = re.compile(r"(?i)\b(fail(ed|ure)?|error|exception)\b")
    new = sorted({_norm(ln) for ln in lines if fail.search(ln)} - known)
    if new:
        return "NEW FAILURES: " + " | ".join(new)[:160]
    return None


def profile_hotspots(cwd, tests_file, work, timeout, top=22):
    prof = os.path.join(work, "profile.out")
    try:
        run_cmd([sys.executable, "-m", "cProfile", "-o", prof, tests_file], cwd, timeout)
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(top)
        lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
        start = next((i for i, ln in enumerate(lines) if "ncalls" in ln), 0)
        return "\n".join(ln.replace(cwd + os.sep, "") for ln in lines[start:])
    except Exception as e:  # noqa: BLE001 - profiling is optional
        return f"(profiling failed: {e})"


S1_INSTRUCTIONS = """You are the lead engineer evolving "Acore", a general reasoning core written in Python \
(general ontology, N worlds, laws with 3+ conditions, agent role hierarchy, hidden-state modelling).

GOAL
- Keep every test in the read-only test suite passing (100% accuracy).
- Bring the suite's own measured runtime below {goal_seconds} seconds.
- Priority: correctness first, then speed. Never trade a passing test for speed.

HARD RULES
1. Edit ONLY the writable files listed below. The test file is read-only and checked by hash.
2. Build general mechanisms. Never special-case test inputs, hardcode expected outputs, read or \
parse the test file, inspect the call stack, or detect that a test harness is running. Never add \
module-level, class-level or otherwise global caches that let one engine instance reuse results \
computed by another instance: every engine must learn and plan on its own. Speed must come from \
better algorithms, data structures and native code, not from replaying earlier runs.
3. Keep every public name, signature and behaviour the tests rely on.
4. Python: standard library and numpy only.
5. All code, identifiers and comments in professional English. If a comment needs a pronoun, use \
"I", "me", "my" - never "we", "us", "our" or "let's".
6. Each turn must be ONE complete, working improvement. A change is kept only if the suite gets \
faster by at least {min_gain}% with every test still passing - a half-finished step is rolled back.
{native_rules}
OUTPUT FORMAT (strict - anything else is ignored)
PLAN: <one sentence describing the change and why it should help>
=== EDIT: relative/path ===
<<<<<<< SEARCH
exact lines copied from the current file
=======
replacement lines
>>>>>>> REPLACE
=== END EDIT ===
(use "=== FILE: relative/path === ... === END FILE ===" only for NEW files)

- SEARCH must match the current file exactly and uniquely; include 2-3 lines of context.
- Do not repeat an approach that the attempt history shows was already rejected.
"""

NATIVE_RULES = """
NATIVE KERNELS (my own driver from stage 3 is available in `npu_driver/`)
- `npu_driver/` holds my CUDA-style C++ runtime (npu_runtime.h, *.cpp), `build.py` (builds the \
shared library into npu_driver/build/) and `npu.py` (ctypes binding). This script rebuilds it \
after every change to those files.
- You may ADD new extern "C" kernels to the runtime and new Python wrappers to npu.py, and call \
them from Acore. Do NOT change or remove the existing API (available_backends, bellman_backup, \
npuInit ... npuGetErrorString): my stage 3 contract test re-runs on every accepted change.
- Kernels used by Acore must use float64/int64 with the same operation order as the numpy code \
they replace, so every decision of the agent stays identical. Never use the Q16.16 chip path \
for Acore's decisions: quantisation changes tie-breaks and breaks tests.
- Acore must import the driver lazily and fall back to its pure Python/numpy path when the \
library is missing or the environment variable ACORE_NATIVE is "0". I run the full suite with \
ACORE_NATIVE=0 on every accepted change: the fallback must pass 25/25 too.
- Amdahl's law: native code only helps where the time is. Read the profile. Value iteration \
alone is a small share; graph imagination (_build_graph, predict, _match_grid) dominates. Batch \
work per ctypes call (a call costs microseconds; never call native code per cell or per edge).
"""

S1_DRIVER_GLOBS = ["npu_driver/*.cpp", "npu_driver/*.h", "npu_driver/*.hpp",
                   "npu_driver/build.py", "npu_driver/npu.py", "npu_driver/__init__.py"]


def s1_snapshot(root, target_name, native):
    snap = {target_name: read_text(os.path.join(root, target_name))}
    if native:
        drv = os.path.join(root, "npu_driver")
        for rel in walk_files(drv):
            snap["npu_driver/" + rel] = read_text(os.path.join(drv, rel))
    return snap


def s1_restore(root, snap, native):
    if native:
        drv = os.path.join(root, "npu_driver")
        for rel in walk_files(drv):
            if "npu_driver/" + rel not in snap:
                os.remove(os.path.join(drv, rel))
    for rel, content in snap.items():
        write_text(os.path.join(root, *rel.split("/")), content)


def build_driver(drv_dir, timeout=900):
    out, code, secs, to = run_cmd([sys.executable, "build.py"], drv_dir, timeout)
    return code == 0, ("timed out" if to else log_excerpt(out, 1200))


def driver_contract_ok(night, drv_dir):
    """Run my stage 3 verifier on a temporary copy of the driver."""
    tmp = os.path.join(night.work, "driver_check")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(drv_dir, os.path.join(tmp, "driver"),
                    ignore=shutil.ignore_patterns(*SKIP_DIRS, *[f"*{e}" for e in SKIP_EXT]))
    docs = os.path.join(night.out, STAGES[3]["dir"], "docs")
    if os.path.isdir(docs):
        shutil.copytree(docs, os.path.join(tmp, "docs"))
    v, facts = verify_driver(night, tmp)
    return v.all_pass, v, facts


def setup_native(night):
    """Copy my VERIFIED stage 3 driver next to Acore and build it."""
    log = night.log
    st3 = night.state["stages"].get("3", {}).get("status")
    src = os.path.join(night.out, STAGES[3]["dir"], "driver")
    if st3 != "VERIFIED" or not os.path.isdir(src):
        log("Native mode off: stage 3 driver is not VERIFIED.")
        return False
    if not night.tools.get("cxx"):
        log("Native mode off: no C++ compiler.")
        return False
    drv = os.path.join(night.root, "npu_driver")
    if not os.path.isdir(drv):
        shutil.copytree(src, drv, ignore=shutil.ignore_patterns(
            *SKIP_DIRS, *[f"*{e}" for e in SKIP_EXT]))
    init = os.path.join(drv, "__init__.py")
    if not os.path.exists(init):
        write_text(init, '"""My CUDA-style driver for my own NPU (stage 3)."""\n')
    ok, detail = build_driver(drv)
    if not ok:
        log("Native mode off: driver build failed:\n" + detail)
        return False
    log("Native mode on: npu_driver/ built.")
    return True


def stage1_acore(night, gate):
    cfg, log = night.cfg, night.log
    root = night.root
    target = os.path.join(root, cfg.target)
    tests = os.path.join(root, cfg.tests)
    for p in (target, tests):
        if not os.path.isfile(p):
            return {"status": "FAILED", "stop_reason": f"missing file {os.path.basename(p)}"}
    target_name, tests_name = os.path.basename(target), os.path.basename(tests)
    work, stamp = night.work, night.stamp
    backup = target + ".bak"
    pristine = os.path.join(work, "tests_pristine.py")
    versions = os.path.join(work, "versions")
    os.makedirs(versions, exist_ok=True)
    native = bool(getattr(cfg, "native", False)) and setup_native(night)
    drv = os.path.join(root, "npu_driver")
    writable = [target_name] + (S1_DRIVER_GLOBS if native else [])

    shutil.copy2(target, os.path.join(versions, f"{stamp}_start_{target_name}"))
    shutil.copy2(tests, pristine)
    test_hash = sha256_of(tests)
    shutil.copy2(target, backup)
    snap = s1_snapshot(root, target_name, native)

    log(f"Stage 1 baseline tests (native={native})...")
    best = run_tests(cfg, root, tests)
    baseline = best
    expected_total = best.total
    log(f"Baseline: {best.short()} (parser: {best.score_source})")
    hotspots = profile_hotspots(root, tests, work, cfg.test_timeout * 3)
    history, accepted = [], []
    last_res = best
    same_err, last_sig, no_improve, it = 0, None, 0, 0
    stop = "unknown"
    native_rules = NATIVE_RULES if native else ""
    instructions = S1_INSTRUCTIONS.format(goal_seconds=cfg.goal_seconds,
                                          min_gain=int(cfg.min_speedup * 100),
                                          native_rules=native_rules)

    def register_error(sig):
        nonlocal same_err, last_sig
        same_err = same_err + 1 if sig == last_sig else 1
        last_sig = sig

    def reject(reason_sig, text, rebuild):
        nonlocal no_improve
        s1_restore(root, snap, native)
        if rebuild and native:
            build_driver(drv)
        if reason_sig:
            register_error(reason_sig)
        no_improve += 1
        history.append(f"#{it}: {text}")
        log(history[-1])

    try:
        while True:
            if best.score >= 1.0 and best.seconds < cfg.goal_seconds:
                stop = "GOAL REACHED"
                break
            if same_err >= 3:
                stop = f"circuit breaker: same error 3x ({last_sig})"
                break
            if no_improve >= cfg.patience:
                stop = f"no improvement in {no_improve} attempts"
                break
            files_txt = "\n".join(f"--- {rel} ---\n{content}" for rel, content in snap.items())
            system = [
                {"type": "text", "text": instructions + f"\nWRITABLE PATHS: {writable}\n"},
                {"type": "text", "text": f"READ-ONLY TEST SUITE `{tests_name}`:\n```python\n"
                                         f"{read_text(pristine)}\n```",
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "CURRENT BEST FILES:\n" + files_txt,
                 "cache_control": {"type": "ephemeral"}},
            ]
            user = "\n".join(
                [f"Current best: {best.short()}. Goal: all tests passing, under "
                 f"{cfg.goal_seconds}s.", "",
                 "cProfile of the current best version (own time; profiler overhead "
                 "inflates totals):", "```", hotspots, "```", "",
                 "Recent attempt history (newest last):"]
                + ([f"- {h}" for h in history[-8:]] or ["- (none yet)"])
                + ["", "Latest test output excerpt:", "```", log_excerpt(last_res.output, 3000),
                   "```", "", "Propose the next single improvement now."])
            why = gate.blocked(system, user, 24000)
            if why:
                stop = why
                break
            it += 1
            log(f"--- Stage 1 iteration {it} | best {best.short()}")
            text, stop_reason = gate.ask(system, user, 24000)
            m = re.search(r"^\s*PLAN:\s*(.+)$", text, re.MULTILINE)
            plan = m.group(1).strip()[:200] if m else "(no plan given)"
            log(f"PLAN: {plan}")
            try:
                if stop_reason == "max_tokens":
                    raise EditError("Response hit max_tokens; propose a smaller change.")
                ops = parse_file_ops(text)
                if not ops:
                    raise EditError("No EDIT/FILE blocks found in the response.")
                changed = apply_file_ops(root, ops, writable, ())
            except EditError as e:
                reject("EDIT: " + _norm(str(e).split("\n")[0])[:80],
                       f"REJECTED (edit not applied) | {plan} | {str(e)[:160]}", False)
                continue
            drv_changed = any(c.startswith("npu_driver/") for c in changed)
            try:
                py_compile.compile(target, cfile=os.path.join(work, "check.pyc"), doraise=True)
                if native and os.path.exists(os.path.join(drv, "npu.py")):
                    py_compile.compile(os.path.join(drv, "npu.py"),
                                       cfile=os.path.join(work, "check2.pyc"), doraise=True)
            except py_compile.PyCompileError as e:
                msg = str(e).strip().splitlines()[-1]
                reject("SYNTAX: " + _norm(msg)[:100],
                       f"REJECTED (syntax error) | {plan} | {msg[:160]}", drv_changed)
                continue
            if drv_changed:
                ok, detail = build_driver(drv)
                if not ok:
                    reject("BUILD: " + _norm(detail.splitlines()[-1] if detail else "")[:100],
                           f"REJECTED (driver build failed) | {plan} | {detail[-300:]}", True)
                    continue
            res = run_tests(cfg, root, tests, expected_total)
            last_res = res
            if sha256_of(tests) != test_hash:
                shutil.copy2(pristine, tests)
                reject("TEST FILE MODIFIED", f"REJECTED (modified the test file) | {plan}",
                       drv_changed)
                continue
            better = (res.score > best.score or (res.score == best.score and
                      res.seconds < best.seconds * (1.0 - cfg.min_speedup)))
            if not better:
                sig = error_signature(res, best.output)
                reason = sig or ("slower" if res.score == best.score else "lower score")
                if not sig:
                    same_err, last_sig = 0, None
                reject(sig, f"ROLLED BACK ({reason}) {res.short()} | {plan}", drv_changed)
                continue
            if native:
                fb = run_tests(cfg, root, tests, expected_total, repeats=1,
                               env_extra={"ACORE_NATIVE": "0"})
                if fb.score < 1.0:
                    reject("FALLBACK BROKEN",
                           f"REJECTED (pure-Python fallback fails: {fb.short()}) | {plan}",
                           drv_changed)
                    continue
                if drv_changed:
                    ok, v, _f = driver_contract_ok(night, drv)
                    if not ok:
                        reject("DRIVER CONTRACT: " + v.signature()[:80],
                               f"REJECTED (stage 3 contract broken: {v.signature()[:200]}) "
                               f"| {plan}", True)
                        continue
            old = best.short()
            best = res
            shutil.copy2(target, backup)
            snap = s1_snapshot(root, target_name, native)
            shutil.copy2(target, os.path.join(versions, f"{stamp}_iter{it:03d}_{target_name}"))
            same_err, last_sig, no_improve = 0, None, 0
            accepted.append(f"{old} -> {best.short()} | {plan}")
            history.append(f"#{it}: ACCEPTED {old} -> {best.short()} | {plan}")
            log(history[-1])
            hotspots = profile_hotspots(root, tests, work, cfg.test_timeout * 3)
    finally:
        s1_restore(root, snap, native)
        if native:
            build_driver(drv)
        if sha256_of(tests) != test_hash:
            shutil.copy2(pristine, tests)

    # Release copy: acore_v5_0.py (+ my driver) and a test file that imports it.
    sd = os.path.join(night.out, STAGES[1]["dir"])
    os.makedirs(sd, exist_ok=True)
    module = os.path.splitext(target_name)[0]
    src = re.sub(r'VERSION = "[^"]*"', 'VERSION = "5.0"', read_text(target), count=1)
    write_text(os.path.join(sd, "acore_v5_0.py"), src)
    write_text(os.path.join(sd, "test_v5_0.py"),
               read_text(pristine).replace(f"from {module} import", "from acore_v5_0 import"))
    write_text(os.path.join(sd, "EVOLUTION_LOG.md"),
               "# Acore evolution log\n\n" + "\n".join(f"- {h}" for h in history) + "\n")
    if native:
        dst = os.path.join(sd, "npu_driver")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(drv, dst, ignore=shutil.ignore_patterns("__pycache__"))
        build_driver(dst)
    final = run_tests(cfg, sd, os.path.join(sd, "test_v5_0.py"))
    fallback = run_tests(cfg, sd, os.path.join(sd, "test_v5_0.py"), repeats=1,
                         env_extra={"ACORE_NATIVE": "0"}) if native else None
    log(f"Stage 1 release check (acore_v5_0.py): {final.short()} | stop: {stop}")
    goal = final.score >= 1.0 and final.seconds < cfg.goal_seconds and \
        (fallback is None or fallback.score >= 1.0)
    status = "VERIFIED" if goal else ("PARTIAL" if final.score >= 1.0 else "FAILED")
    return {
        "status": status, "stop_reason": stop, "iterations": it,
        "tests": f"{final.passed}/{final.total}",
        "baseline_seconds": round(baseline.seconds, 4),
        "best_seconds": round(final.seconds, 4),
        "best_seconds_display": f"{final.seconds:.3f} s",
        "speedup": round(baseline.seconds / final.seconds, 2) if final.seconds else None,
        "goal_under_seconds": cfg.goal_seconds, "goal_met": goal,
        "native_driver": native,
        "pure_python_fallback": fallback.short() if fallback else None,
        "accepted_changes": len(accepted),
        "timing_note": "suite's internal timer, fastest of several runs, measured on my own CPU",
    }


# --------------------------------------------------------------------------
# Verification helpers for build stages (independent of the model)
# --------------------------------------------------------------------------
class Verdict:
    def __init__(self):
        # (name, status, detail); status: pass | fail | skip (required but a tool is
        # missing, so the stage cannot be VERIFIED) | na (optional, not applicable here)
        self.checks = []

    def add(self, name, status, detail=""):
        self.checks.append((name, status, str(detail)[:1500]))

    @property
    def n_pass(self):
        return sum(1 for _n, s, _d in self.checks if s == "pass")

    @property
    def n_fail(self):
        return sum(1 for _n, s, _d in self.checks if s == "fail")

    @property
    def skipped(self):
        return [n for n, s, _d in self.checks if s == "skip"]

    @property
    def score(self):
        total = self.n_pass + self.n_fail
        return self.n_pass / total if total else 0.0

    @property
    def all_pass(self):
        return self.n_fail == 0 and self.n_pass > 0

    def signature(self):
        return " | ".join(f"{n}:{_norm(d)[:60]}" for n, s, d in self.checks if s == "fail")

    def text(self):
        return "\n".join(f"[{s.upper()}] {n}" + (f" - {d}" if d else "")
                         for n, s, d in self.checks)


def need_files(v, sd, patterns):
    ok = True
    for pat in patterns:
        hits = [p for p in walk_files(sd) if fnmatch.fnmatch(p, pat)]
        if hits:
            v.add(f"file {pat}", "pass")
        else:
            v.add(f"file {pat}", "fail", "missing")
            ok = False
    return ok


SKIP_DIRS = {"build", "obj_dir", "__pycache__", "sim_run", "checkpoints", ".git", "runs"}
# Files this script writes itself: never linted, never shown to or writable by the model.
OWN_FILES = {"STAGE_LOG.md", "EVOLUTION_LOG.md"}
SKIP_EXT = {".vvp", ".so", ".dll", ".dylib", ".o", ".obj", ".exe", ".pyc", ".pt", ".bin",
            ".hex", ".lib", ".exp", ".a", ".pkl", ".safetensors", ".tmp", ".vcd"}


def walk_files(sd, opaque=()):
    out = []
    for dirpath, dirnames, filenames in os.walk(sd):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1].lower() in SKIP_EXT or fn in OWN_FILES:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), sd).replace(os.sep, "/")
            if any(rel.startswith(o) for o in opaque):
                continue
            out.append(rel)
    return out


BANNED_PRONOUNS = re.compile(r"(?i)\b(we|us|our|ours|ourselves|let's|we're|we've|we'll)\b")
TURKISH_CHARS = re.compile(r"[ğüşıöçĞÜŞİÖÇ]")


def lint_english(v, sd, rels):
    bad_p, bad_t = [], []
    for rel in rels:
        text = read_text(os.path.join(sd, rel))
        if rel.endswith(".py"):  # only comments and strings matter in code
            text = "\n".join(re.findall(r"#.*|\"\"\"[\s\S]*?\"\"\"|'[^'\n]*'|\"[^\"\n]*\"", text))
        for m in BANNED_PRONOUNS.finditer(text):
            bad_p.append(f"{rel}: '{m.group(0)}'")
        if TURKISH_CHARS.search(text):
            bad_t.append(rel)
    v.add("first-person singular only (no we/us/our)", "fail" if bad_p else "pass",
          "; ".join(bad_p[:8]))
    v.add("English only", "fail" if bad_t else "pass", ", ".join(bad_t[:8]))


# ---------------- Stage 2: chip ----------------
Q = 16


def to_q(x):
    return int(round(x * (1 << Q)))


def bellman_q16(n, a, r, c, d, v):
    """My golden model: V'[i] = max_a (R + ((CONT * V[DST]) >>> 16)), all int32 Q16.16."""
    out = []
    for i in range(n):
        best = None
        for k in range(i * a, i * a + a):
            val = r[k] + ((c[k] * v[d[k]]) >> Q)
            best = val if best is None else max(best, val)
        out.append(best)
    return out


def make_vector_case(seed, n, a):
    rng = random.Random(seed)
    r = [to_q(rng.choice([-1.0, -1.0, -30.0, 100.0, 5.0, rng.uniform(-30, 100)]))
         for _ in range(n * a)]
    c = [to_q(rng.choice([0.0, 0.95, 0.95, rng.random()])) for _ in range(n * a)]
    d = [rng.randrange(n) for _ in range(n * a)]
    v = [to_q(rng.uniform(-600, 2000)) for _ in range(n)]
    words = [n, a] + r + c + d + v
    hexfile = "\n".join(f"{w & 0xFFFFFFFF:08x}" for w in words) + "\n"
    return hexfile, bellman_q16(n, a, r, c, d, v)


CHIP_CASES = [(11, 1, 4), (12, 7, 4), (13, 33, 4), (14, 64, 8), (15, 5, 1)]


def verify_chip(night, sd):
    v = Verdict()
    have = need_files(v, sd, ["rtl/*.v", "tb/tb_npu.v", "docs/ISA.md"])
    tools = night.tools
    rtl = sorted(os.path.join(sd, p) for p in walk_files(sd)
                 if p.startswith("rtl/") and p.endswith((".v", ".sv")))
    tb = os.path.join(sd, "tb", "tb_npu.v")
    if rtl:
        bad = [os.path.basename(p) for p in rtl
               if re.search(r"\$(display|readmemh|fopen|fwrite|finish)", read_text(p))]
        v.add("RTL free of simulation-only system tasks", "fail" if bad else "pass",
              ", ".join(bad))
        has_top = any(re.search(r"\bmodule\s+npu_top\b", read_text(p)) for p in rtl)
        v.add("module npu_top defined in rtl/", "pass" if has_top else "fail")
    if tools["verilator"] and rtl:
        out, code, _, to = run_cmd([tools["verilator"], "--lint-only", "-Wno-fatal",
                                    "--top-module", "npu_top"] + rtl, sd, 300)
        v.add("Verilator lint (rtl)", "pass" if code == 0 else "fail",
              "" if code == 0 else log_excerpt(out, 800))
    else:
        v.add("Verilator lint (rtl)", "na", "optional: verilator not installed")
    cycles = {}
    if not (tools["iverilog"] and tools["vvp"]):
        v.add("Icarus simulation vs golden model", "skip", "iverilog/vvp not installed")
        return v, {"cycles": cycles, "simulator": None}
    if not have:
        v.add("Icarus simulation vs golden model", "fail", "required files missing")
        return v, {"cycles": cycles, "simulator": "Icarus Verilog"}
    simdir = os.path.join(sd, "sim_run")
    os.makedirs(simdir, exist_ok=True)
    out, code, _, _ = run_cmd([tools["iverilog"], "-g2012", "-o", "npu.vvp"] + rtl + [tb],
                              simdir, 300)
    if code != 0:
        v.add("Icarus compile", "fail", log_excerpt(out, 1200))
        return v, {"cycles": cycles, "simulator": "Icarus Verilog"}
    v.add("Icarus compile", "pass")
    for seed, n, a in CHIP_CASES:
        name = f"sim case n={n} A={a}"
        hexfile, expected = make_vector_case(seed, n, a)
        write_text(os.path.join(simdir, "vectors.hex"), hexfile)
        outp = os.path.join(simdir, "output.hex")
        if os.path.exists(outp):
            os.remove(outp)
        out, code, _, to = run_cmd([tools["vvp"], "-n", "npu.vvp"], simdir, 120)
        if to:
            v.add(name, "fail", "simulation timed out (120 s)")
            continue
        if not os.path.exists(outp):
            v.add(name, "fail", "output.hex not written\n" + log_excerpt(out, 600))
            continue
        got = []
        for ln in read_text(outp).split():
            try:
                w = int(ln, 16) & 0xFFFFFFFF
                got.append(w - (1 << 32) if w & 0x80000000 else w)
            except ValueError:
                pass
        exp32 = [((w + (1 << 31)) % (1 << 32)) - (1 << 31) for w in expected]
        if got[:n] != exp32 or len(got) < n:
            bad = [i for i in range(n) if i >= len(got) or got[i] != exp32[i]]
            i = bad[0]
            v.add(name, "fail", f"{len(bad)}/{n} rows wrong; first row {i}: expected "
                                f"{exp32[i]} got {got[i] if i < len(got) else 'nothing'}")
            continue
        m = re.findall(r"CYCLES:\s*(\d+)", out)
        if not m:
            v.add(name, "fail", "correct output but no 'CYCLES: <n>' line printed")
            continue
        cycles[f"n={n},A={a}"] = int(m[-1])
        v.add(name, "pass", f"bit-exact, {m[-1]} cycles")
    return v, {"cycles": cycles, "simulator": "Icarus Verilog"}


# ---------------- Stage 3: driver ----------------
CONTRACT_TEST = r'''
import json, random, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import npu
res = {"backends": {}, "errors": []}
backs = list(npu.available_backends())
res["available"] = backs
for b in backs:
    rng = random.Random(7)
    worst = 0.0
    try:
        for trial in range(40):
            n, a = rng.randint(1, 48), rng.randint(1, 8)
            R = np.array([[rng.choice([-1.0, -30.0, 100.0, rng.uniform(-30, 100)])
                           for _ in range(a)] for _ in range(n)])
            C = np.array([[rng.choice([0.0, 0.95, rng.random()]) for _ in range(a)]
                          for _ in range(n)])
            D = np.array([[rng.randrange(n) for _ in range(a)] for _ in range(n)], dtype=np.int64)
            V = np.array([rng.uniform(-600, 2000) for _ in range(n)])
            out = np.asarray(npu.bellman_backup(R, C, D, V, backend=b), dtype=np.float64)
            ref = np.max(R + C * V[D], axis=1)
            if out.shape != ref.shape:
                raise AssertionError(f"shape {out.shape} vs {ref.shape}")
            worst = max(worst, float(np.max(np.abs(out - ref))))
        n, a = (512 if b != "sim" else 64), 4
        R = np.random.default_rng(1).uniform(-30, 100, (n, a))
        C = np.full((n, a), 0.95)
        D = np.random.default_rng(2).integers(0, n, (n, a))
        V = np.random.default_rng(3).uniform(-600, 2000, n)
        reps = 200 if b != "sim" else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            npu.bellman_backup(R, C, D, V, backend=b)
        micros = (time.perf_counter() - t0) / reps * 1e6
        # Q16.16 quantisation alone can reach ~0.016 at |V| = 2000, so 0.05 is the bound.
        res["backends"][b] = {"max_abs_error": worst, "timed_n": n,
                              "us_per_call_a4": round(micros, 2), "ok": worst <= 0.05}
    except Exception as e:
        res["backends"][b] = {"ok": False, "error": repr(e)[:400]}
print("CONTRACT_JSON " + json.dumps(res))
'''


def verify_driver(night, sd):
    v = Verdict()
    have = need_files(v, sd, ["driver/*.cpp", "driver/npu_runtime.h", "driver/build.py",
                              "driver/npu.py", "docs/DRIVER.md"])
    ddir = os.path.join(sd, "driver")
    facts = {}
    for f in ("build.py", "npu.py"):
        p = os.path.join(ddir, f)
        if os.path.exists(p):
            try:
                py_compile.compile(p, cfile=os.path.join(night.work, "c.pyc"), doraise=True)
                v.add(f"syntax {f}", "pass")
            except py_compile.PyCompileError as e:
                v.add(f"syntax {f}", "fail", str(e)[-400:])
                have = False
    if not night.tools["cxx"]:
        v.add("C++ build", "skip", "no C++ compiler (g++/clang++/cl) installed")
        v.add("contract test", "skip", "needs the C++ build")
        return v, facts
    if not night.tools["numpy"]:
        v.add("contract test", "skip", "numpy not installed")
        return v, facts
    if not have:
        v.add("C++ build", "fail", "required files missing or invalid")
        return v, facts
    out, code, secs, to = run_cmd([sys.executable, "build.py"], ddir, 900)
    if code != 0:
        v.add("C++ build", "fail", "timed out" if to else log_excerpt(out, 1200))
        return v, facts
    v.add("C++ build", "pass", f"{secs:.1f} s")
    script = os.path.join(night.work, "contract_test.py")
    write_text(script, CONTRACT_TEST)  # lives outside the stage dir: the model cannot edit it
    out, code, _, to = run_cmd([sys.executable, script, ddir], ddir, 600)
    m = re.search(r"CONTRACT_JSON (.+)", out)
    if not m:
        v.add("contract test", "fail", "timed out" if to else log_excerpt(out, 1200))
        return v, facts
    res = json.loads(m.group(1))
    facts["backends"] = res["backends"]
    b = res["backends"]
    if "ref" not in b:
        v.add("backend 'ref' (C++ kernel library)", "fail", "not listed by available_backends()")
    else:
        v.add("backend 'ref' (C++ kernel library)", "pass" if b["ref"].get("ok") else "fail",
              json.dumps(b["ref"]))
    chip_ok = night.state.get("stages", {}).get("2", {}).get("status") == "VERIFIED"
    if "sim" in b:
        v.add("backend 'sim' (Verilator co-sim of my chip)", "pass" if b["sim"].get("ok")
              else "fail", json.dumps(b["sim"]))
    elif night.tools["verilator"] and chip_ok:
        v.add("backend 'sim' (Verilator co-sim of my chip)", "fail",
              "verilator is installed and the chip is verified, but no 'sim' backend")
    else:
        v.add("backend 'sim' (Verilator co-sim of my chip)", "na",
              "optional: needs verilator and a VERIFIED chip stage")
    return v, facts


# ---------------- Stage 4: training ----------------
def verify_training(night, sd):
    v = Verdict()
    have = need_files(v, sd, ["train.py", "envs.py", "requirements.txt", "README_TRAINING.md",
                              "bench/bench_gpu.py"])
    facts = {}
    for f in ("train.py", "envs.py", "bench/bench_gpu.py"):
        p = os.path.join(sd, f)
        if os.path.exists(p):
            try:
                py_compile.compile(p, cfile=os.path.join(night.work, "c.pyc"), doraise=True)
                v.add(f"syntax {f}", "pass")
            except py_compile.PyCompileError as e:
                v.add(f"syntax {f}", "fail", str(e)[-400:])
                have = False
    tp = os.path.join(sd, "train.py")
    if os.path.exists(tp):
        src = read_text(tp)
        for needle, label in (("bfloat16", "bf16"), ("torch.compile", "torch.compile"),
                              ("--max-hours", "--max-hours cap"), ("--resume", "--resume"),
                              ("--smoke", "--smoke mode")):
            v.add(f"train.py has {label}", "pass" if needle in src else "fail")
    if not night.tools["torch"]:
        v.add("CPU smoke run (loss must fall)", "skip", "PyTorch not installed")
        return v, facts
    if not have:
        v.add("CPU smoke run (loss must fall)", "fail", "required files missing or invalid")
        return v, facts
    out, code, secs, to = run_cmd([sys.executable, "train.py", "--smoke"], sd, 900)
    a = re.findall(r"LOSS_START\s+([-\d.eE+]+)", out)
    b = re.findall(r"LOSS_END\s+([-\d.eE+]+)", out)
    ok = code == 0 and "SMOKE_OK" in out and a and b and float(b[-1]) < float(a[-1])
    v.add("CPU smoke run (loss must fall)", "pass" if ok else "fail",
          f"loss {a[-1]} -> {b[-1]} in {secs:.0f} s" if ok else
          ("timed out" if to else log_excerpt(out, 1200)))
    if ok:
        facts["smoke"] = {"loss_start": float(a[-1]), "loss_end": float(b[-1])}
    return v, facts


# ---------------- Stage 5: release files ----------------
def assemble_release(night, sd):
    """Copy every verified-or-not artifact into the repo tree (no model tokens)."""
    gh = os.path.join(sd, "github")
    for sub, stage in (("acore", 1), ("chip", 2), ("driver", 3), ("training", 4),
                       ("studio", 7)):
        src = os.path.join(night.out, STAGES[stage]["dir"])
        dst = os.path.join(gh, sub)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                *SKIP_DIRS, *[f"*{e}" for e in SKIP_EXT]))
    save_json(os.path.join(gh, "FACTS.json"), night.facts())


RELEASE_OPAQUE = ("github/acore/", "github/chip/", "github/driver/", "github/training/",
                  "github/studio/",
                  "github/FACTS.json")


def verify_release(night, sd):
    v = Verdict()
    need = ["github/README.md", "github/.gitignore", "github/docs/ARCHITECTURE.md",
            "huggingface/README.md", "publish/publish_github.py", "publish/publish_hf.py",
            "publish/PUBLISH.md"]
    need_files(v, sd, need)
    texts = [p for p in walk_files(sd, RELEASE_OPAQUE)
             if p.endswith((".md", ".txt", ".py"))]
    lint_english(v, sd, texts)
    lint_claims(v, sd, [t for t in texts if t.endswith((".md", ".txt"))])
    freshness_checks(v, night, sd, ["github/README.md"])
    hf = os.path.join(sd, "huggingface", "README.md")
    if os.path.exists(hf):
        v.add("HF model card has YAML front matter", "pass" if read_text(hf).startswith("---")
              else "fail")
    for f in ("publish/publish_github.py", "publish/publish_hf.py"):
        p = os.path.join(sd, f)
        if not os.path.exists(p):
            continue
        src = read_text(p)
        try:
            py_compile.compile(p, cfile=os.path.join(night.work, "c.pyc"), doraise=True)
            v.add(f"syntax {f}", "pass")
        except py_compile.PyCompileError as e:
            v.add(f"syntax {f}", "fail", str(e)[-300:])
        v.add(f"{f} creates PRIVATE repos", "pass" if re.search(r"private", src, re.I) else "fail")
        v.add(f"{f} needs --confirm", "pass" if "--confirm" in src else "fail")
        leak = re.search(r"(ghp_|github_pat_|hf_)[A-Za-z0-9_]{20,}", src)
        v.add(f"{f} has no hardcoded token", "fail" if leak else "pass")
    return v, {}


# ---------------- Stage 6: pitch texts ----------------
def spoken_words(text):
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*\*[^*]*\*\*:", " ", "\n".join(lines))
    return len(re.findall(r"[A-Za-z0-9'’-]+", text))


def verify_pitch(night, sd):
    v = Verdict()
    need_files(v, sd, ["founder_video_script.md", "x_dm.txt", "email.txt"])
    p = os.path.join(sd, "founder_video_script.md")
    if os.path.exists(p):
        n = spoken_words(read_text(p))
        v.add("video script fits 1 minute (120-170 spoken words)",
              "pass" if 120 <= n <= 170 else "fail", f"{n} words")
    p = os.path.join(sd, "x_dm.txt")
    if os.path.exists(p):
        n = len(read_text(p).strip())
        v.add("X DM 300-1000 characters", "pass" if 300 <= n <= 1000 else "fail", f"{n} chars")
    p = os.path.join(sd, "email.txt")
    if os.path.exists(p):
        t = read_text(p).strip()
        v.add("email starts with 'Subject:'", "pass" if t.startswith("Subject:") else "fail")
        n = len(re.findall(r"[A-Za-z0-9'-]+", t.split("\n", 1)[-1]))
        v.add("email body 150-320 words", "pass" if 150 <= n <= 320 else "fail", f"{n} words")
    texts = [f for f in walk_files(sd) if f.endswith((".md", ".txt"))]
    lint_english(v, sd, texts)
    lint_claims(v, sd, texts)
    freshness_checks(v, night, sd, ["founder_video_script.md", "email.txt", "x_dm.txt"])
    return v, {}


def acore_time_display(night):
    s1 = night.state["stages"].get("1", {})
    if s1.get("best_seconds_display"):
        return s1["best_seconds_display"]
    return f"{s1['best_seconds']:.3f} s" if "best_seconds" in s1 else None


def freshness_checks(v, night, sd, rels):
    """The texts must describe the CURRENT system: today's Acore time and the Studio."""
    text = "\n".join(read_text(os.path.join(sd, r)) for r in rels
                     if os.path.exists(os.path.join(sd, r)))
    shown = acore_time_display(night)
    if shown:
        v.add(f"mentions Acore's measured runtime exactly as '{shown}'",
              "pass" if shown in text else "fail")
    st7 = night.state["stages"].get("7", {}).get("status")
    if st7:
        v.add("mentions Acore Studio", "pass" if "Acore Studio" in text else "fail")


# ---------------- Stage 7: Acore Studio ----------------
STUDIO_CONTRACT = r'''
import json, os, re, socket, subprocess, sys, time, urllib.request
studio_dir = sys.argv[1]
sys.path.insert(0, studio_dir)
checks = []
def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": str(detail)[:400]})
def tok(sym, text):
    return re.search(r"(?<![A-Za-z])" + re.escape(sym) + r"(?![A-Za-z])", text) is not None
def door_symbols(s):
    inv = {v: k for k, v in s.palette.items()}
    for ctx, g in s.agent.latent_groups.items():
        if dict(ctx).get((0, 1)) == s.palette["D"] and g.get("map"):
            return sorted({inv.get(dict(e[0]).get((0, 1)), "?") for e, _p in g["map"]})
    return None
def gate_cause(s):
    inv = {v: k for k, v in s.palette.items()}
    keys = s.agent.link_groups().get((s.palette["G"], s.palette["."]), [])
    if len(keys) != 1:
        return None
    return inv.get(dict(keys[0][0][0]).get((0, 1))), keys[0][1]
try:
    from acore_studio import Studio
    s = Studio(world="compound", seed=0)
    worlds = s.list_worlds()
    need = {"open_room", "key_corridor", "switch_room", "timer_room", "compound", "world_full"}
    check("list_worlds() covers the required worlds", need <= set(worlds), worlds)
    a = s.ask("What opens the door?")
    check("fresh agent: door question answered honestly as unknown",
          a.get("source") == "none" and a.get("answer"), a)
    eps = s.run_episodes(20)
    check("run_episodes(20) returns 20 episode records",
          len(eps) == 20 and all({"steps", "success", "hazards"} <= set(e) for e in eps),
          eps[-1] if eps else eps)
    check("agent succeeds in the compound world", any(e["success"] for e in eps),
          [e["steps"] for e in eps])
    syms = door_symbols(s)
    check("engine learned a door law", syms, syms)
    for q in ("What opens the door?", "How can I get the door to open?",
              "why does the door open"):
        a = s.ask(q)
        ok = syms and a.get("source") != "none" and all(tok(x, a.get("answer", "")) for x in syms)
        check(f"door answer matches the engine's law: {q!r}", ok, {"law": syms, "reply": a})
    a = s.ask("What is the capital of France?")
    check("out-of-scope question: no hallucination", a.get("source") == "none", a)
    st = s.step()
    check("step() contract", {"action", "reward", "done", "grid"} <= set(st)
          and isinstance(st["grid"], list) and all(isinstance(c, str) for c in st["grid"][0]), st)
    snap = s.snapshot()
    check("snapshot() contract", {"world", "tick", "stats", "laws"} <= set(snap),
          list(snap))
    w = Studio(world="world_full", seed=0)
    w.run_episodes(10)
    cause = gate_cause(w)
    check("engine learned the gate cause in world_full", cause, cause)
    for q in ("What opens the gate?", "how do I open the gate"):
        a = w.ask(q)
        ok = cause and a.get("source") != "none" and tok(cause[0], a.get("answer", "")) \
            and re.search(r"\b%d\b" % cause[1], a.get("answer", ""))
        check(f"gate answer names the cause and the delay: {q!r}", ok,
              {"cause": cause, "reply": a})
    a = w.ask("Where is my body?")
    check("body question answered from the engine", a.get("source") != "none", a)
except Exception as e:
    import traceback
    check("Studio API raised an exception", False, traceback.format_exc()[-1200:])
proc = None
try:
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]; sock.close()
    proc = subprocess.Popen([sys.executable, "server.py", "--port", str(port)], cwd=studio_dir,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    html = None
    for _ in range(60):
        try:
            html = urllib.request.urlopen(base + "/", timeout=2).read().decode("utf-8")
            break
        except Exception:
            time.sleep(0.5)
    check("server serves the UI at /", html and "<html" in html.lower())
    check("UI discloses 'research prototype'", html and "research prototype" in html.lower())
    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    post("/api/new", {"world": "compound", "seed": 0})
    post("/api/run", {"episodes": 3})
    r = post("/api/ask", {"text": "what have you learned?"})
    check("POST /api/ask returns an answer", isinstance(r, dict) and r.get("answer"), r)
    snap = json.loads(urllib.request.urlopen(base + "/api/snapshot", timeout=10).read().decode())
    check("GET /api/snapshot works", "tick" in snap, list(snap))
except Exception as e:
    check("server test raised an exception", False, repr(e))
finally:
    if proc:
        proc.kill()
print("STUDIO_JSON " + json.dumps(checks, default=str))
'''

CLAIM_RULES = re.compile(
    r"(?i)\b(I am|I'm|this is|it is|Acore is|Studio is)\s+(an?\s+)?(AGI|artificial general "
    r"intelligence|general chat(bot)?|chatgpt)\b|\bAGI\s+(achieved|system|app|application|"
    r"assistant|chatbot)\b|\bfirst AGI\b")


def lint_claims(v, sd, rels):
    bad = []
    for rel in rels:
        for m in CLAIM_RULES.finditer(read_text(os.path.join(sd, rel))):
            bad.append(f"{rel}: '{m.group(0)}'")
    v.add("no overclaim (AGI achieved / AGI app / general chatbot)", "fail" if bad else "pass",
          "; ".join(bad[:6]))


def verify_studio(night, sd):
    v = Verdict()
    have = need_files(v, sd, ["studio/acore_studio.py", "studio/worlds.py", "studio/server.py",
                              "studio/static/index.html", "docs/STUDIO.md"])
    for f in ("studio/acore_studio.py", "studio/worlds.py", "studio/server.py"):
        p = os.path.join(sd, f)
        if os.path.exists(p):
            try:
                py_compile.compile(p, cfile=os.path.join(night.work, "c.pyc"), doraise=True)
                v.add(f"syntax {f}", "pass")
            except py_compile.PyCompileError as e:
                v.add(f"syntax {f}", "fail", str(e)[-400:])
                have = False
    texts = [p for p in walk_files(sd) if p.endswith((".md", ".py", ".html"))]
    lint_english(v, sd, texts)
    lint_claims(v, sd, texts)
    facts = {}
    if not have:
        v.add("Studio functional contract", "fail", "required files missing or invalid")
        return v, facts
    engine_dir = os.path.join(night.out, STAGES[1]["dir"])
    if not os.path.exists(os.path.join(engine_dir, "acore_v5_0.py")):
        v.add("Studio functional contract", "fail", "stage1_acore/acore_v5_0.py not found")
        return v, facts
    script = os.path.join(night.work, "studio_contract.py")
    write_text(script, STUDIO_CONTRACT)  # outside the stage dir: the model cannot edit it
    out, code, secs, to = run_cmd([sys.executable, script, os.path.join(sd, "studio")],
                                  os.path.join(sd, "studio"), 900,
                                  {"ACORE_PATH": engine_dir})
    m = re.search(r"STUDIO_JSON (.+)", out)
    if not m:
        v.add("Studio functional contract", "fail", "timed out" if to else log_excerpt(out, 1200))
        return v, facts
    results = json.loads(m.group(1))
    for c in results:
        v.add(c["name"], "pass" if c["ok"] else "fail", "" if c["ok"] else c["detail"])
    facts["contract_checks"] = f"{sum(c['ok'] for c in results)}/{len(results)} passed"
    facts["kind"] = ("interactive research interface; natural-language questions are answered "
                     "from Acore's own learned model by a rule-based language layer (no LLM)")
    return v, facts


STUDIO_SPEC = """STAGE 7 - ACORE STUDIO: the human-facing interface to my Acore engine.

WHAT IT IS (and is not): a local web app where a person watches Acore live in a micro-universe \
and talks to it in natural language. Every answer is generated from Acore's OWN internal model \
(latent_groups / latent_hypotheses, link_groups, structure_report, reversibility, stats, \
self_values, motor_map) by a rule-based language layer: intent detection with flexible phrasing \
(regex + keyword patterns, synonyms, paraphrases) and templates that verbalise the engine's \
formulas. No LLM, no network. It is NOT a general chatbot and must never claim to be AGI: the UI \
header says "Acore Studio - research prototype". When the engine has not learned something, or a \
question is outside its world, it says so plainly (source "none") instead of guessing.

ENGINE: import ACOREEngine from acore_v5_0 located in the folder given by the environment \
variable ACORE_PATH (fallback: ../../stage1_acore relative to studio/). Add that folder to \
sys.path before importing.

DELIVERABLES (paths relative to this stage folder)
- studio/worlds.py : the micro-universe physics re-implemented exactly as in the test suite in \
CONTEXT (RawGridWorld, SYMBOLS, make_palette, make_action_perm, layouts). World names: \
open_room, key_corridor, chest_corridor, switch_room, timer_room, decoy_lab, compound \
(COMPOUND_WORLD, door rule: P pressed and Q not pressed), world_full (WORLD_FULL), world_a, \
world_b, world_c. For a world name and seed, build the palette with make_palette(seed), the \
actions with make_action_perm(seed), RawGridWorld(layout, pal, dirs, seed=seed, door_rule=...).
- studio/acore_studio.py : class Studio with EXACTLY this API:
      Studio(world="compound", seed=0)    # new engine ACOREEngine(n_actions=4, seed=seed)
      .agent      -> the ACOREEngine instance        .palette -> dict symbol -> grid value
      .list_worlds() -> list[str]
      .new_world(world, seed=0)           # keep the same engine (transfer between worlds)
      .reset_agent()
      .run_episodes(k, max_steps=150) -> list of {"steps": int, "success": bool, "hazards": int}
          (same loop as run_episode in the test suite: act, env.step, observe, end_episode)
      .step() -> {"action": int, "reward": float, "done": bool, "grid": list[list[str]]}
          (grid rendered as SYMBOLS, e.g. "#", ".", "A", "P")
      .ask(text) -> {"answer": str, "source": str, "confidence": str}
      .snapshot() -> {"world": str, "tick": int, "stats": dict, "laws": list[str]}
  ask() must handle, with flexible wording: what opens the door / how to open it / why it \
opens (verbalise the MAP formula with symbol letters, e.g. "the door opens if P was pressed and \
Q was not"), what opens the gate (cause symbol and delay in steps from link_groups), where/what \
my body is, what I am unsure about (open hypotheses), whether an action is reversible, what I \
have learned (stats + laws), help. Anything else -> source "none" with a short list of what I can \
answer. Symbols in answers are the single letters from SYMBOLS.
- studio/server.py : `python server.py --port N` (standard library http.server, default port \
8765) serving static/index.html and JSON routes: GET /api/worlds, POST /api/new {world, seed}, \
POST /api/step, POST /api/run {episodes}, POST /api/ask {text}, GET /api/snapshot.
- studio/static/index.html : single file (inline CSS/JS, no CDN). Grid view, controls (world, \
step, run N episodes), a chat panel for ask(), and a panel with the laws the engine believes.
- docs/STUDIO.md : what it is, how to run it, the question grammar, honest limitations.

CHECK: my own contract test imports Studio, trains it in "compound" and "world_full", compares \
every answer with the engine's internal state, asks out-of-scope questions, and runs the server.
"""


# --------------------------------------------------------------------------
# Stage specifications (what the model must build)
# --------------------------------------------------------------------------
BUILD_INSTRUCTIONS = """You are the lead engineer of a solo founder's deep-tech project. You work \
autonomously overnight; nobody will answer questions, so make sensible decisions yourself and \
record them in the docs.

RULES
1. Everything you write (code, identifiers, comments, docs, messages) is professional English.
2. Use first-person singular for the founder: "I", "me", "my", "own". Never "we", "us", "our", \
"ours" or "let's" - anywhere, including code comments.
3. Never invent results, measurements, benchmark numbers, users or partners. Only cite numbers \
that appear in FACTS. Anything unknown becomes "[TODO: ...]".
4. An independent verifier (not you) checks your work after every turn; its report is in the \
user message. Fix what it reports. Do not try to game it.
5. Work in small verified steps across turns. Prefer EDIT blocks for existing files; use FILE \
blocks for new files or full rewrites of small files. Keep each response well under the limit.

OUTPUT FORMAT (strict - anything else is ignored)
PLAN: <one sentence>
=== FILE: relative/path ===
<complete file content>
=== END FILE ===
=== EDIT: relative/path ===
<<<<<<< SEARCH
exact existing lines
=======
replacement lines
>>>>>>> REPLACE
=== END EDIT ===
=== DELETE: relative/path ===
STATUS: CONTINUE   (or STATUS: DONE when every requirement is met)

Paths are relative to the stage folder. Absolute paths and ".." are rejected.
"""

CHIP_SPEC = """STAGE 2 - CHIP v2.0: my own NPU for Acore's heaviest numeric loop.

WHY: Acore plans by value iteration (see the excerpt below). Each sweep is a Bellman backup:
    V'[i] = max over a of ( R[i,a] + CONT[i,a] * V[ DST[i,a] ] )
with n states (n <= 64 here) and A actions (1 <= A <= 8). This gather + multiply-add + max \
reduction is the kernel my chip accelerates.

DELIVERABLES (paths relative to this stage folder)
- rtl/*.v  : synthesizable Verilog-2005 (Icarus `-g2012` and Verilator compatible). Top module \
`npu_top` with clock, synchronous reset, and a small instruction/data interface of your design. \
The NPU has its OWN ISA (instruction encoding, registers, local SRAMs for R/CONT/DST/V, a \
pipelined datapath: gather, 32x32->64 multiply, arithmetic shift, add, running max). No \
$display/$readmemh/$finish or other simulation-only tasks inside rtl/.
- tb/tb_npu.v : testbench. It must read `vectors.hex` from the current directory with \
$readmemh into a 32-bit memory of depth 4096, load the problem into npu_top ONLY through \
npu_top's ports/ISA (computing the answer in the testbench is forbidden), run it, write \
`output.hex` (n lines, each one 32-bit word as %08x), print `CYCLES: <decimal>` (clock cycles \
from start command to done), then $finish.
- docs/ISA.md : instruction set (encoding table, registers, memory map, example program, cycle \
model). docs/CHIP.md : micro-architecture, pipeline diagram (ASCII), area/timing notes.
- openlane/config.json : an OpenLane config for npu_top (not run tonight; mark it untested).

EXACT DATA CONTRACT (checked bit-exactly by my own golden model)
vectors.hex = one 32-bit two's-complement word per line, in this order:
  n, A, R[0..n*A-1], CONT[0..n*A-1], DST[0..n*A-1], V[0..n-1]      (row-major: index i*A+a)
R, CONT, V are Q16.16 fixed point (int32). DST are plain integers.
Result: out[i] = max_a ( R[i*A+a] + ((CONT[i*A+a] * V[DST[i*A+a]]) >>> 16) )
where the product is a signed 64-bit product and >>> is an arithmetic (floor) shift; the sum is \
taken in 32 bits.

GOOD DESIGN: minimise cycles per backup (pipelining, overlap of gather and MAC). Correctness first.
"""

DRIVER_SPEC = """STAGE 3 - DRIVER v2.0: my own CUDA-style software stack for my NPU.

DELIVERABLES (paths relative to this stage folder)
- driver/npu_runtime.h + driver/*.cpp : a C/C++ runtime with a CUDA-like API, exported with \
extern "C": npuInit, npuGetDeviceCount, npuMalloc, npuFree, npuMemcpyHtoD, npuMemcpyDtoH, \
npuLaunchBellman (one Bellman backup over device buffers), npuSynchronize, npuGetLastCycles, \
npuGetErrorString. Backed by a kernel library:
    * backend "ref": an optimised C++ implementation that is bit-exact with the chip's Q16.16 \
semantics (see the chip contract below),
    * backend "sim": OPTIONAL, only when `verilator` is on PATH and ../../stage2_chip/rtl exists: \
Verilate npu_top and drive it cycle by cycle through the same ISA the chip docs define.
- driver/build.py : builds a shared library into driver/build/ with whatever is available \
(g++ or clang++ on Linux/macOS/WSL, cl.exe on Windows). Exit code 0 on success. Never downloads \
anything.
- driver/npu.py : Python binding via ctypes (standard library + numpy only) exposing exactly:
      available_backends() -> list of str  (always contains "ref" when the library is built)
      bellman_backup(R, CONT, DST, V, backend="ref") -> numpy float64 array of shape (n,)
  with R, CONT float arrays (n, A), DST int array (n, A), V float array (n,). It converts to \
Q16.16, calls the runtime, converts back.
- docs/DRIVER.md : architecture (Python -> runtime -> kernel library/backends), API reference, \
how Acore would call it (an `ACORE_BACKEND=npu` integration plan), build instructions.

CHECK: my own contract test builds the library, imports npu.py and compares every backend with \
numpy float64 on random problems (n <= 48, A <= 8; max abs error must be <= 0.05) and times it \
(n = 512 for "ref", which must support any n; n = 64 for "sim", the chip's SRAM limit).
"""

TRAIN_SPEC = """STAGE 4 - AI v1.5: training script for one rented NVIDIA H100 (Vast.ai).

Acore itself is symbolic and has no weights. The neural part I train is "Acore-WM", a small \
world model that learns the physics of Acore's micro-universes from trajectories: given the \
history of (grid observation, action) it predicts the next grid, the reward and the done flag. \
It is the learned perception/prediction layer that complements Acore's symbolic core.

DELIVERABLES (paths relative to this stage folder)
- envs.py : a self-contained re-implementation of the grid-world physics from the test suite in \
CONTEXT (walls, lava, keys, chests, switches, timers, gates, buttons, doors, flicker noise, \
random palettes, random action permutations) plus a random layout generator, and a fast \
trajectory generator (random + scripted policies, multiprocessing).
- train.py : pure PyTorch (no custom CUDA packages, so the smoke test runs on CPU). Decoder \
transformer over tokenised (observation, action) sequences. Must have: bf16 autocast, \
torch.compile (disabled under --smoke), fused AdamW, cosine schedule with warmup, gradient \
clipping, periodic checkpoints and `--resume`, `--max-hours` (hard stop, default 14), \
`--price-per-hour` (prints estimated total cost before starting and stops before exceeding \
`--max-cost`, default 30), TF32 enabled, data generated on the fly. `--smoke` runs a tiny model \
on CPU for at most ~3 minutes and prints `LOSS_START <x>`, `LOSS_END <y>` and `SMOKE_OK`.
- bench/bench_gpu.py : measures the Bellman backup (same kernel as my chip) with PyTorch on \
whatever GPU it runs on (A100/H100) for several n and A, and writes bench_results.csv. It is \
run later on the real GPU; no numbers are invented now.
- requirements.txt, README_TRAINING.md (Vast.ai step-by-step: choose image, upload, start in \
tmux, resume after interruption, download checkpoints, cost control).
"""

RELEASE_SPEC = """STAGE 5 - RELEASE FILES for a new GitHub repository and a new Hugging Face account.

The folder github/ already contains copies of every stage (acore/, chip/, driver/, training/, \
studio/) and FACTS.json. Do NOT edit those copies. If files from an earlier run exist, UPDATE \
them so they describe the current system in FACTS. Write only:
- github/README.md : the project README. Clear vision (first principles, fabless, vertically \
integrated), architecture diagram (ASCII), the status table of every stage taken from FACTS \
(use the exact status words: VERIFIED / PARTIAL / UNVERIFIED / FAILED), quick start, repo layout, \
roadmap, and an "Acore Studio" section (what it is, how to run it: python studio/server.py). \
Write Acore's runtime exactly as FACTS gives it in "best_seconds_display". Honest: no claim of \
AGI and no "AGI app"; describe the whole project as my path toward it.
- github/docs/ARCHITECTURE.md : how Acore, the NPU, the driver and the world model fit together.
- github/.gitignore
- huggingface/README.md : model card with YAML front matter (license: [TODO], tags), intended \
use, training data (synthetic micro-universes), status "weights not trained yet" unless FACTS \
says otherwise.
- publish/publish_github.py and publish/publish_hf.py : create PRIVATE repositories and upload \
the files (GitHub REST API with the GITHUB_TOKEN environment variable; huggingface_hub with \
HF_TOKEN). Default is a dry run that only prints what would happen; real actions need \
`--confirm`. Never hardcode tokens.
- publish/PUBLISH.md : step-by-step instructions for me.
"""

PITCH_SPEC = """STAGE 6 - MY PITCH TEXTS (I will review and send them myself; nothing is sent tonight).

Write in my voice: a 16-year-old solo founder from Aksaray, Turkey. Confident, concrete, \
humble, zero hype. Use ONLY facts from FACTS (founder facts + measured stage results). A stage \
that is not VERIFIED is described as "in progress". Never claim AGI is achieved. Never invent \
numbers.

DELIVERABLES
Acore Studio (stage 7) is my live demo: people talk to Acore in natural language and every \
answer comes from what Acore itself learned. Describe it exactly like that - never as a general \
chatbot or an AGI. If files from an earlier run exist, UPDATE them to the current FACTS. Quote \
Acore's runtime exactly as FACTS gives it in "best_seconds_display" in the email or the script.

- founder_video_script.md : my 1-minute camera script, 120-170 spoken words. Stage directions \
in [square brackets] (they are not counted). Structure: who I am -> what I already shipped \
(Lossrun) -> what I am building and why first principles -> proof from tonight's measured \
results -> the ask.
- x_dm.txt : a direct message to Cory Levy (@cory), 300-1000 characters, plain text.
- email.txt : first line "Subject: ...", then a 150-320 word email to Cory Levy that can also \
be pasted into the Z Fellows application.
"""


# --------------------------------------------------------------------------
# Generic build-stage engine
# --------------------------------------------------------------------------
def acore_excerpt(night):
    p = os.path.join(night.root, night.cfg.target)
    src = read_text(p)
    a, b = src.find("    def _solve_arrays"), src.find("    def _plan(")
    return src[a:b] if a >= 0 and b > a else src[:20000]


def acore_api_summary(night):
    p = os.path.join(night.out, STAGES[1]["dir"], "acore_v5_0.py")
    if not os.path.exists(p):
        p = os.path.join(night.root, night.cfg.target)
    src = read_text(p)
    lines = []
    for m in re.finditer(r'\n    def (\w+)\(([^)]*)\):\n(?:\s+"""([^\n]*))?', src):
        lines.append(f"def {m.group(1)}({' '.join(m.group(2).split())})"
                     + (f"  # {m.group(3).strip()}" if m.group(3) else ""))
    head = src.split('"""')[1][:3000] if src.count('"""') >= 2 else ""
    return head + "\n\n" + "\n".join(lines)


def stage_context(night, sid):
    """Read-only material each stage needs (kept small for cost)."""
    out = night.out
    parts = [("FACTS (the only numbers you may cite)",
              json.dumps(night.facts(), indent=2, default=str))]
    if sid == 2:
        parts.append(("Acore value iteration (the loop to accelerate)", acore_excerpt(night)))
    if sid == 3:
        chip = os.path.join(out, STAGES[2]["dir"])
        for rel in walk_files(chip):
            if rel.endswith((".v", ".md")):
                parts.append((f"stage2_chip/{rel}", read_text(os.path.join(chip, rel), 30000)))
        parts.append(("Chip data contract", CHIP_SPEC.split("EXACT DATA CONTRACT")[1]))
    if sid == 4:
        parts.append(("Test suite with the micro-universe physics",
                      read_text(os.path.join(night.root, night.cfg.tests), 60000)))
    if sid == 5:
        gh = os.path.join(out, STAGES[5]["dir"], "github")
        parts.append(("Repo tree", "\n".join(walk_files(gh))))
        for rel in ("chip/docs/ISA.md", "driver/docs/DRIVER.md", "training/README_TRAINING.md"):
            p = os.path.join(gh, rel)
            if os.path.exists(p):
                parts.append((rel, read_text(p, 12000)))
    if sid == 7:
        parts.append(("Test suite (physics, layouts, palettes, helper functions)",
                      read_text(os.path.join(night.root, night.cfg.tests), 60000)))
        parts.append(("ACOREEngine public API (signatures + first docstring line)",
                      acore_api_summary(night)))
    if sid == 6:
        readme = os.path.join(out, STAGES[5]["dir"], "github", "README.md")
        if os.path.exists(readme):
            parts.append(("My README (for consistency)", read_text(readme, 15000)))
    return "\n\n".join(f"### {t}\n{body}" for t, body in parts)


BUILD = {
    2: {"spec": CHIP_SPEC, "verify": verify_chip, "max_iters": 25, "writable": None},
    3: {"spec": DRIVER_SPEC, "verify": verify_driver, "max_iters": 25, "writable": None},
    4: {"spec": TRAIN_SPEC, "verify": verify_training, "max_iters": 15, "writable": None},
    5: {"spec": RELEASE_SPEC, "verify": verify_release, "max_iters": 10,
        "writable": ["github/README.md", "github/.gitignore", "github/docs/*",
                     "huggingface/*", "publish/*"]},
    6: {"spec": PITCH_SPEC, "verify": verify_pitch, "max_iters": 8, "writable": None},
    7: {"spec": STUDIO_SPEC, "verify": verify_studio, "max_iters": 22, "writable": None},
}


def safe_rel(rel, writable, opaque):
    rel = rel.replace("\\", "/").strip().strip("`")
    if not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel) \
            or ".." in rel.split("/"):
        raise EditError(f"Rejected path: {rel}")
    if any(rel.startswith(o) for o in opaque) or rel.split("/")[-1] in OWN_FILES:
        raise EditError(f"Read-only path: {rel}")
    if writable and not any(fnmatch.fnmatch(rel, pat) for pat in writable):
        raise EditError(f"Path not allowed in this stage: {rel} (allowed: {writable})")
    return rel


def apply_file_ops(sd, ops, writable, opaque):
    """All-or-nothing: compute every new file in memory, then write."""
    pending = {}
    for kind, rel, body in ops:
        rel = safe_rel(rel, writable, opaque)
        path = os.path.join(sd, *rel.split("/"))
        if kind == "file":
            pending[rel] = body.rstrip("\n") + "\n"
        elif kind == "delete":
            pending[rel] = None
        else:
            cur = pending.get(rel)
            if cur is None:
                if not os.path.exists(path):
                    raise EditError(f"EDIT on a file that does not exist: {rel}")
                cur = read_text(path)
            edits = parse_edits(body)
            if not edits:
                raise EditError(f"EDIT block for {rel} has no SEARCH/REPLACE pairs.")
            try:
                pending[rel] = apply_edits(cur, edits)
            except EditError as e:
                raise EditError(f"{rel}: {e}")
    for rel, content in pending.items():
        path = os.path.join(sd, *rel.split("/"))
        if content is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            write_text(path, content)
    return sorted(pending)


def snapshot(sd, opaque):
    return {rel: read_text(os.path.join(sd, rel)) for rel in walk_files(sd, opaque)}


def restore(sd, snap, opaque):
    for rel in walk_files(sd, opaque):
        if rel not in snap:
            os.remove(os.path.join(sd, rel))
    for rel, content in snap.items():
        write_text(os.path.join(sd, rel), content)


def files_block(sd, opaque, limit=150000):
    rels = walk_files(sd, opaque)
    if not rels:
        return "CURRENT STAGE FILES: (none yet)"
    chunks, used = [], 0
    for rel in rels:
        text = read_text(os.path.join(sd, rel))
        if used + len(text) > limit:
            chunks.append(f"--- {rel} --- (omitted: context limit; edit it with exact lines)")
            continue
        used += len(text)
        chunks.append(f"--- {rel} ---\n{text}")
    return "CURRENT STAGE FILES:\n" + "\n".join(chunks)


def run_build_stage(night, sid, gate):
    log, spec = night.log, BUILD[sid]
    sd = os.path.join(night.out, STAGES[sid]["dir"])
    os.makedirs(sd, exist_ok=True)
    opaque = RELEASE_OPAQUE if sid == 5 else ()
    if sid == 5:
        assemble_release(night, sd)
    verdict, vfacts = spec["verify"](night, sd)
    best = verdict
    snap = snapshot(sd, opaque)
    history = []
    same_err, last_sig, no_improve, it, clean_turns = 0, None, 0, 0, 0
    stop = "unknown"
    max_tokens = 32000
    context = stage_context(night, sid)
    log(f"Stage {sid} start | verifier: {best.n_pass} pass, {best.n_fail} fail, "
        f"{len(best.skipped)} skipped")
    while True:
        if it >= spec["max_iters"]:
            stop = f"max iterations ({spec['max_iters']})"
            break
        if same_err >= 3:
            stop = f"circuit breaker: same failure 3x ({last_sig[:120]})"
            break
        if no_improve >= 6:
            stop = "no progress in 6 turns"
            break
        system = [
            {"type": "text", "text": BUILD_INSTRUCTIONS},
            {"type": "text", "text": spec["spec"] + "\n\nCONTEXT (read-only)\n" + context
             + (f"\n\nWRITABLE PATHS: {spec['writable']}" if spec["writable"] else ""),
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": files_block(sd, opaque),
             "cache_control": {"type": "ephemeral"}},
        ]
        user = "\n".join(
            [f"Independent verifier report (score {best.score:.2f}):", best.text(), "",
             f"Tools on this machine: iverilog={bool(night.tools['iverilog'])}, "
             f"verilator={bool(night.tools['verilator'])}, C++={bool(night.tools['cxx'])}, "
             f"torch={night.tools['torch']}, OS={platform.system()}", "",
             "Turn history (newest last):"]
            + ([f"- {h}" for h in history[-8:]] or ["- (first turn)"])
            + ["", "Do the next step now."])
        why = gate.blocked(system, user, max_tokens)
        if why:
            stop = why
            break
        it += 1
        log(f"--- Stage {sid} turn {it}")
        text, stop_reason = gate.ask(system, user, max_tokens)
        m = re.search(r"^\s*PLAN:\s*(.+)$", text, re.MULTILINE)
        plan = m.group(1).strip()[:200] if m else "(no plan)"
        done_flag = bool(re.search(r"^\s*STATUS:\s*DONE", text, re.MULTILINE))
        log(f"PLAN: {plan}")
        try:
            if stop_reason == "max_tokens":
                raise EditError("Response hit max_tokens: write fewer/smaller files per turn.")
            ops = parse_file_ops(text)
            if not ops and not done_flag:
                raise EditError("No FILE/EDIT/DELETE blocks found.")
            changed = apply_file_ops(sd, ops, spec["writable"], opaque) if ops else []
        except EditError as e:
            sig = "EDIT: " + _norm(str(e).split("\n")[0])[:80]
            same_err, last_sig = (same_err + 1 if sig == last_sig else 1), sig
            no_improve += 1
            history.append(f"#{it}: REJECTED ({str(e)[:200]}) | {plan}")
            log(history[-1])
            continue
        verdict, vf = spec["verify"](night, sd)
        if verdict.score + 1e-9 < best.score:
            restore(sd, snap, opaque)
            sig = verdict.signature()
            same_err, last_sig = (same_err + 1 if sig == last_sig else 1), sig
            no_improve += 1
            history.append(f"#{it}: ROLLED BACK (score {verdict.score:.2f} < {best.score:.2f}); "
                           f"failing: {sig[:300]} | {plan}")
            log(history[-1])
            continue
        improved = verdict.score > best.score + 1e-9
        if not improved and verdict.n_fail:
            sig = verdict.signature()
            same_err, last_sig = (same_err + 1 if sig == last_sig else 1), sig
            no_improve += 1
        else:
            same_err, last_sig = 0, None
            no_improve = 0 if improved else no_improve
        best, vfacts = verdict, vf
        snap = snapshot(sd, opaque)
        history.append(f"#{it}: KEPT score {best.score:.2f} ({best.n_pass} pass/{best.n_fail} "
                       f"fail) files {changed[:6]} | {plan}")
        log(history[-1])
        # Finish when every check passes and the model says DONE, or after two
        # all-pass turns in a row (polishing beyond that is not worth the money).
        clean_turns = clean_turns + 1 if best.all_pass else 0
        if best.all_pass and (done_flag or clean_turns >= 2):
            stop = "all checks pass"
            break
    write_text(os.path.join(sd, "STAGE_LOG.md"),
               f"# Stage {sid} log\n\nStop: {stop}\n\n## Final verifier report\n```\n"
               f"{best.text()}\n```\n\n## Turns\n" + "\n".join(f"- {h}" for h in history) + "\n")
    if best.all_pass and not best.skipped:
        status = "VERIFIED"
    elif best.all_pass:
        status = "UNVERIFIED (skipped: " + ", ".join(best.skipped) + ")"
    elif best.n_pass:
        status = f"PARTIAL ({best.n_pass}/{best.n_pass + best.n_fail} checks)"
    else:
        status = "FAILED"
    log(f"Stage {sid} -> {status} | stop: {stop}")
    result = {"status": status, "stop_reason": stop, "turns": it,
              "checks": [{"name": n, "status": s, "detail": d[:300]} for n, s, d in best.checks]}
    result.update(vfacts or {})
    return result


# --------------------------------------------------------------------------
# Night orchestration
# --------------------------------------------------------------------------
class Night:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = os.path.abspath(cfg.dir)
        self.work = os.path.join(self.root, WORK_DIR_NAME)
        self.out = os.path.join(self.root, OUT_DIR_NAME)
        os.makedirs(self.work, exist_ok=True)
        os.makedirs(self.out, exist_ok=True)
        self.log = Logger(os.path.join(self.work, "night.log"))
        self.stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state_path = os.path.join(self.out, "state.json")
        self.state = load_json(self.state_path, {"stages": {}})
        self.budget = Budget(os.path.join(self.work, "ledger.json"), cfg.cap,
                             cfg.run_budget)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self.call = call_claude
        self.tools = {}
        self.deadline = time.time() + cfg.hours * 3600 if cfg.hours else None

    def facts(self):
        f = {"founder": FOUNDER_FACTS}
        for sid, info in STAGES.items():
            st = self.state["stages"].get(str(sid))
            if st:
                f[info["name"]] = {k: v for k, v in st.items() if k != "checks"}
        return f

    def save(self):
        save_json(self.state_path, self.state)
        save_json(os.path.join(self.out, "FACTS.json"), self.facts())


def sync_driver_back(night):
    """Acore may have added kernels to npu_driver/: copy them into stage 3 and re-verify."""
    drv = os.path.join(night.root, "npu_driver")
    sd = os.path.join(night.out, STAGES[3]["dir"])
    if not (os.path.isdir(drv) and os.path.isdir(os.path.join(sd, "driver"))):
        return
    for rel in walk_files(drv):
        if rel != "__init__.py":
            write_text(os.path.join(sd, "driver", *rel.split("/")),
                       read_text(os.path.join(drv, rel)))
    v, facts = verify_driver(night, sd)
    st = night.state["stages"].setdefault("3", {})
    st["status"] = "VERIFIED" if v.all_pass and not v.skipped else \
        (f"PARTIAL ({v.n_pass}/{v.n_pass + v.n_fail} checks)" if v.n_fail else "UNVERIFIED")
    st["checks"] = [{"name": n, "status": s_, "detail": d[:300]} for n, s_, d in v.checks]
    st.update(facts)
    st["note"] = "re-verified after Acore was linked to the driver"
    night.log(f"Stage 3 re-verified after the Acore link: {st['status']}")
    night.save()


def write_night_report(night, started):
    b = night.budget
    lines = ["# Night Run Report", "",
             f"- Started: {started}",
             f"- Finished: {dt.datetime.now().isoformat(timespec='seconds')}",
             f"- Model: `{MODEL}` (effort: {EFFORT})",
             f"- Spent tonight: ${b.night:.2f} | all runs: ${b.total:.2f} of ${b.cap:.2f}",
             f"- Tokens: cache read {b.tokens['cache_read']:,}, cache write "
             f"{b.tokens['cache_write']:,}, uncached input {b.tokens['input']:,}, output "
             f"{b.tokens['output']:,}", "",
             "| Stage | Status | Stop reason |", "|---|---|---|"]
    for sid, info in STAGES.items():
        st = night.state["stages"].get(str(sid), {})
        lines.append(f"| {sid}. {info['name']} | {st.get('status', 'NOT RUN')} | "
                     f"{st.get('stop_reason', '')} |")
    s1 = night.state["stages"].get("1")
    if s1 and "best_seconds" in s1:
        lines += ["", f"Acore: {s1['tests']} tests, {s1['baseline_seconds']} s -> "
                      f"{s1['best_seconds']} s (goal < {s1['goal_under_seconds']} s: "
                      f"{'met' if s1['goal_met'] else 'NOT met'})."]
    lines += ["", "## Before sending anything",
              "- Read every file in stage5_release and stage6_pitch yourself.",
              "- VERIFIED means my script checked it independently. UNVERIFIED means a tool was "
              "missing, so nobody has run it. PARTIAL/FAILED means it does not work yet.",
              "- Nothing was pushed, published, rented or sent.", "",
              "## Per-stage logs", "Each stage folder contains STAGE_LOG.md (stage 1: "
              "EVOLUTION_LOG.md). Full log: .yonetici/night.log"]
    write_text(os.path.join(night.out, "NIGHT_REPORT.md"), "\n".join(lines) + "\n")


def preflight(night):
    cfg, log = night.cfg, night.log
    night.tools = detect_tools()
    t = night.tools
    log("Tools: " + ", ".join(f"{k}={'yes' if t[k] else 'NO'}" for k in
                              ("iverilog", "vvp", "verilator", "cxx", "numpy", "torch")))
    warn = []
    if not (t["iverilog"] and t["vvp"]):
        warn.append("stage 2 simulation will be UNVERIFIED (install iverilog)")
    if not t["verilator"]:
        warn.append("no Verilator: no lint, no co-sim backend in stage 3")
    if not t["cxx"]:
        warn.append("stage 3 will be UNVERIFIED (install g++)")
    if not t["torch"]:
        warn.append("stage 4 smoke test will be UNVERIFIED (install torch)")
    if not t["numpy"]:
        warn.append("numpy missing: Acore tests and stage 3 cannot run")
    for w in warn:
        log("WARNING: " + w)
    for f in (cfg.target, cfg.tests):
        if not os.path.isfile(os.path.join(night.root, f)):
            log(f"ERROR: {f} not found in {night.root}")
            return False
    if not night.api_key and not cfg.check:
        log("ERROR: ANTHROPIC_API_KEY is not set.")
        return False
    log(f"Budget: ${night.budget.left():.2f} left of the ${night.budget.cap:.2f} cap "
        f"(already spent ${night.budget.total:.2f}).")
    return True


def parse_args():
    ap = argparse.ArgumentParser(description="Autonomous night run for my Acore project.")
    ap.add_argument("--night", action="store_true", help="Run the full chain.")
    ap.add_argument("--check", action="store_true", help="Preflight only, no API calls.")
    ap.add_argument("--dir", default=".", help="Project folder (default: current folder).")
    ap.add_argument("--target", default="acore_v1_9.py")
    ap.add_argument("--tests", default="test_v1.py")
    ap.add_argument("--cap", type=float, default=40.0,
                    help="Hard USD cap across ALL runs of this script (ledger).")
    ap.add_argument("--hours", type=float, default=9.0, help="Night deadline in hours.")
    ap.add_argument("--finish", action="store_true",
                    help="Finish run: Acore + native driver, Studio, then refresh release/pitch.")
    ap.add_argument("--run-budget", type=float, default=25.0,
                    help="USD this run may spend (the all-time --cap still applies).")
    ap.add_argument("--no-native", action="store_true",
                    help="--finish without linking Acore to the C++ driver.")
    ap.add_argument("--stages", default="1,2,3,4,5,6,7")
    ap.add_argument("--redo", action="store_true", help="Also rerun stages already VERIFIED.")
    ap.add_argument("--goal-seconds", type=float, default=0.5)
    ap.add_argument("--min-speedup", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--test-timeout", type=float, default=120.0)
    ap.add_argument("--score-regex", default=r"(\d+)/(\d+) passed in")
    ap.add_argument("--time-regex", default=r"passed in ([\d.]+) ms")
    ap.add_argument("--time-scale", type=float, default=0.001)
    return ap.parse_args()


def main():
    cfg = parse_args()
    if not (cfg.night or cfg.check or cfg.finish):
        sys.exit("Use --check first, then --night or --finish.")
    cfg.native = cfg.finish and not cfg.no_native
    night = Night(cfg)
    log = night.log
    started = dt.datetime.now().isoformat(timespec="seconds")
    log("=" * 70)
    log(f"NIGHT RUN | model {MODEL} | effort {EFFORT} | folder {night.root}")
    if not preflight(night):
        sys.exit(1)
    if cfg.check:
        res = run_tests(cfg, night.root, os.path.join(night.root, cfg.tests))
        log(f"Baseline Acore tests: {res.short()} (parser: {res.score_source})")
        log("Check finished. No API calls were made.")
        return
    keep_awake(log)
    done = lambda sid: night.state["stages"].get(str(sid), {}).get("status") == "VERIFIED"
    if cfg.finish:
        # 5 and 6 always run again: they must describe the system as it is NOW.
        todo = [s for s in FINISH_ORDER if cfg.redo or s in (5, 6) or not done(s)]
        weight = lambda sid: FINISH_WEIGHTS[sid]
    else:
        wanted = [int(s) for s in cfg.stages.split(",") if s.strip()]
        todo = [s for s in wanted if cfg.redo or not done(s)]
        weight = lambda sid: STAGES[sid]["weight"]
    log(f"Stages to run: {todo} | this run may spend ${night.budget.left():.2f}")

    def run_stage(idx, sid):
        weights = sum(weight(s) for s in todo[idx:])
        share = night.budget.left() * weight(sid) / weights
        gate = StageGate(night, share, STAGES[sid]["hours"])
        log("=" * 70)
        log(f"STAGE {sid}: {STAGES[sid]['name']} | budget ${share:.2f} | "
            f"max {STAGES[sid]['hours']} h")
        try:
            result = stage1_acore(night, gate) if sid == 1 else \
                run_build_stage(night, sid, gate)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001 - one broken stage must not end the night
            log(f"Stage {sid} crashed: {e!r}")
            result = {"status": "FAILED", "stop_reason": f"crash: {e!r}"[:300]}
        result["spent_usd"] = round(gate.spent, 4)
        night.state["stages"][str(sid)] = result
        night.save()

    try:
        for idx, sid in enumerate(todo):
            if night.budget.left() < 0.75:
                log("Budget nearly exhausted: stopping the chain.")
                break
            if night.deadline and time.time() > night.deadline:
                log("Deadline reached: stopping the chain.")
                break
            run_stage(idx, sid)
            if sid == 1 and cfg.native:
                sync_driver_back(night)
    except KeyboardInterrupt:
        log("Stopped by user (Ctrl+C).")
    finally:
        night.budget.ledger["runs"].append({"started": started,
                                            "spent": round(night.budget.night, 4)})
        save_json(night.budget.path, night.budget.ledger)
        night.save()
        write_night_report(night, started)
        log("=" * 70)
        for sid in STAGES:
            st = night.state["stages"].get(str(sid), {})
            log(f"  {sid}. {STAGES[sid]['name']:<14} {st.get('status', 'NOT RUN')}")
        log(f"Spent tonight ${night.budget.night:.2f} | total ${night.budget.total:.2f} "
            f"of ${night.budget.cap:.2f}")
        log(f"Report: {os.path.join(night.out, 'NIGHT_REPORT.md')}")


if __name__ == "__main__":
    main()
