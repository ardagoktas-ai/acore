#!/usr/bin/env python3
"""Build my NPU runtime into driver/build/ with whatever compiler is available.

Tries cl.exe (Windows), g++, clang++ and c++ in order, with a few flag variants,
and keeps the first library that both compiles and loads through ctypes in a
fresh Python process. Never downloads anything. Exit code 0 on success.

When `verilator` is on PATH and ../../stage2_chip/rtl exists, the chip RTL is
Verilated first and sim-enabled variants (ref + sim backends) are tried before
the plain ref-only variants, so a broken Verilator setup never blocks the build.
"""
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
RTL = os.path.normpath(os.path.join(HERE, "..", "..", "stage2_chip", "rtl"))
IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"


def lib_name():
    if IS_WIN:
        return "npu_runtime.dll"
    if IS_MAC:
        return "libnpu_runtime.dylib"
    return "libnpu_runtime.so"


def sources():
    # kernels_sim.cpp is only compiled together with Verilator output.
    return sorted(s for s in glob.glob(os.path.join(HERE, "*.cpp"))
                  if not os.path.basename(s).startswith("kernels_sim"))


def run(cmd, cwd, capture=False):
    print("+ " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
    except OSError as e:
        print("  failed to start: %s" % e)
        return (False, "") if capture else False
    if r.stdout and not capture:
        print(r.stdout.strip())
    if capture:
        return r.returncode == 0, (r.stdout or "")
    return r.returncode == 0


def smoke_test(path):
    code = ("import ctypes,sys;l=ctypes.CDLL(sys.argv[1]);c=ctypes.c_int(0);"
            "sys.exit(0 if l.npuGetDeviceCount(ctypes.byref(c))==0 and c.value>=1 else 1)")
    try:
        r = subprocess.run([sys.executable, "-c", code, path],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
    except OSError:
        return False
    if r.returncode != 0 and r.stdout:
        print(r.stdout.strip())
    return r.returncode == 0


def sim_config():
    """Verilate npu_top; return (extra_sources, extra_flags) or None."""
    tool = shutil.which("verilator")
    if not tool or not os.path.isdir(RTL):
        return None
    vfiles = sorted(glob.glob(os.path.join(RTL, "*.v")))
    if not vfiles:
        return None
    obj = os.path.join(BUILD, "sim_obj")
    shutil.rmtree(obj, ignore_errors=True)
    if not run([tool, "--cc", "-Wno-fatal", "--top-module", "npu_top",
                "--Mdir", obj] + vfiles, BUILD):
        print("  verilator failed; building without the sim backend")
        return None
    ok, out = run([tool, "--getenv", "VERILATOR_ROOT"], BUILD, capture=True)
    root = out.strip().splitlines()[-1].strip() if ok and out.strip() else ""
    if not root:
        root = os.environ.get("VERILATOR_ROOT", "")
    inc = os.path.join(root, "include")
    if not os.path.isfile(os.path.join(inc, "verilated.cpp")):
        print("  Verilator include directory not found; building without sim")
        return None
    srcs = sorted(glob.glob(os.path.join(obj, "Vnpu_top*.cpp")))
    srcs.append(os.path.join(inc, "verilated.cpp"))
    threads = os.path.join(inc, "verilated_threads.cpp")
    if os.path.isfile(threads):
        srcs.append(threads)
    srcs.append(os.path.join(HERE, "kernels_sim.cpp"))
    flags = ["-DNPU_HAVE_SIM", "-I" + HERE, "-I" + obj, "-I" + inc,
             "-I" + os.path.join(inc, "vltstd")]
    return srcs, flags


def platform_flags():
    if IS_WIN:
        # Static runtime libraries so the DLL loads without MinGW DLLs on PATH.
        return [["-static-libgcc", "-static-libstdc++", "-static"],
                ["-static-libgcc", "-static-libstdc++"],
                []]
    return [["-fPIC", "-fvisibility=hidden"], ["-fPIC"]]


def gnu_variants(cxx, out, srcs, sim):
    result = []
    if sim:
        sim_srcs, sim_flags = sim
        base = [cxx, "-std=c++17", "-O2", "-ffp-contract=off", "-shared", "-DNPU_BUILD_DLL",
                "-o", out]
        extra = [] if IS_WIN else ["-pthread"]
        for pf in platform_flags():
            result.append(base + sim_flags + srcs + sim_srcs + pf + extra)
    base = [cxx, "-std=c++11", "-O2", "-ffp-contract=off", "-shared", "-DNPU_BUILD_DLL",
            "-o", out] + srcs
    for pf in platform_flags():
        result.append(base + pf)
    return result


def candidates(out, srcs, sim):
    result = []
    order = (["cl", "g++", "clang++", "c++"] if IS_WIN else ["g++", "clang++", "c++"])
    tools = [t for t in order if shutil.which(t)]
    # Sim-enabled variants first (GNU-style compilers only), then ref-only.
    if sim:
        for tool in tools:
            if tool != "cl":
                result.extend(v for v in gnu_variants(tool, out, srcs, sim)
                              if "-DNPU_HAVE_SIM" in v)
    for tool in tools:
        if tool == "cl":
            result.append(["cl", "/nologo", "/O2", "/EHsc", "/LD", "/DNPU_BUILD_DLL"]
                          + srcs + ["/Fe" + out])
        else:
            result.extend(gnu_variants(tool, out, srcs, None))
    return result


def main():
    os.makedirs(BUILD, exist_ok=True)
    out = os.path.join(BUILD, lib_name())
    srcs = sources()
    if not srcs:
        print("error: no C++ sources found")
        return 1
    sim = sim_config()
    cmds = candidates(out, srcs, sim)
    if not cmds:
        print("error: no C++ compiler found (tried cl, g++, clang++, c++)")
        return 1
    for cmd in cmds:
        if os.path.exists(out):
            try:
                os.remove(out)
            except OSError:
                pass
        if run(cmd, BUILD) and os.path.exists(out) and smoke_test(out):
            kind = "ref + sim" if "-DNPU_HAVE_SIM" in cmd else "ref"
            print("built %s (backends: %s)" % (out, kind))
            return 0
        print("  variant failed, trying next")
    print("error: every compiler/flag variant failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
