// kernels_sim.cpp - "sim" backend: drives the Verilated npu_top (Chip v2.0) cycle by
// cycle through the host protocol and ISA defined in stage2_chip/docs/ISA.md.
// Compiled only by build.py when verilator is on PATH and ../../stage2_chip/rtl exists.
// Status: not yet built on a machine with Verilator (see docs/DRIVER.md).
#ifndef NPU_BUILD_DLL
#define NPU_BUILD_DLL
#endif
#include "npu_kernels.h"

#include "Vnpu_top.h"
#include "verilated.h"

// Verilator 4.x needs this symbol from the harness; Verilator 5 does not define it.
double sc_time_stamp() { return 0.0; }

namespace {

Vnpu_top* g_top = 0;
const uint64_t kTimeoutCycles = 100000u;

// Inputs are changed while clk is low (like the testbench's negedge), then one
// rising edge samples them.
void tick() {
    g_top->clk = 1;
    g_top->eval();
    g_top->clk = 0;
    g_top->eval();
}

void hostWrite(uint32_t addr, int32_t data) {
    g_top->host_we = 1;
    g_top->host_addr = addr & 0xFFFu;
    g_top->host_wdata = static_cast<uint32_t>(data);
    tick();
}

npuError_t simBellman(const npu::BellmanArgs& g, uint64_t* cycles) {
    const int n = g.n;
    const int A = g.A;
    if (n < 1 || n > 64 || A < 1 || A > 8) return npuErrorUnsupportedSize;

    if (!g_top) {
        g_top = new Vnpu_top;
        g_top->clk = 0;
        g_top->host_we = 0;
        g_top->host_addr = 0;
        g_top->host_wdata = 0;
        g_top->host_raddr = 0;
        g_top->start = 0;
        g_top->eval();
    }

    // Synchronous reset for every launch: VSEL = 0, sequencer idle.
    g_top->rst = 1;
    g_top->start = 0;
    g_top->host_we = 0;
    for (int k = 0; k < 3; ++k) tick();
    g_top->rst = 0;

    // Problem data: element (i,a) at offset i*8 + a (lane a, word i).
    for (int i = 0; i < n; ++i) {
        for (int a = 0; a < A; ++a) {
            const int k = i * A + a;
            const uint32_t off = static_cast<uint32_t>(i * 8 + a);
            hostWrite(0x000u + off, g.R[k]);
            hostWrite(0x200u + off, g.C[k]);
            hostWrite(0x400u + off, g.D[k]);
        }
    }
    for (int i = 0; i < n; ++i) hostWrite(0x600u + static_cast<uint32_t>(i), g.Vin[i]);

    // Program: CFG n,A ; SWEEP ; HALT
    hostWrite(0x800u, static_cast<int32_t>(0x10000000u | (static_cast<uint32_t>(A) << 8) |
                                           static_cast<uint32_t>(n)));
    hostWrite(0x801u, static_cast<int32_t>(0x20000000u));
    hostWrite(0x802u, 0);
    g_top->host_we = 0;

    // Start pulse, then count cycles until done (same counting as tb_npu.v).
    g_top->start = 1;
    tick();
    g_top->start = 0;
    uint64_t cyc = 1;
    while (!g_top->done) {
        tick();
        if (++cyc > kTimeoutCycles) return npuErrorBackend;
    }

    // Registered read of the current V bank (1-cycle latency).
    for (int i = 0; i < n; ++i) {
        g_top->host_raddr = static_cast<uint32_t>(i);
        tick();
        g.Vout[i] = static_cast<int32_t>(static_cast<uint32_t>(g_top->host_rdata));
    }

    if (cycles) *cycles = cyc;
    return npuSuccess;
}

const npu::Backend kSim = {"sim", 64, 8, simBellman};

}  // namespace

namespace npu {
const Backend* simBackend() { return &kSim; }
}  // namespace npu
