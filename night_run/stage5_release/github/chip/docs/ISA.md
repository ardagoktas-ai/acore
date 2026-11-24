# NPU ISA (Chip v2.0)

My own instruction set for the Bellman backup that Acore's value iteration spends its time in:

    V'[i] = max_a ( R[i*A+a] + ((CONT[i*A+a] * V[DST[i*A+a]]) >>> 16) )

R, CONT and V are Q16.16 (int32). The product is a signed 64-bit product, `>>>` is an arithmetic (floor) shift, and the sum wraps in 32 bits.

## Registers

| Name | Width | Meaning |
|------|-------|---------|
| PC   | 4     | Program counter into IMEM (16 words) |
| LC   | 16    | Loop counter (LOOP / DJNZ) |
| N    | 7     | Number of states, 1..64 |
| A    | 4     | Number of actions, 1..8 |
| VSEL | 1     | Current V bank (0/1); SWEEP reads VSEL, writes !VSEL, then toggles |

## Instruction encoding (32-bit word)

| Opcode [31:28] | Mnemonic | Operand bits | Semantics | Cycles |
|----|----------|--------------|-----------|--------|
| 0x0 | HALT     | -                        | stop, raise `done` | 1 |
| 0x1 | CFG      | [6:0]=N, [11:8]=A        | N, A <- imm | 1 |
| 0x2 | SWEEP    | -                        | one full Bellman backup Vcur -> Vnext, then swap banks | N + 6 |
| 0x3 | LOOP     | [15:0]=count             | LC <- count | 1 |
| 0x4 | DJNZ     | [3:0]=target             | LC <- LC-1; if LC != 0, PC <- target | 1 |
| 0x5-0xF | (reserved) | -                  | executes as NOP | 1 |

Encodings: `HALT = 0x00000000`, `CFG = 0x10000000 | A<<8 | N`, `SWEEP = 0x20000000`, `LOOP = 0x30000000 | count`, `DJNZ = 0x40000000 | target`.

## Memory map (host port, `host_addr[11:0]`, writes accepted only while not busy)

| Range | Memory | Index |
|-------|--------|-------|
| 0x000-0x1FF | R banks    | i*8 + a (lane a, word i) |
| 0x200-0x3FF | CONT banks | i*8 + a |
| 0x400-0x5FF | DST banks  | i*8 + a (low 6 bits used; must be < N) |
| 0x600-0x63F | V, current bank | i |
| 0x800-0x80F | IMEM | instruction address |

The stride is always 8 regardless of A, so every action lives in its own bank (lane) and the host needs no division by A. Words for actions a >= A are ignored by the datapath.

Results are read through `host_raddr[5:0]` -> `host_rdata` (current V bank, one cycle latency). After a SWEEP the current bank holds the new V.

## Host protocol

1. Write R, CONT, DST, V and the program through `host_we/host_addr/host_wdata`.
2. Pulse `start` for one cycle. `busy` goes high.
3. Wait for `done`. Read V through `host_raddr`/`host_rdata`.

## Example programs

One backup (used by the testbench):

    0: CFG   N, A
    1: SWEEP
    2: HALT

K value-iteration sweeps (Jacobi style, ping-pong banks):

    0: CFG   N, A
    1: LOOP  K
    2: SWEEP
    3: DJNZ  2
    4: HALT

Convergence checking (the `1e-3` test in Acore) is not in hardware yet: [TODO: max-|delta| reduction and a conditional branch on it].

## Cycle model

- SWEEP: one state i (all A actions in parallel, 8 lanes) enters the pipeline every cycle, so N issue cycles, plus 1 decode cycle and a 5-cycle drain before the bank swap: N + 6.
- Single-backup program total is N + 9 cycles from `start` to `done` (CFG, SWEEP, HALT plus the start cycle), derived from the RTL structure and independent of A. Measured value: [TODO: record the CYCLES line from the Icarus run].
- Design decision: the earlier serial version needed N*A + 9; the 8-lane version trades 8 multipliers and an 8-read-port V file for an A-fold cycle reduction.
