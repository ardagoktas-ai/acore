# Chip v2.0 micro-architecture

Status: RTL plus testbench; simulated with Icarus only. Results of the golden-model comparison: [TODO: record verifier result]. Not synthesized, no timing or area numbers yet.

## Block diagram

    host port --> [R bank x8] [CONT bank x8] [DST bank x8] [V bank0|bank1] [IMEM]
                       \           |             |            ^    |
                        \          |             |            |    v
    sequencer (PC, LC, N, A, VSEL) --> row address i           |
                                                               |
                 8 lanes x 5-stage pipeline --> max tree ------+

Each lane a (0..7) owns 64-word R, CONT and DST banks holding action a of every state. The host writes element (i,a) at offset i*8 + a.

## Pipeline

    cycle:    t       t+1          t+2          t+3            t+4
            +------+ +----------+ +----------+ +------------+ +------------------+
    state i | S0   | | S1       | | S2       | | S3         | | S4               |
            | i -> | | 8 lanes: | | 8 lanes: | | 8 lanes:   | | mask lanes a>=A  |
            | bank | | R,CONT,  | | V[DST]   | | q = R +    | | (copy lane 0),   |
            | addr | | DST out; | | out;     | | prod[47:16]| | 3-level max tree,|
            |      | | V gather | | 32x32->64| |            | | write Vnext[i]   |
            |      | | issued   | | multiply | |            | |                  |
            +------+ +----------+ +----------+ +------------+ +------------------+
    state i+1        S0           S1           S2             S3 ...

- One state per cycle; gather, multiply, add and max of different states overlap, and all A actions of a state run side by side.
- No data hazards: SWEEP reads bank VSEL and writes bank !VSEL (Jacobi update, matching the golden model), and banks swap only after the pipeline has drained.
- `>>> 16` followed by truncation to 32 bits equals product bits [47:16], so the shifter is pure wiring.
- Inactive lanes (a >= A) are replaced by lane 0's value before the max tree. Lane 0 is always active (A >= 1), so no "minus infinity" constant is needed and the result is unchanged.

## Area / timing notes

- Memories are behavioural arrays. For an ASIC flow the 24 lane banks (64 x 32 each) could become small SRAM macros or latch arrays: [TODO: choose sky130 memory implementation].
- V (2 x 64 words) needs 8 gather read ports plus one host read port and one write port, so it must be a flop register file with read muxes; this is the main area cost of the parallel version besides the multipliers.
- 8 single-cycle 32x32 signed multipliers dominate area; the critical path is expected in S2 (multiplier) or S4 (three signed comparators in series plus the write). Options if timing fails: split the multiplier into two pipelined partial-product stages, and register the max tree after the first level. Each adds one cycle of latency per SWEEP, not per state. Achieved frequency and area: [TODO: after OpenLane run].
- A smaller variant could use L < 8 lanes and ceil(A/L) passes per state. [TODO: evaluate area/cycle trade-off after first synthesis].
- Only 6 bits of DST are used; DST >= N is undefined behaviour by contract.
