// npu_top - Bellman-backup NPU (Chip v2.0), 8-lane version
//
// Computes, for one SWEEP instruction:
//   Vout[i] = max_a ( R[i,a] + ((CONT[i,a] * Vin[DST[i,a]]) >>> 16) )
// with Q16.16 operands, a signed 64-bit product, floor shift and a 32-bit sum.
// All A actions of one state are processed in parallel (one lane per action,
// up to 8 lanes), so one state enters the pipeline every cycle.
//
// Interface:
//   host_we/host_addr/host_wdata : write port into the local memories (only while idle)
//   host_raddr -> host_rdata     : registered read of the current V bank (1-cycle latency)
//   start (pulse)                : run the program in IMEM from address 0
//   busy / done                  : status; done stays high after HALT until the next start
//
// Host memory map (host_addr[11:0]); the element (i,a) lives at offset i*8 + a:
//   0x000-0x1FF  R    [i*8+a]  -> lane a, word i
//   0x200-0x3FF  CONT [i*8+a]
//   0x400-0x5FF  DST  [i*8+a]  (low 6 bits used)
//   0x600-0x63F  V    [i]      (current bank)
//   0x800-0x80F  IMEM [pc]
//
// Pipeline (one state i per cycle, 8 lanes wide):
//   S0 issue i -> per-lane R/CONT/DST read   S1 per-lane V[DST] gather
//   S2 8 x 64-bit multiply                   S3 shift + add
//   S4 masked max tree over lanes, write Vout[i]
//
// Synthesizable Verilog-2005, no simulation-only constructs.

module npu_top (
    input  wire        clk,
    input  wire        rst,          // synchronous, active high
    input  wire        host_we,
    input  wire [11:0] host_addr,
    input  wire [31:0] host_wdata,
    input  wire [5:0]  host_raddr,
    output reg  [31:0] host_rdata,
    input  wire        start,
    output reg         busy,
    output reg         done
);

    localparam [3:0] OP_HALT  = 4'h0;
    localparam [3:0] OP_CFG   = 4'h1;   // n = imm[6:0], A = imm[11:8]
    localparam [3:0] OP_SWEEP = 4'h2;   // one Bellman backup, then swap V banks
    localparam [3:0] OP_LOOP  = 4'h3;   // LC = imm[15:0]
    localparam [3:0] OP_DJNZ  = 4'h4;   // LC = LC-1; if LC != 0 then PC = imm[3:0]

    localparam [1:0] S_IDLE  = 2'd0;
    localparam [1:0] S_RUN   = 2'd1;
    localparam [1:0] S_SWEEP = 2'd2;
    localparam [1:0] S_DRAIN = 2'd3;

    // ------------------------------------------------------------------
    // Shared memories
    // ------------------------------------------------------------------
    reg [31:0] vmem [0:127];   // two banks of 64: {bank, index}
    reg [31:0] imem [0:15];

    // ------------------------------------------------------------------
    // Architectural / control registers
    // ------------------------------------------------------------------
    reg [1:0]  state;
    reg [3:0]  pc;
    reg [15:0] lc;
    reg [6:0]  nreg;
    reg [3:0]  areg;
    reg        vsel;            // current V bank
    reg [5:0]  i_cnt;           // state index being issued

    wire [31:0] ir     = imem[pc];
    wire [3:0]  opcode = ir[31:28];
    wire [15:0] lc_dec = lc - 16'd1;

    wire host_ok = host_we & ~busy;
    wire [2:0] region = host_addr[11:9];

    wire last_i = ({1'b0, i_cnt} == (nreg - 7'd1));
    wire issue  = (state == S_SWEEP);

    // ------------------------------------------------------------------
    // Pipeline tags
    // ------------------------------------------------------------------
    reg        p1_v, p2_v, p3_v, p4_v;
    reg [5:0]  p1_i, p2_i, p3_i, p4_i;

    function [31:0] smax;
        input [31:0] x;
        input [31:0] y;
        begin
            smax = ($signed(x) > $signed(y)) ? x : y;
        end
    endfunction

    // ------------------------------------------------------------------
    // Action lanes
    // ------------------------------------------------------------------
    wire [255:0] q_flat;   // S4 value of each lane
    wire [255:0] m_flat;   // masked: lanes a >= A repeat lane 0

    genvar g;
    generate
        for (g = 0; g < 8; g = g + 1) begin : lane
            reg [31:0] rm [0:63];
            reg [31:0] cm [0:63];
            reg [31:0] dm [0:63];
            reg [31:0] r1, c1, d1;     // S1
            reg [31:0] r2, c2, v2;     // S2
            reg [31:0] r3;             // S3
            reg [63:0] p3;             // S3
            reg [31:0] q4;             // S4

            wire wr = host_ok && (host_addr[2:0] == g);

            wire signed [63:0] c_ext = {{32{c2[31]}}, c2};
            wire signed [63:0] v_ext = {{32{v2[31]}}, v2};
            wire signed [63:0] prod  = c_ext * v_ext;

            always @(posedge clk) begin
                if (wr && region == 3'd0) rm[host_addr[8:3]] <= host_wdata;
                if (wr && region == 3'd1) cm[host_addr[8:3]] <= host_wdata;
                if (wr && region == 3'd2) dm[host_addr[8:3]] <= host_wdata;
            end

            always @(posedge clk) begin
                // S0 -> S1: row read
                r1 <= rm[i_cnt];
                c1 <= cm[i_cnt];
                d1 <= dm[i_cnt];
                // S1 -> S2: gather from the current V bank
                r2 <= r1;
                c2 <= c1;
                v2 <= vmem[{vsel, d1[5:0]}];
                // S2 -> S3: multiply
                r3 <= r2;
                p3 <= prod;
                // S3 -> S4: >>> 16 then truncate = bits [47:16]; 32-bit add
                q4 <= r3 + p3[47:16];
            end

            assign q_flat[32*g +: 32] = q4;
            assign m_flat[32*g +: 32] = (areg > g) ? q4 : q_flat[31:0];
        end
    endgenerate

    // S4: max tree (lane 0 is always valid because A >= 1)
    wire [31:0] m01 = smax(m_flat[ 31:  0], m_flat[ 63: 32]);
    wire [31:0] m23 = smax(m_flat[ 95: 64], m_flat[127: 96]);
    wire [31:0] m45 = smax(m_flat[159:128], m_flat[191:160]);
    wire [31:0] m67 = smax(m_flat[223:192], m_flat[255:224]);
    wire [31:0] m03 = smax(m01, m23);
    wire [31:0] m47 = smax(m45, m67);
    wire [31:0] best = smax(m03, m47);

    wire host_v_we = host_ok & (region == 3'd3);

    // ------------------------------------------------------------------
    // Shared memory ports
    // ------------------------------------------------------------------
    always @(posedge clk) if (host_ok && region == 3'd4) imem[host_addr[3:0]] <= host_wdata;

    always @(posedge clk) begin
        if (p4_v)
            vmem[{~vsel, p4_i}] <= best;
        else if (host_v_we)
            vmem[{vsel, host_addr[5:0]}] <= host_wdata;
    end

    always @(posedge clk) host_rdata <= vmem[{vsel, host_raddr}];

    // ------------------------------------------------------------------
    // Tag pipeline
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            p1_v <= 1'b0; p2_v <= 1'b0; p3_v <= 1'b0; p4_v <= 1'b0;
            p1_i <= 6'd0; p2_i <= 6'd0; p3_i <= 6'd0; p4_i <= 6'd0;
        end else begin
            p1_v <= issue; p1_i <= i_cnt;
            p2_v <= p1_v;  p2_i <= p1_i;
            p3_v <= p2_v;  p3_i <= p2_i;
            p4_v <= p3_v;  p4_i <= p3_i;
        end
    end

    // ------------------------------------------------------------------
    // Sequencer
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            pc    <= 4'd0;
            lc    <= 16'd0;
            nreg  <= 7'd1;
            areg  <= 4'd1;
            vsel  <= 1'b0;
            i_cnt <= 6'd0;
            busy  <= 1'b0;
            done  <= 1'b0;
        end else begin
            case (state)
            S_IDLE: begin
                if (start) begin
                    pc    <= 4'd0;
                    busy  <= 1'b1;
                    done  <= 1'b0;
                    state <= S_RUN;
                end
            end
            S_RUN: begin
                case (opcode)
                OP_HALT: begin
                    busy  <= 1'b0;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                OP_CFG: begin
                    nreg <= ir[6:0];
                    areg <= ir[11:8];
                    pc   <= pc + 4'd1;
                end
                OP_SWEEP: begin
                    i_cnt <= 6'd0;
                    state <= S_SWEEP;
                end
                OP_LOOP: begin
                    lc <= ir[15:0];
                    pc <= pc + 4'd1;
                end
                OP_DJNZ: begin
                    if (lc != 16'd0) lc <= lc_dec;
                    if (lc != 16'd0 && lc_dec != 16'd0)
                        pc <= ir[3:0];
                    else
                        pc <= pc + 4'd1;
                end
                default: pc <= pc + 4'd1;   // reserved opcodes act as NOP
                endcase
            end
            S_SWEEP: begin
                if (last_i)
                    state <= S_DRAIN;
                else
                    i_cnt <= i_cnt + 6'd1;
            end
            S_DRAIN: begin
                if (!p1_v && !p2_v && !p3_v && !p4_v) begin
                    vsel  <= ~vsel;
                    pc    <= pc + 4'd1;
                    state <= S_RUN;
                end
            end
            endcase
        end
    end

endmodule
