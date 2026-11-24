// tb_npu - loads vectors.hex into npu_top through its host port and ISA,
// runs one SWEEP, writes output.hex and prints the cycle count.
`timescale 1ns/1ps

module tb_npu;

    reg         clk = 1'b0;
    reg         rst = 1'b1;
    reg         host_we = 1'b0;
    reg  [11:0] host_addr = 12'd0;
    reg  [31:0] host_wdata = 32'd0;
    reg  [5:0]  host_raddr = 6'd0;
    wire [31:0] host_rdata;
    reg         start = 1'b0;
    wire        busy;
    wire        done;

    npu_top dut (
        .clk(clk), .rst(rst),
        .host_we(host_we), .host_addr(host_addr), .host_wdata(host_wdata),
        .host_raddr(host_raddr), .host_rdata(host_rdata),
        .start(start), .busy(busy), .done(done)
    );

    always #5 clk = ~clk;

    reg [31:0] vec [0:4095];
    integer n, A, nA, j, si, sa, fd, cyc;
    integer off_r, off_c, off_d, off_v;

    task hw(input [11:0] ad, input [31:0] d);
        begin
            @(negedge clk);
            host_we    = 1'b1;
            host_addr  = ad;
            host_wdata = d;
        end
    endtask

    initial begin
        $readmemh("vectors.hex", vec);
        n  = vec[0];
        A  = vec[1];
        if (n < 1 || n > 64 || A < 1 || A > 8) begin
            $display("ERROR: unsupported n=%0d A=%0d", n, A);
            $finish;
        end
        nA    = n * A;
        off_r = 2;
        off_c = 2 + nA;
        off_d = 2 + 2 * nA;
        off_v = 2 + 3 * nA;

        repeat (3) @(posedge clk);
        @(negedge clk) rst = 1'b0;

        // Problem data
        // Element (i,a) goes to host offset i*8 + a (lane a, word i)
        for (si = 0; si < n; si = si + 1)
            for (sa = 0; sa < A; sa = sa + 1) begin
                hw(12'h000 + si * 8 + sa, vec[off_r + si * A + sa]);
                hw(12'h200 + si * 8 + sa, vec[off_c + si * A + sa]);
                hw(12'h400 + si * 8 + sa, vec[off_d + si * A + sa]);
            end
        for (j = 0; j < n;  j = j + 1) hw(12'h600 + j, vec[off_v + j]);

        // Program: CFG n,A ; SWEEP ; HALT
        hw(12'h800, 32'h1000_0000 | (A << 8) | n);
        hw(12'h801, 32'h2000_0000);
        hw(12'h802, 32'h0000_0000);
        @(negedge clk) host_we = 1'b0;

        // Start and count cycles until done
        start = 1'b1;
        @(negedge clk) start = 1'b0;
        cyc = 1;
        while (!done) begin
            @(negedge clk);
            cyc = cyc + 1;
            if (cyc > 100000) begin
                $display("ERROR: timeout");
                $finish;
            end
        end

        // Read back result (1-cycle registered read)
        fd = $fopen("output.hex", "w");
        for (j = 0; j < n; j = j + 1) begin
            host_raddr = j;
            @(negedge clk);
            $fdisplay(fd, "%08x", host_rdata);
        end
        $fclose(fd);

        $display("CYCLES: %0d", cyc);
        $finish;
    end

endmodule
