# Reduced-Precision Belief Propagation for Embedded Systems on a PYNQ-Z2 Board

**Description:**
The following project evaluates reduced and further reduced precision on an ARM based PYNQ-Z2 Board.

Four base algorithms are tested; with Benchmark Belief Propagation algorithms taken from https://github.com/mvandermerwe/BP-GPU-Message-Scheduling
The BP algorithms tested throughout this project include Loopy, Residual, Random & Variable Elimination Belief Propagation. The RnBP (Random) BP algorithm presented is a converted serial version from the original CUDA implementation by Mark Van der Merwe.

The four numerical precisions covered in this project include:Double, Single, Reduced (SDF16), and Further Reduced (SDF8) - Based on  A. S. Molahosseini, J. Lee, and H. Vandierendonck, “Software-Defined Number Formats for High Speed Belief Propagation,” IEEE Transactions on Emerging Topics in Computing, vol. 13, no. 3, pp.
853–865, Jul. 2025, doi: https://doi.org/10.1109/tetc.2025.3528972
This work explores optimal convergence thresholds, speed‑ups, memory savings, and scaling behaviour on embedded ARM hardware, whislt providing a novel variation of serial RnBP.

**CODE TAKEN IS STORED WITHIN THE BENCHMARK_BP FILE.**
**CODE CREATED BY ME IS STORED WITHIN EXTENDED_BP**

**Dependencies:**

-Hardware: PYNQ‑Z2 board (ARM Cortex‑A9, 512 MB RAM)
-OS: Linux (PYNQ image)
-Compiler: GCC 11+ with -std=c++14 -O2
-Libraries: Boost (fibonacci heap for Residual BP), standard C++ libraries

**Building Executables:**
Use the provided Makefile.serial with the INFER flag to select the desired algorithm:

BASH:

**Loopy BP double precision:**
make -f Makefile.serial INFER=loopy SRC=main.cpp
mv main.out loopy_double
**Single precision (Loopy BP):**
make -f Makefile.serial INFER=single SRC=main.cpp
mv main.out loopy_single
**SDF16‑bit Reduced (Loopy BP):**
make -f Makefile.serial INFER=redprec SRC=main.cpp
mv main.out loopy_redprec
**SDF8‑bit Further Reduced (Loopy BP):**
make -f Makefile.serial INFER=furedprec SRC=main.cpp
mv main.out loopy_furedprec
**Residual BP (double):**
make -f Makefile.serial INFER=rbp SRC=main.cpp
mv main.out rbp_double
**Random BP (double):**
make -f Makefile.serial INFER=rnbp SRC=main.cpp
mv main.out rnbp_double
**Variable Elimination:**
make -f Makefile.serial INFER=ve SRC=main.cpp
mv main.out ve_bp
clean with: make -f Makefile.serial clean

**Testing:**
ALL Executables accept the same CLI format:

-./ <graph_file> <timeout_seconds> <threshold_epsilon> [extra_params]

For Example:


-Loopy BP on a 50-node graph with ε= 0.01 & timeout_seconds = 60s: ./loopy_double ising_50_0.txt 60 0.01


-Random BP can have specified high/low parallelism. This is passed as two values a,b where LowP=a/10 and HighP=b/10. Thus to run with 0.3 low parallelism and 0.8 high parallelism, one may run: ./main.out <test_file> <timeout> 3 8


-Variable Elimination has no threshold effect but CLI arguments are required e.g.: ./ve_bp ising_50_0.txt 60 0.01


The repository includes bash scripts to run large experiment batches:


-run_threshold_experiments.sh – tests all 8 thresholds on graphs up to 150 nodes


-final_comparison.sh – compares all algorithms/precisions on the same graph


-measure_memory.sh – measures peak memory usage for each executable


To make scripts executable, run:
chmod +x scripts/*.sh
./scripts/final_comparison.sh

**Memory Measurement:**
Memory is measured by reading the process’s resident set size (RSS) from /proc/self/statm.
The value is multiplied by the ARM page size (4 KB) and printed as Memory: X KB after inference.

**Citation:**
If you use the following code or results, please cite:
[1] A. Qasim, “Reduced-Precision Belief Propagation for Embedded Systems on PYNQ‑Z2 Board”, CSC3002 Dissertation, Queen’s University Belfast, 2026.
[2] M. Van der Merwe, “BP‑GPU‑Message‑Scheduling”, GitHub, 2019.
[3] A. S. Molahosseini, J. Lee, H. Vandierendonck, “Software‑Defined Number Formats for High‑Speed Belief Propagation”, IEEE TETC, vol. 13, no. 3, pp. 853‑865, 2025.
