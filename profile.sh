#!/usr/bin/env bash
# Nsight invocations that work on an LLM decode loop.
#
# The mistake everyone makes: profiling the whole script. Warmup, compilation and
# weight loading swamp the trace and Nsight Compute takes 40 minutes replaying kernels
# you do not care about. Restrict the capture range instead.
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUT="${OUT:-profiles}"
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# 1. Nsight Systems: timeline. Answers "where does wall time go, and is the GPU
#    actually busy?" Start here, always.
#
#    Gaps between kernels on the CUDA row = CPU launch bound. That is a completely
#    different fix from a slow kernel, and you cannot tell them apart without this.
# ---------------------------------------------------------------------------
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=cpu \
  --cuda-memory-usage=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  -o "$OUT/timeline" \
  python bench/harness.py --model "$MODEL" --batch-sizes 1 \
     --prompt-len 512 --gen-len 32 --iters 1 --out /tmp/prof.json

# Stats without opening the GUI:
nsys stats --report cuda_gpu_kern_sum "$OUT/timeline.nsys-rep" | head -40

# ---------------------------------------------------------------------------
# 2. Nsight Compute: per-kernel counters. Only after the timeline says a specific
#    kernel is the problem. Use -k to name it, or ncu will replay everything.
#
#    The section you want first is SpeedOfLight: it gives compute vs memory
#    utilisation as % of peak, which is the roofline answer per kernel.
# ---------------------------------------------------------------------------
ncu \
  --set full \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section LaunchStats \
  --launch-skip 200 \
  --launch-count 20 \
  --target-processes all \
  -o "$OUT/kernels" \
  --force-overwrite \
  python kernels/fused_rmsnorm.py

ncu --import "$OUT/kernels.ncu-rep" --page details | head -60

# ---------------------------------------------------------------------------
# Reading it, in order:
#   1. Is the GPU idle between kernels?        -> launch overhead: CUDA graphs, compile
#   2. Memory % of peak high, compute low?     -> bandwidth-bound: fuse, quantise
#   3. Both low?                               -> occupancy/stalls: check LaunchStats
#   4. Compute high?                           -> you are done, this kernel is fine
#
# To make step 1 readable, wrap regions in NVTX in your own code:
#   with torch.cuda.nvtx.range("decode_step"): ...
# and call torch.cuda.profiler.start()/stop() around the measured section so the
# --capture-range flag above has something to latch onto.
# ---------------------------------------------------------------------------
