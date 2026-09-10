# Affine int4 beats NF4 on accuracy at equal latency (Turing)

*Draft. Everything marked TODO must be filled before publishing. Do not publish with
T4-only numbers — see "Before publishing" at the end.*

---

## Summary

On an NVIDIA T4, a Triton affine (asymmetric) int4 weight-only GEMV runs at parity
with bitsandbytes' NF4 kernel while reconstructing weights 12% more accurately, and
runs 2.0–2.1x faster than cuBLAS fp16. NF4's nonlinear codebook does not pay for
itself here: fine-grained affine quantization with a zero-point per 32 weights gets
better accuracy from a cheaper decode path.

The measurement work turned out to matter more than the kernel work. Three of the
numbers I initially believed were wrong — two from badly constructed baselines, one
from a GPU running at 28% of its rated clock — and each would have produced a
confidently published false result.

---

## Setup

| | |
|---|---|
| GPU | Tesla T4 (Turing, sm_75), Kaggle notebook |
| Clock under load | 1380 MHz of 1590 MHz max, 53°C, verified stable |
| torch | TODO — `torch.__version__` |
| triton | TODO — `triton.__version__` |
| bitsandbytes | TODO — `bitsandbytes.__version__` |
| driver / CUDA | TODO — from `nvidia-smi` |
| Weights tested | `randn` synthetic; Qwen2.5-0.5B `layers.5.mlp.down_proj` (896×4864, kurtosis 4.55) |

Repository: TODO

All timings are the median of 5 `triton.testing.do_bench` calls, with min–max shown.
The card was warmed with 200 large matmuls before any timed region.

---

## The kernel

Weight-only int4, batch size 1 (decode regime), fp16 activations, fp32 accumulate.

**Quantization.** Affine per group of G weights along K:

```
scale = (max - min) / 15
zero  = round(-min / scale)
q     = round(w / scale + zero)   clamped to [0, 15]
```

Dequant is `(q - zero) * scale`. Two fp16 values of metadata per group.

**Packing.** Byte *j* of row *n* holds weight *k=j* in the low nibble and *k=j+K/2* in
the high nibble. A contiguous block of bytes therefore decodes into two contiguous
slices of the activation vector, keeping both `x` loads coalesced. Interleaving
adjacent *k* values instead would force strided access on `x`.

**Kernel shape.** One program per output row, looping over K in blocks of `BLOCK_K`,
accumulating into a `[BLOCK_K]` fp32 vector, single tree reduction and store at the
end. Autotuned over `BLOCK_K ∈ {512, 1024, 2048}`, `num_warps ∈ {2, 4, 8}`,
`num_stages ∈ {2, 3, 4}`. Winner at 4096×4096: `BLOCK_K=512, num_warps=2`.

---

## Correctness

Measured as max relative error against an fp32 evaluation of the *same quantized
weights*, so quantization error and kernel error are separated.

| K | N | rel. error |
|---|---|---|
| 4096 | 4096 | 3.62e-04 |
| 4096 | 11008 | 3.25e-04 |
| 11008 | 4096 | 4.11e-04 |

Nine further configurations across GROUP ∈ {32, 64, 128} passed at 3.16e-04 to
4.21e-04.

**A note on tolerances.** The first version of this test compared the kernel against
an eager fp16 reference with a fixed absolute tolerance of 2e-2. Every bfloat16 case
failed. The kernel was correct — the tolerance was simply smaller than one bf16
mantissa step at the magnitudes involved, and the eager reference was itself the less
accurate of the two, since it rounds twice where the kernel rounds once. Comparing two
lossy implementations to each other and applying an absolute tolerance is not a
correctness test. Ground truth must be higher precision than both, and the tolerance
must be relative to the tensor scale.

---

## Latency

4096×4096, batch 1, median of 5:

| implementation | latency | range | vs fp16 |
|---|---|---|---|
| cuBLAS fp16 | 143.2 µs | [143.2–143.4] | 1.00x |
| affine int4, G=32 | 70.5 µs | [70.0–70.8] | **2.03x** |
| bitsandbytes NF4 | 108.2 µs | [73.7–168.1] | 1.32x |

4096×11008 and 11008×4096:

| implementation | 4096×11008 | 11008×4096 |
|---|---|---|
| cuBLAS fp16 | 368.0 µs | 387.4 µs |
| affine int4, G=32 | 179.3 µs | 184.3 µs |
| bitsandbytes NF4 | 190.3 µs | 189.7 µs |
| int4 vs NF4 | 1.06x | 1.03x |

**The honest reading is parity with NF4, not a win.** The 1.32x at 4096×4096 sits on
an NF4 measurement whose own range spans 73.7–168.1 µs — a 2.3x spread within five
samples, while every other measurement in the same loop on the same card was stable to
under 1%. Something inside `gemv_4bit` is inconsistent at that shape. TODO:
investigate and file upstream if reproducible. The two larger shapes, where NF4 is
stable, both show 1.03–1.06x.

---

## Accuracy

RMSE of reconstructed weights vs fp16 original. Real weights, Qwen2.5-0.5B
`down_proj`, kurtosis 4.55. NF4 baseline: 0.00160.

| mode | group | RMSE | vs NF4 | metadata |
|---|---|---|---|---|
| symmetric | 128 | 0.00216 | 1.355x | 3.1% |
| symmetric | 64 | 0.00194 | 1.218x | 6.2% |
| symmetric | 32 | 0.00172 | 1.080x | 12.5% |
| affine | 128 | 0.00181 | 1.134x | 6.2% |
| affine | 64 | 0.00161 | 1.009x | 12.5% |
| **affine** | **32** | **0.00141** | **0.883x** | 25.0% |

Two things worth drawing out.

**Real weights favour NF4 more than Gaussian weights do, not less.** Symmetric int4
at G=128 is 1.275x worse than NF4 on `randn` and 1.355x worse on real weights. I had
predicted the opposite — that heavy tails would hurt NF4's codebook. The mechanism
runs the other way: real weight distributions are more peaked than Gaussian, so more
mass sits where NF4's nonlinear levels are dense, while a uniform grid has its range
stretched by whichever outlier is largest in each group.

**The zero-point does most of the work.** Going symmetric → affine at fixed group size
buys more than quartering the group size does at fixed mode.

---

## The accuracy/latency curve

Latency measured at 4096×4096 with clocks verified stable; RMSE on synthetic weights.

| group | latency | GB/s | metadata | RMSE |
|---|---|---|---|---|
| 256 | 63.3 µs | 137 | 3.1% | 0.10937 |
| 128 | 63.6 µs | 140 | 6.2% | 0.10056 |
| 64 | 65.3 µs | 145 | 12.5% | 0.09105 |
| 32 | 69.8 µs | 150 | 25.0% | 0.08077 |

8x more metadata costs 10% latency and buys 26% better reconstruction. **G=32 is the
recommended operating point** — the fine-grained configuration is nearly free.

Note what this table also says: achieved bandwidth *rises* as metadata traffic
increases. A bandwidth-bound kernel would show the opposite. See below.

---

## Where the remaining time goes

cuBLAS reaches 235 GB/s on this card for a single-pass read. This kernel achieves
137–150 GB/s. Isolating the phases at BLOCK_K=128, GROUP=32:

| variant | latency |
|---|---|
| loads only, no dequant, vector store | 61.8 µs |
| loads + dequant, vector store | 99.9 µs |
| full kernel (dequant + tree reduction) | 90.6 µs |

The arithmetic costs ~38 µs. The tree reduction costs nothing — the full kernel is
*faster* than the vector-store variant because it writes 4 KB instead of 2 MB.

So this kernel is compute-bound on dequantization, not bandwidth-bound, which is
unusual for a weight-only GEMV and explains why extra metadata traffic is nearly free.

---

## What did not work

Seven hypotheses tested, two confirmed. Each of these was a plausible mechanism that
measurement rejected.

**Single-pass restructuring of the fused RMSNorm kernel** (1.01x). The second-pass
reload was already being served from L2 — a 4 MB cache and an 8 KB row, so it never
reached DRAM. Removing a read that costs nothing saves nothing.

**CUDA graph capture of one kernel** (1.15x). A graph amortises launch cost across
everything inside it; with a single kernel captured, it saves exactly one launch.
Graphs pay off across a whole decode step, not one operator.

**Rows-per-block** (0.96x, monotonically worse from 1 to 8 rows). Fewer, longer-lived
blocks means fewer independent memory requests in flight. A bandwidth-bound kernel
needs concurrency, not larger work units.

**Larger BLOCK_K** (worse at every warp count; 2048 was the worst config tested).
Same reason.

**Split-K with atomic combine** (0.87x at SPLIT=1, degrading to 0.24x at SPLIT=8).
The atomics serialise across programs sharing an output address, and even SPLIT=1 loses
because an atomic replaced a plain store.

**Group-hoisted dequant** (0.20x at G=32, 0.40x at G=64, 0.69x at G=128). The
reassociation is algebraically sound — `Σ(q−z)·s·x = s·(Σqx − z·Σx)` removes most of
the per-element multiplies. But implementing it required iterating group-by-group,
which narrows the inner vector to GROUP_SIZE lanes. The penalty scaling exactly with
group size is the signature. The redundant multiplies were cheaper than the lost
vector width.

**Reducing scale/zero-point loads.** At G=32 the kernel issues 32 loads per distinct
scale. These hit L1 and are not the bottleneck; the 38 µs is arithmetic, and no
formulation found removes it while keeping wide vectors.

---

## Measurement notes

Three results in this project were wrong before they were right. All three would have
been publishable-looking.

**A "floor" measured with atomics came out above the ceiling it bounded.** A
load-only kernel using `tl.atomic_add` to consume its accumulator measured 52.7 µs,
while the full kernel doing strictly more work measured 49.9 µs. A lower bound that
exceeds the measured value is obviously wrong; a lower bound wrong by a mere 2x looks
entirely plausible and would have sent the analysis in the wrong direction. Sanity-check
benchmarks against each other, not just individually.

**A uint8 reduction measured 19 GB/s and was mistaken for memory bandwidth.**
`q.sum(dtype=torch.float32)` on a uint8 tensor goes through a slow PyTorch path. It was
measuring the reduction, not the memory system. A Triton streaming kernel on the same
data gave 159–234 GB/s.

**The GPU was running at 450 MHz of 1590 MHz.** Identical code measured 66.9, 76.2,
and 177.4 µs across runs. No throttle reason was reported by `nvidia-smi` — all flags
read "Not Active" — but the card was at 82°C and clamped. The notebook had a second,
idle T4; pinning to it via `CUDA_VISIBLE_DEVICES=1` gave 1380 MHz under sustained load
and timings stable to under 1%. On shared or virtualised hardware, verify
`clocks.sm` under load before trusting any latency number.

---

## Limits of this result

Measured on one GPU architecture (Turing), one batch size (1), one activation dtype
(fp16), and — for the real-weight accuracy numbers — one layer of one model. Batch
sizes above 1 change the arithmetic intensity and probably the conclusion. No
end-to-end model evaluation was run: RMSE on weights is a proxy for quality, not a
substitute for perplexity or task accuracy.

---

## Before publishing

- [ ] Re-run everything in one clean session so every number comes from one run
- [ ] Fill in all TODO version strings
- [ ] Sweep several layers and both attention and MLP projections — the accuracy claim
      currently rests on a single matrix
- [ ] Re-run on an A100 or L4. Nobody serves production on a T4, and a reviewer will
      say so. Two hours of rented time.
- [ ] Add a perplexity comparison on a real model, int4-G32 vs NF4
- [ ] Investigate the NF4 instability at 4096×4096; file upstream if reproducible
- [ ] Publish the repo with exact reproduction commands
