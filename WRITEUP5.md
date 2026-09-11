# Affine int4 beats NF4 on accuracy at equal latency (Turing)

bitsandbytes' NF4 is the default 4-bit weight format in a large part of the local-LLM
ecosystem, and its selling point is a codebook shaped for normally-distributed weights.
I wrote a plain affine int4 GEMV in Triton to see what that codebook is actually worth,
and found that it depends entirely on block size: NF4 is ahead at 128 weights per
block, the two are level at 64, and below that a distribution-free affine grid wins —
on weight reconstruction across 322 layers, and on end-to-end perplexity for all three
models I tested.

The kernel itself is unremarkable and that is rather the point. What the project is
really about is measurement: seven optimisation ideas that failed, four results that
were wrong before they were right, and a GPU quietly running at 28% of its rated clock
while reporting no fault.

---

## Summary

On an NVIDIA T4, a Triton affine (asymmetric) int4 weight-only GEMV runs at parity
with bitsandbytes' NF4 kernel while reconstructing weights ~5% more accurately at
matched 32-element granularity, and runs 2.0–2.1x faster than cuBLAS fp16. Measured
across every projection weight of Qwen2.5-0.5B and TinyLlama-1.1B, affine int4 wins
on 316 of 322 layers, median ratio ≈0.94x. The advantage survives end-to-end: lower
wikitext-2 perplexity on all three models tested, by 0.37 / 0.15 / 0.03 points.

The advantage depends on granularity and reverses at coarse block sizes: NF4 is better
at 128, the two are equivalent at 64, affine int4 is better at 32. NF4's codebook is
fit to a normal distribution, and small blocks are too small a sample to look normal —
which is where a distribution-free affine grid takes over.

The measurement work turned out to matter more than the kernel work. Four results I
initially believed were wrong — two from badly constructed baselines, one from a GPU
running at 28% of its rated clock, and one from comparing two quantization schemes at
different block sizes. Each would have produced a confidently published false result,
and the last one inflated the headline number by more than a factor of two.

---

## Setup

| | |
|---|---|
| GPU | Tesla T4 (Turing, sm_75), Kaggle notebook |
| Clock under load | 1380 MHz of 1590 MHz max, 53°C, verified stable |
| torch | 2.10.0+cu128 |
| triton | 3.6.0 |
| bitsandbytes | 0.50.2 |
| driver | 580.159.04 |
| Weights tested | Qwen2.5-0.5B (168 weights), TinyLlama-1.1B (154), kurtosis 3.4–300 |
| Perplexity | wikitext-2 test, 2048-token non-overlapping windows, Qwen2.5-0.5B / Qwen2.5-1.5B / TinyLlama-1.1B |

Repository: https://github.com/Asadkhan282/affine-int4-triton

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
under 1%. Something inside `gemv_4bit` is inconsistent at this shape specifically —
possibly a dispatch decision made at runtime, possibly an internal synchronisation.
I did not chase it down, so the 1.32x figure should be treated as unresolved rather
than as a result. The two larger shapes, where NF4 is stable, both show 1.03–1.06x,
and parity is the claim I would defend.

---

## Accuracy

### The comparison has to be at matched granularity

NF4's default block size is 64. My kernel's default group is 32. Comparing those two
directly gives affine int4 a 0.885x RMSE ratio — and that number is meaningless,
because half the difference is granularity rather than format. Twice the metadata
buys accuracy in any scheme.

Sweeping both formats across block sizes on Qwen2.5-0.5B `layers.5.mlp.down_proj`:

| block / group | NF4 RMSE | affine int4 RMSE | ratio |
|---|---|---|---|
| 128 | 0.00169 | 0.00181 | 1.074x |
| 64 | 0.00160 | 0.00161 | 1.009x |
| 32 | 0.00150 | 0.00141 | **0.942x** |

The two formats cross over at 64. NF4 wins at coarse granularity, they are equivalent
at 64, and affine int4 wins at 32. The trend is monotonic across the range with no
reversals. bitsandbytes rejects block sizes below 32, so 32 is the finest point where
a comparison is possible.

**Mechanism.** NF4's codebook is fit to a standard normal distribution, so it pays off
only when each block genuinely looks Gaussian after normalisation. At 128 weights per
block that assumption approximately holds and the nonlinear level spacing helps. At 32
weights a block is too small a sample to resemble any particular distribution, the
assumption breaks down, and a distribution-free affine grid spanning [min, max] wins.
This predicts the advantage should continue growing below 32. bitsandbytes rejects
block sizes under 32, so the prediction cannot be tested against NF4 directly, and it
remains an inference from three points rather than something I have shown.

### Whole-model sweep, matched granularity

Every 2-D projection weight, both formats at 32.

| | Qwen2.5-0.5B | TinyLlama-1.1B |
|---|---|---|
| layers measured | 168 | 154 |
| affine int4 wins | **164 (98%)** | **152 (99%)** |
| median ratio | **0.947x** | **0.932x** |
| mean ratio | 0.950x | 0.935x |
| best / worst | 0.920x / 1.036x | 0.925x / 1.012x |

Combined: **316 of 322 layers across two model families**, median ≈0.94x.

**The six losses share a signature.** All are attention projections, and the two
TinyLlama losses are layer-0 `k_proj` and `q_proj` with kurtosis of 300 and 145 —
far beyond anything in the Qwen set, where the maximum was 42.6. Even there NF4 wins
by only 1%. The practical boundary: affine int4 at group 32 beats NF4 except on
weights with kurtosis in the hundreds, which in practice means a small number of
first-layer attention projections.

**Distribution shape is otherwise not the driver, which surprised me.** Splitting
each model's layers at its median kurtosis gives 0.952x / 0.944x on Qwen and
0.935x / 0.930x on TinyLlama — high half versus low half, essentially identical in
both, and if anything a slight edge to the *low*-kurtosis half. Kurtosis across the
combined set spans 3.4 to 300, so this is not a narrow range failing to reveal an
effect. Tail heaviness matters only at the extreme, and the bulk effect is something
else.

I predicted the opposite twice: first that heavy tails would hurt NF4 in general
(they do not), then that the heavy-tailed losses indicated a general kurtosis effect
(they do not generalise — they are a boundary case). The block-size mechanism above
survives the data; the distributional one does not.

### Metadata cost

Both formats at block 32 carry broadly comparable overhead: affine int4 stores a scale
and a zero-point per group, which is 25% of the packed weight bytes at fp16, while NF4
stores an absmax per block plus its own nested quantization state.

I have not done a byte-for-byte accounting of NF4's metadata, and this is a real gap.
If NF4's total overhead at block 32 turns out to be materially lower than 25%, then
part of the accuracy gap reported below is being bought with memory rather than won by
the format, and the comparison is less clean than it appears. Anyone reproducing this
should measure both before treating the result as settled.

---

## End-to-end perplexity

RMSE is a proxy. This is the question that matters: does better weight reconstruction
produce a better model?

Every eligible `nn.Linear` weight was quantize-dequantized in place, then perplexity
measured on wikitext-2 test with non-overlapping 2048-token windows. Both schemes saw
an identical set of layers — the eligibility test is the intersection of both formats'
constraints, and zero layers were skipped in any run.

| model | fp16 | affine int4 g=32 | NF4 bs=32 | int4 − NF4 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 13.0703 | 14.9539 (+14.4%) | 15.3223 (+17.2%) | **−0.368** |
| Qwen2.5-1.5B | 9.2650 | 10.2147 (+10.3%) | 10.3668 (+11.9%) | **−0.152** |
| TinyLlama-1.1B | 7.9723 | 8.2928 (+4.0%) | 8.3205 (+4.4%) | −0.028 |

Affine int4 is ahead on all three. The RMSE advantage does translate.

### The advantage scales with quantization sensitivity

The three gaps are not equal, and they order perfectly with how much 4-bit hurts each
model:

| degradation under int4 | int4 − NF4 gap |
|---|---|
| 14.4% (Qwen 0.5B) | −0.368 |
| 10.3% (Qwen 1.5B) | −0.152 |
| 4.0% (TinyLlama) | −0.028 |

Where quantization is nearly free, format choice is nearly irrelevant — TinyLlama's
−0.028 is below the noise threshold this evaluation can resolve, so that row is a tie,
not a win. Where quantization does real damage, the format recovers a meaningful
fraction of it.

The mechanism is unsurprising once stated: a format difference can only show up in the
error you are actually incurring. It gives a practitioner a decision rule — if a model
quantizes cleanly, use whatever is convenient; if it does not, choose the format
deliberately.

**Caveats.** Three points is thin for a trend, and the three models differ in
architecture as well as in sensitivity, so sensitivity and architecture are confounded.
A proper test would vary bit width or group size within a single model, moving
sensitivity while holding architecture fixed. I have not run that, so the scaling
relationship should be read as a pattern worth checking rather than an established
one.

### These are not deployment-ready numbers

Both formats here use naive round-to-nearest with no calibration, which is why even the
better of the two costs 14.4% perplexity on Qwen2.5-0.5B. Production methods — GPTQ,
AWQ — use activation statistics to choose quantization parameters and lose far less.

This comparison therefore says something about the two *formats* under identical naive
treatment. It does not say either is ready to deploy, and the degradation figures
should not be read as what 4-bit quantization costs in practice.

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

Four results in this project were wrong before they were right. All four would have
looked entirely publishable.

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
`clocks.sm` **under load** before trusting any latency number. A reading taken after
the workload finishes shows the idle clock and tells you nothing.

**Two quantization schemes were compared at different block sizes.** The first
whole-model sweep put affine int4 at group 32 against NF4 at its default block 64 and
reported 0.885x — a 168/168 sweep, unanimous, and wrong. Half the gap was granularity,
not format. At matched 32 the real figure is 0.947x, and at matched 64 NF4 is very
slightly ahead. This one is the most dangerous of the four because nothing about the
output looks suspicious: the sweep ran cleanly, the statistics were consistent, and
the result was simply comparing two different things. Defaults differing between
libraries is an easy way to produce a confident false result.

---

## Limits of this result

Measured on one GPU architecture (Turing), one batch size (1), and one activation
dtype (fp16). Batch sizes above 1 change the arithmetic intensity and probably the
latency conclusion entirely.

Perplexity was measured on one dataset (wikitext-2) at one sequence length, with no
seed variation and no task-level evaluation. Differences below ~0.05 perplexity should
not be treated as real, which makes the TinyLlama result a tie rather than a win.

The latency numbers come from a shared notebook GPU. They were taken during a window
when the card held 1380 MHz of its 1590 MHz maximum, verified under load, but a
dedicated machine would be better. The accuracy numbers are unaffected by this, being
deterministic.

Only two model families, both small (0.5B and 1.1B). Larger models have different
outlier structure, and the kurtosis-300 boundary case found in TinyLlama's first layer
suggests that structure matters at the extremes.

---

## Reproducing

```bash
git clone https://github.com/Asadkhan282/affine-int4-triton
cd affine-int4-triton
pip install -r requirements.txt

python int4_gemv_asym.py                     # correctness + latency
python sweep_layers.py Qwen/Qwen2.5-0.5B     # per-layer weight accuracy
python perplexity_eval.py Qwen/Qwen2.5-0.5B  # end-to-end perplexity
```

Needs sm_70 or newer — recent PyTorch has dropped Pascal, so the P100 that Kaggle
offers will not run any of this.

Before trusting any latency number you produce, check the clock under load. It is two
lines and it is the difference between a measurement and a random number:

```bash
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu --format=csv
```

## Open questions

Things I would look at next, in roughly the order I think they matter:

- **Does the advantage hold on hardware people deploy on?** Ampere and Ada have tuned
  NF4 paths that Turing does not, and this kernel is arithmetic-bound rather than
  bandwidth-bound, so more memory bandwidth may not help it. The conclusion could
  reverse.
- **Does it survive calibration?** GPTQ and AWQ choose quantization parameters using
  activation statistics. A format advantage under naive round-to-nearest may vanish
  once both formats are calibrated properly.
- **What is NF4's true metadata cost at block 32?** See above — this could narrow the
  gap.
- **Does the sensitivity relationship hold within a single architecture?** Varying bit
  width inside one model would separate sensitivity from architecture.
- **Batch sizes above 1.** Everything here is decode at batch 1. Higher batch changes
  the arithmetic intensity and probably the entire latency picture.
