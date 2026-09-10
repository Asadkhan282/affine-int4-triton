# affine-int4-triton

A Triton implementation of affine (asymmetric) int4 weight-only GEMV for LLM decode,
benchmarked against bitsandbytes NF4 and cuBLAS fp16 on Turing.

**Result:** at matched 32-element granularity, affine int4 reconstructs weights ~5%
more accurately than NF4 — 316 of 322 layers across two model families — while running
at latency parity with NF4 and 2.0–2.1x faster than cuBLAS fp16.

The advantage depends on block size and reverses at coarse granularity: NF4 is better
at 128, the two tie at 64, affine int4 wins at 32. See
[WRITEUP_DRAFT.md](WRITEUP_DRAFT.md) for the numbers, the mechanism, the seven
optimisation attempts that failed, and four measurement errors caught before they
reached a conclusion.

## Layout

| file | purpose |
|---|---|
| `int4_gemv_asym.py` | The main kernel. Affine int4 GEMV with per-group scale and zero-point, plus correctness and latency benchmarks. |
| `sweep_layers.py` | Accuracy sweep across every projection weight of a model, affine int4 vs NF4 at matched block size. |
| `int4_gemv.py` | Symmetric int4 version, kept for the symmetric-vs-affine comparison. |
| `int4_gemv_hoisted.py` | A failed optimisation — group-hoisted dequant, 5x slower. Kept so the negative result is reproducible. |
| `fused_rmsnorm.py`, `verify.py` | Earlier work: fused residual-add + RMSNorm. Delivers its theoretical 1.25x over `F.rms_norm` and ties `torch.compile`. |
| `harness.py`, `roofline.py`, `compare.py`, `profile.sh` | General inference benchmarking utilities. |

## Reproducing

Needs an NVIDIA GPU with sm_70 or newer (Pascal is unsupported by recent PyTorch).

```bash
pip install -r requirements.txt

python int4_gemv_asym.py                              # correctness + latency
python sweep_layers.py Qwen/Qwen2.5-0.5B              # accuracy across all layers
```

## Measurement notes

Two things that cost real time here and generalise beyond this project.

**Verify GPU clocks under load before trusting any latency number.** On a shared
notebook GPU, identical code measured 66.9, 76.2 and 177.4 µs across runs. The card was
clamped to 450 MHz of 1590 MHz while reporting no throttle reason. A clock reading taken
after the workload finishes shows the idle clock and tells you nothing.

**Match the block size when comparing quantization schemes.** NF4 defaults to 64,
this kernel defaults to 32. Comparing the defaults gave a unanimous 168/168 result at
0.885x that was half granularity and half format. At matched granularity the real
figure is 0.947x — still a win, but a much smaller one.

## Status

Latency measured on a shared T4 and would benefit from a dedicated machine. No
end-to-end perplexity evaluation yet, which is the largest gap: RMSE on weights is a
proxy for output quality, not a substitute.
