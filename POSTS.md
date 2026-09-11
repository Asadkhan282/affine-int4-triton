# Ready-to-post text

Two drafts. Edit the tone if you like, but do not inflate the numbers — the honesty is
the part that makes people take it seriously.

---

## 1. r/LocalLLaMA post

**Title:**

> NF4 vs plain affine int4: the codebook only wins above block size 64

**Body:**

I wrote an affine (asymmetric) int4 weight-only GEMV in Triton and benchmarked it
against bitsandbytes NF4 on a T4, mostly to find out what NF4's normal-distribution
codebook is actually buying.

The answer turned out to depend entirely on block size:

| block / group | NF4 RMSE | affine int4 RMSE | ratio |
|---|---|---|---|
| 128 | 0.00169 | 0.00181 | 1.074x |
| 64 | 0.00160 | 0.00161 | 1.009x |
| 32 | 0.00150 | 0.00141 | 0.942x |

They cross over at 64. Sweeping every projection weight in two models, affine int4 at
group 32 beats NF4 at blocksize 32 on **316 of 322 layers**, median ratio 0.94x.

It holds end-to-end on wikitext-2:

| model | fp16 | affine int4 | NF4 |
|---|---|---|---|
| Qwen2.5-0.5B | 13.07 | **14.95** | 15.32 |
| Qwen2.5-1.5B | 9.27 | **10.21** | 10.37 |
| TinyLlama-1.1B | 7.97 | **8.29** | 8.32 |

Interesting bit: the gap scales with how much 4-bit hurts the model. Qwen2.5-0.5B
degrades 14% under int4 and shows a 0.37 ppl gap; TinyLlama degrades 4% and the gap is
0.03, which is within noise. Where quantization is nearly free, the format barely
matters.

My guess at the mechanism: NF4's codebook is fit to a standard normal, so it needs each
block to look Gaussian after normalisation. At 32 weights a block is too small a sample
to look like anything, the assumption breaks, and a plain affine grid spanning [min,
max] does better.

Caveats, and there are several: T4 only, batch size 1, naive round-to-nearest with no
calibration (which is why even the better format costs 14% perplexity on the 0.5B — GPTQ
and AWQ lose far less). I also haven't done a byte-for-byte accounting of NF4's
metadata at block 32, so some of the gap might be bought with memory rather than won by
the format.

Latency is roughly at parity with NF4 and about 2x faster than cuBLAS fp16, but the T4
these were measured on was clock-throttled part of the time, so I trust the accuracy
numbers considerably more than the timings.

Code and full writeup: https://github.com/Asadkhan282/affine-int4-triton

Happy to be told I've measured something wrong — that happened four times during this
project already and the writeup documents each one.

---

## 2. bitsandbytes GitHub issue

File at: https://github.com/bitsandbytes-foundation/bitsandbytes/issues

**Title:**

> NF4 reconstruction is worse than affine int4 below blocksize 64

**Body:**

I've been comparing NF4 against a plain affine (asymmetric) int4 quantizer at matched
block sizes, and the ordering reverses as blocks get smaller.

Measured on Qwen2.5-0.5B `layers.5.mlp.down_proj`, RMSE of reconstructed weights vs
fp16:

| block / group | NF4 | affine int4 | ratio |
|---|---|---|---|
| 128 | 0.00169 | 0.00181 | 1.074x |
| 64 | 0.00160 | 0.00161 | 1.009x |
| 32 | 0.00150 | 0.00141 | 0.942x |

Across every 2-D projection weight in Qwen2.5-0.5B (168) and TinyLlama-1.1B (154), at
matched block size 32, affine int4 has lower RMSE on 316 of 322 layers, median ratio
0.94x. The six exceptions are all attention projections with very high kurtosis —
TinyLlama's layer-0 `k_proj` has kurtosis ~300, and NF4 wins there by about 1%.

It shows up end-to-end as well. Quantize-dequantize of every eligible `nn.Linear`
weight, then wikitext-2 perplexity, 2048-token non-overlapping windows:

| model | fp16 | affine int4 g=32 | NF4 bs=32 |
|---|---|---|---|
| Qwen2.5-0.5B | 13.0703 | 14.9539 | 15.3223 |
| Qwen2.5-1.5B | 9.2650 | 10.2147 | 10.3668 |
| TinyLlama-1.1B | 7.9723 | 8.2928 | 8.3205 |

Possible explanation: the NF4 codebook is fit to a standard normal, so it relies on
each block resembling a normal distribution after normalisation. At 32 elements a block
is too small a sample for that to hold, whereas an affine grid spanning [min, max]
makes no distributional assumption.

Two caveats on my side. I haven't done a byte-for-byte comparison of metadata overhead
at block 32, so part of this gap may be explained by affine int4 storing both a scale
and a zero-point. And this is naive round-to-nearest for both formats, with no
calibration — the picture may differ once activation statistics are used.

Reproduction: https://github.com/Asadkhan282/affine-int4-triton

Versions: bitsandbytes 0.50.2, torch 2.10.0+cu128, triton 3.6.0, driver 580.159.04,
Tesla T4.

Happy to run further tests if any of this would be useful, or to be told where the
comparison is unfair.
