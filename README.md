# inference-bench

A reproducible harness for measuring and improving LLM inference performance on a
single GPU, plus a worked example of a fused Triton kernel that beats PyTorch eager.

The point of this repo is not the code. The point is the **before/after numbers with
traces attached**. That artifact is what turns a cold email into a contract.

## What's here

| Path | Purpose |
|---|---|
| `bench/harness.py` | Measures TTFT, inter-token latency, throughput, peak memory. Manual decode loop so per-token timing is real, not inferred from `generate()`. |
| `bench/roofline.py` | Turns raw latency into achieved memory bandwidth and % of device peak. Tells you whether you are memory-bound or compute-bound before you optimise anything. |
| `bench/compare.py` | Diffs two result JSONs into a table + chart for the writeup. |
| `bench/profile.sh` | Nsight Systems / Compute invocations with flags that actually work on a decode loop. |
| `kernels/fused_rmsnorm.py` | Fused residual-add + RMSNorm in Triton, with correctness test and a benchmark against eager PyTorch. |
| `REPORT_TEMPLATE.md` | The structure of the public writeup. |

## Quickstart

On a rented GPU box (an H100 SXM is ~$2–3/hr spot; an A100 40GB is enough to start):

```bash
pip install -r requirements.txt

# 1. Baseline
python bench/harness.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 --mode eager \
  --batch-sizes 1,4,16,32 --prompt-len 512 --gen-len 128 \
  --out results/baseline.json

# 2. Where is the time going?
python bench/roofline.py results/baseline.json

# 3. Change one thing (compile, quantise, kernel swap, paged KV...)
python bench/harness.py ... --mode compile --out results/compiled.json

# 4. Diff it
python bench/compare.py results/baseline.json results/compiled.json --chart results/delta.png
```

## The kernel example

```bash
python kernels/fused_rmsnorm.py            # correctness + benchmark sweep
```

Fusing the residual add into RMSNorm removes one full read and one full write of the
hidden-state tensor per layer. At decode-time batch sizes this is pure bandwidth
savings on an op that runs `2 * n_layers` times per token.

## What to publish

Do not publish "I made it faster." Publish:

1. Hardware, driver, torch/triton versions, model, exact command lines.
2. Baseline table: TTFT, p50/p95 ITL, tok/s, peak VRAM, achieved GB/s, % of peak.
3. The profile that told you where the time went — screenshot the Nsight timeline.
4. The change, as a diff.
5. After table, same columns.
6. What you tried that did **not** work. This is the part that convinces engineers
   you actually did the work.

Repo public, writeup on your own domain, cross-post to r/LocalLLaMA and X. One good
writeup outperforms fifty cold emails, and it is the answer to "have you done this
before."

## Caveats

Numbers move a lot with driver and library versions. Always re-run the baseline on
the same box, in the same session, as the optimised version. Never compare against a
number you measured last week.
