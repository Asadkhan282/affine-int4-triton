"""
Accuracy sweep: affine int4 (group 32) vs NF4 across every linear layer of a model.

The single-layer result (Qwen2.5-0.5B down_proj, affine int4 at 0.883x NF4's RMSE) is
an anecdote. This sweeps every attention and MLP projection so the claim becomes
"wins on N of M layers, median ratio X" -- which is evidence.

Run on the A100 box:
    python sweep_layers.py 2>&1 | tee accuracy_a100.txt

Requires int4_gemv_asym.py in the same directory.
"""

import statistics
import sys

import torch
import bitsandbytes.functional as F4
from transformers import AutoModelForCausalLM

from int4_gemv_asym import quantize_int4_asym, dequantize_int4_asym

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B"
GROUP = 32

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


def rmse(a, b):
    return (a.float() - b.float()).pow(2).mean().sqrt().item()


def main():
    print(f"model: {MODEL}   group: {GROUP}   device: {torch.cuda.get_device_name(0)}\n")
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)

    header = f"{'layer':<46}{'shape':>14}{'kurt':>7}{'NF4':>11}{'i4g32':>11}{'ratio':>8}"
    print(header)
    print("-" * len(header))

    ratios, skipped = [], 0
    for name, p in m.named_parameters():
        if not any(k in name for k in TARGETS):
            continue

        # biases match the name filter but are 1-D
        if p.dim() != 2:
            continue

        w = p.data.cuda()
        N, K = w.shape

        # packing layout constraints
        if K % GROUP or (K // 2) % GROUP:
            skipped += 1
            continue

        f = w.float()
        kurt = ((f - f.mean()) ** 4).mean().item() / f.var().item() ** 2

        packed, sc, zp = quantize_int4_asym(w, group=GROUP)
        e_i4 = rmse(dequantize_int4_asym(packed, sc, zp, K, group=GROUP), w)

        q, st = F4.quantize_nf4(w)
        e_nf4 = rmse(F4.dequantize_nf4(q, st).view(N, K), w)

        ratio = e_i4 / e_nf4
        ratios.append((ratio, name, kurt))
        print(f"{name:<46}{str((N, K)):>14}{kurt:>7.2f}{e_nf4:>11.5f}{e_i4:>11.5f}{ratio:>8.3f}x")

        del w, packed, sc, zp, q, st
        torch.cuda.empty_cache()

    if not ratios:
        raise SystemExit("no eligible layers found -- check TARGETS and shape constraints")

    vals = [r[0] for r in ratios]
    wins = sum(v < 1.0 for v in vals)

    print(f"\nlayers measured : {len(vals)}   (skipped {skipped} on shape constraints)")
    print(f"median ratio    : {statistics.median(vals):.3f}x")
    print(f"mean ratio      : {statistics.mean(vals):.3f}x")
    print(f"best / worst    : {min(vals):.3f}x / {max(vals):.3f}x")
    print(f"affine int4 wins: {wins}/{len(vals)} layers  ({wins/len(vals)*100:.0f}%)")

    worst = sorted(ratios, reverse=True)[:3]
    print("\nworst three layers (where NF4's codebook helps most):")
    for r, n, k in worst:
        print(f"  {r:.3f}x  kurt={k:6.2f}  {n}")

    # Does the win/loss track distribution shape? If high-kurtosis layers cluster in
    # the losses, that is the mechanism and belongs in the writeup.
    hi = [v for v, _, k in ratios if k > statistics.median([k for _, _, k in ratios])]
    lo = [v for v, _, k in ratios if k <= statistics.median([k for _, _, k in ratios])]
    print(f"\nmedian ratio, high-kurtosis half: {statistics.median(hi):.3f}x")
    print(f"median ratio, low-kurtosis half : {statistics.median(lo):.3f}x")


if __name__ == "__main__":
    main()
