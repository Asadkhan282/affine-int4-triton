"""
End-to-end perplexity: affine int4 (group 32) vs NF4 (blocksize 32) vs fp16.

RMSE on weights says affine int4 reconstructs ~5% better at matched granularity. This
asks the question that actually matters: does that move output quality?

Method: quantize-dequantize every eligible Linear weight in place, then measure
perplexity on wikitext-2 with a fixed sliding window. Both schemes get identical
treatment -- same layers touched, same layers skipped, same evaluation.

Three outcomes, all worth reporting:
  * int4 perplexity clearly lower  -> the RMSE advantage translates, result matters
  * both within noise of each other -> reconstruction error at this level does not
    move output quality, which would mean the RMSE-based comparisons common in the
    quantization literature are measuring the wrong thing
  * int4 worse -> RMSE is not predictive here and the headline needs rewriting

    python perplexity_eval.py                          # Qwen2.5-0.5B
    python perplexity_eval.py TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import copy
import sys

import torch
import torch.nn as nn
import bitsandbytes.functional as F4
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from int4_gemv_asym import quantize_int4_asym, dequantize_int4_asym

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
GROUP = 32
SEQLEN = 2048
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Quantize-dequantize in place
# ---------------------------------------------------------------------------

def eligible(w):
    """Both schemes must see exactly the same set of weights, so the eligibility
    test is the intersection of their constraints."""
    if w.dim() != 2:
        return False
    K = w.shape[1]
    return K % GROUP == 0 and (K // 2) % GROUP == 0


def qdq_affine_int4(w):
    K = w.shape[1]
    packed, sc, zp = quantize_int4_asym(w, group=GROUP)
    return dequantize_int4_asym(packed, sc, zp, K, group=GROUP)


def qdq_nf4(w):
    q, state = F4.quantize_nf4(w, blocksize=GROUP)
    return F4.dequantize_nf4(q, state).view(w.shape).to(w.dtype)


def apply_qdq(model, fn):
    """Replace every eligible nn.Linear weight with its quantize-dequantize roundtrip.
    Returns (touched, skipped)."""
    touched = skipped = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.data
        if not eligible(w):
            skipped += 1
            continue
        module.weight.data = fn(w.to(DEVICE)).to(w.dtype)
        touched += 1
    return touched, skipped


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.inference_mode()
def perplexity(model, input_ids, seqlen=SEQLEN):
    """Standard non-overlapping-window perplexity. Stride == seqlen, so every token
    is predicted exactly once and the number is comparable across runs."""
    model.eval()
    n = input_ids.numel() // seqlen
    nlls = []
    for i in range(n):
        chunk = input_ids[:, i * seqlen : (i + 1) * seqlen].to(DEVICE)
        out = model(chunk, labels=chunk)
        # HF returns mean NLL over (seqlen - 1) predicted positions
        nlls.append(out.loss.float() * (seqlen - 1))
    total = torch.stack(nlls).sum()
    return torch.exp(total / (n * (seqlen - 1))).item()


def main():
    print(f"model  : {MODEL}")
    print(f"group  : {GROUP}   seqlen: {SEQLEN}   device: {torch.cuda.get_device_name(0)}\n")

    tok = AutoTokenizer.from_pretrained(MODEL)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    print(f"eval tokens: {ids.numel():,}  ({ids.numel() // SEQLEN} windows)\n")

    results = {}

    for label, fn in (("fp16 baseline", None),
                      ("affine int4 g=32", qdq_affine_int4),
                      ("NF4 blocksize=32", qdq_nf4)):
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEVICE)

        if fn is not None:
            touched, skipped = apply_qdq(model, fn)
            note = f"{touched} layers quantized, {skipped} skipped"
        else:
            note = "unmodified"

        ppl = perplexity(model, ids)
        results[label] = ppl
        print(f"{label:<20} ppl {ppl:8.4f}   ({note})")

        del model
        torch.cuda.empty_cache()

    base = results["fp16 baseline"]
    i4 = results["affine int4 g=32"]
    nf4 = results["NF4 blocksize=32"]

    print(f"\ndegradation vs fp16")
    print(f"  affine int4 : {i4 - base:+.4f}  ({(i4/base - 1)*100:+.2f}%)")
    print(f"  NF4         : {nf4 - base:+.4f}  ({(nf4/base - 1)*100:+.2f}%)")
    print(f"\naffine int4 minus NF4: {i4 - nf4:+.4f} ppl")

    gap = abs(i4 - nf4)
    if gap < 0.01:
        print("  -> indistinguishable. The RMSE advantage does not translate to")
        print("     output quality at this scale. Report this plainly.")
    elif i4 < nf4:
        print("  -> affine int4 is better end-to-end, consistent with the RMSE result.")
    else:
        print("  -> NF4 is better end-to-end despite worse RMSE. RMSE is not")
        print("     predictive here and the headline needs rewriting.")

    print("\nCaveat: one dataset, one sequence length, no seed variation. Perplexity")
    print("differences under ~0.05 on a model this size should not be treated as real")
    print("without repeating across datasets.")


if __name__ == "__main__":
    main()
