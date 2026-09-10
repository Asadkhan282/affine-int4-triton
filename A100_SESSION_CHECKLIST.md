# A100 session checklist

Target: 90 minutes of paid time, ~$3–6. Everything below is copy-paste in order.
Read the whole thing before you start the instance — the clock is running once it boots.

---

## Before you rent (do this on your laptop, free)

- [ ] Push `int4_gemv_asym.py`, `int4_gemv.py`, `fused_rmsnorm.py` to a GitHub repo
      (private is fine). Cloning is faster and less error-prone than uploading, and the
      `%%writefile` paste failures you hit on Kaggle do not happen with git.
- [ ] Have this checklist open in another window.
- [ ] Know your stopping time. Set a phone timer. Instances bill until you destroy them,
      not until you close the tab.

---

## Pick the instance

Vast.ai or RunPod. Requirements:

- **A100 40GB or 80GB.** L40S or L4 are acceptable alternatives and cheaper — L4 is
  arguably more relevant since it is the card people actually deploy 4-bit models on.
- **A PyTorch/CUDA template**, not a bare Ubuntu image. Saves 20 minutes of driver work.
- **On-demand, not interruptible/spot.** A preemption mid-benchmark wastes more than the
  price difference.
- Check the listed **PCIe bandwidth and reliability score** on Vast; low-scoring hosts
  are shared and will reproduce exactly the noise problem you just escaped.

---

## Setup (~10 min)

```bash
nvidia-smi
```
Confirm the GPU name and that no other process is using it. `Memory-Usage` should be
near zero and `GPU-Util` at 0%. If something else is running, pick another host.

```bash
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv
```
Record this. You will compare it under load later.

```bash
pip install -q triton bitsandbytes transformers accelerate
git clone <your repo> && cd <your repo>
python -c "import torch, triton, bitsandbytes; print(torch.__version__, triton.__version__, bitsandbytes.__version__); print(torch.cuda.get_device_name(0))"
```

---

## Verify clocks under load — DO NOT SKIP (~2 min)

This is the step that invalidated a day of T4 measurements.

```bash
python -c "
import torch
a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
for _ in range(300): torch.mm(a, a)
torch.cuda.synchronize()
" &
sleep 20
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,utilization.gpu --format=csv
wait
```

`clocks.sm` should be at or near `clocks.max.sm` while utilisation is high. If it is
below ~80% of max, destroy the instance and take a different host. Do not proceed and
hope. A throttled card produces numbers that look fine and are worthless.

---

## Run 1 — the main benchmark (~15 min)

```bash
python int4_gemv_asym.py 2>&1 | tee results_a100_run1.txt
python int4_gemv_asym.py 2>&1 | tee results_a100_run2.txt
```

Two runs. If the medians differ by more than ~2%, the host is noisy and every
conclusion needs wider error bars.

**What may change versus the T4, and why it matters more than the raw numbers:**

- The A100 has 1555 GB/s versus 320. Your kernel was compute-bound on dequantization at
  137–150 GB/s, so more bandwidth may not help it at all — the gap to fp16 could widen
  or narrow.
- bitsandbytes has Ampere-tuned paths that Turing lacks. **NF4 may beat your kernel
  here.** If so, that is the result, and the honest claim narrows to "affine int4 wins
  on Turing, where NF4 has no tuned kernel." Narrow claims survive review; broad ones
  invite someone to run it on their own hardware and contradict you.
- Autotune will pick different configs. Record `best_config` from both runs.

---

## Run 2 — accuracy across real layers (~20 min)

The single-layer accuracy claim is the weakest part of the draft. Fix it here.

```python
# save as sweep_layers.py, then: python sweep_layers.py 2>&1 | tee accuracy_a100.txt
import torch, bitsandbytes.functional as F4
from transformers import AutoModelForCausalLM
from int4_gemv_asym import quantize_int4_asym, dequantize_int4_asym

MODEL = "Qwen/Qwen2.5-1.5B"   # bigger than the 0.5B used on Kaggle
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)

def rmse(a, b):
    return (a.float() - b.float()).pow(2).mean().sqrt().item()

print(f"{'layer':<44}{'shape':>14}{'kurt':>7}{'NF4':>10}{'i4g32':>10}{'ratio':>8}")
rows = []
for name, p in m.named_parameters():
    if not any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj")):
        continue
    w = p.data.cuda()
    N, K = w.shape
    if K % 64 or (K // 2) % 32:
        continue
    f = w.float()
    kurt = ((f - f.mean())**4).mean().item() / f.var().item()**2

    packed, sc, zp = quantize_int4_asym(w, group=32)
    e_i4 = rmse(dequantize_int4_asym(packed, sc, zp, K, group=32), w)

    q, st = F4.quantize_nf4(w)
    e_nf4 = rmse(F4.dequantize_nf4(q, st).view(N, K), w)

    rows.append(e_i4 / e_nf4)
    print(f"{name:<44}{str((N,K)):>14}{kurt:>7.2f}{e_nf4:>10.5f}{e_i4:>10.5f}{e_i4/e_nf4:>8.3f}x")

import statistics
print(f"\nlayers: {len(rows)}   median ratio: {statistics.median(rows):.3f}x   "
      f"worst: {max(rows):.3f}x   best: {min(rows):.3f}x")
print(f"layers where affine int4 wins: {sum(r < 1.0 for r in rows)}/{len(rows)}")
```

The last line is the claim. "Affine int4 at G=32 beats NF4 on 47 of 49 layers" is
defensible; one layer is an anecdote.

---

## Run 3 — Nsight profile (~15 min, optional but high value)

You have root here, so `ncu` works — it did not on Kaggle.

```bash
ncu --set full --launch-count 1 --kernel-name regex:gemv \
    --target-processes all python int4_gemv_asym.py 2>&1 | tee ncu_a100.txt
grep -A10 "Speed Of Light" ncu_a100.txt
```

Read the compute-vs-memory utilisation split. Your phase isolation on the T4 inferred
that the kernel is dequant-bound; this measures it directly. A screenshot or the
SpeedOfLight table in the writeup is worth more than any amount of prose asserting it.

---

## Before you destroy the instance

- [ ] `results_a100_run1.txt`, `results_a100_run2.txt`, `accuracy_a100.txt`,
      `ncu_a100.txt` all saved
- [ ] Copy them off the box: `git add -A && git commit -m "a100 results" && git push`,
      or scp to your laptop. **Files on the instance are gone the moment you destroy it.**
- [ ] Record the exact instance type, GPU model, driver, and library versions
- [ ] Verify the clock reading you took under load is written down

```bash
git add -A && git commit -m "A100 results" && git push
```

Then **destroy the instance**, not just stop it. Stopped instances on some providers
still bill for storage.

---

## After

Fold the numbers into `WRITEUP_DRAFT.md`, replacing the T4 tables as the headline and
keeping the T4 numbers as a secondary architecture comparison. If NF4 wins on Ampere,
say so plainly in the summary — a writeup that reports where its own result stops
holding is far more credible than one that quietly tests only the favourable case.
