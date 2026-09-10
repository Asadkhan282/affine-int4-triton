"""
Honest verification and benchmarking for fused_add_rmsnorm.

Two fixes over the version in fused_rmsnorm.py:

1. Correctness is measured against an fp32 ground truth, not against the low-precision
   eager path. Comparing two lossy implementations to each other and applying a fixed
   absolute tolerance tells you nothing -- it fails whenever the tensor magnitude
   happens to exceed the tolerance, which is what happened on bf16.

2. The benchmark compares against baselines a real serving stack would actually use:
   F.rms_norm and a torch.compile'd version. The naive chain of separate ops is kept
   only to show how much of the "speedup" was an artifact of a bad baseline.

    python verify.py
    python verify.py --hidden 8192
"""

import argparse

import torch
import torch.nn.functional as F
import triton

from fused_rmsnorm import eager_add_rmsnorm, fused_add_rmsnorm


def fp32_ground_truth(x, residual, weight, eps=1e-6):
    """Everything in fp32. This is what 'correct' means."""
    r = residual.float() + x.float()
    y = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return y, r


def rel_err(candidate, truth):
    """Max error relative to the scale of the tensor. Dimensionless, comparable
    across dtypes and magnitudes -- unlike a bare absolute tolerance."""
    scale = truth.abs().max().clamp(min=1e-6)
    return ((candidate.float() - truth).abs().max() / scale).item()


def check(M, N, dtype, eps=1e-6):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)

    y_true, r_true = fp32_ground_truth(x, res.clone(), w, eps)
    y_eager, r_eager = eager_add_rmsnorm(x, res.clone(), w, eps)
    y_fused, r_fused = fused_add_rmsnorm(x, res.clone(), w, eps)

    e_eager = rel_err(y_eager, y_true)
    e_fused = rel_err(y_fused, y_true)

    # One mantissa step of the storage dtype, with a small allowance for the
    # accumulated reduction error. This is the physically meaningful bar.
    ulp = 2.0 ** -(8 if dtype is torch.bfloat16 else 11)
    tol = 4 * ulp

    ok = e_fused < tol
    better = "fused better" if e_fused <= e_eager else "eager better"
    print(
        f"  M={M:<6} N={N:<6} {str(dtype).split('.')[-1]:<9} "
        f"rel_err eager={e_eager:.2e} fused={e_fused:.2e} "
        f"tol={tol:.2e}  {better:<13} {'PASS' if ok else 'FAIL'}"
    )
    return ok


# ---------------------------------------------------------------------------
# Baselines a production stack would plausibly run.
# ---------------------------------------------------------------------------

def baseline_f_rmsnorm(x, residual, weight, eps=1e-6):
    residual = residual + x
    return F.rms_norm(residual, (residual.shape[-1],), weight, eps), residual


_compiled = None


def baseline_compiled(x, residual, weight, eps=1e-6):
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(baseline_f_rmsnorm, dynamic=False)
    return _compiled(x, residual, weight, eps)


def bench(M, N, dtype, peak_gb_s):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    itemsize = x.element_size()

    variants = {
        "naive chain": lambda: eager_add_rmsnorm(x, res.clone(), w),
        "F.rms_norm": lambda: baseline_f_rmsnorm(x, res.clone(), w),
        "torch.compile": lambda: baseline_compiled(x, res.clone(), w),
        "triton fused": lambda: fused_add_rmsnorm(x, res.clone(), w),
    }

    times = {}
    for name, fn in variants.items():
        try:
            fn()  # warm up / trigger compilation outside the timed region
            times[name] = triton.testing.do_bench(fn, warmup=25, rep=100)
        except Exception as exc:
            times[name] = None
            print(f"  {name} unavailable: {type(exc).__name__}")

    # The fused kernel moves 4 tensor passes: read x, read res, write res, write y.
    gb = 4 * M * N * itemsize / 1e9
    ref = times.get("F.rms_norm") or times.get("naive chain")

    print(f"  M={M}, N={N}")
    for name, ms in times.items():
        if ms is None:
            continue
        bw = gb / (ms / 1000.0)
        pct = f"{bw / peak_gb_s * 100:5.1f}% of peak" if peak_gb_s else ""
        spd = f"{ref / ms:5.2f}x vs F.rms_norm" if ref else ""
        print(f"    {name:<16} {ms * 1000:9.1f} us   {bw:7.0f} GB/s  {pct}   {spd}")
    print()


PEAK = {"T4": 320, "P100": 732, "A100": 1555, "H100": 3350, "L4": 300, "V100": 900}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096)
    args = ap.parse_args()

    name = torch.cuda.get_device_name(0)
    peak = next((v for k, v in PEAK.items() if k in name), None)
    print(f"{name}   peak HBM ~{peak} GB/s\n")

    N = args.hidden
    print("correctness vs fp32 ground truth")
    ok = True
    for M in (1, 7, 64, 4096):
        for dt in (torch.float16, torch.bfloat16):
            ok &= check(M, N, dt)
    ok &= check(64, 5123, torch.float16)

    print("\ndecode-shaped (launch-latency dominated)")
    for M in (1, 4, 16, 64):
        bench(M, N, torch.float16, peak)

    print("prefill-shaped (bandwidth dominated)")
    for M in (2048, 8192, 32768):
        bench(M, N, torch.float16, peak)

    if not ok:
        raise SystemExit("correctness FAILED against fp32 reference")
    print("all correctness checks passed")


if __name__ == "__main__":
    main()
