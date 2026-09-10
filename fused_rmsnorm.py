"""
Fused residual-add + RMSNorm, Triton.

Every transformer block does this twice:

    residual = residual + x          # elementwise add over [M, N]
    x        = rmsnorm(residual)     # reduction + scale over [M, N]

In eager PyTorch that is two kernel launches and five passes over an [M, N] tensor.
Fused it is one launch and four passes, with the sum-of-squares computed while the
data is already in registers.

At decode time M = batch_size, so these launches are tiny and latency-bound: the win
is as much about removing launch overhead as about bandwidth.

    python kernels/fused_rmsnorm.py            # correctness + benchmark
    python kernels/fused_rmsnorm.py --hidden 8192
"""

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rmsnorm_fwd(
    X,          # [M, N] input (block output)
    RES,        # [M, N] residual, updated in place
    W,          # [N] scale
    Y,          # [M, N] normalised output
    stride_row,
    N,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_row
    RES += row * stride_row
    Y += row * stride_row

    # Pass 1: add residual, write it back, accumulate sum of squares in fp32.
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RES + cols, mask=mask, other=0.0).to(tl.float32)
        r = r + x
        tl.store(RES + cols, r.to(RES.dtype.element_ty), mask=mask)
        acc += r * r

    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)

    # Pass 2: normalise and scale. RES is now hot in L2.
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        r = tl.load(RES + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(Y + cols, (r * rstd * w).to(Y.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    """Returns (normed, residual). `residual` is modified in place, as in vLLM."""
    assert x.shape == residual.shape
    assert x.is_cuda and residual.is_cuda and weight.is_cuda
    x = x.contiguous()
    residual = residual.contiguous()

    *lead, N = x.shape
    M = 1
    for d in lead:
        M *= d
    x2 = x.view(M, N)
    res2 = residual.view(M, N)
    y = torch.empty_like(x2)

    # Cap the block so the two-pass loop stays in registers on large hidden sizes.
    BLOCK = min(triton.next_power_of_2(N), 8192)
    num_warps = 4 if BLOCK <= 2048 else 8
    if BLOCK >= 8192:
        num_warps = 16

    _fused_add_rmsnorm_fwd[(M,)](
        x2, res2, weight, y,
        x2.stride(0), N, eps,
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return y.view_as(x), residual


def eager_add_rmsnorm(x, residual, weight, eps=1e-6):
    residual = residual + x
    v = residual.to(torch.float32)
    normed = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (normed.to(x.dtype) * weight), residual


def check(M, N, dtype, eps=1e-6):
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)

    y_ref, res_ref = eager_add_rmsnorm(x, res.clone(), w, eps)
    y_out, res_out = fused_add_rmsnorm(x, res.clone(), w, eps)

    # bf16 has ~8 bits of mantissa; 2e-2 is the realistic bar, not 1e-5.
    atol = 2e-2 if dtype is torch.bfloat16 else 1e-2
    y_err = (y_out - y_ref).abs().max().item()
    r_err = (res_out - res_ref).abs().max().item()
    ok = y_err < atol and r_err < atol
    print(
        f"  M={M:<6} N={N:<6} {str(dtype).split('.')[-1]:<9} "
        f"max|dy|={y_err:.2e} max|dres|={r_err:.2e}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def bench(M, N, dtype):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    res = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    itemsize = x.element_size()

    ms_eager = triton.testing.do_bench(lambda: eager_add_rmsnorm(x, res.clone(), w))
    ms_fused = triton.testing.do_bench(lambda: fused_add_rmsnorm(x, res.clone(), w))

    # fused: read x, read res, write res, write y  -> 4 passes
    gb_fused = 4 * M * N * itemsize / 1e9
    # eager: add reads x+res writes res (3), norm reads res writes y (2) -> 5 passes
    gb_eager = 5 * M * N * itemsize / 1e9

    print(
        f"  M={M:<6} N={N:<6} eager {ms_eager * 1000:8.1f} us "
        f"({gb_eager / (ms_eager / 1000):7.0f} GB/s)   "
        f"fused {ms_fused * 1000:8.1f} us "
        f"({gb_fused / (ms_fused / 1000):7.0f} GB/s)   "
        f"speedup {ms_eager / ms_fused:5.2f}x"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096, help="hidden size N")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    print(torch.cuda.get_device_name(0))

    N = args.hidden
    print("\ncorrectness")
    all_ok = True
    for M in (1, 7, 64, 4096):
        for dt in (torch.float16, torch.bfloat16):
            all_ok &= check(M, N, dt)
    # non-power-of-two hidden size, to exercise the mask path
    all_ok &= check(64, 5120 + 3, torch.float16)

    print("\ndecode-shaped (M = batch size, launch-latency dominated)")
    for M in (1, 4, 16, 64):
        bench(M, N, torch.float16)

    print("\nprefill-shaped (M = batch * seq, bandwidth dominated)")
    for M in (2048, 8192, 32768):
        bench(M, N, torch.float16)

    if not all_ok:
        raise SystemExit("\ncorrectness FAILED - do not benchmark a wrong kernel")


if __name__ == "__main__":
    main()
