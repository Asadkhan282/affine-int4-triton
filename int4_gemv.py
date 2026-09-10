"""
Symmetric int4 weight-only GEMV for Turing (sm_75).

Motivation, measured on a T4 at K=N=4096:
    byte floor (stream 8MB of packed weights)   52.7 us
    arithmetic 4-bit unpack + scale             61.7 us   (1.17x floor)
    bitsandbytes NF4 gemv_4bit                 126.8 us   (2.40x floor)
    fp16 cuBLAS GEMV                           144.3 us

NF4's lookup-table decode costs ~8x more than arithmetic decode. This kernel uses
arithmetic decode instead. Target is 60-70 us, i.e. roughly 2x over bitsandbytes.

IMPORTANT SCOPE NOTE: this is symmetric int4, NOT NF4. They are different
quantization schemes with different accuracy. A speed comparison against NF4 is only
meaningful alongside the accuracy comparison at the bottom of this file. Do not
report the speedup on its own.

Packing layout: byte j of row n holds weight k=j in the low nibble and k=j+K/2 in the
high nibble. This means a contiguous block of bytes decodes into two contiguous
slices of x, so both x loads stay coalesced. Interleaving adjacent k values instead
would force strided loads on x.

Run:  python int4_gemv.py     (or paste into a Kaggle cell and call main())
"""

import torch
import triton
import triton.language as tl

GROUP = 128  # weights per scale, along K


# ---------------------------------------------------------------------------
# Quantization (done once, offline in a real deployment)
# ---------------------------------------------------------------------------

def quantize_int4(w, group=GROUP):
    """w: [N, K] fp16 -> packed uint8 [N, K//2], scales fp16 [N, K//group].

    Symmetric, per-group along K. Levels are -8..7 stored biased by +8 as 0..15.
    """
    N, K = w.shape
    assert K % group == 0, "K must be divisible by group size"
    assert (K // 2) % group == 0, "K/2 must be divisible by group size for the split layout"

    wg = w.float().view(N, K // group, group)
    scale = wg.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-8)
    q = torch.round(wg / scale).clamp(-8, 7).to(torch.int8).view(N, K)

    biased = (q + 8).to(torch.uint8)
    half = K // 2
    packed = (biased[:, :half] | (biased[:, half:] << 4)).contiguous()
    return packed, scale.view(N, K // group).half().contiguous()


def dequantize_int4(packed, scales, K, group=GROUP):
    """Reference dequant, for correctness checking only."""
    N = packed.shape[0]
    half = K // 2
    lo = (packed & 0xF).to(torch.float32) - 8.0
    hi = ((packed >> 4) & 0xF).to(torch.float32) - 8.0
    q = torch.empty(N, K, device=packed.device, dtype=torch.float32)
    q[:, :half] = lo
    q[:, half:] = hi
    s = scales.float().repeat_interleave(group, dim=1)
    return (q * s).half()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": bk}, num_warps=w, num_stages=st)
        for bk in (512, 1024, 2048)
        for w in (2, 4, 8)
        for st in (2, 3, 4)
    ],
    key=["K", "N"],
)
@triton.jit
def _gemv_int4(
    X, PK, SC, Y,
    K, N,
    stride_pk, stride_sc,
    GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)
    half = K // 2
    pk_row = PK + n * stride_pk
    sc_row = SC + n * stride_sc

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for j0 in range(0, half, BLOCK_K):
        offs = j0 + tl.arange(0, BLOCK_K)
        m = offs < half

        b = tl.load(pk_row + offs, mask=m, other=0)
        lo = (b & 0xF).to(tl.float32) - 8.0
        hi = ((b >> 4) & 0xF).to(tl.float32) - 8.0

        x_lo = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
        x_hi = tl.load(X + half + offs, mask=m, other=0.0).to(tl.float32)

        s_lo = tl.load(sc_row + offs // GROUP_SIZE, mask=m, other=0.0).to(tl.float32)
        s_hi = tl.load(sc_row + (half + offs) // GROUP_SIZE, mask=m, other=0.0).to(tl.float32)

        acc += lo * s_lo * x_lo + hi * s_hi * x_hi

    tl.store(Y + n, tl.sum(acc, axis=0).to(Y.dtype.element_ty))


def gemv_int4(x, packed, scales, K):
    """x: [1, K] or [K] fp16. Returns [1, N] fp16."""
    x = x.reshape(-1)
    N = packed.shape[0]
    y = torch.empty(N, device=x.device, dtype=torch.float16)
    _gemv_int4[(N,)](
        x, packed, scales, y,
        K, N,
        packed.stride(0), scales.stride(0),
        GROUP_SIZE=GROUP,
    )
    return y.view(1, N)


# ---------------------------------------------------------------------------
# Correctness and benchmarks
# ---------------------------------------------------------------------------

def rel_err(c, t):
    return ((c.float() - t.float()).abs().max() / t.float().abs().max().clamp(min=1e-6)).item()


def check(K, N):
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(1, K, device="cuda", dtype=torch.float16)

    packed, scales = quantize_int4(w)
    w_deq = dequantize_int4(packed, scales, K)

    # The kernel is only responsible for matching a dequantize-then-matmul of the
    # SAME quantized weights. Quantization error itself is measured separately.
    ref = torch.nn.functional.linear(x.float(), w_deq.float())
    out = gemv_int4(x, packed, scales, K)

    e = rel_err(out, ref)
    ok = e < 5e-3
    print(f"  K={K:<6} N={N:<6} rel_err vs dequant-matmul = {e:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def bench(K, N, floor_us=None):
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(1, K, device="cuda", dtype=torch.float16)
    packed, scales = quantize_int4(w)

    t_fp16 = triton.testing.do_bench(lambda: torch.nn.functional.linear(x, w))
    t_int4 = triton.testing.do_bench(lambda: gemv_int4(x, packed, scales, K))

    bytes4 = K * N / 2
    print(f"  K={K} N={N}")
    print(f"    fp16 cuBLAS   {t_fp16 * 1000:8.1f} us   {K*N*2/1e9/(t_fp16/1000):6.0f} GB/s")
    print(
        f"    triton int4   {t_int4 * 1000:8.1f} us   {bytes4/1e9/(t_int4/1000):6.0f} GB/s"
        f"   {t_fp16/t_int4:5.2f}x vs fp16"
        + (f"   {t_int4*1000/floor_us:5.2f}x of byte floor" if floor_us else "")
    )

    try:
        import bitsandbytes.functional as F4
        q, st = F4.quantize_nf4(w)
        t_nf4 = triton.testing.do_bench(lambda: F4.gemv_4bit(x, q.t(), state=st))
        print(
            f"    bnb NF4       {t_nf4 * 1000:8.1f} us   {bytes4/1e9/(t_nf4/1000):6.0f} GB/s"
            f"   {t_nf4/t_int4:5.2f}x slower than this kernel"
        )
    except Exception as exc:
        print(f"    bnb NF4 unavailable: {type(exc).__name__}")
    print()


def accuracy_comparison(K=4096, N=4096):
    """The honest part: int4-symmetric is faster than NF4, but is it as accurate?

    NF4's codebook is designed for normally-distributed weights, which is exactly
    what randn produces -- so this test is if anything generous to NF4. Real model
    weights have outliers that change the picture, so repeat this on a real
    checkpoint before publishing any accuracy claim.
    """
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)

    packed, scales = quantize_int4(w)
    w_int4 = dequantize_int4(packed, scales, K)
    e_int4 = (w_int4.float() - w.float()).pow(2).mean().sqrt().item()

    print("weight reconstruction error (RMSE, lower is better)")
    print(f"  int4 symmetric, group={GROUP}: {e_int4:.5f}")

    try:
        import bitsandbytes.functional as F4
        q, st = F4.quantize_nf4(w)
        w_nf4 = F4.dequantize_nf4(q, st).view(N, K)
        e_nf4 = (w_nf4.float() - w.float()).pow(2).mean().sqrt().item()
        print(f"  NF4:                          {e_nf4:.5f}")
        print(f"  ratio int4/NF4:               {e_int4/e_nf4:.3f}x")
        if e_int4 > e_nf4:
            print("  -> int4 is faster but LESS accurate. Report both numbers together.")
    except Exception as exc:
        print(f"  NF4 unavailable: {type(exc).__name__}")


def main():
    print(torch.cuda.get_device_name(0), "\n")
    print("correctness")
    ok = True
    for K, N in ((4096, 4096), (4096, 11008), (11008, 4096)):
        ok &= check(K, N)

    print("\nspeed")
    for K, N, floor in ((4096, 4096, 52.7), (4096, 11008, 100.4), (11008, 4096, 96.2)):
        bench(K, N, floor)

    print("accuracy")
    accuracy_comparison()

    print("\nbest config at 4096x4096:", _gemv_int4.best_config)
    if not ok:
        raise SystemExit("correctness failed")


if __name__ == "__main__":
    main()
