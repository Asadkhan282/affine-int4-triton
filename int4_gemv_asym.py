"""
Asymmetric (affine) int4 weight-only GEMV for Turing (sm_75).

Difference from int4_gemv.py: each group carries a zero-point as well as a scale, so
the 16 levels span [min, max] of the group rather than [-absmax, +absmax]. Real
weight groups are not centred on zero, so this recovers accuracy that symmetric
quantization throws away.

Measured on a T4, Qwen2.5-0.5B layer 5 down_proj (896 x 4864), NF4 RMSE = 0.00160:
    sym  group=128   0.00216   1.352x NF4
    sym  group=32    0.00172   1.077x NF4
    asym group=32    0.00141   0.881x NF4   <- more accurate than NF4

The open question this file answers: what does the zero-point cost in latency?
Symmetric group=128 ran at 49.9 us. Two effects push against each other -- group=32
means 4x more scale traffic (12.5% overhead over the packed weights), and the
zero-point adds a load plus a subtract per element.

Packing layout is unchanged: byte j of row n holds weight k=j in the low nibble and
k=j+K/2 in the high nibble, keeping both x loads contiguous.
"""

import torch
import triton
import triton.language as tl

GROUP = 32


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_int4_asym(w, group=GROUP):
    """w: [N, K] fp16 -> packed uint8 [N, K//2], scales fp16, zeros fp16.

    Levels are 0..15 with no bias. Dequant is (q - zp) * scale.
    """
    N, K = w.shape
    assert K % group == 0, "K must be divisible by group"
    assert (K // 2) % group == 0, "K/2 must be divisible by group for the split layout"

    wg = w.float().view(N, K // group, group)
    lo = wg.amin(dim=-1, keepdim=True)
    hi = wg.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / 15.0).clamp(min=1e-8)
    zp = torch.round(-lo / scale).clamp(0, 15)

    q = torch.round(wg / scale + zp).clamp(0, 15).to(torch.uint8).view(N, K)
    half = K // 2
    packed = (q[:, :half] | (q[:, half:] << 4)).contiguous()

    return (
        packed,
        scale.view(N, K // group).half().contiguous(),
        zp.view(N, K // group).half().contiguous(),
    )


def dequantize_int4_asym(packed, scales, zeros, K, group=GROUP):
    """Reference dequant. Correctness checking only."""
    N = packed.shape[0]
    half = K // 2
    q = torch.empty(N, K, device=packed.device, dtype=torch.float32)
    q[:, :half] = (packed & 0xF).to(torch.float32)
    q[:, half:] = ((packed >> 4) & 0xF).to(torch.float32)
    s = scales.float().repeat_interleave(group, dim=1)
    z = zeros.float().repeat_interleave(group, dim=1)
    return ((q - z) * s).half()


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
def _gemv_int4_asym(
    X, PK, SC, ZP, Y,
    K, N,
    stride_pk, stride_sc,
    GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)
    half = K // 2
    pk_row = PK + n * stride_pk
    sc_row = SC + n * stride_sc
    zp_row = ZP + n * stride_sc

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for j0 in range(0, half, BLOCK_K):
        offs = j0 + tl.arange(0, BLOCK_K)
        m = offs < half

        b = tl.load(pk_row + offs, mask=m, other=0)
        q_lo = (b & 0xF).to(tl.float32)
        q_hi = ((b >> 4) & 0xF).to(tl.float32)

        x_lo = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
        x_hi = tl.load(X + half + offs, mask=m, other=0.0).to(tl.float32)

        g_lo = offs // GROUP_SIZE
        g_hi = (half + offs) // GROUP_SIZE

        s_lo = tl.load(sc_row + g_lo, mask=m, other=0.0).to(tl.float32)
        s_hi = tl.load(sc_row + g_hi, mask=m, other=0.0).to(tl.float32)
        z_lo = tl.load(zp_row + g_lo, mask=m, other=0.0).to(tl.float32)
        z_hi = tl.load(zp_row + g_hi, mask=m, other=0.0).to(tl.float32)

        acc += (q_lo - z_lo) * s_lo * x_lo + (q_hi - z_hi) * s_hi * x_hi

    tl.store(Y + n, tl.sum(acc, axis=0).to(Y.dtype.element_ty))


def gemv_int4_asym(x, packed, scales, zeros, K):
    x = x.reshape(-1)
    N = packed.shape[0]
    y = torch.empty(N, device=x.device, dtype=torch.float16)
    _gemv_int4_asym[(N,)](
        x, packed, scales, zeros, y,
        K, N,
        packed.stride(0), scales.stride(0),
        GROUP_SIZE=GROUP,
    )
    return y.view(1, N)


# ---------------------------------------------------------------------------
# Checks and benchmarks
# ---------------------------------------------------------------------------

def rel_err(c, t):
    return ((c.float() - t.float()).abs().max() / t.float().abs().max().clamp(min=1e-6)).item()


def check(K, N):
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(1, K, device="cuda", dtype=torch.float16)

    packed, scales, zeros = quantize_int4_asym(w)
    w_deq = dequantize_int4_asym(packed, scales, zeros, K)

    ref = torch.nn.functional.linear(x.float(), w_deq.float())
    out = gemv_int4_asym(x, packed, scales, zeros, K)

    e = rel_err(out, ref)
    ok = e < 5e-3
    print(f"  K={K:<6} N={N:<6} rel_err = {e:.2e}  {'PASS' if ok else 'FAIL'}")
    return ok


def bench(K, N, reps=5):
    """Repeat medians, because single do_bench calls on this box swing 30%."""
    torch.manual_seed(0)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)
    x = torch.randn(1, K, device="cuda", dtype=torch.float16)
    packed, scales, zeros = quantize_int4_asym(w)

    def med(fn):
        ts = sorted(triton.testing.do_bench(fn) for _ in range(reps))
        return ts[len(ts) // 2], ts[0], ts[-1]

    t16, lo16, hi16 = med(lambda: torch.nn.functional.linear(x, w))
    t4, lo4, hi4 = med(lambda: gemv_int4_asym(x, packed, scales, zeros, K))

    wbytes = K * N / 2
    sbytes = 2 * 2 * (K // GROUP) * N          # scales + zeros, fp16
    total = wbytes + sbytes

    print(f"  K={K} N={N}   (weights {wbytes/1e6:.1f}MB + meta {sbytes/1e6:.1f}MB "
          f"= {sbytes/wbytes*100:.1f}% overhead)")
    print(f"    fp16 cuBLAS   {t16*1000:8.1f} us  [{lo16*1000:.1f}-{hi16*1000:.1f}]")
    print(f"    asym int4     {t4*1000:8.1f} us  [{lo4*1000:.1f}-{hi4*1000:.1f}]"
          f"   {total/1e9/(t4/1000):6.0f} GB/s   {t16/t4:5.2f}x vs fp16")

    try:
        import bitsandbytes.functional as F4
        q, st = F4.quantize_nf4(w)
        tn, lon, hin = med(lambda: F4.gemv_4bit(x, q.t(), state=st))
        print(f"    bnb NF4       {tn*1000:8.1f} us  [{lon*1000:.1f}-{hin*1000:.1f}]"
              f"   {tn/t4:5.2f}x slower than this kernel")
    except Exception as exc:
        print(f"    bnb NF4 unavailable: {type(exc).__name__}")
    print()


def accuracy_table(w, nf4_rmse=None):
    """Unambiguous sym-vs-asym sweep on a real weight matrix."""
    N, K = w.shape

    def rmse(group, asym):
        wg = w.float().view(N, K // group, group)
        if asym:
            lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
            s = ((hi - lo) / 15.0).clamp(min=1e-8)
            z = torch.round(-lo / s).clamp(0, 15)
            deq = (torch.round(wg / s + z).clamp(0, 15) - z) * s
        else:
            s = (wg.abs().amax(-1, keepdim=True) / 7.0).clamp(min=1e-8)
            deq = torch.round(wg / s).clamp(-8, 7) * s
        return (deq.view(N, K).half().float() - w.float()).pow(2).mean().sqrt().item()

    if nf4_rmse is None:
        try:
            import bitsandbytes.functional as F4
            q, st = F4.quantize_nf4(w)
            nf4_rmse = (F4.dequantize_nf4(q, st).view(N, K).float()
                        - w.float()).pow(2).mean().sqrt().item()
        except Exception:
            nf4_rmse = None

    print(f"  shape {tuple(w.shape)}   NF4 RMSE = "
          + (f"{nf4_rmse:.5f}" if nf4_rmse else "unavailable"))
    print(f"    {'mode':<6}{'group':>7}{'RMSE':>11}{'vs NF4':>10}{'meta':>9}")
    for asym in (False, True):
        for g in (128, 64, 32):
            e = rmse(g, asym)
            ratio = f"{e/nf4_rmse:.3f}x" if nf4_rmse else "-"
            meta = 2 * (2 if asym else 1) * 2 / g   # bytes per weight-byte
            print(f"    {'asym' if asym else 'sym':<6}{g:>7}{e:>11.5f}{ratio:>10}"
                  f"{meta*100:>8.1f}%")


def main():
    print(torch.cuda.get_device_name(0), f"   GROUP={GROUP}\n")

    print("correctness")
    ok = True
    for K, N in ((4096, 4096), (4096, 11008), (11008, 4096)):
        ok &= check(K, N)

    print("\nspeed (median of 5, range shown)")
    for K, N in ((4096, 4096), (4096, 11008), (11008, 4096)):
        bench(K, N)

    print("accuracy on synthetic weights")
    accuracy_table(torch.randn(4096, 4096, device="cuda", dtype=torch.float16))

    print("\nbest config:", _gemv_int4_asym.best_config)
    if not ok:
        raise SystemExit("correctness failed")


if __name__ == "__main__":
    main()
