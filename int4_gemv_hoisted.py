"""
Group-hoisted affine int4 GEMV.

Motivation (measured, T4, K=N=4096, GROUP=32, BLOCK_K=128):
    loads only                 61.8 us
    loads + per-element dequant 99.9 us      <- 38 us of arithmetic
    full kernel (with reduce)   90.6 us

The dequant math, not memory traffic, is the cost. The per-element form evaluates

    sum_k (q_k - z_g) * s_g * x_k

with one subtract and two multiplies per element, and reloads s_g / z_g for every
element even though only BLOCK_K/GROUP distinct values exist per block.

Reassociating within a group:

    sum_{k in g} (q_k - z_g) * s_g * x_k  =  s_g * ( sum_k q_k*x_k  -  z_g * sum_k x_k )

so the inner loop becomes one multiply-accumulate per element into two running sums,
and the scale/zero-point are applied once per group. At GROUP=32 that removes ~31 of
every 32 scale loads and most of the multiplies.

NOTE ON NUMERICS: this changes the accumulation order, so results will not be
bit-identical to the per-element version. Both accumulate in fp32; the hoisted form
actually has a shorter dependency chain per group. Verified against fp32 ground truth
below rather than against the old kernel.

Requires BLOCK_K % GROUP_SIZE == 0 and (K/2) % GROUP_SIZE == 0.
"""

import torch
import triton
import triton.language as tl

from int4_gemv_asym import quantize_int4_asym, dequantize_int4_asym


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": bk}, num_warps=w, num_stages=st)
        for bk in (128, 256, 512, 1024)
        for w in (1, 2, 4, 8)
        for st in (2, 3, 4)
    ],
    key=["K", "N"],
)
@triton.jit
def _gemv_hoisted(
    X, PK, SC, ZP, Y,
    K, N,
    stride_pk, stride_meta,
    GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)
    half = K // 2
    hg = half // GROUP_SIZE
    pk_row = PK + n * stride_pk
    sc_row = SC + n * stride_meta
    zp_row = ZP + n * stride_meta

    # BLOCK_K is a whole number of groups; iterate group-by-group inside the block so
    # the scale and zero-point are scalars, loaded once each.
    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_K // GROUP_SIZE

    total = tl.zeros([1], dtype=tl.float32)

    for j0 in range(0, half, BLOCK_K):
        for gi in range(GROUPS_PER_BLOCK):
            base = j0 + gi * GROUP_SIZE
            offs = base + tl.arange(0, GROUP_SIZE)
            m = offs < half
            gidx = base // GROUP_SIZE

            b = tl.load(pk_row + offs, mask=m, other=0)
            q_lo = (b & 0xF).to(tl.float32)
            q_hi = ((b >> 4) & 0xF).to(tl.float32)

            x_lo = tl.load(X + offs, mask=m, other=0.0).to(tl.float32)
            x_hi = tl.load(X + half + offs, mask=m, other=0.0).to(tl.float32)

            # scalars: one load each per group instead of GROUP_SIZE loads
            s_lo = tl.load(sc_row + gidx).to(tl.float32)
            z_lo = tl.load(zp_row + gidx).to(tl.float32)
            s_hi = tl.load(sc_row + hg + gidx).to(tl.float32)
            z_hi = tl.load(zp_row + hg + gidx).to(tl.float32)

            # one FMA per element into each running sum, scale applied once
            qx_lo = tl.sum(q_lo * x_lo, axis=0)
            sx_lo = tl.sum(x_lo, axis=0)
            qx_hi = tl.sum(q_hi * x_hi, axis=0)
            sx_hi = tl.sum(x_hi, axis=0)

            total += s_lo * (qx_lo - z_lo * sx_lo) + s_hi * (qx_hi - z_hi * sx_hi)

    tl.store(Y + n, tl.sum(total, axis=0).to(Y.dtype.element_ty))


def gemv_hoisted(x, packed, scales, zeros, K, group):
    x = x.reshape(-1)
    N = packed.shape[0]
    y = torch.empty(N, device=x.device, dtype=torch.float16)
    _gemv_hoisted[(N,)](
        x, packed, scales, zeros, y,
        K, N,
        packed.stride(0), scales.stride(0),
        GROUP_SIZE=group,
    )
    return y.view(1, N)


# ---------------------------------------------------------------------------

def main():
    import int4_gemv_asym as base

    print(torch.cuda.get_device_name(0))

    # warm the card so clocks are boosted before anything is timed
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    for _ in range(200):
        torch.mm(a, a)
    torch.cuda.synchronize()

    for group in (32, 64, 128):
        print(f"\n=== GROUP={group} ===")
        for K, N in ((4096, 4096), (4096, 11008), (11008, 4096)):
            torch.manual_seed(0)
            w = torch.randn(N, K, device="cuda", dtype=torch.float16)
            x = torch.randn(1, K, device="cuda", dtype=torch.float16)
            packed, sc, zp = quantize_int4_asym(w, group=group)

            # correctness against fp32 of the same quantized weights
            w_deq = dequantize_int4_asym(packed, sc, zp, K, group=group)
            ref = torch.nn.functional.linear(x.float(), w_deq.float())
            out = gemv_hoisted(x, packed, sc, zp, K, group)
            err = ((out.float() - ref).abs().max() / ref.abs().max()).item()

            base.GROUP = group
            t_old = sorted(
                triton.testing.do_bench(
                    lambda: base._gemv_int4_asym[(N,)](
                        x.reshape(-1), packed, sc, zp,
                        torch.empty(N, device="cuda", dtype=torch.float16),
                        K, N, packed.stride(0), sc.stride(0), GROUP_SIZE=group,
                    )
                )
                for _ in range(5)
            )[2]
            t_new = sorted(
                triton.testing.do_bench(lambda: gemv_hoisted(x, packed, sc, zp, K, group))
                for _ in range(5)
            )[2]

            status = "PASS" if err < 5e-3 else "FAIL"
            print(
                f"  K={K:<6} N={N:<6} err={err:.2e} {status}   "
                f"per-element {t_old*1000:7.1f} us   hoisted {t_new*1000:7.1f} us   "
                f"{t_old/t_new:5.2f}x"
            )

    print("\nbest config:", _gemv_hoisted.best_config)


if __name__ == "__main__":
    main()
