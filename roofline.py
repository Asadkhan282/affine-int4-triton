"""
Roofline check: is decode memory-bound or compute-bound, and how close to peak is it?

Run this before writing a single line of kernel code. If decode is already at 80% of
peak HBM bandwidth, no kernel you write will help and the honest answer to the client
is "change the batching strategy, not the kernels."

Usage:
    python bench/roofline.py results/baseline.json
"""

import argparse
import json
import sys

# Peak spec-sheet HBM bandwidth in GB/s. Real achievable is typically 80-90% of these
# even for a perfect streaming kernel, so treat >75% as "bandwidth-saturated".
PEAK_BW_GB_S = {
    "A100-SXM4-80GB": 2039,
    "A100-SXM4-40GB": 1555,
    "A100-PCIE-40GB": 1555,
    "A100-PCIE-80GB": 1935,
    "H100 80GB HBM3": 3350,
    "H100 PCIe": 2000,
    "H200": 4800,
    "L40S": 864,
    "L4": 300,
    "RTX 4090": 1008,
    "RTX 3090": 936,
    "V100-SXM2-32GB": 900,
    "A10G": 600,
    # Free-tier cards (Kaggle / Colab). Note the T4 is Turing: no bfloat16.
    "Tesla T4": 320,
    "Tesla P100": 732,
}


def lookup_peak(device_name):
    for key, bw in PEAK_BW_GB_S.items():
        if key.lower() in device_name.lower():
            return bw, key
    return None, None


def verdict(fraction):
    if fraction is None:
        return "unknown - add this GPU to PEAK_BW_GB_S"
    if fraction >= 0.75:
        return "bandwidth-saturated: kernel work will not help, change batching/KV layout"
    if fraction >= 0.50:
        return "memory-bound, some headroom: fusion and dtype reduction will pay"
    if fraction >= 0.25:
        return "memory-bound with large headroom: look for launch overhead and stalls"
    return "not bandwidth-limited: suspect CPU launch overhead, sync points, or bad kernels"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="JSON produced by harness.py")
    args = ap.parse_args()

    data = json.loads(open(args.results).read())
    dev = data["device"]["name"]
    peak, matched = lookup_peak(dev)

    print(f"device      : {dev}")
    print(f"peak HBM    : {peak} GB/s" + (f"  (matched '{matched}')" if matched else "  (UNKNOWN)"))
    print(f"model       : {data['model']}  [{data['dtype']}, {data['mode']}]")
    print(
        f"weights     : {data['model_shape']['param_bytes'] / 1e9:.2f} GB\n"
    )

    header = f"{'bs':>4} {'ITL p50 ms':>11} {'GB/s':>9} {'% peak':>8}  verdict"
    print(header)
    print("-" * (len(header) + 40))

    for r in data["results"]:
        achieved = r["achieved_gb_s"]
        frac = achieved / peak if peak else None
        pct = f"{frac * 100:6.1f}%" if frac is not None else "     ?"
        print(
            f"{r['batch_size']:>4} {r['itl_ms_p50']:>11.3f} {achieved:>9.1f} {pct:>8}  {verdict(frac)}"
        )

    print(
        "\nnote: bytes/step assumes every weight is read once per token, which holds for\n"
        "dense decode at small batch. For MoE, quantised, or heavily batched serving,\n"
        "adjust decode_bytes_per_step before trusting these percentages."
    )


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        sys.exit(f"missing file: {e.filename}")
