"""
Diff two harness runs into the table and chart that go in the writeup.

Usage:
    python bench/compare.py results/baseline.json results/optimised.json \
        --chart results/delta.png
"""

import argparse
import json


def pct_change(before, after, lower_is_better=True):
    if before in (None, 0):
        return None
    delta = (after - before) / before * 100.0
    return -delta if lower_is_better else delta


def fmt(v, width=8, prec=2, suffix=""):
    return "n/a".rjust(width) if v is None else f"{v:>{width}.{prec}f}{suffix}"


def guard_comparable(a, b):
    warnings = []
    if a["device"]["name"] != b["device"]["name"]:
        warnings.append(f"different GPUs: {a['device']['name']} vs {b['device']['name']}")
    if a["model"] != b["model"]:
        warnings.append(f"different models: {a['model']} vs {b['model']}")
    if a["device"]["torch"] != b["device"]["torch"]:
        warnings.append(
            f"different torch: {a['device']['torch']} vs {b['device']['torch']}"
        )
    return warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("optimised")
    ap.add_argument("--chart", default=None)
    args = ap.parse_args()

    a = json.loads(open(args.baseline).read())
    b = json.loads(open(args.optimised).read())

    for w in guard_comparable(a, b):
        print(f"WARNING: {w}  -- this comparison is not clean, re-run on one box")

    print(f"\nbaseline : {a['label']}")
    print(f"optimised: {b['label']}")
    print(f"device   : {a['device']['name']}\n")

    by_bs = {r["batch_size"]: r for r in b["results"]}
    rows = []

    head = (
        f"{'bs':>4} | {'TTFT ms':>18} | {'ITL p50 ms':>18} | "
        f"{'tok/s':>18} | {'peak GB':>14}"
    )
    print(head)
    print("-" * len(head))

    for base in a["results"]:
        bs = base["batch_size"]
        opt = by_bs.get(bs)
        if opt is None:
            continue

        ttft = pct_change(base["ttft_ms_mean"], opt["ttft_ms_mean"])
        itl = pct_change(base["itl_ms_p50"], opt["itl_ms_p50"])
        tps = pct_change(
            base["decode_tokens_per_s"], opt["decode_tokens_per_s"], lower_is_better=False
        )
        mem = pct_change(base["peak_memory_gb"], opt["peak_memory_gb"])

        print(
            f"{bs:>4} | {base['ttft_ms_mean']:>7.2f}->{opt['ttft_ms_mean']:<7.2f}{ttft:>+5.0f}% | "
            f"{base['itl_ms_p50']:>7.3f}->{opt['itl_ms_p50']:<7.3f}{itl:>+4.0f}% | "
            f"{base['decode_tokens_per_s']:>7.1f}->{opt['decode_tokens_per_s']:<7.1f}{tps:>+4.0f}% | "
            f"{base['peak_memory_gb']:>5.2f}->{opt['peak_memory_gb']:<5.2f}{mem:>+4.0f}%"
        )
        rows.append((bs, base, opt))

    print("\n(+ve = improvement in all columns)")

    if args.chart and rows:
        make_chart(rows, a["label"], b["label"], args.chart)
        print(f"chart written to {args.chart}")


def make_chart(rows, label_a, label_b, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    bss = [r[0] for r in rows]
    x = np.arange(len(bss))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, key, title, ylabel in (
        (axes[0], "itl_ms_p50", "Inter-token latency (p50)", "ms per token"),
        (axes[1], "decode_tokens_per_s", "Decode throughput", "tokens/s"),
    ):
        ax.bar(x - width / 2, [r[1][key] for r in rows], width, label=label_a)
        ax.bar(x + width / 2, [r[2][key] for r in rows], width, label=label_b)
        ax.set_xticks(x, [str(b) for b in bss])
        ax.set_xlabel("batch size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
