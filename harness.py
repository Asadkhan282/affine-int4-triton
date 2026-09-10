"""
Single-GPU LLM inference benchmark.

Measures prefill (TTFT) and decode (inter-token latency) separately, using a manual
KV-cache loop rather than `model.generate()`, so that per-token timings are measured
rather than averaged out of a single wall-clock number.

Usage:
    python bench/harness.py --model meta-llama/Llama-3.1-8B-Instruct \
        --dtype bfloat16 --mode eager \
        --batch-sizes 1,4,16 --prompt-len 512 --gen-len 128 \
        --out results/baseline.json
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    p.add_argument("--mode", default="eager", choices=["eager", "compile"])
    p.add_argument("--batch-sizes", default="1,4,16")
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--iters", type=int, default=5, help="measured repetitions")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", default="results/run.json")
    p.add_argument("--label", default=None, help="name for this config in reports")
    return p.parse_args()


def device_info():
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        "capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
    }


def model_shape(cfg, dtype_size, param_bytes):
    """Static facts we need to convert latency into bandwidth."""
    n_heads = getattr(cfg, "num_attention_heads")
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    hidden = getattr(cfg, "hidden_size")
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    return {
        "num_layers": cfg.num_hidden_layers,
        "num_attention_heads": n_heads,
        "num_kv_heads": n_kv,
        "hidden_size": hidden,
        "head_dim": head_dim,
        "param_bytes": param_bytes,
        "dtype_bytes": dtype_size,
    }


def kv_bytes(shape, batch, seq_len):
    """Bytes of KV cache read during one decode step."""
    return (
        2
        * shape["num_layers"]
        * shape["num_kv_heads"]
        * shape["head_dim"]
        * seq_len
        * batch
        * shape["dtype_bytes"]
    )


class Timer:
    """CUDA-event timer. Returns milliseconds."""

    def __init__(self):
        self.start_ev = torch.cuda.Event(enable_timing=True)
        self.end_ev = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_ev.record()
        return self

    def __exit__(self, *exc):
        self.end_ev.record()
        torch.cuda.synchronize()
        self.ms = self.start_ev.elapsed_time(self.end_ev)
        return False


@torch.inference_mode()
def run_once(model, input_ids, gen_len):
    """One prefill + gen_len-1 decode steps. Returns (ttft_ms, [per_token_ms])."""
    with Timer() as t_prefill:
        out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    per_token = []
    for _ in range(gen_len - 1):
        with Timer() as t_step:
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        per_token.append(t_step.ms)

    return t_prefill.ms, per_token


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    idx = min(int(round((q / 100.0) * (len(s) - 1))), len(s) - 1)
    return s[idx]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible. This harness must run on a GPU box.")

    dtype = DTYPES[args.dtype]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="cuda:0", attn_implementation="sdpa"
    )
    model.eval()

    if args.mode == "compile":
        model.forward = torch.compile(model.forward, mode="reduce-overhead")

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    shape = model_shape(cfg, torch.finfo(dtype).bits // 8, param_bytes)

    vocab = getattr(cfg, "vocab_size", 32000)
    results = []

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Deterministic synthetic prompt. Avoids tokenizer-length surprises and keeps
        # the shape identical across runs, which is what makes the diff meaningful.
        input_ids = torch.randint(
            0, vocab, (bs, args.prompt_len), device="cuda", dtype=torch.long
        )

        for _ in range(args.warmup):
            run_once(model, input_ids, min(args.gen_len, 8))
        torch.cuda.synchronize()

        ttfts, all_tokens = [], []
        wall_start = time.perf_counter()
        for _ in range(args.iters):
            ttft, per_token = run_once(model, input_ids, args.gen_len)
            ttfts.append(ttft)
            all_tokens.extend(per_token)
        wall = time.perf_counter() - wall_start

        mean_itl = statistics.mean(all_tokens)
        total_tokens = args.iters * args.gen_len * bs
        avg_ctx = args.prompt_len + args.gen_len / 2

        record = {
            "batch_size": bs,
            "prompt_len": args.prompt_len,
            "gen_len": args.gen_len,
            "ttft_ms_mean": round(statistics.mean(ttfts), 3),
            "ttft_ms_p95": round(percentile(ttfts, 95), 3),
            "itl_ms_mean": round(mean_itl, 4),
            "itl_ms_p50": round(percentile(all_tokens, 50), 4),
            "itl_ms_p95": round(percentile(all_tokens, 95), 4),
            "decode_tokens_per_s": round(1000.0 / mean_itl * bs, 2),
            "e2e_tokens_per_s": round(total_tokens / wall, 2),
            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            # Bytes that must cross the memory bus for one decode step: every weight,
            # plus the KV cache for the current context. This is the number that makes
            # decode memory-bound.
            "decode_bytes_per_step": param_bytes + kv_bytes(shape, bs, int(avg_ctx)),
        }
        record["achieved_gb_s"] = round(
            record["decode_bytes_per_step"] / (mean_itl / 1000.0) / 1e9, 1
        )
        results.append(record)
        print(
            f"bs={bs:<4} TTFT {record['ttft_ms_mean']:8.2f} ms   "
            f"ITL p50 {record['itl_ms_p50']:7.3f} ms   "
            f"{record['decode_tokens_per_s']:8.1f} tok/s   "
            f"{record['achieved_gb_s']:7.1f} GB/s   "
            f"{record['peak_memory_gb']:.2f} GB"
        )

    payload = {
        "label": args.label or f"{Path(args.model).name}-{args.dtype}-{args.mode}",
        "model": args.model,
        "dtype": args.dtype,
        "mode": args.mode,
        "device": device_info(),
        "model_shape": shape,
        "results": results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
