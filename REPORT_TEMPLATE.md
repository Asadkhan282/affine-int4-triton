# [Model] inference on [GPU]: [X]% lower latency per token

*[date] — reproduction: [repo link]*

## Summary

One paragraph, numbers first. What you measured, what you changed, what it did.
A reader who stops here should be able to decide whether to keep reading.

> Example shape: "Decode latency for Llama-3.1-8B on an A100-80GB was 42% above what
> the memory bandwidth allows. Fusing the residual add into RMSNorm and enabling CUDA
> graphs cut p50 inter-token latency from 14.2 ms to 9.8 ms at batch 1, a 31%
> reduction, with no change in output logits beyond bf16 noise."

## Setup

Anyone should be able to re-run this. State without exception:

- GPU, exact SKU, and whether SXM or PCIe
- Driver version, CUDA version, `torch.__version__`, `triton.__version__`
- Model and revision hash
- dtype, attention implementation, batch sizes, prompt and generation lengths
- The exact command lines

## Baseline

| bs | TTFT ms | ITL p50 ms | ITL p95 ms | tok/s | peak VRAM | achieved GB/s | % of peak |
|----|---------|------------|------------|-------|-----------|---------------|-----------|
| 1  |         |            |            |       |           |               |           |
| 4  |         |            |            |       |           |               |           |
| 16 |         |            |            |       |           |               |           |

## Where the time actually goes

The Nsight timeline screenshot goes here, with the problem circled. Then say in one
or two sentences what it shows — idle gaps, a dominant kernel, a sync point.

State the roofline conclusion explicitly: memory-bound or compute-bound, and at what
fraction of peak. If you skip this step you are guessing, and experienced readers
will be able to tell.

## The change

Show the diff. Explain *why* it should help in terms of bytes moved or launches
saved, before showing that it did help. A prediction that comes true is far more
convincing than a measurement with a story attached afterwards.

## Results

| bs | ITL p50 before | after | change | tok/s before | after | change |
|----|----------------|-------|--------|--------------|-------|--------|
| 1  |                |       |        |              |       |        |

Chart from `compare.py` here.

## Correctness

Do not skip this section. State how you verified the optimised path produces the same
outputs: max absolute deviation on logits, or a task-level eval score before and
after. A speedup with no correctness evidence reads as a bug to anyone senior.

## What did not work

- Thing you tried, and the number that showed it did not help.
- Thing that helped at batch 1 and regressed at batch 32.

This section is why people will trust the rest of the document.

## Limits of this result

Where it does not generalise: other GPUs, longer contexts, other batch regimes,
other models. Being explicit here costs you nothing and is the difference between
reading as an engineer and reading as a marketer.

---

*Independent consultant. I do inference optimisation and CUDA kernel work —
[contact].*
