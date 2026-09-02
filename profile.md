# Profile Analysis Guide

You are an expert at analyzing SGLang GPU profiling data. Use this guide to understand the profiling toolchain, locate the right files, and answer questions about kernel performance, layer structure, and model behavior.

## Profiling Toolchain Overview

The primary tool is `trace_module_analyzer.py`, which uses nn.Module correlation to classify GPU kernels by their owning module rather than regex pattern matching.

```
Raw Trace (.trace.json.gz)
  │
  ├─► trace_module_analyzer.py ──► report.xlsx  (module-level kernel breakdown)
  │     ├── reads kernel_categories.csv for classification rules
  │     ├── optionally applies fix_rocm_trace_flow.py for ROCm traces
  │     └── optionally generates interactive HTML via visualize_module_tree.py (--model-info)
  │
  ├─► visualize_module_tree.py ──► interactive HTML with tree-table + architecture diagram
  │     (standalone from analysis.xlsx, or triggered by --model-info)
  │
  └─► compare_analysis.py ──► comparison report between two analysis.xlsx files
```

## Tools Reference

### trace_module_analyzer.py — Primary Trace Parser

Correlation-based GPU kernel classification using nn.Module hierarchy from PyTorch profiler traces. Works with both LLM traces (CPU-only module spans) and diffusion model traces (with GPU kernel events).

```bash
# Generate an Excel report
python3 trace_module_analyzer.py trace.json.gz -o report.xlsx

# Include detail sheets for up to 10 module types
python3 trace_module_analyzer.py trace.json.gz -o report.xlsx --max-detail-modules 10

# Show kernel-by-kernel detail for a specific module type
python3 trace_module_analyzer.py trace.json.gz --detail-module WanTransformerBlock

# Pick the 5th occurrence instead of the median
python3 trace_module_analyzer.py trace.json.gz --detail-module WanTransformerBlock --module-index 5

# Generate interactive module tree HTML and serve it
python3 trace_module_analyzer.py trace.json.gz --model-info

# With Excel report + visualization on custom port
python3 trace_module_analyzer.py trace.json.gz -o report.xlsx --model-info --port 9000

# Disable automatic ROCm trace fix
python3 trace_module_analyzer.py trace.json.gz --no-rocm-fix
```

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `trace_file` | *(required)* | Path to trace file (`.json.gz` or `.json`) |
| `-o`, `--output` | None | Output Excel report path (`.xlsx`) |
| `--max-detail-modules` | 3 | Number of module types to generate detail sheets for (0=all) |
| `--detail-module` | None | Specify module types for kernel-by-kernel detail |
| `--module-index` | median | Which occurrence of the module to show detail for |
| `--detail-instance` | None | Instance ids to emit detail sheets for (paired with `--detail-module`) |
| `--list-passes` | off | Print the forward-pass table (index, kind, seqlens, attention kernels) and exit |
| `--phase-index` | None | Restrict to forward passes: `prefill`/`extend`/`decode`, root name, integer index, `NAME#N`/`NAME#N-M`, or `kernel:SUBSTRING` (see [Separating prefill / extend / decode](#separating-prefill--extend--decode---list-passes---phase-index)) |
| `--model-info` | off | Generate interactive module tree HTML and start HTTP server |
| `--port` | 8765 | HTTP server port for `--model-info` |
| `--no-rocm-fix` | off | Disable automatic ROCm trace fix |
| `-v`, `--verbose` | off | Enable debug logging |

### kernel_categories.csv — Kernel Classification Rules

Editable CSV that maps kernel names to categories (e.g. attention, gemm, communication). Each row has a `category` and a `pattern` (regex alternation). Rows are matched top-to-bottom; first match wins.

To add a new pattern, append `|yourpattern` to the relevant row. To add a new category, add a new row. Order matters — place more specific categories above general ones.

### fix_rocm_trace_flow.py — ROCm Trace Fix

Fixes missing CUDA-graph flow events in ROCm/MI355 traces. Automatically applied by `trace_module_analyzer.py` unless `--no-rocm-fix` is passed. Can also be used standalone:

```bash
python3 fix_rocm_trace_flow.py trace.json.gz -o trace_fixed.json.gz
python3 fix_rocm_trace_flow.py trace.json.gz --in-place
```

### visualize_module_tree.py — Interactive Module Tree Visualizer

Reads an `analysis.xlsx` (from `trace_module_analyzer.py`) and generates a self-contained interactive HTML with two views:
1. **Module Tree** — expandable tree-table with timing, kernel counts, and breakdown mini-bars
2. **Architecture** — flowchart-style diagram with folded repeated layers (e.g. decoder blocks)

Used by the `--model-info` flag in `trace_module_analyzer.py`, or standalone.

```bash
# Generate HTML from an existing analysis.xlsx
python3 visualize_module_tree.py analysis.xlsx

# Generate and serve via HTTP
python3 visualize_module_tree.py analysis.xlsx --serve

# Custom output path and port
python3 visualize_module_tree.py analysis.xlsx -o tree.html --serve --port 9000
```

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `xlsx_path` | *(required)* | Path to `analysis.xlsx` from `trace_module_analyzer.py` |
| `-o`, `--output` | `module_tree.html` | Output HTML path |
| `--serve` | off | Start HTTP server after generating HTML |
| `--port` | 8765 | Server port |

### kernel_projection.py — Kernel Improvement Projector

Estimates TTFT/ITL impact from kernel-level improvements.

### Separating prefill / extend / decode (`--list-passes`, `--phase-index`)

In a chunked-prefill trace **every eager forward pass is the same root**
(`nn.Module: <Model>_0`), so a plain `--phase-index <name>` keeps all of them
and the report ends up dominated by the first prefill chunk plus decode.

Start with the pass table:

```bash
python3 trace_module_analyzer.py trace.json.gz --list-passes
```

```
 idx  root                     rel_ts(ms) kind        q       kv   prefix  attention kernels
  801  Qwen3_5MoeForCausalLM_0 #12 12720.5 prefill 16384    16384        0  aiter::fmha_fwd_hd256_fp8_causal_group_gfx950 x15 (14ms)
  803  Qwen3_5MoeForCausalLM_0 #13 13187.8 extend  16384    32768    16384  _ZN7ck_tile6kentryILi1ENS_38FmhaBatchPrefillWi x15 (140ms)
       ... decode x777             8938.8 decode                           CudaGraphReplay_Draft x518, CudaGraphReplay_Target x259 (3505ms)
```

`kind` comes from the attention op's own `max_seqlen_q` / `max_seqlen_k`
arguments: `max_seqlen_k > max_seqlen_q` means the batch carries a prefix, which
*is* the definition of extend. **It never looks at which kernel ran** — that
would be circular, since the usual reason to split the phases is to check
whether extend got routed to the kernel you expect. Reading `kind` and
`attention kernels` side by side is how you spot an extend pass that was never
re-routed (above: extend is still on the old `ck_tile` paged-KV kernel).

The op is matched structurally, not by name — q/k/v are the first three inputs,
k and v share a shape, q and k share a head_dim — so it survives a backend swap.
CUDA-graph replay roots are decode by construction; consecutive ones are
collapsed into a single row.

This mirrors what SGLang branches on itself (`aiter_backend.forward_extend`
tests `any(forward_batch.extend_prefix_lens_cpu)`), read from the seqlens rather
than from the branch taken. Both are batch-wide reductions, so the one blind
spot is a mixed batch whose longest-q request has no prefix while a short-q one
does — `max_q == max_k` and it reads as prefill. The q/kv columns make that
case visible.

Then select:

```bash
python3 trace_module_analyzer.py trace.json.gz -o analysis_prefill.xlsx --phase-index prefill
python3 trace_module_analyzer.py trace.json.gz -o analysis_extend.xlsx  --phase-index extend
python3 trace_module_analyzer.py trace.json.gz -o analysis_decode.xlsx  --phase-index decode

# or pick exact passes off the # column: NAME#N, NAME#N,M, NAME#N-M
python3 trace_module_analyzer.py trace.json.gz -o analysis.xlsx \
    --phase-index 'Qwen3_5MoeForCausalLM_0#1-4' \
    --detail-module Qwen3_5AttentionDecoderLayer --detail-instance 5
```

All forms compose with `--detail-module` / `--detail-instance` as usual.

`kernel:SUBSTRING` also exists (keep every root that launched a matching
kernel). Use it as a **cross-check** — "which passes ran this kernel?" — not as
a phase classifier.

**MTP/EAGLE runs:** the target model and the draft head are *separate* roots —
the CPU blocks on the sampler sync between them, so the head starts long after
the target span ends and carries the last layer's attention kernel. It is
classified on its own seqlens, and the `NAME#N` form additionally pulls in every
non-`CudaGraphReplay` root that starts before the next same-named root. Miss
this and you get 15 of 16 attention layers.

**Eager decode:** if decode is not CUDA-graph captured it has no
`CudaGraphReplay` root and shows up as `extend` (a target-verify step does carry
a prefix). The `q` column disambiguates — it is the speculation depth, not a
chunk size.

### extract_phase_trace.py — Forward-Pass Trace Slicer

Does the same selection but writes a **smaller trace file** instead of a report
— use it when the full capture is too big to open in Perfetto, or to hand a
colleague just the interesting passes. For analysis, prefer `--phase-index`
above: same numbers, one command, no intermediate file.

It labels each eager forward pass by the attention kernel it launched, keeps the
passes you ask for, and follows `correlation` ids so the GPU kernels come along
— GPU work lags its CPU span by a whole iteration on long-context runs, so a
naive time cut would lose it.

```bash
python3 extract_phase_trace.py trace.json.gz --list            # pass table
python3 extract_phase_trace.py trace.json.gz -o EXTEND_only.trace.json.gz \
    --select-kernel extend
python3 extract_phase_trace.py trace.json.gz -o slice.json.gz --select-pass 1 2 3 4
```

| Flag | Default | Description |
|---|---|---|
| `--list` | off | Print the pass table and exit |
| `--root-module` | `Qwen3_5MoeForCausalLM` `Qwen3_5ForCausalLM` | Repeatable; first is the main model, the rest are auxiliary roots (MTP/draft head) |
| `--kernel TAG=SUBSTR` | `prefill=fmha_fwd_hd256`, `extend=FmhaBatchPrefillWithPagedKV` | Attention kernel used to label a pass |
| `--select-kernel` | — | Keep passes carrying this label |
| `--select-pass` | — | Keep these pass indices |
| `--skip-passes` / `--max-passes` | 0 / all | Drop warm-up passes / cap the count |
| `--window` | `iteration` | `iteration` = main span + the MTP/draft span of the same iteration; `span` = main span only |

### compare_per_pass.py — Per-Forward-Pass Normalized Diff

`compare_analysis.py` diffs absolute totals, which only works when both traces
hold the same number of forward passes. When two runs chunk a long prompt
differently (25 extend chunks before a change vs 4 after), normalize first:

```bash
python3 compare_per_pass.py before.xlsx after.xlsx --passes 25 4 \
    --labels BEFORE AFTER --category attention
```

Pass counts come straight from `extract_phase_trace.py --list`.

### compare_analysis.py — Report Comparison

Action-oriented diff of two `analysis.xlsx` reports. Designed for two use cases:
- **Same-platform diff** (e.g. MI355 aiter vs triton): detects kernel replacements (GONE/NEW pairs), same-kernel time changes, and per-category breakdown.
- **Cross-platform diff** (e.g. MI355 vs B200): category-first drill-down to find biggest gaps, then top kernels within each.

```bash
# Terminal report (labels auto-detected from directory names)
python3 compare_analysis.py baseline/analysis.xlsx target/analysis.xlsx

# Custom labels + Excel output
python3 compare_analysis.py a.xlsx b.xlsx --labels MI355-aiter MI355-triton -o diff.xlsx
```

#### Report Sections

1. **Executive Summary** — total kernel time delta and % change
2. **Phase Breakdown** — prefill vs decode split with per-block category and kernel diffs
3. **Kernel Replacements** — side-by-side table of GONE/NEW kernel pairs (with cross-category matching for recategorized kernels), showing net time impact
4. **Category Drill-Down** — per-category delta sorted by magnitude (decreases first, then increases), with top contributing kernels and % of category delta
5. **Kernel Time Changes** — same-name kernels present in both traces with different time, sorted by |delta|
6. **Actionable Summary** — compact replacement table and top same-kernel time changes

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `baseline` | *(required)* | Baseline analysis.xlsx (file A) |
| `target` | *(required)* | Target analysis.xlsx (file B) |
| `-o`, `--output` | None | Output Excel diff report (`.xlsx`) |
| `--labels` | auto-detect | Labels for the two files (e.g. `--labels BF16 FP8`) |

### evaluate_module_parsing.py — Quality Evaluator

Evaluates the output of `trace_module_analyzer.py`. Used by the perf-regression pipeline for quality gating.

```bash
python3 evaluate_module_parsing.py report.xlsx --json
```

## Output File Formats

### trace_module_analyzer.py Output (report.xlsx)

An Excel workbook with module-level kernel breakdown:

- **Summary sheet**: Overall stats — total time, module type breakdown, top kernels per module type
- **Module type sheets**: Per-module-type detail with kernel lists, timing, and percentages
- **Module tree HTML** (with `--model-info`): Interactive visualization generated by `visualize_module_tree.py`, served via HTTP

### evaluation.json — Quality Assessment

JSON with structural scores (S1-S4), per-group metrics, and overall composite score. Produced by `evaluate_module_parsing.py` from `trace_module_analyzer.py` output. Used by perf-regression pipeline for quality gating.

## Typical Benchmark Run Directory Layout

Benchmark runs are stored under `/home/yichiche/benchmark_runs/`. A typical profiling run directory:

```
<version>_<hardware>_<date>_TP<N>_profile/
├── bench_c1.log, bench_c2.log, ...   (server logs per concurrency level)
├── bench_results.jsonl                 (detailed benchmark results)
├── bench_summary.csv                   (throughput/latency per concurrency)
├── server.log, client.log              (server and client logs)
├── orchestrator_output.log             (orchestration log)
├── version_snapshot.json               (version metadata)
└── trace_analysis/
    ├── *.trace.json.gz                 (raw trace file)
    ├── analysis.xlsx                   (trace_module_analyzer output)
    ├── trace_analyzer.log              (analysis log)
    └── evaluation.json                 (quality scores)
```

## How to Analyze Profiling Data

### Step 1: Locate the files

If the user provides a path, use it directly. Otherwise, look for benchmark runs:
```bash
ls /home/yichiche/benchmark_runs/
```

Within a run directory, profiling outputs are in the `trace_analysis/` subdirectory.

### Step 2: Run trace_module_analyzer

For new analysis, use `trace_module_analyzer.py`:
```bash
python3 trace_module_analyzer.py /path/to/trace.json.gz -o report.xlsx -v
```

For existing analysis, read the `analysis.xlsx` and `evaluation.json` files in the `trace_analysis/` subdirectory.

### Step 3: Answer the user's question

Common questions and how to answer them:

| Question | Where to look |
|----------|---------------|
| "What modules take the most time?" | trace_module_analyzer summary sheet |
| "What kernels run in module X?" | trace_module_analyzer --detail-module X |
| "What's the model architecture?" | trace_module_analyzer --model-info, or visualize_module_tree standalone |
| "What's the prefill/decode split?" | trace_module_analyzer summary sheet (phase column) |
| "Is the trace parsing reliable?" | evaluate_module_parsing --json overall score |
| "Which layers are outliers?" | evaluate_module_parsing --json structural rules |
| "How does this compare to baseline?" | compare_analysis.py with two report files (shows replacements, time changes, category drill-down) |

## Key Concepts

- **Prefill**: The initial phase processing the full input prompt. Typically compute-bound with large batch GEMMs.
- **Decode**: Auto-regressive token generation phase. Typically memory-bandwidth-bound.
- **MLA (Multi-head Latent Attention)**: DeepSeek-V3 style compressed KV attention.
- **MHA (Multi-head Attention)**: Standard multi-head attention (Llama, Qwen, Grok-2).
- **GDN (Gated Delta Network)**: Qwen3-Coder-Next linear attention variant.
- **MoE (Mixture of Experts)**: Sparse expert layers with routing (DeepSeek, Grok-2, Qwen3).
- **FC (Fully Connected)**: Dense MLP layers (first/last few layers in MoE models).
- **ALLREDUCE**: Collective communication kernel marking tensor-parallel synchronization points.

If the user provided `$ARGUMENTS`, treat it as their specific question or the path to profiling data to analyze.
