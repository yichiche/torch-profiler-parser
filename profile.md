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
  │     └── optionally adds Model Info tab via model_inspector.py (--model-info)
  │
  ├─► model_inspector.py ──► model structure tree / architecture diagrams
  │     (standalone, or enriches trace_module_analyzer output)
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

# Include a Model Info tab with architecture summary
python3 trace_module_analyzer.py trace.json.gz -o report.xlsx --model-info

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
| `--model-info` | off | Add a Model Info tab with architecture summary |
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

### model_inspector.py — Model Structure Inspector

Static model structure analysis from Python source code, plus trace-based architecture diagrams. Used by `--model-info` flag in `trace_module_analyzer.py`.

```bash
# List classes in a model file
python3 model_inspector.py deepseek_v2.py --list-classes

# Show module hierarchy tree
python3 model_inspector.py deepseek_v2.py --root DeepseekV2ForCausalLM

# PyTorch-profiler-style tree with layer instances expanded
python3 model_inspector.py deepseek_v2.py --profiler-tree

# Generate architecture block diagram from a trace file
python3 model_inspector.py --trace trace.json.gz --arch-diagram
python3 model_inspector.py --trace trace.json.gz --arch-diagram --detailed
```

### kernel_projection.py — Kernel Improvement Projector

Estimates TTFT/ITL impact from kernel-level improvements.

### compare_analysis.py — Report Comparison

Compares two `analysis.xlsx` files to show regressions and improvements between runs.

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
- **Model Info sheet** (with `--model-info`): Architecture summary and optional diagram from `model_inspector.py`

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
| "What's the model architecture?" | trace_module_analyzer --model-info, or model_inspector standalone |
| "What's the prefill/decode split?" | trace_module_analyzer summary sheet (phase column) |
| "Is the trace parsing reliable?" | evaluate_module_parsing --json overall score |
| "Which layers are outliers?" | evaluate_module_parsing --json structural rules |
| "How does this compare to baseline?" | compare_analysis.py with two report files |

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
