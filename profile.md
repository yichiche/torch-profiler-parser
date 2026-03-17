# Profile Analysis Guide

You are an expert at analyzing SGLang GPU profiling data. Use this guide to understand the profiling toolchain, locate the right files, and answer questions about kernel performance, layer structure, and model behavior.

## Profiling Toolchain Overview

The primary tool is `trace_module_analyzer.py`, which uses nn.Module correlation to classify GPU kernels by their owning module rather than regex pattern matching.

```
Raw Trace (.trace.json.gz)
  │
  ├─► trace_module_analyzer.py ──► report.xlsx  (module-level kernel breakdown)
  │     (optionally applies fix_rocm_trace_flow.py for ROCm traces)
  │
  └─► model_inspector.py ──► model structure tree / architecture diagrams
        (standalone, or enriches trace_module_analyzer output)
```

## Tools Reference

### trace_module_analyzer.py — Primary Trace Parser

Correlation-based GPU kernel classification using nn.Module hierarchy from PyTorch profiler traces. Works with both LLM traces (CPU-only module spans) and diffusion model traces (with GPU kernel events).

```bash
# Basic analysis
python3trace_module_analyzer.py trace.json.gz -o report.xlsx

# Enrich with HuggingFace config (adds head counts, vocab size, expert config)
python3trace_module_analyzer.py trace.json.gz -o report.xlsx --config config.json

# Show kernel-by-kernel detail for a specific module type
python3trace_module_analyzer.py trace.json.gz --detail-module WanTransformerBlock

# Show specific instance
python3trace_module_analyzer.py trace.json.gz --detail-module WanTransformerBlock --detail-instance 5

# Show full module tree
python3trace_module_analyzer.py trace.json.gz --show-tree

# Disable automatic ROCm trace fix
python3trace_module_analyzer.py trace.json.gz --no-rocm-fix
```

### fix_rocm_trace_flow.py — ROCm Trace Fix

Fixes missing CUDA-graph flow events in ROCm/MI355 traces. Automatically applied by `trace_module_analyzer.py` unless `--no-rocm-fix` is passed. Can also be used standalone:

```bash
python3fix_rocm_trace_flow.py trace.json.gz -o trace_fixed.json.gz
python3fix_rocm_trace_flow.py trace.json.gz --in-place
```

### model_inspector.py — Model Structure Inspector

Static model structure analysis from Python source code, plus trace-based architecture diagrams.

```bash
# List classes in a model file
python3model_inspector.py deepseek_v2.py --list-classes

# Show module hierarchy tree
python3model_inspector.py deepseek_v2.py --root DeepseekV2ForCausalLM

# PyTorch-profiler-style tree with layer instances expanded
python3model_inspector.py deepseek_v2.py --profiler-tree --config config.json

# Generate architecture block diagram from a trace file
python3model_inspector.py --trace trace.json.gz --arch-diagram
python3model_inspector.py --trace trace.json.gz --arch-diagram --detailed
```

### evaluate_module_parsing.py — Quality Evaluator

Evaluates the output of `trace_module_analyzer.py`. Used by the perf-regression pipeline for quality gating.

```bash
python3evaluate_module_parsing.py report.xlsx --json
```

## Output File Formats

### trace_module_analyzer.py Output (report.xlsx)

An Excel workbook with module-level kernel breakdown:

- **Summary sheet**: Overall stats — total time, module type breakdown, top kernels per module type
- **Module type sheets**: Per-module-type detail with kernel lists, timing, and percentages
- **Model Info sheet** (with `--config`): HuggingFace model configuration details

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
python3trace_module_analyzer.py /path/to/trace.json.gz -o report.xlsx -v
```

For existing analysis, read the `analysis.xlsx` and `evaluation.json` files in the `trace_analysis/` subdirectory.

### Step 3: Answer the user's question

Common questions and how to answer them:

| Question | Where to look |
|----------|---------------|
| "What modules take the most time?" | trace_module_analyzer summary sheet |
| "What kernels run in module X?" | trace_module_analyzer --detail-module X |
| "What's the model architecture?" | model_inspector --profiler-tree or --arch-diagram |
| "What's the prefill/decode split?" | trace_module_analyzer summary sheet |
| "Is the trace parsing reliable?" | evaluate_module_parsing --json overall score |
| "Which layers are outliers?" | evaluate_module_parsing --json structural rules |

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
