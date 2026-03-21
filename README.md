# torch-profiler-parser

Tools for parsing and analyzing PyTorch Profiler traces. Classifies GPU kernels by their owning `nn.Module` using CPU-GPU correlation, producing structured Excel reports with per-module kernel breakdowns.

Built for analyzing large model inference (LLMs, diffusion models, MoE architectures) on NVIDIA and AMD GPUs.

## Tools

| Tool | Description |
|------|-------------|
| `trace_module_analyzer.py` | Primary trace parser. Correlates CPU module spans with GPU kernels to produce per-module kernel breakdown reports. |
| `compare_analysis.py` | Action-oriented diff of two `analysis.xlsx` reports. Detects kernel replacements, same-kernel time changes, and per-category drill-down. |
| `kernel_categories.csv` | Editable kernel classification rules. Each row maps a regex pattern to a category (e.g. attention, gemm, communication). |
| `model_inspector.py` | Static model structure analysis and trace-based architecture diagrams. Used by `--model-info`. |
| `kernel_projection.py` | Project kernel-level improvements to estimate TTFT/ITL impact. |
| `evaluate_module_parsing.py` | Evaluates trace parsing quality with structural scoring (S1-S4). |
| `fix_rocm_trace_flow.py` | Fixes missing CUDA-graph flow events in ROCm traces. Auto-applied by the analyzer. |

## Quick Start

```bash
git clone https://github.com/yichiche/torch-profiler-parser.git
cd torch-profiler-parser
pip install -e .

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

# Evaluate parsing quality
python3 evaluate_module_parsing.py report.xlsx --json
```

## CLI Flags

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

## Comparing Two Reports

`compare_analysis.py` diffs two `analysis.xlsx` files and produces an action-oriented report answering "where does the performance difference come from?"

Two use cases:
- **Same-platform diff** (e.g. MI355 aiter vs triton backend): identifies kernel replacements, same-kernel time changes, and per-category breakdown.
- **Cross-platform diff** (e.g. MI355 vs B200): category-first drill-down to find the biggest gaps, then top kernels within each category.

```bash
# Terminal report
python3 compare_analysis.py baseline/analysis.xlsx target/analysis.xlsx

# With custom labels
python3 compare_analysis.py a.xlsx b.xlsx --labels MI355-aiter MI355-triton

# Terminal + Excel diff report
python3 compare_analysis.py a.xlsx b.xlsx -o diff.xlsx --labels Baseline Optimized
```

Report sections:
1. **Executive Summary** — total kernel time delta and % change
2. **Phase Breakdown** — prefill vs decode split (if phase data available)
3. **Kernel Replacements** — side-by-side table of GONE/NEW kernel pairs with net impact
4. **Category Drill-Down** — per-category delta sorted by magnitude (decreases first, then increases), with top contributing kernels
5. **Kernel Time Changes** — same-name kernels with different time, sorted by |delta|
6. **Actionable Summary** — compact table of replacements and top contributors

## Kernel Categories

Kernel classification rules live in `kernel_categories.csv`. Each row has a `category` and a `pattern` (regex alternation). Rows are matched top-to-bottom; first match wins.

To add a new pattern, append `|yourpattern` to the relevant row. To add a new category, add a new row. Order matters — place more specific categories above general ones.

## Requirements

- Python 3.8+
- `openpyxl` (for Excel report generation)
- `matplotlib` (optional, for architecture diagram PNG in `--model-info`)

## Documentation

See [profile.md](profile.md) for detailed usage, output formats, and analysis workflows.

## License

MIT License. See [LICENSE](LICENSE) for details.
