# torch-profiler-parser

Tools for parsing and analyzing PyTorch Profiler traces. Classifies GPU kernels by their owning `nn.Module` using CPU-GPU correlation, producing structured Excel reports with per-module kernel breakdowns.

Built for analyzing large model inference (LLMs, diffusion models, MoE architectures) on NVIDIA and AMD GPUs.

## Tools

| Tool | Description |
|------|-------------|
| `trace_module_analyzer.py` | Primary trace parser. Correlates CPU module spans with GPU kernels to produce per-module kernel breakdown reports. |
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
