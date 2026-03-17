# torch-profiler-parser

Tools for parsing and analyzing PyTorch Profiler traces. Classifies GPU kernels by their owning `nn.Module` using CPU-GPU correlation, producing structured Excel reports with per-module kernel breakdowns.

Built for analyzing large model inference (LLMs, diffusion models, MoE architectures) on NVIDIA and AMD GPUs.

## Tools

| Tool | Description |
|------|-------------|
| `trace_module_analyzer.py` | Primary trace parser. Correlates CPU module spans with GPU kernels to produce per-module kernel breakdown reports. |
| `model_inspector.py` | Static model structure analysis from Python source, plus trace-based architecture diagrams. |
| `evaluate_module_parsing.py` | Evaluates trace parsing quality with structural scoring (S1-S4). |
| `fix_rocm_trace_flow.py` | Fixes missing CUDA-graph flow events in ROCm traces. Auto-applied by the analyzer. |

## Quick Start

```bash
git clone https://github.com/yichiche/torch-profiler-parser.git
cd torch-profiler-parser
pip install -e .

# Analyze a trace
python3 trace_module_analyzer.py trace.json.gz -o report.xlsx

# With HuggingFace model config for enrichment
python3 trace_module_analyzer.py trace.json.gz -o report.xlsx --config model_path/config.json

# Evaluate parsing quality
python3 evaluate_module_parsing.py report.xlsx --json
```

## Requirements

- Python 3.8+
- `openpyxl` (for Excel report generation)

## Documentation

See [profile.md](profile.md) for detailed usage, output formats, and analysis workflows.

## License

MIT License. See [LICENSE](LICENSE) for details.
