# Trace Analyzer Tests

Tests for `trace_analyzer.py`, covering kernel classification, layer detection, and end-to-end regression against golden baselines.

## Test Files

| File | Description | Trace files needed? |
|------|-------------|---------------------|
| `test_classifier_unit.py` | Unit tests for `KernelClassifier` (type, stage, simplified name) | No |
| `test_layer_detector_unit.py` | Unit tests for `LayerDetector` (boundary detection, half-layer merging, stage determination) | No |
| `test_regression.py` | Regression tests against golden baselines (stage coverage, layer counts, layer types) | Yes |

## Running Tests

Run from the `profile/` directory (or pass the path explicitly):

```bash
# All tests (unit + regression)
cd /home/yichiche/agent-box/profile
pytest

# Unit tests only (no trace files required)
pytest tests/test_classifier_unit.py tests/test_layer_detector_unit.py

# Regression tests only
pytest tests/test_regression.py

# Custom trace file directory (default: /home/yichiche/profile-baseline)
pytest --trace-dir /path/to/traces

# Force re-parse traces instead of using pickle cache
pytest --regen-snapshots

# Include slow-marked tests
pytest -m ""

# Verbose output
pytest -v
```

## Baselines and Snapshots

- **`baselines/`** — Golden baseline JSON files (expected metrics per model). Checked into git.
- **`snapshots/`** — Pickled `AnalysisResult` objects (cached parsed traces). Regenerated from trace files as needed.

### Regenerating Baselines

Use `generate_baselines.py` to regenerate golden baselines from trace files:

```bash
# All models
python tests/generate_baselines.py --trace-dir /path/to/traces --all

# Single model
python tests/generate_baselines.py --trace-dir /path/to/traces --model dsr1_mxfp4_conc8

# Only regenerate pickle caches (skip baseline JSON)
python tests/generate_baselines.py --trace-dir /path/to/traces --all --cache-only

# List registered models
python tests/generate_baselines.py --list
```

## Registered Models

| Model | Status | Architecture |
|-------|--------|-------------|
| `dsr1_mxfp4_conc8` | mature | MLA, 3 FC + 58 MoE layers |
| `dsr1_fp8_conc1` | mature | MLA, 3 FC + 58 MoE layers |
| `dsr1_mxfp4_conc1` | mature | MLA, 3 FC + 58 MoE layers |
| `dsr1_mxfp4_rocm700_conc8` | mature | MLA, 3 FC + 58 MoE layers |
| `grok2` | partial | MHA, alternating attn/MoE half-layers |
| `qwen3_coder_next` | partial | GDN+MHA hybrid, 3 GDN+MoE + 1 MHA+MoE per 4 layers |

Models with `known_broken` status are automatically marked `xfail` in pytest.
