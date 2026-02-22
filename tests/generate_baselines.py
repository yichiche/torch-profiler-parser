#!/usr/bin/env python3
"""Generate golden baseline JSON files and pickle caches for regression tests.

Usage:
    python tests/generate_baselines.py --trace-dir /path/to/traces --all
    python tests/generate_baselines.py --trace-dir /path/to/traces --model dsr1_mxfp4_conc8
    python tests/generate_baselines.py --trace-dir /path/to/traces --model dsr1_mxfp4_conc8 --cache-only
"""

import argparse
import json
import os
import pickle
import sys

# Add parent directory to path so we can import trace_analyzer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace_analyzer import (
    AnalysisResult,
    KernelType,
    LayerType,
    Stage,
    TraceAnalyzer,
)

TRACE_REGISTRY = {
    "dsr1_mxfp4_conc8": {
        "trace_file": "mi355_aiter_0.1.10.post1_rocm720_dsr1-mxfp4_MTP_tp8_IL70kOL200_conc8_1770104060.7451138-TP-0.trace.json.gz",
        "num_hidden_layers": 61,
        "layers_per_block": 1,
        "mtp_qseqlen_decode": True,
        "grid_threshold": 10000,
        "status": "mature",
        "expected_layer_types": {
            "pattern": "first_n_fc_rest_moe",
            "first_n_fc": 3,
        },
        "stage_coverage_min": 95.0,
        "type_coverage_min": 90.0,
    },
    "dsr1_fp8_conc1": {
        "trace_file": "mi355_aiter_0.1.10.post1_rocm720_dsr1-fp8_tp8_IL70kOL200_conc1_1771645120.421048-TP-0.trace.json.gz",
        "num_hidden_layers": 61,
        "layers_per_block": 1,
        "mtp_qseqlen_decode": True,
        "grid_threshold": 10000,
        "status": "mature",
        "expected_layer_types": {
            "pattern": "first_n_fc_rest_moe",
            "first_n_fc": 3,
        },
        "stage_coverage_min": 95.0,
        "type_coverage_min": 50.0,
    },
    "dsr1_mxfp4_conc1": {
        "trace_file": "mi355_aiter_0.1.10.post1_rocm720_dsr1-mxfp4_MTP_tp8_IL70kOL200_conc1_1770135164.3032408-TP-0.trace.json.gz",
        "num_hidden_layers": 61,
        "layers_per_block": 1,
        "mtp_qseqlen_decode": True,
        "grid_threshold": 10000,
        "status": "mature",
        "expected_layer_types": {
            "pattern": "first_n_fc_rest_moe",
            "first_n_fc": 3,
        },
        "stage_coverage_min": 95.0,
        "type_coverage_min": 90.0,
    },
    "dsr1_mxfp4_rocm700_conc8": {
        "trace_file": "mi355_aiter_0.1.9.post1_rocm700_dsr1-mxfp4_MTP_tp8_IL70kOL200_conc8_1770104458.5756042-TP-0.trace.json.gz",
        "num_hidden_layers": 61,
        "layers_per_block": 1,
        "mtp_qseqlen_decode": True,
        "grid_threshold": 10000,
        "status": "mature",
        "expected_layer_types": {
            "pattern": "first_n_fc_rest_moe",
            "first_n_fc": 3,
        },
        "stage_coverage_min": 95.0,
        "type_coverage_min": 90.0,
    },
    "grok2": {
        "trace_file": "mi355_aiter_0.1.10.post1_rocm720_grok2_nonMTP_tp8_IL1kOL1k_conc1_1771648823.7226286-TP-0.trace.json.gz",
        "num_hidden_layers": 64,
        "layers_per_block": 1,
        "mtp_qseqlen_decode": False,
        "grid_threshold": 10000,
        "status": "partial",
        "expected_layer_types": {
            "pattern": "alternating_attn_moe",
        },
        "stage_coverage_min": 50.0,
        "type_coverage_min": 50.0,
    },
    "qwen3_coder_next": {
        "trace_file": "mi355_aiter_0.1.10.post1_rocm720_Qwen3-Coder-Next_nonMTP_tp8_IL70kOL200_conc1_1771642699.0373511-TP-0.trace.json.gz",
        "num_hidden_layers": 48,
        "layers_per_block": 1,
        "mtp_qseqlen_decode": False,
        "grid_threshold": 10000,
        "status": "partial",
        "expected_layer_types": {
            "pattern": "alternating_gdn_mha_moe",
        },
        "stage_coverage_min": 50.0,
        "type_coverage_min": 40.0,
    },
}


def compute_metrics(result: AnalysisResult, registry_entry: dict) -> dict:
    """Compute test metrics from an AnalysisResult."""
    total_time = result.total_time_us
    if total_time == 0:
        return {}

    # Stage coverage: % of kernel time in PREFILL or DECODE (not UNKNOWN)
    prefill_time = result.per_stage_stats.get(Stage.PREFILL, None)
    decode_time = result.per_stage_stats.get(Stage.DECODE, None)
    classified_stage_time = 0.0
    if prefill_time:
        classified_stage_time += prefill_time.total_time_us
    if decode_time:
        classified_stage_time += decode_time.total_time_us
    stage_coverage_pct = 100.0 * classified_stage_time / total_time

    # Type coverage: % of kernel time NOT classified as OTHER
    other_time = result.per_type_stats.get(KernelType.OTHER, None)
    other_time_us = other_time.total_time_us if other_time else 0.0
    type_coverage_pct = 100.0 * (1.0 - other_time_us / total_time)

    # Layer metrics
    prefill_layers = [l for l in result.layers if l.stage == Stage.PREFILL]
    decode_layers = [l for l in result.layers if l.stage == Stage.DECODE]
    total_layers = len(result.layers)

    both_stages_present = len(prefill_layers) > 0 and len(decode_layers) > 0

    # Layer types
    layer_types = [l.layer_type.value for l in result.layers]

    return {
        "stage_coverage_pct": round(stage_coverage_pct, 2),
        "type_coverage_pct": round(type_coverage_pct, 2),
        "both_stages_present": both_stages_present,
        "total_layers": total_layers,
        "prefill_layers": len(prefill_layers),
        "decode_layers": len(decode_layers),
        "layer_types": layer_types,
    }


def generate_baseline(
    model_name: str,
    trace_dir: str,
    cache_only: bool = False,
) -> dict:
    """Generate baseline JSON and pickle cache for a single model."""
    entry = TRACE_REGISTRY[model_name]
    trace_path = os.path.join(trace_dir, entry["trace_file"])

    if not os.path.exists(trace_path):
        print(f"  SKIP: Trace file not found: {trace_path}")
        return {}

    baselines_dir = os.path.join(os.path.dirname(__file__), "baselines")
    snapshots_dir = os.path.join(os.path.dirname(__file__), "snapshots")

    pickle_path = os.path.join(snapshots_dir, f"{model_name}.pkl")

    # Check if we can use cached result
    result = None
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                result = pickle.load(f)
            print(f"  Loaded cached result from {pickle_path}")
        except Exception as e:
            print(f"  Cache load failed ({e}), re-parsing...")
            result = None

    if result is None:
        print(f"  Parsing trace: {trace_path}")
        analyzer = TraceAnalyzer(
            grid_threshold=entry["grid_threshold"],
            mtp_qseqlen_decode=entry["mtp_qseqlen_decode"],
        )
        trace_data = analyzer.load_trace(trace_path)
        result = analyzer.analyze(trace_data, detect_layers=True)

        # Save pickle cache
        with open(pickle_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved cache: {pickle_path}")

    if cache_only:
        print(f"  Cache-only mode, skipping baseline JSON.")
        return {}

    # Compute metrics
    metrics = compute_metrics(result, entry)

    # Build baseline JSON
    baseline = {
        "meta": {
            "model_name": model_name,
            "trace_file": entry["trace_file"],
            "status": entry["status"],
            "num_hidden_layers": entry["num_hidden_layers"],
            "layers_per_block": entry["layers_per_block"],
            "mtp_qseqlen_decode": entry["mtp_qseqlen_decode"],
            "grid_threshold": entry["grid_threshold"],
        },
        "metrics": {
            "stage_coverage_pct": {"min": entry["stage_coverage_min"]},
            "type_coverage_pct": {
                "min": entry["type_coverage_min"],
                "is_secondary": True,
            },
            "both_stages_present": True,
            "layer_count_min": entry["num_hidden_layers"],
            "expected_layer_types": entry["expected_layer_types"],
        },
        "actual_metrics": metrics,
        "known_issues": [],
    }

    # Add known issues for broken models
    if entry["status"] == "known_broken":
        issues = []
        if not metrics.get("both_stages_present", False):
            issues.append("Missing prefill or decode layers")
        if metrics.get("total_layers", 0) < entry["num_hidden_layers"]:
            issues.append(
                f"Only {metrics.get('total_layers', 0)} layer(s) detected, "
                f"expected at least {entry['num_hidden_layers']}"
            )
        baseline["known_issues"] = issues

    # Write baseline JSON
    baseline_path = os.path.join(baselines_dir, f"{model_name}.json")
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  Wrote baseline: {baseline_path}")

    # Print summary
    print(f"  Status: {entry['status']}")
    print(f"  Stage coverage: {metrics.get('stage_coverage_pct', 0):.1f}%")
    print(f"  Type coverage:  {metrics.get('type_coverage_pct', 0):.1f}%")
    print(
        f"  Layers: {metrics.get('total_layers', 0)} total "
        f"({metrics.get('prefill_layers', 0)} prefill, "
        f"{metrics.get('decode_layers', 0)} decode)"
    )
    print(f"  Both stages present: {metrics.get('both_stages_present', False)}")
    if baseline["known_issues"]:
        print(f"  Known issues: {baseline['known_issues']}")

    return baseline


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden baselines for trace_analyzer regression tests."
    )
    parser.add_argument(
        "--trace-dir",
        default="/home/yichiche/profile-baseline",
        help="Directory containing trace files (default: /home/yichiche/profile-baseline)",
    )
    parser.add_argument(
        "--model",
        choices=list(TRACE_REGISTRY.keys()),
        help="Generate baseline for a specific model",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate baselines for all models",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only generate pickle caches, skip baseline JSON",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered models and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Registered models:")
        for name, entry in TRACE_REGISTRY.items():
            print(f"  {name:<30s} status={entry['status']:<14s} file={entry['trace_file']}")
        return

    if not args.model and not args.all:
        parser.error("Specify --model NAME or --all")

    models = [args.model] if args.model else list(TRACE_REGISTRY.keys())

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Generating baseline: {model_name}")
        print(f"{'='*60}")
        generate_baseline(model_name, args.trace_dir, cache_only=args.cache_only)

    print(f"\nDone. Generated {len(models)} baseline(s).")


if __name__ == "__main__":
    main()
