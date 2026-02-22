# Trace Analyzer — Adding Support for a New Model

The trace analyzer (`trace_analyzer.py`) classifies GPU kernel events from torch profiler traces into stages (prefill/decode), kernel types, and transformer layers. Use this guide when debugging the analyzer or adding support for a new model architecture.

## 1. Identify kernel names from a trace

Run the analyzer with `-v` and inspect the output. Look for:
- **Layer-start norms**: The RMSNorm or LayerNorm kernel that fires at the start of each transformer block (e.g., `fused_rmsnorm_kernel`, `fused_dual_residual_rmsnorm_kernel`, `Rmsnorm2dFwd`). Some models fire this norm **twice per block** (once for attention, once for MoE/FC), producing half-layers.
- **Attention kernels**: MLA decode/prefill, flash attention, paged attention, GDN (gated delta network) variants.
- **GDN kernels** (Qwen3-Coder-Next): `chunk_gated_delta_rule_fwd` (prefill), `fused_sigmoid_gating_delta_rule_update` (decode), `_causal_conv1d_fwd`/`_update`, `fused_gdn_gating`, `l2norm_fwd`, `_layer_norm_fwd_1pass`.
- **MoE markers**: MoeSorting, moe_fused_gate, kernel_moe_gemm, moeSoftmax, etc.
- **FC markers**: Activation functions like `act_and_mul_kernel`, `silu_kernel`, `gelu_and_mul_kernel`.
- **Communication**: ALLREDUCE kernels that precede layer-start norms.

## 2. Update pattern tables in `KernelClassifier`

- **`_layer_start_patterns`** (in `LayerDetector.__init__`): Add the model's layer-start norm kernel name. This pattern, when preceded by ALLREDUCE (with optional OTHER/MEMORY kernels in between), triggers a new layer boundary. For models with unfused normalization (e.g. Qwen3-Coder-Next), a fallback detects the first GEMM after ALLREDUCE when no fused norm is found.
- **`_gdn_markers`**: Add GDN-specific kernel names so layer type distinguishes GDN_MOE/GDN_FC from MHA/MLA variants.
- **`_stage_patterns`**: Add patterns that unambiguously indicate prefill or decode (e.g., attention kernels with `qseqlen` suffixes, `_decode_`/`_prefill_` in the name).
- **`_type_patterns`**: Map new kernel names to `KernelType` (ATTENTION, MOE, QUANTIZATION, COMMUNICATION, LINEAR, MEMORY).
- **`_simplify_patterns`**: Add short display names for new kernels.
- **`_moe_markers`** / **`_fc_markers`**: Add MoE/FC indicator kernels so layer type classification works.
- **`_attention_decode_patterns`** / **`_attention_prefill_patterns`**: Add attention kernel patterns that disambiguate decode vs prefill within a layer.

## 3. Handle half-layers (dual-norm models)

Models like Grok-2 use `fused_dual_residual_rmsnorm` which fires twice per transformer block — once before attention, once before MoE. Each triggers a layer boundary, producing:
- An **attention half-layer** (type `Attn`) with attention kernels
- A **MoE/FC half-layer** (type `MoE` or `FC`) with no attention kernels

Two mechanisms handle this:

1. **Half-layer merge** (on by default): Adjacent Attn+MoE or Attn+FC half-layers are automatically merged into full `MHA_MOE` / `MHA_FC` / `MLA_MOE` / `MLA_FC` layers. The merged layer's stage is voted from the combined kernel set.
2. **Stage propagation** (always active): Any remaining UNKNOWN-stage layers inherit the stage of the immediately preceding layer.

The attention type (MLA vs MHA) is auto-detected from kernel names: layers containing MLA-specific kernels (`mla_a8w8`, `set_mla_kv_buffer`, etc.) are labeled MLA; otherwise MHA.

```bash
# Default: half-layers merged, Grok-2 64 blocks → 64 layers per pass
python3 trace_analyzer.py trace.json.gz -v

# To inspect raw half-layers separately:
python3 trace_analyzer.py trace.json.gz --split-half-layers -v
```

## 4. Diagnostic steps

```bash
# Run analyzer with verbose output
python3 trace_analyzer.py /path/to/trace.json.gz -v 2>&1 | head -30

# Check layer count matches expected: num_hidden_layers x passes x (1 or 2 for half-layers)
# Check unknown % is near 0%
# Inspect orphan events (kernels not assigned to any layer)

# Debug specific layers
python3 trace_analyzer.py /path/to/trace.json.gz --debug-layers 0,1,2,3

# Show layer-by-layer breakdown
python3 trace_analyzer.py /path/to/trace.json.gz --show-layer-terminal
```

## Supported model patterns

| Model | Norm pattern | Layers per block | Notes |
|-------|-------------|-----------------|-------|
| DeepSeek-V3 | `_fused_rms_mxfp4_quant` / `_fused_rms_fp8` | 1 (full) | MLA + MoE/FC in single layer |
| Grok-2 | `fused_dual_residual_rmsnorm_kernel` | 2 → 1 (auto-merged) | MHA+MoE; use `--split-half-layers` to see halves |
| Qwen3-Coder-Next | Unfused RMSNorm (GEMM fallback) | 2 → 1 (auto-merged) | GDN+MoE (3/4 layers) + MHA+MoE (1/4 layers) |
| Llama/Qwen | `fused_rmsnorm_kernel` / `Rmsnorm2dFwd` | 1 (full) | Standard transformer block |

## Key architecture concepts

- **Layer detection** (`LayerDetector.detect_layers`): Scans kernel events sequentially. A new layer boundary is triggered when: (1) a `_layer_start_patterns` match appears after an ALLREDUCE kernel (with optional OTHER/MEMORY kernels in between), or (2) fallback: the first LINEAR (GEMM) kernel after ALLREDUCE when only OTHER/MEMORY kernels appear in between and no fused RMSNorm was found (handles unfused normalization as in Qwen3-Coder-Next).
- **Layer type classification** (`_classify_layer_type`): Uses `saw_attention`/`saw_mla`/`saw_gdn`/`saw_moe`/`saw_fc` flags to pick from `MLA_MOE`, `MLA_FC`, `GDN_MOE`, `GDN_FC`, `MHA_MOE`, `MHA_FC`, `ATTN`, `MOE`, `FC`, or `UNKNOWN`.
- **Stage voting** (`_determine_layer_stage`): Counts prefill vs decode votes among the layer's kernels. Ties go to UNKNOWN.
- **Half-layer merge** (`merge_half_layers`, on by default; disable with `--split-half-layers`): Combines adjacent Attn+MoE/FC pairs into single full layers, auto-detecting MLA vs MHA vs GDN from kernel names, re-voting stage from the combined kernels.
- **Stage propagation**: After all layers are detected (and optionally merged), a forward pass propagates known stages to adjacent UNKNOWN layers (handles remaining half-layers that lack stage-indicating kernels).
- **Kernel stage reassignment**: After layer stages are finalized, all kernels within a layer inherit that layer's stage, then per-stage stats are recomputed.
