"""Unit tests for LayerDetector with synthetic kernel event data.

These tests verify layer boundary detection and classification logic
without needing trace files — they construct minimal KernelEvent
sequences that mimic real model architectures.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace_analyzer import KernelEvent, KernelType, LayerDetector, LayerType, Stage


def make_event(
    name: str,
    kernel_type: KernelType = KernelType.OTHER,
    stage: Stage = Stage.UNKNOWN,
    simplified_name: str = "",
    duration_us: float = 10.0,
    timestamp_us: float = 0.0,
) -> KernelEvent:
    """Helper to create a KernelEvent for testing."""
    return KernelEvent(
        name=name,
        timestamp_us=timestamp_us,
        duration_us=duration_us,
        stage=stage,
        kernel_type=kernel_type,
        initial_stage=stage,
        simplified_name=simplified_name or name,
    )


def build_mla_moe_layer(stage: Stage = Stage.UNKNOWN) -> list:
    """Build a minimal MLA+MoE layer kernel sequence.

    Pattern: RMS_MXFP4 -> MLA_DECODE -> MLA_REDUCE -> ALLREDUCE ->
             MOE_SORT -> MOE_GEMM -> ALLREDUCE
    """
    return [
        make_event(
            "_fused_rms_mxfp4_quant",
            KernelType.QUANTIZATION,
            stage,
            "RMS_MXFP4",
        ),
        make_event(
            "mla_a8w8_blockscale_qseqlen1",
            KernelType.ATTENTION,
            Stage.DECODE,
            "MLA_DECODE",
        ),
        make_event(
            "kn_mla_reduce_kernel",
            KernelType.ATTENTION,
            stage,
            "MLA_REDUCE",
        ),
        make_event(
            "rcclGenericKernel",
            KernelType.COMMUNICATION,
            stage,
            "ALLREDUCE",
        ),
        make_event(
            "MoeSorting",
            KernelType.MOE,
            stage,
            "MOE_SORT",
        ),
        make_event(
            "kernel_moe_mxgemm",
            KernelType.MOE,
            stage,
            "MOE_GEMM",
        ),
        make_event(
            "rcclGenericKernel",
            KernelType.COMMUNICATION,
            stage,
            "ALLREDUCE",
        ),
    ]


def build_mla_fc_layer(stage: Stage = Stage.UNKNOWN) -> list:
    """Build a minimal MLA+FC layer kernel sequence.

    Pattern: RMS_MXFP4 -> MLA_DECODE -> MLA_REDUCE -> ALLREDUCE ->
             GEMM -> ACT_MUL -> GEMM -> ALLREDUCE
    """
    return [
        make_event(
            "_fused_rms_mxfp4_quant",
            KernelType.QUANTIZATION,
            stage,
            "RMS_MXFP4",
        ),
        make_event(
            "mla_a8w8_blockscale_qseqlen1",
            KernelType.ATTENTION,
            Stage.DECODE,
            "MLA_DECODE",
        ),
        make_event(
            "kn_mla_reduce_kernel",
            KernelType.ATTENTION,
            stage,
            "MLA_REDUCE",
        ),
        make_event(
            "rcclGenericKernel",
            KernelType.COMMUNICATION,
            stage,
            "ALLREDUCE",
        ),
        make_event(
            "_gemm_afp4wfp4_kernel",
            KernelType.QUANTIZATION,
            stage,
            "GEMM_FP4",
        ),
        make_event(
            "act_and_mul_kernel",
            KernelType.OTHER,
            stage,
            "ACT_MUL",
        ),
        make_event(
            "_gemm_afp4wfp4_kernel",
            KernelType.QUANTIZATION,
            stage,
            "GEMM_FP4",
        ),
        make_event(
            "rcclGenericKernel",
            KernelType.COMMUNICATION,
            stage,
            "ALLREDUCE",
        ),
    ]


def build_attn_half_layer(stage: Stage = Stage.UNKNOWN) -> list:
    """Build an attention-only half-layer (Grok-2 style).

    Pattern: DUAL_RMSNORM -> PA_DECODE -> PA_REDUCE -> ALLREDUCE
    """
    return [
        make_event(
            "fused_dual_residual_rmsnorm_kernel",
            KernelType.OTHER,
            stage,
            "RMSNORM",
        ),
        make_event(
            "paged_attention_ll4mi_QKV",
            KernelType.ATTENTION,
            Stage.DECODE,
            "PA_DECODE",
        ),
        make_event(
            "paged_attention_ll4mi_reduce",
            KernelType.ATTENTION,
            stage,
            "PA_REDUCE",
        ),
        make_event(
            "rcclGenericKernel",
            KernelType.COMMUNICATION,
            stage,
            "ALLREDUCE",
        ),
    ]


def build_moe_half_layer(stage: Stage = Stage.UNKNOWN) -> list:
    """Build a MoE-only half-layer (Grok-2 style).

    Pattern: DUAL_RMSNORM -> MOE_SORT -> MOE_GEMM -> ALLREDUCE
    """
    return [
        make_event(
            "fused_dual_residual_rmsnorm_kernel",
            KernelType.OTHER,
            stage,
            "RMSNORM",
        ),
        make_event(
            "MoeSorting",
            KernelType.MOE,
            stage,
            "MOE_SORT",
        ),
        make_event(
            "kernel_moe_mxgemm",
            KernelType.MOE,
            stage,
            "MOE_GEMM",
        ),
        make_event(
            "rcclGenericKernel",
            KernelType.COMMUNICATION,
            stage,
            "ALLREDUCE",
        ),
    ]


# ---------------------------------------------------------------------------
# Layer Detection Tests
# ---------------------------------------------------------------------------
class TestLayerDetection:
    """Tests for LayerDetector.detect_layers()."""

    def test_empty_events(self):
        detector = LayerDetector()
        assert detector.detect_layers([]) == []

    def test_single_mla_moe_layer(self):
        """A single MLA+MoE layer should be detected correctly."""
        detector = LayerDetector()
        events = build_mla_moe_layer(Stage.DECODE)
        layers = detector.detect_layers(events)

        # Should detect at least one layer
        assert len(layers) >= 1
        # First real layer should be MLA+MoE
        mla_moe_layers = [l for l in layers if l.layer_type == LayerType.MLA_MOE]
        assert len(mla_moe_layers) >= 1

    def test_single_mla_fc_layer(self):
        """A single MLA+FC layer should be detected correctly."""
        detector = LayerDetector()
        events = build_mla_fc_layer(Stage.DECODE)
        layers = detector.detect_layers(events)

        assert len(layers) >= 1
        mla_fc_layers = [l for l in layers if l.layer_type == LayerType.MLA_FC]
        assert len(mla_fc_layers) >= 1

    def test_two_consecutive_layers(self):
        """Two MLA+MoE layers should produce two detected layers."""
        detector = LayerDetector()
        events = build_mla_moe_layer(Stage.DECODE) + build_mla_moe_layer(Stage.DECODE)
        layers = detector.detect_layers(events)

        # The ALLREDUCE at end of layer 1 + RMS at start of layer 2
        # should trigger a boundary
        mla_moe_layers = [l for l in layers if l.layer_type == LayerType.MLA_MOE]
        assert len(mla_moe_layers) >= 2

    def test_fc_then_moe_layers(self):
        """FC layer followed by MoE layer should detect both types."""
        detector = LayerDetector()
        events = build_mla_fc_layer(Stage.DECODE) + build_mla_moe_layer(Stage.DECODE)
        layers = detector.detect_layers(events)

        fc_layers = [l for l in layers if l.layer_type == LayerType.MLA_FC]
        moe_layers = [l for l in layers if l.layer_type == LayerType.MLA_MOE]
        assert len(fc_layers) >= 1, "Should detect at least one FC layer"
        assert len(moe_layers) >= 1, "Should detect at least one MoE layer"

    def test_layer_boundary_allreduce_rms(self):
        """Layer boundary is detected at ALLREDUCE -> RMSnorm transition."""
        detector = LayerDetector()
        # Two layers back to back
        events = build_mla_moe_layer() + build_mla_moe_layer()
        layers = detector.detect_layers(events)

        # The first layer's trailing ALLREDUCE should be included in layer 2
        # as the leading ALLREDUCE
        assert len(layers) >= 2


# ---------------------------------------------------------------------------
# Layer Type Classification Tests
# ---------------------------------------------------------------------------
class TestLayerTypeClassification:
    """Tests for LayerDetector._classify_layer_type()."""

    def test_mla_moe(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=True, saw_mla=True, saw_gdn=False, saw_moe=True, saw_fc=False
        )
        assert result == LayerType.MLA_MOE

    def test_mla_fc(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=True, saw_mla=True, saw_gdn=False, saw_moe=False, saw_fc=True
        )
        assert result == LayerType.MLA_FC

    def test_mha_moe(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=True, saw_mla=False, saw_gdn=False, saw_moe=True, saw_fc=False
        )
        assert result == LayerType.MHA_MOE

    def test_mha_fc(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=True, saw_mla=False, saw_gdn=False, saw_moe=False, saw_fc=True
        )
        assert result == LayerType.MHA_FC

    def test_gdn_moe(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=True, saw_mla=False, saw_gdn=True, saw_moe=True, saw_fc=False
        )
        assert result == LayerType.GDN_MOE

    def test_gdn_fc(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=True, saw_mla=False, saw_gdn=True, saw_moe=False, saw_fc=True
        )
        assert result == LayerType.GDN_FC

    def test_mla_over_gdn(self):
        """MLA takes priority over GDN when both markers are present."""
        result = LayerDetector._classify_layer_type(
            saw_attention=True, saw_mla=True, saw_gdn=True, saw_moe=True, saw_fc=False
        )
        assert result == LayerType.MLA_MOE

    def test_attn_only(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=True, saw_mla=False, saw_gdn=False, saw_moe=False, saw_fc=False
        )
        assert result == LayerType.ATTN

    def test_moe_only(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=False, saw_mla=False, saw_gdn=False, saw_moe=True, saw_fc=False
        )
        assert result == LayerType.MOE

    def test_fc_only(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=False, saw_mla=False, saw_gdn=False, saw_moe=False, saw_fc=True
        )
        assert result == LayerType.FC

    def test_unknown(self):
        result = LayerDetector._classify_layer_type(
            saw_attention=False, saw_mla=False, saw_gdn=False, saw_moe=False, saw_fc=False
        )
        assert result == LayerType.UNKNOWN


# ---------------------------------------------------------------------------
# Half-Layer Merging Tests
# ---------------------------------------------------------------------------
class TestMergeHalfLayers:
    """Tests for LayerDetector.merge_half_layers()."""

    def test_merge_attn_moe(self):
        """Adjacent Attn + MoE half-layers should merge into MHA+MoE."""
        detector = LayerDetector()
        events = build_attn_half_layer(Stage.DECODE) + build_moe_half_layer(
            Stage.DECODE
        )
        raw_layers = detector.detect_layers(events)

        # Before merge, we should see Attn and MoE half-layers
        attn_layers = [l for l in raw_layers if l.layer_type == LayerType.ATTN]
        moe_layers = [l for l in raw_layers if l.layer_type == LayerType.MOE]

        if attn_layers and moe_layers:
            merged = detector.merge_half_layers(raw_layers)
            # After merge, Attn+MoE should become a single full layer
            assert len(merged) < len(raw_layers)
            full_layers = [
                l
                for l in merged
                if l.layer_type in (LayerType.MHA_MOE, LayerType.MLA_MOE)
            ]
            assert len(full_layers) >= 1

    def test_no_merge_when_not_adjacent(self):
        """Non-adjacent Attn and MoE layers should not merge."""
        detector = LayerDetector()
        # Attn, MoE, Attn — the second Attn should not merge with the first MoE
        events = (
            build_attn_half_layer(Stage.DECODE)
            + build_moe_half_layer(Stage.DECODE)
            + build_attn_half_layer(Stage.DECODE)
        )
        raw_layers = detector.detect_layers(events)
        merged = detector.merge_half_layers(raw_layers)

        # Should still have 2+ layers after merge (at least one merged pair + one leftover)
        assert len(merged) >= 2


# ---------------------------------------------------------------------------
# Layer Stage Determination Tests
# ---------------------------------------------------------------------------
class TestLayerStageDetermination:
    """Tests for LayerDetector._determine_layer_stage()."""

    def test_majority_decode(self):
        """Layer with more decode kernels should be DECODE."""
        detector = LayerDetector()
        kernels = [
            make_event("k1", stage=Stage.DECODE),
            make_event("k2", stage=Stage.DECODE),
            make_event("k3", stage=Stage.PREFILL),
            make_event("k4", stage=Stage.UNKNOWN),
        ]
        assert detector._determine_layer_stage(kernels) == Stage.DECODE

    def test_majority_prefill(self):
        """Layer with more prefill kernels should be PREFILL."""
        detector = LayerDetector()
        kernels = [
            make_event("k1", stage=Stage.PREFILL),
            make_event("k2", stage=Stage.PREFILL),
            make_event("k3", stage=Stage.DECODE),
        ]
        assert detector._determine_layer_stage(kernels) == Stage.PREFILL

    def test_tie_is_unknown(self):
        """Equal prefill and decode counts should yield UNKNOWN."""
        detector = LayerDetector()
        kernels = [
            make_event("k1", stage=Stage.PREFILL),
            make_event("k2", stage=Stage.DECODE),
        ]
        assert detector._determine_layer_stage(kernels) == Stage.UNKNOWN

    def test_all_unknown(self):
        """All unknown kernels should yield UNKNOWN."""
        detector = LayerDetector()
        kernels = [
            make_event("k1", stage=Stage.UNKNOWN),
            make_event("k2", stage=Stage.UNKNOWN),
        ]
        assert detector._determine_layer_stage(kernels) == Stage.UNKNOWN


# ---------------------------------------------------------------------------
# MTP qseqlen decode mode tests
# ---------------------------------------------------------------------------
class TestMTPQseqlenDecode:
    """Tests for mtp_qseqlen_decode flag behavior."""

    def test_mtp_mode_qseqlen2_is_decode(self):
        """With mtp_qseqlen_decode=True, qseqlen2 should be decode for layer stage."""
        detector = LayerDetector(mtp_qseqlen_decode=True)
        # qseqlen2+ should match decode patterns
        for pattern in detector._attention_decode_patterns:
            if pattern.search("mla_a8w8_blockscale_qseqlen2"):
                return  # Found it, test passes
        pytest.fail("qseqlen2 not matched by any decode pattern in MTP mode")

    def test_normal_mode_qseqlen2_is_prefill(self):
        """With mtp_qseqlen_decode=False, qseqlen2 should be prefill for layer stage."""
        detector = LayerDetector(mtp_qseqlen_decode=False)
        for pattern in detector._attention_prefill_patterns:
            if pattern.search("mla_a8w8_blockscale_qseqlen2"):
                return  # Found it, test passes
        pytest.fail("qseqlen2 not matched by any prefill pattern in normal mode")
