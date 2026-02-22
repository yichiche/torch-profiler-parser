"""Unit tests for KernelClassifier.

These tests verify kernel name -> KernelType and Stage classification
without needing any trace files. They run instantly.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trace_analyzer import KernelClassifier, KernelType, Stage


@pytest.fixture
def classifier():
    return KernelClassifier(grid_threshold=10000)


# ---------------------------------------------------------------------------
# classify_type tests
# ---------------------------------------------------------------------------
class TestClassifyType:
    """Tests for KernelClassifier.classify_type()."""

    @pytest.mark.parametrize(
        "kernel_name,expected_type",
        [
            # Attention kernels
            ("aiter::mla_decode_fwd_kernel", KernelType.ATTENTION),
            ("mla_a8w8_blockscale_qseqlen1", KernelType.ATTENTION),
            ("decode_attention_kernel", KernelType.ATTENTION),
            ("flash_attn_fwd", KernelType.ATTENTION),
            ("fmha_fwd_kernel", KernelType.ATTENTION),
            ("FmhaBatchPrefill", KernelType.ATTENTION),
            ("set_mla_kv_buffer_kernel", KernelType.ATTENTION),
            ("kn_entry_2c_sbhd", KernelType.ATTENTION),
            ("kn_mla_reduce_kernel", KernelType.ATTENTION),
            ("paged_attention_ll4mi_QKV", KernelType.ATTENTION),
            ("concat_and_cast_mha_k_kernel", KernelType.ATTENTION),
            ("qk_rope_cat_and_cache_kernel", KernelType.ATTENTION),
            # MoE kernels
            ("fused_moe_kernel", KernelType.MOE),
            ("moe_align_block_size", KernelType.MOE),
            ("topk_softmax_kernel", KernelType.MOE),  # 'topk' matches MOE (checked before attention)
            ("MoeSorting", KernelType.MOE),
            ("MoeFlatmm", KernelType.MOE),
            ("kernel_moe_gemm", KernelType.MOE),
            ("kernel_moe_mxgemm", KernelType.MOE),
            ("fused_append_shared_experts", KernelType.MOE),
            ("grouped_topk", KernelType.MOE),
            ("moe_fused_gate", KernelType.MOE),
            ("_moe_mxfp4_sort", KernelType.MOE),
            # Quantization kernels
            ("_fused_rms_mxfp4_quant", KernelType.QUANTIZATION),
            ("_fused_rms_fp8", KernelType.QUANTIZATION),
            ("_gemm_afp4wfp4_kernel", KernelType.QUANTIZATION),
            ("_gemm_a8w8_blockscale", KernelType.QUANTIZATION),
            ("dynamic_per_group_scaled_quant", KernelType.QUANTIZATION),
            ("_dynamic_mxfp4_quant", KernelType.QUANTIZATION),
            ("_batched_gemm_a16wfp4", KernelType.QUANTIZATION),
            ("_batched_gemm_a8w8", KernelType.QUANTIZATION),
            ("_fused_flatten", KernelType.QUANTIZATION),
            # Communication kernels
            ("ncclDevKernel_AllReduce", KernelType.COMMUNICATION),
            ("rcclGenericKernel", KernelType.COMMUNICATION),
            ("cross_device_reduce_1block", KernelType.COMMUNICATION),
            ("all_reduce_kernel", KernelType.COMMUNICATION),
            # Linear kernels
            ("Cijk_Alik_Bljk_SB", KernelType.LINEAR),
            ("Custom_Cijk_Alik", KernelType.LINEAR),
            ("_gemm_a16_w16", KernelType.LINEAR),
            # Memory kernels
            ("memcpy", KernelType.MEMORY),
            ("memset", KernelType.MEMORY),
            ("bfloat16_copy", KernelType.MEMORY),
            ("float8_copy", KernelType.MEMORY),
            # Other / unknown kernels
            ("act_and_mul_kernel", KernelType.OTHER),
            ("Rmsnorm2dFwd", KernelType.OTHER),
            ("some_random_kernel", KernelType.OTHER),
        ],
    )
    def test_classify_type(self, classifier, kernel_name, expected_type):
        assert classifier.classify_type(kernel_name) == expected_type

    def test_type_priority_attention_over_quant(self, classifier):
        """Attention patterns should match before quantization for MLA kernels."""
        # mla_a8w8 contains 'fp8'/'a8w8' but should be ATTENTION
        result = classifier.classify_type("mla_a8w8_blockscale_qseqlen1")
        assert result == KernelType.ATTENTION

    def test_type_priority_moe_over_quant(self, classifier):
        """MoE patterns should match before quantization for MoE gemm kernels."""
        result = classifier.classify_type("kernel_moe_mxgemm")
        assert result == KernelType.MOE


# ---------------------------------------------------------------------------
# classify_stage tests
# ---------------------------------------------------------------------------
class TestClassifyStage:
    """Tests for KernelClassifier.classify_stage()."""

    @pytest.mark.parametrize(
        "kernel_name,grid,expected_stage",
        [
            # Force-unknown patterns (quant gemms with ambiguous stage)
            ("_fused_rms_fp8", None, Stage.UNKNOWN),
            ("_fused_rms_mxfp4", None, Stage.UNKNOWN),
            ("_gemm_a8w8_blockscale", None, Stage.UNKNOWN),
            ("_batched_gemm_a8w8_something", None, Stage.UNKNOWN),
            ("_gemm_afp4wfp4_kernel", None, Stage.UNKNOWN),
            ("_batched_gemm_a16wfp4", None, Stage.UNKNOWN),
            # Prefill-specific kernel names
            ("set_mla_kv_buffer_kernel", None, Stage.PREFILL),
            ("concat_and_cast_mha_k_kernel", None, Stage.PREFILL),
            # qseqlen-based stage (small = decode, large = prefill)
            ("mla_a8w8_blockscale_qseqlen1", None, Stage.DECODE),
            ("mla_a8w8_blockscale_qseqlen2", None, Stage.DECODE),
            ("mla_a8w8_blockscale_qseqlen4", None, Stage.DECODE),
            ("mla_a8w8_blockscale_qseqlen5", None, Stage.PREFILL),
            ("mla_a8w8_blockscale_qseqlen16", None, Stage.PREFILL),
            ("mla_a8w8_blockscale_qseqlen128", None, Stage.PREFILL),
            # Explicit decode/prefill name patterns
            ("some_decode_attention_kernel", None, Stage.DECODE),
            ("paged_attention_ll4mi_QKV", None, Stage.DECODE),
            ("flash_attn_fwd_kernel", None, Stage.PREFILL),
            ("fmha_fwd_kernel", None, Stage.PREFILL),
            ("FmhaBatchPrefill", None, Stage.PREFILL),
            # Grid-based heuristic
            ("some_unknown_kernel", (50000, 1, 1), Stage.DECODE),
            ("some_unknown_kernel", (100, 1, 1), Stage.PREFILL),
            # No information -> UNKNOWN
            ("some_unknown_kernel", None, Stage.UNKNOWN),
            ("some_unknown_kernel", (5000, 1, 1), Stage.UNKNOWN),
        ],
    )
    def test_classify_stage(self, classifier, kernel_name, grid, expected_stage):
        assert classifier.classify_stage(kernel_name, grid) == expected_stage

    def test_force_unknown_overrides_name_pattern(self, classifier):
        """Force-unknown patterns should win even if the name also matches a stage."""
        # _fused_rms_fp8 contains 'fp8' which might match other patterns,
        # but the force-unknown should take priority
        result = classifier.classify_stage("_fused_rms_fp8_prefill_kernel", None)
        assert result == Stage.UNKNOWN

    def test_grid_threshold_boundary(self):
        """Test grid threshold boundary behavior."""
        classifier = KernelClassifier(grid_threshold=5000)

        # Above threshold -> DECODE
        assert classifier.classify_stage("unknown", (5001, 1, 1)) == Stage.DECODE
        # Below 200 -> PREFILL
        assert classifier.classify_stage("unknown", (199, 1, 1)) == Stage.PREFILL
        # In between -> UNKNOWN
        assert classifier.classify_stage("unknown", (3000, 1, 1)) == Stage.UNKNOWN


# ---------------------------------------------------------------------------
# simplify_name tests
# ---------------------------------------------------------------------------
class TestSimplifyName:
    """Tests for KernelClassifier.simplify_name()."""

    @pytest.mark.parametrize(
        "kernel_name,expected_simplified",
        [
            ("rcclGenericKernel", "ALLREDUCE"),
            ("ncclDevKernel_AllReduce", "ALLREDUCE"),
            ("cross_device_reduce_1block", "ALLREDUCE"),
            ("fmha_fwd_kernel", "FMHA"),
            ("mla_a8w8_blockscale_qseqlen1", "MLA_DECODE"),
            ("mla_a8w8_blockscale_qseqlen16", "MLA_DECODE"),  # qseqlen1 pattern matches first (qseqlen1 in qseqlen16)
            ("kn_mla_reduce_kernel", "MLA_REDUCE"),
            ("_fused_rms_mxfp4_quant", "RMS_MXFP4"),
            ("_fused_rms_fp8", "RMS_FP8"),
            ("_gemm_afp4wfp4_reduce_kernel", "GEMM_FP4_RED"),
            ("_gemm_afp4wfp4_kernel", "GEMM_FP4"),
            ("_gemm_a8w8_blockscale_reduce", "GEMM_FP8_RED"),
            ("_gemm_a8w8_blockscale", "GEMM_FP8"),
            ("Rmsnorm2dFwd", "RMSNORM"),
            ("fused_dual_residual_rmsnorm_kernel", "RMSNORM"),
            ("fused_rmsnorm_kernel", "RMSNORM"),
            ("set_mla_kv_buffer_kernel", "MLA_KV_SET"),
            ("MoeSorting", "MOE_SORT"),
            ("kernel_moe_mxgemm", "MOE_GEMM"),
            ("kernel_moe_gemm_kernel", "MOE_GEMM"),
            ("dynamic_per_group_scaled_quant", "DYN_QUANT"),
            ("Cijk_Alik_Bljk", "HIPBLAS_GEMM"),
            ("paged_attention_ll4mi_QKV", "PA_DECODE"),
            ("paged_attention_ll4mi_reduce", "PA_REDUCE"),
            ("FmhaBatchPrefill", "FMHA_PREFILL"),
        ],
    )
    def test_simplify_name(self, classifier, kernel_name, expected_simplified):
        assert classifier.simplify_name(kernel_name) == expected_simplified

    def test_unknown_name_truncation(self, classifier):
        """Unknown kernel names should be truncated to 25 chars."""
        long_name = "a" * 50
        result = classifier.simplify_name(long_name)
        assert len(result) == 25
        assert result == "a" * 25

    def test_short_unknown_name_preserved(self, classifier):
        """Short unknown kernel names should not be modified."""
        name = "short_kernel"
        assert classifier.simplify_name(name) == name
