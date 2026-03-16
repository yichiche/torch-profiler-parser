#!/usr/bin/env python3
"""Trace Module Analyzer — Correlation-based GPU kernel classification.

Uses nn.Module hierarchy from PyTorch traces to automatically classify GPU
kernels by module, eliminating the need for hardcoded regex patterns.

Supports two trace modes:
  - Full mode: kernel events present (diffusion traces) — correlates via cuda_runtime
  - CPU-only mode: no kernel events (LLM traces) — uses cpu_op time containment
"""

import argparse
import bisect
import gzip
import json
import logging
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from fix_rocm_trace_flow import fix_trace as _rocm_fix_trace
    _HAS_ROCM_FIX = True
except ImportError:
    _HAS_ROCM_FIX = False

# Maximum data rows per Excel tab (excluding header).  Keeps file size
# manageable for large traces (e.g. DeepSeek with 24K+ decoder instances).
MAX_ROWS_PER_TAB = 1000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModuleNode:
    """Tree node representing an nn.Module invocation."""
    name: str           # e.g. "WanTransformerBlock_5"
    module_type: str    # e.g. "WanTransformerBlock"
    instance_id: int    # e.g. 5
    ts: float           # start timestamp (us)
    end: float          # ts + dur
    tid: int
    pid: int
    children: List["ModuleNode"] = field(default_factory=list)
    kernels: List[Dict] = field(default_factory=list)
    cpu_ops: List[Dict] = field(default_factory=list)


@dataclass
class KernelDetail:
    """Individual kernel/op record for detail reporting."""
    name: str
    duration: float
    category: str
    module_path: str  # e.g. "WanTransformerBlock_1/WanT2VCrossAttention_0"
    ts: float = 0.0
    phase: str = ""  # "prefill" / "decode" / ""
    input_dims: str = ""  # e.g. "[[1, 75600, 10, 128], ...]"


@dataclass
class ModuleStats:
    """Aggregated statistics for a module node."""
    name: str
    module_type: str
    instance_id: int
    depth: int
    total_kernel_time: float = 0.0
    total_cpu_op_time: float = 0.0
    self_kernel_time: float = 0.0
    self_cpu_op_time: float = 0.0
    kernel_count: int = 0
    cpu_op_count: int = 0
    kernel_breakdown: Dict[str, Tuple[float, int]] = field(default_factory=dict)  # category -> (dur, count)
    kernel_details: List[KernelDetail] = field(default_factory=list)  # all kernels/ops in time order
    children_stats: List["ModuleStats"] = field(default_factory=list)
    phase: str = ""  # "prefill" / "decode" / ""


# ---------------------------------------------------------------------------
# Module tree builder
# ---------------------------------------------------------------------------

class ModuleTreeBuilder:
    """Build nn.Module hierarchy tree from python_function events."""

    MODULE_PREFIX = "nn.Module: "

    def build(self, events: List[Dict]) -> List[ModuleNode]:
        """Build forest from all events (filters for nn.Module internally)."""
        module_events = [
            e for e in events
            if e.get("cat") == "python_function"
            and str(e.get("name", "")).startswith(self.MODULE_PREFIX)
            and e.get("dur") is not None
        ]
        return self.build_from_module_events(module_events)

    def build_from_module_events(self, module_events: List[Dict]) -> List[ModuleNode]:
        """Build forest from pre-filtered nn.Module events (faster for large traces)."""
        if not module_events:
            return []

        # Group by (pid, tid)
        by_thread: Dict[Tuple[int, int], List[Dict]] = defaultdict(list)
        for e in module_events:
            by_thread[(e["pid"], e["tid"])].append(e)

        roots = []
        for (pid, tid), thread_events in by_thread.items():
            thread_events.sort(key=lambda x: (x["ts"], -x.get("dur", 0)))
            thread_roots = self._build_thread(thread_events, pid, tid)
            roots.extend(thread_roots)

        return roots

    def _build_thread(self, events: List[Dict], pid: int, tid: int) -> List[ModuleNode]:
        """Stack-based nesting for events on a single thread."""
        roots = []
        stack: List[ModuleNode] = []

        for e in events:
            ts = e["ts"]
            dur = e.get("dur", 0)
            end = ts + dur
            raw_name = e["name"][len(self.MODULE_PREFIX):]
            module_type, instance_id = self._parse_name(raw_name)

            node = ModuleNode(
                name=raw_name,
                module_type=module_type,
                instance_id=instance_id,
                ts=ts, end=end,
                tid=tid, pid=pid,
            )

            # Pop finished ancestors
            while stack and stack[-1].end <= ts:
                stack.pop()

            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)

            stack.append(node)

        return roots

    @staticmethod
    def _parse_name(raw_name: str) -> Tuple[str, int]:
        m = re.match(r"^(.+?)_(\d+)$", raw_name)
        if m:
            return m.group(1), int(m.group(2))
        return raw_name, 0


# ---------------------------------------------------------------------------
# Kernel correlator (full mode: kernel → cuda_runtime → module)
# ---------------------------------------------------------------------------

class _IntervalIndex:
    """Shared interval index for O(log n) module lookup by timestamp."""

    def __init__(self, roots: List[ModuleNode]):
        self._intervals: Dict[Tuple[int, int], List[Tuple[float, float, ModuleNode]]] = defaultdict(list)
        self._starts_cache: Dict[Tuple[int, int], List[float]] = {}
        self._collect_all_nodes(roots)
        for key in self._intervals:
            self._intervals[key].sort(key=lambda x: (x[0], -x[1]))
            self._starts_cache[key] = [iv[0] for iv in self._intervals[key]]

    def _collect_all_nodes(self, nodes: List[ModuleNode]):
        for node in nodes:
            self._intervals[(node.pid, node.tid)].append((node.ts, node.end, node))
            self._collect_all_nodes(node.children)

    def find_deepest(self, ts: float, pid: int, tid: int) -> Optional[ModuleNode]:
        key = (pid, tid)
        intervals = self._intervals.get(key)
        if not intervals:
            return None
        starts = self._starts_cache[key]
        idx = bisect.bisect_right(starts, ts) - 1
        best: Optional[ModuleNode] = None
        best_span = float("inf")
        for i in range(max(0, idx - 20), min(len(intervals), idx + 20)):
            s, e, node = intervals[i]
            if s <= ts <= e:
                span = e - s
                if span < best_span:
                    best_span = span
                    best = node
            elif s > ts:
                break
        return best


class CpuOpShapeIndex:
    """Map correlation IDs to cpu_op tensor shapes via timestamp containment.

    For each cuda_runtime/cuda_driver launch event, finds the enclosing cpu_op
    on the same thread and extracts its ``Input Dims`` argument.
    """

    def __init__(self, cpu_ops: List[Dict], runtime_events: List[Dict],
                 driver_events: Optional[List[Dict]] = None):
        # Build interval index of cpu_ops that carry Input Dims
        shaped_ops: Dict[Tuple[int, int], List[Tuple[float, float, str]]] = defaultdict(list)
        for op in cpu_ops:
            dims = op.get("args", {}).get("Input Dims")
            if not dims:
                continue
            ts = op["ts"]
            dur = op.get("dur", 0)
            tid = op["tid"]
            pid = op.get("pid", tid)
            dims_str = self._format_dims(dims)
            if dims_str:
                shaped_ops[(pid, tid)].append((ts, ts + dur, dims_str))

        for key in shaped_ops:
            shaped_ops[key].sort(key=lambda x: (x[0], -x[1]))
        self._shaped_ops = shaped_ops
        self._starts_cache: Dict[Tuple[int, int], List[float]] = {
            k: [iv[0] for iv in v] for k, v in shaped_ops.items()
        }

        # Build correlation → shape mapping
        self._corr_to_shape: Dict[int, str] = {}
        all_launches = list(runtime_events)
        if driver_events:
            all_launches.extend(driver_events)
        for e in all_launches:
            corr = e.get("args", {}).get("correlation")
            if corr is None or corr in self._corr_to_shape:
                continue
            ts = e["ts"]
            tid = e["tid"]
            pid = e.get("pid", tid)
            shape = self._find_shape(ts, pid, tid)
            if shape:
                self._corr_to_shape[corr] = shape

    def get_shape(self, correlation_id: int) -> str:
        """Return Input Dims string for a kernel's correlation ID, or ''."""
        return self._corr_to_shape.get(correlation_id, "")

    def _find_shape(self, ts: float, pid: int, tid: int) -> str:
        key = (pid, tid)
        intervals = self._shaped_ops.get(key)
        if not intervals:
            return ""
        starts = self._starts_cache[key]
        idx = bisect.bisect_right(starts, ts) - 1
        # Search nearby intervals for the deepest (narrowest) enclosing cpu_op
        best = ""
        best_span = float("inf")
        for i in range(max(0, idx - 5), min(len(intervals), idx + 5)):
            s, e, dims_str = intervals[i]
            if s <= ts <= e:
                span = e - s
                if span < best_span:
                    best_span = span
                    best = dims_str
            elif s > ts + 100:
                break
        return best

    @staticmethod
    def _format_dims(dims) -> str:
        """Format Input Dims into a concise string, omitting empty entries."""
        if not isinstance(dims, list):
            return ""
        non_empty = [d for d in dims if isinstance(d, list) and len(d) > 0]
        if not non_empty:
            return ""
        return str(non_empty)


class KernelCorrelator:
    """Map GPU kernels to modules via correlation ID chain."""

    def __init__(self, runtime_events: List[Dict], roots: List[ModuleNode],
                 driver_events: Optional[List[Dict]] = None):
        # Build correlation → launch event mapping from both cuda_runtime
        # and cuda_driver events.  On B200, many kernels are launched via
        # cuLaunchKernelEx (cuda_driver) instead of cuda_runtime, so both
        # sources are needed for complete correlation coverage.
        self._corr_to_rt: Dict[int, Dict] = {}
        for e in runtime_events:
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                self._corr_to_rt[corr] = e
        if driver_events:
            for e in driver_events:
                corr = e.get("args", {}).get("correlation")
                if corr is not None and corr not in self._corr_to_rt:
                    self._corr_to_rt[corr] = e

        self._index = _IntervalIndex(roots)

    def correlate(self, kernel_events: List[Dict], roots: List[ModuleNode],
                  shape_index: Optional["CpuOpShapeIndex"] = None) -> int:
        """Assign each kernel to its deepest enclosing module. Returns count matched."""
        matched = 0
        for k in kernel_events:
            corr = k.get("args", {}).get("correlation")
            if corr is None:
                continue
            rt = self._corr_to_rt.get(corr)
            if rt is None:
                continue
            cpu_ts = rt["ts"]
            cpu_tid = rt["tid"]
            cpu_pid = rt.get("pid", cpu_tid)
            module = self._index.find_deepest(cpu_ts, cpu_pid, cpu_tid)
            if module is not None:
                if shape_index is not None:
                    k["_input_dims"] = shape_index.get_shape(corr)
                k["_matched"] = True
                module.kernels.append(k)
                matched += 1
        return matched


# ---------------------------------------------------------------------------
# CUDA graph replay correlator
# ---------------------------------------------------------------------------

class CudaGraphCorrelator:
    """Correlate CUDA-graph-replayed kernels to synthetic layer modules."""

    _COMM_RE = re.compile(
        r"all_reduce|cross_device_reduce|nccl|rccl|broadcast|allgather"
        r"|reduce_scatter|quickreduce|all_to_all", re.IGNORECASE)
    _ATTN_RE = re.compile(
        r"aiter::mla_|mla_a8w8|decode_attention|flash_attn|attention|softmax"
        r"|fmha_|mla_reduce|kv_cache|paged_attention|chunk_gated_delta_rule"
        r"|fused_gdn|kn_get_mla_metadata|kn_mla_reduce|gating_delta_rule",
        re.IGNORECASE)
    _MOE_RE = re.compile(
        r"fused_moe|moe_align|topk|expert|MoeFlatmm|MoeSorting|kernel_moe_gemm"
        r"|kernel_moe_mxgemm|shared_experts|grouped_topk|fused_append_shared_experts",
        re.IGNORECASE)

    def __init__(self, runtime_events):
        self._graph_corrs = set()
        for e in runtime_events:
            name = e.get("name", "")
            if "hipGraphLaunch" in name or "cudaGraphLaunch" in name:
                corr = e.get("args", {}).get("correlation")
                if corr is not None:
                    self._graph_corrs.add(corr)

    @property
    def has_graph_replays(self):
        return bool(self._graph_corrs)

    def correlate(self, gpu_events, capture_roots):
        """Split graph-replayed events into synthetic layer modules.

        Returns (new_roots, matched_count).
        """
        # 1. Partition: graph-replay events grouped by correlation ID
        corr_to_events = defaultdict(list)
        for e in gpu_events:
            corr = e.get("args", {}).get("correlation")
            if corr in self._graph_corrs:
                corr_to_events[corr].append(e)

        if not corr_to_events:
            return [], 0

        # Sort each group by timestamp
        for corr in corr_to_events:
            corr_to_events[corr].sort(key=lambda e: e.get("ts", 0))

        # 2. Deduplicate into templates by kernel count (signature)
        templates = {}  # sig_len -> layer boundaries [(start, end, label), ...]
        for corr, evts in corr_to_events.items():
            sig_len = len(evts)
            if sig_len not in templates:
                names = [e.get("name", "") for e in evts]
                templates[sig_len] = self._detect_layers(names)

        # 3. Extract capture-iteration layer names for naming
        capture_layer_names = self._extract_layer_names(capture_roots)

        # 4. Build synthetic module trees for each replay
        new_roots = []
        matched = 0
        replay_idx = 0
        for corr in sorted(corr_to_events,
                           key=lambda c: corr_to_events[c][0].get("ts", 0)):
            evts = corr_to_events[corr]
            sig_len = len(evts)
            layer_bounds = templates.get(sig_len, [(0, sig_len, "unknown")])

            root = ModuleNode(
                name=f"CudaGraphReplay_{replay_idx}",
                module_type="CudaGraphReplay",
                instance_id=replay_idx,
                ts=evts[0].get("ts", 0),
                end=evts[-1].get("ts", 0) + evts[-1].get("dur", 0),
                tid=evts[0].get("tid", 0),
                pid=evts[0].get("pid", 0),
            )
            root._phase = "decode"

            for layer_i, (start, end, label) in enumerate(layer_bounds):
                if layer_i < len(capture_layer_names):
                    layer_name = capture_layer_names[layer_i]
                else:
                    layer_name = f"Layer_{layer_i}"
                mod_type, inst_id = ModuleTreeBuilder._parse_name(layer_name)

                layer_evts = evts[start:end]
                if not layer_evts:
                    continue

                layer_node = ModuleNode(
                    name=layer_name,
                    module_type=mod_type,
                    instance_id=inst_id,
                    ts=layer_evts[0].get("ts", 0),
                    end=layer_evts[-1].get("ts", 0) + layer_evts[-1].get("dur", 0),
                    tid=root.tid,
                    pid=root.pid,
                )
                layer_node._phase = "decode"
                layer_node.kernels = layer_evts
                matched += len(layer_evts)
                root.children.append(layer_node)

            new_roots.append(root)
            replay_idx += 1

        return new_roots, matched

    def _detect_layers(self, names):
        """COMM-based segmentation + half-layer merging.

        Returns [(start_idx, end_idx, type_label), ...].
        """
        cats = [self._quick_cat(n) for n in names]
        comm_pos = [i for i, c in enumerate(cats) if c == "COMM"]

        if len(comm_pos) < 2:
            return [(0, len(names), "unknown")]

        # Build segments ending at each COMM
        segments = []
        prev = 0
        for cp in comm_pos:
            seg = cats[prev:cp + 1]
            segments.append((prev, cp + 1, "ATTN" in seg, "MOE" in seg))
            prev = cp + 1
        if prev < len(cats):
            seg = cats[prev:]
            segments.append((prev, len(cats), "ATTN" in seg, "MOE" in seg))

        # Decide: half-layer vs full-layer model
        attn_only = sum(1 for s in segments if s[2] and not s[3])
        full_layer = sum(1 for s in segments if s[2] and s[3])

        if attn_only > full_layer:
            return self._merge_half_layers(segments)
        else:
            return [(s[0], s[1], self._seg_label(s[2], s[3])) for s in segments]

    @staticmethod
    def _merge_half_layers(segments):
        """Merge adjacent ATTN-only + non-ATTN pairs into full layers."""
        layers = []
        i = 0
        while i < len(segments):
            start, end, has_attn, has_moe = segments[i]
            if has_attn and not has_moe and i + 1 < len(segments):
                _, next_end, _, next_moe = segments[i + 1]
                layers.append((start, next_end, "MoE" if next_moe else "FC"))
                i += 2
            else:
                layers.append((start, end,
                               CudaGraphCorrelator._seg_label(has_attn, has_moe)))
                i += 1
        return layers

    @staticmethod
    def _seg_label(has_attn, has_moe):
        if has_attn and has_moe:
            return "MoE"
        if has_attn:
            return "Attn"
        if has_moe:
            return "MoE"
        return "other"

    def _quick_cat(self, name):
        if self._COMM_RE.search(name):
            return "COMM"
        if self._ATTN_RE.search(name):
            return "ATTN"
        if self._MOE_RE.search(name):
            return "MOE"
        return "X"

    @staticmethod
    def _extract_layer_names(roots):
        """Extract decoder layer names from capture-iteration module tree."""
        names = []
        for root in roots:
            for child in root.children:
                if ("DecoderLayer" in child.module_type
                        or "TransformerBlock" in child.module_type):
                    names.append(child.name)
        return names


# ---------------------------------------------------------------------------
# CPU-op correlator (CPU-only mode: cpu_op → module via time containment)
# ---------------------------------------------------------------------------

class CpuOpCorrelator:
    """Map cpu_ops to modules via time containment on the same thread."""

    def __init__(self, roots: List[ModuleNode]):
        self._index = _IntervalIndex(roots)

    def correlate(self, cpu_ops: List[Dict], roots: List[ModuleNode]) -> int:
        """Assign cpu_ops to deepest enclosing module. Returns count matched."""
        matched = 0
        find = self._index.find_deepest  # avoid attribute lookup in hot loop
        for op in cpu_ops:
            dur = op.get("dur", 0)
            if dur < 1.0:  # skip trivial ops under 1 us
                continue
            ts = op["ts"]
            tid = op["tid"]
            pid = op.get("pid", tid)
            module = find(ts, pid, tid)
            if module is not None:
                module.cpu_ops.append(op)
                matched += 1
        return matched


# ---------------------------------------------------------------------------
# Module aggregator
# ---------------------------------------------------------------------------

# Simple kernel category classification based on kernel name
_KERNEL_CATEGORIES = [
    ("communication", re.compile(
        r"all_reduce|cross_device_reduce|nccl|rccl|broadcast|allgather|reduce_scatter|quickreduce"
        r"|all_to_all",
        re.IGNORECASE)),
    ("embedding", re.compile(r"embedding|rotary|rope|pos_enc", re.IGNORECASE)),
    ("attention", re.compile(
        r"aiter::mla_|mla_a8w8|decode_attention|flash_attn|attention|softmax|fmha_|FmhaBatchPrefill"
        r"|mla_reduce|kv_cache|flashinfer|set_mla_kv|paged_attention|radix"
        r"|chunk_gated_delta_rule|chunk_fwd_kernel|causal_conv1d|fused_gdn", re.IGNORECASE)),
    ("moe", re.compile(
        r"fused_moe|moe_align|topk|expert|MoeFlatmm|MoeSorting|kernel_moe_gemm|kernel_moe_mxgemm"
        r"|shared_experts|grouped_topk|moe_fused_gate|moeSoftmax", re.IGNORECASE)),
    ("quantization", re.compile(
        r"mxfp4|fp8|quant|_gemm_afp4|_fused_rms_mxfp4|dynamic_per_group_scaled_quant"
        r"|_dynamic_mxfp4|fp4x2|_gemm_a8w8|_batched_gemm_a8w8|_fused_rms_fp8", re.IGNORECASE)),
    ("gemm", re.compile(
        r"Cijk_Alik_Bljk|Cijk_Ailk_Bljk|Cijk_SB_|_gemm_a16_w16|Custom_Cijk|gemm|matmul|nvjet",
        re.IGNORECASE)),
    ("normalization", re.compile(
        r"layer_norm|rmsnorm|rms_norm|batch_norm|group_norm|scale_shift"
        r"|rsqrt|_mean_pow",
        re.IGNORECASE)),
    ("conv", re.compile(r"conv2d|conv3d|convolution|Im2Col", re.IGNORECASE)),
    ("elementwise", re.compile(
        r"vectorized_elementwise|elementwise_kernel|binary_op|unary_op"
        r"|aten::add|aten::mul|aten::sub|aten::div|aten::neg|aten::abs|aten::clamp"
        r"|add_kernel|mul_kernel|sub_kernel|div_kernel|fused_add|residual_add"
        r"|CastKernel|cast_kernel|type_cast|fused_bias"
        r"|_to_copy|fused_copy_|fused_zeros|fused_any|fused_fill|fused_where"
        r"|fused_mul_|fused_pow|fused_neg|fused_abs"
        r"|act_and_mul|silu|gelu|relu|swish|sigmoid_gating",
        re.IGNORECASE)),
]


def _categorize_kernel(name: str) -> str:
    """Categorize a kernel name into a high-level category."""
    for cat, pat in _KERNEL_CATEGORIES:
        if pat.search(name):
            return cat
    return "other"


def _categorize_cpu_op(name: str) -> str:
    """Categorize a cpu_op name into a high-level category."""
    for cat, pat in _KERNEL_CATEGORIES:
        if pat.search(name):
            return cat
    return "other"


class ModuleAggregator:
    """Compute per-module statistics with hierarchical breakdown."""

    def aggregate(self, roots: List[ModuleNode], mode: str) -> List[ModuleStats]:
        """Recursively aggregate stats for all root nodes."""
        return [self._aggregate_node(r, 0, mode, "") for r in roots]

    def _aggregate_node(self, node: ModuleNode, depth: int, mode: str,
                        parent_path: str) -> ModuleStats:
        path = f"{parent_path}/{node.name}" if parent_path else node.name
        node_phase = getattr(node, "_phase", "")
        stats = ModuleStats(
            name=node.name,
            module_type=node.module_type,
            instance_id=node.instance_id,
            depth=depth,
        )

        # Direct kernel stats
        for k in node.kernels:
            dur = k.get("dur", 0)
            kname = k.get("name", "")
            stats.self_kernel_time += dur
            stats.kernel_count += 1
            cat = _categorize_kernel(kname)
            prev_dur, prev_cnt = stats.kernel_breakdown.get(cat, (0.0, 0))
            stats.kernel_breakdown[cat] = (prev_dur + dur, prev_cnt + 1)
            stats.kernel_details.append(KernelDetail(
                name=kname, duration=dur, category=cat,
                module_path=path, ts=k.get("ts", 0), phase=node_phase,
                input_dims=k.get("_input_dims", "")))

        # Direct cpu_op stats
        for op in node.cpu_ops:
            dur = op.get("dur", 0)
            opname = op.get("name", "")
            stats.self_cpu_op_time += dur
            stats.cpu_op_count += 1
            cat = _categorize_cpu_op(opname)
            prev_dur, prev_cnt = stats.kernel_breakdown.get(cat, (0.0, 0))
            stats.kernel_breakdown[cat] = (prev_dur + dur, prev_cnt + 1)
            dims = op.get("args", {}).get("Input Dims", "")
            if dims:
                dims = CpuOpShapeIndex._format_dims(dims)
            stats.kernel_details.append(KernelDetail(
                name=opname, duration=dur, category=cat,
                module_path=path, ts=op.get("ts", 0), phase=node_phase,
                input_dims=dims if dims else ""))

        # Sort own details by timestamp
        stats.kernel_details.sort(key=lambda d: d.ts)

        # Recurse children
        stats.total_kernel_time = stats.self_kernel_time
        stats.total_cpu_op_time = stats.self_cpu_op_time
        for child in node.children:
            child_stats = self._aggregate_node(child, depth + 1, mode, path)
            stats.children_stats.append(child_stats)
            stats.total_kernel_time += child_stats.total_kernel_time
            stats.total_cpu_op_time += child_stats.total_cpu_op_time
            stats.kernel_count += child_stats.kernel_count
            stats.cpu_op_count += child_stats.cpu_op_count
            # Merge child breakdown into parent's total
            for cat, (dur, cnt) in child_stats.kernel_breakdown.items():
                prev_dur, prev_cnt = stats.kernel_breakdown.get(cat, (0.0, 0))
                stats.kernel_breakdown[cat] = (prev_dur + dur, prev_cnt + cnt)

        return stats


# ---------------------------------------------------------------------------
# Phase detector (prefill vs decode for LLM traces)
# ---------------------------------------------------------------------------

class PhaseDetector:
    """Detect prefill vs decode phase for DeepseekV2Model forward passes."""

    def detect_from_markers(self, roots: List[ModuleNode], phase_markers: List[Tuple]):
        """Tag modules using pre-extracted phase markers (ts, phase, tid, pid)."""
        if not phase_markers:
            self._detect_from_cpu_ops(roots)
            return
        phase_markers_sorted = sorted(phase_markers)
        self._tag_nodes_from_phases(roots, phase_markers_sorted)

    def _tag_nodes_from_phases(self, nodes: List[ModuleNode], phases: List[Tuple],
                               parent_phase: Optional[str] = None):
        phase_ts = [p[0] for p in phases]
        for node in nodes:
            if parent_phase:
                # Children inherit parent phase — a forward pass is entirely
                # prefill or decode; independent bisect on children can land on
                # a stale marker when timestamps are close to a boundary.
                node._phase = parent_phase  # type: ignore[attr-defined]
            else:
                idx = bisect.bisect_right(phase_ts, node.ts) - 1
                if 0 <= idx < len(phases):
                    _, phase, _, _ = phases[idx]
                    node._phase = phase  # type: ignore[attr-defined]
            self._tag_nodes_from_phases(
                node.children, phases,
                parent_phase=getattr(node, "_phase", None))

    def _detect_from_cpu_ops(self, nodes: List[ModuleNode]):
        """Fall back to checking cpu_op names for prefill/decode keywords."""
        for node in nodes:
            prefill_count = sum(1 for op in node.cpu_ops if "prefill" in op.get("name", "").lower() or "extend" in op.get("name", "").lower())
            decode_count = sum(1 for op in node.cpu_ops if "decode" in op.get("name", "").lower())
            if prefill_count > decode_count:
                node._phase = "prefill"  # type: ignore[attr-defined]
            elif decode_count > prefill_count:
                node._phase = "decode"  # type: ignore[attr-defined]
            self._detect_from_cpu_ops(node.children)


# ---------------------------------------------------------------------------
# Model info text generator (from trace module roots, no re-load)
# ---------------------------------------------------------------------------

def _generate_model_info(roots: List[ModuleNode],
                         config_path: Optional[str] = None,
                         png_path: Optional[str] = None
                         ) -> Tuple[Optional[str], Optional[str]]:
    """Build architecture diagram text + optional PNG from already-parsed module roots.

    Returns (text, png_path_or_None).  Reuses the trace's module roots so the
    trace file is never re-loaded.
    """
    if not roots:
        return None, None

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from model_inspector import (match_template, template_to_text,
                                     ArchDiagramRenderer)
    except ImportError as e:
        logger.warning("model_inspector not available: %s", e)
        return None, None

    best_root = max(roots, key=lambda r: r.end - r.ts)
    root_type = best_root.module_type

    cfg = {}
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    template = match_template(root_type, cfg, raw_root=best_root)
    if template is None:
        logger.info("No arch template for %s, skipping Model Info tab", root_type)
        return None, None

    text = template_to_text(template)

    actual_png = None
    if png_path:
        try:
            renderer = ArchDiagramRenderer()
            renderer.render_template_to_png(template, png_path)
            if os.path.isfile(png_path):
                actual_png = png_path
        except Exception as e:
            logger.warning("Failed to render arch diagram PNG: %s", e)

    return text, actual_png


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate console and Excel reports from module stats."""

    def print_hierarchy(self, stats_list: List[ModuleStats], mode: str,
                        module_filter: Optional[str] = None, top_n: int = 5):
        """Print hierarchical module report to console."""
        print("\n" + "=" * 70)
        print("  Model Module Hierarchy")
        print("=" * 70)

        for stats in stats_list:
            self._print_node(stats, mode, module_filter, top_n, prefix="", is_last=True)

    def _print_node(self, stats: ModuleStats, mode: str,
                    module_filter: Optional[str], top_n: int,
                    prefix: str, is_last: bool):
        # Apply filter
        if module_filter and module_filter not in stats.module_type:
            # Still recurse children in case they match
            for i, child in enumerate(stats.children_stats):
                self._print_node(child, mode, module_filter, top_n,
                                 prefix, i == len(stats.children_stats) - 1)
            return

        time_val = stats.total_kernel_time if mode == "full" else stats.total_cpu_op_time
        count = stats.kernel_count if mode == "full" else stats.cpu_op_count
        count_label = "kernels" if mode == "full" else "ops"

        connector = "\u2514\u2500 " if is_last else "\u251c\u2500 "
        phase_tag = f" [{getattr(stats, 'phase', '')}]" if getattr(stats, "phase", "") else ""

        print(f"{prefix}{connector}{stats.name}  [{time_val:,.0f} us, {count} {count_label}]{phase_tag}")

        child_prefix = prefix + ("   " if is_last else "\u2502  ")

        # Print breakdown if this node has significant content
        if time_val > 0 and stats.kernel_breakdown:
            sorted_cats = sorted(stats.kernel_breakdown.items(), key=lambda x: -x[1][0])
            for cat, (dur, cnt) in sorted_cats:
                if dur <= 0:
                    continue
                pct = dur / time_val * 100 if time_val > 0 else 0
                print(f"{child_prefix}   {cat:15s} {dur:>12,.0f} us ({pct:5.1f}%, {cnt} items)")

        # Print children
        for i, child in enumerate(stats.children_stats):
            self._print_node(child, mode, module_filter, top_n,
                             child_prefix, i == len(stats.children_stats) - 1)

    def print_layer_detail(self, stats_list: List[ModuleStats], mode: str,
                           detail_module: str, detail_instance: Optional[int] = None):
        """Print kernel-by-kernel detail for a specific module type (one representative instance).

        Shows the full sequence of kernels in time order — the repetitive pattern within a layer.
        """
        # Find matching module instances
        matches: List[ModuleStats] = []
        self._find_modules(stats_list, detail_module, matches)

        if not matches:
            print(f"\nNo modules matching '{detail_module}' found.")
            return

        # Pick the instance to show
        if detail_instance is not None:
            target = [m for m in matches if m.instance_id == detail_instance]
            if not target:
                print(f"\nInstance {detail_instance} not found for '{detail_module}'. "
                      f"Available: {sorted(set(m.instance_id for m in matches))}")
                return
            selected = target[0]
            pick_reason = "user-specified"
        else:
            # Pick instance closest to median (most representative)
            selected, pick_reason = self._pick_median_instance(matches, mode)

        # Collect all kernel details from this module and its descendants, in time order
        all_details = self._collect_all_details(selected)
        all_details.sort(key=lambda d: d.ts)

        time_val = selected.total_kernel_time if mode == "full" else selected.total_cpu_op_time

        # Compute stats across all instances for context
        all_times = [(m.total_kernel_time if mode == "full" else m.total_cpu_op_time)
                     for m in matches]
        mean_t, std_t, min_t, max_t, _ = ReportGenerator._time_stats(all_times)

        # Compute wall time (last end - first start) vs sum of durations
        sum_dur = sum(d.duration for d in all_details)
        if all_details:
            wall_start = min(d.ts for d in all_details)
            wall_end = max(d.ts + d.duration for d in all_details)
            wall_time = wall_end - wall_start
        else:
            wall_time = 0

        print(f"\n{'='*100}")
        print(f"  Layer Detail: {selected.name}  [{len(all_details)} items]")
        print(f"  Kernel sum: {sum_dur:,.0f} us  |  Wall time: {wall_time:,.0f} us  "
              f"(overlap: {sum_dur - wall_time:,.0f} us)")
        print(f"  Selected: {pick_reason}  |  "
              f"All {len(matches)} instances: mean={mean_t:,.0f} std={std_t:,.0f} "
              f"min={min_t:,.0f} max={max_t:,.0f}")
        phase = getattr(selected, 'phase', '')
        if phase:
            print(f"  Phase: {phase}")
        print(f"{'='*100}")
        has_dims = any(d.input_dims for d in all_details)
        if has_dims:
            print(f"  {'#':>4s}  {'Duration (us)':>13s}  {'% wall time':>6s}  {'Category':>15s}  {'Module':30s}  {'Kernel Name':50s}  Input Dims")
            print(f"  {'-'*4}  {'-'*13}  {'-'*6}  {'-'*15}  {'-'*30}  {'-'*50}  {'-'*40}")
        else:
            print(f"  {'#':>4s}  {'Duration (us)':>13s}  {'% wall time':>6s}  {'Category':>15s}  {'Module':30s}  Kernel Name")
            print(f"  {'-'*4}  {'-'*13}  {'-'*6}  {'-'*15}  {'-'*30}  {'-'*50}")

        for i, d in enumerate(all_details, 1):
            pct = d.duration / wall_time * 100 if wall_time > 0 else 0
            leaf = d.module_path.rsplit("/", 1)[-1] if "/" in d.module_path else d.module_path
            kname = d.name[:80]
            if has_dims:
                dims = d.input_dims[:60] if d.input_dims else ""
                print(f"  {i:4d}  {d.duration:13,.1f}  {pct:5.1f}%  {d.category:>15s}  {leaf:30s}  {kname:50s}  {dims}")
            else:
                print(f"  {i:4d}  {d.duration:13,.1f}  {pct:5.1f}%  {d.category:>15s}  {leaf:30s}  {kname}")

    def _pick_median_instance(self, matches: List[ModuleStats],
                              mode: str) -> Tuple[ModuleStats, str]:
        """Pick the instance closest to the median total time."""
        if len(matches) == 1:
            return matches[0], f"only instance (id={matches[0].instance_id})"

        timed = [(m.total_kernel_time if mode == "full" else m.total_cpu_op_time, m)
                 for m in matches]
        timed.sort(key=lambda x: x[0])
        mid = len(timed) // 2
        median_val = timed[mid][0]
        # Find closest to median
        best = min(timed, key=lambda x: abs(x[0] - median_val))
        return best[1], (f"closest to median (id={best[1].instance_id}, "
                         f"{best[0]:,.0f} us, median={median_val:,.0f} us)")

    def _find_modules(self, stats_list: List[ModuleStats], module_type: str,
                      out: List[ModuleStats]):
        for s in stats_list:
            if s.module_type == module_type:
                out.append(s)
            self._find_modules(s.children_stats, module_type, out)

    def _collect_all_details(self, stats: ModuleStats) -> List[KernelDetail]:
        """Collect kernel details from this node and all descendants."""
        result = list(stats.kernel_details)
        for child in stats.children_stats:
            result.extend(self._collect_all_details(child))
        return result

    def print_type_summary(self, stats_list: List[ModuleStats], mode: str,
                           top_n_types: int = 3,
                           detail_modules: Optional[List[str]] = None):
        """Print summary matching the Summary tab: metrics, categories, detail tabs."""
        grand_total = sum(
            (s.total_kernel_time if mode == "full" else s.total_cpu_op_time)
            for s in stats_list)

        # Build wrapper chains and type totals for hierarchy display
        wrapper_types: set = set()
        type_children_map: Dict[str, set] = {}
        top_type_names: set = set()
        root_chains: Dict[str, List[str]] = {}
        type_totals: Dict[str, float] = defaultdict(float)
        if not detail_modules:
            top_type_names, root_chains, type_totals, wrapper_types, type_children_map = \
                self._select_detail_types(stats_list, mode, top_n_types)

        # --- Section 1: Kernel Time Summary (with hierarchy) ---
        print("\n" + "=" * 90)
        print("  Kernel Time Summary")
        print("=" * 90)
        print(f"  {'Metric':<45s} {'Value (us)':>14s} {'%':>7s}  {'Detail Tab'}")
        print("  " + "-" * 85)
        print(f"  {'Total Kernel Time':<45s} {grand_total:>14,.1f} {'100.0%':>7s}")

        root_by_type: Dict[str, List[ModuleStats]] = defaultdict(list)
        for s in stats_list:
            root_by_type[s.module_type].append(s)
        sorted_root_types = sorted(
            root_by_type.items(),
            key=lambda x: -sum(
                (s.total_kernel_time if mode == "full" else s.total_cpu_op_time)
                for s in x[1]))
        for rtype, rlist in sorted_root_types:
            rtype_total = sum(
                (s.total_kernel_time if mode == "full" else s.total_cpu_op_time)
                for s in rlist)
            pct = rtype_total / grand_total * 100 if grand_total > 0 else 0
            chain = root_chains.get(rtype, [rtype])
            print(f"  {rtype:<45s} {rtype_total:>14,.1f} {pct:>6.1f}%")
            if len(chain) > 1:
                for i, ctype in enumerate(chain[1:], 1):
                    is_last = (i == len(chain) - 1)
                    connector = "└── " if is_last else "├── "
                    indent = "    " * i
                    ct = type_totals.get(ctype, 0)
                    ct_pct = ct / grand_total * 100 if grand_total > 0 else 0
                    label = f"{indent}{connector}{ctype}"
                    tab = ctype if ctype in top_type_names else ""
                    print(f"  {label:<45s} {ct:>14,.1f} {ct_pct:>6.1f}%  {tab}")

        # --- Section 2: Category breakdown ---
        global_cat_agg: Dict[str, List] = {}
        self._collect_global_category_agg(stats_list, global_cat_agg, mode)
        sorted_cats = sorted(global_cat_agg.items(), key=lambda x: -x[1][0])

        print("\n" + "=" * 90)
        print("  Kernel Category Breakdown")
        print("=" * 90)
        print(f"  {'Category':<20s} {'Count':>8s} {'Total (us)':>14s} {'Avg (us)':>12s} {'%':>8s}")
        print("  " + "-" * 64)
        for cat, (dur, cnt) in sorted_cats:
            pct = dur / grand_total * 100 if grand_total > 0 else 0
            avg = dur / cnt if cnt > 0 else 0
            print(f"  {cat:<20s} {cnt:>8,d} {dur:>14,.1f} {avg:>12,.1f} {pct:>7.1f}%")
        print()

    def _select_detail_types(self, stats_list: List[ModuleStats], mode: str,
                             top_n_types: int) -> Tuple[set, Dict[str, List[str]],
                                                         Dict[str, float], set,
                                                         Dict[str, set]]:
        """Select detail tab types, prioritizing root chain leaves.

        Returns (top_type_names, root_chains, type_totals,
                 wrapper_types, type_children_map).
        """
        type_agg_flat: Dict[str, List[float]] = defaultdict(list)
        self._collect_by_type(stats_list, type_agg_flat, mode)
        sorted_types_flat = sorted(type_agg_flat.items(), key=lambda x: -sum(x[1]))
        wrapper_types = self._find_wrapper_types(stats_list, mode)
        type_children_map = self._build_type_children(stats_list)
        type_totals = {mtype: sum(times) for mtype, times in sorted_types_flat}

        # Build wrapper chains for each root module type
        root_types = {s.module_type for s in stats_list}
        root_chains: Dict[str, List[str]] = {}
        for rtype in root_types:
            chain = self._trace_wrapper_chain_full(
                rtype, wrapper_types, type_children_map, stats_list, mode)
            root_chains[rtype] = chain

        # Step 1: Reserve slots for each root chain's leaf
        top_type_names: set = set()
        skip_types: set = set(wrapper_types)
        for rtype in root_types:
            chain = root_chains[rtype]
            leaf = chain[-1]
            if leaf not in skip_types and leaf not in top_type_names:
                top_type_names.add(leaf)
                desc = self._all_descendant_types(leaf, type_children_map)
                skip_types.update(desc)

        # Step 2: Fill remaining slots from sorted list
        for mtype, _ in sorted_types_flat:
            if len(top_type_names) >= top_n_types:
                break
            if mtype in skip_types:
                continue
            top_type_names.add(mtype)
            desc = self._all_descendant_types(mtype, type_children_map)
            skip_types.update(desc)

        return top_type_names, root_chains, type_totals, wrapper_types, type_children_map

    def _collect_by_type(self, stats_list: List[ModuleStats],
                         agg: Dict[str, List[float]], mode: str):
        for s in stats_list:
            time_val = s.total_kernel_time if mode == "full" else s.total_cpu_op_time
            agg[s.module_type].append(time_val)
            self._collect_by_type(s.children_stats, agg, mode)

    def _collect_by_type_total(self, stats_list: List[ModuleStats],
                               agg: Dict[str, float], mode: str):
        """Collect total time per module type (summed, not list)."""
        for s in stats_list:
            time_val = s.total_kernel_time if mode == "full" else s.total_cpu_op_time
            agg[s.module_type] += time_val
            self._collect_by_type_total(s.children_stats, agg, mode)

    def export_excel(self, stats_list: List[ModuleStats], mode: str,
                     output_path: str, top_n_types: int = 3,
                     detail_modules: Optional[List[str]] = None,
                     config_path: Optional[str] = None,
                     model_info_roots: Optional[List[ModuleNode]] = None):
        """Export analysis to Excel workbook."""
        try:
            import openpyxl
            import openpyxl.worksheet.properties
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill
        except ImportError:
            logger.error("openpyxl not installed. Run: pip install openpyxl")
            return

        self.top_n_types = top_n_types
        wb = Workbook()
        header_font = Font(bold=True)
        header_fill = PatternFill(start_color="CCE5FF", end_color="CCE5FF", fill_type="solid")
        title_font = Font(bold=True, size=12)

        # Pre-compute grand total and category breakdown (used by Summary + Overview)
        grand_total = sum(
            (s.total_kernel_time if mode == "full" else s.total_cpu_op_time)
            for s in stats_list)
        total_kernel_count = sum(
            s.kernel_count if mode == "full" else s.cpu_op_count
            for s in stats_list)
        # Aggregate kernel categories across entire tree
        global_cat_agg: Dict[str, List] = {}
        self._collect_global_category_agg(stats_list, global_cat_agg, mode)

        # Pre-compute detail tab selection (needed by Summary tab and detail sheets)
        if detail_modules:
            top_type_names = set(detail_modules)
        else:
            top_type_names, _, _, _, _ = \
                self._select_detail_types(stats_list, mode, top_n_types)

        # --- Summary tab (one-pager overview) ---
        ws_summary = wb.active
        ws_summary.title = "Summary"
        summary_last_row = self._write_summary_tab(
            ws_summary, stats_list, mode, grand_total,
            total_kernel_count, global_cat_agg,
            header_font, header_fill, title_font,
            top_type_names)

        # --- Overview sheet (hierarchical type tree, with % of Parent + stats) ---
        ws_ov = wb.create_sheet(title="Overview")
        type_headers = ["Module Type", "Depth", "Count",
                        "Mean (us)", "Std (us)", "Min (us)", "Max (us)",
                        "Total (us)", "% of Parent", "Top Kernel"]
        for col, h in enumerate(type_headers, 1):
            cell = ws_ov.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill

        # Build hierarchical type summary
        root_by_type: Dict[str, List[ModuleStats]] = defaultdict(list)
        for s in stats_list:
            root_by_type[s.module_type].append(s)
        sorted_root_types = sorted(
            root_by_type.items(),
            key=lambda x: -sum(
                (s.total_kernel_time if mode == "full" else s.total_cpu_op_time)
                for s in x[1]))
        type_row = 2
        for rtype, rlist in sorted_root_types:
            root_times = [(s.total_kernel_time if mode == "full" else s.total_cpu_op_time)
                          for s in rlist]
            rep = next((s for s in rlist if s.instance_id == 1), rlist[0])
            rep_time = rep.total_kernel_time if mode == "full" else rep.total_cpu_op_time
            root_rep_kernel = self._get_rep_kernel(rep, mode)
            self._write_type_row(ws_ov, type_row, rtype, 0, len(rlist),
                                 root_times, grand_total, bold=True,
                                 rep_kernel=root_rep_kernel)
            type_row += 1
            if rep.children_stats:
                type_row, _ = self._write_children_hierarchy(
                    ws_ov, rep, mode, type_row, rep_time, stats_list)
        ws_ov.column_dimensions["A"].width = 40
        ws_ov.column_dimensions["J"].width = 80

        # --- Module Tree tab (full tree with collapsible outline groups) ---
        ws_tree = wb.create_sheet(title="Module Tree")
        # Outline settings: collapse buttons at top of group (summaryBelow=False)
        ws_tree.sheet_properties.outlinePr = openpyxl.worksheet.properties.Outline(
            summaryBelow=False, summaryRight=False)
        tree_headers = ["Module", "Total Time (us)", "Kernels/Ops",
                        "Breakdown", "Phase"]
        for col, h in enumerate(tree_headers, 1):
            cell = ws_tree.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill
        tree_row = 2
        seen_tree_roots: set = set()
        for s in stats_list:
            # Show one representative instance per root module type
            if s.module_type in seen_tree_roots:
                continue
            seen_tree_roots.add(s.module_type)
            tree_row = self._write_tree_rows(ws_tree, s, mode, tree_row, 0)
        ws_tree.column_dimensions["A"].width = 50
        ws_tree.column_dimensions["D"].width = 60

        # --- GPU Kernels sheet: global kernel name aggregation ---
        ws3 = wb.create_sheet(title="GPU Kernels")
        bd_headers = ["Kernel Name", "Category", "Total Duration (us)",
                       "Count", "Avg (us)", "% of Total", "Module Types"]
        for col, h in enumerate(bd_headers, 1):
            cell = ws3.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill
        bd_row = 2
        bd_row = self._write_kernel_name_breakdown(ws3, stats_list, mode, bd_row,
                                                   grand_total)
        ws3.column_dimensions["A"].width = 80
        ws3.column_dimensions["G"].width = 50

        # --- Per-module-type detail sheets ---
        type_agg_flat: Dict[str, List[float]] = defaultdict(list)
        self._collect_by_type(stats_list, type_agg_flat, mode)
        type_total_time = {mtype: sum(times) for mtype, times in type_agg_flat.items()}
        seen_types: Dict[Tuple[str, str], ModuleStats] = {}
        self._find_median_instance_per_type(stats_list, seen_types, mode,
                                            force_types=top_type_names)
        # Sort detail tabs by total kernel time descending (highest % first)
        sorted_detail_items = sorted(
            seen_types.items(),
            key=lambda item: -type_total_time.get(item[0][0], 0))
        for (mtype, phase), rep_stats in sorted_detail_items:
            if mtype not in top_type_names:
                continue
            # Include phase suffix in sheet name for LLM traces
            if phase:
                suffix = f" ({phase[:3]})"  # "prefill" -> "(pre)", "decode" -> "(dec)"
                sheet_name = f"{mtype[:28 - len(suffix)]}{suffix}"
            else:
                sheet_name = mtype[:28]  # Excel sheet name limit = 31 chars
            ws_det = wb.create_sheet(title=sheet_name)
            all_details = self._collect_all_details(rep_stats)
            all_details.sort(key=lambda d: d.ts)

            # Compute wall time and sum of durations
            sum_dur = sum(d.duration for d in all_details)
            if all_details:
                wall_start = min(d.ts for d in all_details)
                wall_end = max(d.ts + d.duration for d in all_details)
                wall_time = wall_end - wall_start
            else:
                wall_time = 0

            # Row 1: title
            phase_label = f" [{phase}]" if phase else ""
            ws_det.cell(row=1, column=1,
                        value=f"{rep_stats.name}{phase_label} — {len(all_details)} kernels").font = Font(bold=True, size=12)
            ws_det.cell(row=2, column=1,
                        value=f"Kernel sum: {sum_dur:,.0f} us  |  "
                              f"Wall time: {wall_time:,.0f} us  |  "
                              f"Overlap: {sum_dur - wall_time:,.0f} us  "
                              f"(% uses wall time)")

            # Category summary table (rows 3+)
            cat_agg: Dict[str, float] = defaultdict(float)
            for d in all_details:
                cat_agg[d.category] += d.duration
            sorted_cats = sorted(cat_agg.items(), key=lambda x: -x[1])
            cat_total = sum(cat_agg.values())

            cur_row = 3
            for lbl, col_idx in [("Category", 1), ("Percentage (%)", 2),
                                 ("Total Duration (us)", 3)]:
                cell = ws_det.cell(row=cur_row, column=col_idx, value=lbl)
                cell.font = header_font
                cell.fill = header_fill
            cur_row += 1
            for cat, dur in sorted_cats:
                pct_cat = dur / cat_total * 100 if cat_total > 0 else 0
                ws_det.cell(row=cur_row, column=1, value=cat)
                ws_det.cell(row=cur_row, column=2, value=f"{pct_cat:.0f}%")
                ws_det.cell(row=cur_row, column=3, value=round(dur, 1))
                cur_row += 1
            ws_det.cell(row=cur_row, column=1, value="Total").font = Font(bold=True)
            ws_det.cell(row=cur_row, column=2, value="100%").font = Font(bold=True)
            ws_det.cell(row=cur_row, column=3, value=round(cat_total, 1)).font = Font(bold=True)
            cur_row += 2  # blank row before kernel list

            # Kernel detail headers
            det_headers = ["Module", "Input Dims", "Kernel Name",
                          "Duration (us)", "% of wall time", "Category"]
            for col, h in enumerate(det_headers, 1):
                cell = ws_det.cell(row=cur_row, column=col, value=h)
                cell.font = header_font
                cell.fill = header_fill
            cur_row += 1
            detail_truncated = len(all_details) > MAX_ROWS_PER_TAB
            for i, d in enumerate(all_details[:MAX_ROWS_PER_TAB], 1):
                pct = d.duration / wall_time * 100 if wall_time > 0 else 0
                leaf = d.module_path.rsplit("/", 1)[-1] if "/" in d.module_path else d.module_path
                ws_det.cell(row=cur_row, column=1, value=leaf)
                ws_det.cell(row=cur_row, column=2, value=d.input_dims)
                ws_det.cell(row=cur_row, column=3, value=d.name)
                ws_det.cell(row=cur_row, column=4, value=round(d.duration, 1))
                ws_det.cell(row=cur_row, column=5, value=round(pct, 1))
                ws_det.cell(row=cur_row, column=6, value=d.category)
                cur_row += 1
            if detail_truncated:
                ws_det.cell(row=cur_row, column=1,
                            value=f"... truncated at {MAX_ROWS_PER_TAB} rows")
            ws_det.column_dimensions["A"].width = 35
            ws_det.column_dimensions["B"].width = 60
            ws_det.column_dimensions["C"].width = 80

        # --- Model Info tab (architecture text + diagram from trace roots) ---
        tmp_png = None
        if model_info_roots:
            png_dir = os.path.dirname(os.path.abspath(output_path))
            tmp_png = os.path.join(png_dir, "_arch_diagram_tmp.png")
            model_info_text, png_result = _generate_model_info(
                model_info_roots, config_path=config_path,
                png_path=tmp_png)
            if model_info_text:
                self._write_model_info_tab(wb, model_info_text,
                                           header_font, header_fill,
                                           png_path=png_result)
                logger.info("Model Info tab added")
            else:
                logger.warning("Could not generate model info "
                               "(no matching template)")

        wb.save(output_path)
        print(f"\nExcel report saved to: {output_path}")

        if tmp_png and os.path.isfile(tmp_png):
            try:
                os.unlink(tmp_png)
            except OSError:
                pass

    @staticmethod
    def _time_stats(times: List[float]) -> Tuple[float, float, float, float, float]:
        """Compute mean, std, min, max, total from a list of times."""
        n = len(times)
        total = sum(times)
        if n == 0:
            return 0, 0, 0, 0, 0
        mean = total / n
        if n > 1:
            variance = sum((t - mean) ** 2 for t in times) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0
        return mean, std, min(times), max(times), total

    @staticmethod
    def _top_kernel_name(stats: ModuleStats) -> Tuple[str, float]:
        """Find the kernel name with the highest total duration from self kernels.

        Returns (kernel_name, total_duration).  Aggregates across all instances
        of the same kernel name within this module's own kernel_details.
        """
        if not stats.kernel_details:
            return "", 0.0
        by_name: Dict[str, float] = {}
        for d in stats.kernel_details:
            by_name[d.name] = by_name.get(d.name, 0.0) + d.duration
        if not by_name:
            return "", 0.0
        best_name = max(by_name, key=by_name.get)  # type: ignore[arg-type]
        return best_name, by_name[best_name]

    def _collect_global_category_agg(self, stats_list: List[ModuleStats],
                                     agg: Dict[str, List], mode: str):
        """Aggregate kernel categories globally: cat -> [total_dur, count]."""
        for s in stats_list:
            for d in s.kernel_details:
                entry = agg.get(d.category)
                if entry:
                    entry[0] += d.duration
                    entry[1] += 1
                else:
                    agg[d.category] = [d.duration, 1]
            self._collect_global_category_agg(s.children_stats, agg, mode)

    def _write_summary_tab(self, ws, stats_list: List[ModuleStats], mode: str,
                           grand_total: float, total_kernel_count: int,
                           global_cat_agg: Dict[str, List],
                           header_font, header_fill, title_font,
                           top_type_names: set):
        """Write the Summary tab: kernel time summary with detail tabs, category breakdown."""
        from openpyxl.styles import Font, PatternFill

        # Build wrapper chains and type totals
        root_chains: Dict[str, List[str]] = {}
        type_totals: Dict[str, float] = defaultdict(float)
        self._collect_by_type_total(stats_list, type_totals, mode)
        wrapper_types = self._find_wrapper_types(stats_list, mode)
        type_children_map = self._build_type_children(stats_list)
        root_types = {s.module_type for s in stats_list}
        for rtype in root_types:
            chain = self._trace_wrapper_chain_full(
                rtype, wrapper_types, type_children_map, stats_list, mode)
            root_chains[rtype] = chain
        top_type_set = top_type_names

        row = 1

        # --- Section 1: Kernel Time Summary ---
        ws.cell(row=row, column=1, value="Metric").font = header_font
        ws.cell(row=row, column=2, value="Value (us)").font = header_font
        ws.cell(row=row, column=3, value="Percentage").font = header_font
        ws.cell(row=row, column=4, value="Detail Tab").font = header_font
        for c in range(1, 5):
            ws.cell(row=row, column=c).fill = header_fill
        row += 1

        ws.cell(row=row, column=1, value="Total Kernel Time")
        ws.cell(row=row, column=2, value=round(grand_total, 1))
        ws.cell(row=row, column=3, value="100.0%")
        ws.cell(row=row, column=1).font = Font(bold=True)
        row += 1

        root_by_type: Dict[str, List[ModuleStats]] = defaultdict(list)
        for s in stats_list:
            root_by_type[s.module_type].append(s)
        sorted_root_types = sorted(
            root_by_type.items(),
            key=lambda x: -sum(
                (s.total_kernel_time if mode == "full" else s.total_cpu_op_time)
                for s in x[1]))
        for rtype, rlist in sorted_root_types:
            rtype_total = sum(
                (s.total_kernel_time if mode == "full" else s.total_cpu_op_time)
                for s in rlist)
            pct = rtype_total / grand_total * 100 if grand_total > 0 else 0
            chain = root_chains.get(rtype, [rtype])
            ws.cell(row=row, column=1, value=rtype)
            ws.cell(row=row, column=2, value=round(rtype_total, 1))
            ws.cell(row=row, column=3, value=f"{pct:.1f}%")
            row += 1
            if len(chain) > 1:
                for i, ctype in enumerate(chain[1:], 1):
                    is_last = (i == len(chain) - 1)
                    connector = "└── " if is_last else "├── "
                    indent = "    " * i
                    ct = type_totals.get(ctype, 0)
                    ct_pct = ct / grand_total * 100 if grand_total > 0 else 0
                    ws.cell(row=row, column=1, value=f"{indent}{connector}{ctype}")
                    ws.cell(row=row, column=2, value=round(ct, 1))
                    ws.cell(row=row, column=3, value=f"{ct_pct:.1f}%")
                    if ctype in top_type_set:
                        ws.cell(row=row, column=4, value=ctype)
                        for c in range(1, 5):
                            ws.cell(row=row, column=c).font = Font(bold=True)
                    row += 1

        row += 1  # blank row

        # --- Section 2: Category breakdown ---
        ws.cell(row=row, column=1, value="Category").font = header_font
        ws.cell(row=row, column=2, value="Count").font = header_font
        ws.cell(row=row, column=3, value="Total (us)").font = header_font
        ws.cell(row=row, column=4, value="Avg (us)").font = header_font
        ws.cell(row=row, column=5, value="Percentage").font = header_font
        for c in range(1, 6):
            ws.cell(row=row, column=c).fill = header_fill
        row += 1

        sorted_cats = sorted(global_cat_agg.items(), key=lambda x: -x[1][0])
        for cat, (dur, cnt) in sorted_cats:
            pct = dur / grand_total * 100 if grand_total > 0 else 0
            avg = dur / cnt if cnt > 0 else 0
            ws.cell(row=row, column=1, value=cat)
            ws.cell(row=row, column=2, value=cnt)
            ws.cell(row=row, column=3, value=round(dur, 1))
            ws.cell(row=row, column=4, value=round(avg, 1))
            ws.cell(row=row, column=5, value=f"{pct:.1f}%")
            row += 1

        # Column widths
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 18
        ws.column_dimensions["C"].width = 18
        ws.column_dimensions["D"].width = 25
        ws.column_dimensions["E"].width = 15

        return row

    def _write_model_info_tab(self, wb, model_info_text: str,
                              header_font, header_fill,
                              png_path: Optional[str] = None):
        """Write a 'Model Info' tab: text on the left, arch diagram PNG on the right."""
        from openpyxl.styles import Font, Alignment

        ws = wb.create_sheet(title="Model Info")

        row = 1
        ws.cell(row=row, column=1, value="Architecture (text)").font = Font(bold=True, size=12)
        row += 2

        mono_font = Font(name="Courier New", size=9)
        wrap_align = Alignment(wrap_text=False, vertical="top")

        for line in model_info_text.splitlines():
            ws.cell(row=row, column=1, value=line).font = mono_font
            ws.cell(row=row, column=1).alignment = wrap_align
            row += 1

        ws.column_dimensions["A"].width = 80

        if png_path and os.path.isfile(png_path):
            try:
                from openpyxl.drawing.image import Image as XlImage
                img = XlImage(png_path)
                ws.cell(row=1, column=3,
                        value="Architecture (diagram)").font = Font(bold=True, size=12)
                img.anchor = "C3"
                ws.add_image(img)
            except Exception as e:
                logger.warning("Failed to embed arch diagram PNG: %s", e)

    def _write_type_row(self, ws, row: int, label: str, depth: int, count: int,
                        times: List[float], parent_time: float,
                        bold: bool = False, pct_total: Optional[float] = None,
                        rep_kernel: str = ""):
        """Write one row of the Overview hierarchy with stats columns.

        pct_total: if given, use this as the numerator for % of Parent
                   (for children: sum within one parent, not global total).
        rep_kernel: representative kernel name for this row.
        """
        from openpyxl.styles import Font
        mean, std, tmin, tmax, total = self._time_stats(times)
        pct_num = pct_total if pct_total is not None else total
        pct = pct_num / parent_time * 100 if parent_time > 0 else 0
        ws.cell(row=row, column=1, value=label)
        ws.cell(row=row, column=2, value=depth)
        ws.cell(row=row, column=3, value=count)
        ws.cell(row=row, column=4, value=round(mean, 1))
        ws.cell(row=row, column=5, value=round(std, 1))
        ws.cell(row=row, column=6, value=round(tmin, 1))
        ws.cell(row=row, column=7, value=round(tmax, 1))
        ws.cell(row=row, column=8, value=round(total, 1))
        ws.cell(row=row, column=9, value=round(pct, 1))
        ws.cell(row=row, column=10, value=rep_kernel)
        if bold:
            for c in range(1, 11):
                ws.cell(row=row, column=c).font = Font(bold=True)

    def _collect_all_of_type(self, stats_list: List[ModuleStats],
                            module_type: str) -> List[ModuleStats]:
        """Collect ALL instances of a module type across the entire tree."""
        result = []
        for s in stats_list:
            if s.module_type == module_type:
                result.append(s)
            result.extend(self._collect_all_of_type(s.children_stats, module_type))
        return result

    def _write_children_hierarchy(self, ws, rep: ModuleStats, mode: str,
                                  row: int, parent_time: float,
                                  all_stats: List[ModuleStats],
                                  max_depth: int = 5) -> Tuple[int, str]:
        """Recursively write children rows up to max_depth, with cross-instance stats.

        Returns (next_row, rep_kernel) where rep_kernel is the representative
        kernel name from the highest-time child (bubbled up for parent rows).
        """
        if not rep.children_stats:
            kname, _ = self._top_kernel_name(rep)
            return row, kname

        child_by_type: Dict[str, List[ModuleStats]] = defaultdict(list)
        for child in rep.children_stats:
            child_by_type[child.module_type].append(child)

        sorted_children = sorted(
            child_by_type.items(),
            key=lambda x: -sum(
                (c.total_kernel_time if mode == "full" else c.total_cpu_op_time)
                for c in x[1]))

        rep_self_time = (rep.self_kernel_time if mode == "full"
                         else rep.self_cpu_op_time)
        child_depth = rep.depth + 1

        # Track the highest-time child's representative kernel for bubbling up
        best_child_kernel = ""
        best_child_time = -1.0

        for ctype, clist_in_rep in sorted_children:
            cdepth = clist_in_rep[0].depth
            if cdepth > max_depth:
                continue
            all_of_type = self._collect_all_of_type(all_stats, ctype)
            if all_of_type:
                child_times = [(c.total_kernel_time if mode == "full" else c.total_cpu_op_time)
                               for c in all_of_type]
            else:
                child_times = [(c.total_kernel_time if mode == "full" else c.total_cpu_op_time)
                               for c in clist_in_rep]
            child_indent = "    " * cdepth + "├── "
            rep_child_total = sum(
                (c.total_kernel_time if mode == "full" else c.total_cpu_op_time)
                for c in clist_in_rep)

            # Recurse first to get child's rep_kernel before writing this row
            child_rep = next((c for c in clist_in_rep if c.instance_id == 1), clist_in_rep[0])
            child_rep_time = child_rep.total_kernel_time if mode == "full" else child_rep.total_cpu_op_time

            # For leaf children (no sub-children), use their own top kernel
            # For parent children, recurse to get bubbled-up kernel
            if not child_rep.children_stats:
                child_kernel, _ = self._top_kernel_name(child_rep)
            else:
                # Peek: the recursion below will write sub-rows and return the kernel
                # We need to write THIS row first, then recurse.
                # So compute the kernel from the highest-time grandchild.
                child_kernel = self._get_rep_kernel(child_rep, mode)

            self._write_type_row(ws, row, f"{child_indent}{ctype}",
                                 cdepth, len(child_times), child_times,
                                 parent_time, pct_total=rep_child_total,
                                 rep_kernel=child_kernel)
            row += 1

            row, _ = self._write_children_hierarchy(
                ws, child_rep, mode, row, child_rep_time, all_stats, max_depth)

            if rep_child_total > best_child_time:
                best_child_time = rep_child_total
                best_child_kernel = child_kernel

        # Add "(self)" row if self_time is significant (>0.5% of parent)
        if rep_self_time > 0 and parent_time > 0:
            self_pct = rep_self_time / parent_time * 100
            if self_pct >= 0.5:
                self_indent = "    " * child_depth + "└── "
                from openpyxl.styles import Font
                self_kernel, _ = self._top_kernel_name(rep)
                self._write_type_row(
                    ws, row, f"{self_indent}(self)", child_depth, 1,
                    [rep_self_time], parent_time, pct_total=rep_self_time,
                    rep_kernel=self_kernel)
                for c in range(1, 11):
                    ws.cell(row=row, column=c).font = Font(italic=True, color="888888")
                row += 1
                if rep_self_time > best_child_time:
                    best_child_kernel = self_kernel

        return row, best_child_kernel

    def _get_rep_kernel(self, stats: ModuleStats, mode: str) -> str:
        """Get representative kernel name by following the highest-time child type
        down the tree until a leaf is reached.

        Groups children by module_type (matching the Summary sheet's grouping)
        so that e.g. 4 WanUpBlock instances are summed together before comparing
        against self_time.
        """
        if not stats.children_stats:
            kname, _ = self._top_kernel_name(stats)
            return kname

        # Group children by type and sum their times
        child_by_type: Dict[str, List[ModuleStats]] = defaultdict(list)
        for c in stats.children_stats:
            child_by_type[c.module_type].append(c)

        best_type = ""
        best_type_time = 0.0
        best_type_rep: Optional[ModuleStats] = None
        for ctype, clist in child_by_type.items():
            type_total = sum(
                c.total_kernel_time if mode == "full" else c.total_cpu_op_time
                for c in clist)
            if type_total > best_type_time:
                best_type_time = type_total
                best_type = ctype
                best_type_rep = next(
                    (c for c in clist if c.instance_id == 1), clist[0])

        self_time = stats.self_kernel_time if mode == "full" else stats.self_cpu_op_time

        if self_time > best_type_time or best_type_rep is None:
            kname, _ = self._top_kernel_name(stats)
            return kname

        return self._get_rep_kernel(best_type_rep, mode)

    def _write_tree_rows(self, ws, stats: ModuleStats, mode: str,
                         row: int, depth: int, max_depth: int = 3) -> int:
        """Write tree-structured rows with Excel outline grouping for collapse/expand."""
        if depth > max_depth:
            return row
        if row > MAX_ROWS_PER_TAB + 1:
            ws.cell(row=row, column=1, value=f"... truncated at {MAX_ROWS_PER_TAB} rows")
            return row + 1
        time_val = stats.total_kernel_time if mode == "full" else stats.total_cpu_op_time
        count = stats.kernel_count if mode == "full" else stats.cpu_op_count
        phase = getattr(stats, "phase", "")

        indent = ("    " * depth + "├── ") if depth > 0 else ""
        ws.cell(row=row, column=1, value=f"{indent}{stats.name}")
        ws.cell(row=row, column=2, value=round(time_val, 1))
        ws.cell(row=row, column=3, value=count)

        # Build breakdown string
        if time_val > 0 and stats.kernel_breakdown:
            parts = []
            sorted_cats = sorted(stats.kernel_breakdown.items(), key=lambda x: -x[1][0])
            for cat, (dur, cnt) in sorted_cats:
                if dur <= 0:
                    continue
                pct = dur / time_val * 100
                parts.append(f"{cat}: {dur:,.0f} us ({pct:.1f}%)")
            ws.cell(row=row, column=4, value=", ".join(parts))

        ws.cell(row=row, column=5, value=phase)

        # Set outline level for collapsible grouping (depth > 0 can be collapsed)
        if depth > 0:
            ws.row_dimensions[row].outline_level = min(depth, 8)  # Excel max = 8

        row += 1

        for child in stats.children_stats:
            row = self._write_tree_rows(ws, child, mode, row, depth + 1)

        return row

    @staticmethod
    def _find_wrapper_types(stats_list: List[ModuleStats], mode: str,
                            threshold: float = 0.8) -> set:
        """Find module types that are thin wrappers around a dominant child.

        A type is a wrapper if, in its representative instance, a single child
        type accounts for >threshold of its total time.  These are skipped in
        auto-selection to prefer the smaller repetitive unit.
        Applies transitively: if A wraps B wraps C, both A and B are skipped.
        """
        # Collect one representative per type (first instance seen)
        reps: Dict[str, ModuleStats] = {}
        ReportGenerator._collect_first_instance(stats_list, reps)

        skip: set = set()
        for mtype, rep in reps.items():
            parent_time = (rep.total_kernel_time if mode == "full"
                           else rep.total_cpu_op_time)
            if parent_time <= 0 or not rep.children_stats:
                continue
            # Group children by type within this one instance
            child_by_type: Dict[str, float] = defaultdict(float)
            for c in rep.children_stats:
                ct = c.total_kernel_time if mode == "full" else c.total_cpu_op_time
                child_by_type[c.module_type] += ct
            for ctype, ctime in child_by_type.items():
                if ctime / parent_time > threshold:
                    skip.add(mtype)
                    break
        return skip

    @staticmethod
    def _collect_first_instance(stats_list: List[ModuleStats],
                                out: Dict[str, ModuleStats]):
        """Collect first instance of each module type from the tree."""
        for s in stats_list:
            if s.module_type not in out:
                out[s.module_type] = s
            ReportGenerator._collect_first_instance(s.children_stats, out)

    @staticmethod
    def _build_type_children(stats_list: List[ModuleStats]) -> Dict[str, set]:
        """Build parent_type -> set of direct child types from the stats tree."""
        result: Dict[str, set] = defaultdict(set)
        for s in stats_list:
            for child in s.children_stats:
                result[s.module_type].add(child.module_type)
            child_map = ReportGenerator._build_type_children(s.children_stats)
            for ptype, ctypes in child_map.items():
                result[ptype].update(ctypes)
        return result

    @staticmethod
    def _all_descendant_types(mtype: str, type_children: Dict[str, set]) -> set:
        """Get all transitive descendant types of a given module type."""
        result: set = set()
        stack = list(type_children.get(mtype, set()))
        while stack:
            ct = stack.pop()
            if ct not in result:
                result.add(ct)
                stack.extend(type_children.get(ct, set()))
        return result

    @staticmethod
    def _trace_wrapper_chain(mtype: str, wrapper_types: set,
                             type_children: Dict[str, set],
                             stats_list: List[ModuleStats],
                             mode: str) -> str:
        """Follow wrapper chain from mtype to the first non-wrapper descendant."""
        chain = ReportGenerator._trace_wrapper_chain_full(
            mtype, wrapper_types, type_children, stats_list, mode)
        return chain[-1] if chain else mtype

    @staticmethod
    def _trace_wrapper_chain_full(mtype: str, wrapper_types: set,
                                  type_children: Dict[str, set],
                                  stats_list: List[ModuleStats],
                                  mode: str) -> List[str]:
        """Follow wrapper chain, returning the full list [mtype, child, ..., leaf]."""
        reps: Dict[str, ModuleStats] = {}
        ReportGenerator._collect_first_instance(stats_list, reps)
        chain = [mtype]
        current = mtype
        visited: set = set()
        while current in wrapper_types and current not in visited:
            visited.add(current)
            rep = reps.get(current)
            if not rep or not rep.children_stats:
                break
            child_by_type: Dict[str, float] = defaultdict(float)
            for c in rep.children_stats:
                ct = c.total_kernel_time if mode == "full" else c.total_cpu_op_time
                child_by_type[c.module_type] += ct
            best_child = max(child_by_type, key=child_by_type.get)
            chain.append(best_child)
            current = best_child
        return chain

    def _find_median_instance_per_type(self, stats_list: List[ModuleStats],
                                       seen: Dict[Tuple[str, str], ModuleStats],
                                       mode: str,
                                       force_types: Optional[set] = None):
        """Find median instance of each (module_type, phase) pair.

        Groups by (module_type, phase) so that prefill and decode get separate
        representative instances, producing separate detail sheets.
        force_types: if given, always include these types even if they are leaf modules.
        """
        # First collect all instances per type
        all_by_type: Dict[str, List[ModuleStats]] = defaultdict(list)
        self._collect_instances_by_type(stats_list, all_by_type)
        # Pick median for each (type, phase) pair
        for mtype, instances in all_by_type.items():
            if not any(s.children_stats for s in instances):
                if not (force_types and mtype in force_types):
                    continue
            # Sub-group by phase
            by_phase: Dict[str, List[ModuleStats]] = defaultdict(list)
            for s in instances:
                by_phase[getattr(s, "phase", "")].append(s)
            for phase, phase_instances in by_phase.items():
                timed = sorted(phase_instances,
                               key=lambda s: s.total_kernel_time if mode == "full" else s.total_cpu_op_time)
                mid = len(timed) // 2
                seen[(mtype, phase)] = timed[mid]

    def _collect_instances_by_type(self, stats_list: List[ModuleStats],
                                   out: Dict[str, List[ModuleStats]]):
        for s in stats_list:
            if s.kernel_count > 0:
                out[s.module_type].append(s)
            self._collect_instances_by_type(s.children_stats, out)

    def _write_kernel_name_breakdown(self, ws, stats_list: List[ModuleStats],
                                     mode: str, row: int,
                                     grand_total: float) -> int:
        """Write kernel name breakdown aggregated globally (one row per unique kernel name)."""
        # Aggregate across entire tree: kernel_name -> (total_dur, count, category, module_types)
        global_agg: Dict[str, List] = {}  # name -> [dur, count, category, set(module_types)]
        self._collect_global_kernel_agg(stats_list, global_agg)

        sorted_kernels = sorted(global_agg.items(), key=lambda x: -x[1][0])
        truncated = len(sorted_kernels) > MAX_ROWS_PER_TAB
        sorted_kernels = sorted_kernels[:MAX_ROWS_PER_TAB]
        for kname, (dur, cnt, cat, mtypes) in sorted_kernels:
            pct = dur / grand_total * 100 if grand_total > 0 else 0
            ws.cell(row=row, column=1, value=kname)
            ws.cell(row=row, column=2, value=cat)
            ws.cell(row=row, column=3, value=round(dur, 1))
            ws.cell(row=row, column=4, value=cnt)
            ws.cell(row=row, column=5, value=round(dur / cnt, 1) if cnt > 0 else 0)
            ws.cell(row=row, column=6, value=round(pct, 1))
            ws.cell(row=row, column=7, value=", ".join(sorted(mtypes)))
            row += 1
        if truncated:
            ws.cell(row=row, column=1, value=f"... truncated at {MAX_ROWS_PER_TAB} rows")
            row += 1
        return row

    def _collect_global_kernel_agg(self, stats_list: List[ModuleStats],
                                   agg: Dict[str, List]):
        """Recursively aggregate kernel durations globally by kernel name."""
        for s in stats_list:
            for d in s.kernel_details:
                entry = agg.get(d.name)
                if entry:
                    entry[0] += d.duration
                    entry[1] += 1
                    entry[3].add(s.module_type)
                else:
                    agg[d.name] = [d.duration, 1, d.category, {s.module_type}]
            self._collect_global_kernel_agg(s.children_stats, agg)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class TraceModuleAnalyzer:
    """Main orchestrator: load trace, build module tree, correlate, aggregate, report."""

    def __init__(self, trace_path: str, top_n: int = 5,
                 output_path: Optional[str] = None,
                 detail_modules: Optional[List[str]] = None,
                 detail_instance: Optional[int] = None,
                 top_n_types: int = 3, show_tree: bool = False,
                 auto_fix_rocm: bool = True,
                 config_path: Optional[str] = None,
                 model_info: bool = False):
        self.trace_path = trace_path
        self.top_n = top_n
        self.output_path = output_path
        self.detail_modules = detail_modules or []
        self.detail_instance = detail_instance
        self.top_n_types = top_n_types
        self.show_tree = show_tree
        self.auto_fix_rocm = auto_fix_rocm
        self.config_path = config_path
        self.model_info = model_info

    def run(self):
        # Step 1: Load trace with single-pass categorization
        print(f"Loading trace: {self.trace_path}")
        data = self._load_trace(self.trace_path)
        events = data.get("traceEvents", [])
        print(f"  Total events: {len(events):,}")

        # Auto-detect and fix ROCm traces (missing hipGraphLaunch flow events)
        if self.auto_fix_rocm and _HAS_ROCM_FIX:
            data, n_launches, n_injected = _rocm_fix_trace(data)
            events = data.get("traceEvents", [])
            if n_launches > 0:
                print(f"  ROCm trace detected: {n_launches:,} hipGraphLaunch events, "
                      f"{n_injected:,} flow events injected")
        elif self.auto_fix_rocm and not _HAS_ROCM_FIX:
            logger.debug("fix_rocm_trace_flow not available — skipping ROCm auto-fix")

        # Single-pass categorization: extract everything we need
        kernel_events = []
        runtime_events = []
        driver_events = []
        cpu_ops = []
        module_events = []  # pre-filtered nn.Module events
        gpu_memcpy = []
        gpu_memset = []
        phase_markers = []  # (ts, phase, tid, pid) for prefill/decode detection

        MODULE_PREFIX = "nn.Module: "
        for e in events:
            cat = e.get("cat", "")
            if cat == "kernel":
                kernel_events.append(e)
            elif cat == "cuda_runtime":
                runtime_events.append(e)
            elif cat == "cuda_driver":
                driver_events.append(e)
            elif cat == "cpu_op":
                cpu_ops.append(e)
            elif cat == "gpu_memcpy":
                gpu_memcpy.append(e)
            elif cat == "gpu_memset":
                gpu_memset.append(e)
            elif cat == "python_function":
                name = e.get("name", "")
                if name.startswith(MODULE_PREFIX) and e.get("dur") is not None:
                    module_events.append(e)
                elif "forward_batch_info" in name:
                    if "is_extend" in name:
                        phase_markers.append((e["ts"], "prefill", e["tid"], e.get("pid")))
                    elif "is_decode" in name and "is_decode_or_idle" not in name:
                        phase_markers.append((e["ts"], "decode", e["tid"], e.get("pid")))

        # Free the raw events list — we no longer need it
        del events
        del data

        # Detect mode
        mode = "full" if kernel_events else "cpu_only"
        print(f"  kernel events: {len(kernel_events):,}")
        print(f"  cuda_runtime events: {len(runtime_events):,}")
        print(f"  cuda_driver events: {len(driver_events):,}")
        print(f"  cpu_op events: {len(cpu_ops):,}")
        print(f"  nn.Module events: {len(module_events):,}")
        print(f"  gpu_memcpy events: {len(gpu_memcpy):,}")
        print(f"  gpu_memset events: {len(gpu_memset):,}")
        print(f"  phase markers: {len(phase_markers):,}")
        print(f"  Mode: {mode}")

        # Step 2: Build module hierarchy tree
        if not module_events:
            print("\nWARNING: No nn.Module events found. Trace may not have been captured "
                  "with `with_modules=True`.")
            print("Falling back to basic kernel-only analysis.")
            self._fallback_kernel_only(kernel_events + gpu_memcpy + gpu_memset)
            return

        print("\nBuilding module hierarchy...")
        builder = ModuleTreeBuilder()
        roots = builder.build_from_module_events(module_events)
        del module_events  # free memory
        print(f"  Root modules: {len(roots)}")
        if not roots:
            print("ERROR: Failed to build module tree.")
            return

        # Print root module names
        for r in roots[:10]:
            print(f"    {r.name} (tid={r.tid}, {r.end - r.ts:,.0f} us)")
        if len(roots) > 10:
            print(f"    ... and {len(roots) - 10} more")

        # Step 3: Correlate compute events → modules
        if mode == "full":
            print("\nBuilding cpu_op shape index for Input Dims...")
            shape_index = CpuOpShapeIndex(cpu_ops, runtime_events, driver_events)
            print(f"  Indexed {len(shape_index._corr_to_shape):,} correlation→shape mappings")

            print("Correlating kernels to modules via cuda_runtime + cuda_driver...")
            all_gpu_events = kernel_events + gpu_memcpy + gpu_memset
            correlator = KernelCorrelator(runtime_events, roots,
                                          driver_events=driver_events)
            graph_correlator = CudaGraphCorrelator(runtime_events)
            del runtime_events, driver_events  # free memory
            matched = correlator.correlate(all_gpu_events, roots,
                                           shape_index=shape_index)
            del shape_index  # free memory
            print(f"  Matched {matched:,} / {len(all_gpu_events):,} GPU events to modules")

            # Step 3b: CUDA graph replay correlation
            if graph_correlator.has_graph_replays:
                unmatched = [e for e in all_gpu_events if not e.get("_matched")]
                graph_roots, graph_matched = graph_correlator.correlate(
                    unmatched, roots)
                if graph_roots:
                    roots.extend(graph_roots)
                    matched += graph_matched
                    print(f"  CUDA graph: {graph_matched:,} kernels in "
                          f"{len(graph_roots)} replay(s)")
                    print(f"  Total matched: {matched:,} / "
                          f"{len(all_gpu_events):,} GPU events")
        else:
            print("\nCorrelating cpu_ops to modules via time containment...")
            del runtime_events, driver_events
            correlator = CpuOpCorrelator(roots)
            matched = correlator.correlate(cpu_ops, roots)
            print(f"  Matched {matched:,} / {len(cpu_ops):,} cpu_ops to modules")
            del cpu_ops

        # Step 4: Detect prefill vs decode phase
        phase_detector = PhaseDetector()
        phase_detector.detect_from_markers(roots, phase_markers)
        self._propagate_phase(roots)

        # Step 5: Aggregate stats
        print("\nAggregating module statistics...")
        aggregator = ModuleAggregator()
        stats_list = aggregator.aggregate(roots, mode)
        # Propagate phase from nodes
        self._copy_phase_to_stats(roots, stats_list)

        # Step 6: Output
        reporter = ReportGenerator()
        reporter.print_type_summary(stats_list, mode,
                                    top_n_types=self.top_n_types,
                                    detail_modules=self.detail_modules)

        for dm in self.detail_modules:
            reporter.print_layer_detail(stats_list, mode, dm,
                                        self.detail_instance)

        if self.show_tree:
            # Filter tree to detail modules if specified, otherwise show all
            tree_filter = self.detail_modules[0] if self.detail_modules else None
            reporter.print_hierarchy(stats_list, mode, tree_filter, self.top_n)

        if self.output_path:
            reporter.export_excel(stats_list, mode, self.output_path,
                                  top_n_types=self.top_n_types,
                                  detail_modules=self.detail_modules,
                                  config_path=self.config_path,
                                  model_info_roots=roots if self.model_info else None)

    def _propagate_phase(self, nodes: List[ModuleNode]):
        """Propagate phase from parent to children if not set."""
        for node in nodes:
            phase = getattr(node, "_phase", None)
            if phase:
                self._set_phase_recursive(node.children, phase)
            self._propagate_phase(node.children)

    def _set_phase_recursive(self, nodes: List[ModuleNode], phase: str):
        for node in nodes:
            if not getattr(node, "_phase", None):
                node._phase = phase  # type: ignore[attr-defined]
            self._set_phase_recursive(node.children, phase)

    def _copy_phase_to_stats(self, nodes: List[ModuleNode], stats_list: List[ModuleStats]):
        for node, stats in zip(nodes, stats_list):
            stats.phase = getattr(node, "_phase", "")
            self._copy_phase_to_stats(node.children, stats.children_stats)

    def _fallback_kernel_only(self, kernel_events: List[Dict]):
        """Basic kernel-only analysis when no module events are present."""
        if not kernel_events:
            print("No kernel events found either. Nothing to analyze.")
            return

        total_dur = sum(k.get("dur", 0) for k in kernel_events)
        breakdown: Dict[str, Tuple[float, int]] = {}
        for k in kernel_events:
            cat = _categorize_kernel(k.get("name", ""))
            dur = k.get("dur", 0)
            prev_dur, prev_cnt = breakdown.get(cat, (0.0, 0))
            breakdown[cat] = (prev_dur + dur, prev_cnt + 1)

        print(f"\n{'='*70}")
        print("  Kernel-Only Analysis (no module hierarchy)")
        print(f"{'='*70}")
        print(f"  Total kernels: {len(kernel_events):,}")
        print(f"  Total duration: {total_dur:,.0f} us ({total_dur/1000:,.1f} ms)")
        print(f"\n  {'Category':<20s} {'Duration (us)':>14s} {'Count':>8s} {'%':>7s}")
        print("  " + "-" * 51)
        for cat, (dur, cnt) in sorted(breakdown.items(), key=lambda x: -x[1][0]):
            pct = dur / total_dur * 100 if total_dur > 0 else 0
            print(f"  {cat:<20s} {dur:>14,.0f} {cnt:>8,d} {pct:>6.1f}%")

    @staticmethod
    def _load_trace(path: str) -> Dict[str, Any]:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Trace Module Analyzer — correlation-based GPU kernel classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Diffusion trace (full mode — has kernel events)
  python trace_module_analyzer.py wan2.2.trace.json.gz -o report.xlsx

  # LLM trace (CPU-only mode — no kernel events)
  python trace_module_analyzer.py deepseek.trace.json.gz -o report.xlsx

  # Show kernel-by-kernel detail for a module type (repetitive pattern)
  python trace_module_analyzer.py trace.json.gz --detail-module WanTransformerBlock

  # Show specific instance
  python trace_module_analyzer.py trace.json.gz --detail-module WanTransformerBlock --detail-instance 5

  # Show full tree (optionally filtered by --detail-module)
  python trace_module_analyzer.py trace.json.gz --show-tree
  python trace_module_analyzer.py trace.json.gz --show-tree --detail-module WanDecoder3d
""",
    )
    parser.add_argument("trace_file", help="Path to trace file (.json.gz or .json)")
    parser.add_argument("-o", "--output", dest="output", default=None,
                        help="Output Excel file path (.xlsx)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Top N kernels to show per module (default: 5)")
    parser.add_argument("--top-n-types", type=int, default=3,
                        help="Top N module types for detail sheets (default: 3, 0=all)")
    parser.add_argument("--detail-module", nargs="+", default=None,
                        help="Show kernel-by-kernel detail for module type(s) "
                             "(e.g. --detail-module WanTransformerBlock FSDPT5Block)")
    parser.add_argument("--detail-instance", type=int, default=None,
                        help="Which instance to show detail for (default: 1)")
    parser.add_argument("--show-tree", action="store_true",
                        help="Print full module tree to console (verbose)")
    parser.add_argument("--config", metavar="PATH", default=None,
                        help="Path to HuggingFace config.json (enriches architecture diagram)")
    parser.add_argument("--model-info", action="store_true",
                        help="Add a 'Model Info' tab to the Excel report with the "
                             "architecture diagram text (from model_inspector). "
                             "Uses the trace's module hierarchy — no extra files needed. "
                             "Combine with --config for richer output.")
    parser.add_argument("--no-rocm-fix", action="store_true",
                        help="Disable automatic ROCm trace fix (hipGraphLaunch flow events)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Resolve output path: if relative, save to same folder as trace file
    output_path = args.output
    if output_path and not os.path.isabs(output_path):
        trace_dir = os.path.dirname(os.path.abspath(args.trace_file))
        output_path = os.path.join(trace_dir, output_path)

    try:
        analyzer = TraceModuleAnalyzer(
            trace_path=args.trace_file,
            top_n=args.top_n,
            output_path=output_path,
            detail_modules=args.detail_module,
            detail_instance=args.detail_instance,
            top_n_types=args.top_n_types if args.top_n_types > 0 else 999,
            show_tree=args.show_tree,
            auto_fix_rocm=not args.no_rocm_fix,
            config_path=args.config,
            model_info=args.model_info,
        )
        analyzer.run()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in trace file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
