#!/usr/bin/env python3
"""Merge a CUDA-graph trace with its no-graph counterpart for one decode layer.

When SGLang runs with --disable-cuda-graph, each kernel is launched inside its
real nn.Module context, so the trace carries the full module hierarchy
(DeepseekV2AttentionMLA → RadixAttention, etc.). When CUDA graph is enabled,
the same kernels are replayed inside a captured graph and the module info is
lost — the analyzer can only label them as the synthetic "Layer_N".

This tool produces a per-layer view that combines:
  • module hierarchy + Input Dims  ← from the no-graph trace
  • per-kernel duration              ← from the graph (perf) trace
aligned via longest-common-subsequence on kernel name within one decoder layer.

The output xlsx has the layout illustrated in the screenshot:
  Layer | Module | shape | Kernel_name | call-times | time (us) | percentage (%) | properties

Where "Layer" is the parent module group (e.g. DeepseekV2AttentionMLA,
DeepseekV2MoE), "Module" is the leaf nn.Module (RMSNorm, RadixAttention, ...),
and rows are grouped by Layer with a subtotal time row at the end of each group.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import gzip
import json

from trace_module_analyzer import (  # noqa: E402
    CpuOpShapeIndex,
    CudaGraphCorrelator,
    KernelCorrelator,
    ModuleAggregator,
    ModuleStats,
    ModuleTreeBuilder,
    PhaseDetector,
    PythonSourceIndex,
    _categorize_kernel,
)

try:
    from fix_rocm_trace_flow import fix_trace as _rocm_fix_trace
    _HAS_ROCM_FIX = True
except ImportError:
    _HAS_ROCM_FIX = False


@dataclass
class KernelRow:
    """One kernel as it will appear in the merged report."""
    layer_group: str    # parent module (e.g. DeepseekV2AttentionMLA)
    module: str         # leaf module (e.g. RadixAttention)
    shape: str          # Input Dims (from nograph cpu_op)
    kernel_name: str
    duration_us: float
    category: str       # 'attention', 'gemm', 'moe', 'communication', ...
    source: str = ""    # source-path (nograph)


# ---------------------------------------------------------------------------
# Step 1 — reuse the analyzer to produce a fully-correlated module tree.
# ---------------------------------------------------------------------------

def _load_trace_json(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def analyze_trace(trace_path: str):
    """Load trace, build module tree, correlate kernels, aggregate. Returns stats."""
    print(f"  Loading {trace_path}")
    data = _load_trace_json(trace_path)
    if _HAS_ROCM_FIX:
        data, _, _ = _rocm_fix_trace(data)
    events = data.get("traceEvents", [])

    runtime_events, driver_events, kernel_events = [], [], []
    cpu_ops, module_events, py_events = [], [], []
    gpu_memcpy, gpu_memset = [], []
    phase_markers = []
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
            elif e.get("dur") is not None:
                py_events.append(e)
                if "model_runner" in name:
                    if ": forward_extend" in name:
                        phase_markers.append((e["ts"], e["ts"] + e["dur"],
                                              "prefill", e["tid"], e.get("pid")))
                    elif ": forward_decode" in name:
                        phase_markers.append((e["ts"], e["ts"] + e["dur"],
                                              "decode", e["tid"], e.get("pid")))

    print(f"  events: kernel={len(kernel_events):,} cpu_op={len(cpu_ops):,} "
          f"nnModule={len(module_events):,}")
    roots = ModuleTreeBuilder().build_from_module_events(module_events)

    shape_index = CpuOpShapeIndex(cpu_ops, runtime_events, driver_events)
    source_index = PythonSourceIndex(py_events, runtime_events, driver_events)
    all_gpu = kernel_events + gpu_memcpy + gpu_memset
    correlator = KernelCorrelator(runtime_events, roots, driver_events)
    correlator.correlate(all_gpu, roots,
                         shape_index=shape_index, source_index=source_index)

    graph_corr = CudaGraphCorrelator(runtime_events)
    if graph_corr.has_graph_replays:
        unmatched = [e for e in all_gpu if not e.get("_matched")]
        new_roots, _ = graph_corr.correlate(unmatched, roots)
        roots.extend(new_roots)

    PhaseDetector().detect_from_markers(roots, phase_markers)
    stats = ModuleAggregator().aggregate(roots, mode="full")

    # Propagate phase from nodes onto stats (mirrors TraceModuleAnalyzer)
    def _copy_phase(node, stat):
        ph = getattr(node, "_phase", "")
        if ph:
            stat.phase = ph
        for cn, cs in zip(node.children, stat.children_stats):
            _copy_phase(cn, cs)
    for r, s in zip(roots, stats):
        _copy_phase(r, s)
    return stats


# ---------------------------------------------------------------------------
# Step 2 — locate one instance of a module type by instance_id.
# ---------------------------------------------------------------------------

def find_instance(stats_list: List[ModuleStats],
                  module_type: str,
                  instance_id: int) -> Optional[ModuleStats]:
    for s in stats_list:
        if s.module_type == module_type and s.instance_id == instance_id:
            return s
        found = find_instance(s.children_stats, module_type, instance_id)
        if found is not None:
            return found
    return None


def list_instances(stats_list: List[ModuleStats],
                   module_type: str) -> List[ModuleStats]:
    out: List[ModuleStats] = []

    def _walk(slist):
        for s in slist:
            if s.module_type == module_type:
                out.append(s)
            _walk(s.children_stats)
    _walk(stats_list)
    return out


def _find_best_prof_halves(pf_stats: List[ModuleStats], module_type: str,
                           layer_index: int,
                           ng_kernels: List["FlatKernel"]) -> List[int]:
    """Scan prof Layer instances near 2*N+1 and pick contiguous halves whose
    combined kernel-name set best matches the nograph layer's kernel set."""
    target = {k.name[:80] for k in ng_kernels}
    all_inst = list_instances(pf_stats, module_type)
    by_id = {s.instance_id: s for s in all_inst}
    n_inst = len(all_inst)
    n_target = len(ng_kernels)

    # Decide overall mode by ratio of prof instances per nograph layer.
    # KimiK2.5 has 61 decoder layers; if prof has ~61, it's full-layer mode;
    # if ~122 per iter, it's halves mode. Use total/61 as a hint.
    layers_per_iter = max(1, n_inst // max(1, _estimate_iter_count(all_inst)))
    halves_mode = layers_per_iter > 80  # > ~61 * 1.5

    if halves_mode:
        candidate_bases = [layer_index * 2 + 1, layer_index * 2]
        length_choices = (2,)  # require attn+mlp pair
    else:
        candidate_bases = [layer_index]
        length_choices = (1,)

    best_score = -1.0
    best_halves: List[int] = []
    for base in candidate_bases:
        for start in range(max(0, base - 2), base + 3):
            for length in length_choices:
                ids = [start + k for k in range(length)]
                insts = [by_id.get(i) for i in ids]
                if any(s is None for s in insts):
                    continue
                kernels = []
                for s in insts:
                    kernels.extend(flatten_decoder_layer(s))
                cand = {k.name[:80] for k in kernels}
                if not cand:
                    continue
                inter = len(target & cand)
                union = len(target | cand)
                jacc = inter / union
                dist_penalty = abs(start - base) * 0.01
                score = jacc - dist_penalty
                if score > best_score:
                    best_score = score
                    best_halves = ids
    if not best_halves:
        best_halves = [layer_index]
    return best_halves


def _estimate_iter_count(all_inst: List[ModuleStats]) -> int:
    """Heuristic: a decode trace usually runs N iterations. Return N by
    looking at how many times instance_id resets (or, simplistically, the
    max instance_id over the list, divided into the total count)."""
    if not all_inst:
        return 1
    # If instance_ids restart per iteration, max+1 is one iter's count.
    # Otherwise total/max gives iter count.
    max_id = max(s.instance_id for s in all_inst)
    total = len(all_inst)
    # If max_id == total-1, instance ids are unique → 1 iter (unusual)
    # Usually there are 5 iterations and ids repeat 0..N-1.
    return max(1, total // (max_id + 1))


def find_decode_instance(stats_list: List[ModuleStats],
                         module_type: str,
                         layer_index: int) -> ModuleStats:
    """Return the decode-phase instance with module_type whose ordinal among
    decode instances equals layer_index (0-based in the decode pool).

    For nograph, decode pool of DeepseekV2DecoderLayer typically holds one full
    sequence of 0..N-1 per decode iteration; we just take the layer_index-th
    instance of the first decode iteration (instance_id == layer_index in
    practice). For prof, "Layer" indices count both halves and accumulate
    across iterations.
    """
    all_inst = list_instances(stats_list, module_type)
    decode = [s for s in all_inst if getattr(s, "phase", "") == "decode"]
    if not decode:
        decode = all_inst
    # Prefer exact instance_id match within the first decode iteration
    by_id = [s for s in decode if s.instance_id == layer_index]
    if by_id:
        return by_id[0]
    if layer_index >= len(decode):
        raise SystemExit(
            f"layer_index={layer_index} out of range for {module_type} "
            f"(decode instances: {len(decode)})")
    return decode[layer_index]


# ---------------------------------------------------------------------------
# Step 3 — collect the in-order kernel list for one module instance
# (descend into children, sorted by timestamp).
# ---------------------------------------------------------------------------

@dataclass
class FlatKernel:
    name: str
    duration: float
    category: str
    leaf_module: str       # e.g. RadixAttention_40
    parent_module: str     # the depth-1-under-decoderlayer ancestor name
    shape: str
    source: str
    ts: float


def flatten_kernels(node_stats: ModuleStats,
                    parent_at_layer_level: str = "") -> List[FlatKernel]:
    """Recursively collect kernels under node_stats in trace-time order.

    parent_at_layer_level: the module-name to record as the 'layer_group'
    for direct-and-descendant kernels. When None, the immediate child of the
    top-level DecoderLayer is used; otherwise inherited from caller.
    """
    out: List[FlatKernel] = []
    own_parent = parent_at_layer_level or node_stats.name

    for kd in node_stats.kernel_details:
        out.append(FlatKernel(
            name=kd.name,
            duration=kd.duration,
            category=kd.category,
            leaf_module=node_stats.name,
            parent_module=own_parent,
            shape=getattr(kd, "input_dims", "") or "",
            source=getattr(kd, "source_path", "") or "",
            ts=getattr(kd, "ts", 0.0),
        ))

    for child in node_stats.children_stats:
        # When walking out of the DecoderLayer down, set parent_at_layer_level
        # to the child's name on first descent.
        sub_parent = (own_parent
                      if parent_at_layer_level else child.name)
        out.extend(flatten_kernels(child, sub_parent))

    out.sort(key=lambda k: k.ts)
    return out


_INSTANCE_SUFFIX_RE = __import__("re").compile(r"_\d+$")


def _strip_instance(name: str) -> str:
    return _INSTANCE_SUFFIX_RE.sub("", name)


def flatten_decoder_layer(layer_stats: ModuleStats) -> List[FlatKernel]:
    """Flatten kernels under one DecoderLayer instance with proper layer groups.

    The 'layer_group' (e.g. DeepseekV2AttentionMLA, DeepseekV2MoE) is the
    name of the immediate child sub-module type. Layer-direct kernels get
    synthetic group labels (prepare_attn, prepare_mlp, post_layer) based on
    their time position relative to the first/last sub-module kernel.
    """
    out: List[FlatKernel] = []
    layer_name = layer_stats.name

    for kd in layer_stats.kernel_details:
        out.append(FlatKernel(
            name=kd.name,
            duration=kd.duration,
            category=kd.category,
            leaf_module=layer_name,
            parent_module=layer_name,       # placeholder; renamed below
            shape=getattr(kd, "input_dims", "") or "",
            source=getattr(kd, "source_path", "") or "",
            ts=getattr(kd, "ts", 0.0),
        ))
    for child in layer_stats.children_stats:
        # parent_at_layer_level is the type name (no instance suffix)
        out.extend(flatten_kernels(child,
                                   parent_at_layer_level=_strip_instance(child.name)))

    out.sort(key=lambda k: k.ts)

    sub_module_present = [i for i, k in enumerate(out)
                          if k.parent_module != layer_name]
    if sub_module_present:
        first_sub = sub_module_present[0]
        last_sub = sub_module_present[-1]
        mid_layer_ks = [i for i in range(first_sub + 1, last_sub)
                        if out[i].parent_module == layer_name]
        prepare_mlp_idx = mid_layer_ks[0] if mid_layer_ks else None
        for i, k in enumerate(out):
            if k.parent_module != layer_name:
                continue
            if i < first_sub:
                k.parent_module = "prepare_attn"
            elif prepare_mlp_idx is not None and i == prepare_mlp_idx:
                k.parent_module = "prepare_mlp"
            elif i > last_sub:
                k.parent_module = "post_layer"
            else:
                k.parent_module = "prepare_mlp"
    return out


# ---------------------------------------------------------------------------
# Step 4 — LCS alignment by kernel name.
# ---------------------------------------------------------------------------

def lcs_align(a: List[FlatKernel], b: List[FlatKernel]
              ) -> List[Tuple[Optional[int], Optional[int]]]:
    """Return list of (idx_a, idx_b) pairs covering both sequences.

    Matched pairs have both indices; insertions have (None, j); deletions
    have (i, None). Match key is the (truncated) kernel name.
    """
    def key(k: FlatKernel) -> str:
        # Use first 80 chars to tolerate trivial template differences.
        return k.name[:80]

    n, m = len(a), len(b)
    # Standard LCS DP
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if key(a[i]) == key(b[j]):
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    out: List[Tuple[Optional[int], Optional[int]]] = []
    i = j = 0
    while i < n and j < m:
        if key(a[i]) == key(b[j]):
            out.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            out.append((i, None))
            i += 1
        else:
            out.append((None, j))
            j += 1
    while i < n:
        out.append((i, None))
        i += 1
    while j < m:
        out.append((None, j))
        j += 1
    return out


# ---------------------------------------------------------------------------
# Step 5 — write the merged xlsx.
# ---------------------------------------------------------------------------

CATEGORY_PROPERTY = {
    "attention": "MLA",
    "moe": "MoE",
    "gemm": "GEMM",
    "communication": "COMM",
    "elementwise": "EW",
    "embedding": "EMB",
}


def build_merged_rows(nograph_kernels: List[FlatKernel],
                      prof_kernels: List[FlatKernel]
                      ) -> List[KernelRow]:
    """LCS-align prof against nograph; carry module info from nograph.

    For prof kernels not matched by LCS, infer the module by looking at the
    nograph anchors on either side: any unmatched prof kernels between
    anchors ng[a..b] get labeled with nograph kernels' module info from
    that gap region (positional fallback).
    """
    pairs = lcs_align(nograph_kernels, prof_kernels)

    # Build mapping: prof_idx -> (ng_idx or None)
    prof_to_ng: Dict[int, Optional[int]] = {}
    last_anchor_ng = -1
    next_anchor_ng_for: Dict[int, int] = {}
    # First pass: matches
    for (ig, ip) in pairs:
        if ip is None:
            continue
        prof_to_ng[ip] = ig

    # Walk through pairs in order and assign nograph fallback labels to
    # unmatched prof kernels. For each "gap" of prof kernels between two
    # nograph anchors, distribute the nograph kernels in that gap evenly.
    rows: List[KernelRow] = []
    # Convert pairs into (ng_idx or None, prof_idx or None) ordered list
    # and walk grouped by prof kernel.
    prof_seq: List[Tuple[int, Optional[int]]] = []  # (prof_idx, ng_idx)
    # Collect mapping while preserving order from LCS
    ip_seen = set()
    ng_gap: List[int] = []
    for (ig, ip) in pairs:
        if ip is None:
            if ig is not None:
                ng_gap.append(ig)
            continue
        # When we hit a prof kernel: emit any queued unmatched ng kernels
        # by associating them positionally with prior unmatched prof rows.
        if ig is not None:
            prof_seq.append((ip, ig))
        else:
            prof_seq.append((ip, None))

    # Now produce rows in prof order, with fallback nograph labels for
    # unmatched prof kernels from neighbouring anchors.
    for k, (ip, ig) in enumerate(prof_seq):
        pk = prof_kernels[ip]
        cat = pk.category or _categorize_kernel(pk.name)
        if ig is not None:
            ng = nograph_kernels[ig]
            layer_group = ng.parent_module
            module = ng.leaf_module
            shape = ng.shape
            source = ng.source
        else:
            # Find nearest prior and next matched anchor in prof_seq
            prev_ng = next(
                (prof_seq[j][1] for j in range(k - 1, -1, -1)
                 if prof_seq[j][1] is not None),
                None)
            next_ng = next(
                (prof_seq[j][1] for j in range(k + 1, len(prof_seq))
                 if prof_seq[j][1] is not None),
                None)
            # Candidate nograph kernels in the gap (prev_ng, next_ng)
            lo = (prev_ng + 1) if prev_ng is not None else 0
            hi = next_ng if next_ng is not None else len(nograph_kernels)
            gap_ng = nograph_kernels[lo:hi]
            # Count unmatched prof kernels in this gap to index into
            gap_prof_indices = [
                j for j in range(
                    (next(
                        (jj + 1 for jj in range(k - 1, -1, -1)
                         if prof_seq[jj][1] is not None), 0)),
                    (next(
                        (jj for jj in range(k + 1, len(prof_seq))
                         if prof_seq[jj][1] is not None), len(prof_seq))))
                if prof_seq[j][1] is None]
            if gap_ng:
                try:
                    rel = gap_prof_indices.index(k)
                except ValueError:
                    rel = 0
                ng_idx = min(int(rel * len(gap_ng) / max(1, len(gap_prof_indices))),
                             len(gap_ng) - 1)
                ng = gap_ng[ng_idx]
                layer_group = ng.parent_module
                # Use the gap-region nograph kernel's leaf module, tagged as
                # unmapped (kernel name differs but spatially in same module).
                module = "(unmapped " + _strip_instance(ng.leaf_module) + ")"
                shape = ""
                source = ng.source
            elif prev_ng is not None:
                # Trailing-tail kernels (no next anchor in nograph).
                # post_layer only when it's a *large* comm kernel that
                # is clearly the layer-output allreduce (>10 us).
                # Otherwise inherit prev anchor's group with (unmapped) tag.
                ng = nograph_kernels[prev_ng]
                if cat == "communication" and pk.duration > 10:
                    layer_group = "post_layer"
                    module = ""
                else:
                    layer_group = ng.parent_module
                    module = "(unmapped " + _strip_instance(ng.leaf_module) + ")"
                shape = ""
                source = ""
            elif next_ng is not None:
                ng = nograph_kernels[next_ng]
                layer_group = ng.parent_module
                module = "(unmapped " + _strip_instance(ng.leaf_module) + ")"
                shape = ""
                source = ""
            else:
                layer_group = "(unmatched)"
                module = "(unknown)"
                shape = ""
                source = ""
        rows.append(KernelRow(
            layer_group=layer_group,
            module=module,
            shape=shape,
            kernel_name=pk.name,
            duration_us=pk.duration,
            category=cat,
            source=source,
        ))
    return rows


def write_xlsx(rows: List[KernelRow], output_path: str,
               title: str = "Merged decode layer"):
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Merged Layer"

    bold = Font(bold=True)
    header_fill = PatternFill(start_color="CCE5FF", end_color="CCE5FF",
                              fill_type="solid")
    group_fill = PatternFill(start_color="EFF6FF", end_color="EFF6FF",
                             fill_type="solid")
    subtotal_fill = PatternFill(start_color="F0F0F0", end_color="F0F0F0",
                                fill_type="solid")

    # Title row
    ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=12)

    # Total wall time = sum (treat as approximate; no overlap data here)
    total = sum(r.duration_us for r in rows) or 1.0

    headers = ["Layer", "Module", "shape", "Kernel_name",
               "call-times", "time (us)", "percentage (%)", "properties",
               "source"]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=3, column=col, value=h)
        c.font = bold
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center")

    row_idx = 4
    # Group rows by layer_group; emit in canonical order, unknowns after.
    CANONICAL = [
        "prepare_attn",
        "DeepseekV2AttentionMLA",
        "prepare_mlp",
        "DeepseekV2MoE",
        "DeepseekV2MLP",
        "post_layer",
    ]
    group_to_rows: Dict[str, List[KernelRow]] = {}
    file_order: List[str] = []
    for r in rows:
        if r.layer_group not in group_to_rows:
            group_to_rows[r.layer_group] = []
            file_order.append(r.layer_group)
        group_to_rows[r.layer_group].append(r)
    seen_groups: List[str] = [g for g in CANONICAL if g in group_to_rows]
    for g in file_order:
        if g not in seen_groups:
            seen_groups.append(g)

    for g in seen_groups:
        group_rows = group_to_rows[g]
        group_total = sum(r.duration_us for r in group_rows)

        for i, r in enumerate(group_rows):
            ws.cell(row=row_idx, column=1, value=(g if i == 0 else ""))
            # Strip _NN instance suffix from leaf module name for readability
            mod_disp = r.module
            if mod_disp and not mod_disp.startswith("("):
                mod_disp = _strip_instance(mod_disp)
            ws.cell(row=row_idx, column=2, value=mod_disp)
            ws.cell(row=row_idx, column=3, value=r.shape)
            ws.cell(row=row_idx, column=4, value=r.kernel_name)
            ws.cell(row=row_idx, column=5, value=1)
            ws.cell(row=row_idx, column=6, value=round(r.duration_us, 3))
            pct = r.duration_us / total * 100 if total else 0
            ws.cell(row=row_idx, column=7, value=f"{pct:.1f}%")
            ws.cell(row=row_idx, column=8,
                    value=CATEGORY_PROPERTY.get(r.category, r.category))
            ws.cell(row=row_idx, column=9, value=r.source)
            if i == 0:
                ws.cell(row=row_idx, column=1).fill = group_fill
                ws.cell(row=row_idx, column=1).font = bold
            row_idx += 1
        # subtotal
        ws.cell(row=row_idx, column=6, value=round(group_total, 3)).font = bold
        for c in range(1, 10):
            ws.cell(row=row_idx, column=c).fill = subtotal_fill
        row_idx += 1

    # Grand total
    ws.cell(row=row_idx, column=1, value="TOTAL").font = bold
    ws.cell(row=row_idx, column=6, value=round(total, 3)).font = bold

    widths = [28, 28, 30, 70, 10, 12, 12, 12, 60]
    from openpyxl.utils import get_column_letter
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    wb.save(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Merge a CUDA-graph trace with its no-graph counterpart "
                    "for one decode layer.")
    p.add_argument("--nograph", required=True,
                   help="Trace from a --disable-cuda-graph run (carries module info)")
    p.add_argument("--prof", required=True,
                   help="Trace from the normal run with CUDA graph (real perf)")
    p.add_argument("--layer-index", type=int, default=40,
                   help="Decoder layer index in the nograph trace (default 40)")
    p.add_argument("--nograph-module", default="DeepseekV2DecoderLayer",
                   help="Module type name for a full decoder layer in nograph")
    p.add_argument("--prof-module", default="Layer",
                   help="Synthetic graph layer module name in prof")
    p.add_argument("--prof-halves", type=int, nargs="+", default=None,
                   help="Specific prof Layer indices to take (default: "
                        "2*layer_index+1 and +2, the attn+moe halves)")
    p.add_argument("-o", "--output", required=True, help="Output xlsx path")
    args = p.parse_args()

    print(f"[1/4] Analyzing nograph trace: {args.nograph}")
    ng_stats = analyze_trace(args.nograph)
    ng_layer = find_decode_instance(ng_stats, args.nograph_module,
                                    args.layer_index)
    print(f"      Picked nograph instance: {ng_layer.name}")
    ng_kernels = flatten_decoder_layer(ng_layer)
    print(f"      {len(ng_kernels)} kernels in nograph layer")

    print(f"[2/4] Analyzing prof trace: {args.prof}")
    pf_stats = analyze_trace(args.prof)

    # Determine which prof Layer halves correspond to this decoder layer.
    # If --prof-halves given, use it. Otherwise scan a window around 2N+1
    # and pick the contiguous run whose combined kernel set best matches
    # nograph's kernel set (by short-name Jaccard similarity).
    if args.prof_halves:
        halves = args.prof_halves
        print(f"      Using user-specified prof Layer halves: {halves}")
    else:
        halves = _find_best_prof_halves(
            pf_stats, args.prof_module, args.layer_index, ng_kernels)
        print(f"      Auto-picked prof Layer halves: {halves}")
    pf_kernels: List[FlatKernel] = []
    for h in halves:
        inst = find_instance(pf_stats, args.prof_module, h)
        if inst is None:
            print(f"      WARNING: prof {args.prof_module}_{h} not found, skipping")
            continue
        pf_kernels.extend(flatten_decoder_layer(inst))
    print(f"      {len(pf_kernels)} kernels combined from prof halves")

    print("[3/4] Aligning by kernel name (LCS) ...")
    rows = build_merged_rows(ng_kernels, pf_kernels)
    print(f"      {len(rows)} merged rows")

    print(f"[4/4] Writing {args.output}")
    title = (f"Merged decode layer {args.layer_index}  |  "
             f"nograph instance {ng_layer.name}  |  "
             f"prof halves {halves}")
    write_xlsx(rows, args.output, title=title)
    print("Done.")


if __name__ == "__main__":
    main()
