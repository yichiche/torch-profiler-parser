#!/usr/bin/env python3
"""Side-by-side comparison of two merged-layer xlsx files.

Reads two outputs of merge_graph_nograph.py and produces a single xlsx where
each Layer block lists the kernels from both runs in adjacent column groups,
aligned by leaf-module name (best-effort).

Example:
    python compare_merged.py b200.xlsx mi355.xlsx -o compare.xlsx \
        --labels B200 MI355
"""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


_INSTANCE_RE = re.compile(r"_\d+$")


def _strip(name: str) -> str:
    return _INSTANCE_RE.sub("", name) if name else name


@dataclass
class Row:
    layer: str          # parent group, e.g. DeepseekV2AttentionMLA
    module: str         # leaf, e.g. RadixAttention (may be "(unmapped …)")
    shape: str
    kernel_name: str
    duration_us: float
    pct: str
    properties: str


def load_merged(path: str) -> Tuple[List[Row], float]:
    """Return (rows, total_wall_us) preserving file order."""
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Merged Layer"]
    rows: List[Row] = []
    cur_layer = ""
    total = 0.0
    for r in ws.iter_rows(values_only=True):
        if r[0] == "TOTAL":
            total = r[5] or 0
            continue
        if r[0] == "Layer":  # header row
            continue
        if r[0]:
            cur_layer = r[0]
        # Subtotal rows have no kernel name in col 4
        if r[3] is None:
            continue
        # Defensive: skip any other non-numeric duration row
        try:
            float(r[5] or 0)
        except (TypeError, ValueError):
            continue
        rows.append(Row(
            layer=cur_layer,
            module=str(r[1] or ""),
            shape=str(r[2] or ""),
            kernel_name=str(r[3] or ""),
            duration_us=float(r[5] or 0),
            pct=str(r[6] or ""),
            properties=str(r[7] or ""),
        ))
    wb.close()
    return rows, total


def group_by_layer(rows: List[Row]) -> "dict[str, List[Row]]":
    out: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        out[r.layer].append(r)
    return out


# ---------------------------------------------------------------------------
# Per-Layer side-by-side alignment.
# ---------------------------------------------------------------------------

def align_layer_block(a_rows: List[Row], b_rows: List[Row]
                      ) -> List[Tuple[Optional[Row], Optional[Row]]]:
    """Pair rows from two runs inside one Layer block.

    Strategy: walk both lists in order. At each step pair if the leaf-module
    name matches (stripped); otherwise advance whichever side has a module
    that the other side has somewhere downstream — falling back to greedy
    insertion when leaf modules diverge entirely. This is intentionally
    simple, since within a Layer block the leaf modules usually appear in
    the same conceptual order on both runs.
    """
    pairs: List[Tuple[Optional[Row], Optional[Row]]] = []
    i = j = 0
    # Precompute remaining leaf-module sets for lookahead
    while i < len(a_rows) or j < len(b_rows):
        if i >= len(a_rows):
            pairs.append((None, b_rows[j])); j += 1; continue
        if j >= len(b_rows):
            pairs.append((a_rows[i], None)); i += 1; continue
        a, b = a_rows[i], b_rows[j]
        if a.module == b.module and a.module:
            pairs.append((a, b)); i += 1; j += 1
            continue
        # Look ahead: does B have a's module later, or does A have b's later?
        a_later_in_b = any(b_rows[k].module == a.module
                           for k in range(j + 1, len(b_rows)))
        b_later_in_a = any(a_rows[k].module == b.module
                           for k in range(i + 1, len(a_rows)))
        if a_later_in_b and not b_later_in_a:
            # B's current row has no match in A → emit B alone, advance j
            pairs.append((None, b)); j += 1
        elif b_later_in_a and not a_later_in_b:
            pairs.append((a, None)); i += 1
        else:
            # Both or neither have downstream matches — emit side-by-side
            # (treating them as the "same slot" even though leaf differs).
            pairs.append((a, b)); i += 1; j += 1
    return pairs


def write_compare(a_path: str, b_path: str, output: str,
                  label_a: str, label_b: str):
    a_rows, a_total = load_merged(a_path)
    b_rows, b_total = load_merged(b_path)
    a_by_layer = group_by_layer(a_rows)
    b_by_layer = group_by_layer(b_rows)

    # Canonical Layer order — fall back to file-order for unknown groups.
    CANONICAL = [
        "prepare_attn",
        "DeepseekV2AttentionMLA",
        "prepare_mlp",
        "DeepseekV2MoE",
        "DeepseekV2MLP",
        "post_layer",
    ]
    seen = set(a_by_layer) | set(b_by_layer)
    merged_layers: List[str] = [L for L in CANONICAL if L in seen]
    # Append any unknown groups in file-order
    for L in [r.layer for r in a_rows] + [r.layer for r in b_rows]:
        if L not in merged_layers and L in seen:
            merged_layers.append(L)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Compare"

    bold = Font(bold=True)
    title_font = Font(bold=True, size=12)
    hdr_fill = PatternFill(start_color="CCE5FF", end_color="CCE5FF",
                           fill_type="solid")
    grp_fill = PatternFill(start_color="EFF6FF", end_color="EFF6FF",
                           fill_type="solid")
    sub_fill = PatternFill(start_color="F0F0F0", end_color="F0F0F0",
                           fill_type="solid")
    delta_pos_fill = PatternFill(start_color="FFE5E5", end_color="FFE5E5",
                                 fill_type="solid")  # slower = pink
    delta_neg_fill = PatternFill(start_color="E5FFE5", end_color="E5FFE5",
                                 fill_type="solid")  # faster = green

    # ── Section 1: Summary table at top ──
    ws.cell(row=1, column=1,
            value=f"Side-by-side: {label_a}  vs  {label_b}").font = title_font

    ws.cell(row=3, column=1, value="Layer group").font = bold
    ws.cell(row=3, column=2, value=f"{label_a} (us)").font = bold
    ws.cell(row=3, column=3, value=f"{label_b} (us)").font = bold
    ws.cell(row=3, column=4, value="Δ (us)").font = bold
    ws.cell(row=3, column=5, value="Δ %").font = bold
    for c in range(1, 6):
        ws.cell(row=3, column=c).fill = hdr_fill

    summary_row = 4
    for L in merged_layers:
        at = sum(r.duration_us for r in a_by_layer.get(L, []))
        bt = sum(r.duration_us for r in b_by_layer.get(L, []))
        d = bt - at
        pct = (d / at * 100) if at else float("nan")
        ws.cell(row=summary_row, column=1, value=L)
        ws.cell(row=summary_row, column=2,
                value=round(at, 1) if at else None)
        ws.cell(row=summary_row, column=3,
                value=round(bt, 1) if bt else None)
        ws.cell(row=summary_row, column=4, value=round(d, 1))
        ws.cell(row=summary_row, column=5,
                value=f"{pct:+.1f}%" if at else "")
        if d > 0.5:
            for c in range(1, 6):
                ws.cell(row=summary_row, column=c).fill = delta_pos_fill
        elif d < -0.5:
            for c in range(1, 6):
                ws.cell(row=summary_row, column=c).fill = delta_neg_fill
        summary_row += 1

    # TOTAL row
    d = b_total - a_total
    pct = (d / a_total * 100) if a_total else 0
    ws.cell(row=summary_row, column=1, value="TOTAL").font = bold
    ws.cell(row=summary_row, column=2, value=round(a_total, 1)).font = bold
    ws.cell(row=summary_row, column=3, value=round(b_total, 1)).font = bold
    ws.cell(row=summary_row, column=4, value=round(d, 1)).font = bold
    ws.cell(row=summary_row, column=5,
            value=f"{pct:+.1f}%").font = bold
    for c in range(1, 6):
        ws.cell(row=summary_row, column=c).fill = sub_fill

    # ── Section 2: per-Layer side-by-side ──
    row_idx = summary_row + 3
    # Header for kernel block
    headers = [
        "Layer",
        f"{label_a}: Module", f"{label_a}: Kernel_name",
        f"{label_a}: t (us)", f"{label_a}: %",
        "",  # spacer
        f"{label_b}: Module", f"{label_b}: Kernel_name",
        f"{label_b}: t (us)", f"{label_b}: %",
    ]
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row=row_idx, column=c, value=h)
        cell.font = bold
        cell.fill = hdr_fill
        cell.alignment = Alignment(horizontal="center")
    row_idx += 1

    for L in merged_layers:
        a_grp = a_by_layer.get(L, [])
        b_grp = b_by_layer.get(L, [])
        pairs = align_layer_block(a_grp, b_grp)

        a_total_L = sum(r.duration_us for r in a_grp)
        b_total_L = sum(r.duration_us for r in b_grp)
        first = True

        for (a, b) in pairs:
            ws.cell(row=row_idx, column=1, value=(L if first else ""))
            if first:
                ws.cell(row=row_idx, column=1).fill = grp_fill
                ws.cell(row=row_idx, column=1).font = bold
                first = False

            if a is not None:
                ws.cell(row=row_idx, column=2, value=a.module)
                ws.cell(row=row_idx, column=3, value=a.kernel_name)
                ws.cell(row=row_idx, column=4, value=round(a.duration_us, 2))
                ws.cell(row=row_idx, column=5, value=a.pct)
            if b is not None:
                ws.cell(row=row_idx, column=7, value=b.module)
                ws.cell(row=row_idx, column=8, value=b.kernel_name)
                ws.cell(row=row_idx, column=9, value=round(b.duration_us, 2))
                ws.cell(row=row_idx, column=10, value=b.pct)

            row_idx += 1

        # Subtotal (no per-row delta column)
        ws.cell(row=row_idx, column=4, value=round(a_total_L, 2)).font = bold
        ws.cell(row=row_idx, column=9, value=round(b_total_L, 2)).font = bold
        for c in range(1, 11):
            ws.cell(row=row_idx, column=c).fill = sub_fill
        row_idx += 1

    widths = [28, 26, 55, 10, 8, 2, 26, 55, 10, 8]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    wb.save(output)
    print(f"Wrote {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="First merged xlsx (baseline)")
    p.add_argument("b", help="Second merged xlsx (target)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--labels", nargs=2, default=("A", "B"))
    args = p.parse_args()
    write_compare(args.a, args.b, args.output, args.labels[0], args.labels[1])


if __name__ == "__main__":
    main()
