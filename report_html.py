#!/usr/bin/env python3
"""Generate a self-contained HTML report from two merged-layer xlsx files.

The report has three sections:
  1. Category bar chart (like categorize.png) — kernels grouped by their
     'properties' tag (GEMM, MoE, MLA, COMM, EW, EMB, quantize, other)
     with side-by-side bars per platform.
  2. Layer-group summary table — totals per Layer block + Δ.
  3. Per-Layer side-by-side detail — every kernel pair from the merged xlsx.

Output is a single .html with no external dependencies (inline SVG + CSS).
"""

import argparse
import re
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from html import escape
from typing import Dict, List, Optional, Tuple

import openpyxl

_INSTANCE_RE = re.compile(r"_\d+$")


def _strip(name: str) -> str:
    return _INSTANCE_RE.sub("", name) if name else name


# ── Property/category canonicalization for the bar chart ────────────────────

PROPERTY_DISPLAY = {
    "GEMM": "GEMM",
    "MoE": "MoE",
    "MLA": "MLA",
    "COMM": "COMM",
    "EW": "elementwise",
    "EMB": "embedding",
    "quantization": "quantize",
    "quant": "quantize",
    "normalization": "norm",
    "norm": "norm",
    "other": "others",
    "": "others",
}

CATEGORY_ORDER = ["GEMM", "MoE", "MLA", "COMM", "elementwise",
                  "embedding", "norm", "quantize", "others"]

CATEGORY_COLORS = {
    "GEMM": "#3b82f6",
    "MoE": "#10b981",
    "MLA": "#f59e0b",
    "COMM": "#ef4444",
    "elementwise": "#a855f7",
    "embedding": "#06b6d4",
    "norm": "#0ea5e9",
    "quantize": "#84cc16",
    "others": "#94a3b8",
}


@dataclass
class Row:
    layer: str
    module: str
    shape: str
    kernel_name: str
    duration_us: float
    pct: str
    properties: str


def load_merged(path: str) -> Tuple[List[Row], float]:
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["Merged Layer"]
    rows: List[Row] = []
    total = 0.0
    cur_layer = ""
    for r in ws.iter_rows(values_only=True):
        if r[0] == "TOTAL":
            total = float(r[5] or 0)
            continue
        if r[0] == "Layer":
            continue
        if r[0]:
            cur_layer = r[0]
        if r[3] is None:
            continue
        try:
            d = float(r[5] or 0)
        except (TypeError, ValueError):
            continue
        rows.append(Row(
            layer=cur_layer, module=str(r[1] or ""), shape=str(r[2] or ""),
            kernel_name=str(r[3] or ""), duration_us=d,
            pct=str(r[6] or ""), properties=str(r[7] or ""),
        ))
    wb.close()
    return rows, total


def categorize_rows(rows: List[Row]) -> Dict[str, float]:
    out: Dict[str, float] = defaultdict(float)
    for r in rows:
        cat = PROPERTY_DISPLAY.get(r.properties, r.properties or "others")
        out[cat] += r.duration_us
    return dict(out)


# ── Layer-block alignment (mirrors compare_merged.py) ───────────────────────

def align_layer_block(a_rows: List[Row], b_rows: List[Row]
                      ) -> List[Tuple[Optional[Row], Optional[Row]]]:
    pairs: List[Tuple[Optional[Row], Optional[Row]]] = []
    i = j = 0
    while i < len(a_rows) or j < len(b_rows):
        if i >= len(a_rows):
            pairs.append((None, b_rows[j])); j += 1; continue
        if j >= len(b_rows):
            pairs.append((a_rows[i], None)); i += 1; continue
        a, b = a_rows[i], b_rows[j]
        if a.module == b.module and a.module:
            pairs.append((a, b)); i += 1; j += 1
            continue
        a_later_in_b = any(b_rows[k].module == a.module
                           for k in range(j + 1, len(b_rows)))
        b_later_in_a = any(a_rows[k].module == b.module
                           for k in range(i + 1, len(a_rows)))
        if a_later_in_b and not b_later_in_a:
            pairs.append((None, b)); j += 1
        elif b_later_in_a and not a_later_in_b:
            pairs.append((a, None)); i += 1
        else:
            pairs.append((a, b)); i += 1; j += 1
    return pairs


# ── Inline SVG bar chart ────────────────────────────────────────────────────

def svg_grouped_bars(cats: List[str], a_vals: List[float], b_vals: List[float],
                     label_a: str, label_b: str, title: str) -> str:
    W, H = 900, 380
    pad_left, pad_right, pad_top, pad_bot = 60, 30, 60, 130
    plot_w = W - pad_left - pad_right
    plot_h = H - pad_top - pad_bot
    max_v = max(max(a_vals + b_vals), 1)
    # Round up max for grid
    grid_step = 5.0 if max_v <= 50 else (10.0 if max_v <= 100 else 20.0)
    max_grid = (int(max_v / grid_step) + 1) * grid_step

    n = len(cats)
    group_w = plot_w / n
    bar_w = group_w * 0.32
    gap = group_w * 0.06

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'style="background:#fff;border:1px solid #e5e7eb;border-radius:6px">']
    # Title
    parts.append(f'<text x="{W/2}" y="22" text-anchor="middle" '
                 f'font-size="14" font-weight="bold" fill="#111827">{escape(title)}</text>')

    # Y axis grid + labels
    n_ticks = int(max_grid / grid_step)
    for t in range(n_ticks + 1):
        val = t * grid_step
        y = pad_top + plot_h - (val / max_grid * plot_h)
        parts.append(f'<line x1="{pad_left}" y1="{y}" x2="{pad_left+plot_w}" y2="{y}" '
                     f'stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{pad_left-6}" y="{y+4}" text-anchor="end" '
                     f'font-size="11" fill="#6b7280">{val:g}</text>')
    parts.append(f'<text x="20" y="{pad_top+plot_h/2}" font-size="11" '
                 f'fill="#6b7280" transform="rotate(-90 20 {pad_top+plot_h/2})" '
                 f'text-anchor="middle">TIME(us)</text>')

    # Bars + value labels
    for i, cat in enumerate(cats):
        cx = pad_left + i * group_w + group_w / 2
        a_h = (a_vals[i] / max_grid) * plot_h
        b_h = (b_vals[i] / max_grid) * plot_h
        a_x = cx - bar_w - gap / 2
        b_x = cx + gap / 2
        a_y = pad_top + plot_h - a_h
        b_y = pad_top + plot_h - b_h
        color = CATEGORY_COLORS.get(cat, "#94a3b8")
        # A bar (red-ish for first)
        parts.append(f'<rect x="{a_x}" y="{a_y}" width="{bar_w}" height="{a_h}" '
                     f'fill="#ef4444" stroke="#991b1b" stroke-width="0.5"/>')
        parts.append(f'<text x="{a_x+bar_w/2}" y="{a_y-4}" text-anchor="middle" '
                     f'font-size="10" fill="#111827">{a_vals[i]:.1f}</text>')
        # B bar (green for second)
        parts.append(f'<rect x="{b_x}" y="{b_y}" width="{bar_w}" height="{b_h}" '
                     f'fill="#10b981" stroke="#065f46" stroke-width="0.5"/>')
        parts.append(f'<text x="{b_x+bar_w/2}" y="{b_y-4}" text-anchor="middle" '
                     f'font-size="10" fill="#111827">{b_vals[i]:.1f}</text>')
        # X-axis category label
        parts.append(f'<text x="{cx}" y="{pad_top+plot_h+16}" text-anchor="middle" '
                     f'font-size="12" font-weight="bold" fill="#111827">{escape(cat)}</text>')

    # Embedded data table below the chart
    row1_y = pad_top + plot_h + 36
    row2_y = pad_top + plot_h + 58
    parts.append(f'<rect x="{pad_left-6}" y="{row1_y-14}" width="14" height="14" fill="#ef4444"/>')
    parts.append(f'<text x="{pad_left+12}" y="{row1_y-2}" font-size="11" fill="#111827" '
                 f'font-weight="bold">{escape(label_a)}</text>')
    parts.append(f'<rect x="{pad_left-6}" y="{row2_y-14}" width="14" height="14" fill="#10b981"/>')
    parts.append(f'<text x="{pad_left+12}" y="{row2_y-2}" font-size="11" fill="#111827" '
                 f'font-weight="bold">{escape(label_b)}</text>')
    for i, cat in enumerate(cats):
        cx = pad_left + i * group_w + group_w / 2
        parts.append(f'<text x="{cx}" y="{row1_y-2}" text-anchor="middle" '
                     f'font-size="11" fill="#7f1d1d">{a_vals[i]:.2f}</text>')
        parts.append(f'<text x="{cx}" y="{row2_y-2}" text-anchor="middle" '
                     f'font-size="11" fill="#064e3b">{b_vals[i]:.2f}</text>')

    # Legend at the very bottom
    legend_y = H - 18
    parts.append(f'<rect x="{W/2-80}" y="{legend_y-10}" width="12" height="12" fill="#ef4444"/>')
    parts.append(f'<text x="{W/2-62}" y="{legend_y}" font-size="12">{escape(label_a)}</text>')
    parts.append(f'<rect x="{W/2+10}" y="{legend_y-10}" width="12" height="12" fill="#10b981"/>')
    parts.append(f'<text x="{W/2+28}" y="{legend_y}" font-size="12">{escape(label_b)}</text>')

    parts.append('</svg>')
    return "".join(parts)


# ── HTML generation ─────────────────────────────────────────────────────────

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 24px auto; max-width: 1400px; color: #111827; }
h1 { margin: 0 0 8px 0; font-size: 22px; }
h2 { margin: 28px 0 12px 0; font-size: 18px; color: #1f2937;
     border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }
.subtitle { color: #6b7280; margin-bottom: 18px; }
table { border-collapse: collapse; font-size: 13px; margin-bottom: 8px; }
th, td { padding: 4px 10px; border: 1px solid #e5e7eb; text-align: right; }
th { background: #f3f4f6; font-weight: 600; }
td.l, th.l { text-align: left; }
tr.layer-row td { background: #eff6ff; font-weight: 600; }
tr.subtotal td { background: #f3f4f6; font-weight: 600; }
tr.total td { background: #d1d5db; font-weight: 700; }
.slower { background: #fee2e2 !important; }
.faster { background: #d1fae5 !important; }
.dim { color: #9ca3af; }
.kname { font-family: ui-monospace, Consolas, monospace; font-size: 11.5px;
         max-width: 320px; overflow: hidden; text-overflow: ellipsis;
         white-space: nowrap; }
.spacer { background: #fafafa; border-top: none; border-bottom: none; width: 8px; }
"""


def fmt_pct(base: float, target: float) -> str:
    if base <= 0:
        return ""
    return f"{(target-base)/base*100:+.1f}%"


def render_summary_table(merged_layers: List[str],
                         a_by_layer: Dict[str, List[Row]],
                         b_by_layer: Dict[str, List[Row]],
                         a_total: float, b_total: float,
                         label_a: str, label_b: str) -> str:
    parts = ['<table>',
             f'<tr><th class="l">Layer group</th>'
             f'<th>{escape(label_a)} (us)</th>'
             f'<th>{escape(label_b)} (us)</th>'
             f'<th>Δ (us)</th><th>Δ %</th></tr>']
    for L in merged_layers:
        at = sum(r.duration_us for r in a_by_layer.get(L, []))
        bt = sum(r.duration_us for r in b_by_layer.get(L, []))
        d = bt - at
        cls = "slower" if d > 0.5 else ("faster" if d < -0.5 else "")
        parts.append(
            f'<tr class="{cls}"><td class="l">{escape(L)}</td>'
            f'<td>{at:.1f}</td><td>{bt:.1f}</td>'
            f'<td>{d:+.1f}</td><td>{fmt_pct(at, bt)}</td></tr>')
    td = b_total - a_total
    parts.append(
        f'<tr class="total"><td class="l">TOTAL</td>'
        f'<td>{a_total:.1f}</td><td>{b_total:.1f}</td>'
        f'<td>{td:+.1f}</td><td>{fmt_pct(a_total, b_total)}</td></tr>')
    parts.append('</table>')
    return "".join(parts)


def render_detail_table(merged_layers: List[str],
                        a_by_layer: Dict[str, List[Row]],
                        b_by_layer: Dict[str, List[Row]],
                        label_a: str, label_b: str) -> str:
    parts = ['<table>',
             f'<tr><th class="l">Layer</th>'
             f'<th class="l">{escape(label_a)}: Module</th>'
             f'<th class="l">{escape(label_a)}: Kernel</th>'
             f'<th>t (us)</th><th>%</th>'
             f'<th class="spacer"></th>'
             f'<th class="l">{escape(label_b)}: Module</th>'
             f'<th class="l">{escape(label_b)}: Kernel</th>'
             f'<th>t (us)</th><th>%</th></tr>']
    for L in merged_layers:
        a_grp = a_by_layer.get(L, [])
        b_grp = b_by_layer.get(L, [])
        pairs = align_layer_block(a_grp, b_grp)
        first = True
        for (a, b) in pairs:
            cells = [
                f'<td class="l">{escape(L) if first else ""}</td>',
                f'<td class="l">{escape(a.module) if a else ""}</td>',
                f'<td class="l kname">{escape(a.kernel_name) if a else ""}</td>',
                f'<td>{a.duration_us:.2f}</td>' if a else '<td></td>',
                f'<td>{escape(a.pct) if a else ""}</td>',
                '<td class="spacer"></td>',
                f'<td class="l">{escape(b.module) if b else ""}</td>',
                f'<td class="l kname">{escape(b.kernel_name) if b else ""}</td>',
                f'<td>{b.duration_us:.2f}</td>' if b else '<td></td>',
                f'<td>{escape(b.pct) if b else ""}</td>',
            ]
            cls = "layer-row" if first else ""
            parts.append(f'<tr class="{cls}">{"".join(cells)}</tr>')
            first = False
        # Subtotal
        at = sum(r.duration_us for r in a_grp)
        bt = sum(r.duration_us for r in b_grp)
        parts.append(
            '<tr class="subtotal">'
            '<td class="l"></td><td></td><td class="l">subtotal</td>'
            f'<td>{at:.2f}</td><td></td>'
            '<td class="spacer"></td><td></td><td class="l">subtotal</td>'
            f'<td>{bt:.2f}</td><td></td></tr>')
    parts.append('</table>')
    return "".join(parts)


def build_html(a_path: str, b_path: str, output: str,
               label_a: str, label_b: str, title: str):
    a_rows, a_total = load_merged(a_path)
    b_rows, b_total = load_merged(b_path)
    a_by_layer = defaultdict(list)
    for r in a_rows: a_by_layer[r.layer].append(r)
    b_by_layer = defaultdict(list)
    for r in b_rows: b_by_layer[r.layer].append(r)

    CANONICAL = ["prepare_attn", "DeepseekV2AttentionMLA", "prepare_mlp",
                 "DeepseekV2MoE", "DeepseekV2MLP", "post_layer"]
    seen = set(a_by_layer) | set(b_by_layer)
    merged_layers = [L for L in CANONICAL if L in seen]
    for L in [r.layer for r in a_rows] + [r.layer for r in b_rows]:
        if L not in merged_layers and L in seen:
            merged_layers.append(L)

    a_cat = categorize_rows(a_rows)
    b_cat = categorize_rows(b_rows)
    all_cats = [c for c in CATEGORY_ORDER if c in a_cat or c in b_cat]
    for c in list(a_cat) + list(b_cat):
        if c not in all_cats:
            all_cats.append(c)
    a_vals = [a_cat.get(c, 0.0) for c in all_cats]
    b_vals = [b_cat.get(c, 0.0) for c in all_cats]

    chart_title = (f"Profiling DeepseekV2DecoderLayer in decode  —  "
                   f"{label_a} vs {label_b}")
    svg = svg_grouped_bars(all_cats, a_vals, b_vals,
                           label_a, label_b, chart_title)

    summary = render_summary_table(merged_layers, a_by_layer, b_by_layer,
                                   a_total, b_total, label_a, label_b)
    detail = render_detail_table(merged_layers, a_by_layer, b_by_layer,
                                 label_a, label_b)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{escape(title)}</title>
<style>{CSS}</style></head><body>
<h1>{escape(title)}</h1>
<p class="subtitle">{escape(label_a)} vs {escape(label_b)}  ·  one DECODE layer (median instance)</p>

<h2>Kernel time by category</h2>
{svg}

<h2>Layer-group totals</h2>
{summary}

<h2>Per-Layer side-by-side detail</h2>
{detail}

</body></html>"""
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="Merged xlsx for run A (baseline)")
    p.add_argument("b", help="Merged xlsx for run B (target)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--labels", nargs=2, default=("A", "B"))
    p.add_argument("--title", default="Profile compare report")
    args = p.parse_args()
    build_html(args.a, args.b, args.output, args.labels[0], args.labels[1],
               args.title)


if __name__ == "__main__":
    main()
