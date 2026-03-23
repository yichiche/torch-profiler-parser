#!/usr/bin/env python3
"""Interactive Module Tree Visualizer.

Reads an analysis.xlsx (from trace_module_analyzer.py) and generates
a self-contained interactive HTML visualization of the Module Tree.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import openpyxl


@dataclass
class TreeNode:
    id: int
    name: str
    module_type: str
    color_category: str
    depth: int
    time_us: float
    kernel_count: int
    breakdown: List[Tuple[str, float, float]]  # (category, time_us, pct)
    phase: Optional[str]
    children: List["TreeNode"] = field(default_factory=list)
    pct_of_parent: float = 100.0


# ---------------------------------------------------------------------------
# Module type -> color category mapping
# ---------------------------------------------------------------------------

MODULE_COLOR_MAP = {
    # Attention
    "Qwen3_5GatedDeltaNet": "attention",
    "Qwen3_5AttentionDecoderLayer": "attention",
    "RadixAttention": "attention",
    "RadixLinearAttention": "attention",
    "MRotaryEmbedding": "attention",
    # Linear
    "ColumnParallelLinear": "linear",
    "RowParallelLinear": "linear",
    "MergedColumnParallelLinear": "linear",
    "QKVParallelLinear": "linear",
    "ReplicatedLinear": "linear",
    "Linear": "linear",
    # MoE
    "Qwen2MoeSparseMoeBlock": "moe",
    "FlashInferFusedMoE": "moe",
    "FusedMoE": "moe",
    "Qwen2MoeMLP": "moe",
    "TopK": "moe",
    # Normalization
    "GemmaRMSNorm": "normalization",
    "RMSNorm": "normalization",
    # Embedding
    "VocabParallelEmbedding": "embedding",
    # Decoder layer
    "Qwen3_5LinearDecoderLayer": "decoder-layer",
    # Model root
    "Qwen3_5MoeForCausalLM": "model-root",
    # Infrastructure
    "CudaGraphReplay": "infrastructure",
    "LogitsProcessor": "infrastructure",
    "Sampler": "infrastructure",
}


def classify_module(module_type: str) -> str:
    if module_type in MODULE_COLOR_MAP:
        return MODULE_COLOR_MAP[module_type]
    lower = module_type.lower()
    if "attention" in lower or "attn" in lower:
        return "attention"
    if "norm" in lower:
        return "normalization"
    if "moe" in lower or "expert" in lower:
        return "moe"
    if "mlp" in lower or "ffn" in lower:
        return "moe"  # MLP inside MoE is common
    if "linear" in lower:
        return "linear"
    if "embed" in lower:
        return "embedding"
    if "decoder" in lower or "layer" in lower:
        return "decoder-layer"
    return "default"


def extract_module_type(name: str) -> str:
    m = re.match(r"^(.+?)_(\d+)$", name)
    return m.group(1) if m else name


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_BREAKDOWN_RE = re.compile(r"(\w+):\s+([\d,]+(?:\.\d+)?)\s+us\s+\(([\d.]+)%\)")


def parse_breakdown(raw: Optional[str]) -> List[Tuple[str, float, float]]:
    if not raw:
        return []
    result = []
    for m in _BREAKDOWN_RE.finditer(raw):
        cat = m.group(1)
        time_us = float(m.group(2).replace(",", ""))
        pct = float(m.group(3))
        result.append((cat, time_us, pct))
    return result


def parse_module_entry(raw: str) -> Tuple[int, str]:
    stripped = raw.lstrip()
    leading = len(raw) - len(stripped)
    depth = leading // 4
    name = stripped
    for prefix in ["├── ", "└── "]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return depth, name.strip()


# ---------------------------------------------------------------------------
# Reading Excel
# ---------------------------------------------------------------------------


def read_module_tree(xlsx_path: str) -> List[dict]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Module Tree"]
    rows = []
    first = True
    for row in ws.iter_rows(values_only=True):
        if first:
            first = False
            continue  # skip header
        module_raw = row[0]
        if module_raw is None:
            continue
        if "truncated" in str(module_raw).lower():
            continue
        time_us = row[1] if row[1] is not None else 0.0
        kernel_count = row[2] if row[2] is not None else 0
        breakdown_raw = row[3]
        phase = row[4] if len(row) > 4 else None
        if phase and not phase.strip():
            phase = None
        rows.append(
            {
                "module_raw": str(module_raw),
                "time_us": float(time_us),
                "kernel_count": int(kernel_count),
                "breakdown_raw": breakdown_raw,
                "phase": phase,
            }
        )
    wb.close()
    return rows


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------

_node_counter = 0


def _make_node(depth: int, name: str, time_us: float, kernel_count: int,
               breakdown_raw: Optional[str], phase: Optional[str]) -> TreeNode:
    global _node_counter
    module_type = extract_module_type(name)
    node = TreeNode(
        id=_node_counter,
        name=name,
        module_type=module_type,
        color_category=classify_module(module_type),
        depth=depth,
        time_us=time_us,
        kernel_count=kernel_count,
        breakdown=parse_breakdown(breakdown_raw),
        phase=phase,
    )
    _node_counter += 1
    return node


def build_tree(rows: List[dict]) -> List[TreeNode]:
    global _node_counter
    _node_counter = 0
    roots: List[TreeNode] = []
    stack: List[Tuple[int, TreeNode]] = []

    for row in rows:
        depth, name = parse_module_entry(row["module_raw"])
        node = _make_node(depth, name, row["time_us"], row["kernel_count"],
                          row["breakdown_raw"], row["phase"])

        while stack and stack[-1][0] >= depth:
            stack.pop()

        if stack:
            stack[-1][1].children.append(node)
        else:
            roots.append(node)

        stack.append((depth, node))

    return roots


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(roots: List[TreeNode]) -> float:
    """Compute pct_of_parent for all nodes. Returns global max_time_us."""
    max_time = 0.0

    def walk(node: TreeNode, parent_time: float):
        nonlocal max_time
        if node.time_us > max_time:
            max_time = node.time_us
        if parent_time > 0:
            node.pct_of_parent = round(node.time_us / parent_time * 100, 1)
        else:
            node.pct_of_parent = 0.0
        for child in node.children:
            walk(child, node.time_us)

    for root in roots:
        root.pct_of_parent = 100.0
        if root.time_us > max_time:
            max_time = root.time_us
        for child in root.children:
            walk(child, root.time_us)

    return max_time


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def nodes_to_json(roots: List[TreeNode]) -> List[dict]:
    def serialize(node: TreeNode) -> dict:
        return {
            "id": node.id,
            "name": node.name,
            "moduleType": node.module_type,
            "colorCategory": node.color_category,
            "depth": node.depth,
            "timeUs": node.time_us,
            "kernelCount": node.kernel_count,
            "breakdown": [
                {"cat": cat, "timeUs": t, "pct": p} for cat, t, p in node.breakdown
            ],
            "phase": node.phase,
            "pctOfParent": node.pct_of_parent,
            "children": [serialize(c) for c in node.children],
        }

    return [serialize(r) for r in roots]


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------


def compute_summary(roots: List[TreeNode]) -> dict:
    total_time = sum(r.time_us for r in roots)
    total_kernels = sum(r.kernel_count for r in roots)
    module_count = 0
    prefill_time = 0.0
    decode_time = 0.0

    def walk(node):
        nonlocal module_count, prefill_time, decode_time
        module_count += 1
        # Only count time for nodes that are roots (to avoid double counting)
        for child in node.children:
            walk(child)

    for root in roots:
        walk(root)
        if root.phase == "prefill":
            prefill_time += root.time_us
        elif root.phase == "decode":
            decode_time += root.time_us

    # For phase time, also check children of roots without phase
    def collect_phase_times(node):
        nonlocal prefill_time, decode_time
        if node.phase == "prefill" and node.depth > 0:
            pass  # already counted via parent
        elif node.phase == "decode" and node.depth > 0:
            pass
        for child in node.children:
            collect_phase_times(child)

    return {
        "totalTime": round(total_time, 1),
        "prefillTime": round(prefill_time, 1),
        "decodeTime": round(decode_time, 1),
        "totalKernels": total_kernels,
        "moduleCount": module_count,
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------


def generate_html(roots: List[TreeNode], max_time_us: float, basename: str) -> str:
    tree_json = json.dumps(nodes_to_json(roots), separators=(",", ":"))
    summary = json.dumps(compute_summary(roots), separators=(",", ":"))

    return HTML_TEMPLATE.replace("__TREE_DATA__", tree_json).replace(
        "__MAX_TIME__", str(max_time_us)
    ).replace("__BASENAME__", basename).replace("__SUMMARY__", summary)


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Module Tree - __BASENAME__</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg-primary:#1a1a2e;
  --bg-secondary:#16213e;
  --bg-toolbar:#0f3460;
  --bg-row-hover:#1e2a4a;
  --text-primary:#e0e0e0;
  --text-secondary:#a0a0b0;
  --text-dim:#666;
  --border-color:#2a2a4a;
  --cat-attention:#f0a500;
  --cat-linear:#06d6a0;
  --cat-moe:#9b5de5;
  --cat-normalization:#4ea8de;
  --cat-embedding:#00c9db;
  --cat-decoder-layer:#64748b;
  --cat-model-root:#e0e0e0;
  --cat-infrastructure:#6c757d;
  --cat-default:#888;
  --bd-communication:#e76f51;
  --bd-quantization:#e9c46a;
  --bd-attention:#f0a500;
  --bd-gemm:#76c893;
  --bd-normalization:#4ea8de;
  --bd-elementwise:#b8b8d0;
  --bd-embedding:#00c9db;
  --bd-moe:#9b5de5;
  --bd-other:#6c757d;
}
body{
  background:var(--bg-primary);color:var(--text-primary);
  font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
  font-size:13px;line-height:1.4;
}
.app{display:flex;flex-direction:column;height:100vh}
/* Toolbar */
.toolbar{
  background:var(--bg-toolbar);padding:12px 16px;
  display:flex;flex-wrap:wrap;align-items:center;gap:12px;
  border-bottom:1px solid var(--border-color);
}
.toolbar h1{font-size:15px;font-weight:600;margin-right:16px;white-space:nowrap}
.toolbar button{
  background:rgba(255,255,255,0.1);color:var(--text-primary);
  border:1px solid var(--border-color);border-radius:6px;
  padding:4px 10px;cursor:pointer;font-size:12px;transition:background .15s;
}
.toolbar button:hover{background:rgba(255,255,255,0.2)}
.toolbar button.active{background:rgba(78,168,222,0.3);border-color:#4ea8de}
.toolbar input[type=text]{
  background:rgba(255,255,255,0.08);color:var(--text-primary);
  border:1px solid var(--border-color);border-radius:6px;
  padding:4px 10px;font-size:12px;width:180px;outline:none;
}
.toolbar input[type=text]:focus{border-color:#4ea8de}
.toolbar select{
  background:rgba(255,255,255,0.08);color:var(--text-primary);
  border:1px solid var(--border-color);border-radius:6px;
  padding:4px 8px;font-size:12px;outline:none;
}
.toolbar label{font-size:12px;display:flex;align-items:center;gap:4px;cursor:pointer}
.toolbar .sep{width:1px;height:20px;background:var(--border-color)}
.ctrl-group{display:flex;align-items:center;gap:6px}
.ctrl-label{font-size:11px;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.5px}
/* Summary */
.summary-bar{
  background:var(--bg-secondary);padding:8px 16px;
  font-size:12px;color:var(--text-secondary);
  border-bottom:1px solid var(--border-color);
  display:flex;gap:24px;
}
.summary-bar span{display:flex;align-items:center;gap:4px}
.summary-bar .val{color:var(--text-primary);font-weight:600}
/* Column headers */
.col-headers{
  display:grid;
  grid-template-columns:1fr 100px 80px 70px 220px;
  padding:6px 16px 6px 40px;
  background:var(--bg-secondary);
  border-bottom:2px solid var(--border-color);
  font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:0.5px;color:var(--text-secondary);
}
.col-headers span:nth-child(n+2){text-align:right}
.col-headers span:last-child{text-align:left;padding-left:8px}
/* Tree container */
.tree-container{flex:1;overflow-y:auto;overflow-x:hidden}
/* Tree row */
.tree-row{
  position:relative;display:grid;
  grid-template-columns:1fr 100px 80px 70px 220px;
  align-items:center;min-height:32px;
  padding:0 16px;
  border-bottom:1px solid rgba(42,42,74,0.5);
  cursor:pointer;transition:background .12s;
}
.tree-row:hover{background:var(--bg-row-hover)}
/* Time bar background */
.time-bar{
  position:absolute;left:0;top:0;bottom:0;
  opacity:0.10;border-radius:0 4px 4px 0;
  pointer-events:none;z-index:0;min-width:2px;
}
/* Module cell */
.cell-module{
  display:flex;align-items:center;gap:6px;
  overflow:hidden;position:relative;z-index:1;
}
.toggle{
  width:16px;height:16px;display:inline-flex;align-items:center;
  justify-content:center;font-size:10px;flex-shrink:0;
  color:var(--text-secondary);transition:transform .15s;user-select:none;
}
.toggle.expanded{transform:rotate(90deg)}
.toggle.leaf{visibility:hidden}
.color-dot{
  width:8px;height:8px;border-radius:50%;flex-shrink:0;
}
.module-name{
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  font-size:13px;
}
.phase-badge{
  font-size:9px;padding:1px 6px;border-radius:8px;
  font-weight:600;text-transform:uppercase;flex-shrink:0;
  margin-left:4px;
}
.phase-prefill{background:rgba(46,196,182,0.25);color:#2ec4b6}
.phase-decode{background:rgba(240,165,0,0.25);color:#f0a500}
/* Metric cells */
.cell-time,.cell-pct,.cell-kernels{
  text-align:right;font-variant-numeric:tabular-nums;
  position:relative;z-index:1;font-size:12px;
}
.cell-pct{color:var(--text-secondary)}
/* Breakdown mini-bar */
.cell-breakdown{position:relative;z-index:1;padding-left:8px}
.mini-bar{
  display:flex;height:14px;border-radius:3px;overflow:hidden;
  width:200px;background:rgba(255,255,255,0.04);
}
.mini-seg{height:100%;min-width:1px}
/* Tooltip */
.tooltip{
  display:none;position:fixed;z-index:1000;
  background:#0d1b2a;border:1px solid var(--border-color);
  border-radius:8px;padding:12px;max-width:350px;
  font-size:12px;pointer-events:none;
  box-shadow:0 8px 32px rgba(0,0,0,0.5);
}
.tooltip .tt-name{font-weight:600;font-size:13px;margin-bottom:4px}
.tooltip .tt-phase{font-size:11px;margin-bottom:4px}
.tooltip .tt-metrics{margin-bottom:8px}
.tooltip .tt-sep{border-top:1px solid var(--border-color);margin:6px 0}
.tooltip .tt-bd-row{display:flex;align-items:center;gap:6px;margin:2px 0}
.tooltip .tt-bd-dot{width:8px;height:8px;border-radius:2px;flex-shrink:0}
.tooltip .tt-bd-cat{flex:1}
.tooltip .tt-bd-val{color:var(--text-secondary);text-align:right}
/* Legend */
.legend{
  background:var(--bg-secondary);padding:8px 16px;
  border-top:1px solid var(--border-color);
  display:flex;flex-wrap:wrap;gap:12px;font-size:11px;
}
.legend-item{display:flex;align-items:center;gap:4px}
.legend-dot{width:8px;height:8px;border-radius:50%}
/* No results */
.no-results{padding:40px;text-align:center;color:var(--text-secondary);font-size:14px}
</style>
</head>
<body>
<div class="app">
  <div class="toolbar">
    <h1>Module Tree: __BASENAME__</h1>
    <div class="sep"></div>
    <div class="ctrl-group">
      <button id="btn-expand-all">Expand All</button>
      <button id="btn-collapse-all">Collapse All</button>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group">
      <span class="ctrl-label">Depth</span>
      <button class="depth-btn" data-depth="0">0</button>
      <button class="depth-btn" data-depth="1">1</button>
      <button class="depth-btn" data-depth="2">2</button>
      <button class="depth-btn" data-depth="3">3</button>
      <button class="depth-btn" data-depth="4">4</button>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group">
      <span class="ctrl-label">Search</span>
      <input type="text" id="search-input" placeholder="Filter modules...">
    </div>
    <div class="sep"></div>
    <div class="ctrl-group">
      <span class="ctrl-label">Sort</span>
      <select id="sort-select">
        <option value="original">Original</option>
        <option value="time-desc">Time ↓</option>
        <option value="time-asc">Time ↑</option>
        <option value="kernels-desc">Kernels ↓</option>
      </select>
    </div>
    <div class="sep"></div>
    <div class="ctrl-group">
      <label><input type="checkbox" class="phase-cb" value="prefill" checked> Prefill</label>
      <label><input type="checkbox" class="phase-cb" value="decode" checked> Decode</label>
      <label><input type="checkbox" class="phase-cb" value="" checked> Other</label>
    </div>
  </div>
  <div class="summary-bar" id="summary-bar"></div>
  <div class="col-headers">
    <span>Module</span>
    <span>Time (μs)</span>
    <span>% Parent</span>
    <span>Kernels</span>
    <span style="padding-left:8px">Breakdown</span>
  </div>
  <div class="tree-container" id="tree-container"></div>
  <div class="legend" id="legend"></div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
const TREE_DATA = __TREE_DATA__;
const MAX_TIME = __MAX_TIME__;
const SUMMARY = __SUMMARY__;

// Category colors (module type)
const CAT_COLORS = {
  attention:'#f0a500',linear:'#06d6a0',moe:'#9b5de5',
  normalization:'#4ea8de',embedding:'#00c9db','decoder-layer':'#64748b',
  'model-root':'#e0e0e0',infrastructure:'#6c757d','default':'#888'
};

// Breakdown colors (kernel operation categories)
const BD_COLORS = {
  communication:'#e76f51',quantization:'#e9c46a',attention:'#f0a500',
  gemm:'#76c893',normalization:'#4ea8de',elementwise:'#b8b8d0',
  embedding:'#00c9db',moe:'#9b5de5',other:'#6c757d'
};

// State
let expandedNodes = new Set();
let allNodes = [];    // flat DFS order
let nodeMap = {};     // id -> node
let sortMode = 'original';
let searchFilter = '';
let phaseFilter = {prefill:true,decode:true,'':true};
let activeDepth = -1;

// Flatten tree to DFS-ordered list
function flattenTree(roots, parentId) {
  const sorted = sortChildren(roots);
  for (const node of sorted) {
    node.parentId = parentId;
    allNodes.push(node);
    nodeMap[node.id] = node;
    if (node.children && node.children.length > 0) {
      flattenTree(node.children, node.id);
    }
  }
}

function sortChildren(children) {
  if (sortMode === 'original') return children;
  const s = [...children];
  if (sortMode === 'time-desc') s.sort((a,b) => b.timeUs - a.timeUs);
  if (sortMode === 'time-asc') s.sort((a,b) => a.timeUs - b.timeUs);
  if (sortMode === 'kernels-desc') s.sort((a,b) => b.kernelCount - a.kernelCount);
  return s;
}

function rebuildFlat() {
  allNodes = [];
  nodeMap = {};
  flattenTree(TREE_DATA, null);
}

function isVisible(node) {
  let cur = node;
  while (cur.parentId !== null && cur.parentId !== undefined) {
    if (!expandedNodes.has(cur.parentId)) return false;
    cur = nodeMap[cur.parentId];
  }
  return true;
}

// Search: when active, show matches + ancestors
function getVisibleSet() {
  if (!searchFilter) return null; // null means no filter
  const matchIds = new Set();
  const visibleIds = new Set();
  // Mark matches
  for (const n of allNodes) {
    if (n.name.toLowerCase().includes(searchFilter)) {
      matchIds.add(n.id);
      visibleIds.add(n.id);
      // Walk up ancestors
      let cur = n;
      while (cur.parentId !== null && cur.parentId !== undefined) {
        visibleIds.add(cur.parentId);
        expandedNodes.add(cur.parentId);
        cur = nodeMap[cur.parentId];
      }
    }
  }
  return visibleIds;
}

function matchesPhase(node) {
  const p = node.phase || '';
  return phaseFilter[p] !== false;
}

function fmtNum(n) {
  if (n === null || n === undefined) return '-';
  return n.toLocaleString('en-US', {maximumFractionDigits:1});
}

function getCatColor(cat) { return CAT_COLORS[cat] || CAT_COLORS['default']; }
function getBdColor(cat) { return BD_COLORS[cat] || BD_COLORS['other']; }

function createRow(node) {
  const row = document.createElement('div');
  row.className = 'tree-row';
  row.dataset.id = node.id;

  // Time bar
  const barW = MAX_TIME > 0 ? (node.timeUs / MAX_TIME) * 100 : 0;
  const bar = document.createElement('div');
  bar.className = 'time-bar';
  bar.style.width = Math.max(barW, node.timeUs > 0 ? 0.15 : 0) + '%';
  bar.style.backgroundColor = getCatColor(node.colorCategory);
  row.appendChild(bar);

  // Module cell
  const modCell = document.createElement('div');
  modCell.className = 'cell-module';
  modCell.style.paddingLeft = (node.depth * 24) + 'px';

  const toggle = document.createElement('span');
  const hasChildren = node.children && node.children.length > 0;
  toggle.className = 'toggle' + (hasChildren ? (expandedNodes.has(node.id) ? ' expanded' : '') : ' leaf');
  toggle.textContent = '▶';
  toggle.dataset.id = node.id;
  modCell.appendChild(toggle);

  const dot = document.createElement('span');
  dot.className = 'color-dot';
  dot.style.backgroundColor = getCatColor(node.colorCategory);
  modCell.appendChild(dot);

  const nameSpan = document.createElement('span');
  nameSpan.className = 'module-name';
  nameSpan.textContent = node.name;
  modCell.appendChild(nameSpan);

  if (node.phase) {
    const badge = document.createElement('span');
    badge.className = 'phase-badge phase-' + node.phase;
    badge.textContent = node.phase;
    modCell.appendChild(badge);
  }
  row.appendChild(modCell);

  // Time
  const timeCell = document.createElement('div');
  timeCell.className = 'cell-time';
  timeCell.textContent = fmtNum(node.timeUs);
  row.appendChild(timeCell);

  // Pct
  const pctCell = document.createElement('div');
  pctCell.className = 'cell-pct';
  pctCell.textContent = node.pctOfParent < 100 ? node.pctOfParent.toFixed(1) + '%' : '';
  row.appendChild(pctCell);

  // Kernels
  const kernelCell = document.createElement('div');
  kernelCell.className = 'cell-kernels';
  kernelCell.textContent = node.kernelCount;
  row.appendChild(kernelCell);

  // Breakdown mini-bar
  const bdCell = document.createElement('div');
  bdCell.className = 'cell-breakdown';
  if (node.breakdown && node.breakdown.length > 0) {
    const miniBar = document.createElement('div');
    miniBar.className = 'mini-bar';
    for (const seg of node.breakdown) {
      const s = document.createElement('div');
      s.className = 'mini-seg';
      s.style.width = seg.pct + '%';
      s.style.backgroundColor = getBdColor(seg.cat);
      s.title = seg.cat + ': ' + fmtNum(seg.timeUs) + ' us (' + seg.pct + '%)';
      miniBar.appendChild(s);
    }
    bdCell.appendChild(miniBar);
  }
  row.appendChild(bdCell);

  return row;
}

function render() {
  const container = document.getElementById('tree-container');
  const searchSet = getVisibleSet();
  const frag = document.createDocumentFragment();
  let count = 0;

  for (const node of allNodes) {
    if (!matchesPhase(node)) continue;
    if (searchSet !== null) {
      if (!searchSet.has(node.id)) continue;
    } else {
      if (!isVisible(node)) continue;
    }
    frag.appendChild(createRow(node));
    count++;
  }

  container.innerHTML = '';
  if (count === 0) {
    const nr = document.createElement('div');
    nr.className = 'no-results';
    nr.textContent = 'No modules match the current filters.';
    container.appendChild(nr);
  } else {
    container.appendChild(frag);
  }
}

function toggleNode(id) {
  if (expandedNodes.has(id)) {
    expandedNodes.delete(id);
    collapseDescendants(id);
  } else {
    expandedNodes.add(id);
  }
  activeDepth = -1;
  updateDepthButtons();
  render();
}

function collapseDescendants(id) {
  const node = nodeMap[id];
  if (!node || !node.children) return;
  for (const c of node.children) {
    expandedNodes.delete(c.id);
    collapseDescendants(c.id);
  }
}

function expandToDepth(d) {
  expandedNodes.clear();
  for (const n of allNodes) {
    if (n.depth < d && n.children && n.children.length > 0) {
      expandedNodes.add(n.id);
    }
  }
  activeDepth = d;
  updateDepthButtons();
  render();
}

function updateDepthButtons() {
  document.querySelectorAll('.depth-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.depth) === activeDepth);
  });
}

// Tooltip
const tooltip = document.getElementById('tooltip');

function showTooltip(e, node) {
  let html = '<div class="tt-name">' + esc(node.name) + '</div>';
  if (node.phase) {
    html += '<div class="tt-phase">Phase: <strong>' + node.phase + '</strong></div>';
  }
  html += '<div class="tt-metrics">Time: <strong>' + fmtNum(node.timeUs) + ' μs</strong>';
  if (node.pctOfParent < 100) html += ' (' + node.pctOfParent.toFixed(1) + '% of parent)';
  html += '<br>Kernels: <strong>' + node.kernelCount + '</strong></div>';

  if (node.breakdown && node.breakdown.length > 0) {
    html += '<div class="tt-sep"></div>';
    for (const seg of node.breakdown) {
      html += '<div class="tt-bd-row">' +
        '<span class="tt-bd-dot" style="background:' + getBdColor(seg.cat) + '"></span>' +
        '<span class="tt-bd-cat">' + seg.cat + '</span>' +
        '<span class="tt-bd-val">' + fmtNum(seg.timeUs) + ' μs (' + seg.pct + '%)</span>' +
        '</div>';
    }
  }
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  positionTooltip(e);
}

function positionTooltip(e) {
  const r = tooltip.getBoundingClientRect();
  let x = e.clientX + 16;
  let y = e.clientY - 8;
  if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 16;
  if (y + r.height > window.innerHeight - 8) y = window.innerHeight - r.height - 8;
  if (y < 8) y = 8;
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}

function hideTooltip() { tooltip.style.display = 'none'; }

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// Summary
function renderSummary() {
  const sb = document.getElementById('summary-bar');
  sb.innerHTML =
    '<span>Total: <span class="val">' + fmtNum(SUMMARY.totalTime) + ' μs</span></span>' +
    '<span>Modules: <span class="val">' + SUMMARY.moduleCount + '</span></span>' +
    '<span>Kernels: <span class="val">' + fmtNum(SUMMARY.totalKernels) + '</span></span>';
}

// Legend
function renderLegend() {
  const legend = document.getElementById('legend');
  const cats = [
    ['attention','Attention'],['linear','Linear'],['moe','MoE'],
    ['normalization','Norm'],['embedding','Embedding'],
    ['decoder-layer','Decoder Layer'],['model-root','Model Root'],
    ['infrastructure','Infrastructure']
  ];
  legend.innerHTML = cats.map(([k,v]) =>
    '<span class="legend-item"><span class="legend-dot" style="background:' +
    getCatColor(k) + '"></span>' + v + '</span>'
  ).join('');

  // Breakdown colors
  legend.innerHTML += '<span style="margin-left:16px;color:var(--text-dim)">|</span>';
  const bds = [
    ['communication','Comm'],['gemm','GEMM'],['attention','Attn'],
    ['moe','MoE'],['quantization','Quant'],['normalization','Norm'],
    ['elementwise','Elem'],['embedding','Embed'],['other','Other']
  ];
  legend.innerHTML += bds.map(([k,v]) =>
    '<span class="legend-item"><span class="legend-dot" style="background:' +
    getBdColor(k) + ';border-radius:2px"></span>' + v + '</span>'
  ).join('');
}

// Debounce
function debounce(fn, ms) {
  let t;
  return function(...args) { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// Init
function init() {
  rebuildFlat();
  // Default: expand depth 0
  for (const n of allNodes) {
    if (n.depth === 0 && n.children && n.children.length > 0) {
      expandedNodes.add(n.id);
    }
  }
  activeDepth = 1;
  updateDepthButtons();
  renderSummary();
  renderLegend();
  render();

  // Events
  document.getElementById('tree-container').addEventListener('click', e => {
    const row = e.target.closest('.tree-row');
    if (!row) return;
    const id = parseInt(row.dataset.id);
    const node = nodeMap[id];
    if (node && node.children && node.children.length > 0) {
      toggleNode(id);
    }
  });

  document.getElementById('tree-container').addEventListener('mouseover', e => {
    const row = e.target.closest('.tree-row');
    if (row) {
      const id = parseInt(row.dataset.id);
      showTooltip(e, nodeMap[id]);
    }
  });

  document.getElementById('tree-container').addEventListener('mousemove', e => {
    if (tooltip.style.display === 'block') positionTooltip(e);
  });

  document.getElementById('tree-container').addEventListener('mouseout', e => {
    const row = e.target.closest('.tree-row');
    if (row && !row.contains(e.relatedTarget)) hideTooltip();
  });

  document.getElementById('btn-expand-all').addEventListener('click', () => {
    for (const n of allNodes) {
      if (n.children && n.children.length > 0) expandedNodes.add(n.id);
    }
    activeDepth = -1; updateDepthButtons(); render();
  });

  document.getElementById('btn-collapse-all').addEventListener('click', () => {
    expandedNodes.clear();
    activeDepth = 0; updateDepthButtons(); render();
  });

  document.querySelectorAll('.depth-btn').forEach(btn => {
    btn.addEventListener('click', () => expandToDepth(parseInt(btn.dataset.depth)));
  });

  const debouncedSearch = debounce(() => {
    searchFilter = document.getElementById('search-input').value.toLowerCase().trim();
    render();
  }, 300);
  document.getElementById('search-input').addEventListener('input', debouncedSearch);

  document.getElementById('sort-select').addEventListener('change', e => {
    sortMode = e.target.value;
    rebuildFlat();
    render();
  });

  document.querySelectorAll('.phase-cb').forEach(cb => {
    cb.addEventListener('change', () => {
      phaseFilter[cb.value] = cb.checked;
      render();
    });
  });
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def serve_html(html_path: str, port: int):
    """Start a FastAPI/uvicorn server to serve the generated HTML."""
    import socket

    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        import uvicorn
    except ImportError:
        print("\n  Install fastapi and uvicorn to use --serve:")
        print("    pip install fastapi uvicorn")
        print("\n  Falling back to built-in http.server...\n")
        _serve_html_fallback(html_path, port)
        return

    html_content = open(html_path, "r", encoding="utf-8").read()
    app = FastAPI(title="Module Tree Viewer")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(content=html_content)

    hostname = socket.gethostname()
    print(f"\n  Serving at:")
    print(f"    Local:  http://localhost:{port}")
    print(f"    Remote: http://{hostname}:{port}")
    print(f"\n  Press Ctrl+C to stop.\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def _serve_html_fallback(html_path: str, port: int):
    """Fallback server using stdlib http.server."""
    import http.server
    import socket

    directory = os.path.dirname(os.path.abspath(html_path))
    filename = os.path.basename(html_path)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(302)
                self.send_header("Location", f"/{filename}")
                self.end_headers()
                return
            super().do_GET()

        def log_message(self, format, *args):
            pass

    hostname = socket.gethostname()
    print(f"\n  Serving at:")
    print(f"    Local:  http://localhost:{port}")
    print(f"    Remote: http://{hostname}:{port}")
    print(f"\n  Press Ctrl+C to stop.\n")

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


def main(xlsx_path: str, output_path: str, serve: bool = False, port: int = 8765):
    print(f"Reading {xlsx_path}...")
    rows = read_module_tree(xlsx_path)
    print(f"  {len(rows)} rows read from 'Module Tree' sheet")

    roots = build_tree(rows)
    print(f"  {len(roots)} root modules")

    max_time = compute_metrics(roots)
    print(f"  Max time: {max_time:.1f} us")

    basename = os.path.basename(xlsx_path)
    html = generate_html(roots, max_time, basename)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Output written to {output_path}")

    if serve:
        serve_html(output_path, port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interactive module tree visualization"
    )
    parser.add_argument("xlsx_path", help="Path to analysis.xlsx")
    parser.add_argument(
        "-o", "--output",
        help="Output HTML path (default: <input_dir>/module_tree.html)",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start a local HTTP server and open in browser",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port for HTTP server (default: 8765)",
    )
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(args.xlsx_path) or ".", "module_tree.html"
    )
    main(args.xlsx_path, output, serve=args.serve, port=args.port)
