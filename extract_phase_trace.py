#!/usr/bin/env python3
"""Slice a PyTorch/roctracer trace down to selected eager forward passes.

Motivation
----------
For *analysis*, prefer ``trace_module_analyzer.py --phase-index extend`` -- it
selects the same passes in one command with no intermediate file.  Reach for
this tool when you need a smaller trace *file*: the full capture is too big to
open in Perfetto, or you want to hand a colleague just the interesting passes.

It cuts the trace itself:

1. Find every ``nn.Module: <root-module>_0`` span (one per eager forward pass).
2. Label each pass by which GPU attention kernel it launched
   (e.g. ``fmha_fwd_hd256...`` = plain prefill chunk,
   ``FmhaBatchPrefillWithPagedKVCache`` = extend / prefix-cached chunk).
3. Keep only the passes you ask for, together with:
     - every CPU event inside those passes' CPU spans,
     - every GPU event whose ``correlation`` was launched inside them
       (GPU work lags the CPU span, so a plain time cut would lose it),
     - the matching ``ac2g`` flow events and all metadata events.

The result is a normal trace.json.gz that `trace_module_analyzer.py` can chew
on directly, giving an apples-to-apples before/after comparison.

Examples
--------
  # what passes are in here?
  python3 extract_phase_trace.py in.trace.json.gz --list

  # keep only the paged-KV (extend) passes
  python3 extract_phase_trace.py in.trace.json.gz -o extend.trace.json.gz \
      --select-kernel FmhaBatchPrefillWithPagedKVCache

  # keep specific pass indices (as printed by --list)
  python3 extract_phase_trace.py in.trace.json.gz -o extend.trace.json.gz \
      --select-pass 1 2 3 4 --max-passes 4
"""

import argparse
import gzip
import os
import re
import sys
import time

TS_RE = re.compile(r'"ts": ([0-9.eE+-]+)')
CAT_RE = re.compile(r'"cat": "([^"]*)"')
PH_RE = re.compile(r'"ph": "([^"]*)"')
CORR_RE = re.compile(r'"correlation": (\d+)')
FLOWID_RE = re.compile(r'"id": (\d+)')

# cats that live on the GPU timeline -- they are kept via `correlation`, not ts
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def iter_events(path, progress_every=0):
    """Yield (blob_text, is_event) for the traceEvents array.

    The file is pretty-printed with one event object per ``  { ... },`` block,
    so a 2-space-indented closing brace reliably terminates an event.
    """
    started = False
    buf = []
    n = 0
    t0 = time.time()
    with _open(path) as f:
        for line in f:
            if not started:
                yield line, False
                if '"traceEvents": [' in line:
                    started = True
                continue
            if line[:3] == "  ]":
                # tail of the file
                yield line, False
                for rest in f:
                    yield rest, False
                return
            buf.append(line)
            if line[:3] == "  }":
                n += 1
                if progress_every and n % progress_every == 0:
                    print(f"    ...{n:,} events ({time.time()-t0:.0f}s)",
                          file=sys.stderr, flush=True)
                yield "".join(buf), True
                buf = []


def parse_blob(blob):
    ph = PH_RE.search(blob)
    cat = CAT_RE.search(blob)
    ts = TS_RE.search(blob)
    return (ph.group(1) if ph else None,
            cat.group(1) if cat else None,
            float(ts.group(1)) if ts else None)


TS_DUR_RE = re.compile(r'"ts": ([0-9.]+), "dur": ([0-9.]+)')


def scan_passes(path, root_modules, kernel_pats):
    """Line-scan 1: forward-pass spans + the correlation ids of tagged kernels.

    The event objects are pretty-printed as
        {  / "ph"..."name"...  / "ts"..."dur"...  / "args": {  / <args>  / }  / },
    so ``ts``/``dur`` sit one line after the name and the args (carrying
    ``correlation``) three lines after it.

    Returns ({module_name: [(ts, dur), ...]}, {correlation: tag}).
    """
    markers = {f'nn.Module: {m}_0': m for m in root_modules}
    spans = {m: [] for m in root_modules}
    kern_corr = {}          # correlation -> tag
    pending = None          # (kind, tag, countdown)
    with _open(path) as f:
        for line in f:
            if pending is not None:
                kind, tag, n = pending
                if n == 1:      # ts / dur line
                    m = TS_DUR_RE.search(line)
                    if m and kind == "M":
                        spans[tag].append((float(m.group(1)), float(m.group(2))))
                        pending = None
                        continue
                    pending = (kind, tag, 2)
                    continue
                if n < 3:       # "args": { line
                    pending = (kind, tag, n + 1)
                    continue
                m = CORR_RE.search(line)
                if m:
                    kern_corr[int(m.group(1))] = tag
                pending = None
                continue
            if '"name":' not in line:
                continue
            for marker, mod in markers.items():
                if marker in line:
                    pending = ("M", mod, 1)
                    break
            else:
                if '"cat": "kernel"' in line:
                    for tag, pat in kernel_pats:
                        if pat in line:
                            pending = ("K", tag, 1)
                            break
    for v in spans.values():
        v.sort()
    return spans, kern_corr


def build_windows(spans, root_modules, mode):
    """One list of (start, end) intervals per forward pass.

    The main model span does not cover the whole iteration: on an MTP/EAGLE
    run the CPU launches the target model, blocks on the sampler sync for
    hundreds of ms, and only then runs the draft head -- which shows up as a
    *separate* root span shortly before the next main span.  Widening the
    window all the way to the next main span would sweep in every decode CUDA
    graph replay that happened during the wait, so instead each pass gets the
    main span plus the auxiliary root spans that fall in the same gap.
    """
    main = spans[root_modules[0]]
    starts = [ts for ts, _ in main]
    bounds = starts[1:] + [float("inf")]
    per_pass = [[(ts, ts + dur)] for ts, dur in main]
    if mode == "span":
        return per_pass
    aux = sorted(s for m in root_modules[1:] for s in spans[m])
    for i, (lo, hi) in enumerate(zip(starts, bounds)):
        tail = [(a, a + d) for a, d in aux if lo <= a < hi and a + d > per_pass[i][0][1]]
        if tail:
            # everything from the first aux span to the next pass belongs to
            # this iteration (draft head, logits processor, sampler, ...)
            per_pass[i].append((min(t[0] for t in tail), min(hi, max(t[1] for t in tail))))
    return per_pass


def scan_launch_ts(path, kern_corr):
    """Line-scan 2: CPU launch timestamp for each correlation of interest."""
    launch = {}
    ts = None
    pending = 0
    with _open(path) as f:
        for line in f:
            if pending:
                pending -= 1
                if pending == 2:                    # "ts"/"dur" line
                    m = TS_DUR_RE.search(line)
                    ts = float(m.group(1)) if m else None
                elif pending == 0:                  # args line
                    c = CORR_RE.search(line)
                    if c and ts is not None:
                        cid = int(c.group(1))
                        if cid in kern_corr:
                            launch[cid] = ts
                continue
            if '"cat": "cuda_runtime"' in line or '"cat": "hip_runtime"' in line:
                pending = 3
    return launch


def label_passes(per_pass, kern_corr, launch):
    """Attach kernel tags by *CPU launch* time -- GPU execution lags by a
    whole iteration on long-context runs, so GPU ts would mislabel."""
    out = [{"intervals": iv, "ts": iv[0][0], "tags": {}} for iv in per_pass]
    for cid, tag in kern_corr.items():
        lts = launch.get(cid)
        if lts is None:
            continue
        for p in out:
            if any(a <= lts < b for a, b in p["intervals"]):
                p["tags"][tag] = p["tags"].get(tag, 0) + 1
                break
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Slice a trace down to selected eager forward passes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("trace")
    ap.add_argument("-o", "--output", help="output trace (.json.gz)")
    ap.add_argument("--root-module", action="append", default=None,
                    help="nn.Module name that wraps one eager forward pass "
                         "(repeatable; the first is the main model, the rest "
                         "are auxiliary roots such as the MTP/draft head). "
                         "Default: Qwen3_5MoeForCausalLM Qwen3_5ForCausalLM")
    ap.add_argument("--kernel", action="append", default=None,
                    metavar="TAG=SUBSTRING",
                    help="attention kernel used to label a pass; repeatable. "
                         "Default: prefill=fmha_fwd_hd256 and "
                         "extend=FmhaBatchPrefillWithPagedKV")
    ap.add_argument("--select-kernel", default=None,
                    help="keep passes whose label matches this TAG "
                         "(or a raw kernel substring)")
    ap.add_argument("--select-pass", nargs="+", type=int, default=None,
                    help="keep these pass indices (as shown by --list)")
    ap.add_argument("--max-passes", type=int, default=None,
                    help="keep at most N of the selected passes")
    ap.add_argument("--skip-passes", type=int, default=0,
                    help="drop the first N selected passes (warm-up)")
    ap.add_argument("--window", choices=("iteration", "span"),
                    default="iteration",
                    help="'iteration' (default) = main model span + the "
                         "auxiliary root spans (MTP/draft head) of the same "
                         "iteration; 'span' = main nn.Module span only")
    ap.add_argument("--list", action="store_true",
                    help="only print the pass table and exit")
    args = ap.parse_args()

    if args.kernel:
        kernel_pats = []
        for spec in args.kernel:
            tag, _, pat = spec.partition("=")
            kernel_pats.append((tag, pat or tag))
    else:
        kernel_pats = [("prefill", "fmha_fwd_hd256"),
                       ("extend", "FmhaBatchPrefillWithPagedKV")]

    roots = args.root_module or ["Qwen3_5MoeForCausalLM", "Qwen3_5ForCausalLM"]
    print(f"[1/4] scanning forward passes in {args.trace} ...", flush=True)
    spans, kern_corr = scan_passes(args.trace, roots, kernel_pats)
    if not spans[roots[0]]:
        sys.exit(f"ERROR: no 'nn.Module: {roots[0]}_0' spans found. "
                 f"Pass --root-module <ModelClassName>.")
    for m in roots:
        print(f"  {len(spans[m]):>5} spans of {m}")
    print(f"  {len(kern_corr)} tagged attention kernels")

    print("[2/4] resolving kernel launch timestamps ...", flush=True)
    launch = scan_launch_ts(args.trace, kern_corr)

    per_pass = build_windows(spans, roots, args.window)
    passes = label_passes(per_pass, kern_corr, launch)
    t0 = passes[0]["ts"]

    print(f"  {'idx':>4} {'rel_ts(us)':>14} {'window(us)':>12}  label")
    for i, p in enumerate(passes):
        label = ", ".join(f"{k}x{v}" for k, v in sorted(p["tags"].items())) or "-"
        width = sum(b - a for a, b in p["intervals"])
        print(f"  {i:>4} {p['ts']-t0:>14.1f} {width:>12.1f}  {label}")

    if args.list:
        return
    if not args.output:
        sys.exit("ERROR: -o/--output is required (or use --list)")

    # ---- choose passes -------------------------------------------------
    if args.select_pass is not None:
        chosen = [i for i in args.select_pass if 0 <= i < len(passes)]
    elif args.select_kernel:
        sel = args.select_kernel
        chosen = [i for i, p in enumerate(passes)
                  if sel in p["tags"] or any(sel in t for t in p["tags"])]
    else:
        chosen = [i for i, p in enumerate(passes) if p["tags"]]
    chosen = chosen[args.skip_passes:]
    if args.max_passes:
        chosen = chosen[:args.max_passes]
    if not chosen:
        sys.exit("ERROR: selection matched no passes")
    windows = sorted(iv for i in chosen for iv in passes[i]["intervals"])
    print(f"  keeping passes: {chosen}")
    print(f"  window coverage: "
          f"{sum(b - a for a, b in windows)/1000:.1f} ms of CPU timeline")

    def in_window(ts):
        # windows are few (tens), a linear scan is fine and avoids bisect edge
        for a, b in windows:
            if a <= ts < b:
                return True
            if ts < a:
                return False
        return False

    # ---- pass 2: collect correlations launched inside those windows ----
    print("[3/4] collecting correlation ids ...", flush=True)
    keep_corr = set()
    for blob, is_event in iter_events(args.trace, progress_every=5_000_000):
        if not is_event:
            continue
        if '"correlation"' not in blob:
            continue
        ph, cat, ts = parse_blob(blob)
        if cat in GPU_CATS or ts is None or not in_window(ts):
            continue
        m = CORR_RE.search(blob)
        if m:
            keep_corr.add(int(m.group(1)))
    print(f"  {len(keep_corr):,} correlations kept")

    # ---- pass 3: write the sliced trace --------------------------------
    print(f"[4/4] writing {args.output} ...", flush=True)
    opener = (lambda p: gzip.open(p, "wt", compresslevel=4)) \
        if args.output.endswith(".gz") else (lambda p: open(p, "w"))
    kept = dropped = 0
    with opener(args.output) as out:
        pending_event = None  # buffer so we can fix the trailing comma
        for blob, is_event in iter_events(args.trace):
            if not is_event:
                if blob[:3] == "  ]" and pending_event is not None:
                    out.write(pending_event.rstrip().rstrip(",") + "\n")
                    pending_event = None
                out.write(blob)
                continue
            ph, cat, ts = parse_blob(blob)
            keep = False
            if ph in ("M", "i") or cat == "Trace":
                keep = True
            elif ph in ("s", "f", "t"):
                m = FLOWID_RE.search(blob)
                keep = bool(m) and int(m.group(1)) in keep_corr
            elif cat in GPU_CATS:
                m = CORR_RE.search(blob)
                keep = bool(m) and int(m.group(1)) in keep_corr
            elif ts is not None:
                keep = in_window(ts)
            if not keep:
                dropped += 1
                continue
            kept += 1
            if pending_event is not None:
                out.write(pending_event)
            pending_event = blob if blob.rstrip().endswith(",") \
                else blob.rstrip() + ",\n"
        if pending_event is not None:
            out.write(pending_event.rstrip().rstrip(",") + "\n")

    size = os.path.getsize(args.output)
    print(f"  kept {kept:,} events, dropped {dropped:,} "
          f"-> {args.output} ({size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
