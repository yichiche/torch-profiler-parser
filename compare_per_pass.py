#!/usr/bin/env python3
"""Per-forward-pass normalized diff of two trace_module_analyzer reports.

`compare_analysis.py` diffs absolute totals, which is only meaningful when both
traces contain the same number of forward passes.  When two runs chunk a long
prompt differently (e.g. 25 extend chunks before a change vs 4 after), divide by
the pass count first -- that is what this does.

Usage:
  python3 compare_per_pass.py before.xlsx after.xlsx --passes 25 4
  python3 compare_per_pass.py before.xlsx after.xlsx --passes 25 4 --category attention
"""

import argparse
import collections

import openpyxl


def load_kernels(path):
    ws = openpyxl.load_workbook(path, read_only=True)["GPU Kernels"]
    rows = list(ws.iter_rows(values_only=True))
    out = []
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        out.append({"name": str(r[0]), "cat": str(r[1]),
                    "total": float(r[2]), "count": int(r[3])})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--passes", nargs=2, type=int, required=True,
                    metavar=("N_BEFORE", "N_AFTER"),
                    help="forward-pass count in each trace (from "
                         "extract_phase_trace.py --list)")
    ap.add_argument("--labels", nargs=2, default=["BEFORE", "AFTER"])
    ap.add_argument("--category", default=None,
                    help="only list kernels of this category")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    B, A = load_kernels(args.before), load_kernels(args.after)
    nb, na = args.passes
    lb, la = args.labels

    cb, ca = collections.Counter(), collections.Counter()
    for r in B:
        cb[r["cat"]] += r["total"]
    for r in A:
        ca[r["cat"]] += r["total"]

    print(f"\n=== category totals, us per forward pass "
          f"({lb}: {nb} passes, {la}: {na} passes) ===")
    print(f"{'category':16s} {lb+' us/pass':>16s} {la+' us/pass':>16s} "
          f"{'delta':>12s} {'ratio':>7s}")
    tb = ta = 0.0
    for k in sorted(set(cb) | set(ca), key=lambda k: -cb[k]):
        b, a = cb[k] / nb, ca[k] / na
        tb += b
        ta += a
        print(f"{k:16s} {b:16,.0f} {a:16,.0f} {a-b:12,.0f} "
              f"{(a/b if b else float('inf')):7.2f}")
    print(f"{'TOTAL':16s} {tb:16,.0f} {ta:16,.0f} {ta-tb:12,.0f} {ta/tb:7.2f}")

    print(f"\n=== top kernels, us per forward pass"
          f"{' [' + args.category + ']' if args.category else ''} ===")
    for label, rows, n in ((lb, B, nb), (la, A, na)):
        sel = [r for r in rows if not args.category or r["cat"] == args.category]
        print(f"-- {label}")
        for r in sorted(sel, key=lambda r: -r["total"])[:args.top]:
            print(f"   {r['total']/n:12,.0f} us/pass  n={r['count']/n:6.1f}  "
                  f"avg={r['total']/r['count']:9,.0f} us  {r['cat']:14s} "
                  f"{r['name'][:60]}")


if __name__ == "__main__":
    main()
