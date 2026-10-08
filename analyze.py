#!/usr/bin/env python3
"""Print the full statistical report for one or more result files; optionally write figures.

  python analyze.py results/gen.jsonl results/v1_adv.jsonl --plots figures/
"""
import argparse, pandas as pd
from datapilot.analysis import load_results, report

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--baseline", default="single")
ap.add_argument("--plots", help="directory for PNG figures (needs matplotlib)")
a = ap.parse_args()

df = pd.concat([load_results(p) for p in a.paths], ignore_index=True)
print(report(df, baseline=a.baseline))
if "legacy_correct" in df and df.legacy_correct.notna().any():
    d = df[df.legacy_correct.notna()]
    print("\n== v1 vs v2 grader disagreement ==")
    print(pd.crosstab(d.legacy_correct, d.correct, rownames=["v1 grader"], colnames=["v2 grader"]))
if a.plots:
    from datapilot.plots import make_plots
    for p in make_plots(df, a.plots):
        print("wrote", p)
