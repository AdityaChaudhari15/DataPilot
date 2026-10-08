#!/usr/bin/env python3
"""Generate scalable-difficulty task suites with computed ground truth -> data/tasks_gen.json"""
import argparse
from datapilot.generators import make_suite
from datapilot.tasks import save_tasks

ap = argparse.ArgumentParser()
ap.add_argument("--seq-depths", type=int, nargs="*", default=[3, 6, 10, 15])
ap.add_argument("--breadth-groups", type=int, nargs="*", default=[3, 6, 10, 14])
ap.add_argument("--per-level", type=int, default=10)
ap.add_argument("--out", default="data/tasks_gen.json")
a = ap.parse_args()
tasks = make_suite("sequential", a.seq_depths, a.per_level) + make_suite("breadth", a.breadth_groups, a.per_level)
save_tasks(tasks, a.out)
print(f"wrote {len(tasks)} tasks -> {a.out}")
