"""Programmatic task generators with a difficulty knob and computed (never hand-typed) ground truth.

Why: v1 hit a ceiling (100% accuracy) on short tasks. Literature predicts the architectural effect
depends on task structure: sequential dependency (multi-agent tends to hurt) vs. independent
sub-problems / long context (multi-agent can help). These generators let you vary exactly that.
"""
from __future__ import annotations
import random
from .tasks import Task


def sequential_chain(depth: int, seed: int) -> Task:
    """Strictly dependent steps: each percentage change applies to the previous result."""
    rng = random.Random(seed)
    value = start = rng.choice([800, 1000, 1200, 1500, 2000])
    steps = []
    for i in range(depth):
        pct = rng.choice([5, 8, 10, 12, 15, 20, 25])
        up = (i % 2 == 0) if rng.random() < 0.7 else (rng.random() < 0.5)
        value = value * (1 + pct / 100) if up else value * (1 - pct / 100)
        steps.append(f"{'increase' if up else 'decrease'} by {pct}%")
    prompt = (f"A quantity starts at {start}. Apply these changes strictly in order, each applied to the "
              f"result of the previous step: " + "; then ".join(steps) + ". What is the final value?")
    return Task(task_id=f"GS_{depth:02d}_{seed}", suite="gen_sequential", level=depth,
                category="sequential_chain", prompt=prompt, expected={"final_value": round(value, 2)},
                tol=0.05, tags=["sequential", "generated"])


def breadth_aggregate(n_groups: int, rows_per_group: int, seed: int) -> Task:
    """Independent sub-problems over a long table: per-group revenue, then global max and total."""
    rng = random.Random(seed)
    names = [f"G{i + 1}" for i in range(n_groups)]
    revenue, lines = {}, []
    for g in names:
        total = 0.0
        lines.append(f"Group {g}:")
        for r in range(rows_per_group):
            units, price = rng.randint(5, 60), rng.choice([4.5, 7.25, 9.99, 12.5, 15.0, 22.4])
            total += units * price
            lines.append(f"  item {r + 1}: units={units}, unit_price={price}")
        revenue[g] = total
    best = max(revenue, key=revenue.get)
    prompt = ("Each group below lists items with units sold and unit price (revenue = units x unit price). "
              "Compute each group's total revenue, then report which group has the highest revenue "
              "and the total revenue across all groups.\n\n" + "\n".join(lines))
    return Task(task_id=f"GB_{n_groups:02d}x{rows_per_group:02d}_{seed}", suite="gen_breadth",
                level=n_groups * rows_per_group, category="breadth_aggregate", prompt=prompt,
                expected={"best_group": [best.lower()], "total_revenue": round(sum(revenue.values()), 2)},
                tol=0.05, tags=["parallelizable", "long_context", "generated"])


def make_suite(kind: str, levels: list[int], seeds_per_level: int, base_seed: int = 1000) -> list[Task]:
    tasks = []
    for lv in levels:
        for s in range(seeds_per_level):
            seed = base_seed + lv * 100 + s
            tasks.append(sequential_chain(lv, seed) if kind == "sequential"
                         else breadth_aggregate(lv, 6, seed))
    return tasks
