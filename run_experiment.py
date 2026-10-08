#!/usr/bin/env python3
"""Run an experiment. Resumable: re-running the same command skips finished runs.

Examples
  python run_experiment.py --tasks data/tasks_v1.json --suites adversarial12 --archs single reflection multi_agent --dry-run
  python run_experiment.py --tasks data/tasks_v1.json --suites adversarial12 --reps 3 --out results/v1_adv.jsonl
  python run_experiment.py --tasks data/tasks_gen.json --archs single single_thinking self_consistency multi_agent
"""
import argparse, os, random, sys
from datapilot.env import load_env
load_env()
from datapilot.config import ModelConfig, RunConfig
from agentlab.tasks import load_tasks
from agentlab.runner import plan_jobs, run_experiment

ALL = ["single", "single_thinking", "reflection", "multi_agent", "self_consistency", "parallel"]

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", required=True)
ap.add_argument("--suites", nargs="*", help="e.g. core60 hard14 adversarial12 gen_sequential gen_breadth")
ap.add_argument("--archs", nargs="+", default=["single", "reflection", "multi_agent"], choices=ALL)
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--workers", type=int, default=1, help="1 for valid latency; >1 only for accuracy sweeps")
ap.add_argument("--model", default=ModelConfig.name)
ap.add_argument("--thinking", action="store_true", help="enable model reasoning for all non-'single_thinking' arms")
ap.add_argument("--no-system-role", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit", type=int, help="random subset of N tasks (seeded): use for the serial latency phase")
ap.add_argument("--out", default="results/run.jsonl")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

tasks = load_tasks(a.tasks, a.suites)
if a.limit and a.limit < len(tasks):
    tasks = random.Random(a.seed).sample(tasks, a.limit)
model = ModelConfig(name=a.model, thinking=a.thinking, use_system_role=not a.no_system_role)
run = RunConfig(workers=a.workers, reps=a.reps, seed=a.seed)
jobs = plan_jobs(tasks, a.archs, a.reps, a.seed)
print(f"{len(tasks)} tasks x {len(a.archs)} archs x {a.reps} reps = {len(jobs)} runs | model={model.name} thinking={model.thinking}")
if a.dry_run:
    from agentlab.analysis import estimate_tokens
    est = estimate_tokens(a.archs, len(tasks) * a.reps)
    if est:
        print("rough token budget from v1 means:", {k: f"{v/1e6:.2f}M" for k, v in est.items()}, "(other arms: unknown)")
    sys.exit(0)
if model.api_key_env not in os.environ:
    sys.exit(f"Set {model.api_key_env} first (Colab: os.environ['NVIDIA_API_KEY'] = userdata.get('NVIDIA_API_KEY2')).")

from agentlab.tools import build_langchain_tools
from agentlab.architectures import build_architectures
graphs = build_architectures(model, run, build_langchain_tools(), a.archs)
run_experiment(tasks, a.archs, graphs, model, run, a.out)
print("done ->", a.out, "| analyse with: python analyze.py", a.out)
