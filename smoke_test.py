#!/usr/bin/env python3
"""LIVE pre-flight check (uses your API key; ~10-20 model calls). Run this BEFORE any sweep.

  python smoke_test.py                      # all six architectures on one easy task
  python smoke_test.py --archs single reflection

It checks: endpoint reachable, system role accepted, thinking toggle works, token usage is reported,
every architecture returns a parsable FINAL ANSWER, and prints per-role token usage.
"""
import argparse, os, sys
from datapilot.env import load_env
load_env()
from datapilot.config import ModelConfig, RunConfig, config_fingerprint
from datapilot.generators import sequential_chain

ALL = ["single", "single_thinking", "reflection", "multi_agent", "self_consistency", "parallel"]
ap = argparse.ArgumentParser()
ap.add_argument("--archs", nargs="+", default=ALL, choices=ALL)
ap.add_argument("--model", default=ModelConfig.name)
ap.add_argument("--no-system-role", action="store_true")
a = ap.parse_args()

cfg = ModelConfig(name=a.model, use_system_role=not a.no_system_role)
if cfg.api_key_env not in os.environ:
    sys.exit(f"Set {cfg.api_key_env} (environment variable or .env file) first.")

from langchain_core.messages import HumanMessage
from datapilot.architectures import make_llm, _prefix, build_architectures
from datapilot.tools import build_langchain_tools
from datapilot.runner import run_one

ok_all = True
def check(label, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label} {detail}")

print("1) Endpoint, system role, usage reporting")
try:
    r = make_llm(cfg).invoke(_prefix("Answer with one word.", cfg.use_system_role) + [HumanMessage(content="Capital of France?")])
    check("endpoint reachable + system role accepted", True, f"-> {str(r.content)[:40]!r}")
    check("usage_metadata reported", bool(r.usage_metadata), str(r.usage_metadata))
except Exception as e:
    check("endpoint call", False, repr(e)[:300])
    print("  hint: if the error mentions roles/templates, rerun with --no-system-role")
    sys.exit(1)

print("2) Thinking toggle")
q = [HumanMessage(content="What is 17 * 23? Think carefully, then give the number.")]
off = make_llm(cfg, thinking=False).invoke(q)
on = make_llm(cfg, thinking=True).invoke(q)
o_off, o_on = (off.usage_metadata or {}).get("output_tokens", 0), (on.usage_metadata or {}).get("output_tokens", 0)
print(f"  output tokens: thinking off={o_off}, on={o_on}")
print(f"  extra fields on 'on' reply: {[k for k in on.additional_kwargs]}")
check("thinking ON uses more output tokens (or exposes a reasoning field)", o_on > o_off or bool(on.additional_kwargs))
print("  NOTE: if ON == OFF, your endpoint may ignore enable_thinking; the single_thinking arm is then not a real control.")

print("3) One task through each architecture")
task = sequential_chain(4, seed=7)
print("  task:", task.prompt[:110], "...", "| expected:", task.expected)
run = RunConfig(workers=1)
graphs = build_architectures(cfg, run, build_langchain_tools(), a.archs)
fp = config_fingerprint(cfg, run)
for arch in a.archs:
    rec = run_one(arch, graphs[arch], task, 1, run, fp)
    print(f"\n  == {arch}: status={rec['status']} correct={rec['correct']} tokens={rec['total_tokens']} "
          f"llm_calls={rec['llm_calls']} tool_calls={rec.get('tool_calls')} latency={rec['latency_s']:.1f}s")
    if rec["status"] in ("error", "recursion_limit"):
        check(f"{arch} ran", False, str(rec.get("error"))[:300]); continue
    print("     usage_by_node:", rec["usage_by_node"])
    print("     final_text tail:", repr(rec["final_text"][-160:]))
    check(f"{arch}: FINAL ANSWER parsed", rec["status"] != "format_error")
    check(f"{arch}: usage complete", rec["usage_complete"])
    check(f"{arch}: used a tool", (rec.get("tool_calls") or 0) > 0 or arch == "parallel")

print("\nRESULT:", "ALL CHECKS PASSED - safe to start a sweep" if ok_all else "SOME CHECKS FAILED - fix before sweeping")
sys.exit(0 if ok_all else 1)
