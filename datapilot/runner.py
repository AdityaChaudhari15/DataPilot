"""Resumable, order-randomised experiment runner. One JSON line per (architecture, task, repetition)."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json, random, threading, time

from .config import ModelConfig, RunConfig, config_fingerprint
from .evaluators import grade, legacy_grade
from .tasks import Task, build_prompt

_lock = threading.Lock()


def job_key(arch: str, task_id: str, rep: int) -> str:
    return f"{arch}|{task_id}|{rep}"


def plan_jobs(tasks: list[Task], archs: list[str], reps: int, seed: int) -> list[tuple[str, Task, int]]:
    """Randomise task order and architecture order within each (task, rep) group: removes
    time-of-day / endpoint-load confounds from the paired latency comparison."""
    rng = random.Random(seed)
    groups = [(t, r) for t in tasks for r in range(1, reps + 1)]
    rng.shuffle(groups)
    jobs = []
    for t, r in groups:
        order = archs[:]
        rng.shuffle(order)
        jobs += [(a, t, r) for a in order]
    return jobs


def _tool_names(messages) -> list[str]:
    names = []
    for m in messages:
        for c in getattr(m, "tool_calls", None) or []:
            if c.get("name"):
                names.append(c["name"])
    return names


def run_one(arch: str, graph, task: Task, rep: int, run: RunConfig, fingerprint: str) -> dict:
    from langchain_core.messages import HumanMessage
    rec = {"architecture": arch, "task_id": task.task_id, "suite": task.suite, "level": task.level,
           "tags": task.tags, "rep": rep, "fingerprint": fingerprint, "workers": run.workers,
           "started_at": time.time()}
    status, state, err, attempts, latency = "ok", None, None, 0, None
    for attempts in range(1, 4):
        t0 = time.perf_counter()
        try:
            state = graph.invoke({"messages": [HumanMessage(content=build_prompt(task))], "usage": []},
                                 config={"recursion_limit": run.recursion_limit})
            latency = time.perf_counter() - t0
            err = None
            break
        except Exception as e:                      # noqa: BLE001  (record, don't crash the sweep)
            latency = time.perf_counter() - t0
            err = repr(e)
            if type(e).__name__ == "GraphRecursionError":
                status = "recursion_limit"
                break
            time.sleep(2 ** attempts)
    if state is None:
        status = status if status != "ok" else "error"
        rec.update(status=status, error=err, attempts=attempts, latency_s=latency, correct=False,
                   input_tokens=None, output_tokens=None, total_tokens=None, llm_calls=None)
        return rec

    usage = state.get("usage", [])
    final = state.get("final_text") or (lambda c: c if isinstance(c, str) else str(c))(state["messages"][-1].content)
    g = grade(task, final)
    rec.update(
        status=g.status if g.status == "format_error" else "ok",
        error=None, attempts=attempts, latency_s=latency,
        correct=g.correct, field_results=g.fields,
        legacy_correct=legacy_grade(task.v1_evaluator, task.v1_expected, final) if task.v1_evaluator else None,
        final_text=final,
        input_tokens=sum(u["input"] for u in usage), output_tokens=sum(u["output"] for u in usage),
        total_tokens=sum(u["total"] for u in usage), llm_calls=len(usage),
        usage_complete=not any(u["missing"] for u in usage),
        usage_by_node=_by_node(usage), tool_calls=len(_tool_names(state["messages"])),
        tools_used=sorted(set(_tool_names(state["messages"]))), meta=state.get("meta"))
    return rec


def _by_node(usage: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for u in usage:
        d = out.setdefault(u["node"], {"calls": 0, "input": 0, "output": 0, "total": 0})
        d["calls"] += 1
        for k in ("input", "output", "total"):
            d[k] += u[k]
    return out


def run_experiment(tasks, archs, graphs: dict, model: ModelConfig, run: RunConfig, out_path: str | Path,
                   log=print) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add(job_key(r["architecture"], r["task_id"], r["rep"]))
    fp = config_fingerprint(model, run)
    jobs = [j for j in plan_jobs(tasks, archs, run.reps, run.seed) if job_key(j[0], j[1].task_id, j[2]) not in done]
    log(f"{len(jobs)} runs to do ({len(done)} already in {out.name}); config {fp}; workers={run.workers}")

    def work(job):
        arch, task, rep = job
        rec = run_one(arch, graphs[arch], task, rep, run, fp)
        with _lock, out.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    n = 0
    with ThreadPoolExecutor(max_workers=run.workers) as ex:
        for fut in as_completed([ex.submit(work, j) for j in jobs]):
            r = fut.result()
            n += 1
            log(f"[{n}/{len(jobs)}] {r['architecture']:<16} {r['task_id']:<14} status={r['status']:<15} "
                f"correct={r['correct']!s:<5} tokens={r['total_tokens']} latency={r['latency_s']:.1f}s")
    return n
