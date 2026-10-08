"""Task schema, loading, prompt building and answer-leakage detection."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json, re

Expected = dict  # field -> number | list[str] (accepted phrasings)


@dataclass
class Task:
    task_id: str
    suite: str                      # core60 | hard14 | adversarial12 | gen_sequential | gen_breadth ...
    level: int                      # 1..6 for v1 tasks; difficulty knob for generated tasks
    category: str
    prompt: str
    expected: Expected              # e.g. {"final_price": 960, "report": ["incorrect"]}
    tol: float = 0.01               # absolute tolerance for numeric fields
    tags: list[str] = field(default_factory=list)   # e.g. ["sequential"], ["parallelizable"]
    expected_tools: list[str] = field(default_factory=list)
    acceptable_tool_sets: list[list[str]] = field(default_factory=list)
    v1_evaluator: str = ""          # numeric | structured | text (kept for legacy comparison)
    v1_expected: object = None      # original v1 ground truth (kept for legacy comparison)


def load_tasks(path: str | Path, suites: list[str] | None = None) -> list[Task]:
    rows = json.loads(Path(path).read_text())
    tasks = [Task(**r) for r in rows]
    ids = [t.task_id for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task_id values in task file.")
    return [t for t in tasks if suites is None or t.suite in suites]


def save_tasks(tasks: list[Task], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(t) for t in tasks], indent=1, ensure_ascii=False))


def answer_format_instruction(task: Task) -> str:
    """Identical suffix for every architecture: makes grading deterministic and echo-proof."""
    keys = []
    for k, v in task.expected.items():
        keys.append(f'"{k}": <number>' if isinstance(v, (int, float)) else f'"{k}": "<short text>"')
    spec = "{" + ", ".join(keys) + "}"
    return ("\n\nEnd your reply with exactly one final line in this format "
            f"(valid JSON, no extra text after it):\nFINAL ANSWER: {spec}")


def build_prompt(task: Task) -> str:
    return task.prompt.rstrip() + answer_format_instruction(task)


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(str(text).replace(",", ""))]


def leaked_numeric_values(task: Task, tol: float | None = None) -> list[float]:
    """Expected numeric answers that already appear verbatim in the prompt (grader-leakage risk)."""
    tol = task.tol if tol is None else tol
    prompt_nums = numbers_in(task.prompt)
    vals = [float(v) for v in task.expected.values() if isinstance(v, (int, float))]
    return [v for v in vals if any(abs(p - v) <= tol for p in prompt_nums)]


def leaked_text_values(task: Task) -> list[str]:
    """Expected text answers that already appear in the prompt (e.g. 'Report B' in a Report A/B task)."""
    p = task.prompt.lower()
    out = []
    for v in task.expected.values():
        if isinstance(v, list):
            out += [s for s in v if s.lower() in p]
    return out
