"""Convert the replayed v1 notebook tasks (tasks_all_raw.json) into the v2 schema.
Run scripts/extract_from_notebook.py first (it writes tasks_all_raw.json)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentlab.tasks import Task, save_tasks, leaked_numeric_values, leaked_text_values
from agentlab.evaluators import legacy_grade

UNDETERMINED = ["cannot be determined", "cannot be uniquely determined", "not enough information",
                "insufficient information", "impossible to determine", "cannot determine",
                "cannot know"]
SUITES = {"RESEARCH_TASKS": "core60", "HARD_VALIDATION_TASKS": "hard14", "ADVERSARIAL_TASKS": "adversarial12"}


def to_expected(t):
    e, ev = t["expected_answer"], t["evaluator_type"]
    if t["task_id"] in ("A7_05", "A7_12", "H5_04", "H6_04"):      # v1 later patched these to variant lists
        return {"answer": UNDETERMINED}
    if ev == "numeric":
        return {"answer": float(e)}
    if ev == "text":
        return {"answer": e if isinstance(e, list) else [str(e).lower()]}
    return {k: (float(v) if isinstance(v, (int, float)) else [str(v).lower()]) for k, v in e.items()}


raw = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "tasks_all_raw.json").read_text())
tasks = []
for key, suite in SUITES.items():
    for t in raw[key]:
        tasks.append(Task(
            task_id=t["task_id"], suite=suite, level=t["complexity_level"], category=t["category"],
            prompt=t["task"], expected=to_expected(t), expected_tools=t["expected_tools"],
            acceptable_tool_sets=t["acceptable_tool_sets"], v1_evaluator=t["evaluator_type"],
            v1_expected=t["expected_answer"],
            tags=["sequential"] + (["error_detection"] if t["tests_error_detection"] else [])
                 + (["conflicting"] if t["contains_conflicting_information"] else [])))
out = Path(__file__).resolve().parents[1] / "data" / "tasks_v1.json"
save_tasks(tasks, out)
echo = [t.task_id for t in tasks if leaked_numeric_values(t) or leaked_text_values(t)]
print(f"wrote {len(tasks)} tasks -> {out}")
print(f"{len(echo)} tasks have an expected answer already present in the prompt: {echo}")
