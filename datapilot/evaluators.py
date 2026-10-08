"""Answer grading.

v2 grader: the model must end with `FINAL ANSWER: {json}`; each expected field is matched
against the *parsed* field only (never against the whole response), so echoing the prompt,
mentioning a number in passing, or writing "not incorrect" cannot score.

`legacy_grade` re-implements the v1 grader so v1-vs-v2 disagreement can be measured.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json, re
from .tasks import Task, numbers_in

_FINAL = re.compile(r"FINAL ANSWER\s*:\s*(\{.*\})", re.IGNORECASE | re.DOTALL)
_NEGATORS = {"not", "no", "never", "isn't", "isnt", "wasn't", "wasnt", "n't"}
_ENTITY = re.compile(r"\b(report|model|system|claim|option|dataset)\s+([a-z0-9])\b")


def normalize(text: str) -> str:
    t = str(text).lower().strip()
    t = re.sub(r"[*_`\"'.,;:!?()\[\]]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_final_answer(text: str) -> dict | None:
    """Return the JSON object on the last 'FINAL ANSWER:' line, or None if absent/invalid."""
    if not text:
        return None
    matches = list(_FINAL.finditer(text))
    if not matches:
        return None
    blob = matches[-1].group(1).strip()
    # take the first balanced {...} (models sometimes add trailing text)
    depth = 0
    for i, ch in enumerate(blob):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            blob = blob[: i + 1]
            break
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _to_number(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    nums = numbers_in(str(value))
    return nums[0] if len(nums) == 1 else None


def _phrase_present(field_text: str, phrase: str) -> bool:
    """Whole-phrase match that rejects negated mentions ('not incorrect', 'not Report B')."""
    ft, ph = normalize(field_text), normalize(phrase)
    for m in re.finditer(r"\b" + re.escape(ph) + r"\b", ft):
        before = ft[: m.start()].split()[-2:]
        if not any(tok in _NEGATORS for tok in before):
            return True
    return False


def _text_match(value, variants: list[str]) -> bool:
    text = str(value)
    # reject answers naming several different entities ("Report A or Report B")
    entities = {m.group(0) for m in _ENTITY.finditer(normalize(text))}
    if len(entities) > 1:
        return False
    return any(_phrase_present(text, v) for v in variants)


@dataclass
class Grade:
    correct: bool
    status: str                       # ok | wrong | format_error
    fields: dict = field(default_factory=dict)


def grade(task: Task, response_text: str) -> Grade:
    parsed = parse_final_answer(response_text)
    if parsed is None:
        return Grade(False, "format_error")
    results = {}
    for key, expected in task.expected.items():
        got = parsed.get(key)
        if got is None:
            results[key] = False
        elif isinstance(expected, (int, float)):
            num = _to_number(got)
            results[key] = num is not None and abs(num - float(expected)) <= task.tol
        else:
            results[key] = _text_match(got, expected)
    ok = all(results.values())
    return Grade(ok, "ok" if ok else "wrong", results)


# ----------------------------- v1 grader (for replication / disagreement analysis) -----------------------------

def legacy_grade(evaluator_type: str, expected, response_text: str) -> bool:
    """v1 behaviour: any number anywhere in the response; substring text matching."""
    if not response_text:
        return False
    nums, text = numbers_in(response_text), normalize(response_text)
    if evaluator_type == "numeric":
        return any(abs(n - float(expected)) <= 1e-6 for n in nums)
    if evaluator_type == "text":
        variants = expected if isinstance(expected, (list, tuple, set)) else [expected]
        return any(normalize(v) in text for v in variants)
    if evaluator_type == "structured":
        for value in expected.values():
            if isinstance(value, (int, float)):
                if not any(abs(n - float(value)) <= 1e-6 for n in nums):
                    return False
            elif normalize(value) not in text:
                return False
        return True
    return False
