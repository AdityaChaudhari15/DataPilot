import json, sys, unittest
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datapilot.tools import PURE_TOOLS
from datapilot.tasks import load_tasks, build_prompt, leaked_numeric_values, Task
from datapilot.evaluators import grade, legacy_grade, parse_final_answer
from datapilot.generators import sequential_chain, breadth_aggregate
from datapilot import analysis as an
from datapilot.runner import plan_jobs, _by_node
from datapilot.architectures import SelfConsistency, ParallelOrchestrator, _canon

TASKS = {t.task_id: t for t in load_tasks(ROOT / "data" / "tasks_v1.json")}


def fa(**kw):
    return "Some reasoning.\nFINAL ANSWER: " + json.dumps(kw)


class TestTools(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(PURE_TOOLS["add"](2, 3), 5)
        self.assertEqual(PURE_TOOLS["calculate_percentage"](9, 72), 12.5)
        self.assertEqual(PURE_TOOLS["compare_values"](1, 2), "b is greater than a")
        with self.assertRaises(ValueError):
            PURE_TOOLS["divide"](1, 0)
        with self.assertRaises(ValueError):
            PURE_TOOLS["find_max"]([])
        self.assertEqual(len(PURE_TOOLS), 10)


class TestTasks(unittest.TestCase):
    def test_counts_and_ids(self):
        suites = pd.Series([t.suite for t in TASKS.values()]).value_counts().to_dict()
        self.assertEqual(suites, {"core60": 60, "hard14": 14, "adversarial12": 12})

    def test_prompt_has_format_instruction(self):
        self.assertIn("FINAL ANSWER:", build_prompt(TASKS["A7_02"]))
        self.assertIn('"final_price": <number>', build_prompt(TASKS["A7_02"]))

    def test_known_leakage(self):
        self.assertEqual(sorted(leaked_numeric_values(TASKS["A7_01"])), [9.0, 12.5])
        self.assertEqual(leaked_numeric_values(TASKS["A7_02"]), [])


class TestEvaluators(unittest.TestCase):
    def test_correct_and_wrong(self):
        t = TASKS["A7_02"]
        self.assertTrue(grade(t, fa(final_price=960, report="Incorrect")).correct)
        self.assertTrue(grade(t, fa(final_price="$960.00", report="the report is incorrect")).correct)
        self.assertFalse(grade(t, fa(final_price=1000, report="Correct")).correct)

    def test_negation_is_rejected(self):
        self.assertFalse(grade(TASKS["A7_02"], fa(final_price=960, report="not incorrect")).correct)

    def test_swapped_labels_fail_v2_but_pass_v1(self):
        t = TASKS["A7_01"]
        wrong = ("The improvement is 12.5 percentage points and 9% relative.\n"
                 + fa(percentage_point_improvement=12.5, relative_improvement=9))
        self.assertFalse(grade(t, wrong).correct)
        self.assertTrue(legacy_grade(t.v1_evaluator, t.v1_expected, wrong))   # v1 false positive

    def test_prompt_echo_never_scores(self):
        for t in TASKS.values():
            self.assertFalse(grade(t, t.prompt).correct, t.task_id)
            self.assertEqual(grade(t, t.prompt).status, "format_error")

    def test_v1_grader_accepts_some_prompt_echoes(self):
        echo_pass = [t.task_id for t in TASKS.values()
                     if t.v1_evaluator and legacy_grade(t.v1_evaluator, t.v1_expected, t.prompt)]
        self.assertGreaterEqual(len(echo_pass), 20)      # documents the v1 weakness (23 observed)
        self.assertIn("A7_01", echo_pass)

    def test_underdetermined_variants_and_ambiguous_choice(self):
        self.assertTrue(grade(TASKS["A7_12"], fa(answer="This cannot be determined from the data")).correct)
        self.assertFalse(grade(TASKS["A7_12"], fa(answer=42)).correct)
        t = TASKS["L5_01"]                                   # expects 'report b'
        self.assertTrue(grade(t, fa(answer="Report B")).correct)
        self.assertFalse(grade(t, fa(answer="Report A or Report B")).correct)
        self.assertFalse(grade(t, fa(answer="not Report B")).correct)

    def test_parser(self):
        self.assertIsNone(parse_final_answer("no marker"))
        self.assertIsNone(parse_final_answer("FINAL ANSWER: {bad json"))
        self.assertEqual(parse_final_answer('FINAL ANSWER: {"a": 1} trailing text'), {"a": 1})


class TestGenerators(unittest.TestCase):
    def test_sequential_ground_truth(self):
        import re
        for seed in range(20):
            t = sequential_chain(5, seed)
            start = float(re.search(r"starts at (\d+)", t.prompt).group(1))
            val = start
            for kind, pct in re.findall(r"(increase|decrease) by (\d+)%", t.prompt):
                val *= 1 + (1 if kind == "increase" else -1) * int(pct) / 100
            self.assertAlmostEqual(val, t.expected["final_value"], places=1)

    def test_breadth_ground_truth(self):
        import re
        for seed in range(10):
            t = breadth_aggregate(5, 6, seed)
            totals, cur = {}, None
            for line in t.prompt.splitlines():
                if line.startswith("Group "):
                    cur = line.split()[1].rstrip(":")
                    totals[cur] = 0.0
                m = re.search(r"units=(\d+), unit_price=([\d.]+)", line)
                if m:
                    totals[cur] += int(m.group(1)) * float(m.group(2))
            self.assertEqual([max(totals, key=totals.get).lower()], t.expected["best_group"])
            self.assertAlmostEqual(sum(totals.values()), t.expected["total_revenue"], places=1)

    def test_generated_tasks_have_no_leak(self):
        for t in [sequential_chain(6, s) for s in range(30)] + [breadth_aggregate(4, 6, s) for s in range(30)]:
            self.assertEqual(leaked_numeric_values(t, tol=0.5), [], t.task_id)


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        v1 = pd.read_csv(ROOT / "data" / "adversarial_v1_runs.csv")
        v1["status"], v1["usage_complete"], v1["workers"] = "ok", True, 3
        v1["correct"] = v1.v1_success.astype(bool)
        v1["rep"], v1["llm_calls"], v1["tool_calls"] = 1, 0, 0
        v1["compute_valid"] = True
        self.df = v1

    def test_wilson_matches_paper_discussion(self):
        lo, hi = an.wilson(15, 15)
        self.assertAlmostEqual(lo, 0.796, places=3)
        self.assertEqual(hi, 1.0)

    def test_regression_paper_paired_numbers(self):
        r = an.paired_difference(self.df, "total_tokens", "multi_agent", "single")
        self.assertAlmostEqual(r["mean_diff"], 3089, delta=1)
        self.assertTrue(2500 < r["ci_lo"] < 2700 and 3400 < r["ci_hi"] < 3600)
        r = an.paired_difference(self.df, "total_tokens", "reflection", "single")
        self.assertAlmostEqual(r["mean_diff"], 1616, delta=1)
        r = an.paired_difference(self.df, "latency_s", "multi_agent", "single")
        self.assertAlmostEqual(r["mean_diff"], 94.64, delta=0.01)
        self.assertLess(r["p_signflip"], 0.05)
        r = an.paired_difference(self.df, "latency_s", "reflection", "single")
        self.assertAlmostEqual(r["mean_diff"], 39.23, delta=0.01)
        self.assertLess(r["ci_lo"], 0)                      # CI includes zero, as in the paper

    def test_overhead_denominator_sensitivity(self):
        m = self.df.groupby("architecture").total_tokens.apply(lambda s: s.astype(float).mean())
        self.assertAlmostEqual(m["multi_agent"] / m["single"] - 1, 1.267, places=3)   # 12-task basis, not 119.2%

    def test_fixed_overhead_model_matches_paper(self):
        m = an.fit_overhead(self.df, "multi_agent")
        self.assertAlmostEqual(m["intercept"], 2913, delta=5)
        self.assertAlmostEqual(m["slope"], 1.07, delta=0.01)
        self.assertAlmostEqual(m["r2"], 0.80, delta=0.01)
        self.assertAlmostEqual(m["spearman_baseline_vs_ratio"], -0.88, delta=0.01)
        r = an.fit_overhead(self.df, "reflection")
        self.assertAlmostEqual(r["intercept"], 1467, delta=5)
        self.assertAlmostEqual(r["r2"], 0.94, delta=0.01)
        self.assertIn("note", an.fit_overhead(self.df.head(2), "multi_agent"))

    def test_stats_helpers(self):
        self.assertAlmostEqual(an.mcnemar_exact(5, 0), 0.0625)
        self.assertAlmostEqual(an.signflip_pvalue([1, 2, 3, 4, 5]), 0.0625)
        self.assertEqual(an.min_detectable_gain(60), 0.17)
        self.assertGreaterEqual(an.min_detectable_gain(12), 0.6)
        self.assertEqual(an.tasks_needed(0.2), 48)


class TestRunnerAndArchitectures(unittest.TestCase):
    def test_plan_jobs_complete_deterministic_randomised(self):
        tasks = list(TASKS.values())[:8]
        archs = ["single", "reflection", "multi_agent"]
        a, b = plan_jobs(tasks, archs, 2, seed=1), plan_jobs(tasks, archs, 2, seed=1)
        self.assertEqual([(x, t.task_id, r) for x, t, r in a], [(x, t.task_id, r) for x, t, r in b])
        self.assertEqual(len(a), 8 * 2 * 3)
        self.assertEqual(len({(x, t.task_id, r) for x, t, r in a}), len(a))
        first_arch = {j[0] for j in a[::3]}
        self.assertGreater(len(first_arch), 1)            # architecture order is shuffled

    def test_by_node(self):
        u = [{"node": "a", "input": 1, "output": 2, "total": 3, "missing": False},
             {"node": "a", "input": 1, "output": 1, "total": 2, "missing": False}]
        self.assertEqual(_by_node(u)["a"], {"calls": 2, "input": 2, "output": 3, "total": 5})

    def test_self_consistency_majority(self):
        class Msg:  # minimal stand-in for a LangChain message
            def __init__(s, c): s.content = c
        answers = iter([fa(x=1), fa(x=2), fa(x=2)])
        class G:
            def invoke(s, inputs, config=None):
                return {"messages": [Msg(next(answers))], "usage": [{"node": "n", "input": 1, "output": 1, "total": 2, "missing": False}]}
        out = SelfConsistency(G(), 3).invoke({"messages": []})
        self.assertEqual(parse_final_answer(out["final_text"]), {"x": 2})
        self.assertEqual(len(out["usage"]), 3)

    def test_parallel_split_fallbacks(self):
        po = ParallelOrchestrator(lambda s, u, n: ('Sure: ["a", "b"] done', {"node": n}), None, k=4)
        self.assertEqual(po._split("task")[0], ["a", "b"])
        po = ParallelOrchestrator(lambda s, u, n: ("not json", {"node": n}), None, k=4)
        self.assertEqual(po._split("task")[0], ["task"])

    def test_canon_ignores_formatting(self):
        self.assertEqual(_canon(fa(x=1.001, y="Report B.")), _canon(fa(x=1.0, y="report b")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
