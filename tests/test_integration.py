"""End-to-end test of architectures -> runner -> analysis using tests/fake_stack.py (strict fakes).
Skipped automatically when the real langgraph is installed (use smoke_test.py for the live check)."""
import importlib.util, json, os, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

REAL = importlib.util.find_spec("langgraph") is not None


@unittest.skipIf(REAL, "real langgraph installed; run smoke_test.py for the live check")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fake_stack
        from datapilot.tasks import load_tasks
        from datapilot.generators import make_suite
        v1 = load_tasks(ROOT / "data" / "tasks_v1.json", ["adversarial12"])[:3]
        gen = make_suite("sequential", [3, 6], 1) + make_suite("breadth", [3], 1)
        cls.tasks = v1 + gen
        cls.wrong_id = gen[0].task_id

        def answer_for(text):
            for t in cls.tasks:
                if t.prompt in text:
                    if t.task_id == cls.wrong_id:
                        return {k: 0 for k in t.expected}                      # deliberately wrong
                    return {k: (v if isinstance(v, (int, float)) else v[0]) for k, v in t.expected.items()}
            return {}
        cls.fake = fake_stack.install(answer_for)
        os.environ["NVIDIA_API_KEY"] = "dummy"
        from datapilot.config import ModelConfig, RunConfig
        from datapilot.tools import build_langchain_tools
        from datapilot.architectures import build_architectures
        cls.model, cls.run_cfg = ModelConfig(), RunConfig(workers=2, reps=2, recursion_limit=40)
        cls.archs = ["single", "single_thinking", "reflection", "multi_agent", "self_consistency", "parallel"]
        cls.graphs = build_architectures(cls.model, cls.run_cfg, build_langchain_tools(), cls.archs)
        cls.out = Path(tempfile.mkdtemp()) / "r.jsonl"
        from datapilot.runner import run_experiment
        cls.n_first = run_experiment(cls.tasks, cls.archs, cls.graphs, cls.model, cls.run_cfg, cls.out, log=lambda *_: None)
        cls.recs = [json.loads(l) for l in cls.out.read_text().splitlines()]

    def test_all_runs_recorded(self):
        self.assertEqual(self.n_first, len(self.tasks) * 6 * 2)
        self.assertEqual(len(self.recs), self.n_first)

    def test_grading_end_to_end(self):
        for r in self.recs:
            self.assertEqual(r["correct"], r["task_id"] != self.wrong_id, (r["architecture"], r["task_id"]))
            self.assertEqual(r["status"], "ok")
            self.assertTrue(r["usage_complete"])
            self.assertGreater(r["total_tokens"], 0)

    def test_per_role_usage(self):
        want = {"single": {"single"}, "single_thinking": {"single_thinking"}, "reflection": {"agent", "checker"},
                "multi_agent": {"planner", "analyst", "reporter"},
                "self_consistency": {"sc_sample"}, "parallel": {"orchestrator", "sub_analyst", "reporter"}}
        for r in self.recs:
            self.assertEqual(set(r["usage_by_node"]), want[r["architecture"]], r["architecture"])
        sc = next(r for r in self.recs if r["architecture"] == "self_consistency")
        self.assertEqual(sc["usage_by_node"]["sc_sample"]["calls"], 2 * 3)      # 3 samples x (tool-call turn + final)

    def test_model_settings(self):
        kws = [k for k in self.fake.created_llms]
        self.assertTrue(any(k["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True for k in kws))
        self.assertTrue(any(k["temperature"] == 0.7 for k in kws))
        self.assertTrue(all(k["max_tokens"] == 4096 and k["max_retries"] == 3 for k in kws))

    def test_resume_skips_finished_runs(self):
        from datapilot.runner import run_experiment
        again = run_experiment(self.tasks, self.archs, self.graphs, self.model, self.run_cfg, self.out, log=lambda *_: None)
        self.assertEqual(again, 0)
        self.assertEqual(len(self.out.read_text().splitlines()), self.n_first)

    def test_recursion_limit_recorded_without_fake_zero_tokens(self):
        from datapilot.config import RunConfig
        from datapilot.runner import run_one
        rec = run_one("multi_agent", self.graphs["multi_agent"], self.tasks[0], 1, RunConfig(recursion_limit=2), "fp")
        self.assertEqual(rec["status"], "recursion_limit")
        self.assertIsNone(rec["total_tokens"])
        self.assertFalse(rec["correct"])

    def test_analysis_pipeline(self):
        from datapilot import analysis as an
        df = an.load_results(self.out)
        self.assertTrue(df.compute_valid.all())
        acc = an.accuracy_table(df)
        self.assertEqual(set(acc.index), set(self.archs))
        self.assertAlmostEqual(acc.loc["single", "accuracy"], (len(self.tasks) - 1) / len(self.tasks))
        txt = an.report(df)
        for needle in ("ACCURACY", "COST", "POWER", "TOKENS PER ROLE", "ACCURACY BY SUITE"):
            self.assertIn(needle, txt)
        self.assertIn("note", an.fit_overhead(df, "multi_agent"))     # fake model has constant tokens: no fit possible
        self.assertEqual(an.latency_diagnostics(df)["max_workers_used"], 2)

    def test_plots(self):
        from datapilot import analysis as an
        from datapilot.plots import make_plots
        paths = make_plots(an.load_results(self.out), Path(tempfile.mkdtemp()))
        self.assertGreaterEqual(len(paths), 3)
        self.assertTrue(all(p.exists() and p.stat().st_size > 1000 for p in paths))


if __name__ == "__main__":
    unittest.main(verbosity=2)
