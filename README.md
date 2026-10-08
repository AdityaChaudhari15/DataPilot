# DataPilot: single vs. reflective vs. multi-agent LLM analysts

Refactor of `single_multi_ai_agents_research.ipynb` into a tested package, plus the extra arms and task suites
needed for the follow-up study (compute-matched controls, parallelizable tasks, scalable difficulty).
Python 3.10+.

```
datapilot/
  config.py         ModelConfig (thinking toggle, tokens, retries) and RunConfig (recursion cap, workers, SC samples)
  tools.py          the 10 deterministic tools (pure functions + LangChain adapter)
  tasks.py          Task schema, FINAL-ANSWER prompt suffix, answer-leakage detector
  evaluators.py     v2 grader (parsed-field matching, negation/ambiguity rejection) + v1 grader (for comparison)
  generators.py     sequential_chain and breadth_aggregate generators, computed ground truth
  architectures.py  single | single_thinking | reflection | multi_agent | self_consistency | parallel
  runner.py         resumable JSONL runner, randomised order, retries, per-role token accounting
  analysis.py       paired bootstrap, sign-flip, McNemar, Wilson, power, overhead model, per-level tables
  plots.py          figures (Pareto, accuracy/tokens by difficulty)
run_experiment.py   run a sweep         make_suite.py   generate tasks       analyze.py   report + figures
smoke_test.py       LIVE pre-flight     tests/          32 offline tests     data/        migrated tasks + v1 log
```

## 1. Setup
```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                 # then paste your NVIDIA key into .env  (Windows: copy .env.example .env)
```

## 2. Verify before spending anything (in this order)
```bash
python -m unittest discover -s tests -v      # 32 tests. 8 integration tests use a built-in fake LangGraph and are
                                             # skipped automatically once the real langgraph is installed.
python smoke_test.py                         # LIVE, ~20 model calls. Must end with "ALL CHECKS PASSED".
python run_experiment.py --tasks data/tasks_v1.json --suites adversarial12 --dry-run
```
`smoke_test.py` checks: endpoint + system role, that `usage_metadata` is reported, that the thinking toggle changes
output, and that each of the six architectures returns a parsable `FINAL ANSWER` with complete token usage.
If it reports that thinking ON == OFF, your endpoint ignores `enable_thinking`; then `single_thinking` is not a
real control and you should say so in the paper.

## 3. Recommended experiment (budget-aware order)
```bash
python make_suite.py --per-level 10                       # 80 generated tasks -> data/tasks_gen.json (use 14 for ~112)

# A. pilot: 1 rep, find problems cheaply (check format_error rate and the ceiling warning)
python run_experiment.py --tasks data/tasks_gen.json --suites gen_sequential --archs single reflection multi_agent \
       --reps 1 --workers 4 --limit 12 --out results/pilot.jsonl
python analyze.py results/pilot.jsonl

# B. main accuracy + token sweep (parallel is fine: concurrency does not change tokens)
python run_experiment.py --tasks data/tasks_gen.json --suites gen_sequential \
       --archs single single_thinking self_consistency reflection multi_agent --reps 3 --workers 4 --out results/seq.jsonl
python run_experiment.py --tasks data/tasks_gen.json --suites gen_breadth \
       --archs single single_thinking self_consistency reflection multi_agent parallel --reps 3 --workers 4 --out results/breadth.jsonl

# C. your original adversarial set re-run with the v2 grader (reps give run-to-run noise)
python run_experiment.py --tasks data/tasks_v1.json --suites adversarial12 \
       --archs single single_thinking reflection multi_agent --reps 3 --workers 4 --out results/adv.jsonl

# D. latency, serial, on a subsample (never mix with the parallel runs above)
python run_experiment.py --tasks data/tasks_gen.json --archs single reflection multi_agent \
       --workers 1 --limit 20 --out results/latency.jsonl

python analyze.py results/seq.jsonl results/breadth.jsonl results/adv.jsonl --plots figures/
python analyze.py results/latency.jsonl
```
Every command is resumable: re-run the same line after a crash or rate-limit and finished runs are skipped.
For a second model add `--model <id>` (check the exact ID in the NVIDIA catalog) and a different `--out`.

## 4. Reading the report
- **ACCURACY (intention-to-treat)**: errors and format failures count as wrong. `failed_runs` and `format_errors` are shown.
  A high `format_errors` count means the model is not following the final-line format: read `final_text` in the JSONL.
- **WARNING ceiling**: baseline >= 95% means accuracy comparisons say nothing; use deeper chains / more groups or a smaller model.
- **COST + paired differences**: mean diff, bootstrap CI, sign-flip p, dz, task count.
- **overhead model**: `tokens_arch = intercept + slope x single_tokens`. In your v1 data: multi-agent 2,913 + 1.07x (R2 0.80).
  Check whether it replicates on generated tasks; the generators let single-agent cost vary widely.
- **ACCURACY BY SUITE / LEVEL**: the table for the predictions (sequential: multi-agent should not beat single;
  breadth: `parallel` should help only when single-agent accuracy drops, and less against `single_thinking`/`self_consistency`).
- **TOKENS PER ROLE**: which role consumes the overhead (input tokens = context re-reading).
- **v1 vs v2 grader disagreement** (adversarial set): shows how many v1 "correct" labels the strict grader rejects.

## 5. Troubleshooting
| Symptom | Fix |
|---|---|
| `smoke_test` role/template error | add `--no-system-role` (also to `run_experiment`) |
| many `format_error` | the model is not ending with the `FINAL ANSWER: {json}` line. Read `final_text` in the JSONL to see what it writes instead; if it is systematic, report it as a finding for that architecture |
| HTTP 429 / timeouts | lower `--workers`; runner retries 3 times with backoff, then records `status=error` |
| `usage_complete=False` rows | endpoint did not return usage; those runs are excluded from token statistics |
| `recursion_limit` rows | graph hit 40 steps (`RunConfig.recursion_limit`); counts as incorrect, excluded from compute stats |

## 6. What changed vs the notebook
Strict `FINAL ANSWER` JSON grading (v1 accepted any number anywhere); explicit thinking flag and a thinking arm; compute-matched
self-consistency arm; parallel orchestrator arm; system-role prompts; failed runs stored as `None` not 0 and excluded from compute
stats; recursion cap; randomised, seeded order; worker count stored per run; repetitions with task-level analysis; per-role token
accounting; generated tasks with computed, unit-tested ground truth; fixed-overhead model and per-level tables; the v1 grade is stored
beside the v2 grade so grader changes can never be silent again.

## 7. Known limits
The LangGraph builders were exercised against a strict fake of LangChain/LangGraph, not the real packages or a live endpoint.
Use `smoke_test.py` as the real check. `tasks_v1.json` carries the ground truth as it stood in the last notebook state; the
L4 tool-strategy patch (Input 50) could not be replayed offline, so tool-strategy metadata for some L4 tasks may be older than
the notebook's. Answer correctness fields are unaffected.
