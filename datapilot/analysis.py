"""Statistics for paired architecture comparisons (numpy/pandas only).

Design rules (Miller 2024, "Adding Error Bars to Evals"): analyse paired per-task differences, treat the
task (not the run) as the unit of analysis, and report intervals plus the effect size you could have detected.
"""
from __future__ import annotations
from math import comb, ceil, sqrt
from statistics import NormalDist
import json
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- intervals / tests

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Valid for independent runs only (use task-level bootstrap with repeats)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / d, (centre + half) / d)


def bootstrap_ci(values, n_boot: int = 10_000, conf: float = 0.95, seed: int = 42) -> tuple[float, float]:
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    a = (1 - conf) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def signflip_pvalue(diffs, n_mc: int = 100_000, seed: int = 0) -> float:
    """Two-sided paired permutation test on the mean difference (exact when n <= 16)."""
    d = np.asarray(diffs, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    obs = abs(d.sum())
    if n <= 16:
        signs = (((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1)
        stats = np.abs(signs @ d)
    else:
        rng = np.random.default_rng(seed)
        stats = np.abs((rng.integers(0, 2, size=(n_mc, n)) * 2 - 1) @ d)
    return float((stats >= obs - 1e-12).mean())


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p from discordant counts (a right/b wrong, a wrong/b right)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


# ----------------------------------------------------------------------------- power

def tasks_needed(delta: float, reverse_discordance: float = 0.02, alpha: float = 0.05, power: float = 0.8) -> int:
    """Paired tasks needed to detect an accuracy gain `delta` (McNemar normal approximation)."""
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return ceil(z * z * (delta + 2 * reverse_discordance) / delta ** 2)


def min_detectable_gain(n_tasks: int, reverse_discordance: float = 0.02, **kw) -> float:
    for d in np.arange(0.01, 1.0, 0.01):
        if tasks_needed(float(d), reverse_discordance, **kw) <= n_tasks:
            return round(float(d), 2)
    return float("nan")


# ----------------------------------------------------------------------------- data

def load_results(path) -> pd.DataFrame:
    with open(path) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    df = pd.DataFrame(rows)
    complete = df["usage_complete"].fillna(False).astype(bool) if "usage_complete" in df else True
    df["compute_valid"] = df["status"].isin(["ok", "format_error"]) & df["total_tokens"].notna() & complete
    return df


def _task_matrix(df: pd.DataFrame, metric: str, only_valid: bool) -> pd.DataFrame:
    d = df[df["compute_valid"]] if only_valid else df
    d = d.assign(**{metric: d[metric].astype(float)})
    return d.groupby(["task_id", "architecture"])[metric].mean().unstack("architecture")


# ----------------------------------------------------------------------------- tables

def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Intention-to-treat accuracy: errors/timeouts/format failures count as incorrect."""
    mat = _task_matrix(df, "correct", only_valid=False)
    rows = []
    for arch in mat.columns:
        per_task = mat[arch].dropna()
        lo, hi = bootstrap_ci(per_task)
        runs = df[df.architecture == arch]
        k, n = int(runs.correct.sum()), len(runs)
        wl, wh = wilson(k, n)
        rows.append({"architecture": arch, "tasks": len(per_task), "runs": n, "correct": k,
                     "accuracy": per_task.mean(), "task_boot_lo": lo, "task_boot_hi": hi,
                     "wilson_lo(runs)": wl, "wilson_hi(runs)": wh,
                     "format_errors": int((runs.status == "format_error").sum()),
                     "failed_runs": int(runs.status.isin(["error", "recursion_limit"]).sum())})
    return pd.DataFrame(rows).set_index("architecture")


def paired_difference(df: pd.DataFrame, metric: str, a: str, b: str, only_valid: bool = True) -> dict:
    """Mean of (a - b) over tasks present for both architectures, with CI, p-value and dz."""
    mat = _task_matrix(df, metric, only_valid).dropna(subset=[a, b])
    diff = (mat[a] - mat[b]).to_numpy()
    lo, hi = bootstrap_ci(diff)
    sd = diff.std(ddof=1) if len(diff) > 1 else float("nan")
    return {"metric": metric, "a": a, "b": b, "n_tasks": len(diff), "mean_diff": diff.mean(),
            "ci_lo": lo, "ci_hi": hi, "p_signflip": signflip_pvalue(diff),
            "dz": diff.mean() / sd if sd and sd > 0 else float("nan"),
            "ratio_of_means": mat[a].mean() / mat[b].mean() if mat[b].mean() else float("nan")}


def accuracy_mcnemar(df: pd.DataFrame, a: str, b: str) -> dict:
    mat = _task_matrix(df, "correct", only_valid=False).dropna(subset=[a, b]) >= 0.5
    only_a = int((mat[a] & ~mat[b]).sum())
    only_b = int((~mat[a] & mat[b]).sum())
    return {"a": a, "b": b, "n_tasks": len(mat), "only_a_correct": only_a, "only_b_correct": only_b,
            "p_mcnemar_exact": mcnemar_exact(only_a, only_b)}


def cost_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    v = df[df["compute_valid"]]
    g = v.groupby("architecture")
    out = pd.DataFrame({"mean_tokens": g.total_tokens.mean(), "mean_latency_s": g.latency_s.mean(),
                        "mean_llm_calls": g.llm_calls.mean(), "mean_tool_calls": g.tool_calls.mean()})
    acc = df.groupby("architecture").correct.mean()
    out["accuracy"] = acc
    n_correct = df.groupby("architecture").correct.sum()
    out["tokens_per_correct"] = v.groupby("architecture").total_tokens.sum() / n_correct.replace(0, np.nan)
    return out


def role_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Where do the tokens go? mean input/output tokens per role per architecture."""
    rows = []
    for _, r in df[df["compute_valid"]].iterrows():
        for node, u in (r.get("usage_by_node") or {}).items():
            rows.append({"architecture": r["architecture"], "node": node, **u})
    if not rows:
        return pd.DataFrame()
    n_runs = df[df["compute_valid"]].groupby("architecture").size()
    t = pd.DataFrame(rows).groupby(["architecture", "node"])[["calls", "input", "output", "total"]].sum()
    return t.div(n_runs.reindex(t.index.get_level_values(0)).to_numpy(), axis=0).round(1)


def latency_diagnostics(df: pd.DataFrame) -> dict:
    v = df[df["compute_valid"]]
    rho = v["total_tokens"].rank().corr(v["latency_s"].rank())
    tps = (v.total_tokens / v.latency_s).groupby(v.architecture).median().round(1).to_dict()
    return {"spearman_tokens_latency": round(float(rho), 2), "median_tokens_per_sec": tps,
            "max_workers_used": int(v["workers"].max()),
            "warning": "workers>1: latency includes endpoint queueing; rerun latency with workers=1"
            if v["workers"].max() > 1 else None}


def ceiling_warning(df: pd.DataFrame, baseline: str = "single", threshold: float = 0.95) -> str | None:
    acc = df[df.architecture == baseline].correct.mean()
    if acc >= threshold:
        return (f"Baseline accuracy {acc:.0%} >= {threshold:.0%}: accuracy comparisons are uninformative; "
                "raise task difficulty (generators) or use a weaker model.")
    return None


def report(df: pd.DataFrame, baseline: str = "single") -> str:
    out = ["== ACCURACY (intention-to-treat) ==", accuracy_table(df).round(3).to_string(), ""]
    if (w := ceiling_warning(df, baseline)):
        out += ["WARNING: " + w, ""]
    out += ["== COST ==", cost_efficiency(df).round(2).to_string(), ""]
    others = [a for a in df.architecture.unique() if a != baseline]
    for a in others:
        for m in ("total_tokens", "latency_s"):
            r = paired_difference(df, m, a, baseline)
            out.append(f"{a} - {baseline} | {m}: {r['mean_diff']:+.1f} [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] "
                       f"p={r['p_signflip']:.4f} dz={r['dz']:.2f} n_tasks={r['n_tasks']}")
        mc = accuracy_mcnemar(df, a, baseline)
        out.append(f"{a} vs {baseline} | accuracy: only_{a}={mc['only_a_correct']} only_{baseline}={mc['only_b_correct']} "
                   f"p={mc['p_mcnemar_exact']:.3f}")
    n = df.task_id.nunique()
    mdg = min_detectable_gain(n)
    mdg_txt = f"about {mdg:.0%}" if mdg == mdg else "not computable (sample too small to detect any gain below 100 points)"
    out += ["", f"== POWER == with {n} tasks the smallest accuracy gain detectable (80% power) is {mdg_txt}", "", "== LATENCY DIAGNOSTICS ==", str(latency_diagnostics(df))]
    rb = role_breakdown(df)
    if len(rb):
        out += ["", "== TOKENS PER ROLE (mean per run) ==", rb.to_string()]
    return "\n".join(out)


# ----------------------------------------------------------------------------- v2 additions

def fit_overhead(df: pd.DataFrame, arch: str, baseline: str = "single") -> dict:
    """Fixed-plus-proportional cost model: tokens_arch = intercept + slope * tokens_baseline (per task).

    intercept ~ fixed orchestration tax; slope ~ 1 means 'baseline work plus a constant'.
    Also reports Spearman(baseline tokens, ratio) to show whether relative overhead shrinks with task cost.
    """
    mat = _task_matrix(df, "total_tokens", True)
    if arch not in mat.columns or baseline not in mat.columns:
        return {"arch": arch, "n_tasks": 0, "note": "architecture or baseline missing from data"}
    mat = mat.dropna(subset=[arch, baseline])
    x, y = mat[baseline].to_numpy(float), mat[arch].to_numpy(float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or y.var() == 0:
        return {"arch": arch, "n_tasks": n, "note": "too few tasks or no variance for a fit"}
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else float("nan")
    se_slope = sqrt(resid.var(ddof=2) / ((x - x.mean()) ** 2).sum()) if n > 2 else float("nan")
    ratio = y / x
    rho = pd.Series(x).rank().corr(pd.Series(ratio).rank())
    return {"arch": arch, "n_tasks": n, "intercept": float(intercept), "slope": float(slope),
            "slope_ci95_halfwidth": float(1.96 * se_slope), "r2": float(r2),
            "spearman_baseline_vs_ratio": float(rho), "median_ratio": float(np.median(ratio))}


def by_level_table(df: pd.DataFrame, metric: str = "correct") -> pd.DataFrame:
    """Mean metric per (suite, level, architecture), averaged over tasks then repetitions."""
    d = df if metric == "correct" else df[df["compute_valid"]]
    per_task = d.groupby(["suite", "level", "task_id", "architecture"])[metric].mean().reset_index()
    return per_task.groupby(["suite", "level", "architecture"])[metric].mean().unstack("architecture")


def estimate_tokens(archs: list[str], n_runs_per_arch: int) -> dict:
    """Very rough budget using the v1 means (12 adversarial tasks): single 2.4k, reflection 4.1k, multi 5.5k."""
    base = {"single": 2439, "reflection": 4055, "multi_agent": 5528}
    base["self_consistency"] = 3 * base["single"]
    out = {a: base[a] * n_runs_per_arch for a in archs if a in base}
    return out


_orig_report = report


def report(df: pd.DataFrame, baseline: str = "single") -> str:  # noqa: F811  (extends the v1 report)
    out = [_orig_report(df, baseline)]
    for a in [x for x in df.architecture.unique() if x != baseline]:
        f = fit_overhead(df, a, baseline)
        if "slope" in f:
            out.append(f"overhead model {a}: tokens = {f['intercept']:.0f} + {f['slope']:.2f} x {baseline}_tokens "
                       f"(R2={f['r2']:.2f}, slope +/-{f['slope_ci95_halfwidth']:.2f}); "
                       f"spearman(baseline, ratio)={f['spearman_baseline_vs_ratio']:.2f}; median ratio {f['median_ratio']:.2f}")
    if df["suite"].nunique() > 1 or df["level"].nunique() > 1:
        out += ["", "== ACCURACY BY SUITE / LEVEL ==", by_level_table(df).round(2).to_string()]
    return "\n".join(out)
