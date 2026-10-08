"""Optional matplotlib figures (matplotlib only needed if you call these)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def make_plots(df: pd.DataFrame, outdir: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .analysis import by_level_table, cost_efficiency
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    paths = []

    ce = cost_efficiency(df)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for arch, r in ce.iterrows():
        ax.scatter(r.mean_tokens, r.accuracy * 100, s=70)
        ax.annotate(arch, (r.mean_tokens, r.accuracy * 100), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean tokens per run"); ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy vs. cost")
    ax.grid(alpha=.3); fig.tight_layout(); p = out / "pareto_accuracy_vs_tokens.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(p)

    for metric, label in (("correct", "Accuracy"), ("total_tokens", "Mean tokens")):
        t = by_level_table(df, metric)
        for suite, g in t.groupby(level="suite"):
            g = g.droplevel("suite")
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for arch in g.columns:
                ax.plot(g.index, g[arch] * (100 if metric == "correct" else 1), marker="o", label=arch)
            ax.set_xlabel("Difficulty level"); ax.set_ylabel(label + (" (%)" if metric == "correct" else ""))
            ax.set_title(f"{label} by difficulty: {suite}"); ax.grid(alpha=.3); ax.legend(fontsize=7)
            fig.tight_layout(); p = out / f"{metric}_by_level_{suite}.png"; fig.savefig(p, dpi=160); plt.close(fig); paths.append(p)
    return paths
