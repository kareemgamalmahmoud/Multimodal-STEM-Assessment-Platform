"""
Visualization plots for the results analysis.

Generates:
  1. score_distributions.png  — side-by-side histograms of predicted scores per experiment
  2. calibration_curve.png    — ASTRA pre- vs post-calibration scatter vs human scores
  3. language_breakdown.png   — bar chart of QWK per language per experiment
  4. self_consistency_var.png — variance of ASTRA votes per sample (Exp 4 only)
"""

import json
from pathlib import Path

import numpy as np


EXPERIMENT_COLORS = {
    "exp1_baseline_direct": "#e05c5c",
    "exp2_chain_of_thought": "#f0a500",
    "exp3_rubric_decomposed": "#4a90d9",
    "exp4_astra_ours": "#2ecc71",
}

DISPLAY_NAMES = {
    "exp1_baseline_direct": "Baseline",
    "exp2_chain_of_thought": "Chain-of-Thought",
    "exp3_rubric_decomposed": "Rubric-Decomposed",
    "exp4_astra_ours": "ASTRA (Ours)",
}


def load_predictions(pred_path: Path) -> list[dict]:
    with open(pred_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_score_distributions(all_preds: dict[str, list[dict]], output_dir: Path):
    """Side-by-side histograms comparing predicted score distributions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(all_preds)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, (exp_key, preds) in zip(axes, all_preds.items()):
            pred_scores = [p["total_score"] for p in preds]
            human_scores = [float(p["human_score"]) for p in preds if p.get("human_score") is not None]
            color = EXPERIMENT_COLORS.get(exp_key, "#888")
            ax.hist(pred_scores, bins=20, alpha=0.7, color=color, label="LLM Score", edgecolor="white")
            if human_scores:
                ax.hist(human_scores, bins=20, alpha=0.5, color="#333", label="Human Score", edgecolor="white")
            ax.set_title(DISPLAY_NAMES.get(exp_key, exp_key), fontsize=11)
            ax.set_xlabel("Score", fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel("Count", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Predicted vs Human Score Distributions by Method", fontsize=13, y=1.02)
        fig.tight_layout()
        out_path = output_dir / "score_distributions.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  [SKIP] matplotlib not available — skipping score_distributions.png")


def plot_calibration_curve(astra_preds: list[dict], output_dir: Path):
    """
    Scatter plot: human scores vs ASTRA scores (pre and post calibration).
    Illustrates the effectiveness of the bias calibration step.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labeled = [p for p in astra_preds if p.get("human_score") is not None]
        if not labeled:
            print("  [SKIP] No labeled samples for calibration curve.")
            return

        human = np.array([float(p["human_score"]) for p in labeled])
        post_cal = np.array([p["total_score"] for p in labeled])
        pre_cal = np.array([p.get("pre_calibration_score", p["total_score"]) for p in labeled])

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, scores, title, color in zip(
            axes,
            [pre_cal, post_cal],
            ["Pre-Calibration", "Post-Calibration (ASTRA)"],
            ["#e05c5c", "#2ecc71"],
        ):
            ax.scatter(human, scores, alpha=0.5, s=25, color=color)
            lims = [0, 1]
            ax.plot(lims, lims, "k--", alpha=0.4, label="Perfect agreement")
            ax.set_xlabel("Human Score", fontsize=11)
            ax.set_ylabel("LLM Score", fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            corr = np.corrcoef(human, scores)[0, 1]
            ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.7))
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        fig.suptitle("ASTRA Calibration Effect", fontsize=13)
        fig.tight_layout()
        out_path = output_dir / "calibration_curve.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  [SKIP] matplotlib not available")


def plot_language_breakdown(all_metrics: list[dict], output_dir: Path):
    """Bar chart of Pearson r per language track per experiment."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        langs = set()
        for m in all_metrics:
            langs.update(m.get("per_language", {}).keys())
        langs = sorted(langs)

        if not langs:
            print("  [SKIP] No per-language metrics to plot.")
            return

        x = np.arange(len(langs))
        width = 0.8 / max(len(all_metrics), 1)

        fig, ax = plt.subplots(figsize=(max(8, 2 * len(langs)), 5))
        for i, m in enumerate(all_metrics):
            exp_key = m.get("method", f"exp{i+1}")
            color = EXPERIMENT_COLORS.get(exp_key, f"C{i}")
            values = [m.get("per_language", {}).get(lang, {}).get("pearson_r", float("nan")) for lang in langs]
            # Replace NaN with 0 for plotting
            plot_values = [v if not np.isnan(v) else 0.0 for v in values]
            bars = ax.bar(x + i * width, plot_values, width * 0.9, label=DISPLAY_NAMES.get(exp_key, exp_key), color=color, alpha=0.85)

        ax.set_xticks(x + width * (len(all_metrics) - 1) / 2)
        ax.set_xticklabels([l.capitalize() for l in langs], fontsize=11)
        ax.set_ylabel("Pearson r", fontsize=11)
        ax.set_title("Scoring Agreement by Language Track", fontsize=13)
        ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = output_dir / "language_breakdown.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  [SKIP] matplotlib not available")


def plot_self_consistency_variance(astra_preds: list[dict], output_dir: Path):
    """
    Histogram of vote score variance across ASTRA's N voting passes.
    Low variance → self-consistency achieved. High variance → uncertain sample.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        variances = []
        for p in astra_preds:
            votes = p.get("raw_votes", [])
            if len(votes) > 1:
                variances.append(float(np.var(votes)))

        if not variances:
            print("  [SKIP] No raw_votes data for self-consistency plot.")
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(variances, bins=20, color="#4a90d9", edgecolor="white", alpha=0.85)
        ax.axvline(float(np.mean(variances)), color="red", linestyle="--", label=f"Mean variance = {np.mean(variances):.4f}")
        ax.set_xlabel("Score Variance Across Votes", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("ASTRA Self-Consistency: Vote Variance Distribution", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = output_dir / "self_consistency_variance.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  [SKIP] matplotlib not available")
