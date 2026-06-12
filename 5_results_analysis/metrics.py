"""
Metrics computation for STEM assessment scoring evaluation.

Metrics used:
  QWK  — Quadratic Weighted Kappa: standard metric for ordinal agreement tasks
          (used in essay scoring competitions like ASAP/Kaggle)
  r    — Pearson correlation coefficient: linear agreement between LLM and human scores
  RMSE — Root Mean Squared Error: absolute accuracy in score units

All metrics are computed on samples that have both a predicted score and a human score.
Per-language breakdowns are provided for Arabic, English, and Mixed.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """Load a predictions.json file into a DataFrame."""
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def compute_qwk(
    human_scores: list[float],
    pred_scores: list[float],
    n_bins: int = 5,
) -> float:
    """
    Compute Quadratic Weighted Kappa (QWK).

    Scores are binned into n_bins categories before computing kappa.
    QWK = 1 means perfect agreement, 0 = chance, negative = worse than chance.
    """
    if len(human_scores) < 2:
        return float("nan")
    human_binned = [round(s * (n_bins - 1)) for s in human_scores]
    pred_binned = [round(s * (n_bins - 1)) for s in pred_scores]
    try:
        return float(cohen_kappa_score(human_binned, pred_binned, weights="quadratic"))
    except Exception:
        return float("nan")


def compute_pearson(human_scores: list[float], pred_scores: list[float]) -> float:
    """Pearson correlation coefficient between human and predicted scores."""
    if len(human_scores) < 2:
        return float("nan")
    try:
        return float(np.corrcoef(human_scores, pred_scores)[0, 1])
    except Exception:
        return float("nan")


def compute_rmse(human_scores: list[float], pred_scores: list[float]) -> float:
    """Root Mean Squared Error."""
    if not human_scores:
        return float("nan")
    return float(np.sqrt(np.mean((np.array(human_scores) - np.array(pred_scores)) ** 2)))


def compute_all_metrics(df: pd.DataFrame, method_name: str = "") -> dict:
    """
    Compute QWK, Pearson r, and RMSE for a predictions DataFrame.
    Also computes per-language-track breakdowns.

    Args:
        df:          DataFrame with columns: total_score, human_score, detected_language.
        method_name: Label for this experiment (used in output).

    Returns:
        Dict with overall and per-language metrics.
    """
    labeled = df.dropna(subset=["human_score"]).copy()
    labeled["human_score"] = labeled["human_score"].astype(float)
    labeled["total_score"] = labeled["total_score"].astype(float)

    if labeled.empty:
        return {"method": method_name, "n": 0, "note": "no labeled samples"}

    human = labeled["human_score"].tolist()
    pred = labeled["total_score"].tolist()

    overall = {
        "method": method_name,
        "n": len(labeled),
        "qwk": round(compute_qwk(human, pred), 4),
        "pearson_r": round(compute_pearson(human, pred), 4),
        "rmse": round(compute_rmse(human, pred), 4),
        "mean_human": round(float(np.mean(human)), 4),
        "mean_pred": round(float(np.mean(pred)), 4),
        "per_language": {},
    }

    # Per-language breakdown
    lang_col = "detected_language" if "detected_language" in labeled.columns else None
    if lang_col:
        for lang in sorted(labeled[lang_col].dropna().unique()):
            sub = labeled[labeled[lang_col] == lang]
            if len(sub) < 2:
                continue
            lh = sub["human_score"].tolist()
            lp = sub["total_score"].tolist()
            overall["per_language"][lang] = {
                "n": len(sub),
                "qwk": round(compute_qwk(lh, lp), 4),
                "pearson_r": round(compute_pearson(lh, lp), 4),
                "rmse": round(compute_rmse(lh, lp), 4),
            }

    return overall


def load_all_experiment_metrics(experiments_dir: Path) -> list[dict]:
    """
    Walk the 4_experiments/ directory and compute metrics for each experiment
    that has a predictions.json file.
    """
    results = []
    exp_dirs = sorted(experiments_dir.glob("exp*/results/predictions.json"))

    for pred_path in exp_dirs:
        exp_name = pred_path.parent.parent.name
        try:
            df = load_predictions(pred_path)
            metrics = compute_all_metrics(df, method_name=exp_name)
            results.append(metrics)
            print(f"  {exp_name:35s} | QWK={metrics.get('qwk', 'N/A'):<7}  "
                  f"r={metrics.get('pearson_r', 'N/A'):<7}  "
                  f"RMSE={metrics.get('rmse', 'N/A')}")
        except Exception as e:
            print(f"  [ERROR] {exp_name}: {e}")

    return results
