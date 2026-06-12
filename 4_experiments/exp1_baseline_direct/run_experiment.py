"""
Experiment 1 — Baseline Direct Scoring.

The simplest possible evaluation: show the handwritten image to Qwen2-VL
and ask it to assign a score in a single model call. No explicit reasoning,
no rubric, no transcript.

This establishes a naive lower-bound baseline. Expected weaknesses:
  - No structured reasoning → inconsistent scores
  - Poor Arabic comprehension without explicit language guidance
  - Score calibration not grounded to rubric criteria

Run:
    python run_experiment.py [--limit N] [--split SPLIT]
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

# Add repo root and pipeline dir to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = REPO_ROOT / "3_evaluation_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from llm_client import QwenVLClient
from evaluator import evaluate_sample

EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_metadata(config: dict) -> pd.DataFrame:
    processed_dir = REPO_ROOT / config["data"]["processed_dir"].lstrip("../../")
    meta_path = REPO_ROOT / "data" / "fermat_processed" / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {meta_path}. Run Stage 1 first."
        )
    df = pd.read_csv(meta_path)

    split = config["data"].get("split", "all")
    if split != "all" and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)

    return df


def run(config: dict, limit: int = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata(config)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)

    print(f"\n[Exp 1] Baseline Direct Scoring")
    print(f"  Samples : {len(df)}")
    print(f"  Model   : {config['model']['id']}")
    print(f"  Output  : {RESULTS_DIR}\n")

    # Load model once (shared singleton)
    client = QwenVLClient.get_instance(
        model_id=config["model"]["id"],
        use_4bit=config["model"].get("use_4bit", True),
    )

    predictions = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exp1-Baseline"):
        result = evaluate_sample(row=row, method="baseline", client=client)
        if result:
            predictions.append(result)

    # Save predictions
    pred_path = RESULTS_DIR / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Quick summary
    scored = [p for p in predictions if p.get("human_score") is not None]
    if scored:
        import numpy as np
        pred_scores = [p["total_score"] for p in scored]
        human_scores = [float(p["human_score"]) for p in scored]
        from sklearn.metrics import cohen_kappa_score
        try:
            qwk = cohen_kappa_score(
                [round(s * 4) for s in human_scores],
                [round(s * 4) for s in pred_scores],
                weights="quadratic",
            )
        except Exception:
            qwk = float("nan")
        corr = float(np.corrcoef(human_scores, pred_scores)[0, 1])
        rmse = float(np.sqrt(np.mean((np.array(human_scores) - np.array(pred_scores)) ** 2)))
        summary = {"method": "baseline", "n": len(scored), "qwk": round(qwk, 4), "pearson_r": round(corr, 4), "rmse": round(rmse, 4)}
    else:
        summary = {"method": "baseline", "n": len(predictions), "note": "no human scores available for metrics"}

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Exp 1] Done. Predictions: {len(predictions)}")
    print(f"  {summary}")
    print(f"  Results → {RESULTS_DIR}\n")


def main():
    parser = argparse.ArgumentParser(description="Exp 1: Baseline Direct Scoring")
    parser.add_argument("--limit", type=int, default=None, help="Process only N samples")
    parser.add_argument("--split", type=str, default=None, help="Dataset split override")
    args = parser.parse_args()

    config = load_config()
    if args.split:
        config["data"]["split"] = args.split

    run(config, limit=args.limit)


if __name__ == "__main__":
    main()
