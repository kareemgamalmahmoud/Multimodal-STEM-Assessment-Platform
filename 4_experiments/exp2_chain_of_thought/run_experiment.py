"""
Experiment 2 — Chain-of-Thought (CoT) Scoring.

Extends Exp1 by asking the model to reason step-by-step before assigning a score.
Prompt structure: TRANSCRIBE → ANALYZE → SCORE.

Scientific grounding:
  Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
  NeurIPS 2022. https://arxiv.org/abs/2201.11903

Expected improvements over Exp1:
  - Explicit reasoning chain improves score consistency (less hallucinated scores)
  - Error identification provides interpretable feedback

Expected limitations vs Exp3/4:
  - No explicit rubric → criterion weighting is model-internal and uncontrolled
  - Arabic text may still be under-evaluated without language-aware prompting

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


def load_metadata() -> pd.DataFrame:
    meta_path = REPO_ROOT / "data" / "fermat_processed" / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found. Run Stage 1 first.")
    return pd.read_csv(meta_path)


def run(config: dict, limit: int = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata()
    split = config["data"].get("split", "all")
    if split != "all" and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)

    print(f"\n[Exp 2] Chain-of-Thought Scoring")
    print(f"  Samples : {len(df)}")
    print(f"  Model   : {config['model']['id']}")
    print(f"  Output  : {RESULTS_DIR}\n")

    client = QwenVLClient.get_instance(
        model_id=config["model"]["id"],
        use_4bit=config["model"].get("use_4bit", True),
    )

    predictions = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exp2-CoT"):
        result = evaluate_sample(row=row, method="cot", client=client)
        if result:
            predictions.append(result)

    pred_path = RESULTS_DIR / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Metrics
    scored = [p for p in predictions if p.get("human_score") is not None]
    if scored:
        import numpy as np
        from sklearn.metrics import cohen_kappa_score
        pred_scores = [p["total_score"] for p in scored]
        human_scores = [float(p["human_score"]) for p in scored]
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
        summary = {"method": "chain_of_thought", "n": len(scored), "qwk": round(qwk, 4), "pearson_r": round(corr, 4), "rmse": round(rmse, 4)}
    else:
        summary = {"method": "chain_of_thought", "n": len(predictions), "note": "no human scores available"}

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Exp 2] Done. Predictions: {len(predictions)}")
    print(f"  {summary}")
    print(f"  Results → {RESULTS_DIR}\n")


def main():
    parser = argparse.ArgumentParser(description="Exp 2: Chain-of-Thought Scoring")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()
    config = load_config()
    if args.split:
        config["data"]["split"] = args.split
    run(config, limit=args.limit)


if __name__ == "__main__":
    main()
