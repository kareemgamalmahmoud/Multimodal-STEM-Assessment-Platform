"""
Experiment 3 — Rubric-Decomposed Scoring.

Two-phase pipeline:
  Phase A: Generate a weighted rubric for each unique question.
  Phase B: Evaluate each rubric criterion independently via the LLM.

Scientific grounding:
  Yavuz et al. (2024). Utilizing LLMs for EFL Essay Grading. arXiv:2501.07244.
  — The paper uses LLMs with rubric-aligned prompts for essay scoring.
  We extend this to multimodal STEM (image input) and multiple criteria.

Also draws on analytic scoring theory (as opposed to holistic scoring):
  Weigle, S.C. (2002). Assessing Writing. Cambridge University Press.
  — Analytic rubrics improve consistency and diagnostic value.

Expected improvement over Exp2:
  - Explicit rubric → each criterion has a defined weight and description
  - Per-criterion calls reduce the burden on a single complex prompt
  - Scores are more interpretable and diagnostic

Expected limitation vs Exp4:
  - Single-pass scoring (no voting) → still susceptible to LLM hallucination variance
  - No Arabic-specific language routing

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
        raise FileNotFoundError("metadata.csv not found. Run Stage 1 first.")
    return pd.read_csv(meta_path)


def compute_metrics(predictions: list) -> dict:
    scored = [p for p in predictions if p.get("human_score") is not None]
    if not scored:
        return {"note": "no human scores available"}
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
    return {
        "n": len(scored),
        "qwk": round(float(qwk), 4),
        "pearson_r": round(float(np.corrcoef(human_scores, pred_scores)[0, 1]), 4),
        "rmse": round(float(np.sqrt(np.mean((np.array(human_scores) - np.array(pred_scores)) ** 2))), 4),
    }


def run(config: dict, limit: int = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata()
    split = config["data"].get("split", "all")
    if split != "all" and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)

    print(f"\n[Exp 3] Rubric-Decomposed Scoring")
    print(f"  Samples  : {len(df)}")
    print(f"  Model    : {config['model']['id']}")
    print(f"  Output   : {RESULTS_DIR}\n")

    client = QwenVLClient.get_instance(
        model_id=config["model"]["id"],
        use_4bit=config["model"].get("use_4bit", True),
    )

    predictions = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exp3-Rubric"):
        result = evaluate_sample(row=row, method="rubric_decomposed", client=client)
        if result:
            predictions.append(result)

    pred_path = RESULTS_DIR / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    metrics = compute_metrics(predictions)
    summary = {"method": "rubric_decomposed", **metrics}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Exp 3] Done. Predictions: {len(predictions)}")
    print(f"  {summary}")
    print(f"  Results → {RESULTS_DIR}\n")


def main():
    parser = argparse.ArgumentParser(description="Exp 3: Rubric-Decomposed Scoring")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()
    config = load_config()
    if args.split:
        config["data"]["split"] = args.split
    run(config, limit=args.limit)


if __name__ == "__main__":
    main()
