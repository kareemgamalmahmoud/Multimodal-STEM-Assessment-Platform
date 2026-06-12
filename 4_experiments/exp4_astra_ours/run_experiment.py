"""
Experiment 4 — ASTRA: Adaptive Script-aware Two-stage Rubric Assessment.

Our novel contribution. The full ASTRA pipeline runs as follows:

  Phase A — Calibration (if human scores are available):
    Run Exp 3 (rubric-decomposed) predictions on a calibration split.
    Compute per-language offsets: offset[lang] = mean(human) - mean(llm).

  Phase B — Main evaluation:
    For each sample:
      1. Script-aware transcription (language-conditioned VLM prompt).
      2. Auto-generate rubric (cached from Exp 3 where possible).
      3. Self-consistency voting: score the rubric N=5 times at T=0.7.
      4. Aggregate by median per criterion.
      5. Apply calibration offset.

Scientific grounding:
  [SC]  Wang et al. 2023 — self-consistency voting
  [CAL] Ahuja et al. 2023 — multilingual LLM bias calibration
  [JDG] Zheng et al. 2023 — LLM-as-judge design principles
  [YAV] Yavuz et al. 2024 — rubric-based LLM grading baseline

Run:
    python run_experiment.py [--limit N] [--split SPLIT] [--no-calibration]
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
EXP4_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(EXP4_DIR))

from llm_client import QwenVLClient
from rubric_generator import generate_rubric
from astra_pipeline import run_astra_sample, compute_calibration_offsets

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
        return {"note": "no human scores available for metrics"}
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

    # Per-language breakdown
    lang_metrics = {}
    for lang in set(p.get("detected_language", "unknown") for p in scored):
        lang_preds = [p for p in scored if p.get("detected_language") == lang]
        if not lang_preds:
            continue
        lp = [x["total_score"] for x in lang_preds]
        lh = [float(x["human_score"]) for x in lang_preds]
        lang_metrics[lang] = {
            "n": len(lp),
            "pearson_r": round(float(np.corrcoef(lh, lp)[0, 1]), 4) if len(lp) > 1 else None,
            "rmse": round(float(np.sqrt(np.mean((np.array(lh) - np.array(lp)) ** 2))), 4),
        }

    return {
        "n": len(scored),
        "qwk": round(float(qwk), 4),
        "pearson_r": round(float(np.corrcoef(human_scores, pred_scores)[0, 1]), 4),
        "rmse": round(float(np.sqrt(np.mean((np.array(human_scores) - np.array(pred_scores)) ** 2))), 4),
        "per_language": lang_metrics,
    }


def run(config: dict, limit: int = None, no_calibration: bool = False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata()
    split = config["data"].get("split", "all")
    if split != "all" and "split" in df.columns:
        df = df[df["split"] == split].reset_index(drop=True)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)

    n_votes = config["astra"]["n_votes"]
    vote_temperature = config["model"]["vote_temperature"]
    cal_fraction = config["astra"]["calibration_split_ratio"]
    cal_min = config["astra"]["calibration_min_samples"]

    print(f"\n[Exp 4] ASTRA — Adaptive Script-aware Two-stage Rubric Assessment")
    print(f"  Samples      : {len(df)}")
    print(f"  n_votes      : {n_votes}")
    print(f"  vote_temp    : {vote_temperature}")
    print(f"  calibration  : {'disabled' if no_calibration else f'enabled (fraction={cal_fraction})'}")
    print(f"  Model        : {config['model']['id']}")
    print(f"  Output       : {RESULTS_DIR}\n")

    client = QwenVLClient.get_instance(
        model_id=config["model"]["id"],
        use_4bit=config["model"].get("use_4bit", True),
    )

    # -----------------------------------------------------------------------
    # Phase A: Calibration
    # -----------------------------------------------------------------------
    calibration_offsets = {}
    cal_path = RESULTS_DIR / "calibration_offsets.json"

    if not no_calibration:
        print("[Phase A] Computing calibration offsets ...")
        labeled_df = df[df["human_score"].notna()].copy() if "human_score" in df.columns else pd.DataFrame()

        if not labeled_df.empty:
            # Quick single-pass scoring for calibration split only
            n_cal = max(1, int(len(labeled_df) * cal_fraction))
            cal_df = labeled_df.iloc[:n_cal]

            cal_predictions = []
            for idx, row in tqdm(cal_df.iterrows(), total=len(cal_df), desc="Calibration"):
                image_path = REPO_ROOT / "data" / "fermat_processed" / str(row["image_path"])
                if not image_path.exists():
                    continue
                lang = str(row.get("language_track", "english"))
                rubric = generate_rubric(
                    question=str(row.get("question", "")),
                    reference_answer=str(row.get("reference_answer", "")),
                    client=client,
                    language_track=lang,
                )
                result = run_astra_sample(
                    sample_id=str(row["id"]),
                    image_path=image_path,
                    question=str(row.get("question", "")),
                    reference_answer=str(row.get("reference_answer", "")),
                    language_track=lang,
                    client=client,
                    rubric=rubric,
                    n_votes=1,                # single pass for calibration (speed)
                    vote_temperature=0.1,
                    calibration_offsets={},   # no offset during calibration
                )
                result["human_score"] = row.get("human_score")
                cal_predictions.append(result)

            if cal_predictions:
                cal_preds_df = pd.DataFrame(cal_predictions)
                calibration_offsets = compute_calibration_offsets(
                    cal_preds_df, calibration_fraction=1.0, min_samples=cal_min
                )
        else:
            print("  [Phase A] No labeled samples — calibration skipped.")

        with open(cal_path, "w") as f:
            json.dump(calibration_offsets, f, indent=2)
        print(f"  Calibration offsets: {calibration_offsets}\n")

    # -----------------------------------------------------------------------
    # Phase B: Main evaluation (full dataset)
    # -----------------------------------------------------------------------
    print("[Phase B] ASTRA main evaluation ...")
    predictions = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exp4-ASTRA"):
        image_path = REPO_ROOT / "data" / "fermat_processed" / str(row["image_path"])
        if not image_path.exists():
            continue

        lang = str(row.get("language_track", "english"))

        try:
            rubric = generate_rubric(
                question=str(row.get("question", "")),
                reference_answer=str(row.get("reference_answer", "")),
                client=client,
                language_track=lang,
            )
            result = run_astra_sample(
                sample_id=str(row["id"]),
                image_path=image_path,
                question=str(row.get("question", "")),
                reference_answer=str(row.get("reference_answer", "")),
                language_track=lang,
                client=client,
                rubric=rubric,
                n_votes=n_votes,
                vote_temperature=vote_temperature,
                calibration_offsets=calibration_offsets,
            )
            result["human_score"] = row.get("human_score")
            predictions.append(result)
        except Exception as e:
            print(f"  [ERROR] {row['id']}: {e}")

    # Save predictions
    pred_path = RESULTS_DIR / "predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Metrics
    metrics = compute_metrics(predictions)
    summary = {"method": "astra", "n_votes": n_votes, **metrics}
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Exp 4] Done. Predictions: {len(predictions)}")
    print(f"  Overall QWK    : {metrics.get('qwk', 'N/A')}")
    print(f"  Pearson r      : {metrics.get('pearson_r', 'N/A')}")
    print(f"  RMSE           : {metrics.get('rmse', 'N/A')}")
    if "per_language" in metrics:
        for lang, lm in metrics["per_language"].items():
            print(f"  [{lang}] Pearson r={lm.get('pearson_r')}  RMSE={lm.get('rmse')}  n={lm['n']}")
    print(f"  Results → {RESULTS_DIR}\n")


def main():
    parser = argparse.ArgumentParser(description="Exp 4: ASTRA")
    parser.add_argument("--limit", type=int, default=None, help="Process only N samples")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--no-calibration", action="store_true", help="Disable bias calibration")
    args = parser.parse_args()

    config = load_config()
    if args.split:
        config["data"]["split"] = args.split
    run(config, limit=args.limit, no_calibration=args.no_calibration)


if __name__ == "__main__":
    main()
