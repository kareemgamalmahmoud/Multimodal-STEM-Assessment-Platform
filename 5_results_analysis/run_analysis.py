"""
Stage 5 — Results Analysis Runner.

Aggregates predictions from all four experiments and produces:
  - comparison_table.txt (ASCII + LaTeX)
  - metrics_all_experiments.json
  - score_distributions.png
  - calibration_curve.png
  - language_breakdown.png
  - self_consistency_variance.png

Run from the 5_results_analysis/ directory:
    python run_analysis.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "4_experiments"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Add this directory to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import compute_all_metrics
from comparison_table import generate_tables
from visualizations import (
    plot_score_distributions,
    plot_calibration_curve,
    plot_language_breakdown,
    plot_self_consistency_variance,
)

EXPERIMENT_DIRS = [
    ("exp1_baseline_direct", "exp1_baseline_direct"),
    ("exp2_chain_of_thought", "exp2_chain_of_thought"),
    ("exp3_rubric_decomposed", "exp3_rubric_decomposed"),
    ("exp4_astra_ours", "exp4_astra_ours"),
]


def main():
    print("=" * 60)
    print("Stage 5 — Results Analysis")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_preds = {}
    astra_preds = []

    print("\n[INFO] Loading experiment predictions ...\n")
    for exp_dir_name, exp_key in EXPERIMENT_DIRS:
        pred_path = EXPERIMENTS_DIR / exp_dir_name / "results" / "predictions.json"

        if not pred_path.exists():
            print(f"  [SKIP] {exp_dir_name}: predictions.json not found (run the experiment first)")
            continue

        try:
            # load_predictions returns a DataFrame (for metrics); also keep the raw
            # list for visualization functions that expect list[dict].
            import pandas as pd
            with open(pred_path, "r", encoding="utf-8") as fh:
                preds_list = json.load(fh)

            df = pd.DataFrame(preds_list)

            metrics = compute_all_metrics(df, method_name=exp_dir_name)
            all_metrics.append(metrics)
            all_preds[exp_dir_name] = preds_list   # list[dict] for plots

            if "astra" in exp_dir_name:
                astra_preds = preds_list

            print(f"  {exp_dir_name:<35} | n={metrics.get('n', 0):<5} | "
                  f"QWK={str(metrics.get('qwk', 'N/A')):<7} | "
                  f"r={str(metrics.get('pearson_r', 'N/A')):<7} | "
                  f"RMSE={str(metrics.get('rmse', 'N/A'))}")

        except Exception as e:
            print(f"  [ERROR] {exp_dir_name}: {e}")

    if not all_metrics:
        print("\n[WARN] No experiment results found. Run at least one experiment first.")
        print("       Example: cd ../4_experiments/exp1_baseline_direct && python run_experiment.py --limit 5")
        return

    print(f"\n[INFO] Generating outputs ...\n")

    # Tables
    generate_tables(all_metrics, OUTPUT_DIR)

    # Plots
    if all_preds:
        plot_score_distributions(all_preds, OUTPUT_DIR)
        plot_language_breakdown(all_metrics, OUTPUT_DIR)

    if astra_preds:
        plot_calibration_curve(astra_preds, OUTPUT_DIR)
        plot_self_consistency_variance(astra_preds, OUTPUT_DIR)

    print(f"\n[DONE] All outputs saved to: {OUTPUT_DIR}")
    print("\nStage 5 — Analysis complete.\n")


if __name__ == "__main__":
    main()
