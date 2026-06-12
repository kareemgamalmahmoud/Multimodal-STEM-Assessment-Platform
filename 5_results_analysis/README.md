# Stage 5 — Results Analysis

Loads predictions from all four experiments and produces:
- Aggregate metrics table (QWK, Pearson r, RMSE)
- Per-language breakdown (Arabic vs English vs Mixed)
- Score distribution plots
- Calibration curve plots

## Run

```bash
python run_analysis.py
```

## Outputs

`output/`
- `comparison_table.txt` — ASCII table + LaTeX source
- `metrics_all_experiments.json` — machine-readable metrics
- `score_distributions.png` — side-by-side score histograms
- `calibration_curve.png` — ASTRA pre/post calibration comparison
- `language_breakdown.png` — metric breakdown by language track
