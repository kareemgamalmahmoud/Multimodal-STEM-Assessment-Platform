# Stage 4 — Experiments

Four experiments comparing different scoring methodologies. Each is self-contained and produces a `results/predictions.json` file.

## Experiment Overview

| Exp | Method | Key Idea | Scientific Grounding |
|-----|--------|----------|----------------------|
| 1 | **Baseline** | Single VLM call, no reasoning | Establishes a naive floor |
| 2 | **Chain-of-Thought** | Transcribe → reason → score | Wei et al. 2022 (NeurIPS) |
| 3 | **Rubric-Decomposed** | Auto-rubric, per-criterion scoring | Yavuz et al. 2024 |
| 4 | **ASTRA** (ours) | Script routing + self-consistency voting + bias calibration | Wang et al. 2023 + Ahuja et al. 2023 |

## Run Order

Experiments 1–4 can run in any order. ASTRA (4) uses calibration computed from experiments 3, so to get calibrated ASTRA results run experiment 3 first, then 4.

```bash
# From repo root:
cd 4_experiments

python exp1_baseline_direct/run_experiment.py
python exp2_chain_of_thought/run_experiment.py
python exp3_rubric_decomposed/run_experiment.py
python exp4_astra_ours/run_experiment.py
```

## Common Arguments

All experiment runners accept:
```
--limit N       Process only first N samples (for quick testing)
--split train   Which split to evaluate ('train', 'test', 'validation', 'demo')
```

## Output Structure

Each experiment writes to its own `results/` directory:
```
exp<N>_<name>/results/
  predictions.json   ← list of ScoringResult dicts
  summary.json       ← aggregate metrics
```
