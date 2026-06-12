# Multimodal STEM Assessment Platform

An end-to-end pipeline for evaluating handwritten STEM answers using Vision-Language Models, with specialized support for **Arabic script** and **English scientific notation**.

---

## Methodology Summary

We propose **ASTRA** (**A**daptive **S**cript-aware **T**wo-stage **R**ubric **A**ssessment), a novel scoring pipeline that extends LLM-based grading (Yavuz et al. 2024) to the multimodal, multilingual setting. ASTRA introduces:
1. **Script-aware routing** — detects Arabic, English, or mixed-script content before scoring
2. **Self-consistency voting** — runs the rubric evaluator N times at temperature > 0 and aggregates via majority vote (Wang et al. 2023)
3. **Bias-calibrated scoring** — applies an empirical per-language offset learned from human scores

---

## Repo Structure

```
.
├── 1_data_preparation/     ← Download FERMAT, EDA, image preprocessing
├── 2_ocr_htr/              ← Arabic HTR (TrOCR) + math OCR (pix2tex) + region detection
├── 3_evaluation_pipeline/  ← Shared rubric generator, Qwen2-VL client, scorer
├── 4_experiments/
│   ├── exp1_baseline_direct/        ← Direct VLM scoring (no reasoning)
│   ├── exp2_chain_of_thought/       ← CoT prompting
│   ├── exp3_rubric_decomposed/      ← Per-criterion rubric scoring
│   └── exp4_astra_ours/             ← ASTRA: our full novel method
├── 5_results_analysis/     ← QWK, Pearson, RMSE, comparison table, plots
├── literature_survey/      ← sources.md: all referenced papers and models
├── docs/                   ← Methodology report + future improvements
├── environment.yml         ← Conda environment
└── requirements.txt        ← pip requirements (for Colab)
```

---

## Quick Start (Google Colab / T4 GPU)

```bash
# 1. Clone repo
git clone https://github.com/<your-username>/Multimodal-STEM-Assessment-Platform.git
cd Multimodal-STEM-Assessment-Platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Stage 1: Download and preprocess data
cd 1_data_preparation
python download_dataset.py
python preprocess.py

# 4. Run Stage 2: OCR / HTR on all images
cd ../2_ocr_htr
python run_ocr.py

# 5. Run Stage 3 is a library — no standalone runner

# 6. Run experiments (start with ASTRA)
cd ../4_experiments/exp4_astra_ours
python run_experiment.py

# 7. Analyze results
cd ../../5_results_analysis
python run_analysis.py
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 8 GB (with 4-bit quant) | 15 GB (Google Colab T4) |
| RAM | 16 GB | 32 GB |
| Disk | 20 GB | 50 GB |

All models are loaded in **4-bit quantization** by default using `bitsandbytes`.

---

## Key Dependencies

- `transformers` >= 4.45 — Qwen2-VL, TrOCR
- `qwen-vl-utils` — Qwen2-VL vision preprocessing
- `bitsandbytes` — 4-bit quantization
- `pix2tex` — LaTeX math OCR
- `datasets` — HuggingFace dataset loading
- `Pillow`, `opencv-python` — image processing
- `scikit-learn` — metrics (QWK, Pearson)

See `environment.yml` or `requirements.txt` for the full list.

---

## Novel Contribution: ASTRA

See [docs/methodology.md](docs/methodology.md) for the full methodology write-up.

The scientific grounding for each ASTRA component:

| Component | Source |
|---|---|
| Rubric-based LLM grading | Yavuz et al. 2024 |
| Self-consistency voting | Wang et al. 2023 (NeurIPS) |
| Bias calibration for multilingual LLMs | Ahuja et al. 2023 (MEGA) |
| LLM-as-judge design | Zheng et al. 2023 (MT-Bench) |

---

## Environment

This project was developed and tested on **Google Colab (free tier, T4 GPU)**. No local GPU is required.

To reproduce the environment on Colab:

```bash
pip install -r requirements.txt
```

The `environment.yml` is also provided for local Conda setups if needed.

---

## Literature Survey

See [literature_survey/sources.md](literature_survey/sources.md) for all referenced papers, models, and datasets with links.
