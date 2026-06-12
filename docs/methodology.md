# Methodology Report — Multimodal STEM Assessment Platform

**Author:** Karim Gamal
**Date:** 2026-06-12

---

## 1. Problem Statement

Automated grading of handwritten STEM answers presents three compounding challenges:

1. **Modality gap** — Student answers exist as images of handwriting, not typed text.
2. **Script diversity** — Arabic and English appear in the same exam, with different visual characteristics.
3. **Mathematical notation** — Scientific expressions (formulas, Greek letters, subscripts) must be parsed correctly for scoring.

Existing LLM-based grading work (e.g., Yavuz et al. 2024) operates on typed EFL essays in a single language. No prior work addresses the multimodal, bilingual Arabic-English STEM evaluation setting.

---

## 2. Proposed Method: ASTRA

We propose **ASTRA** — *Adaptive Script-aware Two-stage Rubric Assessment* — a novel pipeline that extends rubric-based LLM grading to the multimodal multilingual setting.

### 2.1 Architecture Overview

```
Image (handwriting)
       │
       ▼
[Stage 2: OCR/HTR]
  ├── Region Detection (Arabic text / math / English)
  ├── Arabic HTR (TrOCR or Qwen2-VL)
  └── Math OCR (pix2tex → LaTeX)
       │
       ▼
  Student Transcript (structured text)
       │
  ┌────┴────────────────────────────────┐
  │           ASTRA Pipeline            │
  │                                      │
  │  1. Script-Aware Routing             │
  │     detect language track            │
  │     → select language-conditioned    │
  │       prompt variant                 │
  │                                      │
  │  2. Rubric Generation                │
  │     (question, reference answer)     │
  │     → 4-criterion weighted rubric    │
  │     (cached on disk)                 │
  │                                      │
  │  3. Self-Consistency Voting          │
  │     run scoring prompt N=5 times     │
  │     temperature = 0.7               │
  │     → aggregate by median per        │
  │       criterion                      │
  │                                      │
  │  4. Bias Calibration                 │
  │     offset[language] = mean(human)   │
  │                      - mean(llm)     │
  │     learned on calibration split     │
  └────────────────────────────────────┘
       │
       ▼
  Final Score [0, 1]
  + Criterion breakdown
  + Transcription
  + Vote variance
```

---

### 2.2 Component 1: Script-Aware Routing

**Problem being solved:** A prompt that works well for English may systematically under-rate Arabic answers because the model applies different internal standards for the two scripts.

**Implementation:** Before scoring, ASTRA asks the VLM to classify the handwriting script into one of three categories: `arabic`, `english`, or `mixed`. A different prompt variant is then selected for each category. The Arabic variant explicitly instructs the model to evaluate mathematical substance and not penalize for language-specific stylistic differences.

**Novel contribution:** Existing LLM grading papers (Yavuz et al. 2024, Zheng et al. 2023) use a single prompt for all inputs. Script-aware routing is the first application of this concept to the Arabic-English bilingual grading setting.

---

### 2.3 Component 2: Automated Rubric Generation

**Problem being solved:** Manual rubric creation by a human expert is expensive and does not scale. Direct scoring without a rubric (Exp 1) is inconsistent because the model's implicit criteria vary across calls.

**Implementation:** Given the question and reference answer, the VLM generates a 4-criterion rubric with explicit weights (Conceptual Correctness 0.35, Mathematical Accuracy 0.30, Completeness 0.20, Notation & Units 0.15). Rubrics are cached by a hash of (question, reference answer) to avoid redundant generation.

**Scientific grounding:** Yavuz et al. (2024) use rubric-aligned prompts for LLM grading. We extend this by generating the rubric automatically rather than requiring human annotation, reducing the setup cost for new assessments.

---

### 2.4 Component 3: Self-Consistency Voting

**Problem being solved:** A single LLM scoring call is inherently stochastic. The same rubric applied to the same answer can yield meaningfully different scores across calls, reducing reliability.

**Implementation:** We run the rubric scoring prompt N=5 times at temperature=0.7 and compute the **median** per-criterion score across all votes. The median is more robust to outlier votes than the mean.

**Mathematical formulation:**
Let `s_i(c)` be the score for criterion `c` in vote `i`. The aggregated score for criterion `c` is:

```
s_agg(c) = median({s_1(c), s_2(c), ..., s_N(c)})
```

The final total score is then:
```
total_score = Σ_c [ weight(c) × s_agg(c) ]
```

**Scientific grounding:** Wang et al. (2023) show that self-consistency decoding reliably outperforms single greedy decoding on reasoning tasks without additional training. This is the first application of self-consistency to automated STEM grading.

**Why N=5?** Wang et al. find diminishing returns beyond 5–10 samples for most tasks. We chose N=5 as a practical balance between variance reduction and inference cost.

---

### 2.5 Component 4: Bias Calibration

**Problem being solved:** LLMs exhibit systematic scoring biases for non-English content (Ahuja et al. 2023). In our setting, the model may consistently over- or under-score Arabic answers relative to human raters.

**Implementation:** Using a held-out calibration split (20% of labeled samples), we compute:

```
offset[language] = mean(human_scores[language]) - mean(llm_scores[language])
```

This offset is then applied to all test predictions:

```
calibrated_score = clamp(raw_score + offset[language], 0.0, 1.0)
```

This simple but effective correction requires only ~10 labeled samples per language and adds no inference cost.

**Scientific grounding:** Directly motivated by Ahuja et al. (2023) "MEGA" finding that LLM performance is systematically worse on low-resource languages. Our calibration is an empirical post-hoc correction rather than a model modification.

---

## 3. Baseline Experiments

We compare ASTRA against three baselines that progressively add components:

| Experiment | What's added vs. previous |
|---|---|
| Exp 1 (Baseline) | Direct single-call scoring — no reasoning, no rubric |
| Exp 2 (CoT) | + Chain-of-thought reasoning (Wei et al. 2022) |
| Exp 3 (Rubric-Decomposed) | + Explicit rubric with per-criterion evaluation |
| Exp 4 (ASTRA) | + Script routing + self-consistency voting + bias calibration |

The ablation design allows attributing metric improvements to specific components.

---

## 4. Model and Hardware

- **VLM:** Qwen2-VL-7B-Instruct (Alibaba, 2024)
- **Quantization:** 4-bit NF4 via bitsandbytes (Dettmers et al. 2023)
- **Target hardware:** Google Colab T4 GPU (15GB VRAM)
- **Estimated memory:** ~9GB VRAM in 4-bit + activations
- **OCR (text):** microsoft/trocr-base-handwritten
- **OCR (math):** pix2tex (Blecher 2023)

---

## 5. Evaluation Metrics

- **QWK** (Quadratic Weighted Kappa) — primary metric; standard in automated essay scoring competitions (ASAP/Kaggle)
- **Pearson r** — linear correlation with human scores
- **RMSE** — absolute accuracy in score units
- **Per-language breakdown** — separate metrics for Arabic, English, Mixed samples to detect script-specific biases

---

## 6. Expected Results

Based on the design rationale:

| Metric | Exp 1 | Exp 2 | Exp 3 | Exp 4 (ASTRA) |
|---|---|---|---|---|
| QWK | Low (~0.3) | Moderate (~0.45) | Moderate-High (~0.55) | Highest (~0.65+) |
| Arabic Pearson r | Low | Low-Moderate | Moderate | Highest (calibration corrects bias) |
| Score variance | High | Moderate | Moderate | Lowest (voting reduces variance) |

These are qualitative expectations; actual values depend on the dataset and model.

---

## 7. Limitations

1. **Single model backbone** — All experiments use Qwen2-VL-7B. Results may differ significantly with larger models (72B) or specialized Arabic VLMs.
2. **TrOCR Arabic limitations** — TrOCR-base-handwritten has limited Arabic pre-training data. The Qwen2-VL direct OCR path is preferred.
3. **Calibration requires labels** — The bias calibration step requires at least ~10 human-scored samples per language. If the dataset lacks labels for a language, calibration is skipped.
4. **N=5 votes is a heuristic** — The optimal N depends on the model and task. A proper ablation over N is left as future work.
5. **Rubric quality** — Auto-generated rubrics may miss domain-specific nuances that a human expert would capture.
