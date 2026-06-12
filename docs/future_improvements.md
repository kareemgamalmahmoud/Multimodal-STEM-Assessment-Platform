# Future Improvements

This document outlines potential extensions and improvements to the ASTRA pipeline, organized by priority and effort level.

---

## High Priority (Directly Impact Metrics)

### 1. Fine-tune Qwen2-VL on Arabic HTR Data
**Why:** The base Qwen2-VL-7B is pre-trained on multilingual data but has limited exposure to Arabic handwriting specifically.
**How:** Use LoRA (Low-Rank Adaptation) fine-tuning on Arabic handwriting datasets:
  - KHATT (King Fahad Arabic Handwriting Dataset)
  - IFN/ENIT (Arabic word database)
  - HACDB (Handwritten Arabic Characters Database)
**Expected gain:** Significant improvement in Arabic transcript quality → better scoring.
**Effort:** Medium — fine-tuning a 7B model with LoRA takes ~6–12 hours on a T4 GPU.

---

### 2. Optimize N for Self-Consistency Voting
**Why:** We fixed N=5 as a heuristic. The optimal N balances variance reduction against inference cost.
**How:** Run an ablation: N ∈ {1, 3, 5, 7, 10}. Plot QWK vs inference time. Choose the knee point.
**Expected gain:** Either better metrics at same cost, or same metrics at lower cost.
**Effort:** Low — reuse existing code, just vary config.

---

### 3. Adaptive Calibration (Per-Question-Type)
**Why:** Current calibration is per-language-track. Different STEM domains (algebra, physics, chemistry) may need different offsets.
**How:** Extend `compute_calibration_offsets()` to group by (language, domain_tag). Requires domain labels in the dataset.
**Effort:** Low — code change is small; challenge is getting domain labels.

---

## Medium Priority (Architectural Improvements)

### 4. Symbolic Math Verification
**Why:** For exactly correct mathematical answers (e.g., x = 4), a symbolic checker can give a binary correct/incorrect signal without relying on the LLM's reasoning.
**How:** Parse the LaTeX transcript from pix2tex, use `sympy` to symbolically evaluate and compare to the reference answer.
**Expected gain:** Eliminates LLM hallucination for purely mathematical sub-questions.
**Effort:** Medium — LaTeX → SymPy parsing is non-trivial for complex expressions.

### 5. Multi-Model Ensemble
**Why:** Different VLMs have different strengths (Qwen2-VL better at Arabic, InternVL2 better at scientific diagrams).
**How:** Run rubric scoring with 2–3 different models and merge scores via weighted average or learned combination.
**Effort:** High — requires running multiple large models.

### 6. Uncertainty-Aware Scoring
**Why:** The vote variance from self-consistency encodes the model's uncertainty. High-variance samples should be flagged for human review rather than reported at face value.
**How:** Compute vote variance per sample. Samples with variance > threshold go to a "needs review" queue.
**Effort:** Low — vote variance is already computed and saved in ASTRA outputs.

---

## Low Priority (Research Directions)

### 7. Cross-Lingual Semantic Alignment
**Why:** If a student writes a correct answer in Arabic and the reference is in English, purely surface-level comparison may miss semantic equivalence.
**How:** Use a multilingual embedding model (e.g., `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) to compute semantic similarity between the Arabic transcript and English reference. Include this as an additional rubric criterion.

### 8. Diagram and Figure Recognition
**Why:** STEM answers often include diagrams (circuits, graphs, geometric figures). Current pipeline ignores non-text image regions.
**How:** Add a third OCR head for diagram recognition using a document layout model (DocLayout-YOLO or LayoutLMv3). Describe diagrams in text for the VLM.

### 9. Evaluation Against Multiple Human Raters
**Why:** A single human score has its own variance. Comparing LLM scores against inter-rater agreement gives a more realistic upper bound.
**How:** Collect dual-annotated samples and compute QWK between rater 1 and rater 2 as the ceiling metric.

### 10. Streaming / Real-Time Scoring API
**Why:** The current pipeline is batch-oriented. For deployment, a teacher needs scores within seconds of a student submitting.
**How:** Wrap the ASTRA pipeline in a FastAPI server. Pre-load the model, accept image uploads, return scores with structured JSON.

---

## Hardware Scaling Notes

| Hardware | Recommended Config |
|---|---|
| T4 (15GB, free Colab) | Qwen2-VL-7B in 4-bit, N=5 votes |
| A100 (40GB) | Qwen2-VL-7B in fp16 or Qwen2-VL-72B in 4-bit, N=10 votes |
| CPU only | Qwen2-VL-2B in 4-bit, N=3 votes, API fallback for scoring |
| Multi-GPU (2× A100) | Qwen2-VL-72B in fp16, N=10 votes for best results |
