# Literature Survey — Multimodal STEM Assessment Platform

This document collects all papers, models, and datasets referenced in this project.
For each source, we include: citation, relevance to our work, and a one-paragraph summary.

---

## Primary Papers

### 1. Yavuz et al. (2024) — LLMs for EFL Essay Grading
**Citation:** Yavuz, M. F., et al. (2024). *Utilizing LLMs for EFL essay grading.* arXiv:2501.07244.
**Link:** https://arxiv.org/abs/2501.07244
**Relevance:** Direct baseline for our work. Core methodology that we extend.

**Summary:** The paper investigates using GPT-family models to grade English as a Foreign Language (EFL) essays against rubric criteria. The authors prompt the model with the essay, rubric, and reference answer, then collect a score. They find that LLMs can approach human inter-rater reliability on well-defined rubric dimensions. Our work extends this approach to: (1) multimodal input (handwritten images instead of typed text), (2) Arabic-English bilingual content, and (3) a self-consistency voting mechanism to reduce LLM grading variance.

---

### 2. Wei et al. (2022) — Chain-of-Thought Prompting
**Citation:** Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
**Link:** https://arxiv.org/abs/2201.11903
**Relevance:** Scientific grounding for Experiment 2 (CoT scoring).

**Summary:** Chain-of-Thought (CoT) prompting instructs a language model to produce intermediate reasoning steps before giving a final answer. Wei et al. demonstrate that this significantly improves performance on reasoning tasks (arithmetic, commonsense reasoning, symbolic manipulation). We adapt CoT for automated grading: the model is asked to transcribe the handwriting, analyze the student's reasoning step-by-step, and then assign a score. This provides more consistent and interpretable outputs compared to direct scoring (Exp 1).

---

### 3. Wang et al. (2023) — Self-Consistency Improves CoT Reasoning
**Citation:** Wang, X., et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023.
**Link:** https://arxiv.org/abs/2203.11171
**Relevance:** Core innovation in ASTRA — the self-consistency voting component.

**Summary:** Self-consistency is a simple decoding strategy that samples multiple diverse reasoning chains (at temperature > 0) and selects the most consistent answer via majority vote. Wang et al. show this reliably outperforms single-pass greedy decoding on a range of reasoning tasks without additional training. In ASTRA, we apply this principle to grading: we run the scoring prompt N=5 times at temperature=0.7 and aggregate criterion scores via the median. This reduces variance introduced by LLM hallucination in grading, analogous to averaging across multiple human raters.

---

### 4. Ahuja et al. (2023) — MEGA: Multilingual Evaluation of Generative AI
**Citation:** Ahuja, K., et al. (2023). *MEGA: Multilingual Evaluation of Generative AI.* EMNLP 2023.
**Link:** https://arxiv.org/abs/2303.12528
**Relevance:** Motivates the bias calibration component of ASTRA.

**Summary:** MEGA benchmarks generative AI models (GPT-4, etc.) across 70+ languages and finds systematic performance gaps between high-resource languages (English) and lower-resource ones (Arabic, Hindi, etc.). Models consistently score non-English content differently than human evaluators, indicating language-specific biases. This empirically motivates ASTRA's calibration step: we compute per-language mean residuals between LLM scores and human scores on a calibration split, then apply a correction offset. This directly addresses the Ahuja et al. finding without requiring model fine-tuning.

---

### 5. Zheng et al. (2023) — Judging LLM-as-a-Judge with MT-Bench
**Citation:** Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023.
**Link:** https://arxiv.org/abs/2306.05685
**Relevance:** Informs our LLM-as-judge pipeline design and known failure modes to avoid.

**Summary:** This paper systematically analyzes the strengths and failure modes of using LLMs as automated evaluators. Key failure modes include: position bias (preferring answers listed first), verbosity bias (preferring longer answers), and inconsistency across multiple calls. Their recommendations — position-balanced prompts, multi-call voting, and careful rubric design — directly shaped our ASTRA design: we use per-criterion prompts (avoiding position bias in multi-answer comparison), self-consistency voting (addressing inconsistency), and auto-generated rubrics with explicit weights.

---

### 6. Li et al. (2021) — TrOCR: Transformer-based OCR
**Citation:** Li, M., et al. (2021). *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models.* AAAI 2023.
**Link:** https://arxiv.org/abs/2109.10282
**Relevance:** Stage 2 OCR component for text recognition.

**Summary:** TrOCR is a seq2seq model that combines a ViT image encoder with a BERT/GPT text decoder for OCR. Pre-trained on large synthetic and real handwriting datasets, it achieves state-of-the-art results on handwriting recognition benchmarks (IAM, SROIE). We use the `microsoft/trocr-base-handwritten` checkpoint as the text recognition backbone in our two-headed OCR pipeline (Stage 2). Its primary limitation is reduced performance on Arabic script, which motivates our use of Qwen2-VL as the primary OCR path.

---

### 7. Wang et al. (2024) — Qwen2-VL Technical Report
**Citation:** Wang, P., et al. (2024). *Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution.* arXiv:2409.12191.
**Link:** https://arxiv.org/abs/2409.12191
**Relevance:** Primary backbone model for all experiments.

**Summary:** Qwen2-VL is a 7B (and 72B) parameter Vision-Language Model from Alibaba that supports images at any resolution via a Naive Dynamic Resolution mechanism. It demonstrates strong performance on OCR, document understanding, and multilingual tasks including Arabic. Crucially, it supports 4-bit quantization via bitsandbytes, enabling deployment on a single consumer-grade T4 GPU (15GB VRAM). We use `Qwen/Qwen2-VL-7B-Instruct` as the backbone for OCR (Stage 2), rubric generation (Stage 3), and all four experiments.

---

### 8. Blecher (2023) — pix2tex: LaTeX OCR
**Citation:** Blecher, L. (2023). *pix2tex: Using a ViT to convert images of equations into LaTeX.* GitHub.
**Link:** https://github.com/lukas-blecher/LaTeX-OCR
**Relevance:** Math expression recognition component in Stage 2.

**Summary:** pix2tex is a lightweight ViT + GPT-2 model trained specifically to recognize mathematical expressions and output LaTeX source code. It handles fractions, integrals, summations, Greek letters, and scientific notation reliably. We use it to process detected math regions from the handwritten images, producing LaTeX strings that are then included in the student transcript passed to the evaluation pipeline.

---

## Dataset

### 9. AI4Bharat FERMAT Dataset
**Citation:** AI4Bharat (2024). *FERMAT: Free-form Evaluation of Reasoning in Mathematics and Analytical Tasks.*
**Link:** https://github.com/AI4Bharat/FERMAT
**HuggingFace:** https://huggingface.co/datasets/ai4bharat/FERMAT (if available)
**Relevance:** Primary evaluation dataset.

**Summary:** FERMAT is a benchmark dataset for evaluating mathematical reasoning from free-form handwritten student answers. It includes images of handwritten responses paired with questions, reference answers, and human-assigned scores. We use FERMAT as our primary evaluation corpus and extend its evaluation framework to the multimodal multilingual Arabic-English setting. The dataset provides a standardized test bed for comparing our four scoring methods against human annotator scores.

---

## Additional References

### 10. Weigle (2002) — Assessing Writing
**Citation:** Weigle, S.C. (2002). *Assessing Writing.* Cambridge University Press.
**Relevance:** Educational assessment theory — analytic vs holistic scoring.

**Summary:** Foundational text on writing assessment methodology. Distinguishes between holistic scoring (single overall impression score) and analytic scoring (separate scores for each criterion: content, organization, vocabulary, grammar). Analytic scoring provides more diagnostic information and higher inter-rater reliability. Our rubric-decomposed approach (Exp 3) and ASTRA (Exp 4) are grounded in analytic scoring principles extended to STEM domains.

---

### 11. Dettmers et al. (2023) — QLoRA / bitsandbytes
**Citation:** Dettmers, T., et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.
**Link:** https://arxiv.org/abs/2305.14314
**Relevance:** 4-bit quantization enabling our pipeline to run on a T4 GPU.

**Summary:** QLoRA introduces 4-bit NF4 quantization for LLMs, enabling 7B parameter models to run on a single 15–20GB GPU with minimal accuracy degradation. The `bitsandbytes` library implements this. We apply 4-bit quantization to Qwen2-VL-7B throughout our pipeline, reducing GPU memory usage from ~28GB (fp16) to ~9GB, making it feasible to run all experiments on Google Colab's free T4 GPU.

---

### 12. Liang et al. (2022) — HELM: Holistic Evaluation of Language Models
**Citation:** Liang, P., et al. (2022). *Holistic Evaluation of Language Models.* TMLR 2023.
**Link:** https://arxiv.org/abs/2211.09110
**Relevance:** Framework for evaluating LLMs across multiple dimensions — informs our multi-metric approach.

**Summary:** HELM proposes evaluating LLMs holistically across accuracy, calibration, robustness, fairness, and efficiency. Our results analysis follows this spirit by reporting multiple metrics (QWK, Pearson r, RMSE) across multiple subgroups (language tracks) rather than a single aggregate score. The per-language breakdown in Stage 5 directly echoes HELM's subgroup analysis philosophy.

---

## Models Considered (HuggingFace)

| Model | HF ID | Role | Notes |
|---|---|---|---|
| Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct` | Primary VLM | Arabic-capable, fits T4 in 4-bit |
| TrOCR-base-handwritten | `microsoft/trocr-base-handwritten` | Text HTR | Limited Arabic support |
| pix2tex | (pip: `pix2tex`) | Math LaTeX OCR | Lightweight, fast |
| InternVL2-8B | `OpenGVLab/InternVL2-8B` | Alternative VLM | Strong doc understanding |
| Phi-3-Vision-128k | `microsoft/Phi-3-vision-128k-instruct` | Lightweight alternative | 4B params, less Arabic coverage |
| Qwen2-VL-2B-Instruct | `Qwen/Qwen2-VL-2B-Instruct` | Ultra-lightweight option | For CPU or 8GB VRAM |
| AraBERT | `aubmindlab/bert-base-arabertv2` | Arabic text embedding | For text-only Arabic baseline |
