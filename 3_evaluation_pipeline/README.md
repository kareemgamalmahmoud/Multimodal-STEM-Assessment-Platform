# Stage 3 — Evaluation Pipeline

Shared library used by all four experiments. Contains the Qwen2-VL client, prompt templates, rubric generator, and scorer.

**This stage has no standalone runner** — it is imported by the Stage 4 experiment scripts.

## Modules

| Module | Purpose |
|---|---|
| `llm_client.py` | Qwen2-VL inference wrapper (4-bit quant, batching, temperature control) |
| `prompt_templates.py` | All prompt templates for experiments 1–4 |
| `rubric_generator.py` | Auto-generate a weighted rubric from (question, reference answer) |
| `scorer.py` | Score a student answer against a rubric using the LLM |
| `evaluator.py` | End-to-end orchestrator: image → OCR → rubric → score |

## Design Notes

- All experiments share the same `QwenVLClient` (same model weights, loaded once).
- Prompts are versioned in `prompt_templates.py` so each experiment's prompt is auditable.
- The `scorer.py` output is always a dict with keys: `total_score`, `criterion_scores`, `reasoning`.
- Scores are normalized to [0, 1] regardless of the rubric's raw scale.
