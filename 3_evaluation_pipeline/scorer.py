"""
Scorer — assigns a score to a student answer using the LLM and a rubric.

Provides four scoring functions corresponding to the four experiments:
  score_baseline()         → Exp 1: direct single-call score
  score_cot()              → Exp 2: chain-of-thought score
  score_rubric_decomposed() → Exp 3: per-criterion rubric scoring
  score_astra()            → Exp 4: ASTRA (self-consistency voting + calibration)

All functions return a ScoringResult dataclass.
"""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CriterionScore:
    name: str
    score: float
    weight: float
    justification: str = ""


@dataclass
class ScoringResult:
    sample_id: str
    total_score: float                          # normalized [0, 1]
    criterion_scores: List[CriterionScore] = field(default_factory=list)
    reasoning: str = ""
    transcription: str = ""
    detected_language: str = "unknown"
    calibrated: bool = False
    calibration_offset: float = 0.0
    n_votes: int = 1                            # for ASTRA
    raw_votes: List[float] = field(default_factory=list)  # per-vote totals
    method: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "id": self.sample_id,
            "total_score": round(self.total_score, 4),
            "criterion_scores": [
                {
                    "name": c.name,
                    "score": round(c.score, 4),
                    "weight": c.weight,
                    "justification": c.justification,
                }
                for c in self.criterion_scores
            ],
            "reasoning": self.reasoning,
            "transcription": self.transcription,
            "detected_language": self.detected_language,
            "calibrated": self.calibrated,
            "calibration_offset": self.calibration_offset,
            "n_votes": self.n_votes,
            "raw_votes": [round(v, 4) for v in self.raw_votes],
            "method": self.method,
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalize_score(value: float) -> float:
    """
    Normalize a score to [0.0, 1.0] regardless of the scale the model used.
    - If model used 0–4 scale (common exam grade scale), divide by 4.
    - If model used 0–5 scale, divide by 5.
    - If model used 0–10 scale, divide by 10.
    - If already in [0, 1], keep as-is.
    This guards against the known issue where Qwen2-VL ignores the 0–1 instruction.
    """
    if value <= 1.0:
        return _clamp(value)
    elif value <= 4.0:
        return _clamp(value / 4.0)
    elif value <= 5.0:
        return _clamp(value / 5.0)
    elif value <= 10.0:
        return _clamp(value / 10.0)
    else:
        return _clamp(value / 100.0)


def _weighted_total(criterion_scores: List[CriterionScore]) -> float:
    if not criterion_scores:
        return 0.0
    total_weight = sum(c.weight for c in criterion_scores) or 1.0
    return sum(c.score * c.weight for c in criterion_scores) / total_weight


def _parse_criterion_scores(raw: dict, rubric: dict) -> List[CriterionScore]:
    """
    Map model output criterion scores back to rubric criteria with weights.
    """
    criteria_map = {c["name"]: c["weight"] for c in rubric.get("criteria", [])}
    results = []
    for item in raw.get("criterion_scores", []):
        name = item.get("name", "unknown")
        results.append(CriterionScore(
            name=name,
            score=_normalize_score(float(item.get("score", 0.0))),
            weight=criteria_map.get(name, 1.0 / max(len(criteria_map), 1)),
            justification=item.get("justification", ""),
        ))
    return results


# ---------------------------------------------------------------------------
# Experiment 1: Baseline
# ---------------------------------------------------------------------------

def score_baseline(
    sample_id: str,
    image_path: Path,
    question: str,
    reference_answer: str,
    language_track: str = "english",
    client=None,
) -> ScoringResult:
    """Direct single-call scoring with no explicit reasoning."""
    from llm_client import QwenVLClient, parse_json_from_response
    from prompt_templates import prompt_baseline_score

    client = client or QwenVLClient.get_instance()
    prompt = prompt_baseline_score(question, reference_answer, language_track)
    response = client.generate_from_image(image_path, prompt, temperature=0.1)
    parsed = parse_json_from_response(response)

    raw_score = _clamp(float(parsed.get("total_score", 0.0)))
    return ScoringResult(
        sample_id=sample_id,
        total_score=raw_score,
        reasoning=parsed.get("brief_justification", ""),
        detected_language=language_track,
        method="baseline",
    )


# ---------------------------------------------------------------------------
# Experiment 2: Chain-of-Thought
# ---------------------------------------------------------------------------

def score_cot(
    sample_id: str,
    image_path: Path,
    question: str,
    reference_answer: str,
    language_track: str = "english",
    client=None,
) -> ScoringResult:
    """Chain-of-thought scoring: transcribe → reason → score."""
    from llm_client import QwenVLClient, parse_json_from_response
    from prompt_templates import prompt_cot_transcribe_and_score

    client = client or QwenVLClient.get_instance()
    prompt = prompt_cot_transcribe_and_score(question, reference_answer)
    response = client.generate_from_image(image_path, prompt, max_new_tokens=1024, temperature=0.1)
    parsed = parse_json_from_response(response)

    raw_score = _clamp(float(parsed.get("total_score", 0.0)))
    return ScoringResult(
        sample_id=sample_id,
        total_score=raw_score,
        reasoning=parsed.get("step_by_step_analysis", ""),
        transcription=parsed.get("transcription", ""),
        detected_language=language_track,
        method="chain_of_thought",
    )


# ---------------------------------------------------------------------------
# Experiment 3: Rubric-decomposed
# ---------------------------------------------------------------------------

def score_rubric_decomposed(
    sample_id: str,
    image_path: Path,
    question: str,
    reference_answer: str,
    transcript: str,
    rubric: dict,
    language_track: str = "english",
    client=None,
) -> ScoringResult:
    """Score each rubric criterion independently in separate model calls."""
    from llm_client import QwenVLClient, parse_json_from_response
    from prompt_templates import prompt_rubric_score_criterion

    client = client or QwenVLClient.get_instance()
    criterion_scores = []

    for criterion in rubric.get("criteria", []):
        prompt = prompt_rubric_score_criterion(
            question=question,
            reference_answer=reference_answer,
            student_transcript=transcript,
            criterion_name=criterion["name"],
            criterion_description=criterion["description"],
        )
        response = client.generate_text_only(prompt, max_new_tokens=256, temperature=0.1)
        parsed = parse_json_from_response(response)

        criterion_scores.append(CriterionScore(
            name=criterion["name"],
            score=_clamp(float(parsed.get("score", 0.0))),
            weight=criterion["weight"],
            justification=parsed.get("justification", ""),
        ))

    total = _weighted_total(criterion_scores)
    return ScoringResult(
        sample_id=sample_id,
        total_score=_clamp(total),
        criterion_scores=criterion_scores,
        transcription=transcript,
        detected_language=language_track,
        method="rubric_decomposed",
    )


# ---------------------------------------------------------------------------
# Experiment 4: ASTRA
# ---------------------------------------------------------------------------

def score_astra(
    sample_id: str,
    image_path: Path,
    question: str,
    reference_answer: str,
    rubric: dict,
    language_track: str = "english",
    n_votes: int = 5,
    vote_temperature: float = 0.7,
    calibration_offset: float = 0.0,
    client=None,
) -> ScoringResult:
    """
    ASTRA: Adaptive Script-aware Two-stage Rubric Assessment.

    Pipeline:
      1. Transcribe the image with a language-conditioned prompt.
      2. Run the rubric scoring prompt N times at temperature=vote_temperature.
      3. Aggregate via majority vote (median score per criterion).
      4. Apply calibration offset.

    Self-consistency grounding: Wang et al. 2023
    Calibration grounding: Ahuja et al. 2023 (MEGA)
    """
    from llm_client import QwenVLClient, parse_json_from_response
    from prompt_templates import (
        prompt_astra_transcribe,
        prompt_astra_score_all_criteria,
    )

    client = client or QwenVLClient.get_instance()

    # --- Step 1: Script-aware transcription ---
    transcribe_prompt = prompt_astra_transcribe(language_track)
    transcript = client.generate_from_image(
        image_path, transcribe_prompt, max_new_tokens=512, temperature=0.1
    )

    # --- Step 2: Self-consistency voting ---
    score_prompt = prompt_astra_score_all_criteria(
        question=question,
        reference_answer=reference_answer,
        student_transcript=transcript,
        rubric=rubric,
        language_track=language_track,
    )

    all_vote_results = []
    raw_votes = []

    for _ in range(n_votes):
        response = client.generate_text_only(
            score_prompt, max_new_tokens=512, temperature=vote_temperature
        )
        parsed = parse_json_from_response(response)
        all_vote_results.append(parsed)
        raw_votes.append(_normalize_score(float(parsed.get("total_score", 0.0))))

    # --- Step 3: Aggregate — median per criterion (robust to outliers) ---
    criterion_names = [c["name"] for c in rubric.get("criteria", [])]
    criterion_weights = {c["name"]: c["weight"] for c in rubric.get("criteria", [])}

    aggregated_criterion_scores = []
    for name in criterion_names:
        per_vote_scores = []
        per_vote_justifications = []
        for vote in all_vote_results:
            for cs in vote.get("criterion_scores", []):
                if cs.get("name") == name:
                    per_vote_scores.append(_normalize_score(float(cs.get("score", 0.0))))
                    per_vote_justifications.append(cs.get("justification", ""))
                    break

        median_score = statistics.median(per_vote_scores) if per_vote_scores else 0.0
        best_justification = per_vote_justifications[0] if per_vote_justifications else ""

        aggregated_criterion_scores.append(CriterionScore(
            name=name,
            score=median_score,
            weight=criterion_weights.get(name, 0.25),
            justification=best_justification,
        ))

    total_score = _weighted_total(aggregated_criterion_scores)

    # --- Step 4: Apply bias calibration offset ---
    calibrated_score = _clamp(total_score + calibration_offset)

    return ScoringResult(
        sample_id=sample_id,
        total_score=calibrated_score,
        criterion_scores=aggregated_criterion_scores,
        transcription=transcript,
        detected_language=language_track,
        calibrated=(calibration_offset != 0.0),
        calibration_offset=calibration_offset,
        n_votes=n_votes,
        raw_votes=raw_votes,
        method="astra",
    )
