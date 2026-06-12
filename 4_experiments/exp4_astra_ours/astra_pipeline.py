"""
ASTRA Pipeline — Adaptive Script-aware Two-stage Rubric Assessment.

This module implements the novel components of ASTRA:
  1. Script-aware routing
  2. Self-consistency voting
  3. Bias calibration

It is called by run_experiment.py and can also be imported as a library.

Scientific foundations:
  [SC]  Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning.
        ICLR 2023. https://arxiv.org/abs/2203.11171
  [CAL] Ahuja et al. (2023). MEGA: Multilingual Evaluation of Generative AI.
        EMNLP 2023. https://arxiv.org/abs/2303.12528
  [JDG] Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
        NeurIPS 2023. https://arxiv.org/abs/2306.05685
"""

import json
import statistics
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Component 1: Script-Aware Routing
# ---------------------------------------------------------------------------

def detect_script_from_image(image_path: Path, client=None) -> str:
    """
    Detect the dominant script in a handwritten image.

    Primary method: ask Qwen2-VL directly (most accurate).
    Fallback: use the language_track column from metadata (set in Stage 1).

    Returns: 'arabic', 'english', or 'mixed'.
    """
    if client is None:
        from llm_client import QwenVLClient
        client = QwenVLClient.get_instance()

    prompt = (
        "Look at this handwritten text. Respond with ONLY one word: "
        "'arabic', 'english', or 'mixed' depending on the script used."
    )
    try:
        response = client.generate_from_image(
            image_path, prompt, max_new_tokens=10, temperature=0.0
        )
        lang = response.strip().lower()
        if lang in ("arabic", "english", "mixed"):
            return lang
    except Exception:
        pass
    return "unknown"


def route_language(metadata_language_track: str, detected_language: str) -> str:
    """
    Combine metadata-provided language hint with the VLM-detected language.
    Metadata takes precedence if it was explicitly set; otherwise use detection.
    """
    if metadata_language_track and metadata_language_track not in ("unknown", "nan", ""):
        return metadata_language_track
    return detected_language or "english"


# ---------------------------------------------------------------------------
# Component 2: Self-Consistency Voting
# ---------------------------------------------------------------------------

def run_voting_round(
    client,
    score_prompt: str,
    n_votes: int,
    temperature: float,
) -> list[dict]:
    """
    Run the scoring prompt N times at a given temperature.
    Returns a list of N parsed JSON dicts (one per vote).

    Motivation [SC]: Sampling multiple outputs and aggregating reduces the
    variance introduced by LLM non-determinism, leading to more reliable scores.
    """
    from llm_client import parse_json_from_response

    votes = []
    for i in range(n_votes):
        try:
            response = client.generate_text_only(
                score_prompt, max_new_tokens=512, temperature=temperature
            )
            parsed = parse_json_from_response(response)
            if parsed and "criterion_scores" in parsed:
                votes.append(parsed)
        except Exception as e:
            print(f"    [WARN] Vote {i+1} failed: {e}")
    return votes


def aggregate_votes_median(votes: list[dict], rubric: dict) -> dict:
    """
    Aggregate N voting results by taking the median score per criterion.

    Why median (not mean)?
    The median is more robust to outlier votes caused by LLM hallucination.
    One poorly-calibrated vote out of five won't dominate the final score.
    This is analogous to rejecting outlier judges in human scoring panels.

    Returns a dict with keys: 'criterion_scores' (list), 'total_score' (float).
    """
    criteria = rubric.get("criteria", [])
    weight_map = {c["name"]: c["weight"] for c in criteria}

    # Collect all per-criterion scores across votes
    per_criterion: dict[str, list[float]] = {c["name"]: [] for c in criteria}
    justifications: dict[str, str] = {}

    for vote in votes:
        for cs in vote.get("criterion_scores", []):
            name = cs.get("name")
            if name in per_criterion:
                per_criterion[name].append(float(cs.get("score", 0.0)))
                if name not in justifications:
                    justifications[name] = cs.get("justification", "")

    aggregated_criterion_scores = []
    total_weighted = 0.0
    total_weight = 0.0

    for c in criteria:
        name = c["name"]
        scores_for_criterion = per_criterion.get(name, [])
        if scores_for_criterion:
            median_score = float(statistics.median(scores_for_criterion))
        else:
            median_score = 0.0

        weight = weight_map.get(name, 0.0)
        total_weighted += median_score * weight
        total_weight += weight

        aggregated_criterion_scores.append({
            "name": name,
            "score": round(max(0.0, min(1.0, median_score)), 4),
            "weight": weight,
            "justification": justifications.get(name, ""),
            "n_votes_contributing": len(scores_for_criterion),
        })

    total_score = total_weighted / total_weight if total_weight > 0 else 0.0

    return {
        "criterion_scores": aggregated_criterion_scores,
        "total_score": round(max(0.0, min(1.0, total_score)), 4),
        "n_votes": len(votes),
        "raw_votes": [round(float(v.get("total_score", 0.0)), 4) for v in votes],
    }


# ---------------------------------------------------------------------------
# Component 3: Bias Calibration
# ---------------------------------------------------------------------------

def compute_calibration_offsets(
    predictions_df: pd.DataFrame,
    calibration_fraction: float = 0.2,
    min_samples: int = 10,
) -> dict:
    """
    Compute empirical per-language calibration offsets from a subset of
    labeled samples (those with human_score available).

    Calibration procedure [CAL]:
      For each language track:
        offset = mean(human_scores) - mean(llm_scores)
      Then apply: calibrated_score = llm_score + offset

    This corrects for systematic over/under-scoring of non-English content,
    a bias documented in Ahuja et al. 2023 (MEGA benchmark).

    Args:
        predictions_df:      DataFrame with columns: total_score, human_score, detected_language.
        calibration_fraction: Fraction of labeled samples to use for calibration.
        min_samples:          Minimum labeled samples per language to compute a reliable offset.

    Returns:
        Dict: language_track → offset float.
        Example: {'arabic': 0.12, 'english': -0.03, 'mixed': 0.05}
    """
    labeled = predictions_df.dropna(subset=["human_score"]).copy()
    if labeled.empty:
        print("  [CAL] No labeled samples available. Using zero offsets.")
        return {}

    # Use the first calibration_fraction as calibration set
    n_cal = max(1, int(len(labeled) * calibration_fraction))
    cal_df = labeled.iloc[:n_cal]

    offsets = {}
    for lang in cal_df["detected_language"].unique():
        lang_df = cal_df[cal_df["detected_language"] == lang]
        if len(lang_df) < min_samples:
            print(f"  [CAL] Insufficient samples for '{lang}' ({len(lang_df)} < {min_samples}). Skipping.")
            continue
        mean_llm = lang_df["total_score"].mean()
        mean_human = lang_df["human_score"].astype(float).mean()
        offset = float(mean_human - mean_llm)
        offsets[lang] = round(offset, 4)
        print(f"  [CAL] Language '{lang}': mean_human={mean_human:.3f}, "
              f"mean_llm={mean_llm:.3f}, offset={offset:+.3f}  (n={len(lang_df)})")

    return offsets


def apply_calibration(score: float, language_track: str, offsets: dict) -> float:
    """Apply the calibration offset for the given language track."""
    offset = offsets.get(language_track, 0.0)
    return max(0.0, min(1.0, score + offset))


# ---------------------------------------------------------------------------
# Full ASTRA pipeline entry point
# ---------------------------------------------------------------------------

def run_astra_sample(
    sample_id: str,
    image_path: Path,
    question: str,
    reference_answer: str,
    language_track: str,
    client,
    rubric: dict,
    n_votes: int = 5,
    vote_temperature: float = 0.7,
    calibration_offsets: Optional[dict] = None,
) -> dict:
    """
    Run the full ASTRA pipeline for a single sample.

    Returns a dict ready for serialization.
    """
    from prompt_templates import (
        prompt_astra_transcribe,
        prompt_astra_score_all_criteria,
    )

    # Step 1: Script-aware transcription
    transcribe_prompt = prompt_astra_transcribe(language_track)
    transcript = client.generate_from_image(
        image_path, transcribe_prompt, max_new_tokens=512, temperature=0.1
    )

    # Step 2: Build scoring prompt and run N votes
    score_prompt = prompt_astra_score_all_criteria(
        question=question,
        reference_answer=reference_answer,
        student_transcript=transcript,
        rubric=rubric,
        language_track=language_track,
    )
    votes = run_voting_round(client, score_prompt, n_votes=n_votes, temperature=vote_temperature)

    if not votes:
        # Graceful fallback if all votes failed
        return {
            "id": sample_id,
            "total_score": 0.0,
            "criterion_scores": [],
            "transcription": transcript,
            "detected_language": language_track,
            "method": "astra",
            "error": "all votes failed",
        }

    # Step 3: Aggregate by median
    aggregated = aggregate_votes_median(votes, rubric)

    # Step 4: Bias calibration
    raw_total = aggregated["total_score"]
    cal_offsets = calibration_offsets or {}
    calibrated_total = apply_calibration(raw_total, language_track, cal_offsets)

    return {
        "id": sample_id,
        "total_score": calibrated_total,
        "pre_calibration_score": raw_total,
        "criterion_scores": aggregated["criterion_scores"],
        "transcription": transcript,
        "detected_language": language_track,
        "calibrated": (language_track in cal_offsets),
        "calibration_offset": cal_offsets.get(language_track, 0.0),
        "n_votes": aggregated["n_votes"],
        "raw_votes": aggregated["raw_votes"],
        "method": "astra",
    }
