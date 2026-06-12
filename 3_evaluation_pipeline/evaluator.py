"""
End-to-end evaluator — orchestrates the full pipeline for a single sample.

Used by experiment runners to process one row from the metadata CSV.
Handles:
  - Image path resolution
  - Transcript loading (from Stage 2 output)
  - Rubric generation / loading
  - Calling the appropriate scorer
  - Returning a ScoringResult
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "fermat_processed"
TRANSCRIPT_DIR = REPO_ROOT / "data" / "transcripts"


def load_transcript_for_sample(sample_id: str) -> Optional[dict]:
    """Load the OCR transcript JSON for a sample. Returns None if not found."""
    path = TRANSCRIPT_DIR / f"{sample_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_sample(
    row: pd.Series,
    method: str,
    client=None,
    calibration_offsets: Optional[dict] = None,
    n_votes: int = 5,
) -> Optional[dict]:
    """
    Run evaluation for one metadata row.

    Args:
        row:                  A row from metadata.csv (id, question, reference_answer,
                              image_path, language_track, human_score).
        method:               One of 'baseline', 'cot', 'rubric_decomposed', 'astra'.
        client:               QwenVLClient instance. If None, uses the singleton.
        calibration_offsets:  Dict mapping language_track → offset float (for ASTRA).
        n_votes:              Number of voting passes for ASTRA.

    Returns:
        Dict representation of ScoringResult, or None on unrecoverable error.
    """
    from scorer import (
        score_baseline, score_cot, score_rubric_decomposed, score_astra
    )
    from rubric_generator import generate_rubric
    from llm_client import QwenVLClient

    client = client or QwenVLClient.get_instance()

    sample_id = str(row["id"])
    question = str(row.get("question", ""))
    reference_answer = str(row.get("reference_answer", ""))
    language_track = str(row.get("language_track", "english"))
    image_path = PROCESSED_DIR / str(row["image_path"])

    if not image_path.exists():
        print(f"  [SKIP] Image missing: {image_path}")
        return None

    try:
        if method == "baseline":
            result = score_baseline(
                sample_id=sample_id,
                image_path=image_path,
                question=question,
                reference_answer=reference_answer,
                language_track=language_track,
                client=client,
            )

        elif method == "cot":
            result = score_cot(
                sample_id=sample_id,
                image_path=image_path,
                question=question,
                reference_answer=reference_answer,
                language_track=language_track,
                client=client,
            )

        elif method in ("rubric_decomposed", "astra"):
            # Both rubric-based methods need a transcript from Stage 2
            transcript_data = load_transcript_for_sample(sample_id)
            transcript = (
                transcript_data["transcript"] if transcript_data
                else "[OCR transcript not available — run Stage 2 first]"
            )

            rubric = generate_rubric(
                question=question,
                reference_answer=reference_answer,
                client=client,
                language_track=language_track,
            )

            if method == "rubric_decomposed":
                result = score_rubric_decomposed(
                    sample_id=sample_id,
                    image_path=image_path,
                    question=question,
                    reference_answer=reference_answer,
                    transcript=transcript,
                    rubric=rubric,
                    language_track=language_track,
                    client=client,
                )
            else:  # astra
                offset = (calibration_offsets or {}).get(language_track, 0.0)
                result = score_astra(
                    sample_id=sample_id,
                    image_path=image_path,
                    question=question,
                    reference_answer=reference_answer,
                    rubric=rubric,
                    language_track=language_track,
                    n_votes=n_votes,
                    vote_temperature=0.7,
                    calibration_offset=offset,
                    client=client,
                )
        else:
            raise ValueError(f"Unknown method: {method}")

        out = result.to_dict()
        out["human_score"] = row.get("human_score")
        return out

    except Exception as e:
        print(f"  [ERROR] {sample_id} ({method}): {e}")
        return None
