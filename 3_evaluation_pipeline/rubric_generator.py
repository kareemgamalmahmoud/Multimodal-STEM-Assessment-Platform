"""
Rubric Generator — auto-generate a weighted scoring rubric from a question and reference answer.

Used by Experiment 3 (rubric-decomposed) and Experiment 4 (ASTRA).
The generated rubric is cached to disk to avoid redundant LLM calls
when re-running experiments.

Rubric format:
    {
        "criteria": [
            {"name": "...", "description": "...", "weight": 0.X},
            ...
        ]
    }
All weights must sum to 1.0.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RUBRIC_CACHE_DIR = REPO_ROOT / "data" / "rubric_cache"


def _rubric_cache_key(question: str, reference_answer: str) -> str:
    """Stable cache key for a (question, reference) pair."""
    content = f"{question.strip()}|||{reference_answer.strip()}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_cached_rubric(cache_key: str) -> Optional[dict]:
    """Return a previously-generated rubric or None if not cached."""
    path = RUBRIC_CACHE_DIR / f"{cache_key}.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_rubric_to_cache(cache_key: str, rubric: dict):
    """Persist a generated rubric to disk."""
    RUBRIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = RUBRIC_CACHE_DIR / f"{cache_key}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rubric, f, ensure_ascii=False, indent=2)


def validate_rubric(rubric: dict) -> bool:
    """
    Check that the rubric has the expected structure and weights sum to ~1.0.
    Returns True if valid.
    """
    criteria = rubric.get("criteria", [])
    if not criteria:
        return False
    total_weight = sum(c.get("weight", 0.0) for c in criteria)
    if not (0.95 <= total_weight <= 1.05):
        return False
    for c in criteria:
        if "name" not in c or "description" not in c or "weight" not in c:
            return False
    return True


def normalize_rubric_weights(rubric: dict) -> dict:
    """Rescale weights so they sum to exactly 1.0."""
    criteria = rubric.get("criteria", [])
    total = sum(c.get("weight", 0.0) for c in criteria) or 1.0
    for c in criteria:
        c["weight"] = round(c.get("weight", 0.0) / total, 4)
    return rubric


def generate_rubric(
    question: str,
    reference_answer: str,
    client=None,
    prompt_fn=None,
    language_track: str = "english",
    use_cache: bool = True,
) -> dict:
    """
    Generate a scoring rubric for the given question + reference answer.

    Args:
        question:         The exam question text.
        reference_answer: The model/reference answer.
        client:           QwenVLClient instance. If None, uses the singleton.
        prompt_fn:        Function(question, reference, language_track) → prompt string.
                          If None, uses the ASTRA rubric prompt by default.
        language_track:   'arabic', 'english', or 'mixed'.
        use_cache:        If True, check disk cache before calling the model.

    Returns:
        Rubric dict with 'criteria' list.
    """
    cache_key = _rubric_cache_key(question, reference_answer)

    if use_cache:
        cached = load_cached_rubric(cache_key)
        if cached and validate_rubric(cached):
            return cached

    # Import here to avoid circular dependencies
    if client is None:
        from llm_client import QwenVLClient, parse_json_from_response
        client = QwenVLClient.get_instance()
    else:
        from llm_client import parse_json_from_response

    if prompt_fn is None:
        from prompt_templates import prompt_astra_generate_rubric
        prompt_fn = prompt_astra_generate_rubric

    prompt = prompt_fn(question, reference_answer, language_track)
    response = client.generate_text_only(prompt, max_new_tokens=512, temperature=0.1)
    rubric = parse_json_from_response(response)

    if not validate_rubric(rubric):
        # Provide a safe fallback rubric if the model output is malformed
        rubric = _fallback_rubric()

    rubric = normalize_rubric_weights(rubric)

    if use_cache:
        save_rubric_to_cache(cache_key, rubric)

    return rubric


def _fallback_rubric() -> dict:
    """
    Hardcoded rubric used when the model fails to generate a valid one.
    Covers the four core STEM assessment dimensions.
    """
    return {
        "criteria": [
            {
                "name": "Conceptual Correctness",
                "description": "Does the student demonstrate understanding of the core concept?",
                "weight": 0.35,
            },
            {
                "name": "Mathematical Accuracy",
                "description": "Are computations, formulas, and numerical results correct?",
                "weight": 0.30,
            },
            {
                "name": "Completeness",
                "description": "Is the answer fully developed with all required steps shown?",
                "weight": 0.20,
            },
            {
                "name": "Notation and Units",
                "description": "Are scientific notation, symbols, and units used correctly?",
                "weight": 0.15,
            },
        ]
    }
