"""
Text Assembler — merge per-region OCR outputs into a single structured transcript.

Given a list of (Region, text) pairs from the region detector + HTR/math-OCR,
this module:
  1. Sorts regions by vertical position (top to bottom reading order).
  2. Separates Arabic/English prose from LaTeX math expressions.
  3. Produces a final JSON transcript for each sample.

The resulting transcript is the input to Stage 3 (evaluation pipeline).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from region_detector import Region


@dataclass
class Transcript:
    """Structured OCR output for one student answer image."""
    sample_id: str
    transcript: str                        # Full text in reading order
    latex_expressions: List[str] = field(default_factory=list)
    arabic_segments: List[str] = field(default_factory=list)
    english_segments: List[str] = field(default_factory=list)
    detected_language: str = "unknown"    # 'arabic', 'english', 'mixed', 'unknown'
    ocr_mode: str = "trocr+pix2tex"

    def to_dict(self) -> dict:
        return {
            "id": self.sample_id,
            "transcript": self.transcript,
            "latex_expressions": self.latex_expressions,
            "arabic_segments": self.arabic_segments,
            "english_segments": self.english_segments,
            "detected_language": self.detected_language,
            "ocr_mode": self.ocr_mode,
        }


def assemble_transcript(
    sample_id: str,
    region_text_pairs: list[tuple[Region, str]],
    ocr_mode: str = "trocr+pix2tex",
) -> Transcript:
    """
    Build a Transcript from a list of (Region, recognized_text) pairs.

    Regions are sorted by vertical position. Math regions get their text
    wrapped in $$ ... $$ delimiters in the combined transcript.

    Args:
        sample_id:         Unique sample identifier.
        region_text_pairs: List of (Region, text_or_latex) tuples.
        ocr_mode:          Label describing which OCR engines were used.

    Returns:
        A Transcript dataclass instance.
    """
    # Sort top-to-bottom
    sorted_pairs = sorted(region_text_pairs, key=lambda rt: rt[0].y)

    lines = []
    latex_exprs = []
    arabic_segs = []
    english_segs = []

    for region, text in sorted_pairs:
        text = text.strip()
        if not text:
            continue

        if region.label == "math":
            wrapped = f"$${text}$$" if not text.startswith("$$") else text
            lines.append(wrapped)
            latex_exprs.append(text)
        elif region.label == "arabic":
            lines.append(text)
            arabic_segs.append(text)
        else:  # english or unknown
            lines.append(text)
            english_segs.append(text)

    # Determine overall language
    if arabic_segs and not english_segs:
        detected_lang = "arabic"
    elif english_segs and not arabic_segs:
        detected_lang = "english"
    elif arabic_segs and english_segs:
        detected_lang = "mixed"
    else:
        detected_lang = "unknown"

    full_transcript = "\n".join(lines)

    return Transcript(
        sample_id=sample_id,
        transcript=full_transcript,
        latex_expressions=latex_exprs,
        arabic_segments=arabic_segs,
        english_segments=english_segs,
        detected_language=detected_lang,
        ocr_mode=ocr_mode,
    )


def assemble_from_qwen_output(
    sample_id: str,
    raw_text: str,
    detected_language: str = "unknown",
) -> Transcript:
    """
    Build a Transcript from Qwen2-VL's direct OCR output (single string).
    Used when running in --mode qwen2vl.

    Qwen2-VL returns the full text including math; we parse out LaTeX blocks.
    """
    import re

    # Extract LaTeX blocks ($$...$$)
    latex_exprs = re.findall(r"\$\$(.*?)\$\$", raw_text, re.DOTALL)
    clean_text = re.sub(r"\$\$.*?\$\$", lambda m: m.group(0), raw_text)  # keep in transcript

    return Transcript(
        sample_id=sample_id,
        transcript=clean_text.strip(),
        latex_expressions=[e.strip() for e in latex_exprs],
        detected_language=detected_language,
        ocr_mode="qwen2vl",
    )


def save_transcript(transcript: Transcript, out_dir: Path):
    """Save transcript as a JSON file in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{transcript.sample_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)
    return out_path


def load_transcript(transcript_path: Path) -> dict:
    """Load a saved transcript JSON file."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_transcripts(transcript_dir: Path) -> dict[str, dict]:
    """
    Load all transcript JSON files from a directory.
    Returns a dict mapping sample_id → transcript dict.
    """
    transcripts = {}
    for path in sorted(transcript_dir.glob("*.json")):
        data = load_transcript(path)
        transcripts[data["id"]] = data
    return transcripts
