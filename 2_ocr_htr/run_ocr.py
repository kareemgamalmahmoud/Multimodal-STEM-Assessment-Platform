"""
Stage 2 — End-to-end OCR / HTR runner.

Two modes:
  --mode trocr     (default) Region detection → TrOCR (text) + pix2tex (math)
  --mode qwen2vl              Qwen2-VL direct end-to-end OCR (single model, better quality)

Usage:
    python run_ocr.py
    python run_ocr.py --mode qwen2vl
    python run_ocr.py --mode trocr --limit 50   # process only first 50 samples

Outputs:
    ../data/transcripts/<sample_id>.json  — one JSON transcript per sample
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "fermat_processed"
TRANSCRIPT_DIR = REPO_ROOT / "data" / "transcripts"

# Allow imports from this stage
sys.path.insert(0, str(Path(__file__).parent))

from region_detector import detect_regions
from arabic_htr import recognize_handwriting
from math_ocr import recognize_math
from text_assembler import (
    assemble_transcript,
    assemble_from_qwen_output,
    save_transcript,
)


# ---------------------------------------------------------------------------
# Mode 1: TrOCR + pix2tex (two-headed)
# ---------------------------------------------------------------------------

def run_trocr_mode(metadata_df: pd.DataFrame, limit: int = None) -> int:
    """
    For each sample: detect regions → HTR (TrOCR) or math OCR (pix2tex)
    per region → assemble transcript.
    """
    rows = metadata_df.iterrows()
    count = 0

    for idx, row in tqdm(rows, total=len(metadata_df), desc="TrOCR+pix2tex"):
        if limit and count >= limit:
            break

        sample_id = str(row["id"])
        img_path = PROCESSED_DIR / str(row["image_path"])

        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        try:
            # Detect regions
            regions = detect_regions(img_path)

            region_text_pairs = []
            for region in regions:
                if region.label == "math":
                    text = recognize_math(img_path)  # uses pix2tex on full image slice
                else:
                    text = recognize_handwriting(img_path)
                region_text_pairs.append((region, text))

            # Assemble
            transcript = assemble_transcript(
                sample_id=sample_id,
                region_text_pairs=region_text_pairs,
                ocr_mode="trocr+pix2tex",
            )
            save_transcript(transcript, TRANSCRIPT_DIR)
            count += 1

        except Exception as e:
            print(f"  [ERROR] {sample_id}: {e}")

    return count


# ---------------------------------------------------------------------------
# Mode 2: Qwen2-VL end-to-end OCR
# ---------------------------------------------------------------------------

QWEN_OCR_PROMPT = (
    "You are an expert OCR system. Transcribe ALL handwritten content in this image exactly as written.\n"
    "- Preserve Arabic text in Arabic script.\n"
    "- Convert mathematical expressions to LaTeX, wrapped in $$ ... $$.\n"
    "- Preserve line breaks.\n"
    "- Do NOT add explanations or comments — output only the transcribed text."
)


def run_qwen2vl_mode(metadata_df: pd.DataFrame, limit: int = None) -> int:
    """
    Use Qwen2-VL for direct end-to-end OCR: single model pass per image.
    Much better at Arabic and mixed-script content than TrOCR.
    """
    # Lazy import to avoid loading the model when using trocr mode
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from transformers import BitsAndBytesConfig
    from qwen_vl_utils import process_vision_info

    print("[Qwen2-VL] Loading model (4-bit quantization) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    print("[Qwen2-VL] Model ready.\n")

    count = 0
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Qwen2-VL OCR"):
        if limit and count >= limit:
            break

        sample_id = str(row["id"])
        img_path = PROCESSED_DIR / str(row["image_path"])
        language_track = str(row.get("language_track", "unknown"))

        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": QWEN_OCR_PROMPT},
                    ],
                }
            ]

            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)

            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            raw_text = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            transcript = assemble_from_qwen_output(
                sample_id=sample_id,
                raw_text=raw_text,
                detected_language=language_track,
            )
            save_transcript(transcript, TRANSCRIPT_DIR)
            count += 1

        except Exception as e:
            print(f"  [ERROR] {sample_id}: {e}")

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2 — OCR / HTR Runner")
    parser.add_argument(
        "--mode",
        choices=["trocr", "qwen2vl"],
        default="trocr",
        help="OCR mode: 'trocr' (TrOCR + pix2tex) or 'qwen2vl' (single VLM, recommended)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N samples (useful for quick tests)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Stage 2 — OCR/HTR  [mode: {args.mode}]")
    print("=" * 60)

    meta_path = PROCESSED_DIR / "metadata.csv"
    if not meta_path.exists():
        print(f"[ERROR] metadata.csv not found at {meta_path}. Run Stage 1 first.")
        sys.exit(1)

    df = pd.read_csv(meta_path)
    print(f"[INFO] {len(df)} samples to process.\n")

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "trocr":
        processed = run_trocr_mode(df, limit=args.limit)
    else:
        processed = run_qwen2vl_mode(df, limit=args.limit)

    print(f"\n[DONE] Transcripts generated: {processed}")
    print(f"  Output directory: {TRANSCRIPT_DIR}")
    print("\nStage 2 — OCR complete.\n")


if __name__ == "__main__":
    main()
