"""
Download the FERMAT dataset from HuggingFace or GitHub and organize it locally.

FERMAT (AI4Bharat) is a benchmark for evaluating free-form mathematical reasoning
from handwritten student answers. We use it as our primary evaluation dataset.

Reference: https://github.com/AI4Bharat/FERMAT
"""

import os
import sys
import json
import shutil
import requests
from pathlib import Path

# Add repo root to path for cross-stage imports
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fermat_raw"

# HuggingFace dataset identifier (try this first)
HF_DATASET_ID = "ai4bharat/FERMAT"

# GitHub fallback
GITHUB_REPO_URL = "https://github.com/AI4Bharat/FERMAT"


def try_huggingface_download(save_dir: Path) -> bool:
    """
    Attempt to load FERMAT via the HuggingFace datasets library.
    Returns True on success, False if the dataset is not available on HF Hub.
    """
    try:
        from datasets import load_dataset
        print(f"[INFO] Trying HuggingFace Hub: {HF_DATASET_ID}")
        dataset = load_dataset(HF_DATASET_ID, trust_remote_code=True)

        save_dir.mkdir(parents=True, exist_ok=True)

        # Save each split as JSONL for easy inspection
        for split_name, split_data in dataset.items():
            out_path = save_dir / f"{split_name}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for row in split_data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  Saved {len(split_data)} rows → {out_path}")

        # If dataset has image columns, dump them as files
        _extract_images(dataset, save_dir)

        print("[INFO] HuggingFace download successful.")
        return True

    except Exception as e:
        print(f"[WARN] HuggingFace download failed: {e}")
        return False


def _extract_images(dataset, save_dir: Path):
    """
    If the dataset contains PIL Image columns, save them as PNG files.
    Column names checked: 'image', 'student_answer_image', 'handwriting_image'.
    """
    from PIL import Image as PILImage

    image_columns = ["image", "student_answer_image", "handwriting_image", "img"]

    for split_name, split_data in dataset.items():
        available_cols = split_data.column_names
        img_col = next((c for c in image_columns if c in available_cols), None)

        if img_col is None:
            print(f"  [INFO] No image column found in split '{split_name}'. Skipping image extraction.")
            continue

        img_dir = save_dir / "images" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Extracting images from column '{img_col}' in split '{split_name}' ...")
        for idx, row in enumerate(split_data):
            img_obj = row[img_col]
            if img_obj is None:
                continue
            # HuggingFace returns PIL Images directly for image features
            if isinstance(img_obj, PILImage.Image):
                img_obj.save(img_dir / f"{idx:05d}.png")
            elif isinstance(img_obj, dict) and "bytes" in img_obj:
                # Some datasets store images as {"bytes": ..., "path": ...}
                import io
                img = PILImage.open(io.BytesIO(img_obj["bytes"]))
                img.save(img_dir / f"{idx:05d}.png")

        print(f"  Images saved to {img_dir}")


def try_github_clone(save_dir: Path) -> bool:
    """
    Fallback: clone the FERMAT GitHub repo directly.
    Requires git to be installed.
    """
    try:
        import subprocess
        save_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_REPO_URL, str(save_dir / "FERMAT_github")],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"[WARN] Git clone failed:\n{result.stderr}")
            return False
        print(f"[INFO] GitHub clone successful → {save_dir / 'FERMAT_github'}")
        _restructure_github_clone(save_dir / "FERMAT_github", save_dir)
        return True
    except Exception as e:
        print(f"[WARN] GitHub clone failed: {e}")
        return False


def _restructure_github_clone(clone_dir: Path, save_dir: Path):
    """
    GitHub clone has its own layout; normalize it to match our expected structure:
    images/ → per-sample PNG files
    *.json / *.csv → metadata files
    """
    # Copy any JSON/CSV metadata files up
    for ext in ["*.json", "*.csv", "*.jsonl"]:
        for f in clone_dir.rglob(ext):
            shutil.copy2(f, save_dir / f.name)
            print(f"  Copied metadata: {f.name}")

    # Copy image directories
    for img_dir in clone_dir.rglob("images"):
        if img_dir.is_dir():
            dest = save_dir / "images" / img_dir.parent.name
            shutil.copytree(img_dir, dest, dirs_exist_ok=True)
            print(f"  Copied images: {img_dir} → {dest}")


def create_dummy_dataset(save_dir: Path):
    """
    Create a tiny synthetic dataset for pipeline testing when neither HF nor GitHub works.
    This lets the rest of the pipeline run end-to-end without real data.
    """
    print("[INFO] Creating synthetic demo dataset for pipeline testing ...")
    save_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    samples = [
        {
            "id": "demo_001",
            "question": "Solve: 2x + 5 = 13",
            "reference_answer": "x = 4",
            "human_score": 1.0,
            "language_track": "english",
        },
        {
            "id": "demo_002",
            "question": "What is the derivative of f(x) = x^2 + 3x?",
            "reference_answer": "f'(x) = 2x + 3",
            "human_score": 0.8,
            "language_track": "english",
        },
        {
            "id": "demo_003",
            "question": "احسب مساحة المثلث ذي القاعدة 6 والارتفاع 4",
            "reference_answer": "المساحة = 12",
            "human_score": 0.9,
            "language_track": "arabic",
        },
        {
            "id": "demo_004",
            "question": "State Newton's second law of motion.",
            "reference_answer": "F = ma",
            "human_score": 0.5,
            "language_track": "english",
        },
        {
            "id": "demo_005",
            "question": "ما هي قيمة cos(0)؟",
            "reference_answer": "cos(0) = 1",
            "human_score": 1.0,
            "language_track": "arabic",
        },
    ]

    img_dir = save_dir / "images" / "demo"
    img_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        # Create a minimal handwriting-like grayscale image with the answer text
        img = Image.new("L", (400, 100), color=240)
        draw = ImageDraw.Draw(img)
        # Draw placeholder text (real handwriting would come from the dataset)
        draw.text((10, 30), f"Answer: {sample['reference_answer']}", fill=20)
        img_path = img_dir / f"{sample['id']}.png"
        img.save(img_path)
        sample["image_path"] = str(img_path.relative_to(save_dir))

    # Save metadata
    meta_path = save_dir / "demo.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[INFO] Demo dataset created: {len(samples)} samples → {save_dir}")


def main():
    print("=" * 60)
    print("FERMAT Dataset Download")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try HuggingFace first (preferred)
    success = try_huggingface_download(DATA_DIR)

    if not success:
        # Try GitHub clone
        success = try_github_clone(DATA_DIR)

    if not success:
        # Fall back to synthetic demo data so the pipeline can still run
        print("[WARN] Both download methods failed. Creating synthetic demo dataset.")
        create_dummy_dataset(DATA_DIR)

    # Write a manifest so downstream stages know where data lives
    manifest = {
        "data_dir": str(DATA_DIR),
        "download_status": "real" if success else "demo",
    }
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[INFO] Manifest written: {manifest_path}")
    print("Stage 1 — Download complete.\n")


if __name__ == "__main__":
    main()
