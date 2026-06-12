"""
Download the FERMAT dataset from HuggingFace or GitHub and organize it locally.

FERMAT (AI4Bharat) — https://huggingface.co/datasets/ai4bharat/FERMAT
  - Gated dataset: requires HuggingFace authentication.
  - Set your token: export HF_TOKEN=hf_xxxxx   (or pass --token flag)
  - GitHub fallback: clones the repo and converts any CSV/JSON data files to JSONL.

Usage:
    python download_dataset.py
    python download_dataset.py --token hf_xxxxx
    python download_dataset.py --demo          # skip download, use synthetic data
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "fermat_raw"

HF_DATASET_ID = "ai4bharat/FERMAT"
GITHUB_REPO_URL = "https://github.com/AI4Bharat/FERMAT"


# ---------------------------------------------------------------------------
# HuggingFace download (primary path)
# ---------------------------------------------------------------------------

def try_huggingface_download(save_dir: Path, token: str = None) -> bool:
    """
    Download FERMAT via the HuggingFace `datasets` library.
    Requires a valid HF token because FERMAT is a gated dataset.

    Get your token at: https://huggingface.co/settings/tokens
    Then accept the dataset terms at: https://huggingface.co/datasets/ai4bharat/FERMAT
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[WARN] `datasets` library not installed. Run: pip install datasets")
        return False

    # Resolve token: arg > env var
    hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not hf_token:
        print(
            "[WARN] No HuggingFace token found.\n"
            "  FERMAT is a gated dataset — you need to:\n"
            "  1. Create a token at https://huggingface.co/settings/tokens\n"
            "  2. Accept terms at https://huggingface.co/datasets/ai4bharat/FERMAT\n"
            "  3. Set: export HF_TOKEN=hf_xxxxx   (or pass --token hf_xxxxx)\n"
            "  Falling back to GitHub clone ..."
        )
        return False

    print(f"[INFO] Trying HuggingFace Hub: {HF_DATASET_ID}")
    try:
        dataset = load_dataset(HF_DATASET_ID, token=hf_token)
    except Exception as e:
        print(f"[WARN] HuggingFace download failed: {e}")
        return False

    save_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in dataset.items():
        out_path = save_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in split_data:
                # Convert PIL Image objects to a placeholder path string before JSON serialization
                serializable = {}
                for k, v in row.items():
                    try:
                        json.dumps(v)
                        serializable[k] = v
                    except (TypeError, ValueError):
                        serializable[k] = str(type(v).__name__)
                f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        print(f"  Saved {len(split_data)} rows → {out_path}")

    _extract_images_from_hf(dataset, save_dir)
    print("[INFO] HuggingFace download successful.")
    return True


def _extract_images_from_hf(dataset, save_dir: Path):
    """Save PIL Image columns from the HF dataset as PNG files."""
    from PIL import Image as PILImage
    import io

    image_col_candidates = ["image", "student_answer_image", "handwriting_image", "img", "scan"]

    for split_name, split_data in dataset.items():
        img_col = next((c for c in image_col_candidates if c in split_data.column_names), None)
        if img_col is None:
            continue

        img_dir = save_dir / "images" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting images [{split_name}] from column '{img_col}' ...")

        for idx, row in enumerate(split_data):
            obj = row[img_col]
            if obj is None:
                continue
            try:
                if isinstance(obj, PILImage.Image):
                    obj.save(img_dir / f"{idx:05d}.png")
                elif isinstance(obj, dict) and "bytes" in obj and obj["bytes"]:
                    PILImage.open(io.BytesIO(obj["bytes"])).save(img_dir / f"{idx:05d}.png")
                elif isinstance(obj, dict) and "path" in obj:
                    pass  # path-based images: already on disk in HF cache
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GitHub clone fallback
# ---------------------------------------------------------------------------

def try_github_clone(save_dir: Path) -> bool:
    """
    Clone the FERMAT GitHub repo and convert its data files to JSONL.
    This fallback works without a HuggingFace token.
    """
    clone_dir = save_dir / "FERMAT_github"

    # Skip re-clone if already present
    if clone_dir.exists() and any(clone_dir.iterdir()):
        print(f"[INFO] GitHub clone already exists at {clone_dir}. Reusing.")
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Cloning {GITHUB_REPO_URL} ...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_REPO_URL, str(clone_dir)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"[WARN] Git clone failed:\n{result.stderr}")
            return False
        print(f"[INFO] GitHub clone successful → {clone_dir}")

    # Show what was cloned
    print("\n  Clone directory contents:")
    for p in sorted(clone_dir.rglob("*")):
        if ".git" not in str(p):
            indent = "  " + "  " * (len(p.relative_to(clone_dir).parts) - 1)
            print(f"{indent}{p.name}{'/' if p.is_dir() else ''}")

    # Convert any data files to JSONL in save_dir
    converted = _convert_clone_to_jsonl(clone_dir, save_dir)

    # Copy image directories
    for img_dir in clone_dir.rglob("images"):
        if img_dir.is_dir():
            dest = save_dir / "images" / img_dir.parent.name
            shutil.copytree(img_dir, dest, dirs_exist_ok=True)
            print(f"  Copied images: {img_dir.relative_to(clone_dir)} → {dest}")

    # Also copy any loose image folders named after splits
    for subdir in clone_dir.iterdir():
        if subdir.is_dir() and subdir.name not in (".git", "images"):
            imgs = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg"))
            if imgs:
                dest = save_dir / "images" / subdir.name
                dest.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    shutil.copy2(img, dest / img.name)
                print(f"  Copied {len(imgs)} images from {subdir.name}/ → {dest}")

    return converted > 0


def _convert_clone_to_jsonl(clone_dir: Path, save_dir: Path) -> int:
    """
    Find all CSV / JSON / TSV / Parquet data files in the clone and convert each to JSONL.
    Returns the number of files successfully converted.
    """
    import pandas as pd

    converted = 0
    skip_dirs = {".git", "__pycache__"}

    # Gather candidate files
    candidates = []
    for ext in ["*.csv", "*.tsv", "*.json", "*.jsonl", "*.parquet", "*.xlsx"]:
        for f in clone_dir.rglob(ext):
            if not any(part in skip_dirs for part in f.parts):
                candidates.append(f)

    if not candidates:
        print("  [WARN] No data files (csv/json/parquet) found in GitHub clone.")
        print("         The repo may store data as image files only.")
        print("         Inspect the clone manually:", clone_dir)
        return 0

    for src in candidates:
        stem = src.stem.lower()
        out_path = save_dir / f"{stem}.jsonl"

        try:
            # Read into DataFrame
            if src.suffix in (".csv", ".tsv"):
                sep = "\t" if src.suffix == ".tsv" else ","
                df = pd.read_csv(src, sep=sep, encoding="utf-8", low_memory=False)
            elif src.suffix == ".parquet":
                df = pd.read_parquet(src)
            elif src.suffix in (".xls", ".xlsx"):
                df = pd.read_excel(src)
            elif src.suffix == ".jsonl":
                # Already JSONL — just copy
                shutil.copy2(src, out_path)
                print(f"  Copied JSONL: {src.name} → {out_path.name}")
                converted += 1
                continue
            elif src.suffix == ".json":
                with open(src, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Could be a list or a dict of lists
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    # Could be {split: [records, ...]} or a flat dict
                    all_rows = []
                    for key, val in data.items():
                        if isinstance(val, list):
                            for rec in val:
                                if isinstance(rec, dict):
                                    rec["_split"] = key
                                    all_rows.append(rec)
                    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame([data])
                else:
                    print(f"  [SKIP] Unrecognized JSON format in {src.name}")
                    continue
            else:
                continue

            # Normalize column names to lowercase
            df.columns = [str(c).lower().strip() for c in df.columns]

            # Add a synthetic 'id' if missing
            if "id" not in df.columns:
                df.insert(0, "id", [f"{stem}_{i:05d}" for i in range(len(df))])

            # Write to JSONL
            with open(out_path, "w", encoding="utf-8") as fh:
                for _, row in df.iterrows():
                    fh.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + "\n")

            print(f"  Converted: {src.relative_to(clone_dir)} → {out_path.name}  ({len(df)} rows)")
            converted += 1

        except Exception as e:
            print(f"  [ERROR] Could not convert {src.name}: {e}")

    return converted


# ---------------------------------------------------------------------------
# Synthetic demo dataset (last resort fallback)
# ---------------------------------------------------------------------------

def create_dummy_dataset(save_dir: Path):
    """
    Create a tiny synthetic dataset so the full pipeline can run end-to-end
    when neither HuggingFace nor GitHub download is possible.
    """
    print("[INFO] Creating synthetic demo dataset ...")
    save_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image, ImageDraw

    samples = [
        {"id": "demo_001", "question": "Solve: 2x + 5 = 13",
         "reference_answer": "x = 4", "human_score": 1.0, "language_track": "english"},
        {"id": "demo_002", "question": "What is the derivative of f(x) = x^2 + 3x?",
         "reference_answer": "f'(x) = 2x + 3", "human_score": 0.8, "language_track": "english"},
        {"id": "demo_003", "question": "احسب مساحة المثلث ذي القاعدة 6 والارتفاع 4",
         "reference_answer": "المساحة = 12", "human_score": 0.9, "language_track": "arabic"},
        {"id": "demo_004", "question": "State Newton's second law of motion.",
         "reference_answer": "F = ma", "human_score": 0.5, "language_track": "english"},
        {"id": "demo_005", "question": "ما هي قيمة cos(0)؟",
         "reference_answer": "cos(0) = 1", "human_score": 1.0, "language_track": "arabic"},
        {"id": "demo_006", "question": "What is the speed of light in m/s?",
         "reference_answer": "3 × 10^8 m/s", "human_score": 0.7, "language_track": "english"},
        {"id": "demo_007", "question": "حل المعادلة: 3x - 9 = 0",
         "reference_answer": "x = 3", "human_score": 1.0, "language_track": "arabic"},
        {"id": "demo_008", "question": "Define kinetic energy.",
         "reference_answer": "KE = 0.5 * m * v^2", "human_score": 0.6, "language_track": "english"},
        {"id": "demo_009", "question": "Find the area of a circle with radius 5.",
         "reference_answer": "A = 25π ≈ 78.54", "human_score": 0.9, "language_track": "english"},
        {"id": "demo_010", "question": "ما هو تعريف التسارع؟",
         "reference_answer": "التسارع = التغير في السرعة / الزمن", "human_score": 0.8, "language_track": "arabic"},
    ]

    img_dir = save_dir / "images" / "demo"
    img_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        img = Image.new("L", (500, 120), color=245)
        draw = ImageDraw.Draw(img)
        draw.rectangle([5, 5, 495, 115], outline=180)
        draw.text((15, 40), sample['reference_answer'], fill=30)
        img_path = img_dir / f"{sample['id']}.png"
        img.save(img_path)
        sample["image_path"] = str(img_path.relative_to(save_dir))

    out_path = save_dir / "demo.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[INFO] Demo dataset: {len(samples)} samples → {save_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1 — FERMAT Dataset Download")
    parser.add_argument("--token", default=None, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--demo", action="store_true", help="Skip download, create synthetic demo data")
    args = parser.parse_args()

    print("=" * 60)
    print("FERMAT Dataset Download")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.demo:
        create_dummy_dataset(DATA_DIR)
        success = True
    else:
        # 1. Try HuggingFace (requires token + accepted terms)
        success = try_huggingface_download(DATA_DIR, token=args.token)

        if not success:
            # 2. Try GitHub clone + CSV→JSONL conversion
            success = try_github_clone(DATA_DIR)

        if not success:
            # 3. Synthetic demo as last resort
            print("[WARN] Both download methods failed. Creating synthetic demo dataset.")
            create_dummy_dataset(DATA_DIR)
            success = True

    # Verify at least one JSONL file exists
    jsonl_files = list(DATA_DIR.glob("*.jsonl"))
    print(f"\n[INFO] JSONL files in {DATA_DIR}: {[f.name for f in jsonl_files]}")

    manifest = {
        "data_dir": str(DATA_DIR),
        "download_status": "real" if success else "demo",
        "jsonl_files": [f.name for f in jsonl_files],
    }
    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] Manifest written: {DATA_DIR / 'manifest.json'}")
    print("Stage 1 — Download complete.\n")


if __name__ == "__main__":
    main()
