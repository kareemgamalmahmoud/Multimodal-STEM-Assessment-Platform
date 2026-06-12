"""
Image preprocessing pipeline for FERMAT handwritten answer images.

Processing steps applied to each image:
  1. Grayscale conversion
  2. Otsu binarization (adaptive thresholding for uneven lighting)
  3. Deskew (correct tilt up to ±15 degrees using Hough transform)
  4. Border padding (uniform white border)
  5. Resize to fixed height (preserving aspect ratio)

Output: ../data/fermat_processed/
  - images/<id>.png       (processed images)
  - metadata.csv          (all sample metadata with updated image paths)

Run after download_dataset.py.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "fermat_raw"
PROCESSED_DIR = REPO_ROOT / "data" / "fermat_processed"

TARGET_HEIGHT = 128   # pixels — tall enough for handwriting detail, small enough for fast inference
PAD_SIZE = 10         # white border in pixels


def load_metadata(raw_dir: Path) -> list[dict]:
    """Load metadata from any supported format (delegates to explore_dataset loader)."""
    # Reuse the multi-format loader from explore_dataset
    import importlib.util, sys as _sys
    loader_path = Path(__file__).parent / "explore_dataset.py"
    spec = importlib.util.spec_from_file_location("explore_dataset", loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_data_files(raw_dir)




def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's global thresholding to produce a binary image.
    Pixels below the threshold → black (foreground ink).
    Pixels above → white (paper background).
    """
    try:
        import cv2
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except ImportError:
        # Fallback: manual Otsu via numpy
        hist, _ = np.histogram(gray.flatten(), 256, [0, 256])
        hist = hist.astype(float) / hist.sum()
        best_thresh, best_var = 0, 0.0
        for t in range(1, 256):
            w0, w1 = hist[:t].sum(), hist[t:].sum()
            if w0 == 0 or w1 == 0:
                continue
            mu0 = (hist[:t] * np.arange(t)).sum() / w0
            mu1 = (hist[t:] * np.arange(t, 256)).sum() / w1
            var = w0 * w1 * (mu0 - mu1) ** 2
            if var > best_var:
                best_var, best_thresh = var, t
        return (gray > best_thresh).astype(np.uint8) * 255


def deskew(binary: np.ndarray) -> np.ndarray:
    """
    Detect and correct skew angle using the Hough line transform.
    Falls back to no correction if cv2 is unavailable or no lines detected.
    """
    try:
        import cv2
        # Invert so ink is white (for line detection)
        inv = cv2.bitwise_not(binary)
        lines = cv2.HoughLinesP(
            inv, rho=1, theta=np.pi / 180, threshold=50,
            minLineLength=30, maxLineGap=10,
        )
        if lines is None:
            return binary

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 15:  # ignore near-vertical lines
                    angles.append(angle)

        if not angles:
            return binary

        median_angle = float(np.median(angles))
        h, w = binary.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            binary, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderValue=255,  # fill with white
        )
        return rotated
    except ImportError:
        return binary


def resize_fixed_height(img_array: np.ndarray, target_h: int) -> np.ndarray:
    """Resize image to target_h pixels tall, preserving aspect ratio."""
    try:
        import cv2
        h, w = img_array.shape[:2]
        if h == 0:
            return img_array
        scale = target_h / h
        new_w = max(1, int(w * scale))
        return cv2.resize(img_array, (new_w, target_h), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img_array)
        h, w = img_array.shape[:2]
        scale = target_h / h
        new_w = max(1, int(w * scale))
        pil = pil.resize((new_w, target_h), PILImage.LANCZOS)
        return np.array(pil)


def add_padding(img_array: np.ndarray, pad: int) -> np.ndarray:
    """Add a uniform white border around the image."""
    return np.pad(img_array, pad_width=pad, mode="constant", constant_values=255)


def preprocess_image(image_path: Path) -> np.ndarray:
    """
    Full preprocessing pipeline for one image.
    Returns a processed numpy array (uint8, single channel).
    """
    from PIL import Image

    # Load as grayscale
    img = Image.open(image_path).convert("L")
    gray = np.array(img, dtype=np.uint8)

    # Binarize
    binary = otsu_binarize(gray)

    # Deskew
    straight = deskew(binary)

    # Resize to fixed height
    resized = resize_fixed_height(straight, TARGET_HEIGHT)

    # Pad
    padded = add_padding(resized, PAD_SIZE)

    return padded


def detect_language_from_metadata(record: dict) -> str:
    """
    Determine language track from record metadata.
    Priority: explicit language field > heuristic from question text.
    """
    for key in ["language_track", "language", "lang"]:
        if key in record and record[key]:
            return str(record[key]).lower()

    # Heuristic: count Arabic Unicode characters in question
    question = str(record.get("question", "") or record.get("query", "") or "")
    arabic_chars = sum(1 for c in question if "؀" <= c <= "ۿ")
    total_alpha = sum(1 for c in question if c.isalpha())
    if total_alpha == 0:
        return "unknown"
    ratio = arabic_chars / total_alpha
    if ratio > 0.6:
        return "arabic"
    elif ratio > 0.1:
        return "mixed"
    return "english"


def main():
    import pandas as pd
    from PIL import Image

    print("=" * 60)
    print("FERMAT — Image Preprocessing")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_img_dir = PROCESSED_DIR / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    records = load_metadata(RAW_DIR)
    if not records:
        print(f"[ERROR] No metadata found in {RAW_DIR}. Run download_dataset.py first.")
        sys.exit(1)

    print(f"[INFO] Processing {len(records)} samples ...\n")

    processed_rows = []
    skipped = 0

    # Gather all available image files for matching by ID
    all_images = {}
    for img_path in (RAW_DIR / "images").rglob("*.png") if (RAW_DIR / "images").exists() else []:
        all_images[img_path.stem] = img_path
    for img_path in (RAW_DIR / "images").rglob("*.jpg") if (RAW_DIR / "images").exists() else []:
        all_images[img_path.stem] = img_path

    for i, rec in enumerate(records):
        sample_id = str(rec.get("id", f"sample_{i:05d}"))

        # Resolve image path
        raw_img_path = None
        # 1) Explicit path in metadata
        for col in ["image_path", "img_path"]:
            if col in rec and rec[col]:
                candidate = RAW_DIR / str(rec[col])
                if candidate.exists():
                    raw_img_path = candidate
                    break
        # 2) Match by sample ID in collected images
        if raw_img_path is None and sample_id in all_images:
            raw_img_path = all_images[sample_id]

        if raw_img_path is None:
            skipped += 1
            if skipped <= 5:
                print(f"  [SKIP] No image for {sample_id}")
            continue

        try:
            processed = preprocess_image(raw_img_path)
            out_path = out_img_dir / f"{sample_id}.png"
            Image.fromarray(processed).save(out_path)

            # Build metadata row
            row = {
                "id": sample_id,
                "question": rec.get("question") or rec.get("query") or "",
                "reference_answer": (
                    rec.get("reference_answer") or rec.get("answer") or rec.get("solution") or ""
                ),
                "image_path": str(out_path.relative_to(PROCESSED_DIR)),
                "human_score": rec.get("human_score") or rec.get("score") or rec.get("label") or None,
                "language_track": detect_language_from_metadata(rec),
            }
            processed_rows.append(row)

            if (i + 1) % 100 == 0 or i < 5:
                print(f"  [{i+1:>5}/{len(records)}] {sample_id} → {out_path.name}")

        except Exception as e:
            print(f"  [ERROR] Failed on {sample_id}: {e}")
            skipped += 1

    # Save metadata CSV
    df = pd.DataFrame(processed_rows)
    csv_path = PROCESSED_DIR / "metadata.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\n[DONE] Processed: {len(processed_rows)}  |  Skipped: {skipped}")
    print(f"  Metadata → {csv_path}")
    print(f"  Images   → {out_img_dir}")

    lang_counts = df["language_track"].value_counts()
    print("\n  Language track distribution:")
    for lang, cnt in lang_counts.items():
        print(f"    {lang:<12}: {cnt}")

    print("\nStage 1 — Preprocessing complete.\n")


if __name__ == "__main__":
    main()
