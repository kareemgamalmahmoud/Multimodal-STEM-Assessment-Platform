"""
Region Detector — classify image sub-regions as Arabic text, English text, or math expression.

Strategy:
  1. Connected-Component Analysis (CCA) to find text lines.
  2. For each line region, compute features:
       - Aspect ratio (math tends to be wide, Arabic text denser)
       - Pixel density
       - Presence of math-like characters (operators, digits only, etc.)
  3. A lightweight heuristic classifier assigns a label: 'arabic', 'english', 'math'.

This avoids training a dedicated CNN region classifier, keeping hardware requirements low.
For production, replace with a trained layout analysis model (e.g., DocLayout-YOLO).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Region:
    """One detected region from an image."""
    x: int       # left pixel
    y: int       # top pixel
    w: int       # width
    h: int       # height
    label: str   # 'arabic', 'english', or 'math'
    confidence: float


def load_binary_image(image_path: Path) -> np.ndarray:
    """Load image and convert to binary (black ink on white background)."""
    try:
        import cv2
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except ImportError:
        from PIL import Image
        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        threshold = arr.mean()
        return (arr < threshold).astype(np.uint8) * 255


def find_line_regions(binary: np.ndarray, min_height: int = 8) -> List[tuple]:
    """
    Segment the image into horizontal text lines using horizontal projection.
    Returns list of (y_start, y_end) row ranges where ink is present.
    """
    # Horizontal projection: count black pixels per row
    ink_rows = (binary < 128).astype(int).sum(axis=1)

    in_line = False
    line_starts = []
    y_start = 0
    for y, count in enumerate(ink_rows):
        if count > 0 and not in_line:
            y_start = y
            in_line = True
        elif count == 0 and in_line:
            if y - y_start >= min_height:
                line_starts.append((y_start, y))
            in_line = False
    if in_line and (len(ink_rows) - y_start) >= min_height:
        line_starts.append((y_start, len(ink_rows)))

    return line_starts


def classify_region(line_crop: np.ndarray) -> tuple[str, float]:
    """
    Heuristic classifier for one line crop.
    Returns (label, confidence) where label is 'arabic', 'english', or 'math'.

    Heuristics:
      - Math lines tend to have isolated symbols, low pixel density, wide aspect ratio.
      - Arabic lines have a connected cursive baseline, moderate pixel density.
      - English lines are intermediate.
    """
    h, w = line_crop.shape
    if h == 0 or w == 0:
        return "unknown", 0.0

    # Ink pixel density (fraction of black pixels in the bounding box)
    ink_pixels = (line_crop < 128).sum()
    density = ink_pixels / (h * w)

    # Aspect ratio
    aspect = w / max(h, 1)

    # Horizontal gap ratio: how many column-runs of white space exist?
    ink_cols = (line_crop < 128).sum(axis=0)
    gap_count = sum(
        1 for i in range(1, len(ink_cols)) if ink_cols[i - 1] > 0 and ink_cols[i] == 0
    )
    gap_ratio = gap_count / max(w, 1)

    # Math signals: very low density, many isolated components, high gap ratio
    if density < 0.08 and gap_ratio > 0.25:
        return "math", 0.75

    # Arabic signals: higher density (cursive), moderate gaps
    if density > 0.12 and gap_ratio < 0.25:
        return "arabic", 0.70

    # Default: English (printed/handwritten Latin)
    return "english", 0.60


def detect_regions(image_path: Path) -> List[Region]:
    """
    Main entry point.  Returns a list of Region objects for the given image.
    Each Region has (x, y, w, h, label, confidence).
    """
    binary = load_binary_image(image_path)
    h_total, w_total = binary.shape

    line_ranges = find_line_regions(binary)

    regions = []
    for y_start, y_end in line_ranges:
        line_crop = binary[y_start:y_end, :]
        label, conf = classify_region(line_crop)
        regions.append(Region(
            x=0, y=y_start,
            w=w_total, h=y_end - y_start,
            label=label, confidence=conf,
        ))

    return regions


def crop_region(image_path: Path, region: Region) -> np.ndarray:
    """Return the numpy array crop for a given Region."""
    binary = load_binary_image(image_path)
    return binary[region.y: region.y + region.h, region.x: region.x + region.w]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python region_detector.py <image_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    found = detect_regions(path)
    for r in found:
        print(f"  [{r.label:<8}] y={r.y:>4}–{r.y+r.h:<4}  conf={r.confidence:.2f}")
