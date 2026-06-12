"""
Exploratory Data Analysis (EDA) for the FERMAT dataset.

Produces:
  - Console summary: sample counts, score distribution, language breakdown
  - ../data/eda/score_distribution.png
  - ../data/eda/sample_grid.png   (random image grid)
  - ../data/eda/stats.json        (machine-readable stats)

Run after download_dataset.py.
"""

import json
import os
import random
from pathlib import Path
from collections import Counter

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "fermat_raw"
EDA_DIR = REPO_ROOT / "data" / "eda"

# Candidate column names across different dataset versions
SCORE_COLUMNS = ["human_score", "score", "label", "grade"]
TEXT_COLUMNS = ["question", "query", "problem"]
ANSWER_COLUMNS = ["reference_answer", "answer", "solution", "ref"]
IMAGE_COLUMNS = ["image_path", "image", "img_path"]
LANG_COLUMNS = ["language_track", "language", "lang"]


def load_jsonl_files(data_dir: Path) -> list[dict]:
    """Load all JSONL files from the raw data directory."""
    records = []
    for path in data_dir.glob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def resolve_column(record: dict, candidates: list[str]):
    """Return the value of the first matching key from candidates."""
    for col in candidates:
        if col in record:
            return record[col]
    return None


def detect_language_heuristic(text: str) -> str:
    """
    Detect script type from a text string using Unicode range checks.
    Arabic Unicode block: U+0600–U+06FF
    Returns: 'arabic', 'english', or 'mixed'
    """
    if not text:
        return "unknown"
    arabic_chars = sum(1 for c in text if "؀" <= c <= "ۿ")
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return "unknown"
    arabic_ratio = arabic_chars / total_alpha
    if arabic_ratio > 0.6:
        return "arabic"
    elif arabic_ratio > 0.1:
        return "mixed"
    return "english"


def analyze_records(records: list[dict]) -> dict:
    """Compute summary statistics over all records."""
    scores = []
    languages = []
    has_image = []

    for rec in records:
        # Score
        score = resolve_column(rec, SCORE_COLUMNS)
        if score is not None:
            try:
                scores.append(float(score))
            except (ValueError, TypeError):
                pass

        # Language: use explicit column or detect from question text
        lang = resolve_column(rec, LANG_COLUMNS)
        if lang is None:
            question = resolve_column(rec, TEXT_COLUMNS) or ""
            lang = detect_language_heuristic(str(question))
        languages.append(str(lang))

        # Image presence
        img = resolve_column(rec, IMAGE_COLUMNS)
        has_image.append(img is not None)

    stats = {
        "total_samples": len(records),
        "samples_with_scores": len(scores),
        "samples_with_images": sum(has_image),
        "language_distribution": dict(Counter(languages)),
    }

    if scores:
        stats["score_stats"] = {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "median": float(np.median(scores)),
        }

    return stats, scores, languages


def plot_score_distribution(scores: list[float], save_path: Path):
    """Save a histogram of human scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(scores, bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Human Score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("FERMAT — Human Score Distribution", fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except ImportError:
        print("  [SKIP] matplotlib not available — skipping score plot")


def plot_sample_grid(records: list[dict], data_dir: Path, save_path: Path, n: int = 9):
    """Save a grid of n random sample images from the dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        # Collect image paths
        image_paths = []
        for rec in records:
            img_val = resolve_column(rec, IMAGE_COLUMNS)
            if img_val and isinstance(img_val, str):
                p = data_dir / img_val
                if p.exists():
                    image_paths.append(p)

        # Also scan the images/ subdirectory
        for p in (data_dir / "images").rglob("*.png") if (data_dir / "images").exists() else []:
            image_paths.append(p)

        if not image_paths:
            print("  [SKIP] No images found for sample grid.")
            return

        selected = random.sample(image_paths, min(n, len(image_paths)))
        cols = 3
        rows = (len(selected) + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten() if rows * cols > 1 else [axes]

        for ax, img_path in zip(axes, selected):
            img = Image.open(img_path).convert("L")
            ax.imshow(img, cmap="gray")
            ax.set_title(img_path.name, fontsize=8)
            ax.axis("off")

        for ax in axes[len(selected):]:
            ax.axis("off")

        fig.suptitle("FERMAT — Sample Handwritten Answers", fontsize=14)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {save_path}")
    except ImportError:
        print("  [SKIP] matplotlib/PIL not available — skipping sample grid")


def main():
    print("=" * 60)
    print("FERMAT Dataset — Exploratory Data Analysis")
    print("=" * 60)

    EDA_DIR.mkdir(parents=True, exist_ok=True)

    records = load_jsonl_files(RAW_DIR)
    if not records:
        print(f"[ERROR] No JSONL files found in {RAW_DIR}. Run download_dataset.py first.")
        return

    print(f"\n[INFO] Loaded {len(records)} records from {RAW_DIR}\n")

    stats, scores, languages = analyze_records(records)

    print("--- Summary Statistics ---")
    print(f"  Total samples       : {stats['total_samples']}")
    print(f"  With scores         : {stats['samples_with_scores']}")
    print(f"  With images         : {stats['samples_with_images']}")

    print("\n  Language distribution:")
    for lang, count in sorted(stats["language_distribution"].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats["total_samples"]
        print(f"    {lang:<12} : {count:>5}  ({pct:.1f}%)")

    if "score_stats" in stats:
        s = stats["score_stats"]
        print(f"\n  Score stats:")
        print(f"    min={s['min']:.3f}  max={s['max']:.3f}  mean={s['mean']:.3f}  std={s['std']:.3f}")

    # Save stats JSON
    stats_path = EDA_DIR / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    # Plots
    if scores:
        plot_score_distribution(scores, EDA_DIR / "score_distribution.png")

    plot_sample_grid(records, RAW_DIR, EDA_DIR / "sample_grid.png")

    print("\nStage 1 — EDA complete.\n")


if __name__ == "__main__":
    main()
