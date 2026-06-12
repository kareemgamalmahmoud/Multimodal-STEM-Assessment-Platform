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
SCORE_COLUMNS = ["human_score", "score", "label", "grade", "mark", "marks", "total_score"]
TEXT_COLUMNS = ["question", "query", "problem", "question_text", "prompt"]
ANSWER_COLUMNS = ["reference_answer", "answer", "solution", "ref", "gold_answer", "correct_answer"]
IMAGE_COLUMNS = ["image_path", "image", "img_path", "scan", "handwriting_image", "student_answer_image"]
LANG_COLUMNS = ["language_track", "language", "lang", "script"]


def load_data_files(data_dir: Path) -> list[dict]:
    """
    Load records from any supported format in data_dir:
    JSONL (preferred) → JSON → CSV → TSV → Parquet.
    Also recursively checks the FERMAT_github subdirectory as fallback.
    """
    records = []

    def _read_jsonl(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return rows

    def _read_csv(path, sep=","):
        try:
            import pandas as pd
            df = pd.read_csv(path, sep=sep, low_memory=False)
            df.columns = [str(c).lower().strip() for c in df.columns]
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"  [WARN] Could not read {path.name}: {e}")
            return []

    def _read_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Could be {split: [records]} format
                all_rows = []
                for v in data.values():
                    if isinstance(v, list):
                        all_rows.extend(v)
                return all_rows if all_rows else [data]
        except Exception as e:
            print(f"  [WARN] Could not read {path.name}: {e}")
            return []

    def _read_parquet(path):
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            df.columns = [str(c).lower().strip() for c in df.columns]
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"  [WARN] Could not read {path.name}: {e}")
            return []

    # Search order: data_dir itself, then FERMAT_github subdirectory
    search_dirs = [data_dir]
    github_clone = data_dir / "FERMAT_github"
    if github_clone.exists():
        search_dirs.append(github_clone)
        for sub in github_clone.iterdir():
            if sub.is_dir() and sub.name != ".git":
                search_dirs.append(sub)

    skip_names = {"manifest.json"}

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for path in sorted(search_dir.iterdir()):
            if path.is_dir() or path.name in skip_names:
                continue
            if path.suffix == ".jsonl":
                rows = _read_jsonl(path)
                if rows:
                    records.extend(rows)
                    print(f"  Loaded {len(rows)} rows from {path.relative_to(data_dir)}")
            elif path.suffix == ".csv":
                rows = _read_csv(path, sep=",")
                if rows:
                    records.extend(rows)
                    print(f"  Loaded {len(rows)} rows from {path.relative_to(data_dir)}")
            elif path.suffix == ".tsv":
                rows = _read_csv(path, sep="\t")
                if rows:
                    records.extend(rows)
                    print(f"  Loaded {len(rows)} rows from {path.relative_to(data_dir)}")
            elif path.suffix == ".json" and path.name != "manifest.json":
                rows = _read_json(path)
                if rows:
                    records.extend(rows)
                    print(f"  Loaded {len(rows)} rows from {path.relative_to(data_dir)}")
            elif path.suffix == ".parquet":
                rows = _read_parquet(path)
                if rows:
                    records.extend(rows)
                    print(f"  Loaded {len(rows)} rows from {path.relative_to(data_dir)}")

        if records:
            break  # found data in this directory — don't keep searching deeper

    return records


# Keep old name as alias for compatibility
def load_jsonl_files(data_dir: Path) -> list[dict]:
    return load_data_files(data_dir)


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

    print(f"[INFO] Scanning {RAW_DIR} for data files ...")
    records = load_data_files(RAW_DIR)
    if not records:
        print(
            f"\n[ERROR] No data found in {RAW_DIR}.\n"
            "  Options:\n"
            "  1. Set HF_TOKEN and re-run: python download_dataset.py --token hf_xxxxx\n"
            "  2. Use demo data:            python download_dataset.py --demo\n"
            "  3. Check the clone manually: ls data/fermat_raw/FERMAT_github/"
        )
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
