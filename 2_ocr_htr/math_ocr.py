"""
Mathematical Expression Recognition using pix2tex (LaTeX-OCR).

pix2tex (Blecher 2023) is a ViT + GPT-2 model trained to convert images of
mathematical expressions into LaTeX source code. It handles:
  - Inline math (single-line formulas)
  - Display math (multi-line, fractions, integrals, etc.)
  - Scientific notation (e.g., 6.02 × 10²³)

Model: naver-clova-ix/donut-base-finetuned-cord-v2 (pix2tex internal)
Install: pip install pix2tex

Reference: Blecher L. (2023). pix2tex: Using a ViT to convert images of equations
           into LaTeX. https://github.com/lukas-blecher/LaTeX-OCR
"""

from pathlib import Path
from typing import Optional

import numpy as np


class MathOCR:
    """
    Wrapper around pix2tex for LaTeX recognition from math expression images.
    Lazy-loads on first use.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from pix2tex.cli import LatexOCR
            print("[MathOCR] Loading pix2tex LatexOCR model ...")
            self._model = LatexOCR()
            print("[MathOCR] Model ready.")
        except ImportError:
            print(
                "[MathOCR] pix2tex not installed. "
                "Run: pip install pix2tex\n"
                "         Falling back to placeholder output."
            )
            self._model = None

    def recognize(self, image_input) -> str:
        """
        Convert an image of a math expression to LaTeX.

        Args:
            image_input: PIL Image, numpy array, or Path to image file.

        Returns:
            LaTeX string (e.g., '2x + 5 = 13') or empty string on failure.
        """
        self._load()

        from PIL import Image

        # Normalize to PIL
        if isinstance(image_input, Path):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        if self._model is None:
            # Graceful fallback: return a human-readable placeholder
            return "[math expression — pix2tex not available]"

        try:
            latex = self._model(pil_img)
            return latex.strip() if latex else ""
        except Exception as e:
            print(f"[MathOCR] Recognition error: {e}")
            return ""

    def recognize_batch(self, images: list) -> list[str]:
        """Recognize a list of math expression images sequentially."""
        return [self.recognize(img) for img in images]


def wrap_latex(latex: str) -> str:
    """
    Wrap a raw LaTeX expression in display math delimiters if not already wrapped.
    This standardizes the output for downstream use.
    """
    s = latex.strip()
    if not s:
        return s
    if s.startswith("$$") or s.startswith("\\["):
        return s
    return f"$${s}$$"


# Module-level singleton
_math_ocr_instance: Optional[MathOCR] = None


def get_math_ocr() -> MathOCR:
    global _math_ocr_instance
    if _math_ocr_instance is None:
        _math_ocr_instance = MathOCR()
    return _math_ocr_instance


def recognize_math(image_input, wrap: bool = False) -> str:
    """Convenience function: recognize math from a single image."""
    latex = get_math_ocr().recognize(image_input)
    return wrap_latex(latex) if wrap else latex


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python math_ocr.py <image_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    result = recognize_math(path, wrap=True)
    print(f"LaTeX:\n{result}")
