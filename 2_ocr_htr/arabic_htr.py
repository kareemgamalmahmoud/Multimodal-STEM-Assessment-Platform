"""
Arabic / English Handwriting Text Recognition using Microsoft TrOCR.

TrOCR (Li et al. 2021) is a Transformer-based OCR model pre-trained on large
synthetic and real handwriting datasets. We use the base handwritten variant.

Model: microsoft/trocr-base-handwritten
  - Input: RGB image (any size, auto-resized internally)
  - Output: Unicode text string

For Arabic-specific content, we note that TrOCR-base-handwritten has limited
Arabic training data. A production system should fine-tune on Arabic HTR
datasets (e.g., KHATT, IFN/ENIT). We include this as an important baseline
and document it as a future improvement in docs/future_improvements.md.

Qwen2-VL (used in Stage 3/4) performs substantially better on Arabic handwriting
and is the primary recognition path in our pipeline.
"""

from pathlib import Path
from typing import Optional

import numpy as np


TROCR_MODEL_ID = "microsoft/trocr-base-handwritten"


class ArabicHTR:
    """
    Wrapper around TrOCR for handwriting recognition.
    Lazy-loads the model on first call to avoid OOM when only the
    Qwen2-VL path is used.
    """

    def __init__(self, model_id: str = TROCR_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        """Load model and processor on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TrOCR] Loading {self.model_id} on {self.device} ...")

        self._processor = TrOCRProcessor.from_pretrained(self.model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        print("[TrOCR] Model ready.")

    def recognize(self, image_input, max_new_tokens: int = 128) -> str:
        """
        Recognize handwritten text from an image.

        Args:
            image_input: PIL Image, numpy array (H x W uint8), or Path to image file.
            max_new_tokens: maximum number of output tokens.

        Returns:
            Recognized text string.
        """
        self._load()

        import torch
        from PIL import Image

        # Normalize input to PIL RGB
        if isinstance(image_input, Path):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        pixel_values = self._processor(
            images=pil_img, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
            )

        text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return text

    def recognize_batch(self, images: list, max_new_tokens: int = 128) -> list[str]:
        """Recognize a batch of images for throughput efficiency."""
        self._load()

        import torch
        from PIL import Image

        pil_images = []
        for img in images:
            if isinstance(img, Path):
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                pil_images.append(img.convert("RGB"))

        pixel_values = self._processor(
            images=pil_images, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
            )

        texts = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [t.strip() for t in texts]


# Module-level singleton (lazy-initialized)
_htr_instance: Optional[ArabicHTR] = None


def get_htr(model_id: str = TROCR_MODEL_ID) -> ArabicHTR:
    global _htr_instance
    if _htr_instance is None:
        _htr_instance = ArabicHTR(model_id=model_id)
    return _htr_instance


def recognize_handwriting(image_input, model_id: str = TROCR_MODEL_ID) -> str:
    """Convenience function: recognize handwriting in a single image."""
    return get_htr(model_id).recognize(image_input)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python arabic_htr.py <image_path>")
        sys.exit(1)
    path = Path(sys.argv[1])
    text = recognize_handwriting(path)
    print(f"Recognized text:\n{text}")
