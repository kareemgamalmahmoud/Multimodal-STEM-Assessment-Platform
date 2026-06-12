"""
Qwen2-VL inference client with 4-bit quantization support.

This module provides a single shared model instance used across all experiments.
The model is loaded lazily (on first use) and cached as a module-level singleton
to avoid reloading 7B parameters between experiment runs.

Model: Qwen/Qwen2-VL-7B-Instruct
  - Supports Arabic natively (pre-trained on multilingual data)
  - Handles images + text in the same forward pass
  - Loaded in 4-bit NF4 quantization: ~8 GB VRAM on a T4 GPU

Reference:
  Wang et al. (2024). Qwen2-VL: Enhancing Vision-Language Model's Perception
  of the World at Any Resolution. arXiv:2409.12191.
"""

import os
from pathlib import Path
from typing import Optional, Union

import torch

# Qwen model identifier
QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

# Generation defaults
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.1   # near-deterministic for scoring tasks


class QwenVLClient:
    """
    Thread-unsafe singleton wrapper for Qwen2-VL inference.

    Usage:
        client = QwenVLClient.get_instance()
        response = client.generate_from_image(image_path, prompt)
    """

    _instance: Optional["QwenVLClient"] = None

    def __init__(self, model_id: str = QWEN_MODEL_ID, use_4bit: bool = True):
        self.model_id = model_id
        self.use_4bit = use_4bit
        self._model = None
        self._processor = None

    @classmethod
    def get_instance(cls, model_id: str = QWEN_MODEL_ID, use_4bit: bool = True) -> "QwenVLClient":
        """Return the shared singleton, loading it on first call."""
        if cls._instance is None:
            cls._instance = cls(model_id=model_id, use_4bit=use_4bit)
        return cls._instance

    def _load(self):
        """Load model + processor once. Subsequent calls are no-ops."""
        if self._model is not None:
            return

        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        print(f"[QwenVL] Loading {self.model_id} (4-bit={self.use_4bit}) ...")

        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if not self.use_4bit else None,
            trust_remote_code=True,
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        print("[QwenVL] Model ready.\n")

    def generate_from_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """
        Run one image + text prompt through Qwen2-VL and return the response string.

        Args:
            image_path:     Path to the image file (PNG, JPG, etc.)
            prompt:         Text prompt / instruction.
            max_new_tokens: Maximum tokens in the generated response.
            temperature:    Sampling temperature. 0.0 = greedy (deterministic).

        Returns:
            The model's response as a plain string.
        """
        self._load()

        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        do_sample = temperature > 0.01
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
            )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output[0].strip()

    def generate_text_only(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """
        Text-only generation (no image). Used for rubric generation and
        post-transcription scoring where the transcript is already text.
        """
        self._load()

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input],
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        do_sample = temperature > 0.01
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
            )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def parse_json_from_response(response: str) -> dict:
    """
    Extract the first JSON object from a model response string.
    The model is prompted to output JSON, but sometimes adds surrounding text.
    """
    import re, json

    # Try to find a JSON block
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Fall back: try the whole string
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"raw_response": response}
