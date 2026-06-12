# Stage 2 — OCR / Handwriting Text Recognition

Converts preprocessed handwriting images into structured text transcripts.

## Architecture

Two-headed recognition pipeline:
1. **Region Detector** — classifies image patches as Arabic text, English text, or math expression
2. **Arabic HTR** — Microsoft TrOCR (fine-tuned for handwriting) for Arabic-script regions
3. **Math OCR** — pix2tex (LaTeX-OCR) for mathematical expression regions
4. **Text Assembler** — merges outputs into a unified JSON transcript per sample

Alternatively, `run_ocr.py` can use **Qwen2-VL directly** for end-to-end OCR (single model, supports Arabic + math), which is recommended when GPU memory allows.

## Scripts

| Script | Purpose |
|---|---|
| `region_detector.py` | Split image into text vs math regions |
| `arabic_htr.py` | Run TrOCR on Arabic/English handwritten text |
| `math_ocr.py` | Run pix2tex (LaTeX-OCR) on math regions |
| `text_assembler.py` | Merge region transcripts into final JSON |
| `run_ocr.py` | End-to-end runner (calls all of the above) |

## Run

```bash
# Full OCR pipeline
python run_ocr.py

# Or: just the end-to-end Qwen2-VL path (faster, single model)
python run_ocr.py --mode qwen2vl
```

## Output

`../data/transcripts/` — one JSON file per sample:
```json
{
  "id": "sample_001",
  "transcript": "2x + 5 = 13\nx = 4",
  "latex_expressions": ["2x + 5 = 13", "x = 4"],
  "detected_language": "english",
  "ocr_mode": "trocr+pix2tex"
}
```
