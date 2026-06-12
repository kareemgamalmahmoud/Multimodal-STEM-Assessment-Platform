# Stage 1 — Data Preparation

Downloads the FERMAT dataset, performs exploratory data analysis, and preprocesses images for downstream OCR and evaluation stages.

## Scripts

| Script | Purpose |
|---|---|
| `download_dataset.py` | Download FERMAT from HuggingFace Hub |
| `explore_dataset.py` | EDA: sample counts, score distribution, language breakdown |
| `preprocess.py` | Image cleaning: grayscale, binarize, deskew, pad, resize |

## Run Order

```bash
python download_dataset.py   # creates ../data/fermat_raw/
python explore_dataset.py    # prints stats, saves sample grid to ../data/eda/
python preprocess.py         # creates ../data/fermat_processed/
```

## Outputs

- `../data/fermat_raw/` — raw FERMAT dataset files
- `../data/eda/` — EDA figures and stats JSON
- `../data/fermat_processed/` — preprocessed images + metadata CSV

## Notes

- Images are saved as PNG (lossless) after preprocessing.
- The metadata CSV (`metadata.csv`) has columns:
  `id, question, reference_answer, image_path, human_score, language_track`
- `language_track` is one of: `arabic`, `english`, `mixed` — detected heuristically from any available text metadata.
