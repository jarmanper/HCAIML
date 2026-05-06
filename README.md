# finsent-compare — VADER vs FinBERT on Financial PhraseBank

A small, reproducible study that puts a rule-based sentiment baseline
(**VADER**) head-to-head against a Transformer fine-tuned for financial text
(**FinBERT**, [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert))
on the **Financial PhraseBank** (Malo et al., 2014). Built as the honors
extension for an undergraduate ML / neural networks course.

## Why this comparison is interesting

Generic sentiment models pick up on words like "beats," "crushes," "falls,"
"misses" and treat them with their everyday meaning. In an earnings headline
that often happens to line up — *"Acme beats expectations"* genuinely is
positive — but financial language is full of cases where it doesn't:
*"costs rose,"* *"margins compressed,"* *"guidance tightened"*. A lexicon-only
model can't represent the contextual flip; the verbs aren't in any sentiment
lexicon at all, or they're filed under a polarity that's wrong for the
finance domain. That's exactly where a domain-tuned encoder like FinBERT
earns its keep: every token's representation is contextual, so "compressed"
in *"margins compressed"* doesn't get the same vector as "compressed" in
some unrelated sentence about a video file.

This project quantifies that gap on a clean benchmark and surfaces the
specific sentences where it shows up, so the comparison is something more
substantive than a single F1 number.

## Repo layout

```
finsent-compare/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│   ├── raw/         # PhraseBank lands here, or HF fallback in code
│   └── processed/   # cleaned parquet output
├── src/
│   ├── config.py            # paths, seeds, label maps, model ids
│   ├── data_loader.py       # latin-1 parse + HF fallback + stratified split
│   ├── preprocess.py        # light text normalization for the VADER path
│   ├── vader_model.py       # rule-based baseline + tunable thresholds
│   ├── finbert_model.py     # HF batched inference + id2label remap
│   ├── evaluate.py          # metrics, predictions frame, edge cases
│   └── visualize.py         # the four required figures
├── notebooks/
│   └── comparative_analysis.ipynb
├── figures/      # generated PNGs
├── results/      # metrics.json, predictions.csv, disagreements.csv
└── tests/        # pytest suite
```

## Setup

Tested with Python 3.10 / 3.11.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### CUDA

The default install uses the CPU `torch` wheel. If you have a CUDA-enabled
GPU and want the fast path, install the matching `torch` wheel from
<https://pytorch.org/get-started/locally/> *before* running
`pip install -r requirements.txt`. The pipeline auto-detects CUDA at
inference time.

### conda (optional)

```bash
conda env create -f environment.yml
conda activate finsent-compare
```

## Getting the dataset

You have two options and the loader picks whichever is available:

1. **Local file** (preferred for reproducibility — the original release): drop
   `Sentences_75Agree.txt` into `data/raw/`. The original release page is at
   <https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10>;
   it has been flaky historically. The `Sentences_50Agree.txt` file is also
   accepted, for the optional ablation. Files are latin-1 encoded — the
   loader handles that explicitly, you don't need to convert.
2. **Hugging Face fallback**: if no local file is present, the loader pulls
   `financial_phrasebank` configuration `sentences_75agree` from the HF hub.
   This requires network access on first run; cached afterwards.

Either way, the cleaned dataset is persisted to
`data/processed/phrasebank.parquet` for downstream tooling.

## Reproduce

One command, from the repo root:

```bash
jupyter nbconvert --to notebook --execute \
  notebooks/comparative_analysis.ipynb \
  --output comparative_analysis.ipynb
```

That walks the notebook end-to-end: load + clean PhraseBank, run VADER, run
FinBERT, compute metrics, produce the four figures, write
`results/metrics.json`, `results/predictions.csv`,
`results/disagreements.csv`, and refresh `figures/`. Open the same notebook
afterwards to read the executed outputs.

## Tests

```bash
pytest -q
```

Covers the loader (latin-1 parsing, label mapping, dedup, split
reproducibility) and the evaluator (per-class and macro metrics on
hand-derived examples).

## Expected runtime

| Step                        | CPU (laptop)        | CUDA GPU       |
|-----------------------------|---------------------|----------------|
| Loader + parquet write      | ~1 s                | ~1 s           |
| VADER over the test set     | < 5 s               | < 5 s          |
| FinBERT over the test set   | ~2–4 min            | < 30 s         |
| First-time FinBERT download | ~440 MB, one-off    | ~440 MB        |

The model cache lives under `~/.cache/huggingface` after the first run.

## Results

Filled in from `results/metrics.json` after the first end-to-end run. Until
then, placeholders.

| Model   | Accuracy | Macro F1 | F1 (negative) | F1 (neutral) | F1 (positive) |
|---------|----------|----------|---------------|--------------|---------------|
| VADER   | _tbd_    | _tbd_    | _tbd_         | _tbd_        | _tbd_         |
| FinBERT | _tbd_    | _tbd_    | _tbd_         | _tbd_        | _tbd_         |

## References

- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
  *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.*
  Journal of the Association for Information Science and Technology.
- Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained
  Language Models.* arXiv:1908.10063.
- Hutto, C. J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based
  Model for Sentiment Analysis of Social Media Text.* ICWSM.
