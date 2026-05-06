# finsent-compare

A small, reproducible study that puts a rule-based sentiment baseline
(**VADER**) head-to-head against a Transformer fine-tuned for financial text
(**FinBERT**, `ProsusAI/finbert`) on the **Financial PhraseBank** (Malo et al.).
Built as the honors extension for an undergraduate ML / neural networks course.

> Status: scaffolding in place. Pipeline modules and the notebook are landing
> in subsequent commits — see the project plan / commit history for the order.

## Why this comparison is interesting

Generic sentiment models pick up on words like *"beats"*, *"crushes"*, *"falls"*
and treat them with their everyday meaning. In an earnings headline,
*"Acme beats expectations"* is positive and *"Acme falls short"* is negative —
which lines up with intuition — but *"Costs rise"* or *"Margins compress"* tend
to fool a lexicon-only model because the literal verbs aren't lexicon-negative.
This is exactly where a domain-tuned encoder like FinBERT earns its keep: it
learns those finance-specific contextual cues that a bag-of-words approach
can't represent.

## What's here so far

```
finsent-compare/
├── README.md
├── requirements.txt
├── environment.yml
├── .gitignore
├── data/
│   ├── raw/         # PhraseBank lands here (or HF fallback in code)
│   └── processed/
├── src/             # all pipeline logic — being filled in commit by commit
├── notebooks/       # comparative_analysis.ipynb
├── figures/         # generated PNGs
├── results/         # metrics.json, predictions.csv, disagreements.csv
└── tests/           # pytest suite
```

## Setup

Tested on Python 3.10 / 3.11.

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CUDA users: install the matching `torch` wheel from
<https://pytorch.org/get-started/locally/> instead of the CPU default before
running `pip install -r requirements.txt`.

## How to reproduce my numbers

Once the pipeline modules and notebook land:

```bash
jupyter nbconvert --to notebook --execute notebooks/comparative_analysis.ipynb --output comparative_analysis.ipynb
```

That command runs the entire study top-to-bottom and refreshes everything
under `results/` and `figures/`.

## Results

A populated table will land here after the first end-to-end run.

| Model   | Accuracy | Macro-F1 | F1 (neg) | F1 (neu) | F1 (pos) |
|---------|----------|----------|----------|----------|----------|
| VADER   | _tbd_    | _tbd_    | _tbd_    | _tbd_    | _tbd_    |
| FinBERT | _tbd_    | _tbd_    | _tbd_    | _tbd_    | _tbd_    |

## References

- Malo, P. et al. *Good Debt or Bad Debt: Detecting Semantic Orientations in
  Economic Texts.* JASIST, 2014.
- Araci, D. *FinBERT: Financial Sentiment Analysis with Pre-trained Language
  Models.* arXiv:1908.10063, 2019.
- Hutto, C.J. & Gilbert, E. *VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text.* ICWSM, 2014.
