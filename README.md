# FAERS Drug Safety Signal Detection
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20analysis-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20Logistic%20Regression-6A5ACD)
![openFDA](https://img.shields.io/badge/openFDA-API-0A6EBD)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker&logoColor=white)

A pharmacovigilance pipeline that runs three complementary analyses on
adverse-event reports: statistical **disproportionality (signal detection)**,
a **predictive model** for report seriousness from structured fields, and an
**NLP baseline** that predicts seriousness from free-text narrative alone.

## Why this project

Drug safety teams do all three of these things, usually with different
tools. Putting them in one pipeline, on the same dataset, makes it possible
to directly compare what structured fields vs. free text each contribute —
and to validate the statistical method against known ground truth before
trusting it on anything new.

## Data — please read this before the results

Two data sources are supported, switchable in `config.yaml` with no code
changes:

1. **`openfda`** — a live pull from the [openFDA drug/event API](https://open.fda.gov/apis/drug/event/).
   `src/fetch_data.py` implements this against the real API; it needs normal
   internet access, which the sandboxed environment this project was
   originally built in did not have (only package registries and GitHub
   were reachable — no route to `api.fda.gov`). It works from a normal
   laptop or CI runner.
2. **`synthetic`** (the default fallback, and what generated the results
   below) — a calibrated simulator in `src/synthetic_data.py` that
   generates FAERS-shaped case reports (drug, MedDRA-style reaction term,
   age, sex, seriousness outcome, narrative text). It is **not real FDA
   data** and says so in its own docstring. It's deliberately seeded with
   17 drug-adverse-event pairs that are well-documented in the
   pharmacovigilance literature (e.g. simvastatin–rhabdomyolysis,
   warfarin–haemorrhage, clozapine–agranulocytosis) specifically so the
   disproportionality method can be checked against a known answer — see
   the validation result below.

Also worth knowing: openFDA's public API exposes structured fields only,
not free-text case narratives, so the NLP stage (`src/nlp_narrative.py`)
only runs meaningfully against the synthetic path or against a real
narrative-bearing dataset you supply — it's written to work on any CSV with
`narrative_text` / `outcome_serious` columns.

Run `python3 main.py` with `data.source: openfda` on a machine with normal
internet access to reproduce this whole pipeline on live FDA data.

## Pipeline

```
fetch_data → disproportionality → predictive_model → nlp_narrative
```

| Stage | Method | Output |
|---|---|---|
| Disproportionality | PRR, ROR (95% CI), chi-square per drug-reaction pair; Evans-criteria signal flagging | `results/tables/disproportionality_signals.csv`, `results/figures/01_top_signals.png` |
| Predictive model | Gradient-boosted trees on age/sex/drug class/reaction/year | `results/figures/02_roc_curve.png`, `results/figures/03_feature_importance.png`, `results/model_card.md` |
| NLP baseline | TF-IDF + logistic regression on narrative text alone | `results/figures/04_nlp_top_tokens.png`, `results/nlp_report.txt` |

## Results (synthetic data, n = 20,000 reports)

**Signal detection validation** — the whole point of building this against
calibrated synthetic data first: does the statistical method actually find
what it's supposed to find?

> Tested 660 drug-reaction pairs → flagged 18 as signals (PRR ≥ 2, χ² ≥ 4,
> ≥ 3 cases) → **recovered 17/17 (100%)** of the literature-documented
> signals baked into the data, plus one additional pair (Apixaban–
> Hepatotoxicity) that did not clear the threshold and was correctly left
> unflagged.

**Predictive model (structured fields):** ROC-AUC 0.756, PR-AUC 0.496.
Reaction term dominates feature importance (expected — some reaction types
are inherently more likely to be coded serious), with patient age
contributing meaningfully as well.

**NLP baseline (narrative text only):** ROC-AUC 0.847 — outperforming the
structured-fields model despite using none of its features, which suggests
free-text narratives carry real incremental signal about seriousness beyond
what gets captured in coded fields. (See `results/model_card.md` for full
metrics and documented limitations, including why this comparison is
data-source-specific and would need re-validation on real narratives.)

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

**With Docker:**
```bash
docker build -t faers-signal-detection .
docker run -v $(pwd)/results:/pipeline/results faers-signal-detection
```

Re-parameterize signal-detection thresholds, model hyperparameters, or the
openFDA drug query list in `config.yaml`.

## Project structure

```
.
├── main.py                    # orchestrates all 4 stages
├── config.yaml                 # data source, thresholds, model params
├── requirements.txt
├── Dockerfile
├── src/
│   ├── fetch_data.py           # openFDA live pull, with synthetic fallback
│   ├── synthetic_data.py       # calibrated FAERS-style simulator
│   ├── disproportionality.py   # PRR / ROR / chi-square signal detection
│   ├── predictive_model.py     # structured-field seriousness classifier
│   └── nlp_narrative.py        # TF-IDF narrative-only classifier
└── results/
    ├── figures/
    ├── tables/
    ├── model_card.md
    └── nlp_report.txt
```

## Stack

Python · pandas/NumPy · scikit-learn · TF-IDF/logistic regression ·
gradient-boosted trees · openFDA API · Docker

## Author

Rhutika Patil — M.S. Bioinformatics, NC State University
[linkedin.com/in/rhutika-patil](https://linkedin.com/in/rhutika-patil) ·
[github.com/Rhutikapatil](https://github.com/Rhutikapatil)
