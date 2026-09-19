"""
Stage 2 — Predictive model for report seriousness.

Trains a gradient-boosted classifier on structured case fields (age, sex,
drug class, reaction term, report year) to predict whether a report will be
classified as "serious" (hospitalization, life-threatening, death, or
disability) — a triage-style model that could help prioritize which
incoming reports a pharmacovigilance reviewer looks at first.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def main(config_path: str, input_csv: str, output_fig_roc: str, output_fig_importance: str, model_card_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["predictive_model"]

    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["patient_age", "patient_sex", "drug_class", "reaction_pt"])

    cat_cols = ["patient_sex", "drug_class", "reaction_pt"]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = enc.fit_transform(df[cat_cols])
    feature_names = list(enc.get_feature_names_out(cat_cols)) + ["patient_age", "report_year"]
    X = np.hstack([X_cat, df[["patient_age", "report_year"]].to_numpy()])
    y = df["outcome_serious"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y
    )

    clf = GradientBoostingClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        random_state=cfg["random_state"],
    )
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Non-serious", "Serious"])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="GradientBoosting")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title(f"ROC curve — seriousness prediction (AUC = {auc:.3f})")
    fig.tight_layout()
    Path(output_fig_roc).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig_roc, dpi=150)
    plt.close(fig)

    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    ax2.barh(importances.index[::-1], importances.values[::-1])
    ax2.set_title("Top 15 features — seriousness classifier")
    ax2.set_xlabel("Feature importance")
    fig2.tight_layout()
    fig2.savefig(output_fig_importance, dpi=150)
    plt.close(fig2)

    serious_rate = df["outcome_serious"].mean()
    card = f"""# Model Card — FAERS Report Seriousness Classifier

## Purpose
Predicts whether an adverse event report will be classified as "serious"
(hospitalization, life-threatening, death, or disability) from structured
case fields available at intake: patient age, sex, drug class, reaction
term, and report year. Intended as a **triage aid** to help a
pharmacovigilance reviewer prioritize incoming reports — not a clinical or
regulatory decision-making tool.

## Data
- {len(df):,} case-reaction rows, {serious_rate:.1%} labeled serious.
- Data source: see `data/README` in this repo — synthetic by default in
  this build (see `src/synthetic_data.py`), or live openFDA reports if run
  with `data.source: openfda` and normal internet access.
- Train/test split: {int((1 - cfg['test_size']) * 100)}/{int(cfg['test_size'] * 100)}, stratified on the target.

## Model
Gradient-boosted decision trees (scikit-learn `GradientBoostingClassifier`),
{cfg['n_estimators']} estimators, max depth {cfg['max_depth']}, learning
rate {cfg['learning_rate']}. Features one-hot encode sex/drug class/reaction
term; age and report year are used as-is.

## Performance (held-out test set)
- ROC-AUC: **{auc:.3f}**
- Average precision (PR-AUC): **{ap:.3f}**

```
{report}
```

## Limitations
- Trained on {"synthetic, literature-calibrated" if "synthetic" in input_csv else "openFDA"} data — a real
  deployment must be retrained and re-validated on the target reporting
  system's real, current data before any operational use.
- Reaction term and drug class are strong predictors here partly because the
  synthetic generator injects a higher base seriousness rate for specific
  reaction types (rhabdomyolysis, haemorrhage, etc.) — the feature
  importance plot should be read as "what the model learned from this
  dataset," not as an independent clinical finding.
- No adjustment for reporting bias, duplicate reports, or causality
  (a report being "serious" is not the same as the drug having caused it).
- Not validated for any subgroup fairness properties; age/sex are included
  as predictors and that trade-off (predictive value vs. protected-attribute
  use) would need explicit review before any real deployment.
"""
    Path(model_card_path).write_text(card)

    print(f"ROC-AUC: {auc:.3f} | PR-AUC: {ap:.3f}")
    print(report)


if __name__ == "__main__":
    main(*sys.argv[1:6])
