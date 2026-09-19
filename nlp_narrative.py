"""
Stage 3 — NLP on report narrative text.

Baseline text-classification pipeline (TF-IDF + logistic regression)
predicting report seriousness from free-text narrative alone, with no
structured fields. Reports the most predictive tokens per class, which is
the kind of interpretable baseline worth building before reaching for an
LLM: it's fast, auditable, and gives you a floor to beat.

Note: openFDA's public API does not expose free-text case narratives (only
structured fields), so this stage only runs when narrative_text is present
— i.e. on the synthetic data path. Swap in a real narrative corpus (e.g. a
licensed FAERS case-narrative extract, or your own clinical text) by
pointing `input_csv` at a file with the same `narrative_text` /
`outcome_serious` columns; nothing else in this script needs to change.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def main(config_path: str, input_csv: str, output_fig: str, output_report: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["nlp"]

    df = pd.read_csv(input_csv)
    if "narrative_text" not in df.columns or df["narrative_text"].isna().all():
        Path(output_report).write_text(
            "No narrative_text column available for this data source (openFDA's public "
            "API does not expose free-text narratives) — NLP stage skipped. Run against "
            "synthetic data or a narrative-bearing dataset to exercise this stage.\n"
        )
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "NLP stage skipped: no narrative text in this data source",
                ha="center", va="center", wrap=True)
        ax.axis("off")
        Path(output_fig).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_fig, dpi=150)
        return

    df = df.dropna(subset=["narrative_text"])
    X_text = df["narrative_text"]
    y = df["outcome_serious"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=cfg["max_features"], stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    y_proba = clf.predict_proba(X_test_vec)[:, 1]
    y_pred = clf.predict(X_test_vec)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=["Non-serious", "Serious"])

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_serious = feature_names[np.argsort(coefs)[-15:]]
    top_nonserious = feature_names[np.argsort(coefs)[:15]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].barh(top_serious, np.sort(coefs)[-15:], color="#d62728")
    axes[0].set_title("Top tokens predicting 'serious'")
    axes[1].barh(top_nonserious[::-1], np.sort(coefs)[:15][::-1], color="#1f77b4")
    axes[1].set_title("Top tokens predicting 'non-serious'")
    fig.suptitle(f"TF-IDF + logistic regression on narrative text (AUC = {auc:.3f})")
    fig.tight_layout()
    Path(output_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=150)
    plt.close(fig)

    Path(output_report).write_text(
        f"NLP narrative-only seriousness classifier\n"
        f"==========================================\n"
        f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n"
        f"ROC-AUC: {auc:.3f}\n\n{report}\n\n"
        f"Top tokens -> 'serious': {', '.join(top_serious[::-1])}\n"
        f"Top tokens -> 'non-serious': {', '.join(top_nonserious)}\n"
    )

    print(f"NLP narrative model ROC-AUC: {auc:.3f}")
    print(report)


if __name__ == "__main__":
    main(*sys.argv[1:5])
