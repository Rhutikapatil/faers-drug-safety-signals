"""
Runs the full FAERS drug-safety signal-detection pipeline end to end:
    fetch_data -> disproportionality -> predictive_model -> nlp_narrative
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import fetch_data
import disproportionality
import predictive_model
import nlp_narrative
import yaml

CONFIG = "config.yaml"


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    raw_csv = cfg["data"]["synthetic"]["raw_csv"]

    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)

    print("\n=== Stage 0: data acquisition ===")
    fetch_data.main(CONFIG)

    print("\n=== Stage 1: disproportionality / signal detection ===")
    disproportionality.main(
        CONFIG, raw_csv,
        "results/tables/disproportionality_signals.csv",
        "results/figures/01_top_signals.png",
        "results/tables/signal_detection_validation.csv",
    )

    print("\n=== Stage 2: predictive model (seriousness) ===")
    predictive_model.main(
        CONFIG, raw_csv,
        "results/figures/02_roc_curve.png",
        "results/figures/03_feature_importance.png",
        "results/model_card.md",
    )

    print("\n=== Stage 3: NLP on narrative text ===")
    nlp_narrative.main(
        CONFIG, raw_csv,
        "results/figures/04_nlp_top_tokens.png",
        "results/nlp_report.txt",
    )

    print("\nPipeline complete. See results/ for tables, figures, and the model card.")


if __name__ == "__main__":
    main()
