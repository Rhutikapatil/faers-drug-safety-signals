"""
Stage 1 — Disproportionality (signal detection) analysis.

Standard pharmacovigilance approach: for every drug-reaction pair, build the
2x2 contingency table

                  reaction X       all other reactions
    drug D            a                    b
    other drugs       c                    d

and compute the Proportional Reporting Ratio (PRR), Reporting Odds Ratio
(ROR) with a 95% CI, and a chi-square statistic. A pair is flagged as a
"signal" under the standard Evans criteria: n >= 3 cases, PRR >= 2, and
chi-square >= 4.

Because this dataset can fall back to a calibrated synthetic generator
(see synthetic_data.py), this script also reports recall against the
literature-documented KNOWN_SIGNALS list as a validation check — i.e. does
the method actually recover the associations it was calibrated to contain.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_data import KNOWN_SIGNALS


def compute_signals(df: pd.DataFrame, min_cases: int, prr_thresh: float, chi2_thresh: float) -> pd.DataFrame:
    total = len(df)
    pair_counts = df.groupby(["drug_name", "reaction_pt"]).size().rename("a").reset_index()
    drug_totals = df.groupby("drug_name").size().rename("drug_total")
    reaction_totals = df.groupby("reaction_pt").size().rename("reaction_total")

    pair_counts = pair_counts.merge(drug_totals, on="drug_name").merge(reaction_totals, on="reaction_pt")
    pair_counts = pair_counts[pair_counts["a"] >= min_cases].copy()

    a = pair_counts["a"]
    b = pair_counts["drug_total"] - a
    c = pair_counts["reaction_total"] - a
    d = total - a - b - c

    prr = (a / (a + b)) / (c / (c + d))
    ror = (a * d) / (b * c)
    se_log_ror = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ror_ci_low = np.exp(np.log(ror) - 1.96 * se_log_ror)
    ror_ci_high = np.exp(np.log(ror) + 1.96 * se_log_ror)

    n = total
    expected_a = (a + b) * (a + c) / n
    chi2 = ((a - expected_a) ** 2) / expected_a + \
           ((b - (a + b) * (b + d) / n) ** 2) / ((a + b) * (b + d) / n) + \
           ((c - (a + c) * (c + d) / n) ** 2) / ((a + c) * (c + d) / n) + \
           ((d - (b + d) * (c + d) / n) ** 2) / ((b + d) * (c + d) / n)

    pair_counts["PRR"] = prr
    pair_counts["ROR"] = ror
    pair_counts["ROR_CI_low"] = ror_ci_low
    pair_counts["ROR_CI_high"] = ror_ci_high
    pair_counts["chi2"] = chi2
    pair_counts["is_signal"] = (pair_counts["PRR"] >= prr_thresh) & (pair_counts["chi2"] >= chi2_thresh)
    pair_counts["is_known_literature_signal"] = pair_counts.apply(
        lambda r: (r["drug_name"], r["reaction_pt"]) in KNOWN_SIGNALS, axis=1
    )

    return pair_counts.sort_values("PRR", ascending=False)


def main(config_path: str, input_csv: str, output_table: str, output_fig: str, output_validation: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["disproportionality"]

    df = pd.read_csv(input_csv)
    results = compute_signals(df, cfg["min_case_count"], cfg["prr_threshold"], cfg["chi2_threshold"])

    Path(output_table).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_table, index=False)

    # Validation: recall of the disproportionality method against the known
    # literature signals baked into the synthetic generator.
    known_in_data = set(KNOWN_SIGNALS)
    flagged = set(zip(results.loc[results.is_signal, "drug_name"], results.loc[results.is_signal, "reaction_pt"]))
    recovered = known_in_data & flagged
    recall = len(recovered) / len(known_in_data) if known_in_data else float("nan")

    with open(output_validation, "w") as f:
        f.write(f"known_literature_signals,{len(known_in_data)}\n")
        f.write(f"recovered_by_prr_chi2_method,{len(recovered)}\n")
        f.write(f"recall,{recall:.3f}\n")
        f.write(f"total_pairs_tested,{len(results)}\n")
        f.write(f"total_pairs_flagged_as_signal,{int(results.is_signal.sum())}\n")

    top20 = results.sort_values("chi2", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    labels = [f"{d} - {r}" for d, r in zip(top20.drug_name, top20.reaction_pt)]
    colors = ["#d62728" if k else "#1f77b4" for k in top20.is_known_literature_signal]
    ax.barh(labels[::-1], top20.PRR[::-1], color=colors[::-1])
    ax.axvline(cfg["prr_threshold"], color="gray", linestyle="--", linewidth=1, label=f"PRR = {cfg['prr_threshold']} (signal threshold)")
    ax.set_xlabel("Proportional Reporting Ratio (PRR)")
    ax.set_title("Top 20 drug-reaction pairs by signal strength\n(red = matches a literature-documented signal)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_fig, dpi=150)
    plt.close(fig)

    print(f"Tested {len(results)} drug-reaction pairs; flagged {int(results.is_signal.sum())} as signals.")
    print(f"Recovered {len(recovered)}/{len(known_in_data)} literature-documented signals (recall={recall:.1%}).")


if __name__ == "__main__":
    main(*sys.argv[1:6])
