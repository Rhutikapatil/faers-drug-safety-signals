"""
Stage 0 — Data acquisition.

Tries a live pull from the openFDA drug/event API; if that's unreachable
(as in the sandboxed environment this project was originally built in) or
`data.source` is explicitly set to "synthetic", falls back to the
calibrated synthetic generator in synthetic_data.py. Either way the output
schema is identical, so every downstream stage is agnostic to which path
was used — swap data sources without touching any other script.
"""

import sys
from pathlib import Path

import pandas as pd
import requests
import yaml

from synthetic_data import generate as generate_synthetic


def _openfda_reachable(endpoint: str, timeout: float = 6.0) -> bool:
    """Actually attempt an HTTP(S) request through whatever network path
    `requests` will use (including any configured proxy) — a raw TCP socket
    check isn't representative in sandboxed environments that proxy HTTPS
    and reject specific hosts at the CONNECT layer."""
    try:
        resp = requests.get(endpoint, params={"limit": 1}, timeout=timeout)
        return resp.status_code < 500
    except requests.exceptions.RequestException:
        return False


def fetch_openfda(cfg: dict) -> pd.DataFrame:
    """Pull real adverse event reports from the openFDA API for the configured
    drug list. Requires normal internet egress to api.fda.gov."""
    endpoint = cfg["data"]["openfda_endpoint"]
    rows = []
    for drug in cfg["data"]["openfda_query_drugs"]:
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug}"',
            "limit": 100,
        }
        resp = requests.get(endpoint, params=params, timeout=30)
        resp.raise_for_status()
        for result in resp.json().get("results", []):
            patient = result.get("patient", {})
            reactions = patient.get("reaction", [])
            drugs = patient.get("drug", [])
            serious = str(result.get("serious", "2")) == "1"
            for r in reactions:
                rows.append({
                    "case_id": result.get("safetyreportid"),
                    "drug_name": drug.title(),
                    "drug_class": None,
                    "reaction_pt": r.get("reactionmeddrapt"),
                    "patient_age": patient.get("patientonsetage"),
                    "patient_sex": {"1": "Male", "2": "Female"}.get(patient.get("patientsex"), "Unknown"),
                    "outcome_serious": serious,
                    "outcome_type": result.get("seriousnessdeath") and "Death" or (
                        "Hospitalization" if result.get("seriousnesshospitalization") else
                        ("Serious (other)" if serious else "Non-serious")
                    ),
                    "report_year": (result.get("receivedate") or "0")[:4],
                    "narrative_text": None,  # openFDA's public API does not expose free-text narratives
                })
    return pd.DataFrame(rows)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    source = cfg["data"]["source"]
    out_path = Path(cfg["data"]["synthetic"]["raw_csv"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_openfda = source == "openfda" or (
        source == "auto" and _openfda_reachable(cfg["data"]["openfda_endpoint"])
    )

    if use_openfda:
        print("openFDA reachable — pulling live adverse event reports...")
        df = fetch_openfda(cfg)
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} real openFDA report-reaction rows -> {out_path}")
    else:
        print("openFDA not reachable from this environment — generating calibrated "
              "synthetic dataset instead (see src/synthetic_data.py docstring).")
        sc = cfg["data"]["synthetic"]
        df = generate_synthetic(sc["n_reports"], seed=sc["random_seed"])
        df.to_csv(out_path, index=False)
        print(f"Wrote {len(df)} synthetic report rows -> {out_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_path)
