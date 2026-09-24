"""
Synthetic FAERS-style case report generator.

IMPORTANT — data provenance: this module produces SIMULATED data. It does not
contain, and is not derived from, any real FDA Adverse Event Reporting System
(FAERS) report. It exists because the build environment for this project had
no network route to api.fda.gov (only package registries and GitHub were
reachable), so `fetch_data.py` falls back to this generator when a live
openFDA pull isn't possible. The generator is calibrated against real,
published pharmacovigilance findings (see KNOWN_SIGNALS below) specifically
so the rest of the pipeline can be validated against a ground truth: a
disproportionality method that's worth anything should recover these
well-documented drug-ADR associations from the simulated reports.

Run this same pipeline against real data any time by setting
`data.source: openfda` in config.yaml on a machine with normal internet
access — fetch_data.py's `fetch_openfda()` implements that path against the
live API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DRUGS = {
    "Simvastatin": "Statin",
    "Atorvastatin": "Statin",
    "Warfarin": "Anticoagulant",
    "Apixaban": "Anticoagulant",
    "Metformin": "Antidiabetic",
    "Glipizide": "Antidiabetic",
    "Lisinopril": "ACE inhibitor",
    "Enalapril": "ACE inhibitor",
    "Ibuprofen": "NSAID",
    "Naproxen": "NSAID",
    "Clozapine": "Atypical antipsychotic",
    "Quetiapine": "Atypical antipsychotic",
    "Ciprofloxacin": "Fluoroquinolone",
    "Levofloxacin": "Fluoroquinolone",
    "Isotretinoin": "Retinoid",
    "Amoxicillin": "Penicillin antibiotic",
    "Sertraline": "SSRI",
    "Fluoxetine": "SSRI",
    "Levothyroxine": "Thyroid hormone",
    "Omeprazole": "Proton pump inhibitor",
}

REACTIONS = [
    "Rhabdomyolysis", "Myalgia", "Hepatotoxicity",
    "Haemorrhage", "Bruising", "International normalised ratio increased",
    "Lactic acidosis", "Nausea", "Diarrhoea",
    "Angioedema", "Cough", "Hyperkalaemia",
    "Gastrointestinal haemorrhage", "Dyspepsia", "Renal impairment",
    "Agranulocytosis", "Weight increased", "Sedation",
    "Tendon rupture", "Tendinitis", "QT prolongation",
    "Depression", "Dry skin", "Photosensitivity reaction",
    "Rash", "Anaphylactic reaction", "Pruritus",
    "Insomnia", "Headache", "Dizziness",
    "Hypothyroidism", "Osteoporosis", "Pneumonia",
]

# Literature-documented drug-ADR signals (used to calibrate the simulation
# and, downstream, to check whether disproportionality analysis recovers
# them). This is the "gold standard" the disproportionality script scores
# itself against.
KNOWN_SIGNALS = [
    ("Simvastatin", "Rhabdomyolysis"), ("Atorvastatin", "Rhabdomyolysis"),
    ("Warfarin", "Haemorrhage"), ("Warfarin", "International normalised ratio increased"),
    ("Metformin", "Lactic acidosis"),
    ("Lisinopril", "Angioedema"), ("Enalapril", "Angioedema"), ("Lisinopril", "Cough"),
    ("Ibuprofen", "Gastrointestinal haemorrhage"), ("Naproxen", "Gastrointestinal haemorrhage"),
    ("Clozapine", "Agranulocytosis"),
    ("Ciprofloxacin", "Tendon rupture"), ("Levofloxacin", "Tendon rupture"),
    ("Isotretinoin", "Depression"), ("Isotretinoin", "Dry skin"),
    ("Quetiapine", "Weight increased"), ("Quetiapine", "Sedation"),
]

SERIOUS_REACTIONS = {
    "Rhabdomyolysis", "Haemorrhage", "Gastrointestinal haemorrhage", "Lactic acidosis",
    "Agranulocytosis", "Anaphylactic reaction", "Tendon rupture", "QT prolongation",
    "Hepatotoxicity", "Renal impairment",
}

OPENING_TEMPLATES = [
    "{age}-year-old {sex} patient developed {reaction} approximately {days} days after starting {drug}.",
    "Report describes a {sex} patient, age {age}, who experienced {reaction} while on {drug} therapy.",
    "{sex} patient (age {age}) presented with {reaction} following {drug} use.",
    "A {sex} patient, {age} years old, noted {reaction} associated with {drug}.",
    "Case involves a {sex} patient (age {age}) with onset of {reaction} during treatment with {drug}.",
]

# Severity-leaning phrase pools. These are sampled *probabilistically*, not
# deterministically, by class — real-world narrative language and eventual
# seriousness coding are correlated but far from perfectly separable (a
# report can read alarmingly and still be coded non-serious, and vice
# versa), and the synthetic data should reflect that rather than making the
# NLP task trivial.
SEVERE_SOUNDING_PHRASES = [
    "Patient was admitted to hospital for management.",
    "Required urgent medical intervention.",
    "Patient was hospitalized and monitored closely.",
    "Event required emergency department evaluation.",
    "Symptoms were severe and prompted immediate treatment.",
    "Drug was discontinued and patient was observed overnight.",
]
MILD_SOUNDING_PHRASES = [
    "Symptom was mild and resolved without intervention.",
    "No hospitalization was required; patient continued therapy at a reduced dose.",
    "Event resolved after a brief period of observation.",
    "Patient was managed as an outpatient.",
    "Symptom improved after a short course of supportive care.",
    "Patient was advised to monitor symptoms and follow up as needed.",
]
NEUTRAL_FILLER_PHRASES = [
    "Concomitant medications were reviewed and no interactions were identified.",
    "Patient has a documented history of similar therapy in the past.",
    "Reporter noted no other relevant medical history.",
    "Follow-up information was requested from the reporting clinician.",
    "Causality assessment is ongoing at the time of this report.",
    "",  # sometimes no filler at all
]


def _build_narrative(age: int, sex: str, reaction: str, drug: str, days: int, serious: bool, rng: np.random.Generator) -> str:
    opening = rng.choice(OPENING_TEMPLATES).format(
        age=age, sex=sex.lower(), reaction=reaction.lower(), drug=drug, days=days
    )
    # Imperfect correlation: most serious reports lean severe-sounding, but
    # a meaningful minority read mild; symmetric for non-serious reports.
    if serious:
        severity_clause = rng.choice(SEVERE_SOUNDING_PHRASES) if rng.random() < 0.72 else rng.choice(MILD_SOUNDING_PHRASES)
    else:
        severity_clause = rng.choice(MILD_SOUNDING_PHRASES) if rng.random() < 0.78 else rng.choice(SEVERE_SOUNDING_PHRASES)
    filler = rng.choice(NEUTRAL_FILLER_PHRASES)
    return " ".join(p for p in [opening, severity_clause, filler] if p)


def _pick_reaction(drug: str, rng: np.random.Generator) -> str:
    """Sample a reaction for `drug`, with elevated probability for its known signals."""
    weights = np.ones(len(REACTIONS))
    for d, r in KNOWN_SIGNALS:
        if d == drug and r in REACTIONS:
            weights[REACTIONS.index(r)] *= 12.0
    weights /= weights.sum()
    return rng.choice(REACTIONS, p=weights)


def generate(n_reports: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drug_names = list(DRUGS.keys())

    rows = []
    for i in range(n_reports):
        drug = rng.choice(drug_names)
        reaction = _pick_reaction(drug, rng)
        age = int(np.clip(rng.normal(52, 18), 2, 95))
        sex = rng.choice(["Female", "Male"], p=[0.56, 0.44])

        base_serious_p = 0.55 if reaction in SERIOUS_REACTIONS else 0.12
        serious = rng.random() < base_serious_p

        if serious:
            outcome = rng.choice(
                ["Hospitalization", "Life-threatening", "Death", "Disability"],
                p=[0.62, 0.22, 0.08, 0.08],
            )
        else:
            outcome = "Non-serious"

        narrative = _build_narrative(
            age=age, sex=sex, reaction=reaction, drug=drug,
            days=int(rng.integers(1, 90)), serious=serious, rng=rng,
        )

        rows.append({
            "case_id": f"SIM{100000 + i}",
            "drug_name": drug,
            "drug_class": DRUGS[drug],
            "reaction_pt": reaction,
            "patient_age": age,
            "patient_sex": sex,
            "outcome_serious": serious,
            "outcome_type": outcome,
            "report_year": int(rng.integers(2021, 2026)),
            "narrative_text": narrative,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate(20000, seed=42)
    print(df.head())
    print(df.shape)
    print(f"Serious rate: {df.outcome_serious.mean():.1%}")
