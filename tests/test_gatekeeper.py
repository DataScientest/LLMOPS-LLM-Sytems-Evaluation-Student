"""Test hors-ligne du gatekeeper de dérive du chapitre 2 (Evidently 0.7.23).

Sur cette branche, `src/app.py` contient la solution du chapitre 1 et `src/check_drift.py`
est écrit par l'étudiant (cours, chapitre 2) : il n'existe pas encore. Ce test :
  1. reproduit l'évaluateur de dérive de la solution du cours (Sentiment, TextLength,
     RegExp email + TextEvals + DataDriftPreset avec les mêmes seuils) sur un golden
     dataset de 5 requêtes (référence) / 5 requêtes (courant) : vrai
     `Report(...).run(...).json()` Evidently 0.7.23 ;
  2. applique la règle de décision de `check_drift.py` décrite dans le cours, réécrite ici
     dans `drift_gate()` avec la clé `metric_name` (la clé `metric_id` a été renommée en
     0.7.23 : le script du cours ne détecterait plus aucune dérive).
"""
import json
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import RegExp, Sentiment, TextLength
from evidently.presets import DataDriftPreset, TextEvals

GOLDEN = json.loads((Path(__file__).parent / "golden_gatekeeper.json").read_text())
assert len(GOLDEN["reference"]) == len(GOLDEN["current"]) == 5

EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
DRIFT_SHARE_THRESHOLD = 0.1
STAT_THRESHOLD = 0.1


def run_drift_report(reference: list[str], current: list[str]) -> dict:
    data_definition = DataDefinition(text_columns=["user_query"])

    def dataset(queries):
        return Dataset.from_pandas(
            pd.DataFrame({"user_query": queries}),
            data_definition=data_definition,
            descriptors=[
                Sentiment("user_query", alias="Sentiment"),
                TextLength("user_query", alias="Length"),
                RegExp("user_query", reg_exp=EMAIL_REGEX, alias="Email_PII"),
            ],
        )

    report = Report([
        TextEvals(),
        DataDriftPreset(text_method="perc_text_content_drift", text_threshold=0.1, drift_share=0.1, threshold=0.4),
    ])
    snapshot = report.run(reference_data=dataset(reference), current_data=dataset(current))
    return json.loads(snapshot.json())


def drift_gate(report: dict) -> bool:
    """True si le déploiement doit être bloqué (règle de check_drift.py du cours)."""
    failed = False
    for m in report["metrics"]:
        metric_name = m.get("metric_name", "")
        if metric_name.startswith("DriftedColumnsCount") and m["value"]["share"] > DRIFT_SHARE_THRESHOLD:
            failed = True
        if metric_name.startswith("ValueDrift(column=Sentiment") and m["value"] < STAT_THRESHOLD:
            failed = True
        if metric_name.startswith("ValueDrift(column=Email_PII") and m["value"] < STAT_THRESHOLD:
            failed = True
    return failed


def test_drift_gate_pass():
    report = run_drift_report(GOLDEN["reference"], GOLDEN["current"])
    assert drift_gate(report) is False


def test_drift_gate_fail():
    # Golden altéré (semaine B du cours) : clients agressifs, fuite d'emails, autre langue
    altered = [
        "Your service is terrible! I want a refund now!",
        "Can you send the invoice to my email: john.doe@gmail.com?",
        "I am very angry with the delay. Fix it!",
        "Hola, ¿pueden ayudarme con mi pedido?",
        "Contact me at support@scam.com",
    ]
    report = run_drift_report(GOLDEN["reference"], altered)
    assert drift_gate(report) is True


def test_report_json_uses_metric_name():
    """Garde-fou de migration : `metric_name` (plus `metric_id`) ; ValueDrift inclut la méthode."""
    report = run_drift_report(GOLDEN["reference"], GOLDEN["current"])
    names = [m["metric_name"] for m in report["metrics"]]
    assert all("metric_id" not in m for m in report["metrics"])
    assert any(n.startswith("DriftedColumnsCount(") for n in names)
    assert any(n.startswith("ValueDrift(column=Sentiment,") for n in names)
