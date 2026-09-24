"""Test hors-ligne du gatekeeper du chapitre 1 (Evidently 0.7.23).

Sur cette branche, `src/app.py` et `src/check_limits.py` sont écrits par l'étudiant
(cours, chapitre 1) : ils sont vides / absents. Ce test :
  1. reproduit l'évaluateur de la solution du cours (TextLength, OOVWordsPercentage,
     Sentiment, RegExp de refus + preset TextEvals) sur un golden dataset de 5 échanges :
     vrai `Report(...).run(...).json()` Evidently 0.7.23 ;
  2. applique la règle de décision de `check_limits.py` décrite dans le cours, réécrite
     ici dans `limits_gate()` et adaptée au JSON d'Evidently 0.7.23 :
       - la clé `metric_id` est devenue `metric_name` ;
       - le descripteur RegExp est booléen : ses comptes sont sous "true"/"false"
         (et non plus "1"/"0").
     Avec le script du cours tel quel, aucune règle ne serait plus évaluée.
"""
import json
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import OOVWordsPercentage, RegExp, Sentiment, TextLength
from evidently.presets import TextEvals

GOLDEN = json.loads((Path(__file__).parent / "golden_gatekeeper.json").read_text())
assert len(GOLDEN) == 5

REFUSAL_REGEX = r"(?i)(cannot answer|I don't know)"
OOV_MAX = 20


def run_smoke_report(rows: list[dict]) -> dict:
    dataset = Dataset.from_pandas(
        pd.DataFrame(rows),
        data_definition=DataDefinition(text_columns=["question", "answer"]),
        descriptors=[
            TextLength("answer", alias="Length"),
            OOVWordsPercentage("answer", alias="OOV"),
            Sentiment("answer", alias="Sentiment"),
            RegExp("answer", reg_exp=REFUSAL_REGEX, alias="Refusal_Regex"),
        ],
    )
    return json.loads(Report([TextEvals()]).run(reference_data=None, current_data=dataset).json())


def limits_gate(report: dict) -> bool:
    """True si le déploiement doit être rejeté (règle de check_limits.py du cours)."""
    failed = False
    for m in report["metrics"]:
        metric_name = m.get("metric_name")
        if metric_name == "RowCount()" and m.get("value", 0) <= 0:
            failed = True
        if metric_name == "MeanValue(column=OOV)" and m.get("value", 0) > OOV_MAX:
            failed = True
        if metric_name == "UniqueValueCount(column=Refusal_Regex)":
            if m.get("value", {}).get("counts", {}).get("true", 0) > 0:
                failed = True
    return failed


def test_limits_gate_pass():
    assert limits_gate(run_smoke_report(GOLDEN)) is False


def test_limits_gate_fail_refusal():
    altered = [dict(g) for g in GOLDEN]
    altered[1]["answer"] = "Sorry, I don't know if we deliver there."
    assert limits_gate(run_smoke_report(altered)) is True


def test_limits_gate_fail_oov():
    # Hallucination de jargon : mots inventés -> OOV moyen > 20 %
    altered = [dict(g, answer="Zorblax the flimtrap via quandex prottle snirf.") for g in GOLDEN]
    assert limits_gate(run_smoke_report(altered)) is True


def test_report_json_uses_metric_name():
    """Garde-fou de migration : `metric_name` (plus `metric_id`), comptes booléens "true"/"false"."""
    report = run_smoke_report(GOLDEN)
    metrics = {m["metric_name"]: m["value"] for m in report["metrics"]}
    assert all("metric_id" not in m for m in report["metrics"])
    assert metrics["RowCount()"] == 5
    assert metrics["UniqueValueCount(column=Refusal_Regex)"]["counts"] == {"false": 5.0}
