"""Test hors-ligne du gatekeeper structurel du chapitre 3 (Evidently 0.7.23).

Sur cette branche, `src/app.py` et `src/check_structure.py` sont écrits par l'étudiant
(cours, chapitre 3) : ils n'existent pas encore. Ce test :
  1. reproduit l'évaluateur de la solution du cours (IsValidJSON, JSONSchemaMatch, RegExp
     + preset TextEvals) sur 5 prédictions figées (tests/golden_gatekeeper.json) au lieu
     d'appeler Gemma 3 via Ollama : vrai `Report(...).run(...).json()` Evidently 0.7.23 ;
  2. applique la règle de décision de `check_structure.py` décrite dans le cours
     (syntaxe >= 90 %, schéma >= 80 %, "Strict_Start_JSON" informatif), réécrite ici
     dans `structure_gate()`.

Écart constaté avec la solution du cours : les descripteurs IsValidJSON / JSONSchemaMatch /
RegExp sont booléens ; Evidently ne produit pas `MeanValue(column=Syntax_OK)` mais
`UniqueValueCount(column=Syntax_OK)` (déjà vrai en 0.7.2). Le script du cours, qui cherche
`MeanValue(...)` via la clé `metric_id` (renommée `metric_name` en 0.7.23), ne rejette donc
jamais rien. `structure_gate()` lit le taux de succès dans `value["shares"]["true"]`.
"""
import json
from pathlib import Path

import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import IsValidJSON, JSONSchemaMatch, RegExp
from evidently.presets import TextEvals

GOLDEN = json.loads((Path(__file__).parent / "golden_gatekeeper.json").read_text())
assert len(GOLDEN) == 5

SCHEMA = {"vendor": str, "total": float}
THRESHOLDS = {"Syntax_OK": 0.9, "Schema_Compliance": 0.8}


def run_structural_report(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    dataset = Dataset.from_pandas(
        df,
        data_definition=DataDefinition(text_columns=["prediction"]),
        descriptors=[
            IsValidJSON("prediction", alias="Syntax_OK"),
            JSONSchemaMatch("prediction", expected_schema=SCHEMA, alias="Schema_Compliance"),
            RegExp("prediction", reg_exp=r"^\{", alias="Strict_Start_JSON"),
        ],
    )
    return json.loads(Report([TextEvals()]).run(reference_data=None, current_data=dataset).json())


def success_rate(report: dict, column: str) -> float:
    for m in report["metrics"]:
        if m["metric_name"] == f"UniqueValueCount(column={column})":
            return m["value"]["shares"].get("true", 0.0)
    raise KeyError(column)


def structure_gate(report: dict) -> bool:
    """True si le déploiement doit être rejeté (règle de check_structure.py du cours)."""
    return any(success_rate(report, col) < threshold for col, threshold in THRESHOLDS.items())


def test_structure_gate_pass():
    report = run_structural_report(GOLDEN)
    assert success_rate(report, "Syntax_OK") == 1.0
    assert success_rate(report, "Schema_Compliance") == 1.0
    assert success_rate(report, "Strict_Start_JSON") == 1.0
    assert structure_gate(report) is False


def test_structure_gate_fail():
    # Golden altéré : modèle bavard (texte avant le JSON) + clé inventée + montant en texte
    altered = [dict(g) for g in GOLDEN]
    altered[0]["prediction"] = 'Here is the data: {"vendor": "CloudCorp", "total": 560.0}'
    altered[1]["prediction"] = '{"company": "Luigi\'s", "total": 42.5}'
    altered[2]["prediction"] = '{"vendor": "Netflix", "total": "15.99"}'
    report = run_structural_report(altered)
    assert success_rate(report, "Syntax_OK") == 0.8
    assert structure_gate(report) is True


def test_report_json_uses_metric_name():
    """Garde-fou de migration : `metric_name` (plus `metric_id`), pas de MeanValue pour les booléens."""
    report = run_structural_report(GOLDEN)
    names = {m["metric_name"] for m in report["metrics"]}
    assert all("metric_id" not in m for m in report["metrics"])
    assert "UniqueValueCount(column=Syntax_OK)" in names
    assert "MeanValue(column=Syntax_OK)" not in names
