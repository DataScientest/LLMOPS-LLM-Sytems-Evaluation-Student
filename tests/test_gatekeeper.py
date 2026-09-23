"""Test hors-ligne du gatekeeper sémantique du chapitre 4 (Evidently 0.7.23).

- Évaluateur : sur cette branche `src/eval/eval_rag.py` est volontairement vide (l'étudiant
  l'écrit). Le test REPRODUIT donc la construction des rapports de la solution du cours
  (ContextRelevance -> "Context Precision", FaithfulnessLLMEval -> "Faithfulness",
  CompletenessLLMEval -> "Answer Relevance", preset TextEvals) : vrais
  `Report(...).run(...)` Evidently 0.7.23 sauvegardés en JSON.
- Juge LLM : remplacé par un juge factice déterministe (hors-ligne).
- Gatekeeper : le VRAI `src/eval/check_semantic.py` (exit 0 = PASS, exit 1 = FAIL).
Cas PASS : golden dataset (5 exemples). Cas FAIL : golden altéré (mauvais retrieval,
hallucinations) et rapports sans les scores du juge (juge en erreur).

Faithfulness / Answer Relevance sont des descripteurs catégoriels : avec `include_score=True`
(solution du cours), Evidently ajoute les colonnes `<alias> score` et donc les métriques
`MeanValue(column=Faithfulness score)` / `MeanValue(column=Answer Relevance score)`.
Ce score mesure la catégorie négative (0.0 = FAITHFUL/COMPLETE, 1.0 = UNFAITHFUL/INCOMPLETE) :
check_semantic.py compare `1 - score` aux seuils.
"""
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
from evidently import Dataset, DataDefinition, Report
from evidently.descriptors import CompletenessLLMEval, ContextRelevance, FaithfulnessLLMEval
from evidently.legacy.utils.llm.wrapper import LLMResult, OpenAIWrapper
from evidently.presets import TextEvals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))

import check_semantic  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "golden_gatekeeper.json").read_text())
assert len(GOLDEN) == 5
EVAL_MODEL = os.getenv("EVAL_MODEL", "groq-llama3")


# ── Juge LLM factice (hors-ligne, déterministe) ─────────────────────────────
# Remplace OpenAIWrapper.complete (le même point d'entrée que le patch "JSON mode"
# du cours) : les descripteurs LLM d'Evidently (ContextRelevance, FaithfulnessLLMEval,
# CompletenessLLMEval) s'exécutent réellement, seul l'appel réseau est simulé.
# Score = part des mots significatifs du texte évalué présents dans la référence
# (CONTEXT pour ContextRelevance, SOURCE pour Faithfulness/Completeness).

_STOP = {"what", "is", "a", "an", "the", "of", "to", "and", "in", "for", "that", "this",
         "are", "it", "its", "we", "us", "our", "with", "be", "by", "on", "as", "all", "yes",
         "how", "does", "do", "which", "use", "can", "you", "new"}


def _words(text: str) -> set[str]:
    # racine grossière (5 premières lettres) : "scale"/"scaled", "route"/"routed"
    return {w[:5] for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOP}


def _between(prompt: str, start: str, end: str) -> str:
    # dernière occurrence : la consigne "Classify text between ___text_starts_here___ and ..."
    # cite aussi les marqueurs
    matches = re.findall(re.escape(start) + r"[ \t]*\n(.*?)\n\s*" + re.escape(end), prompt, re.S)
    return matches[-1] if matches else ""


async def fake_judge_complete(self, messages, *args, **kwargs):
    prompt = messages[-1].content
    text = _between(prompt, "___text_starts_here___", "___text_ends_here___")
    reference = (_between(prompt, "-----context_starts-----", "-----context_ends-----")
                 or _between(prompt, "-----source_starts-----", "-----source_finishes-----"))
    words = _words(text)
    score = round(len(words & _words(reference)) / len(words), 2) if words else 0.0
    positive, negative = re.search(r"into two categories: (\w+) and (\w+)", prompt).groups()
    # l'ordre des catégories varie selon le descripteur : on repère la catégorie "positive"
    if positive.startswith(("IR", "UN", "IN")):
        positive, negative = negative, positive
    category = positive if score >= 0.5 else negative
    # Same convention as the real prompt: "0.0 is absolute FAITHFUL and 1.0 is absolute
    # UNFAITHFUL" -> the returned score is the likelihood of the negative category.
    direction = re.search(r"where 0\.0 is absolute (\w+)", prompt)
    if direction and direction.group(1) == positive:
        score = round(1 - score, 2)
    return LLMResult(f'{{"category": "{category}", "score": {score}, "reasoning": "fake judge"}}', 0, 0)


def _write_reports(rows: list[dict], out_dir: Path) -> list[str]:
    """Reproduit eval_rag.py (solution du cours) et renvoie les chemins des rapports JSON."""
    df = pd.DataFrame(rows)
    data_def = DataDefinition(text_columns=["question", "context", "response"])
    dataset = Dataset.from_pandas(df, data_definition=data_def, descriptors=[
        ContextRelevance("question", "context", output_scores=True, aggregation_method="mean",
                         method="llm", method_params={"provider": "openai", "model": EVAL_MODEL},
                         alias="Context Precision"),
    ])
    dataset_llm = Dataset.from_pandas(df, data_definition=data_def, descriptors=[
        FaithfulnessLLMEval("response", context="context", provider="openai", model=EVAL_MODEL,
                            include_score=True, alias="Faithfulness"),
        CompletenessLLMEval("response", context="context", provider="openai", model=EVAL_MODEL,
                            include_score=True, alias="Answer Relevance"),
    ])
    paths = [out_dir / "chapter4_context_precision.json", out_dir / "chapter4_semantic_report.json"]
    Report(metrics=[TextEvals()]).run(reference_data=None, current_data=dataset).save_json(str(paths[0]))
    Report(metrics=[TextEvals()]).run(reference_data=None, current_data=dataset_llm).save_json(str(paths[1]))
    return [str(p) for p in paths]


def _gate_exit_code(paths):
    with pytest.raises(SystemExit) as exc:
        check_semantic.check_semantic_quality(paths)
    return exc.value.code


def test_semantic_gate_pass(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(OpenAIWrapper, "complete", fake_judge_complete)
    assert _gate_exit_code(_write_reports(GOLDEN, tmp_path)) == 0
    out = capsys.readouterr().out
    for axis in ("Context Precision", "Faithfulness", "Answer Relevance"):
        assert f"[PASSED] {axis}" in out


def test_semantic_gate_fail(monkeypatch, tmp_path, capsys):
    # Golden altéré : le retrieval renvoie un contexte hors-sujet pour chaque question
    off_topic = "To add a new document, send a POST request to /ingest."
    altered = [{**g, "context": off_topic} for g in GOLDEN[:3]] + GOLDEN[3:]
    monkeypatch.setattr(OpenAIWrapper, "complete", fake_judge_complete)
    assert _gate_exit_code(_write_reports(altered, tmp_path)) == 1
    assert "[FAILED] Context Precision" in capsys.readouterr().out


def test_semantic_gate_fail_hallucination(monkeypatch, tmp_path, capsys):
    # Golden altéré : les réponses inventent des faits absents du contexte
    invented = "Yes, we use PostgreSQL version 16 with Kafka streaming for billing."
    altered = [{**g, "response": invented} for g in GOLDEN[:3]] + GOLDEN[3:]
    monkeypatch.setattr(OpenAIWrapper, "complete", fake_judge_complete)
    assert _gate_exit_code(_write_reports(altered, tmp_path)) == 1
    out = capsys.readouterr().out
    assert "[PASSED] Context Precision" in out  # retrieval is fine, only the answers are wrong
    assert "[FAILED] Faithfulness" in out


def test_semantic_gate_fail_when_judge_scores_missing(monkeypatch, tmp_path, capsys):
    # Juge en erreur (clé invalide, quota 429...) : le rapport sémantique ne contient aucune
    # métrique de score -> le gate doit échouer au lieu de valider sur Context Precision seul.
    monkeypatch.setattr(OpenAIWrapper, "complete", fake_judge_complete)
    ctx_report, llm_report = _write_reports(GOLDEN, tmp_path)
    Path(llm_report).write_text(json.dumps({"metrics": []}))
    assert _gate_exit_code([ctx_report, llm_report]) == 1
    out = capsys.readouterr().out
    assert "[MISSING] MeanValue(column=Faithfulness score)" in out
    assert "[MISSING] MeanValue(column=Answer Relevance score)" in out


def test_report_json_uses_metric_name(monkeypatch, tmp_path):
    """Garde-fou de migration : Evidently 0.7.23 expose `metric_name` (plus `metric_id`)."""
    monkeypatch.setattr(OpenAIWrapper, "complete", fake_judge_complete)
    names = []
    for path in _write_reports(GOLDEN, tmp_path):
        metrics = json.loads(Path(path).read_text())["metrics"]
        assert all("metric_id" not in m for m in metrics)
        names += [m["metric_name"] for m in metrics]
    assert "MeanValue(column=Context Precision)" in names
    assert "UniqueValueCount(column=Faithfulness)" in names
    assert "MeanValue(column=Faithfulness score)" in names
    assert "MeanValue(column=Answer Relevance score)" in names


@pytest.mark.live
def test_semantic_gate_live_judge(tmp_path):
    """Juge LLM réel : EVAL_MODEL, OPENAI_BASE_URL et OPENAI_API_KEY doivent être définis
    (ex. proxy LiteLLM : OPENAI_BASE_URL=http://localhost:4000/v1, EVAL_MODEL=groq-llama3)."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY absent")
    assert _gate_exit_code(_write_reports(GOLDEN, tmp_path)) == 0
