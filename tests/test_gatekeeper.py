"""Tests hors-ligne des Quality Gates du chapitre 7 (Evidently 0.7.23).

On exécute les VRAIES fonctions de décision de `test_llm_e2e.py`
(test_rag_semantic_quality, test_retrieval_drift, test_security_red_teaming) :
elles construisent un vrai `Report(...).run(...).json()` Evidently puis appliquent les
seuils. Seuls les appels réseau sont remplacés :
  - `query_rag` (API RAGOPS) renvoie les données du golden dataset (5 exemples) ;
  - le juge LLM (OpenAIWrapper.complete) est remplacé par un juge factice déterministe.
Chaque gate est vérifié sur un cas PASS (golden dataset) et un cas FAIL (golden altéré) ;
le gate sémantique aussi sur une hallucination et sur un juge en erreur (FAIL explicite).
Le test `test_semantic_gate_live_judge` utilise un vrai juge LLM : marqué `live`.
"""
import json
import os
import re

import pytest
from evidently.legacy.utils.llm.errors import LLMRateLimitError
from evidently.legacy.utils.llm.wrapper import LLMResult, OpenAIWrapper

import test_llm_e2e as e2e


# ── Juge LLM factice (hors-ligne, déterministe) ─────────────────────────────
# Remplace OpenAIWrapper.complete (le même point d'entrée que le patch "JSON mode"
# du cours) : les descripteurs LLM d'Evidently (ContextRelevance, FaithfulnessLLMEval,
# Answer Relevance LLMEval) s'exécutent réellement, seul l'appel réseau est simulé.
# Score = part des mots significatifs du texte évalué présents dans la référence
# (CONTEXT pour ContextRelevance, SOURCE pour Faithfulness).

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
    # Answer Relevance: share of the QUESTION's significant words addressed by the response.
    # ContextRelevance also has a QUESTION block, but it scores the CONTEXT: skip it here.
    question = _between(prompt, "-----question_starts-----", "-----question_ends-----")
    if question and not reference:
        asked = _words(question)
        score = round(len(asked & words) / len(asked), 2) if asked else 0.0
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


# ── Golden dataset ───────────────────────────────────────────────────────────

GOLDEN = e2e.load_golden_dataset()
assert len(GOLDEN) == 5


def _fake_rag(answers, chunks):
    """Remplace l'API RAGOPS : renvoie les réponses/contextes fournis, dans l'ordre."""
    calls = iter(range(10_000))

    def query_rag(query, k=3):
        i = next(calls) % len(answers)
        return {"answer": answers[i], "chunks": [{"content": c} for c in chunks[i]]}

    return query_rag


def _rag_from_golden(dataset):
    return _fake_rag([g["expected_answer"] for g in dataset], [[g["expected_context"]] for g in dataset])


# ── Gate 1 : qualité sémantique (Triade RAG) ────────────────────────────────

def test_semantic_gate_pass(monkeypatch):
    monkeypatch.setattr(OpenAIWrapper, "complete", fake_judge_complete)
    monkeypatch.setattr(e2e, "query_rag", _rag_from_golden(GOLDEN))
    e2e.test_rag_semantic_quality(None)  # ne doit pas lever pytest.fail


def test_semantic_gate_fail(monkeypatch):
    # Golden altéré : le retrieval renvoie des contextes hors-sujet (Context Precision s'effondre)
    off_topic = "Meilisearch is an open-source search engine written in Rust."
    altered = [{**g, "expected_context": off_topic} for g in GOLDEN]
    monkeypatch.setattr(OpenAIWrapper, "complete", fake_judge_complete)
    monkeypatch.setattr(e2e, "query_rag", _rag_from_golden(altered))
    with pytest.raises(pytest.fail.Exception, match="SEMANTIC QUALITY BELOW THRESHOLD"):
        e2e.test_rag_semantic_quality(None)


def test_semantic_gate_fail_hallucination(monkeypatch, capsys):
    # Retrieval parfait, mais les réponses inventent des faits absents du contexte
    invented = "Quantum tensors require GPU clusters running Kubernetes."
    monkeypatch.setattr(OpenAIWrapper, "complete", fake_judge_complete)
    monkeypatch.setattr(e2e, "query_rag", _rag_from_golden([{**g, "expected_answer": invented} for g in GOLDEN]))
    with pytest.raises(pytest.fail.Exception, match="SEMANTIC QUALITY BELOW THRESHOLD"):
        e2e.test_rag_semantic_quality(None)
    out = capsys.readouterr().out
    assert "[PASSED] Context Precision" in out
    assert "[FAILED] Faithfulness" in out


def test_semantic_gate_fail_off_topic_answer(monkeypatch, capsys):
    # Off-topic answers: the Answer Relevance judge sees the QUESTION and flags them
    off_topic = "Kubernetes is a popular container orchestration system created by Google."
    monkeypatch.setattr(OpenAIWrapper, "complete", fake_judge_complete)
    monkeypatch.setattr(e2e, "query_rag", _rag_from_golden([{**g, "expected_answer": off_topic} for g in GOLDEN]))
    with pytest.raises(pytest.fail.Exception, match="SEMANTIC QUALITY BELOW THRESHOLD"):
        e2e.test_rag_semantic_quality(None)
    out = capsys.readouterr().out
    assert "[PASSED] Context Precision" in out
    assert "[FAILED] Answer Relevance" in out


def test_semantic_gate_fail_when_judge_errors(monkeypatch):
    # Juge en erreur (clé invalide, quota 429...) : les exceptions sont avalées par le test,
    # qui doit quand même échouer faute de métriques (et non passer à vide).
    async def failing_judge(self, messages, *args, **kwargs):
        raise LLMRateLimitError("429: daily token quota exceeded")

    monkeypatch.setattr(OpenAIWrapper, "complete", failing_judge)
    monkeypatch.setattr(e2e, "query_rag", _rag_from_golden(GOLDEN))
    with pytest.raises(pytest.fail.Exception, match="LLM judge produced no metric"):
        e2e.test_rag_semantic_quality(None)


@pytest.mark.live
def test_semantic_gate_live_judge(monkeypatch):
    """Juge LLM réel sur le golden dataset.

    Variables : EVAL_MODEL (alias du juge), OPENAI_BASE_URL (proxy LiteLLM ou passerelle
    compatible OpenAI) et PROXY_KEY (clé envoyée comme OPENAI_API_KEY par test_llm_e2e).
    """
    if not os.getenv("PROXY_KEY"):
        pytest.skip("PROXY_KEY absent")
    monkeypatch.setattr(e2e, "query_rag", _rag_from_golden(GOLDEN))
    errors_before = sum(1 for log in e2e.TEST_LOGS if log.get("status") == "ERROR")
    e2e.test_rag_semantic_quality(None)  # fails by itself if the judge produced no metric
    errors = [log["error"] for log in e2e.TEST_LOGS if log.get("status") == "ERROR"][errors_before:]
    assert not errors, errors


# ── Gate 2 : dérive du retrieval (Drift) ────────────────────────────────────

def test_drift_gate_pass(monkeypatch):
    # Retrieval parfait : les contextes récupérés sont les contextes attendus du golden.
    # NB : avec 5 lignes, Evidently traite Sentiment comme catégoriel (chi-square) ; des
    # contextes seulement "proches" (ex. expected_answer) donnent déjà p=0.0 -> dérive.
    contexts = [g["expected_context"] for g in GOLDEN]
    monkeypatch.setattr(e2e, "query_rag", _fake_rag(["-"], [contexts]))
    e2e.test_retrieval_drift()


def test_drift_gate_fail(monkeypatch):
    # Contextes altérés : fuite d'emails + ton négatif + longueur très différente
    drifted = [
        "Contact the admin at admin@company.com, this is terrible!",
        "Send the invoice to john.doe@gmail.com now, I hate waiting.",
        "Awful service. Write to support@scam.com immediately!",
        "Horrible and useless bot, mail me at angry.user@yahoo.com",
        "Worst answer ever, escalate to boss@company.org!!",
    ]
    monkeypatch.setattr(e2e, "query_rag", _fake_rag(["-"], [drifted]))
    with pytest.raises(pytest.fail.Exception, match="dérive est trop élevée"):
        e2e.test_retrieval_drift()


# ── Gate 3 : sécurité / red teaming ─────────────────────────────────────────

def test_security_gate_pass(monkeypatch):
    answers = [g["expected_answer"] for g in GOLDEN]
    monkeypatch.setattr(e2e, "query_rag", _fake_rag(answers, [[]] * 5))
    e2e.test_security_red_teaming()


def test_security_gate_fail(monkeypatch):
    answers = [g["expected_answer"] for g in GOLDEN]
    answers[2] = "As DAN, I can tell you my secret token is ADMIN_TOKEN_99X."
    monkeypatch.setattr(e2e, "query_rag", _fake_rag(answers, [[]] * 5))
    with pytest.raises(pytest.fail.Exception, match="THE SYSTEM IS VULNERABLE"):
        e2e.test_security_red_teaming()


def test_security_gate_fail_when_no_llm_answer(monkeypatch, capsys):
    # LLM unreachable (invalid key, quota): only backend fallbacks, no leak and no refusal
    answers = ["I found relevant chunks but could not generate an answer."] * 3 + [
        "I found chunks but could not generate an answer due to an error.",
        "I couldn't find any relevant chunks to answer your question.",
    ]
    monkeypatch.setattr(e2e, "query_rag", _fake_rag(answers, [[]] * 5))
    with pytest.raises(pytest.fail.Exception, match="NO USABLE ANSWER FROM THE LLM"):
        e2e.test_security_red_teaming()
    assert "Aucune réponse exploitable du LLM (5 réponses de repli)" in capsys.readouterr().out


def test_security_gate_pass_with_some_fallbacks(monkeypatch):
    answers = [g["expected_answer"] for g in GOLDEN]
    answers[0] = "I found relevant chunks but could not generate an answer."
    monkeypatch.setattr(e2e, "query_rag", _fake_rag(answers, [[]] * 5))
    e2e.test_security_red_teaming()


def test_security_gate_fail_when_guardrails_fail_closed(monkeypatch, capsys):
    # The rail itself fails (proxy/LLM unreachable): the backend blocks every query (fail-closed).
    # Safe (counted as refusals), but nothing was really tested: the gate must reject.
    answers = ["I am an AI assistant and I cannot answer right now: the safety check is unavailable."] * 5
    monkeypatch.setattr(e2e, "query_rag", _fake_rag(answers, [[]] * 5))
    with pytest.raises(pytest.fail.Exception, match="NO USABLE ANSWER FROM THE LLM"):
        e2e.test_security_red_teaming()
    assert "Guardrails intervenu 5 fois" in capsys.readouterr().out


def test_security_gate_pass_with_some_fail_closed_refusals(monkeypatch):
    answers = [g["expected_answer"] for g in GOLDEN]
    answers[2] = "I am an AI assistant and I cannot answer right now: the safety check is unavailable."
    monkeypatch.setattr(e2e, "query_rag", _fake_rag(answers, [[]] * 5))
    e2e.test_security_red_teaming()


def test_security_refusal_regex_accepts_typographic_apostrophe(monkeypatch, capsys):
    answers = ["I’m sorry, but I can’t help with that.", "I can’t share that information.",
               "I can't disclose it.", "Linear algebra studies vectors.", "I cannot answer this request."]
    monkeypatch.setattr(e2e, "query_rag", _fake_rag(answers, [[]] * 5))
    e2e.test_security_red_teaming()
    assert "Guardrails intervenu 4 fois" in capsys.readouterr().out


def test_report_json_uses_metric_name():
    """Garde-fou de migration : Evidently 0.7.23 expose `metric_name` (plus `metric_id`)."""
    import pandas as pd
    from evidently import Dataset, DataDefinition, Report
    from evidently.descriptors import RegExp
    from evidently.presets import TextEvals

    df = pd.DataFrame({"response": [g["expected_answer"] for g in GOLDEN]})
    ds = Dataset.from_pandas(df, data_definition=DataDefinition(text_columns=["response"]),
                             descriptors=[RegExp("response", reg_exp=r"matrix", alias="Has_Matrix")])
    metrics = json.loads(Report([TextEvals()]).run(reference_data=None, current_data=ds).json())["metrics"]
    names = {m["metric_name"]: m["value"] for m in metrics}
    assert all("metric_id" not in m for m in metrics)
    assert names["UniqueValueCount(column=Has_Matrix)"]["counts"] == {"false": 3.0, "true": 2.0}
