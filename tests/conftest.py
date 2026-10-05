"""Configuration pytest commune.

- Les tests de `test_llm_e2e.py` montent toute la stack Docker (Testcontainers) et
  appellent un juge LLM réel : ils sont marqués `live` et désélectionnés par défaut
  (`addopts = -m 'not live'` dans pyproject.toml). Pour les lancer : `pytest -m live`.
- Les descripteurs Evidently hors-ligne (OOV, Sentiment...) ont besoin de ressources NLTK :
  on les télécharge une seule fois si elles sont absentes.
"""
from pathlib import Path

import nltk
import pytest

_NLTK_RESOURCES = {
    "words": "corpora/words",
    "stopwords": "corpora/stopwords",
    "punkt_tab": "tokenizers/punkt_tab",
    "vader_lexicon": "sentiment/vader_lexicon.zip",
    "wordnet": "corpora/wordnet.zip",
}


def pytest_configure(config):
    for name, path in _NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def pytest_collection_modifyitems(config, items):
    for item in items:
        if Path(str(item.fspath)).name == "test_llm_e2e.py":
            item.add_marker(pytest.mark.live)
