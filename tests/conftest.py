"""Configuration pytest commune : ressources NLTK nécessaires aux descripteurs
Evidently hors-ligne (Sentiment, OOV...), téléchargées une seule fois si absentes."""
import nltk

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
