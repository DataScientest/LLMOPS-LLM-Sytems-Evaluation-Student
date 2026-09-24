"""Unit tests of the NeMo Guardrails pre-screen of /search (fail-closed), without LLM or network.

LLMRails is replaced by a mock: no rail is built, no LLM, proxy or database is called.
Run them in the backend image (it ships nemoguardrails and the app dependencies):

    docker compose run --rm --no-deps backend python -m unittest discover -s tests -v
"""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# The app reads these variables at import time (docker compose sets them); the defaults allow a
# plain `docker run`. The clients are lazy: no connection is opened by these tests.
os.environ.setdefault("LITELLM_MODEL", "groq-llama3")
os.environ.setdefault("EMBEDDING_MODEL_NAME", "local-embeddings")
os.environ.setdefault("REDIS_URL", "redis://redis:6379")
os.environ.setdefault("MEILI_URL", "http://meilisearch:7700")

from app.api import search  # noqa: E402
from app.models.search import SearchRequest  # noqa: E402

QUERY = "What is a matrix?"


def mocked_rails(content=None, error=None):
    """Patch LLMRails with a mock whose generate_async returns `content` or raises `error`."""
    rails = MagicMock()
    if error is not None:
        rails.generate_async = AsyncMock(side_effect=error)
    else:
        rails.generate_async = AsyncMock(return_value={"role": "assistant", "content": content})
    # reset the lazily built instance so that get_rails() calls the mocked constructor
    return patch.multiple(search, _rails=None, LLMRails=MagicMock(return_value=rails))


class CheckGuardrailsTest(unittest.IsolatedAsyncioTestCase):
    async def test_rail_error_blocks_the_query(self):
        with mocked_rails(error=ConnectionError("LiteLLM proxy unreachable")), \
                self.assertLogs(search.logger, level="ERROR") as logs:
            self.assertEqual(await search.check_guardrails(QUERY), search.GUARDRAIL_UNAVAILABLE)
        self.assertIn("fail-closed", logs.output[0])

    async def test_rails_init_error_blocks_the_query(self):
        with patch.multiple(search, _rails=None, LLMRails=MagicMock(side_effect=ValueError("bad config"))), \
                self.assertLogs(search.logger, level="ERROR"):
            self.assertEqual(await search.check_guardrails(QUERY), search.GUARDRAIL_UNAVAILABLE)

    async def test_nemo_internal_error_answer_blocks_the_query(self):
        with mocked_rails(content=search.NEMO_INTERNAL_ERROR), self.assertLogs(search.logger, level="ERROR"):
            self.assertEqual(await search.check_guardrails(QUERY), search.GUARDRAIL_UNAVAILABLE)

    async def test_jailbreak_refusal_is_returned(self):
        with mocked_rails(content=search.GUARDRAIL_REFUSAL):
            self.assertEqual(await search.check_guardrails(QUERY), search.GUARDRAIL_REFUSAL)

    async def test_safe_query_passes(self):
        with mocked_rails(content="A matrix is a 2-D array of numbers."):
            self.assertIsNone(await search.check_guardrails(QUERY))


class RagRouteTest(unittest.IsolatedAsyncioTestCase):
    async def test_rail_error_never_reaches_the_rag_pipeline(self):
        rag_search = AsyncMock()
        with mocked_rails(error=RuntimeError("401 invalid proxy key")), \
                patch.object(search, "rag_search", rag_search), \
                self.assertLogs(search.logger, level="ERROR"):
            response = await search.rag_route(SearchRequest(query=QUERY, k=3))
        rag_search.assert_not_called()
        self.assertEqual(response.answer, search.GUARDRAIL_UNAVAILABLE)
        self.assertEqual(response.search_method, "guardrails_unavailable")
        self.assertEqual(response.chunks, [])

    async def test_jailbreak_is_blocked_by_guardrails(self):
        rag_search = AsyncMock()
        with mocked_rails(content=search.GUARDRAIL_REFUSAL), patch.object(search, "rag_search", rag_search):
            response = await search.rag_route(SearchRequest(query=QUERY, k=3))
        rag_search.assert_not_called()
        self.assertEqual(response.search_method, "blocked_by_guardrails")


if __name__ == "__main__":
    unittest.main()
