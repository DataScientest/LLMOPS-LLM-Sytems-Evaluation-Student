import httpx
from typing import List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.utils.hashing import md5_hash
from app.utils.cache import get_json, set_json

_CACHE_TTL = 3600  # seconds

async def _request_embeddings(texts: List[str]) -> Optional[List[dict]]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(
            f"{settings.PROXY_URL}/v1/embeddings",
            json={"model": "local-embeddings", "input": texts, "encoding_format": "float"},
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {settings.PROXY_KEY}"}
        )
        if r.status_code != 200:
            logger.error(f"Embedding request failed: {r.status_code} - {r.text}")
            return None
        return r.json().get("data", [])

async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate sentence embeddings via LiteLLM (TEI backend), with Redis cache.
    Uses utils.hashing + utils.cache for keys & JSON storage.
    """
    results: List[Optional[List[float]]] = [None] * len(texts)
    uncached, idxs = [], []

    # 1) cache lookups
    for i, t in enumerate(texts):
        key = f"embedding:{md5_hash(t)}"
        cached_vec = get_json(key)
        if cached_vec is not None:
            results[i] = cached_vec
        else:
            uncached.append(t)
            idxs.append(i)

    # 2) remote call if needed with batching (TEI/LiteLLM has a max batch size, e.g., 32)
    BATCH_SIZE = 32
    if uncached:
        import asyncio
        for i in range(0, len(uncached), BATCH_SIZE):
            batch = uncached[i : i + BATCH_SIZE]
            batch_idxs = idxs[i : i + BATCH_SIZE]
            
            data = await _request_embeddings(batch)
            if data is None:
                logger.error(f"Batch {i//BATCH_SIZE + 1} failed. Skipping remaining batches.")
                break

            for idx, emb in zip(batch_idxs, data):
                vec = emb.get("embedding", emb)
                if isinstance(vec, dict) and "default" in vec:
                    vec = vec["default"]
                results[idx] = vec
                set_json(f"embedding:{md5_hash(texts[idx])}", vec, _CACHE_TTL)
            
            # Avoid hammering TEI too hard
            if i + BATCH_SIZE < len(uncached):
                await asyncio.sleep(0.1)

    # 3) flatten and filter
    return [v for v in results if v is not None]
