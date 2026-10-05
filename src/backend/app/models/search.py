from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = 5
    use_embeddings: Optional[bool] = True

class ChunkHit(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str  # preview (first 300 characters)
    full_content: str = ""  # full chunk text, as given to the LLM
    metadata: Dict[str, Any] = {}

class SearchResponse(BaseModel):
    answer: str
    chunks: List[ChunkHit]
    total_chunks_found: int
    cached: bool
    search_method: str  # "text" | "vector" | "hybrid"

class DirectSearchResult(BaseModel):
    hits: List[Dict[str, Any]]
    total: int
    query: str
    index_used: str
    search_method: str
