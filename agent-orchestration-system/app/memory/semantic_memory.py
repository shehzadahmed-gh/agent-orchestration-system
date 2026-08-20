"""
Semantic memory - the ChromaDB half of the "two persistence layers" split
(see app/db/models.py for the PostgreSQL/structured half).

This is long-term, cross-session recall: every finalized interaction is
embedded and stored here, and future runs retrieve the most similar past
entries as context before the supervisor routes the request. It is
deliberately separate from the LangGraph checkpointer (which persists a
single thread's exact working state for pause/resume) - this store is
about recalling relevant *past* interactions, potentially from entirely
different threads/tasks.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from app.embeddings import EmbeddingProvider


class SemanticMemory:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        collection_name: str,
        url: str | None = None,
        local_path: str | None = None,
        top_k: int = 3,
    ):
        import chromadb

        if url:
            host, _, port = url.replace("http://", "").replace("https://", "").partition(":")
            self._client = chromadb.HttpClient(host=host, port=int(port or 8000))
        else:
            self._client = chromadb.PersistentClient(path=local_path or "./data/chroma_local")

        # We supply our own embeddings explicitly, so no implicit embedding
        # function (and therefore no implicit model download) is ever needed.
        self._collection = self._client.get_or_create_collection(collection_name)
        self._embedder = embedder
        self._default_top_k = top_k

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        entry_id = str(uuid.uuid4())
        vector = self._embedder.embed_one(text)
        self._collection.upsert(
            ids=[entry_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[{**(metadata or {}), "created_at": time.time()}],
        )
        return entry_id

    def search(self, query: str, top_k: int | None = None, min_similarity: float = 0.3) -> list[str]:
        """
        Returns the most similar past entries, best-first - but only ones
        that actually clear `min_similarity`. Chroma's `.query()` always
        returns its top-N *nearest* neighbours regardless of whether any of
        them are meaningfully close, so without this gate a query on a
        totally unrelated topic would still silently return "the least
        dissimilar" memories instead of nothing.

        The embedder L2-normalizes every vector, so Chroma's default
        (squared-L2) distance recovers true cosine similarity via
        `1 - distance / 2`: 1.0 for identical, 0.0 for orthogonal/unrelated,
        negative for opposed. `min_similarity=0.3` is tuned for the offline
        hashing embedder, which has no learned semantics - recalibrate
        (likely upward, since real embeddings separate topics more cleanly)
        when EMBEDDING_BACKEND=openai.
        """
        if not query:
            return []
        try:
            count = self._collection.count()
        except Exception:
            count = 0
        if count == 0:
            return []
        vector = self._embedder.embed_one(query)
        n_results = min(top_k or self._default_top_k, count)
        results = self._collection.query(query_embeddings=[vector], n_results=n_results)
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        matches = []
        for doc, distance in zip(documents, distances):
            cosine_similarity = 1.0 - (distance / 2.0)
            if cosine_similarity >= min_similarity:
                matches.append(doc)
        return matches
