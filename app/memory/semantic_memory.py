"""Long-term memory across tasks, stored in ChromaDB.

Every finished interaction is embedded and stored. Before routing a new
request, the most similar past entries are fetched as context. This is
separate from the LangGraph checkpointer, which only stores the state of one
paused run so it can be resumed.
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

        # Embeddings are always passed in, so Chroma never needs to download
        # its default embedding model.
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
        """Most similar past entries, best first, that reach `min_similarity`.

        Chroma always returns the top-k nearest entries, even when none of
        them are related, so results below the threshold are dropped.

        Vectors are L2-normalised, so Chroma's squared-L2 distance d converts
        to cosine similarity as 1 - d / 2. The 0.3 threshold was picked by
        trial against the hashing embedder and would need re-tuning for
        OpenAI embeddings.
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
        for doc, distance in zip(documents, distances, strict=True):
            cosine_similarity = 1.0 - (distance / 2.0)
            if cosine_similarity >= min_similarity:
                matches.append(doc)
        return matches
