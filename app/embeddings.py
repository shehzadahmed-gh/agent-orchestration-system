"""Embedding providers for semantic memory.

LocalHashingEmbeddingProvider needs no model download or API key, but it is a
hashed bag of words with no learned meaning. Set EMBEDDING_BACKEND=openai for
real embeddings. This is a small copy of the same module in rag-hybrid-search,
kept here so the two services don't depend on each other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import HashingVectorizer


class EmbeddingProvider(ABC):
    name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalHashingEmbeddingProvider(EmbeddingProvider):
    name = "local-hashing"

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._vectorizer = HashingVectorizer(
            n_features=dimension, norm="l2", alternate_sign=True, ngram_range=(1, 2), lowercase=True
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._vectorizer.transform(texts).toarray().tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def build_embedding_provider(settings) -> EmbeddingProvider:
    if settings.embedding_backend == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDING_BACKEND=openai requires OPENAI_API_KEY to be set.")
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key, model=settings.openai_embedding_model)
    return LocalHashingEmbeddingProvider(dimension=settings.embedding_dimension)
