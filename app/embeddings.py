"""
Same offline-by-default embedding pattern used in the rag-hybrid-search
project: a dependency-light local provider so semantic memory works with no
API key, and a real OpenAI-backed provider as the production swap-in. Kept
as a small local copy (rather than importing the sibling project) so this
service has no cross-repo dependency.
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
