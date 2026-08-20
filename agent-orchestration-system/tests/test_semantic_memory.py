import tempfile

from app.embeddings import LocalHashingEmbeddingProvider
from app.memory.semantic_memory import SemanticMemory


def _memory(tmp_path):
    embedder = LocalHashingEmbeddingProvider(dimension=384)
    return SemanticMemory(embedder=embedder, collection_name="test_memory", local_path=str(tmp_path))


def test_search_on_empty_memory_returns_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        memory = _memory(tmp)
        assert memory.search("anything") == []


def test_add_and_recall_related_entry():
    with tempfile.TemporaryDirectory() as tmp:
        memory = _memory(tmp)
        memory.add("Q: What is 12 * (3 + 4) / 2?\nA: 12 * (3 + 4) / 2 = 42.0", {"route": "math"})
        results = memory.search("What is 12 * (3 + 4) / 2?")
        assert results, "expected the near-identical query to recall the stored entry"
        assert "42.0" in results[0]


def test_relevance_threshold_filters_out_unrelated_queries():
    with tempfile.TemporaryDirectory() as tmp:
        memory = _memory(tmp)
        memory.add(
            "Q: I like structuring my study schedule around 90-minute focus blocks.\nA: Noted.",
            {"route": "general"},
        )
        # An unrelated query (in a totally different domain) should not pull
        # back the study-schedule memory just because it's the "closest"
        # thing in a tiny collection - this is the bug the min_similarity
        # gate exists to prevent.
        results = memory.search("Please delete my account and process a refund immediately.")
        assert results == []


def test_empty_query_returns_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        memory = _memory(tmp)
        memory.add("some content", {})
        assert memory.search("") == []
