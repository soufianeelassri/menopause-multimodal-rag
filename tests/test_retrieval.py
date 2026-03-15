"""Tests for the retrieval pipeline modules."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.retrieval.hybrid import _min_max_normalize


class TestMinMaxNormalize:
    """Tests for score normalization."""

    def test_basic_normalization(self) -> None:
        """Should normalize to [0, 1] range."""
        result = _min_max_normalize([1.0, 2.0, 3.0])
        assert result == [0.0, 0.5, 1.0]

    def test_equal_scores(self) -> None:
        """Equal scores should normalize to zeros."""
        result = _min_max_normalize([5.0, 5.0, 5.0])
        assert result == [0.0, 0.0, 0.0]

    def test_empty_list(self) -> None:
        """Empty input should return empty output."""
        assert _min_max_normalize([]) == []

    def test_single_element(self) -> None:
        """Single element should normalize to zero."""
        assert _min_max_normalize([3.0]) == [0.0]

    def test_negative_scores(self) -> None:
        """Should handle negative scores."""
        result = _min_max_normalize([-1.0, 0.0, 1.0])
        assert result == [0.0, 0.5, 1.0]

    def test_large_range(self) -> None:
        """Should handle large score ranges."""
        result = _min_max_normalize([0.0, 100.0])
        assert result == [0.0, 1.0]


class TestBM25Index:
    """Tests for BM25 sparse index."""

    def test_build_and_search(self) -> None:
        """Should build index and return relevant results."""
        from src.retrieval.bm25_index import BM25Index

        index = BM25Index()
        docs = [
            "Hot flashes are common during menopause",
            "Exercise can help manage weight during menopause",
            "Python is a programming language",
        ]
        index.build_from_documents(docs)

        results = index.search("hot flashes menopause", k=2)
        assert len(results) == 2
        # First result should be the most relevant
        assert results[0][0] == 0  # Index of hot flashes doc

    def test_search_returns_scores(self) -> None:
        """Search results should include scores."""
        from src.retrieval.bm25_index import BM25Index

        index = BM25Index()
        index.build_from_documents(["test document about menopause"])
        results = index.search("menopause")
        assert len(results) > 0
        assert isinstance(results[0][1], float)

    def test_empty_index(self) -> None:
        """Should handle search on unloaded index."""
        from src.retrieval.bm25_index import BM25Index

        index = BM25Index()
        assert index.search("test") == []
        assert not index.is_loaded

    def test_get_document(self) -> None:
        """Should retrieve document by index."""
        from src.retrieval.bm25_index import BM25Index

        index = BM25Index()
        index.build_from_documents(["doc one", "doc two"])
        assert index.get_document(0) == "doc one"
        assert index.get_document(1) == "doc two"
        assert index.get_document(99) == ""  # Out of range


class TestDocumentRepacker:
    """Tests for semantic repacking."""

    def test_repack_small_set(self, sample_documents: list[Document]) -> None:
        """Should handle small document sets."""
        from src.retrieval.repacker import DocumentRepacker

        repacker = DocumentRepacker()

        # With 2 or fewer docs, should return as-is
        result = repacker.repack_by_similarity(sample_documents[:2])
        assert len(result) == 2

    def test_token_limit_repacking(self, sample_documents: list[Document]) -> None:
        """Token-limited repacking should respect budget."""
        from src.retrieval.repacker import DocumentRepacker

        repacker = DocumentRepacker()
        result = repacker.repack_by_token_limit(sample_documents, max_tokens=50)

        # Should return fewer docs than input
        assert len(result) <= len(sample_documents)
        assert len(result) >= 1

    def test_token_limit_empty(self) -> None:
        """Should handle empty document list."""
        from src.retrieval.repacker import DocumentRepacker

        repacker = DocumentRepacker()
        result = repacker.repack_by_token_limit([])
        assert result == []
