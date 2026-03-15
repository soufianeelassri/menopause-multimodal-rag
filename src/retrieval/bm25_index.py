"""BM25 sparse retrieval index.

Loads a pre-built BM25 index from pickle for fast startup.
Paper Eq. 2: BM25 with k1=1.5, b=0.75.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BM25Index:
    """BM25Okapi sparse retrieval index.

    Loads from a serialized pickle file built during ingestion.
    Eliminates the startup delay of initializing BM25 from ChromaDB.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._bm25: Any = None
        self._corpus: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._loaded = False

    def load(self, index_path: Path | None = None) -> bool:
        """Load the BM25 index from pickle.

        Args:
            index_path: Path to the pickle file. Defaults to settings.

        Returns:
            True if loaded successfully, False otherwise.
        """
        path = index_path or self._settings.bm25_index_path

        if not path.exists():
            logger.warning("bm25_index_not_found", path=str(path))
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self._bm25 = data["bm25"]
            self._corpus = data["corpus"]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._loaded = True

            logger.info(
                "bm25_index_loaded",
                documents=len(self._corpus),
                path=str(path),
            )
            return True

        except Exception as e:
            logger.error("bm25_load_error", error=str(e))
            return False

    def build_from_documents(self, documents: list[str]) -> None:
        """Build a BM25 index from a list of document texts.

        Used as fallback when no pickle file exists.

        Args:
            documents: List of document text strings.
        """
        self._corpus = documents
        self._tokenized_corpus = [doc.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self._settings.bm25_k1,
            b=self._settings.bm25_b,
        )
        self._loaded = True

        logger.info("bm25_index_built", documents=len(documents))

    @property
    def is_loaded(self) -> bool:
        """Whether the BM25 index is loaded and ready."""
        return self._loaded

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search the BM25 index for relevant documents.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of (document_index, score) tuples sorted by score descending.
        """
        if not self._loaded or self._bm25 is None:
            logger.warning("bm25_not_loaded")
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        top_k = min(k, len(scores))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [(idx, float(scores[idx])) for idx in top_indices]

    def get_document(self, index: int) -> str:
        """Retrieve a document by its index.

        Args:
            index: Document index in the corpus.

        Returns:
            Document text content.
        """
        if 0 <= index < len(self._corpus):
            return self._corpus[index]
        return ""

    def search_documents(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Search and return Document objects with scores.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples.
        """
        results = self.search(query, k)
        return [
            (Document(page_content=self.get_document(idx)), score)
            for idx, score in results
        ]
