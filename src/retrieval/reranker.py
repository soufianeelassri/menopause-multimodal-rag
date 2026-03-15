"""Cross-encoder reranking for retrieved documents.

Uses ms-marco-MiniLM-L-6-v2 to reorder passages by query-document relevance.
Paper Table IV: NDCG=0.847, MRR=0.792, latency=13.2s.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentReranker:
    """Cross-encoder document reranker.

    Scores (query, passage) pairs using a cross-encoder model and
    reorders documents by relevance score.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any = None

    def _load_model(self) -> Any:
        """Load the cross-encoder model on first use.

        Returns:
            Loaded CrossEncoder model.
        """
        if self._model is None:
            model_name = self._settings.reranker_model_name
            logger.info("loading_reranker", model=model_name)
            self._model = CrossEncoder(model_name)
            logger.info("reranker_loaded", model=model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """Rerank documents by cross-encoder relevance score.

        Args:
            query: The search query.
            documents: Documents to rerank.
            top_k: Maximum number of documents to return.
                   Defaults to all documents.

        Returns:
            Documents reordered by relevance, with 'rerank_score' in metadata.
        """
        if not documents:
            return []

        model = self._load_model()

        # Create (query, passage) pairs for scoring
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)

        # Attach scores and sort
        scored_docs: list[tuple[Document, float]] = []
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = round(float(score), 4)
            scored_docs.append((doc, float(score)))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        result = [doc for doc, _ in scored_docs]

        logger.info(
            "reranking_complete",
            query=query[:80],
            input_docs=len(documents),
            output_docs=len(result),
        )

        return result
