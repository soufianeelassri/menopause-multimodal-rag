"""Semantic repacking for diverse context selection.

Uses agglomerative clustering on BGE embeddings to group retrieved passages
by topic, then selects the highest-scoring passage from each cluster.
Paper Table V: Coherence=8.7/10, Diversity=9.2/10.
"""

from __future__ import annotations

import numpy as np
from langchain_core.documents import Document
from sklearn.cluster import AgglomerativeClustering

from src.config.settings import Settings, get_settings
from src.retrieval.embeddings import EmbeddingModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentRepacker:
    """Semantic clustering repacker for diverse context delivery.

    Replaces K-means with agglomerative clustering for automatic
    cluster count determination based on document diversity.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def repack_by_similarity(
        self,
        documents: list[Document],
        distance_threshold: float = 1.5,
    ) -> list[Document]:
        """Repack documents using agglomerative clustering for diversity.

        Groups semantically similar passages, then selects the
        highest-scoring representative from each cluster.

        Args:
            documents: Documents to repack (should have rerank_score in metadata).
            distance_threshold: Clustering distance threshold.
                Higher values = fewer clusters = less diversity.

        Returns:
            Representative documents, one per cluster, sorted by relevance.
        """
        if len(documents) <= 2:
            return documents

        # Embed all documents
        embedding_model = EmbeddingModel(self._settings)
        texts = [doc.page_content for doc in documents]
        embeddings = np.array(embedding_model.embed_documents(texts))

        # Determine cluster count adaptively
        n_docs = len(documents)
        max_clusters = min(n_docs, max(2, n_docs // 2))

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )

        try:
            labels = clustering.fit_predict(embeddings)
        except Exception as e:
            logger.warning("clustering_fallback", error=str(e))
            return documents

        # Group documents by cluster
        clusters: dict[int, list[Document]] = {}
        for doc, label in zip(documents, labels):
            clusters.setdefault(int(label), []).append(doc)

        # Select best document from each cluster (by rerank_score)
        representatives: list[Document] = []
        for cluster_id, cluster_docs in sorted(clusters.items()):
            best_doc = max(
                cluster_docs,
                key=lambda d: d.metadata.get("rerank_score", 0.0),
            )
            best_doc.metadata["cluster_id"] = cluster_id
            best_doc.metadata["cluster_size"] = len(cluster_docs)
            representatives.append(best_doc)

        # Sort representatives by rerank_score
        representatives.sort(
            key=lambda d: d.metadata.get("rerank_score", 0.0),
            reverse=True,
        )

        logger.info(
            "repacking_complete",
            input_docs=len(documents),
            clusters=len(clusters),
            output_docs=len(representatives),
        )

        return representatives

    def repack_by_token_limit(
        self,
        documents: list[Document],
        max_tokens: int | None = None,
    ) -> list[Document]:
        """Group documents until a token limit is reached.

        A simpler alternative to similarity-based repacking that
        greedily selects documents up to the token budget.

        Args:
            documents: Documents to repack.
            max_tokens: Maximum total tokens. Defaults to settings.

        Returns:
            Documents fitting within the token budget.
        """
        max_tokens = max_tokens or self._settings.max_tokens_per_group
        selected: list[Document] = []
        total_tokens = 0

        for doc in documents:
            # Rough token estimate: words * 1.3
            doc_tokens = int(len(doc.page_content.split()) * 1.3)
            if total_tokens + doc_tokens > max_tokens and selected:
                break
            selected.append(doc)
            total_tokens += doc_tokens

        logger.info(
            "token_repacking_complete",
            input_docs=len(documents),
            output_docs=len(selected),
            total_tokens=total_tokens,
        )

        return selected
