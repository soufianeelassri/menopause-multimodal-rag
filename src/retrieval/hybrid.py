"""Hybrid retrieval combining dense and sparse search with score fusion.

Implements the paper's Eq. 1: S_final = alpha * S_dense + (1-alpha) * S_sparse
with min-max normalization and configurable alpha (default 0.7, Table II).

Pipeline: Query -> Dense(top-k) + BM25(top-k) -> Score Fusion -> Return fused results
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config.settings import Settings, get_settings
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import EmbeddingModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] range using min-max scaling.

    Args:
        scores: Raw scores to normalize.

    Returns:
        Normalized scores. Returns zeros if all scores are equal.
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    if score_range == 0:
        return [0.0] * len(scores)

    return [(s - min_score) / score_range for s in scores]


def hybrid_retrieve(
    query: str,
    vectorstore: Chroma,
    bm25_index: BM25Index,
    settings: Settings | None = None,
    k: int | None = None,
) -> list[Document]:
    """Perform hybrid dense+sparse retrieval with alpha-weighted score fusion.

    Implements Eq. 1 from the paper:
        S_final = alpha * S_dense_norm + (1 - alpha) * S_sparse_norm

    Both dense and sparse scores are min-max normalized to [0,1] before fusion.

    Args:
        query: Search query string.
        vectorstore: ChromaDB vector store for dense retrieval.
        bm25_index: BM25 index for sparse retrieval.
        settings: Application settings. Uses defaults if not provided.
        k: Number of results to return. Defaults to settings.retrieval_top_k.

    Returns:
        List of Document objects sorted by fused score descending,
        with 'fused_score', 'dense_score', and 'sparse_score' in metadata.
    """
    settings = settings or get_settings()
    k = k or settings.retrieval_top_k
    alpha = settings.hybrid_alpha

    # Retrieve candidates from both indexes (fetch more than k for fusion)
    fetch_k = k * 5

    # --- Dense retrieval with scores ---
    embedding_model = EmbeddingModel(settings)
    query_embedding = embedding_model.embed_query(query)

    dense_results: list[tuple[Document, float]] = (
        vectorstore.similarity_search_by_vector_with_relevance_scores(
            query_embedding, k=fetch_k
        )
    )

    # --- Sparse retrieval with scores ---
    sparse_results: list[tuple[Document, float]] = []
    if bm25_index.is_loaded:
        sparse_results = bm25_index.search_documents(query, k=fetch_k)

    if not dense_results and not sparse_results:
        logger.warning("no_retrieval_results", query=query[:100])
        return []

    # --- Build score maps keyed by document content ---
    dense_scores: dict[str, float] = {}
    dense_docs: dict[str, Document] = {}
    for doc, score in dense_results:
        content_key = doc.page_content[:200]  # Use prefix as key
        dense_scores[content_key] = score
        dense_docs[content_key] = doc

    sparse_scores: dict[str, float] = {}
    sparse_docs: dict[str, Document] = {}
    for doc, score in sparse_results:
        content_key = doc.page_content[:200]
        sparse_scores[content_key] = score
        sparse_docs[content_key] = doc

    # --- Collect all unique document keys ---
    all_keys = set(dense_scores.keys()) | set(sparse_scores.keys())

    # --- Min-max normalize scores ---
    dense_vals = [dense_scores.get(k, 0.0) for k in all_keys]
    sparse_vals = [sparse_scores.get(k, 0.0) for k in all_keys]

    dense_norm = _min_max_normalize(dense_vals)
    sparse_norm = _min_max_normalize(sparse_vals)

    # --- Fuse scores: S_final = alpha * S_dense + (1 - alpha) * S_sparse ---
    fused_results: list[tuple[str, float, float, float]] = []
    for key, d_norm, s_norm in zip(all_keys, dense_norm, sparse_norm):
        fused_score = alpha * d_norm + (1 - alpha) * s_norm
        fused_results.append((key, fused_score, d_norm, s_norm))

    # --- Sort by fused score descending ---
    fused_results.sort(key=lambda x: x[1], reverse=True)

    # --- Build output documents with scores in metadata ---
    output: list[Document] = []
    for content_key, fused_score, d_score, s_score in fused_results[:k]:
        doc = dense_docs.get(content_key) or sparse_docs.get(content_key)
        if doc is None:
            continue

        doc.metadata["fused_score"] = round(fused_score, 4)
        doc.metadata["dense_score"] = round(d_score, 4)
        doc.metadata["sparse_score"] = round(s_score, 4)
        output.append(doc)

    logger.info(
        "hybrid_retrieval_complete",
        query=query[:80],
        dense_candidates=len(dense_results),
        sparse_candidates=len(sparse_results),
        fused_results=len(output),
        alpha=alpha,
    )

    return output
