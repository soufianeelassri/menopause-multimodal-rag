"""BGE-large-en-v1.5 embedding wrapper for dense retrieval.

Provides LangChain-compatible embedding interface with lazy model loading
and GPU support detection. Paper Table I: 92.3% precision, d=1024.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingModel(Embeddings):
    """Sentence-transformers embedding wrapper for BGE-large-en-v1.5.

    Implements the LangChain Embeddings interface with lazy initialization
    to avoid loading the model until it's actually needed.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any = None

    def _load_model(self) -> Any:
        """Load the sentence-transformers model on first use.

        Returns:
            Loaded SentenceTransformer model.
        """
        if self._model is None:
            model_name = self._settings.embedding_model_name
            logger.info("loading_embedding_model", model=model_name)

            self._model = SentenceTransformer(model_name)
            logger.info(
                "embedding_model_loaded",
                model=model_name,
                device=str(self._model.device),
                dimension=self._model.get_sentence_embedding_dimension(),
            )
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Query string to embed.

        Returns:
            List of float values representing the embedding vector.
        """
        model = self._load_model()
        embedding = model.encode(
            text, normalize_embeddings=True
        )
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors.
        """
        model = self._load_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
            batch_size=32,
        )
        return embeddings.tolist()
