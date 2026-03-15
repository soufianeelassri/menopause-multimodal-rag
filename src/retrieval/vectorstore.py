"""ChromaDB vector store management.

Provides factory functions for creating and loading ChromaDB collections
with persistent storage to data/indices/chroma_db/.
"""

from __future__ import annotations

from langchain_chroma import Chroma

from src.config.settings import Settings, get_settings
from src.retrieval.embeddings import EmbeddingModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_vectorstore(settings: Settings | None = None) -> Chroma:
    """Create or load the ChromaDB vector store.

    Args:
        settings: Application settings. Uses defaults if not provided.

    Returns:
        Configured Chroma vector store instance.
    """
    settings = settings or get_settings()
    embedding_model = EmbeddingModel(settings)

    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embedding_model,
        persist_directory=str(settings.chroma_db_dir),
    )

    logger.info(
        "vectorstore_loaded",
        collection=settings.collection_name,
        persist_dir=str(settings.chroma_db_dir),
    )

    return vectorstore
