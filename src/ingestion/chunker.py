"""Semantic text chunking with token-aware splitting.

Implements the paper's chunking strategy: 512-token windows with
64-token overlap using the BGE tokenizer for accurate token counting.
"""

from __future__ import annotations

from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Token-aware document chunker using BGE tokenizer.

    Splits text into chunks of configurable token size with overlap,
    preserving document metadata through the splitting process.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._tokenizer = self._load_tokenizer()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _load_tokenizer(self) -> Any:
        """Load the BGE tokenizer for accurate token counting.

        Returns:
            HuggingFace tokenizer instance.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self._settings.embedding_model_name
        )
        logger.info(
            "tokenizer_loaded",
            model=self._settings.embedding_model_name,
        )
        return tokenizer

    def _token_length(self, text: str) -> int:
        """Count tokens in text using the BGE tokenizer.

        Args:
            text: Input text to tokenize.

        Returns:
            Number of tokens.
        """
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Split a single text string into token-aware chunks.

        Args:
            text: Text content to split.
            metadata: Metadata to attach to each chunk.

        Returns:
            List of Document objects with chunk metadata.
        """
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}
        docs = self._splitter.create_documents(
            texts=[text],
            metadatas=[base_metadata],
        )

        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i
            doc.metadata["token_count"] = self._token_length(doc.page_content)

        return docs

    def chunk_elements(
        self,
        text_elements: list[dict[str, Any]],
    ) -> list[Document]:
        """Split a list of text elements into token-aware chunks.

        Each element is individually chunked, preserving its metadata.

        Args:
            text_elements: List of dicts with 'content' and 'metadata' keys.

        Returns:
            List of Document objects with chunk and source metadata.
        """
        all_chunks: list[Document] = []

        for element in text_elements:
            content = element.get("content", "")
            metadata = element.get("metadata", {}).copy()
            metadata["element_type"] = element.get("type", "text")

            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)

        logger.info(
            "chunking_complete",
            input_elements=len(text_elements),
            output_chunks=len(all_chunks),
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )

        return all_chunks
