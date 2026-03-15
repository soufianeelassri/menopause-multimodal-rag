"""Full ingestion pipeline orchestration.

Orchestrates: parse PDFs → chunk text → caption tables/images → index to
ChromaDB → build BM25 index. Supports idempotent re-execution via manifest.
"""

from __future__ import annotations

import pickle
import uuid
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.config.settings import Settings, get_settings
from src.ingestion.captioner import ImageCaptioner, TableCaptioner
from src.ingestion.chunker import DocumentChunker
from src.ingestion.parser import PDFParser
from src.retrieval.embeddings import EmbeddingModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """End-to-end document ingestion pipeline.

    Processes PDFs through extraction, chunking, captioning, and indexing
    with idempotent execution support.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._parser = PDFParser(self._settings)
        self._chunker: DocumentChunker | None = None
        self._table_captioner: TableCaptioner | None = None
        self._image_captioner: ImageCaptioner | None = None

    def _get_chunker(self) -> DocumentChunker:
        """Lazy-initialize the document chunker."""
        if self._chunker is None:
            self._chunker = DocumentChunker(self._settings)
        return self._chunker

    def _get_table_captioner(self) -> TableCaptioner:
        """Lazy-initialize the table captioner."""
        if self._table_captioner is None:
            self._table_captioner = TableCaptioner(self._settings)
        return self._table_captioner

    def _get_image_captioner(self) -> ImageCaptioner:
        """Lazy-initialize the image captioner."""
        if self._image_captioner is None:
            self._image_captioner = ImageCaptioner(self._settings)
        return self._image_captioner

    def _parse_pdfs(
        self, pdf_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Parse all PDFs with idempotency.

        Args:
            pdf_dir: Directory containing PDFs.

        Returns:
            List of parsed element sets per PDF.
        """
        return self._parser.process_pdfs(pdf_dir)

    def _chunk_texts(
        self, parsed_results: list[dict[str, Any]]
    ) -> list[Document]:
        """Chunk all text elements from parsed PDFs.

        Args:
            parsed_results: Output from _parse_pdfs().

        Returns:
            List of chunked Document objects.
        """
        chunker = self._get_chunker()
        all_chunks: list[Document] = []

        for result in parsed_results:
            text_elements = result.get("text_elements", [])
            chunks = chunker.chunk_elements(text_elements)
            all_chunks.extend(chunks)

        logger.info("total_text_chunks", count=len(all_chunks))
        return all_chunks

    def _caption_tables(
        self, parsed_results: list[dict[str, Any]]
    ) -> list[Document]:
        """Generate summaries for all tables and wrap as Documents.

        Args:
            parsed_results: Output from _parse_pdfs().

        Returns:
            List of Document objects containing table summaries.
        """
        captioner = self._get_table_captioner()
        all_table_docs: list[Document] = []

        for result in parsed_results:
            table_elements = result.get("table_elements", [])
            if not table_elements:
                continue

            summaries = captioner.summarize_tables(table_elements)

            for summary, table_el in zip(summaries, table_elements):
                metadata = table_el.get("metadata", {}).copy()
                metadata["element_type"] = "table_summary"
                metadata["original_content"] = table_el["content"]
                doc = Document(page_content=summary, metadata=metadata)
                all_table_docs.append(doc)

        logger.info("total_table_summaries", count=len(all_table_docs))
        return all_table_docs

    def _caption_images(
        self, parsed_results: list[dict[str, Any]]
    ) -> list[Document]:
        """Generate captions for all images and wrap as Documents.

        Args:
            parsed_results: Output from _parse_pdfs().

        Returns:
            List of Document objects containing image captions.
        """
        captioner = self._get_image_captioner()
        all_image_docs: list[Document] = []

        for result in parsed_results:
            image_elements = result.get("image_elements", [])
            if not image_elements:
                continue

            captions, base64_list = captioner.caption_images(image_elements)

            for caption, b64, img_el in zip(
                captions, base64_list, image_elements
            ):
                metadata = img_el.get("metadata", {}).copy()
                metadata["element_type"] = "image_caption"
                metadata["image_base64"] = b64
                doc = Document(page_content=caption, metadata=metadata)
                all_image_docs.append(doc)

        logger.info("total_image_captions", count=len(all_image_docs))
        return all_image_docs

    def _index_to_vectorstore(
        self, documents: list[Document]
    ) -> None:
        """Index all documents into ChromaDB.

        Args:
            documents: List of Document objects to index.
        """
        if not documents:
            logger.warning("no_documents_to_index")
            return

        embedding_model = EmbeddingModel(self._settings)
        chroma_db_dir = str(self._settings.chroma_db_dir)

        # Filter metadata to ChromaDB-compatible types
        filtered_docs: list[Document] = []
        for doc in documents:
            clean_metadata: dict[str, Any] = {}
            for key, value in doc.metadata.items():
                if key == "image_base64":
                    continue  # Skip large base64 strings
                if key == "original_content":
                    continue  # Skip raw HTML tables
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                elif value is None:
                    clean_metadata[key] = ""
                else:
                    clean_metadata[key] = str(value)

            filtered_docs.append(
                Document(page_content=doc.page_content, metadata=clean_metadata)
            )

        vectorstore = Chroma.from_documents(
            documents=filtered_docs,
            embedding=embedding_model,
            collection_name=self._settings.collection_name,
            persist_directory=chroma_db_dir,
        )

        logger.info(
            "vectorstore_indexed",
            documents=len(filtered_docs),
            persist_dir=chroma_db_dir,
        )

    def _build_bm25_index(self, documents: list[Document]) -> None:
        """Build and serialize BM25 sparse index.

        Args:
            documents: List of Document objects to index.
        """
        if not documents:
            logger.warning("no_documents_for_bm25")
            return

        corpus = [doc.page_content for doc in documents]
        tokenized_corpus = [doc.lower().split() for doc in corpus]

        bm25 = BM25Okapi(
            tokenized_corpus,
            k1=self._settings.bm25_k1,
            b=self._settings.bm25_b,
        )

        index_data = {
            "bm25": bm25,
            "corpus": corpus,
            "tokenized_corpus": tokenized_corpus,
        }

        index_path = self._settings.bm25_index_path
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump(index_data, f)

        logger.info(
            "bm25_index_built",
            documents=len(corpus),
            path=str(index_path),
        )

    def run(
        self,
        pdf_dir: Path | None = None,
        skip_captioning: bool = False,
    ) -> dict[str, int]:
        """Execute the full ingestion pipeline.

        Args:
            pdf_dir: Directory containing PDFs. Defaults to settings.raw_pdf_dir.
            skip_captioning: If True, skip table/image captioning (faster for testing).

        Returns:
            Dictionary with counts of processed elements.
        """
        logger.info("ingestion_pipeline_started")

        # Step 1: Parse PDFs
        logger.info("step_1_parsing_pdfs")
        parsed_results = self._parse_pdfs(pdf_dir)

        # Step 2: Chunk text elements
        logger.info("step_2_chunking_texts")
        text_chunks = self._chunk_texts(parsed_results)

        # Step 3: Caption tables and images
        table_docs: list[Document] = []
        image_docs: list[Document] = []

        if not skip_captioning:
            logger.info("step_3_captioning_tables")
            table_docs = self._caption_tables(parsed_results)

            logger.info("step_4_captioning_images")
            image_docs = self._caption_images(parsed_results)
        else:
            logger.info("skipping_captioning")

        # Step 5: Combine all documents
        all_documents = text_chunks + table_docs + image_docs
        logger.info("total_documents", count=len(all_documents))

        # Step 6: Index to vectorstore
        logger.info("step_5_indexing_vectorstore")
        self._index_to_vectorstore(all_documents)

        # Step 7: Build BM25 index
        logger.info("step_6_building_bm25")
        self._build_bm25_index(all_documents)

        stats = {
            "pdfs_processed": len(parsed_results),
            "text_chunks": len(text_chunks),
            "table_summaries": len(table_docs),
            "image_captions": len(image_docs),
            "total_documents": len(all_documents),
        }

        logger.info("ingestion_pipeline_completed", **stats)
        return stats
