"""PDF document parser using the Unstructured library.

Extracts text, tables, and images from PDFs using high-resolution strategy.
Implements SHA-256 hash-based idempotency to skip already-processed documents.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PDFParser:
    """High-resolution PDF element extractor.

    Uses the Unstructured library's partition_pdf with hi_res strategy
    to extract text, tables, and images as separate element streams.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._output_dir = self._settings.processed_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        """Compute SHA-256 hash of a file for idempotency checking.

        Args:
            filepath: Path to the file to hash.

        Returns:
            Hex string of the SHA-256 hash.
        """
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_manifest(self) -> dict[str, str]:
        """Load the processing manifest tracking already-processed PDFs.

        Returns:
            Dictionary mapping file hashes to filenames.
        """
        manifest_path = self._settings.manifest_path
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        """Save the processing manifest to disk.

        Args:
            manifest: Dictionary mapping file hashes to filenames.
        """
        self._settings.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _serialize_metadata(metadata: Any) -> dict[str, Any]:
        """Convert Unstructured metadata to JSON-serializable dictionary.

        Args:
            metadata: Raw metadata from Unstructured elements.

        Returns:
            JSON-serializable dictionary.
        """
        if hasattr(metadata, "to_dict"):
            raw = metadata.to_dict()
        elif isinstance(metadata, dict):
            raw = metadata
        else:
            raw = {"raw": str(metadata)}

        serializable: dict[str, Any] = {}
        for key, value in raw.items():
            try:
                json.dumps(value)
                serializable[key] = value
            except (TypeError, ValueError):
                serializable[key] = str(value)
        return serializable

    def extract_pdf_elements(
        self,
        pdf_path: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract text, table, and image elements from a PDF.

        Uses Unstructured's partition_pdf with hi_res strategy for
        optimal extraction quality.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Tuple of (text_elements, table_elements, image_elements).
            Each element is a dict with 'content', 'type', and 'metadata' keys.
        """
        logger.info("parsing_pdf", filename=pdf_path.name)

        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=1500,
            new_after_n_chars=1000,
            combine_text_under_n_chars=500,
        )

        text_elements: list[dict[str, Any]] = []
        table_elements: list[dict[str, Any]] = []
        image_elements: list[dict[str, Any]] = []

        for element in elements:
            element_type = type(element).__name__
            metadata = self._serialize_metadata(element.metadata)
            metadata["source_file"] = pdf_path.name

            if element_type == "Table":
                table_data = {
                    "content": (
                        element.metadata.text_as_html
                        if hasattr(element.metadata, "text_as_html")
                        and element.metadata.text_as_html
                        else str(element)
                    ),
                    "type": "table",
                    "metadata": metadata,
                }
                table_elements.append(table_data)
            elif element_type == "Image":
                image_data = {
                    "content": (
                        element.metadata.image_base64
                        if hasattr(element.metadata, "image_base64")
                        and element.metadata.image_base64
                        else str(element)
                    ),
                    "type": "image",
                    "metadata": metadata,
                }
                image_elements.append(image_data)
            else:
                text_content = str(element)
                if text_content.strip():
                    text_data = {
                        "content": text_content,
                        "type": "text",
                        "metadata": metadata,
                    }
                    text_elements.append(text_data)

        logger.info(
            "pdf_parsed",
            filename=pdf_path.name,
            texts=len(text_elements),
            tables=len(table_elements),
            images=len(image_elements),
        )

        return text_elements, table_elements, image_elements

    def _save_elements(
        self,
        pdf_path: Path,
        text_elements: list[dict[str, Any]],
        table_elements: list[dict[str, Any]],
        image_elements: list[dict[str, Any]],
    ) -> Path:
        """Save extracted elements to a JSON file for caching.

        Args:
            pdf_path: Original PDF path (used for naming).
            text_elements: Extracted text elements.
            table_elements: Extracted table elements.
            image_elements: Extracted image elements.

        Returns:
            Path to the saved JSON file.
        """
        output_path = self._output_dir / f"{pdf_path.stem}_elements.json"
        data = {
            "source_file": pdf_path.name,
            "text_elements": text_elements,
            "table_elements": table_elements,
            "image_elements": image_elements,
        }
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def load_cached_elements(
        self, pdf_path: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Load previously extracted elements from cache.

        Args:
            pdf_path: Original PDF path to look up cached results.

        Returns:
            Tuple of (text, table, image) elements if cached, None otherwise.
        """
        cache_path = self._output_dir / f"{pdf_path.stem}_elements.json"
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return (
                data["text_elements"],
                data["table_elements"],
                data["image_elements"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("cache_load_failed", path=str(cache_path), error=str(e))
            return None

    def process_pdfs(
        self,
        pdf_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Process all PDFs in a directory with idempotency.

        Skips PDFs whose SHA-256 hash already exists in the manifest.
        Saves extracted elements to data/processed/ for caching.

        Args:
            pdf_dir: Directory containing PDFs. Defaults to settings.raw_pdf_dir.

        Returns:
            List of all processed element sets (text, table, image per PDF).
        """
        pdf_dir = pdf_dir or self._settings.raw_pdf_dir
        manifest = self._load_manifest()
        all_results: list[dict[str, Any]] = []

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        logger.info("processing_pdfs", total=len(pdf_files))

        for pdf_path in pdf_files:
            file_hash = self.compute_file_hash(pdf_path)

            if file_hash in manifest:
                logger.info("skipping_processed", filename=pdf_path.name)
                cached = self.load_cached_elements(pdf_path)
                if cached:
                    text_els, table_els, image_els = cached
                    all_results.append({
                        "source_file": pdf_path.name,
                        "text_elements": text_els,
                        "table_elements": table_els,
                        "image_elements": image_els,
                    })
                continue

            try:
                text_els, table_els, image_els = self.extract_pdf_elements(pdf_path)
                self._save_elements(pdf_path, text_els, table_els, image_els)

                manifest[file_hash] = pdf_path.name
                self._save_manifest(manifest)

                all_results.append({
                    "source_file": pdf_path.name,
                    "text_elements": text_els,
                    "table_elements": table_els,
                    "image_elements": image_els,
                })

            except Exception as e:
                logger.error(
                    "pdf_processing_error",
                    filename=pdf_path.name,
                    error=str(e),
                )

        logger.info("processing_complete", total_processed=len(all_results))
        return all_results
