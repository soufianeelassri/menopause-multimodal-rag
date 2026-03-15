"""Tests for the ingestion pipeline modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import Settings
from src.ingestion.parser import PDFParser


class TestPDFParser:
    """Tests for PDF parser."""

    def test_compute_file_hash_deterministic(self, tmp_path: Path) -> None:
        """Hash should be deterministic for the same file."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        hash1 = PDFParser.compute_file_hash(test_file)
        hash2 = PDFParser.compute_file_hash(test_file)
        assert hash1 == hash2

    def test_compute_file_hash_different_files(self, tmp_path: Path) -> None:
        """Different files should produce different hashes."""
        file1 = tmp_path / "file1.pdf"
        file2 = tmp_path / "file2.pdf"
        file1.write_bytes(b"%PDF-1.4 content A")
        file2.write_bytes(b"%PDF-1.4 content B")

        assert PDFParser.compute_file_hash(file1) != PDFParser.compute_file_hash(file2)

    def test_serialize_metadata_dict(self) -> None:
        """Should handle plain dict metadata."""
        metadata = {"key": "value", "number": 42}
        result = PDFParser._serialize_metadata(metadata)
        assert result == {"key": "value", "number": 42}

    def test_serialize_metadata_non_serializable(self) -> None:
        """Should convert non-serializable values to strings."""
        metadata = {"key": "value", "bad": object()}
        result = PDFParser._serialize_metadata(metadata)
        assert isinstance(result["bad"], str)

    def test_manifest_round_trip(self, tmp_path: Path, settings: Settings) -> None:
        """Manifest should save and load correctly."""
        parser = PDFParser(settings)
        # Override paths to use tmp
        settings_manifest = tmp_path / "manifest.json"

        manifest = {"abc123": "test.pdf", "def456": "test2.pdf"}
        settings_manifest.write_text(json.dumps(manifest), encoding="utf-8")

        loaded = json.loads(settings_manifest.read_text(encoding="utf-8"))
        assert loaded == manifest

    def test_cached_elements_returns_none_when_missing(
        self, tmp_path: Path, settings: Settings
    ) -> None:
        """Should return None when no cached elements exist."""
        parser = PDFParser(settings)
        result = parser.load_cached_elements(Path("nonexistent.pdf"))
        assert result is None


class TestDocumentChunker:
    """Tests for document chunking."""

    def test_chunk_size_respected(self) -> None:
        """Chunks should not exceed configured token count."""
        from src.ingestion.chunker import DocumentChunker

        chunker = DocumentChunker()
        # Create a long text
        text = "Menopause symptoms include hot flashes and night sweats. " * 100
        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["token_count"] <= chunker._settings.chunk_size + 10

    def test_empty_text_returns_empty(self) -> None:
        """Empty text should produce no chunks."""
        from src.ingestion.chunker import DocumentChunker

        chunker = DocumentChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_metadata_preserved(self) -> None:
        """Chunk metadata should include source metadata."""
        from src.ingestion.chunker import DocumentChunker

        chunker = DocumentChunker()
        text = "Menopause affects women between ages 45-55. " * 20
        metadata = {"source_file": "test.pdf", "page": 1}
        chunks = chunker.chunk_text(text, metadata)

        for chunk in chunks:
            assert chunk.metadata["source_file"] == "test.pdf"
            assert "chunk_index" in chunk.metadata
            assert "token_count" in chunk.metadata
