"""Shared pytest fixtures for MenoGuide tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.config.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create test settings with temporary directories."""
    return Settings(
        gemini_api_key="test-key",
        hf_api_token="test-token",
        log_level="DEBUG",
        _env_file=None,
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Hot flashes are the most common symptom of menopause, "
            "affecting up to 80% of women during the menopausal transition.",
            metadata={"source_file": "test1.pdf", "element_type": "text"},
        ),
        Document(
            page_content="Hormone replacement therapy (HRT) can effectively reduce "
            "vasomotor symptoms but carries risks including increased "
            "breast cancer risk with long-term use.",
            metadata={"source_file": "test2.pdf", "element_type": "text"},
        ),
        Document(
            page_content="Regular physical activity has been shown to reduce the "
            "frequency and severity of hot flashes by approximately 50%.",
            metadata={"source_file": "test3.pdf", "element_type": "text"},
        ),
        Document(
            page_content="Cognitive behavioral therapy (CBT) is effective for "
            "managing sleep disturbances during menopause.",
            metadata={"source_file": "test4.pdf", "element_type": "text"},
        ),
        Document(
            page_content="Soy isoflavones may provide modest relief from "
            "menopausal symptoms through phytoestrogen activity.",
            metadata={"source_file": "test5.pdf", "element_type": "text"},
        ),
    ]


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test response")
    return mock
