"""Tests for the generation pipeline modules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.generation.classifier import QueryClass, classify_query
from src.generation.prompts import (
    ANSWER_PROMPT,
    DIRECT_RESPONSE_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
)


class TestQueryClassifier:
    """Tests for query classification."""

    def test_rag_required_classification(self, mock_llm: MagicMock) -> None:
        """Clinical queries should be classified as RAG_REQUIRED."""
        mock_llm.invoke.return_value = MagicMock(content="RAG_REQUIRED")
        result = classify_query("What causes hot flashes?", mock_llm)
        assert result == QueryClass.RAG_REQUIRED

    def test_direct_response_classification(self, mock_llm: MagicMock) -> None:
        """Greetings should be classified as DIRECT_RESPONSE."""
        mock_llm.invoke.return_value = MagicMock(content="DIRECT_RESPONSE")
        result = classify_query("Hello!", mock_llm)
        assert result == QueryClass.DIRECT_RESPONSE

    def test_out_of_scope_classification(self, mock_llm: MagicMock) -> None:
        """Unrelated queries should be classified as OUT_OF_SCOPE."""
        mock_llm.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
        result = classify_query("What is the capital of France?", mock_llm)
        assert result == QueryClass.OUT_OF_SCOPE

    def test_fallback_to_rag(self, mock_llm: MagicMock) -> None:
        """Unparseable responses should default to RAG_REQUIRED."""
        mock_llm.invoke.return_value = MagicMock(content="UNKNOWN_CATEGORY")
        result = classify_query("test query", mock_llm)
        assert result == QueryClass.RAG_REQUIRED

    def test_error_handling(self, mock_llm: MagicMock) -> None:
        """LLM errors should default to RAG_REQUIRED."""
        mock_llm.invoke.side_effect = Exception("API error")
        result = classify_query("test query", mock_llm)
        assert result == QueryClass.RAG_REQUIRED


class TestPrompts:
    """Tests for prompt templates."""

    def test_answer_prompt_variables(self) -> None:
        """RAG answer prompt should have correct input variables."""
        assert set(ANSWER_PROMPT.input_variables) == {
            "context", "chat_history", "question"
        }

    def test_direct_response_prompt_variables(self) -> None:
        """Direct response prompt should have correct input variables."""
        assert set(DIRECT_RESPONSE_PROMPT.input_variables) == {
            "chat_history", "question"
        }

    def test_out_of_scope_response_content(self) -> None:
        """Out-of-scope response should mention menopause capabilities."""
        assert "menopause" in OUT_OF_SCOPE_RESPONSE.lower()
        assert "symptoms" in OUT_OF_SCOPE_RESPONSE.lower()


class TestResponseGenerator:
    """Tests for the response generator."""

    def test_out_of_scope_returns_static(self, mock_llm: MagicMock) -> None:
        """OUT_OF_SCOPE should return the static response."""
        from src.generation.generator import ResponseGenerator

        generator = ResponseGenerator(mock_llm)
        result = generator.generate(
            query="What is 2+2?",
            query_class=QueryClass.OUT_OF_SCOPE,
        )
        assert result == OUT_OF_SCOPE_RESPONSE

    def test_direct_response_not_out_of_scope(self, mock_llm: MagicMock) -> None:
        """DIRECT_RESPONSE should not return the out-of-scope static text."""
        from src.generation.generator import ResponseGenerator

        mock_llm.invoke.return_value = MagicMock(content="Hello! I'm MenoGuide.")
        generator = ResponseGenerator(mock_llm)

        # DIRECT_RESPONSE path invokes the prompt|llm chain
        # With MagicMock, the pipe operator creates a mock chain
        # that returns a mock; verify routing is not OUT_OF_SCOPE
        result = generator.generate(
            query="Hello!",
            query_class=QueryClass.DIRECT_RESPONSE,
        )
        assert result != OUT_OF_SCOPE_RESPONSE

    def test_format_context(self, mock_llm: MagicMock) -> None:
        """Context formatting should include source attribution."""
        from src.generation.generator import ResponseGenerator

        generator = ResponseGenerator(mock_llm)
        docs = [
            Document(
                page_content="Test content",
                metadata={"source_file": "test.pdf", "element_type": "text"},
            )
        ]
        context = generator._format_context(docs)
        assert "test.pdf" in context
        assert "Test content" in context

    def test_format_empty_context(self, mock_llm: MagicMock) -> None:
        """Empty document list should return appropriate message."""
        from src.generation.generator import ResponseGenerator

        generator = ResponseGenerator(mock_llm)
        context = generator._format_context([])
        assert "No relevant context" in context

    def test_history_windowing(self, mock_llm: MagicMock) -> None:
        """History should be capped at max_turns."""
        from src.generation.generator import ResponseGenerator

        generator = ResponseGenerator(mock_llm)

        # Create 20 message pairs (40 messages)
        history = []
        for i in range(20):
            history.append(HumanMessage(content=f"Question {i}"))
            history.append(AIMessage(content=f"Answer {i}"))

        formatted = generator._format_history(history, max_turns=5)
        # History excludes last message (current query), then takes last 5 pairs
        # Messages 0-39, [:-1] = 0-38, [-(10):] = 29-38
        # = A14, Q15, A15, Q16, A16, Q17, A17, Q18, A18, Q19
        assert "Question 19" in formatted  # Last question (but Answer 19 excluded)
        assert "Question 15" in formatted  # Within window
        assert "Question 14" not in formatted  # Outside window
