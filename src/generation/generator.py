"""RAG response generation with query classification and streaming.

Orchestrates the full generation flow: classify query → route to
appropriate handler → generate response with optional streaming.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import Settings, get_settings
from src.generation.classifier import QueryClass, classify_query
from src.generation.prompts import (
    ANSWER_PROMPT,
    DIRECT_RESPONSE_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
)
from src.utils.helpers import format_chat_history_for_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_llm(settings: Settings | None = None) -> ChatGoogleGenerativeAI:
    """Create a configured Gemini 2.0 Flash LLM instance.

    Args:
        settings: Application settings. Uses defaults if not provided.

    Returns:
        Configured ChatGoogleGenerativeAI instance.
    """
    settings = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.llm_model_name,
        google_api_key=settings.gemini_api_key,
        temperature=settings.llm_temperature,
    )


class ResponseGenerator:
    """Full RAG generation pipeline with query classification.

    Handles three query types:
    - RAG_REQUIRED: Retrieves context and generates grounded response
    - DIRECT_RESPONSE: Generates conversational response without retrieval
    - OUT_OF_SCOPE: Returns polite redirect to menopause topics

    Args:
        llm: LLM instance for generation.
        settings: Application settings.
    """

    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._settings = settings or get_settings()

    def _format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string.

        Args:
            documents: Retrieved and repacked documents.

        Returns:
            Formatted context string with source attributions.
        """
        if not documents:
            return "No relevant context found in the corpus."

        context_parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source_file", "Unknown source")
            element_type = doc.metadata.get("element_type", "text")
            context_parts.append(
                f"[Source {i} — {source} ({element_type})]:\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    def _format_history(
        self,
        conversation_history: list[HumanMessage | AIMessage],
        max_turns: int = 10,
    ) -> str:
        """Format conversation history with a sliding window.

        Args:
            conversation_history: List of message objects.
            max_turns: Maximum message pairs to include.

        Returns:
            Formatted history string.
        """
        if not conversation_history:
            return "No previous conversation."
        return format_chat_history_for_prompt(
            conversation_history, max_turns=max_turns
        )

    def generate(
        self,
        query: str,
        context_documents: list[Document] | None = None,
        conversation_history: list[HumanMessage | AIMessage] | None = None,
        query_class: QueryClass | None = None,
    ) -> str:
        """Generate a response for the given query.

        Args:
            query: User's input query.
            context_documents: Retrieved documents (for RAG_REQUIRED).
            conversation_history: Previous conversation messages.
            query_class: Pre-computed classification. Auto-classifies if None.

        Returns:
            Generated response string.
        """
        if query_class is None:
            query_class = classify_query(query, self._llm, self._settings)

        history = self._format_history(conversation_history or [])

        if query_class == QueryClass.OUT_OF_SCOPE:
            return OUT_OF_SCOPE_RESPONSE

        if query_class == QueryClass.DIRECT_RESPONSE:
            chain = DIRECT_RESPONSE_PROMPT | self._llm
            result = chain.invoke({
                "question": query,
                "chat_history": history,
            })
            return result.content

        # RAG_REQUIRED
        context = self._format_context(context_documents or [])
        chain = ANSWER_PROMPT | self._llm
        result = chain.invoke({
            "question": query,
            "context": context,
            "chat_history": history,
        })
        return result.content

    def generate_stream(
        self,
        query: str,
        context_documents: list[Document] | None = None,
        conversation_history: list[HumanMessage | AIMessage] | None = None,
        query_class: QueryClass | None = None,
    ) -> Generator[str, None, None]:
        """Generate a streaming response for the given query.

        Args:
            query: User's input query.
            context_documents: Retrieved documents (for RAG_REQUIRED).
            conversation_history: Previous conversation messages.
            query_class: Pre-computed classification. Auto-classifies if None.

        Yields:
            Response text chunks as they are generated.
        """
        if query_class is None:
            query_class = classify_query(query, self._llm, self._settings)

        history = self._format_history(conversation_history or [])

        if query_class == QueryClass.OUT_OF_SCOPE:
            yield OUT_OF_SCOPE_RESPONSE
            return

        if query_class == QueryClass.DIRECT_RESPONSE:
            chain = DIRECT_RESPONSE_PROMPT | self._llm
            for chunk in chain.stream({
                "question": query,
                "chat_history": history,
            }):
                content = (
                    chunk.content
                    if hasattr(chunk, "content")
                    else str(chunk)
                )
                if content:
                    yield content
            return

        # RAG_REQUIRED
        context = self._format_context(context_documents or [])
        chain = ANSWER_PROMPT | self._llm
        for chunk in chain.stream({
            "question": query,
            "context": context,
            "chat_history": history,
        }):
            content = (
                chunk.content
                if hasattr(chunk, "content")
                else str(chunk)
            )
            if content:
                yield content
