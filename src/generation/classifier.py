"""LLM-based query classifier for intelligent routing.

Routes queries to RAG pipeline, direct generation, or out-of-scope handling.
Paper Section III-C: 96% routing accuracy on 50-query evaluation set.
"""

from __future__ import annotations

from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QueryClass(str, Enum):
    """Query classification categories.

    RAG_REQUIRED: Clinical, factual, symptom-specific menopause queries
        that need retrieval from the corpus.
    DIRECT_RESPONSE: Greetings, meta-questions, or general menopause
        queries answerable without retrieval.
    OUT_OF_SCOPE: Queries unrelated to menopause or women's health.
    """

    RAG_REQUIRED = "rag_required"
    DIRECT_RESPONSE = "direct_response"
    OUT_OF_SCOPE = "out_of_scope"


CLASSIFICATION_PROMPT = """You are a query router for a menopause health information system \
powered by peer-reviewed PLOS ONE articles.

Classify the following user query into exactly one category:

1. RAG_REQUIRED: The query is factual, clinical, or symptom-specific about menopause \
and requires evidence from peer-reviewed sources to answer accurately.
   Examples: "What causes hot flashes?", "Is HRT safe for women with a history of \
breast cancer?", "What are the symptoms of perimenopause?", "How does menopause \
affect cardiovascular risk?"

2. DIRECT_RESPONSE: The query is a greeting, meta-question about the system, \
or a general menopause question that can be answered from general knowledge \
without specific peer-reviewed evidence.
   Examples: "Hello", "How do you work?", "Thanks for the help", "What can you help me with?"

3. OUT_OF_SCOPE: The query is unrelated to menopause, women's health, or the \
system's capabilities.
   Examples: "What's the capital of France?", "Help me write Python code", \
"What's the weather today?"

Query: "{query}"

Respond with ONLY the category name: RAG_REQUIRED, DIRECT_RESPONSE, or OUT_OF_SCOPE."""


def classify_query(
    query: str,
    llm: ChatGoogleGenerativeAI,
    settings: Settings | None = None,
) -> QueryClass:
    """Classify a user query for routing.

    Args:
        query: The user's input query.
        llm: LLM instance for classification.
        settings: Application settings.

    Returns:
        QueryClass enum value indicating the routing decision.
    """
    prompt = CLASSIFICATION_PROMPT.format(query=query)

    try:
        response = llm.invoke(prompt)
        result = response.content.strip().upper()

        if "RAG_REQUIRED" in result:
            classification = QueryClass.RAG_REQUIRED
        elif "DIRECT" in result:
            classification = QueryClass.DIRECT_RESPONSE
        elif "OUT_OF_SCOPE" in result or "OUT" in result:
            classification = QueryClass.OUT_OF_SCOPE
        else:
            # Default to RAG_REQUIRED for safety (avoids missing retrieval)
            classification = QueryClass.RAG_REQUIRED
            logger.warning(
                "classification_fallback",
                raw_response=result[:50],
                defaulted_to="RAG_REQUIRED",
            )

        logger.info(
            "query_classified",
            query=query[:80],
            classification=classification.value,
        )
        return classification

    except Exception as e:
        logger.error("classification_error", error=str(e))
        return QueryClass.RAG_REQUIRED
