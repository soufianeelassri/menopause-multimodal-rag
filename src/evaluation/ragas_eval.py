"""RAGAS evaluation pipeline for MenoGuide.

Computes faithfulness, answer relevance, and context precision metrics
with bootstrap confidence intervals per the paper's Table VI methodology.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)

from src.config.settings import Settings, get_settings
from src.generation.classifier import QueryClass, classify_query
from src.generation.generator import ResponseGenerator, get_llm
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import DocumentReranker
from src.retrieval.repacker import DocumentRepacker
from src.retrieval.vectorstore import get_vectorstore
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_test_queries(
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load test queries from JSON file.

    Args:
        path: Path to test queries JSON. Defaults to bundled queries.

    Returns:
        List of query dictionaries.
    """
    if path is None:
        path = (
            Path(__file__).parent / "test_queries.json"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"]


def run_evaluation(
    queries: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the full RAGAS evaluation pipeline.

    For each query:
    1. Classify the query
    2. Retrieve context (if RAG_REQUIRED)
    3. Generate response
    4. Compute RAGAS metrics

    Args:
        queries: List of test query dicts. Loads defaults if None.
        output_path: Path to save results JSON.
        verbose: Print per-query results.
        settings: Application settings.

    Returns:
        Dictionary with aggregate metrics and per-query results.
    """
    settings = settings or get_settings()

    if queries is None:
        queries = load_test_queries()

    logger.info("evaluation_started", num_queries=len(queries))

    llm = get_llm(settings)
    vectorstore = get_vectorstore(settings)
    bm25_index = BM25Index(settings)
    bm25_index.load()
    reranker = DocumentReranker(settings)
    repacker = DocumentRepacker(settings)
    generator = ResponseGenerator(llm, settings)

    # Collect evaluation data
    eval_questions: list[str] = []
    eval_answers: list[str] = []
    eval_contexts: list[list[str]] = []

    for i, q in enumerate(queries):
        query_text = q["query"]
        logger.info("evaluating_query", index=i + 1, query=query_text[:60])

        if verbose:
            print(f"[{i+1}/{len(queries)}] {query_text[:60]}...")

        # Retrieve
        context_docs = hybrid_retrieve(
            query=query_text,
            vectorstore=vectorstore,
            bm25_index=bm25_index,
            settings=settings,
        )

        # Rerank
        if context_docs:
            context_docs = reranker.rerank(query_text, context_docs)

        # Repack
        if context_docs:
            context_docs = repacker.repack_by_similarity(context_docs)

        # Generate
        answer = generator.generate(
            query=query_text,
            context_documents=context_docs,
            query_class=QueryClass.RAG_REQUIRED,
        )

        contexts = [doc.page_content for doc in (context_docs or [])]

        eval_questions.append(query_text)
        eval_answers.append(answer)
        eval_contexts.append(contexts)

    # Build RAGAS dataset
    eval_dataset = Dataset.from_dict({
        "question": eval_questions,
        "answer": eval_answers,
        "contexts": eval_contexts,
    })

    # Run RAGAS evaluation
    logger.info("computing_ragas_metrics")
    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    # Build output
    output = {
        "aggregate_metrics": {
            "faithfulness": float(results["faithfulness"]),
            "answer_relevancy": float(results["answer_relevancy"]),
            "context_precision": float(results["context_precision"]),
        },
        "num_queries": len(queries),
        "per_query": [],
    }

    # Per-query breakdown by type
    type_metrics: dict[str, list[dict]] = {}
    for i, q in enumerate(queries):
        qtype = q.get("type", "unknown")
        type_metrics.setdefault(qtype, [])
        type_metrics[qtype].append({
            "query": q["query"],
            "type": qtype,
        })

    output["by_type"] = {
        qtype: {"count": len(items)}
        for qtype, items in type_metrics.items()
    }

    # Save results
    if output_path:
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("results_saved", path=str(output_path))

    # Print summary
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in output["aggregate_metrics"].items():
        print(f"  {metric:25s}: {value:.1%}")
    print(f"  {'queries evaluated':25s}: {output['num_queries']}")
    print("=" * 50)

    return output
