# metrics.py
# Description: Retrieval quality metrics for the RAG evaluation harness.
#
# Ported from tldw_server2 rag_service/retrieval_metrics.py (P1, RAG server-port programme).
"""Retrieval quality metrics for the RAG pipeline.

Pure stateless functions for computing standard information retrieval metrics:
- Precision@K: Proportion of retrieved documents that are relevant
- Recall@K: Proportion of relevant documents that were retrieved
- MRR: Mean Reciprocal Rank - position of first relevant result
- NDCG@K: Normalized Discounted Cumulative Gain - ranking quality

All functions are pure (no side effects) and require no LLM calls.
Ported from RAGnarok-AI's evaluator pattern, adapted for tldw_server2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate precision at K.

    Precision@K measures the proportion of retrieved documents (in top K)
    that are relevant.

    Args:
        retrieved_ids: List of retrieved document IDs, ordered by relevance.
        relevant_ids: List of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0

    relevant_set = set(relevant_ids)
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)

    return relevant_retrieved / len(retrieved_at_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate recall at K.

    Recall@K measures the proportion of relevant documents that were
    retrieved in the top K results.

    Args:
        retrieved_ids: List of retrieved document IDs, ordered by relevance.
        relevant_ids: List of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    relevant_retrieved = len(retrieved_at_k & relevant_set)

    return relevant_retrieved / len(relevant_set)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Calculate Mean Reciprocal Rank.

    MRR measures the position of the first relevant document in the
    retrieved results. Returns 1/rank of the first relevant result.

    Args:
        retrieved_ids: List of retrieved document IDs, ordered by relevance.
        relevant_ids: List of ground truth relevant document IDs.

    Returns:
        MRR score between 0.0 and 1.0. Returns 0.0 if no relevant doc found.
    """
    if not retrieved_ids or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.

    NDCG@K measures the quality of the ranking, giving higher scores
    when relevant documents appear earlier in the results.
    Uses binary relevance (1 if relevant, 0 if not).

    Args:
        retrieved_ids: List of retrieved document IDs, ordered by relevance.
        relevant_ids: List of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    retrieved_at_k = retrieved_ids[:k]

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_at_k, start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    # Calculate IDCG (Ideal DCG) - best possible ranking
    num_relevant_at_k = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_relevant_at_k + 1))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def f1_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate F1 score at K.

    Harmonic mean of precision@K and recall@K.

    Args:
        retrieved_ids: List of retrieved document IDs, ordered by relevance.
        relevant_ids: List of ground truth relevant document IDs.
        k: Number of top results to consider.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)

    if p + r == 0:
        return 0.0

    return 2 * (p * r) / (p + r)


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregated retrieval evaluation metrics.

    Attributes:
        precision: Precision@K score.
        recall: Recall@K score.
        mrr: Mean Reciprocal Rank score.
        ndcg: NDCG@K score.
        f1: F1@K score.
        k: The K value used for @K metrics.
    """

    precision: float
    recall: float
    mrr: float
    ndcg: float
    f1: float
    k: int

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary suitable for metadata/JSON."""
        return {
            "precision_at_k": self.precision,
            "recall_at_k": self.recall,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg,
            "f1_at_k": self.f1,
            "k": self.k,
        }


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 10,
) -> RetrievalMetrics:
    """Evaluate retrieval quality for a single query result.

    Computes all retrieval metrics (precision, recall, MRR, NDCG, F1) for
    the given document IDs against ground truth.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevant_ids: List of ground truth relevant document IDs.
        k: Number of top results to consider for @K metrics. Defaults to 10.

    Returns:
        RetrievalMetrics containing all computed metrics.

    Raises:
        ValueError: If k is less than 1.
    """
    if k < 1:
        msg = f"k must be at least 1, got {k}"
        raise ValueError(msg)

    return RetrievalMetrics(
        precision=precision_at_k(retrieved_ids, relevant_ids, k),
        recall=recall_at_k(retrieved_ids, relevant_ids, k),
        mrr=mrr(retrieved_ids, relevant_ids),
        ndcg=ndcg_at_k(retrieved_ids, relevant_ids, k),
        f1=f1_at_k(retrieved_ids, relevant_ids, k),
        k=k,
    )


def evaluate_retrieval_batch(
    results: list[tuple[list[str], list[str]]],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate retrieval quality across multiple queries.

    Computes averaged metrics across a batch of query results.

    Args:
        results: List of (retrieved_ids, relevant_ids) tuples.
        k: Number of top results to consider.

    Returns:
        Dictionary with averaged metric values, keyed by the
        `RetrievalMetrics` field names (``precision``, ``recall``, ``mrr``,
        ``ndcg``, ``f1``) plus ``num_queries`` and ``k``.

        Deviation from the ported server module: the server's
        `evaluate_retrieval_batch` returns `avg_`-prefixed, `_at_k`-suffixed
        keys (e.g. ``avg_precision_at_k``). This port uses the shorter
        `RetrievalMetrics` field names instead, per the harness's own
        known-answer test (Task 2 of the RAG-port P1 plan).
    """
    if not results:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "f1": 0.0,
            "num_queries": 0,
            "k": k,
        }

    metrics_list = [
        evaluate_retrieval(retrieved, relevant, k)
        for retrieved, relevant in results
    ]

    n = len(metrics_list)
    return {
        "precision": sum(m.precision for m in metrics_list) / n,
        "recall": sum(m.recall for m in metrics_list) / n,
        "mrr": sum(m.mrr for m in metrics_list) / n,
        "ndcg": sum(m.ndcg for m in metrics_list) / n,
        "f1": sum(m.f1 for m in metrics_list) / n,
        "num_queries": n,
        "k": k,
    }
