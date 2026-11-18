"""Retrieval evaluation metrics: Recall@k, MRR, nDCG, Precision, MAP, Hit Rate."""

from collections.abc import Sequence

import numpy as np

from src.utils.common import RetrievedDoc


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall@k: fraction of relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = sum(1 for d in top_k if d in relevant_ids)
    return found / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Precision@k: fraction of top-k that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    found = sum(1 for d in top_k if d in relevant_ids)
    return found / len(top_k)


def mrr_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """MRR@k: reciprocal rank of first relevant doc in top-k."""
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """nDCG@k with binary relevance."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)

    ideal_len = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_len))

    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Average Precision for a single query."""
    if not relevant_ids:
        return 0.0

    num_relevant_found = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            num_relevant_found += 1
            sum_precision += num_relevant_found / (i + 1)

    return sum_precision / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Hit Rate@k: 1 if any relevant doc in top-k, else 0."""
    return 1.0 if any(d in relevant_ids for d in retrieved_ids[:k]) else 0.0


def compute_retrieval_metrics(
    all_retrieved: list[list[RetrievedDoc]],
    all_relevant_ids: list[set[str]],
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, float]:
    """Compute all retrieval metrics averaged over queries."""
    n = len(all_retrieved)
    assert n == len(all_relevant_ids), "Mismatch between queries and relevance"

    metrics: dict[str, list[float]] = {}

    for retrieved, relevant in zip(all_retrieved, all_relevant_ids, strict=True):
        ids = [r.doc_id for r in retrieved]
        for k in k_values:
            metrics.setdefault(f"recall@{k}", []).append(recall_at_k(ids, relevant, k))
            metrics.setdefault(f"precision@{k}", []).append(precision_at_k(ids, relevant, k))
            metrics.setdefault(f"mrr@{k}", []).append(mrr_at_k(ids, relevant, k))
            metrics.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(ids, relevant, k))
            metrics.setdefault(f"hit_rate@{k}", []).append(hit_rate_at_k(ids, relevant, k))
        metrics.setdefault("map", []).append(average_precision(ids, relevant))

    return {name: float(np.mean(values)) for name, values in metrics.items()}


def compute_per_query_retrieval(
    retrieved: list[RetrievedDoc],
    relevant_ids: set[str],
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
) -> dict[str, float]:
    """Compute retrieval metrics for a single query."""
    ids = [r.doc_id for r in retrieved]
    result = {}
    for k in k_values:
        result[f"recall@{k}"] = recall_at_k(ids, relevant_ids, k)
        result[f"mrr@{k}"] = mrr_at_k(ids, relevant_ids, k)
        result[f"ndcg@{k}"] = ndcg_at_k(ids, relevant_ids, k)
    result["map"] = average_precision(ids, relevant_ids)
    return result
