import math

import pytest

from src.evaluation.retrieval_metrics import (
    average_precision,
    compute_retrieval_metrics,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.common import RetrievedDoc

RANKED = ["a", "x", "b", "y"]
RELEVANT = {"a", "b"}


def docs(*ids: str) -> list[RetrievedDoc]:
    return [RetrievedDoc(doc_id=d, text=d, score=0.0, rank=r) for r, d in enumerate(ids, start=1)]


def test_metrics_hand_computed() -> None:
    assert recall_at_k(RANKED, RELEVANT, 1) == 0.5
    assert recall_at_k(RANKED, RELEVANT, 3) == 1.0
    assert precision_at_k(RANKED, RELEVANT, 4) == 0.5
    assert mrr_at_k(["x", "a"], RELEVANT, 2) == 0.5
    assert mrr_at_k(["x", "a"], RELEVANT, 1) == 0.0
    assert hit_rate_at_k(RANKED, RELEVANT, 1) == 1.0
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, RELEVANT, 3) == pytest.approx((1 + 1 / math.log2(4)) / idcg)
    assert average_precision(RANKED, RELEVANT) == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_compute_retrieval_metrics_averages_queries() -> None:
    m = compute_retrieval_metrics([docs("a"), docs("x")], [{"a"}, {"a"}], k_values=(1,))
    assert m["recall@1"] == 0.5
    assert m["mrr@1"] == 0.5


class _Fixed:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        return docs(*self.ids[:top_k])


def test_hybrid_rrf_order() -> None:
    # c = 1/63 + 1/61 > b = 1/62 + 1/63 > a = 1/61 > d = 1/62
    hybrid = HybridRetriever(_Fixed(["a", "b", "c"]), _Fixed(["c", "d", "b"]), fusion="rrf")  # type: ignore[arg-type]
    out = hybrid.retrieve("q", top_k=4)
    assert [d.doc_id for d in out] == ["c", "b", "a", "d"]
    assert [d.rank for d in out] == [1, 2, 3, 4]
