"""Hybrid retriever that fuses BM25 and dense retrieval results."""

from collections import defaultdict
from typing import Any, Literal

import numpy as np

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)

FusionMethod = Literal["rrf", "cc", "dbsf"]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever that fuses sparse and dense results."""

    name: str = "hybrid"

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        fusion: FusionMethod = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.5,
        candidate_k_multiplier: int = 3,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.fusion: FusionMethod = fusion
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.candidate_k_multiplier = candidate_k_multiplier

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Build indices for both the BM25 and dense retrievers."""
        logger.info("Building hybrid index (%s fusion) ...", self.fusion)
        self.bm25_retriever.build_index(doc_ids, documents)
        self.dense_retriever.build_index(doc_ids, documents)
        logger.info("Hybrid index construction complete.")

    @staticmethod
    def _collect_candidates(
        bm25_results: list[RetrievedDoc],
        dense_results: list[RetrievedDoc],
    ) -> dict[str, dict[str, Any]]:
        """Merge candidate lists, keeping the best text and per-retriever info."""
        candidates: dict[str, dict[str, Any]] = {}

        for doc in bm25_results:
            candidates[doc.doc_id] = {
                "text": doc.text,
                "bm25_score": doc.score,
                "bm25_rank": doc.rank,
                "dense_score": 0.0,
                "dense_rank": None,
            }

        for doc in dense_results:
            if doc.doc_id in candidates:
                candidates[doc.doc_id]["dense_score"] = doc.score
                candidates[doc.doc_id]["dense_rank"] = doc.rank
            else:
                candidates[doc.doc_id] = {
                    "text": doc.text,
                    "bm25_score": 0.0,
                    "bm25_rank": None,
                    "dense_score": doc.score,
                    "dense_rank": doc.rank,
                }

        return candidates

    def _fuse_rrf(
        self,
        bm25_results: list[RetrievedDoc],
        dense_results: list[RetrievedDoc],
    ) -> dict[str, float]:
        """Reciprocal Rank Fusion."""
        scores: dict[str, float] = defaultdict(float)

        for doc in bm25_results:
            scores[doc.doc_id] += 1.0 / (self.rrf_k + doc.rank)

        for doc in dense_results:
            scores[doc.doc_id] += 1.0 / (self.rrf_k + doc.rank)

        return dict(scores)

    @staticmethod
    def _min_max_normalize(values: list[float]) -> list[float]:
        """Scale *values* to [0, 1] via min-max normalization."""
        if not values:
            return values
        lo, hi = min(values), max(values)
        span = hi - lo
        if span == 0.0:
            return [0.5] * len(values)
        return [(v - lo) / span for v in values]

    def _fuse_cc(self, candidates: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Convex Combination with min-max normalised scores."""
        doc_ids = list(candidates.keys())
        bm25_raw = [candidates[did]["bm25_score"] for did in doc_ids]
        dense_raw = [candidates[did]["dense_score"] for did in doc_ids]

        bm25_norm = self._min_max_normalize(bm25_raw)
        dense_norm = self._min_max_normalize(dense_raw)

        return {
            did: self.alpha * dn + (1.0 - self.alpha) * bn
            for did, bn, dn in zip(doc_ids, bm25_norm, dense_norm, strict=True)
        }

    @staticmethod
    def _z_score_normalize(values: list[float]) -> list[float]:
        """Standardise *values* to zero mean and unit variance."""
        if not values:
            return values
        arr = np.array(values, dtype=np.float64)
        std = float(arr.std())
        if std == 0.0:
            return [0.0] * len(values)
        mean = float(arr.mean())
        return [float((v - mean) / std) for v in values]

    def _fuse_dbsf(self, candidates: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Distribution-Based Score Fusion."""
        doc_ids = list(candidates.keys())
        bm25_raw = [candidates[did]["bm25_score"] for did in doc_ids]
        dense_raw = [candidates[did]["dense_score"] for did in doc_ids]

        bm25_z = self._z_score_normalize(bm25_raw)
        dense_z = self._z_score_normalize(dense_raw)

        return {
            did: self.alpha * dz + (1.0 - self.alpha) * bz
            for did, bz, dz in zip(doc_ids, bm25_z, dense_z, strict=True)
        }

    def _fuse(
        self,
        bm25_results: list[RetrievedDoc],
        dense_results: list[RetrievedDoc],
    ) -> dict[str, float]:
        """Dispatch to the configured fusion strategy."""
        candidates = self._collect_candidates(bm25_results, dense_results)

        if self.fusion == "rrf":
            return self._fuse_rrf(bm25_results, dense_results)
        elif self.fusion == "cc":
            return self._fuse_cc(candidates)
        elif self.fusion == "dbsf":
            return self._fuse_dbsf(candidates)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion!r}")

    def _build_results(
        self,
        fused_scores: dict[str, float],
        candidates: dict[str, dict[str, Any]],
        top_k: int,
    ) -> list[RetrievedDoc]:
        """Sort candidates by fused score and return top-k ``RetrievedDoc``s."""
        sorted_ids = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)[:top_k]

        results: list[RetrievedDoc] = []
        for rank, doc_id in enumerate(sorted_ids, start=1):
            info = candidates[doc_id]
            results.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    text=info["text"],
                    score=fused_scores[doc_id],
                    rank=rank,
                    method=f"hybrid_{self.fusion}",
                    metadata={
                        "bm25_score": info["bm25_score"],
                        "dense_score": info["dense_score"],
                        "bm25_rank": info["bm25_rank"],
                        "dense_rank": info["dense_rank"],
                    },
                )
            )
        return results

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Retrieve from both sub-retrievers, fuse, and return top-k."""
        candidate_k = top_k * self.candidate_k_multiplier

        with Timer() as t:
            bm25_results = self.bm25_retriever.retrieve(query, top_k=candidate_k)
            dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k)

            candidates = self._collect_candidates(bm25_results, dense_results)
            fused_scores = self._fuse(bm25_results, dense_results)

        results = self._build_results(fused_scores, candidates, top_k)

        logger.debug(
            "Hybrid [%s] retrieve: %d candidates fused in %.1f ms",
            self.fusion,
            len(candidates),
            t.elapsed_ms,
        )
        return results

    def retrieve_batch(self, queries: list[str], top_k: int = 5) -> list[list[RetrievedDoc]]:
        """Batch hybrid retrieval."""
        candidate_k = top_k * self.candidate_k_multiplier
        logger.info(
            "Hybrid batch retrieval [%s]: %d queries, top_k=%d", self.fusion, len(queries), top_k
        )

        with Timer() as t:
            bm25_batch = self.bm25_retriever.retrieve_batch(queries, top_k=candidate_k)
            dense_batch = self.dense_retriever.retrieve_batch(queries, top_k=candidate_k)

            batch_results: list[list[RetrievedDoc]] = []
            for bm25_results, dense_results in zip(bm25_batch, dense_batch, strict=True):
                candidates = self._collect_candidates(bm25_results, dense_results)
                fused_scores = self._fuse(bm25_results, dense_results)
                batch_results.append(self._build_results(fused_scores, candidates, top_k))

        logger.info(
            "Hybrid batch retrieval finished in %.2f s (avg %.1f ms/query).",
            t.elapsed,
            t.elapsed_ms / max(len(queries), 1),
        )
        return batch_results
