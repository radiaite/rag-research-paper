"""BM25 sparse retriever using rank_bm25."""

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.base import BaseRetriever
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    """Okapi BM25 sparse retriever."""

    name: str = "bm25"

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Any | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._tokenize = tokenizer or _tokenize

        self._index: BM25Okapi | None = None
        self.doc_ids: list[str] = []
        self.documents: list[str] = []
        self._tokenized_corpus: list[list[str]] = []

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Tokenize the corpus and build a BM25Okapi index."""
        if len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and documents ({len(documents)}) "
                "must have the same length."
            )

        logger.info(
            "Building BM25 index over %d documents (k1=%.2f, b=%.2f) ...",
            len(documents),
            self.k1,
            self.b,
        )

        with Timer() as t:
            self.doc_ids = list(doc_ids)
            self.documents = list(documents)
            self._tokenized_corpus = [self._tokenize(doc) for doc in documents]
            self._index = BM25Okapi(
                self._tokenized_corpus,
                k1=self.k1,
                b=self.b,
            )

        logger.info("BM25 index built in %.2f s (%d documents).", t.elapsed, len(documents))

    def ensure_index(self) -> BM25Okapi:
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")
        return self._index

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Score every document against *query* and return the top-k results."""
        index = self.ensure_index()

        tokenized_query = self._tokenize(query)
        scores: np.ndarray = index.get_scores(tokenized_query)

        k = min(top_k, len(self.doc_ids))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[RetrievedDoc] = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                RetrievedDoc(
                    doc_id=self.doc_ids[idx],
                    text=self.documents[idx],
                    score=float(scores[idx]),
                    rank=rank,
                    method=self.name,
                )
            )
        return results

    def retrieve_batch(self, queries: list[str], top_k: int = 5) -> list[list[RetrievedDoc]]:
        """Retrieve for a batch of queries sequentially."""
        self.ensure_index()
        logger.info("BM25 batch retrieval: %d queries, top_k=%d", len(queries), top_k)

        with Timer() as t:
            results = [self.retrieve(q, top_k) for q in queries]

        logger.info(
            "BM25 batch retrieval finished in %.2f s (avg %.1f ms/query).",
            t.elapsed,
            t.elapsed_ms / max(len(queries), 1),
        )
        return results
