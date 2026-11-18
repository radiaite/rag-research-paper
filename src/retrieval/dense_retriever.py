"""Dense (embedding-based) retriever backed by FAISS."""

import json
from pathlib import Path

import faiss
import numpy as np

from src.retrieval.base import BaseEmbedder, BaseRetriever
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """FAISS-backed dense retriever."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        name: str = "dense",
        embed_batch_size: int = 512,
    ) -> None:
        self.name: str = name
        self.embedder = embedder
        self.embed_batch_size = embed_batch_size

        self._index: faiss.Index | None = None
        self.doc_ids: list[str] = []
        self.documents: list[str] = []

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Embed the corpus and build a FAISS inner-product index."""
        if len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and documents ({len(documents)}) "
                "must have the same length."
            )

        logger.info("Building dense index [%s] over %d documents ...", self.name, len(documents))

        self.doc_ids = list(doc_ids)
        self.documents = list(documents)

        with Timer() as t_embed:
            embeddings = self._embed_in_batches(documents)

        logger.info(
            "Embedding completed in %.2f s (%d docs, dim=%d).",
            t_embed.elapsed,
            len(documents),
            embeddings.shape[1],
        )

        with Timer() as t_index:
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(embeddings)

        logger.info(
            "FAISS index built in %.2f s (%d vectors).", t_index.elapsed, self._index.ntotal
        )

    def _embed_in_batches(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* in chunks of ``self.embed_batch_size``."""
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.embed_batch_size):
            batch = texts[start : start + self.embed_batch_size]
            all_embeddings.append(self.embedder.embed_documents(batch))
        return np.vstack(all_embeddings).astype(np.float32)

    def ensure_index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")
        return self._index

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Embed *query*, search the FAISS index, and return top-k results."""
        index = self.ensure_index()

        query_vec = self.embedder.embed_queries([query]).astype(np.float32)
        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vec, k)

        results: list[RetrievedDoc] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=True), start=1):
            if idx == -1:  # FAISS sentinel: fewer than k matches
                break
            results.append(
                RetrievedDoc(
                    doc_id=self.doc_ids[idx],
                    text=self.documents[idx],
                    score=float(score),
                    rank=rank,
                    method=self.name,
                )
            )
        return results

    def retrieve_batch(self, queries: list[str], top_k: int = 5) -> list[list[RetrievedDoc]]:
        """Batch retrieval: embed all queries at once, then search."""
        index = self.ensure_index()
        logger.info(
            "Dense batch retrieval [%s]: %d queries, top_k=%d", self.name, len(queries), top_k
        )

        with Timer() as t:
            query_vecs = self.embedder.embed_queries(queries).astype(np.float32)
            k = min(top_k, index.ntotal)
            all_scores, all_indices = index.search(query_vecs, k)

        batch_results: list[list[RetrievedDoc]] = []
        for q_idx in range(len(queries)):
            results: list[RetrievedDoc] = []
            for rank, (score, doc_idx) in enumerate(
                zip(all_scores[q_idx], all_indices[q_idx], strict=True), start=1
            ):
                if doc_idx == -1:
                    break
                results.append(
                    RetrievedDoc(
                        doc_id=self.doc_ids[doc_idx],
                        text=self.documents[doc_idx],
                        score=float(score),
                        rank=rank,
                        method=self.name,
                    )
                )
            batch_results.append(results)

        logger.info(
            "Dense batch retrieval finished in %.2f s (avg %.1f ms/query).",
            t.elapsed,
            t.elapsed_ms / max(len(queries), 1),
        )
        return batch_results

    def save_index(self, directory: str | Path) -> None:
        """Persist the FAISS index and associated metadata to disk."""
        index = self.ensure_index()

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with Timer() as t:
            faiss.write_index(index, str(directory / "index.faiss"))
            (directory / "doc_ids.json").write_text(json.dumps(self.doc_ids))
            (directory / "documents.json").write_text(json.dumps(self.documents))

        logger.info("Dense index [%s] saved to %s in %.2f s.", self.name, directory, t.elapsed)

    def load_index(self, directory: str | Path) -> None:
        """Load a previously saved FAISS index and metadata."""
        directory = Path(directory)

        with Timer() as t:
            index = self._index = faiss.read_index(str(directory / "index.faiss"))
            self.doc_ids = json.loads((directory / "doc_ids.json").read_text())
            self.documents = json.loads((directory / "documents.json").read_text())

        logger.info(
            "Dense index [%s] loaded from %s in %.2f s (%d vectors, dim=%d).",
            self.name,
            directory,
            t.elapsed,
            index.ntotal,
            index.d,
        )
