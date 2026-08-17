"""Dense (embedding-based) retriever backed by FAISS."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.retrieval.base import BaseEmbedder, BaseRetriever
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """FAISS-backed dense retriever.

    Uses a :class:`BaseEmbedder` to encode documents and queries, then
    performs maximum inner-product search via ``faiss.IndexFlatIP``.
    Embeddings are assumed to be L2-normalised so that inner product is
    equivalent to cosine similarity.

    Parameters
    ----------
    embedder : BaseEmbedder
        Embedding model used to encode texts.
    name : str
        Human-readable retriever name (defaults to ``"dense"``).
    embed_batch_size : int
        Number of documents encoded in a single call to the embedder
        (useful for memory management with large corpora).
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        name: str = "dense",
        embed_batch_size: int = 512,
    ) -> None:
        self.name: str = name
        self.embedder = embedder
        self.embed_batch_size = embed_batch_size

        # Populated by build_index / load_index
        self._index: faiss.IndexFlatIP | None = None
        self._doc_ids: list[str] = []
        self._documents: list[str] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Embed the corpus and build a FAISS inner-product index.

        Parameters
        ----------
        doc_ids : list[str]
            Unique identifier for each document.
        documents : list[str]
            Raw document texts, same order as *doc_ids*.
        """
        if len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and documents ({len(documents)}) "
                "must have the same length."
            )

        logger.info("Building dense index [%s] over %d documents ...",
                     self.name, len(documents))

        self._doc_ids = list(doc_ids)
        self._documents = list(documents)

        # Embed in batches to control peak memory.
        with Timer() as t_embed:
            embeddings = self._embed_in_batches(documents)

        logger.info("Embedding completed in %.2f s (%d docs, dim=%d).",
                     t_embed.elapsed, len(documents), embeddings.shape[1])

        # Build FAISS index.
        with Timer() as t_index:
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(embeddings)

        logger.info("FAISS index built in %.2f s (%d vectors).",
                     t_index.elapsed, self._index.ntotal)

    def _embed_in_batches(self, texts: list[str]) -> np.ndarray:
        """Embed *texts* in chunks of ``self.embed_batch_size``."""
        all_embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.embed_batch_size):
            batch = texts[start : start + self.embed_batch_size]
            all_embeddings.append(self.embedder.embed_documents(batch))
        return np.vstack(all_embeddings).astype(np.float32)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Embed *query*, search the FAISS index, and return top-k results.

        Parameters
        ----------
        query : str
            Natural-language query string.
        top_k : int
            Number of documents to return.

        Returns
        -------
        list[RetrievedDoc]
            Results sorted by descending similarity score.
        """
        self._ensure_index()

        query_vec = self.embedder.embed_queries([query]).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results: list[RetrievedDoc] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:
                # FAISS returns -1 when fewer than k results exist.
                break
            results.append(
                RetrievedDoc(
                    doc_id=self._doc_ids[idx],
                    text=self._documents[idx],
                    score=float(score),
                    rank=rank,
                    method=self.name,
                )
            )
        return results

    def retrieve_batch(
        self, queries: list[str], top_k: int = 5
    ) -> list[list[RetrievedDoc]]:
        """Batch retrieval: embed all queries at once, then search.

        Significantly faster than sequential ``retrieve`` calls because
        query embedding and FAISS search are both vectorised.

        Parameters
        ----------
        queries : list[str]
            Batch of query strings.
        top_k : int
            Number of results per query.

        Returns
        -------
        list[list[RetrievedDoc]]
            One result list per query.
        """
        self._ensure_index()
        logger.info("Dense batch retrieval [%s]: %d queries, top_k=%d",
                     self.name, len(queries), top_k)

        with Timer() as t:
            query_vecs = self.embedder.embed_queries(queries).astype(np.float32)
            k = min(top_k, self._index.ntotal)
            all_scores, all_indices = self._index.search(query_vecs, k)

        batch_results: list[list[RetrievedDoc]] = []
        for q_idx in range(len(queries)):
            results: list[RetrievedDoc] = []
            for rank, (score, doc_idx) in enumerate(
                zip(all_scores[q_idx], all_indices[q_idx]), start=1
            ):
                if doc_idx == -1:
                    break
                results.append(
                    RetrievedDoc(
                        doc_id=self._doc_ids[doc_idx],
                        text=self._documents[doc_idx],
                        score=float(score),
                        rank=rank,
                        method=self.name,
                    )
                )
            batch_results.append(results)

        logger.info("Dense batch retrieval finished in %.2f s (avg %.1f ms/query).",
                     t.elapsed, t.elapsed_ms / max(len(queries), 1))
        return batch_results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, directory: str | Path) -> None:
        """Persist the FAISS index and associated metadata to disk.

        Creates three files inside *directory*:
        - ``index.faiss``  -- the FAISS binary index
        - ``doc_ids.json`` -- ordered list of document IDs
        - ``documents.json`` -- ordered list of document texts

        Parameters
        ----------
        directory : str or Path
            Target directory (created if it does not exist).
        """
        self._ensure_index()

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with Timer() as t:
            faiss.write_index(self._index, str(directory / "index.faiss"))
            with open(directory / "doc_ids.json", "w") as f:
                json.dump(self._doc_ids, f)
            with open(directory / "documents.json", "w") as f:
                json.dump(self._documents, f)

        logger.info("Dense index [%s] saved to %s in %.2f s.",
                     self.name, directory, t.elapsed)

    def load_index(self, directory: str | Path) -> None:
        """Load a previously saved FAISS index and metadata.

        Parameters
        ----------
        directory : str or Path
            Directory that contains ``index.faiss``, ``doc_ids.json``,
            and ``documents.json``.
        """
        directory = Path(directory)

        with Timer() as t:
            self._index = faiss.read_index(str(directory / "index.faiss"))
            with open(directory / "doc_ids.json") as f:
                self._doc_ids = json.load(f)
            with open(directory / "documents.json") as f:
                self._documents = json.load(f)

        logger.info(
            "Dense index [%s] loaded from %s in %.2f s (%d vectors, dim=%d).",
            self.name,
            directory,
            t.elapsed,
            self._index.ntotal,
            self._index.d,
        )
