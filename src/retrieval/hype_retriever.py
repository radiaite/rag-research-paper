"""HyPE (Hypothetical Prompt Embeddings) retriever.

Instead of embedding documents directly, this retriever generates synthetic
questions for each document chunk at indexing time, embeds those questions,
and stores them in a FAISS index mapped back to the originating document.
At query time the real query is embedded and matched against the synthetic
question embeddings, which often yields better alignment than document
embeddings.

Reference: Inspired by the HyDE inversion -- embed the *questions* a
document could answer rather than the document itself.
"""

from __future__ import annotations

import os
from typing import Any

import faiss
import numpy as np
from openai import AzureOpenAI

from src.retrieval.base import BaseEmbedder, BaseRetriever
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)

DEFAULT_HYPE_PROMPT = (
    "Given the following text passage, generate {n} diverse questions that "
    "this passage could answer. Return each question on its own line, "
    "numbered (e.g. 1. ... 2. ...). Do not include any other text.\n\n"
    "Passage:\n{text}\n\n"
    "Questions:"
)


class HyPERetriever(BaseRetriever):
    """Hypothetical Prompt Embeddings retriever.

    At index time, for every document chunk an LLM generates N synthetic
    questions that the chunk could answer.  Those questions are embedded
    and stored in a FAISS inner-product index.  Each entry in the index
    maps back to the original document.

    At query time the real user query is embedded normally and matched
    against the synthetic-question embeddings.  Results are deduplicated
    by ``doc_id``, keeping the best score per document.

    Parameters
    ----------
    embedder : BaseEmbedder
        Embedding model for synthetic questions and queries.
    llm_model : str
        OpenAI chat model used for synthetic question generation.
    num_queries_per_chunk : int
        Number of synthetic questions to generate for each document.
    prompt_template : str
        Format string with ``{text}`` and ``{n}`` placeholders.
    temperature : float
        Sampling temperature for the LLM.
    max_tokens : int
        Maximum tokens for each LLM response.
    openai_kwargs : dict
        Extra keyword arguments forwarded to ``OpenAI()``.
    """

    name: str = "hype"

    def __init__(
        self,
        embedder: BaseEmbedder,
        llm_model: str = "gpt-4.1-mini",
        num_queries_per_chunk: int = 5,
        prompt_template: str = DEFAULT_HYPE_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 512,
        openai_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.embedder = embedder
        self.llm_model = llm_model
        self.num_queries_per_chunk = max(1, num_queries_per_chunk)
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
        )

        # Populated by build_index
        self._index: faiss.IndexFlatIP | None = None
        self._index_doc_ids: list[str] = []       # one entry per FAISS vector
        self._doc_id_to_text: dict[str, str] = {}  # original doc texts
        self._doc_ids: list[str] = []              # unique, ordered

    # ------------------------------------------------------------------
    # Synthetic question generation
    # ------------------------------------------------------------------

    def _generate_synthetic_queries(self, text: str, n: int) -> list[str]:
        """Generate *n* synthetic questions for *text* using the LLM.

        Parameters
        ----------
        text : str
            The document passage.
        n : int
            Number of questions to request.

        Returns
        -------
        list[str]
            Parsed questions (may be fewer than *n* if parsing fails for
            some lines).
        """
        prompt = self.prompt_template.format(text=text, n=n)

        try:
            response = self._client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content or ""
        except Exception:
            logger.exception("HyPE: LLM call failed for synthetic query generation.")
            return []

        # Parse numbered lines like "1. What is ..." or plain lines.
        questions: list[str] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading numbering (e.g. "1. ", "1) ", "- ").
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned:
                questions.append(cleaned)

        return questions[:n]

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Generate synthetic queries for each document, embed them, and build index.

        For each document, N synthetic questions are generated and embedded.
        The FAISS index stores these question embeddings, with a parallel
        list tracking which ``doc_id`` each embedding belongs to.

        Parameters
        ----------
        doc_ids : list[str]
            Unique identifier for each document.
        documents : list[str]
            Raw document texts.
        """
        if len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and documents ({len(documents)}) "
                "must have the same length."
            )

        logger.info(
            "HyPE: building index over %d documents "
            "(%d synthetic queries each) ...",
            len(documents), self.num_queries_per_chunk,
        )

        self._doc_ids = list(doc_ids)
        self._doc_id_to_text = dict(zip(doc_ids, documents))

        # Step 1: generate synthetic questions for every document.
        all_synth_queries: list[str] = []
        self._index_doc_ids = []

        with Timer() as t_gen:
            for i, (did, doc) in enumerate(zip(doc_ids, documents)):
                queries = self._generate_synthetic_queries(
                    doc, self.num_queries_per_chunk
                )
                if not queries:
                    # Fallback: use a truncated version of the document itself.
                    logger.warning(
                        "HyPE: no synthetic queries for doc %s; "
                        "falling back to document text.", did,
                    )
                    queries = [doc[:512]]

                all_synth_queries.extend(queries)
                self._index_doc_ids.extend([did] * len(queries))

                if (i + 1) % 50 == 0:
                    logger.info("HyPE: generated queries for %d/%d docs.",
                                i + 1, len(documents))

        logger.info(
            "HyPE: generated %d synthetic queries in %.2f s.",
            len(all_synth_queries), t_gen.elapsed,
        )

        # Step 2: embed all synthetic queries.
        with Timer() as t_embed:
            embeddings = self.embedder.embed_queries(all_synth_queries)
            embeddings = embeddings.astype(np.float32)

        logger.info(
            "HyPE: embedded %d synthetic queries in %.2f s (dim=%d).",
            embeddings.shape[0], t_embed.elapsed, embeddings.shape[1],
        )

        # Step 3: build FAISS inner-product index.
        with Timer() as t_index:
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(embeddings)

        logger.info(
            "HyPE: FAISS index built in %.2f s (%d vectors).",
            t_index.elapsed, self._index.ntotal,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        if self._index is None:
            raise RuntimeError(
                "Index has not been built yet. Call build_index() first."
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Embed the query, search synthetic-question index, deduplicate by doc.

        The FAISS search may return multiple entries for the same document
        (one per synthetic question).  Results are deduplicated by
        ``doc_id``, keeping the highest score for each document.

        Parameters
        ----------
        query : str
            Natural-language query string.
        top_k : int
            Number of unique documents to return.

        Returns
        -------
        list[RetrievedDoc]
            Deduplicated results sorted by descending score.
        """
        self._ensure_index()

        query_vec = self.embedder.embed_queries([query]).astype(np.float32)

        # Fetch more candidates than top_k to account for deduplication.
        candidate_k = min(
            top_k * self.num_queries_per_chunk,
            self._index.ntotal,
        )
        scores, indices = self._index.search(query_vec, candidate_k)

        # Deduplicate: keep best score per doc_id.
        best_per_doc: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                break
            did = self._index_doc_ids[idx]
            if did not in best_per_doc or score > best_per_doc[did]:
                best_per_doc[did] = float(score)

        # Sort by score descending and take top_k.
        sorted_docs = sorted(best_per_doc.items(), key=lambda x: x[1], reverse=True)
        sorted_docs = sorted_docs[:top_k]

        results: list[RetrievedDoc] = []
        for rank, (did, score) in enumerate(sorted_docs, start=1):
            results.append(
                RetrievedDoc(
                    doc_id=did,
                    text=self._doc_id_to_text[did],
                    score=score,
                    rank=rank,
                    method=self.name,
                    metadata={
                        "num_queries_per_chunk": self.num_queries_per_chunk,
                    },
                )
            )
        return results
