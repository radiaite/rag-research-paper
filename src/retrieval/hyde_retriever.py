"""HyDE (Hypothetical Document Embeddings) retriever.

Generates a hypothetical answer passage for a query using an LLM, embeds that
passage as if it were a document, and searches the dense index with the
resulting embedding.  When ``num_generations > 1``, multiple hypothetical
documents are generated and their embeddings are averaged before search.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (2022).  https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from openai import AzureOpenAI

from src.retrieval.base import BaseRetriever
from src.retrieval.dense_retriever import DenseRetriever
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)

DEFAULT_HYDE_PROMPT = (
    "Please write a short passage that directly answers the following question. "
    "The passage should be factual, detailed, and roughly the length of a "
    "typical encyclopedia paragraph.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)


class HyDERetriever(BaseRetriever):
    """Hypothetical Document Embeddings retriever.

    Wraps an existing :class:`DenseRetriever` and augments retrieval by
    first generating a hypothetical answer document with an LLM, embedding
    that hypothetical document, and using its embedding vector to query the
    dense index.

    Parameters
    ----------
    dense_retriever : DenseRetriever
        A dense retriever whose index has already been built (or will be
        built via :meth:`build_index`).
    llm_model : str
        OpenAI chat model used for hypothetical document generation.
    num_generations : int
        Number of hypothetical documents to generate per query.  When
        greater than 1 the embeddings are averaged before search.
    prompt_template : str
        A Python format string containing a ``{query}`` placeholder.
    temperature : float
        Sampling temperature for the LLM.
    max_tokens : int
        Maximum number of tokens for the generated passage.
    openai_kwargs : dict
        Extra keyword arguments forwarded to ``OpenAI()``.
    """

    name: str = "hyde"

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        llm_model: str = "gpt-4.1-mini",
        num_generations: int = 1,
        prompt_template: str = DEFAULT_HYDE_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 256,
        openai_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.llm_model = llm_model
        self.num_generations = max(1, num_generations)
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
        )

    # ------------------------------------------------------------------
    # Index construction (delegates to inner dense retriever)
    # ------------------------------------------------------------------

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Build the dense index.

        Simply delegates to the wrapped :class:`DenseRetriever`.  If the
        inner retriever already has a built index this is a no-op-safe
        operation (FAISS will rebuild).

        Parameters
        ----------
        doc_ids : list[str]
            Unique identifier for each document.
        documents : list[str]
            Raw document texts.
        """
        logger.info("HyDE: delegating index build to inner dense retriever.")
        self.dense_retriever.build_index(doc_ids, documents)

    # ------------------------------------------------------------------
    # Hypothetical document generation
    # ------------------------------------------------------------------

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Call the LLM to produce a single hypothetical answer passage.

        Parameters
        ----------
        query : str
            The user's natural-language question.

        Returns
        -------
        str
            A generated passage that attempts to answer *query*.
        """
        prompt = self.prompt_template.format(query=query)

        try:
            response = self._client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            text = response.choices[0].message.content or ""
            return text.strip()
        except Exception:
            logger.exception("HyDE: LLM call failed for query: %s", query)
            # Fall back to the original query so retrieval still works.
            return query

    def _generate_hypothetical_docs(self, query: str) -> list[str]:
        """Generate ``num_generations`` hypothetical documents.

        Parameters
        ----------
        query : str
            The user's natural-language question.

        Returns
        -------
        list[str]
            Generated passages (length equals ``self.num_generations``).
        """
        docs: list[str] = []
        for i in range(self.num_generations):
            doc = self._generate_hypothetical_doc(query)
            logger.debug("HyDE generation %d/%d (%d chars).",
                         i + 1, self.num_generations, len(doc))
            docs.append(doc)
        return docs

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Generate a hypothetical document, embed it, and search the index.

        When ``num_generations > 1``, multiple hypothetical documents are
        generated.  Their embeddings are averaged (centroid) and the
        resulting vector is used for a single FAISS search.

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
        self.dense_retriever._ensure_index()

        # Step 1: generate hypothetical document(s).
        with Timer() as t_gen:
            hypo_docs = self._generate_hypothetical_docs(query)

        logger.info("HyDE: generated %d hypothetical doc(s) in %.2f s.",
                     len(hypo_docs), t_gen.elapsed)

        # Step 2: embed the hypothetical docs *as documents* (not queries)
        # because they are meant to look like corpus passages.
        with Timer() as t_embed:
            hypo_embeddings = self.dense_retriever.embedder.embed_documents(hypo_docs)
            # Average when multiple generations.
            if len(hypo_docs) > 1:
                query_vec = hypo_embeddings.mean(axis=0, keepdims=True)
            else:
                query_vec = hypo_embeddings
            # Normalise for cosine similarity (IP index assumes unit vectors).
            norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
            norm = np.where(norm == 0.0, 1.0, norm)
            query_vec = (query_vec / norm).astype(np.float32)

        logger.debug("HyDE: embedding took %.1f ms.", t_embed.elapsed_ms)

        # Step 3: search FAISS directly.
        k = min(top_k, self.dense_retriever._index.ntotal)
        scores, indices = self.dense_retriever._index.search(query_vec, k)

        results: list[RetrievedDoc] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:
                break
            results.append(
                RetrievedDoc(
                    doc_id=self.dense_retriever._doc_ids[idx],
                    text=self.dense_retriever._documents[idx],
                    score=float(score),
                    rank=rank,
                    method=self.name,
                    metadata={"num_generations": self.num_generations},
                )
            )
        return results
