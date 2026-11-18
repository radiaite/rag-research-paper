"""Multi-Query retriever with Reciprocal Rank Fusion."""

import os
from collections import defaultdict
from typing import Any

from openai import AzureOpenAI, OpenAIError

from src.retrieval.base import BaseRetriever
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)

DEFAULT_MULTI_QUERY_PROMPT = (
    "You are a helpful assistant that generates alternative search queries. "
    "Given the following question, generate {n} alternative phrasings that "
    "capture the same information need but use different wording or "
    "perspectives. Return each query on its own line, numbered "
    "(e.g. 1. ... 2. ...). Do not include any other text.\n\n"
    "Original question: {query}\n\n"
    "Alternative queries:"
)


class MultiQueryRetriever(BaseRetriever):
    """Multi-Query retriever with RAG-Fusion."""

    name: str = "multi_query"

    def __init__(
        self,
        inner_retriever: BaseRetriever,
        llm_model: str = "gpt-4.1-mini",
        num_queries: int = 3,
        rrf_k: int = 60,
        prompt_template: str = DEFAULT_MULTI_QUERY_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 256,
        include_original: bool = True,
    ) -> None:
        self.inner_retriever = inner_retriever
        self.llm_model = llm_model
        self.num_queries = max(1, num_queries)
        self.rrf_k = rrf_k
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.include_original = include_original

        self.llm_failures = 0
        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_LLM_ENDPOINT"],
        )

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Build the index by delegating to the inner retriever."""
        logger.info(
            "MultiQuery: delegating index build to inner retriever (%s).",
            self.inner_retriever.name,
        )
        self.inner_retriever.build_index(doc_ids, documents)

    def _generate_query_variants(self, query: str, n: int) -> list[str]:
        """Generate *n* alternative phrasings of *query* using the LLM."""
        prompt = self.prompt_template.format(query=query, n=n)

        try:
            response = self._client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content or ""
        except OpenAIError:
            self.llm_failures += 1
            logger.exception("MultiQuery: LLM call failed for query variant generation.")
            return []

        variants: list[str] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned:
                variants.append(cleaned)

        return variants[:n]

    @staticmethod
    def _rrf_fuse(
        result_lists: list[list[RetrievedDoc]],
        k: int,
    ) -> dict[str, dict[str, Any]]:
        """Merge multiple ranked lists via Reciprocal Rank Fusion."""
        scores: dict[str, float] = defaultdict(float)
        texts: dict[str, str] = {}
        contributions: dict[str, int] = defaultdict(int)

        for result_list in result_lists:
            for doc in result_list:
                scores[doc.doc_id] += 1.0 / (k + doc.rank)
                texts.setdefault(doc.doc_id, doc.text)
                contributions[doc.doc_id] += 1

        return {
            did: {
                "text": texts[did],
                "score": scores[did],
                "contributing_lists": contributions[did],
            }
            for did in scores
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Generate query variants, retrieve for each, and RRF-fuse results."""
        with Timer() as t_gen:
            variants = self._generate_query_variants(query, self.num_queries)

        logger.info(
            "MultiQuery: generated %d variant(s) in %.2f s.",
            len(variants),
            t_gen.elapsed,
        )

        all_queries: list[str] = []
        if self.include_original:
            all_queries.append(query)
        all_queries.extend(variants)

        if not all_queries:
            logger.warning("MultiQuery: no variants generated; falling back to original query.")
            all_queries = [query]

        with Timer() as t_retrieve:
            result_lists: list[list[RetrievedDoc]] = []
            for q in all_queries:
                results = self.inner_retriever.retrieve(q, top_k=top_k)
                result_lists.append(results)

        logger.debug(
            "MultiQuery: retrieved for %d queries in %.2f s.",
            len(all_queries),
            t_retrieve.elapsed,
        )

        fused = self._rrf_fuse(result_lists, k=self.rrf_k)

        sorted_docs = sorted(
            fused.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )[:top_k]

        results: list[RetrievedDoc] = []
        for rank, (did, info) in enumerate(sorted_docs, start=1):
            results.append(
                RetrievedDoc(
                    doc_id=did,
                    text=info["text"],
                    score=info["score"],
                    rank=rank,
                    method=self.name,
                    metadata={
                        "num_query_variants": len(all_queries),
                        "contributing_lists": info["contributing_lists"],
                        "rrf_k": self.rrf_k,
                    },
                )
            )
        return results

    def retrieve_batch(self, queries: list[str], top_k: int = 5) -> list[list[RetrievedDoc]]:
        """Batch multi-query retrieval."""
        logger.info(
            "MultiQuery batch retrieval: %d queries, top_k=%d",
            len(queries),
            top_k,
        )

        with Timer() as t:
            results = [self.retrieve(q, top_k) for q in queries]

        logger.info(
            "MultiQuery batch retrieval finished in %.2f s (avg %.1f ms/query).",
            t.elapsed,
            t.elapsed_ms / max(len(queries), 1),
        )
        return results
