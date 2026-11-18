"""HyDE (Hypothetical Document Embeddings) retriever."""

import os

import numpy as np
from openai import AzureOpenAI, OpenAIError

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
    """Hypothetical Document Embeddings retriever."""

    name: str = "hyde"

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        llm_model: str = "gpt-4.1-mini",
        num_generations: int = 1,
        prompt_template: str = DEFAULT_HYDE_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.llm_model = llm_model
        self.num_generations = max(1, num_generations)
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm_failures = 0
        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_LLM_ENDPOINT"],
        )

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Build the dense index."""
        logger.info("HyDE: delegating index build to inner dense retriever.")
        self.dense_retriever.build_index(doc_ids, documents)

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Call the LLM to produce a single hypothetical answer passage."""
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
        except OpenAIError:
            self.llm_failures += 1
            logger.exception("HyDE: LLM call failed for query: %s", query)
            return query  # fall back so retrieval still runs

    def _generate_hypothetical_docs(self, query: str) -> list[str]:
        """Generate ``num_generations`` hypothetical documents."""
        docs: list[str] = []
        for i in range(self.num_generations):
            doc = self._generate_hypothetical_doc(query)
            logger.debug("HyDE generation %d/%d (%d chars).", i + 1, self.num_generations, len(doc))
            docs.append(doc)
        return docs

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Generate a hypothetical document, embed it, and search the index."""
        index = self.dense_retriever.ensure_index()

        with Timer() as t_gen:
            hypo_docs = self._generate_hypothetical_docs(query)

        logger.info(
            "HyDE: generated %d hypothetical doc(s) in %.2f s.", len(hypo_docs), t_gen.elapsed
        )

        # Embed as documents (not queries) since they pose as corpus passages.
        with Timer() as t_embed:
            hypo_embeddings = self.dense_retriever.embedder.embed_documents(hypo_docs)
            if len(hypo_docs) > 1:
                query_vec = hypo_embeddings.mean(axis=0, keepdims=True)
            else:
                query_vec = hypo_embeddings
            # Re-normalise: IP index assumes unit vectors.
            norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
            norm = np.where(norm == 0.0, 1.0, norm)
            query_vec = (query_vec / norm).astype(np.float32)

        logger.debug("HyDE: embedding took %.1f ms.", t_embed.elapsed_ms)

        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vec, k)

        results: list[RetrievedDoc] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=True), start=1):
            if idx == -1:
                break
            results.append(
                RetrievedDoc(
                    doc_id=self.dense_retriever.doc_ids[idx],
                    text=self.dense_retriever.documents[idx],
                    score=float(score),
                    rank=rank,
                    method=self.name,
                    metadata={"num_generations": self.num_generations},
                )
            )
        return results
