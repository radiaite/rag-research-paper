"""HyPE (Hypothetical Prompt Embeddings) retriever."""

import os

import faiss
import numpy as np
from openai import AzureOpenAI, OpenAIError

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
    """Hypothetical Prompt Embeddings retriever."""

    name: str = "hype"

    def __init__(
        self,
        embedder: BaseEmbedder,
        llm_model: str = "gpt-4.1-mini",
        num_queries_per_chunk: int = 5,
        prompt_template: str = DEFAULT_HYPE_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        self.embedder = embedder
        self.llm_model = llm_model
        self.num_queries_per_chunk = max(1, num_queries_per_chunk)
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm_failures = 0
        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_LLM_ENDPOINT"],
        )

        self._index: faiss.IndexFlatIP | None = None
        self._index_doc_ids: list[str] = []
        self._doc_id_to_text: dict[str, str] = {}
        self.doc_ids: list[str] = []

    def _generate_synthetic_queries(self, text: str, n: int) -> list[str]:
        """Generate *n* synthetic questions for *text* using the LLM."""
        prompt = self.prompt_template.format(text=text, n=n)

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
            logger.exception("HyPE: LLM call failed for synthetic query generation.")
            return []

        questions: list[str] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            cleaned = line.lstrip("0123456789.-) ").strip()
            if cleaned:
                questions.append(cleaned)

        return questions[:n]

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Generate synthetic queries for each document, embed them, and build index."""
        if len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and documents ({len(documents)}) "
                "must have the same length."
            )

        logger.info(
            "HyPE: building index over %d documents (%d synthetic queries each) ...",
            len(documents),
            self.num_queries_per_chunk,
        )

        self.doc_ids = list(doc_ids)
        self._doc_id_to_text = dict(zip(doc_ids, documents, strict=True))

        all_synth_queries: list[str] = []
        self._index_doc_ids = []

        with Timer() as t_gen:
            for i, (did, doc) in enumerate(zip(doc_ids, documents, strict=True)):
                queries = self._generate_synthetic_queries(doc, self.num_queries_per_chunk)
                if not queries:
                    logger.warning(
                        "HyPE: no synthetic queries for doc %s; falling back to document text.",
                        did,
                    )
                    queries = [doc[:512]]

                all_synth_queries.extend(queries)
                self._index_doc_ids.extend([did] * len(queries))

                if (i + 1) % 50 == 0:
                    logger.info("HyPE: generated queries for %d/%d docs.", i + 1, len(documents))

        logger.info(
            "HyPE: generated %d synthetic queries in %.2f s.",
            len(all_synth_queries),
            t_gen.elapsed,
        )

        with Timer() as t_embed:
            embeddings = self.embedder.embed_queries(all_synth_queries)
            embeddings = embeddings.astype(np.float32)

        logger.info(
            "HyPE: embedded %d synthetic queries in %.2f s (dim=%d).",
            embeddings.shape[0],
            t_embed.elapsed,
            embeddings.shape[1],
        )

        with Timer() as t_index:
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dimension)
            self._index.add(embeddings)

        logger.info(
            "HyPE: FAISS index built in %.2f s (%d vectors).",
            t_index.elapsed,
            self._index.ntotal,
        )

    def ensure_index(self) -> faiss.IndexFlatIP:
        if self._index is None:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")
        return self._index

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Embed the query, search synthetic-question index, deduplicate by doc."""
        index = self.ensure_index()

        query_vec = self.embedder.embed_queries([query]).astype(np.float32)

        # Oversample because synthetic-question hits will deduplicate by doc_id.
        candidate_k = min(
            top_k * self.num_queries_per_chunk,
            index.ntotal,
        )
        scores, indices = index.search(query_vec, candidate_k)

        best_per_doc: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                break
            did = self._index_doc_ids[idx]
            if did not in best_per_doc or score > best_per_doc[did]:
                best_per_doc[did] = float(score)

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
