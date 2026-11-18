"""Contextual retriever inspired by Anthropic's Contextual Retrieval."""

import os

from openai import AzureOpenAI, OpenAIError

from src.retrieval.base import BaseEmbedder, BaseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.common import RetrievedDoc, Timer, get_logger

logger = get_logger(__name__)

DEFAULT_CONTEXT_PROMPT = (
    "Here is the full document:\n"
    "<document>\n{document}\n</document>\n\n"
    "Here is a chunk from that document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short, succinct context (2-3 sentences) to situate this "
    "chunk within the overall document for the purposes of improving search "
    "retrieval of the chunk. Answer only with the context, nothing else."
)

DEFAULT_DOCUMENT_CONTEXT_PROMPT = (
    "Here is a document:\n"
    "<document>\n{document}\n</document>\n\n"
    "Please provide a concise summary context (2-3 sentences) that captures "
    "the key topics and entities in this document, for the purpose of "
    "improving search retrieval. Answer only with the context, nothing else."
)


class ContextualRetriever(BaseRetriever):
    """Contextual Retrieval: context-enriched hybrid search."""

    name: str = "contextual"

    def __init__(
        self,
        embedder: BaseEmbedder,
        hybrid_retriever: HybridRetriever | None = None,
        llm_model: str = "gpt-4.1-mini",
        prompt_template: str = DEFAULT_CONTEXT_PROMPT,
        document_prompt_template: str = DEFAULT_DOCUMENT_CONTEXT_PROMPT,
        use_chunked_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 128,
    ) -> None:
        self.embedder = embedder
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        self.document_prompt_template = document_prompt_template
        self.use_chunked_mode = use_chunked_mode
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm_failures = 0
        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_LLM_ENDPOINT"],
        )

        if hybrid_retriever is not None:
            self._hybrid = hybrid_retriever
        else:
            bm25 = BM25Retriever()
            dense = DenseRetriever(embedder, name="contextual_dense")
            self._hybrid = HybridRetriever(
                bm25_retriever=bm25,
                dense_retriever=dense,
                fusion="rrf",
            )

        self._original_texts: dict[str, str] = {}

    def _generate_context(
        self,
        full_document: str,
        chunk: str | None = None,
    ) -> str:
        """Generate a short context prefix for a document or chunk."""
        if chunk is not None:
            prompt = self.prompt_template.format(
                document=full_document,
                chunk=chunk,
            )
        else:
            prompt = self.document_prompt_template.format(
                document=full_document,
            )

        try:
            response = self._client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            context = response.choices[0].message.content or ""
            return context.strip()
        except OpenAIError:
            self.llm_failures += 1
            logger.exception("Contextual: LLM context generation failed; using empty context.")
            return ""

    def _contextualize_corpus(
        self,
        doc_ids: list[str],
        documents: list[str],
        full_documents: list[str] | None = None,
    ) -> list[str]:
        """Prepend LLM-generated context to each document."""
        contextualized: list[str] = []

        with Timer() as t:
            for i, (did, doc) in enumerate(zip(doc_ids, documents, strict=True)):
                if self.use_chunked_mode and full_documents is not None:
                    context = self._generate_context(
                        full_document=full_documents[i],
                        chunk=doc,
                    )
                else:
                    context = self._generate_context(
                        full_document=doc,
                        chunk=None,
                    )

                contextualized_text = f"{context}\n\n{doc}" if context else doc

                contextualized.append(contextualized_text)
                self._original_texts[did] = doc

                if (i + 1) % 50 == 0:
                    logger.info(
                        "Contextual: generated context for %d/%d documents.",
                        i + 1,
                        len(documents),
                    )

        logger.info(
            "Contextual: corpus contextualisation complete in %.2f s (%d documents).",
            t.elapsed,
            len(documents),
        )
        return contextualized

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Contextualise each document and build the hybrid index."""
        if len(doc_ids) != len(documents):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and documents ({len(documents)}) "
                "must have the same length."
            )

        logger.info(
            "Contextual: building index over %d documents ...",
            len(documents),
        )

        contextualized_docs = self._contextualize_corpus(doc_ids, documents)
        self._hybrid.build_index(doc_ids, contextualized_docs)

        logger.info("Contextual: index build complete.")

    def build_index_chunked(
        self,
        doc_ids: list[str],
        chunks: list[str],
        full_documents: list[str],
    ) -> None:
        """Build index with chunked documents and their parent texts."""
        if len(doc_ids) != len(chunks) or len(chunks) != len(full_documents):
            raise ValueError("doc_ids, chunks, and full_documents must have the same length.")

        logger.info(
            "Contextual (chunked): building index over %d chunks ...",
            len(chunks),
        )

        self.use_chunked_mode = True
        contextualized = self._contextualize_corpus(
            doc_ids,
            chunks,
            full_documents=full_documents,
        )
        self._hybrid.build_index(doc_ids, contextualized)

        logger.info("Contextual (chunked): index build complete.")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Retrieve from the contextualized hybrid index."""
        results = self._hybrid.retrieve(query, top_k=top_k)

        tagged: list[RetrievedDoc] = []
        for doc in results:
            original_text = self._original_texts.get(doc.doc_id, doc.text)
            metadata = dict(doc.metadata)
            metadata["original_text"] = original_text
            metadata["contextualized"] = True
            tagged.append(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=doc.score,
                    rank=doc.rank,
                    method=self.name,
                    metadata=metadata,
                )
            )
        return tagged
