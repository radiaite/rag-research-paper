"""Contextual retriever inspired by Anthropic's Contextual Retrieval.

Prepends a short, LLM-generated context summary to each document chunk
before embedding and indexing.  This extra context helps the embedding and
BM25 models disambiguate chunks that would otherwise lose meaning when
separated from their surrounding document.

The contextualized chunks are fed into a :class:`HybridRetriever` (BM25 +
dense) so that both lexical and semantic matching benefit from the added
context.

Reference: Anthropic, "Introducing Contextual Retrieval" (2024).
https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

import os
from typing import Any

from openai import AzureOpenAI

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

# When the corpus consists of whole documents (no chunking), the prompt
# generates a summary-style context for the document itself.
DEFAULT_DOCUMENT_CONTEXT_PROMPT = (
    "Here is a document:\n"
    "<document>\n{document}\n</document>\n\n"
    "Please provide a concise summary context (2-3 sentences) that captures "
    "the key topics and entities in this document, for the purpose of "
    "improving search retrieval. Answer only with the context, nothing else."
)


class ContextualRetriever(BaseRetriever):
    """Contextual Retrieval: context-enriched hybrid search.

    For each document (or chunk), an LLM generates a brief context prefix
    that is prepended to the text before embedding and BM25 indexing.  A
    :class:`HybridRetriever` then handles the actual search over the
    contextualized corpus.

    If a pre-built :class:`HybridRetriever` is provided, its sub-retrievers
    are reused.  Otherwise, a new ``BM25Retriever`` and ``DenseRetriever``
    are created internally.

    Parameters
    ----------
    embedder : BaseEmbedder
        Embedding model for the dense component.
    hybrid_retriever : HybridRetriever or None
        Optional pre-configured hybrid retriever.  When ``None``, one is
        built internally using *embedder* and default BM25 settings.
    llm_model : str
        OpenAI chat model for context generation.
    prompt_template : str
        Format string with ``{document}`` and ``{chunk}`` placeholders.
        Used when documents are chunked and both the parent document and
        the chunk are available.
    document_prompt_template : str
        Format string with a ``{document}`` placeholder.  Used when
        the corpus consists of whole documents (no chunking).
    use_chunked_mode : bool
        If ``True``, expects ``(full_doc, chunk)`` pairs via a
        dedicated helper.  If ``False`` (default), treats each entry in
        ``documents`` as a standalone text and uses
        ``document_prompt_template``.
    temperature : float
        Sampling temperature for the LLM.
    max_tokens : int
        Maximum tokens per context generation call.
    openai_kwargs : dict
        Extra keyword arguments forwarded to ``OpenAI()``.
    """

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
        openai_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.embedder = embedder
        self.llm_model = llm_model
        self.prompt_template = prompt_template
        self.document_prompt_template = document_prompt_template
        self.use_chunked_mode = use_chunked_mode
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
        )

        # Build or reuse the hybrid retriever.
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

        # Mapping from contextualized text back to original text.
        self._original_texts: dict[str, str] = {}  # doc_id -> original text

    # ------------------------------------------------------------------
    # Context generation
    # ------------------------------------------------------------------

    def _generate_context(
        self,
        full_document: str,
        chunk: str | None = None,
    ) -> str:
        """Generate a short context prefix for a document or chunk.

        Parameters
        ----------
        full_document : str
            The full document text (or the only text when not chunking).
        chunk : str or None
            The specific chunk to contextualise.  When ``None``, the
            document-level prompt template is used.

        Returns
        -------
        str
            A short context string (2-3 sentences).
        """
        if chunk is not None:
            prompt = self.prompt_template.format(
                document=full_document, chunk=chunk,
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
        except Exception:
            logger.exception(
                "Contextual: LLM context generation failed; using empty context."
            )
            return ""

    # ------------------------------------------------------------------
    # Corpus contextualisation
    # ------------------------------------------------------------------

    def _contextualize_corpus(
        self,
        doc_ids: list[str],
        documents: list[str],
        full_documents: list[str] | None = None,
    ) -> list[str]:
        """Prepend LLM-generated context to each document.

        Parameters
        ----------
        doc_ids : list[str]
            Document identifiers.
        documents : list[str]
            Document (or chunk) texts.
        full_documents : list[str] or None
            When in chunked mode, the full parent documents corresponding
            to each chunk.  Must be the same length as *documents*.

        Returns
        -------
        list[str]
            Contextualized texts (context prefix + original text).
        """
        contextualized: list[str] = []

        with Timer() as t:
            for i, (did, doc) in enumerate(zip(doc_ids, documents)):
                if self.use_chunked_mode and full_documents is not None:
                    context = self._generate_context(
                        full_document=full_documents[i], chunk=doc,
                    )
                else:
                    context = self._generate_context(
                        full_document=doc, chunk=None,
                    )

                if context:
                    contextualized_text = f"{context}\n\n{doc}"
                else:
                    contextualized_text = doc

                contextualized.append(contextualized_text)
                self._original_texts[did] = doc

                if (i + 1) % 50 == 0:
                    logger.info(
                        "Contextual: generated context for %d/%d documents.",
                        i + 1, len(documents),
                    )

        logger.info(
            "Contextual: corpus contextualisation complete in %.2f s "
            "(%d documents).",
            t.elapsed, len(documents),
        )
        return contextualized

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Contextualise each document and build the hybrid index.

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
        """Build index with chunked documents and their parent texts.

        This is the full Contextual Retrieval workflow: each chunk gets
        context derived from its parent document before being indexed.

        Parameters
        ----------
        doc_ids : list[str]
            Identifier for each chunk (should be unique per chunk).
        chunks : list[str]
            Chunk texts.
        full_documents : list[str]
            The full parent document for each chunk (same length as
            *chunks*; documents with multiple chunks will repeat).
        """
        if len(doc_ids) != len(chunks) or len(chunks) != len(full_documents):
            raise ValueError(
                "doc_ids, chunks, and full_documents must have the same length."
            )

        logger.info(
            "Contextual (chunked): building index over %d chunks ...",
            len(chunks),
        )

        self.use_chunked_mode = True
        contextualized = self._contextualize_corpus(
            doc_ids, chunks, full_documents=full_documents,
        )
        self._hybrid.build_index(doc_ids, contextualized)

        logger.info("Contextual (chunked): index build complete.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Retrieve from the contextualized hybrid index.

        The query is passed as-is to the hybrid retriever (no query
        transformation).  Results are returned with the method name set
        to ``"contextual"``.

        Parameters
        ----------
        query : str
            Natural-language query string.
        top_k : int
            Number of documents to return.

        Returns
        -------
        list[RetrievedDoc]
            Hybrid search results over the contextualized corpus.
        """
        results = self._hybrid.retrieve(query, top_k=top_k)

        # Re-tag results with our method name and attach original text
        # in metadata.
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
