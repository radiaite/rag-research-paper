"""Chunking strategies for document preprocessing."""

from dataclasses import dataclass
from typing import Any

from src.utils.common import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A document chunk with parent reference."""

    chunk_id: str
    doc_id: str
    text: str
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] | None = None


def whole_document(doc_id: str, text: str) -> list[Chunk]:
    """No chunking — treat the entire document as one chunk."""
    return [Chunk(chunk_id=doc_id, doc_id=doc_id, text=text, end_char=len(text))]


def fixed_size_chunks(
    doc_id: str,
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    tokenizer: str = "char",
) -> list[Chunk]:
    """Split text into fixed-size chunks with overlap."""
    if tokenizer == "word":
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_c{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                )
            )
            start += chunk_size - overlap
            if start >= len(words):
                break
        return chunks
    else:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}_c{len(chunks)}",
                    doc_id=doc_id,
                    text=text[start:end],
                    start_char=start,
                    end_char=end,
                )
            )
            start += chunk_size - overlap
            if start >= len(text):
                break
        return chunks


def sentence_chunks(doc_id: str, text: str) -> list[Chunk]:
    """Split text into sentences using NLTK."""
    import nltk

    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)

    chunks = []
    offset = 0
    for sent in sentences:
        idx = text.find(sent, offset)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_s{len(chunks)}",
                doc_id=doc_id,
                text=sent,
                start_char=idx,
                end_char=idx + len(sent),
            )
        )
        offset = idx + len(sent)
    return chunks


def parent_child_chunks(
    doc_id: str,
    text: str,
    parent_size: int = 1024,
    child_size: int = 256,
    overlap: int = 32,
) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent and child chunks."""
    parents = fixed_size_chunks(doc_id, text, parent_size, 0, "word")
    all_children = []

    for parent in parents:
        children = fixed_size_chunks(parent.chunk_id, parent.text, child_size, overlap, "word")
        for child in children:
            child.metadata = {"parent_chunk_id": parent.chunk_id, "original_doc_id": doc_id}
        all_children.extend(children)

    return parents, all_children


def sentence_window_chunks(doc_id: str, text: str, window_size: int = 2) -> list[Chunk]:
    """Sentence-level chunks with surrounding window expansion."""
    import nltk

    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)

    chunks = []
    for i, sent in enumerate(sentences):
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        window_text = " ".join(sentences[start:end])

        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}_sw{i}",
                doc_id=doc_id,
                text=window_text,
                metadata={"center_sentence": sent, "window_start": start, "window_end": end},
            )
        )
    return chunks


def chunk_corpus(
    doc_ids: list[str],
    documents: list[str],
    strategy: str = "whole_doc",
    **kwargs,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Chunk an entire corpus using the specified strategy."""
    all_chunk_ids = []
    all_chunk_texts = []
    chunk_to_doc = {}

    for doc_id, text in zip(doc_ids, documents, strict=True):
        if strategy == "whole_doc":
            chunks = whole_document(doc_id, text)
        elif strategy == "fixed":
            chunks = fixed_size_chunks(
                doc_id,
                text,
                kwargs.get("chunk_size", 512),
                kwargs.get("chunk_overlap", 64),
            )
        elif strategy == "sentence":
            chunks = sentence_chunks(doc_id, text)
        elif strategy == "sentence_window":
            chunks = sentence_window_chunks(doc_id, text, kwargs.get("window_size", 2))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for chunk in chunks:
            all_chunk_ids.append(chunk.chunk_id)
            all_chunk_texts.append(chunk.text)
            chunk_to_doc[chunk.chunk_id] = doc_id

    logger.info(f"Chunked {len(doc_ids)} docs → {len(all_chunk_ids)} chunks (strategy={strategy})")
    return all_chunk_ids, all_chunk_texts, chunk_to_doc
