"""Load and preprocess the T²-RAGBench dataset from HuggingFace."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from src.utils.common import DATA_DIR, get_logger

logger = get_logger(__name__)

SUBSETS = ["FinQA", "ConvFinQA", "TAT-DQA"]


@dataclass
class QAItem:
    """A single question-answer pair with its ground truth context."""

    id: str
    subset: str
    question: str
    answer: str
    context_id: str
    gold_context: str
    table: str = ""
    pre_text: str = ""
    post_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """A single document in the corpus."""

    doc_id: str
    text: str
    subset: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class T2RAGBenchData:
    """Container for the full dataset."""

    qa_items: list[QAItem]
    corpus: dict[str, Document]  # context_id -> Document
    subsets: dict[str, list[QAItem]]  # subset_name -> list of QAItems

    @property
    def num_queries(self) -> int:
        return len(self.qa_items)

    @property
    def num_documents(self) -> int:
        return len(self.corpus)

    def get_subset(self, name: str) -> list[QAItem]:
        return self.subsets.get(name, [])

    def summary(self) -> str:
        lines = [f"T²-RAGBench: {self.num_queries} queries, {self.num_documents} documents"]
        for subset, items in self.subsets.items():
            lines.append(f"  {subset}: {len(items)} queries")
        return "\n".join(lines)


def load_t2ragbench(
    subsets: list[str] | None = None,
    split: str | None = None,
    cache_dir: str | Path | None = None,
) -> T2RAGBenchData:
    """Load T²-RAGBench from HuggingFace and build corpus."""
    subsets = subsets or SUBSETS
    cache_dir = str(cache_dir or DATA_DIR / "raw")

    all_qa: list[QAItem] = []
    corpus: dict[str, Document] = {}
    by_subset: dict[str, list[QAItem]] = {}

    for subset_name in subsets:
        logger.info(f"Loading subset: {subset_name}")
        ds = load_dataset("G4KMU/t2-ragbench", subset_name, cache_dir=cache_dir)

        available_splits = list(ds.keys())
        splits_to_use = [split] if split and split in available_splits else available_splits

        subset_items: list[QAItem] = []
        for sp in splits_to_use:
            for row in ds[sp]:
                context_id = row["context_id"]

                qa = QAItem(
                    id=row["id"],
                    subset=subset_name,
                    question=row["question"],
                    answer=str(row.get("program_answer", row.get("original_answer", ""))),
                    context_id=context_id,
                    gold_context=row["context"],
                    table=row.get("table", "") or "",
                    pre_text=row.get("pre_text", "") or "",
                    post_text=row.get("post_text", "") or "",
                    metadata={
                        "split": sp,
                        "company_name": row.get("company_name", ""),
                        "company_sector": row.get("company_sector", ""),
                        "report_year": str(row.get("report_year", "")),
                        "file_name": row.get("file_name", ""),
                    },
                )
                subset_items.append(qa)

                if context_id not in corpus:
                    corpus[context_id] = Document(
                        doc_id=context_id,
                        text=row["context"],
                        subset=subset_name,
                        metadata={
                            "company_name": row.get("company_name", ""),
                            "report_year": str(row.get("report_year", "")),
                        },
                    )

        all_qa.extend(subset_items)
        by_subset[subset_name] = subset_items
        logger.info(f"  {subset_name}: {len(subset_items)} QA pairs, corpus now {len(corpus)} docs")

    data = T2RAGBenchData(qa_items=all_qa, corpus=corpus, subsets=by_subset)
    logger.info(data.summary())
    return data


def save_corpus_texts(data: T2RAGBenchData, output_dir: Path | None = None) -> Path:
    """Save corpus as a JSON file for easy reuse."""
    output_dir = output_dir or DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_records = [
        {
            "doc_id": doc.doc_id,
            "text": doc.text,
            "subset": doc.subset,
            **doc.metadata,
        }
        for doc in data.corpus.values()
    ]

    path = output_dir / "corpus.json"
    pd.DataFrame(corpus_records).to_json(path, orient="records", indent=2)
    logger.info(f"Saved corpus ({len(corpus_records)} docs) to {path}")
    return path


def save_queries(data: T2RAGBenchData, output_dir: Path | None = None) -> Path:
    """Save queries as a JSON file for easy reuse."""
    output_dir = output_dir or DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    query_records = [
        {
            "id": qa.id,
            "subset": qa.subset,
            "question": qa.question,
            "answer": qa.answer,
            "context_id": qa.context_id,
        }
        for qa in data.qa_items
    ]

    path = output_dir / "queries.json"
    pd.DataFrame(query_records).to_json(path, orient="records", indent=2)
    logger.info(f"Saved queries ({len(query_records)}) to {path}")
    return path
