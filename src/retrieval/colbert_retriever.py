"""ColBERTv2 retriever via RAGatouille."""

from pathlib import Path

from src.retrieval.base import BaseRetriever
from src.utils.common import DATA_DIR, RetrievedDoc, get_logger

logger = get_logger(__name__)


class ColBERTRetriever(BaseRetriever):
    """ColBERTv2 late-interaction retriever using RAGatouille."""

    name = "colbert"

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        index_name: str = "t2ragbench",
        index_root: str | Path | None = None,
    ):
        self.model_name = model_name
        self.index_name = index_name
        self.index_root = str(index_root or DATA_DIR / "processed" / "colbert_index")
        self.doc_ids: list[str] = []
        self.documents: list[str] = []
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ragatouille import RAGPretrainedModel

            self._model = RAGPretrainedModel.from_pretrained(self.model_name)
        return self._model

    def build_index(self, doc_ids: list[str], documents: list[str]) -> None:
        """Build ColBERT index."""
        model = self._load_model()
        self.doc_ids = doc_ids
        self.documents = documents

        logger.info(f"Building ColBERT index for {len(documents)} documents...")
        model.index(
            collection=documents,
            document_ids=doc_ids,
            index_name=self.index_name,
            split_documents=False,
        )
        logger.info("ColBERT index built.")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDoc]:
        """Retrieve using ColBERT late interaction."""
        model = self._load_model()
        results = model.search(query=query, k=top_k)

        retrieved = []
        for rank, r in enumerate(results):
            retrieved.append(
                RetrievedDoc(
                    doc_id=str(r.get("document_id", r.get("doc_id", ""))),
                    text=r.get("content", ""),
                    score=r.get("score", 0.0),
                    rank=rank,
                    method="colbert",
                )
            )
        return retrieved

    def retrieve_batch(self, queries: list[str], top_k: int = 5) -> list[list[RetrievedDoc]]:
        """Batch retrieval (sequential — ColBERT doesn't natively batch search)."""
        return [self.retrieve(q, top_k) for q in queries]
