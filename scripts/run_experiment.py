"""Run a single retrieval experiment on T²-RAGBench."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import chunk_corpus
from src.data_loader import load_t2ragbench
from src.evaluation.retrieval_metrics import (
    compute_per_query_retrieval,
    compute_retrieval_metrics,
)
from src.utils.common import (
    RESULTS_DIR,
    ExperimentResult,
    Timer,
    get_logger,
    load_config,
    set_seed,
)

logger = get_logger(__name__)


def build_retriever(method: str, config: dict, doc_ids: list[str], documents: list[str]):
    """Factory: build and index a retriever by method name.

    Imports are lazy so each method only requires its own dependencies.
    """
    emb_config = config["embedding_models"].get(
        config.get("_embedding_key", "openai_large")
    )

    if method == "bm25":
        from src.retrieval.bm25_retriever import BM25Retriever
        retriever = BM25Retriever(
            k1=config["bm25"]["k1"], b=config["bm25"]["b"]
        )
        retriever.build_index(doc_ids, documents)

    elif method == "dense":
        from src.retrieval.base import create_embedder
        from src.retrieval.dense_retriever import DenseRetriever
        embedder = create_embedder(emb_config)
        retriever = DenseRetriever(embedder=embedder, name=f"dense_{emb_config['model']}")
        retriever.build_index(doc_ids, documents)

    elif method == "hybrid":
        from src.retrieval.base import create_embedder
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        embedder = create_embedder(emb_config)
        bm25 = BM25Retriever(k1=config["bm25"]["k1"], b=config["bm25"]["b"])
        bm25.build_index(doc_ids, documents)
        dense = DenseRetriever(embedder=embedder, name=f"dense_{emb_config['model']}")
        dense.build_index(doc_ids, documents)
        retriever = HybridRetriever(
            bm25_retriever=bm25,
            dense_retriever=dense,
            fusion=config["hybrid"]["fusion_method"],
            rrf_k=config["hybrid"]["rrf_k"],
            alpha=config["hybrid"]["cc_alpha"],
        )

    elif method == "colbert":
        from src.retrieval.colbert_retriever import ColBERTRetriever
        retriever = ColBERTRetriever()
        retriever.build_index(doc_ids, documents)

    elif method == "hyde":
        from src.retrieval.base import create_embedder
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.hyde_retriever import HyDERetriever
        embedder = create_embedder(emb_config)
        dense = DenseRetriever(embedder=embedder, name="dense_hyde")
        dense.build_index(doc_ids, documents)
        retriever = HyDERetriever(
            dense_retriever=dense,
            llm_model=config["hyde"]["llm_model"],
            prompt_template=config["hyde"]["prompt_template"],
            num_generations=config["hyde"]["num_generations"],
        )

    elif method == "multi_query":
        from src.retrieval.base import create_embedder
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.multi_query_retriever import MultiQueryRetriever
        embedder = create_embedder(emb_config)
        bm25 = BM25Retriever(k1=config["bm25"]["k1"], b=config["bm25"]["b"])
        bm25.build_index(doc_ids, documents)
        dense = DenseRetriever(embedder=embedder, name="dense_mq")
        dense.build_index(doc_ids, documents)
        inner = HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
        retriever = MultiQueryRetriever(
            inner_retriever=inner,
            llm_model=config["multi_query"]["llm_model"],
            num_queries=config["multi_query"]["num_queries"],
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return retriever


@click.command()
@click.option("--method", required=True, help="Retrieval method name")
@click.option("--config", "config_path", default=None, help="Config YAML path")
@click.option("--embedding", "emb_key", default="openai_large", help="Embedding model key")
@click.option("--reranker", "reranker_key", default="none", help="Reranker key (none/cohere/local)")
@click.option("--top-k", default=5, type=int, help="Number of documents to retrieve")
@click.option("--subset", default=None, help="Specific subset (FinQA/ConvFinQA/TAT-DQA)")
@click.option("--max-queries", default=None, type=int, help="Limit queries (for testing)")
@click.option("--output-name", default=None, help="Custom output filename")
def main(
    method: str,
    config_path: str | None,
    emb_key: str,
    reranker_key: str,
    top_k: int,
    subset: str | None,
    max_queries: int | None,
    output_name: str | None,
):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    cfg["_embedding_key"] = emb_key

    # Load data
    logger.info("Loading T²-RAGBench...")
    subsets = [subset] if subset else cfg["dataset"]["subsets"]
    data = load_t2ragbench(subsets=subsets)

    # Build corpus
    doc_ids = list(data.corpus.keys())
    documents = [data.corpus[did].text for did in doc_ids]

    # Chunk if needed
    chunk_strategy = cfg["chunking"]["strategy"]
    chunk_ids, chunk_texts, chunk_to_doc = chunk_corpus(
        doc_ids, documents,
        strategy=chunk_strategy,
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
    )

    # Build retriever
    logger.info(f"Building retriever: {method} (embedding: {emb_key})")
    with Timer() as index_timer:
        retriever = build_retriever(method, cfg, chunk_ids, chunk_texts)
    logger.info(f"Index built in {index_timer.elapsed:.1f}s")

    # Setup reranker (lazy import)
    #
    # _rerank_top_k is read from the config and deliberately not used: the
    # rerank call below passes the experiment's own top_k, and the candidate
    # pool is fixed at max(top_k * 3, 20). So `top_n` in configs/default.yaml
    # has no effect here. Kept, under a leading underscore, because it is the
    # only visible trace at this call site that the config key is inert.
    #
    # Why this is not simply wired up, and which figure it costs us, is in
    # CONTRIBUTING.md → "Known gaps".
    if reranker_key != "none" and reranker_key in cfg["rerankers"]:
        from src.reranking.reranker import create_reranker
        reranker = create_reranker(cfg["rerankers"][reranker_key])
        _rerank_top_k = cfg["rerankers"][reranker_key].get("top_n", top_k)
    else:
        from src.reranking.reranker import NoReranker
        reranker = NoReranker()
        _rerank_top_k = top_k

    # Prepare queries
    qa_items = data.qa_items[:max_queries] if max_queries else data.qa_items
    queries = [qa.question for qa in qa_items]
    relevant_ids = []
    for qa in qa_items:
        if chunk_strategy == "whole_doc":
            relevant_ids.append({qa.context_id})
        else:
            # Find all chunks that belong to the gold document
            relevant_chunks = {
                cid for cid, did in chunk_to_doc.items() if did == qa.context_id
            }
            relevant_ids.append(relevant_chunks)

    # Run retrieval
    logger.info(f"Retrieving for {len(queries)} queries (top_k={top_k})...")
    all_retrieved = []
    latencies = []

    for query in tqdm(queries, desc=f"Retrieving [{method}]"):
        with Timer() as t:
            # Retrieve more candidates for reranking
            candidates = retriever.retrieve(query, top_k=max(top_k * 3, 20))
            results = reranker.rerank(query, candidates, top_k=top_k)
        all_retrieved.append(results)
        latencies.append(t.elapsed_ms)

    # Compute metrics
    k_values = cfg["evaluation"]["k_values"]
    retrieval_metrics = compute_retrieval_metrics(all_retrieved, relevant_ids, k_values)

    # Per-query results
    per_query = []
    for i, (qa, retrieved) in enumerate(zip(qa_items, all_retrieved)):
        pq = compute_per_query_retrieval(retrieved, relevant_ids[i], k_values)
        pq["query_id"] = qa.id
        pq["subset"] = qa.subset
        pq["latency_ms"] = latencies[i]
        pq["retrieved_ids"] = [r.doc_id for r in retrieved]
        per_query.append(pq)

    # Build result
    method_label = method
    if reranker_key != "none":
        method_label += f"+{reranker_key}"

    result = ExperimentResult(
        method=method_label,
        config={
            "method": method,
            "embedding": emb_key,
            "reranker": reranker_key,
            "top_k": top_k,
            "chunking": chunk_strategy,
            "subset": subset or "all",
        },
        retrieval_metrics=retrieval_metrics,
        per_query_results=per_query,
        wall_clock_seconds=sum(latencies) / 1000,
        index_time_seconds=index_timer.elapsed,
        num_queries=len(queries),
        avg_latency_ms=float(sum(latencies) / len(latencies)) if latencies else 0,
    )

    # Save
    fname = output_name or f"{method_label}_{emb_key}_{chunk_strategy}"
    output_path = RESULTS_DIR / f"{fname}.json"
    result.save(output_path)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Method: {method_label}")
    logger.info(f"Embedding: {emb_key} | Reranker: {reranker_key}")
    logger.info(f"Queries: {result.num_queries} | Avg latency: {result.avg_latency_ms:.1f}ms")
    logger.info(f"Index time: {result.index_time_seconds:.1f}s")
    logger.info(f"{'='*60}")
    for metric, value in sorted(retrieval_metrics.items()):
        logger.info(f"  {metric}: {value:.4f}")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
