"""Run a single retrieval experiment on T²-RAGBench."""

import contextlib

import click
from tqdm import tqdm

from src.chunking import chunk_corpus
from src.data_loader import load_t2ragbench
from src.evaluation.retrieval_metrics import (
    compute_per_query_retrieval,
    compute_retrieval_metrics,
)
from src.utils.common import (
    PROJECT_ROOT,
    RESULTS_DIR,
    ExperimentResult,
    Timer,
    collect_provenance,
    get_logger,
    load_config,
    set_seed,
)

logger = get_logger(__name__)

INDEX_CACHE_ROOT = PROJECT_ROOT / "data" / "indexes"


def _get_or_build_dense(
    embedder, emb_key: str, chunk_strategy: str, doc_ids: list[str], documents: list[str], name: str
):
    """Build a DenseRetriever, loading from cache if available."""
    from src.retrieval.dense_retriever import DenseRetriever

    max_seq = getattr(embedder, "max_seq_length", None)
    suffix = f"_seq{max_seq}" if max_seq else ""
    cache_dir = INDEX_CACHE_ROOT / f"{emb_key}_{chunk_strategy}{suffix}"
    retriever = DenseRetriever(embedder=embedder, name=name)

    if cache_dir.exists() and (cache_dir / "index.faiss").exists():
        logger.info(f"Loading cached dense index from {cache_dir}")
        retriever.load_index(cache_dir)
        if len(retriever.doc_ids) != len(doc_ids):
            logger.warning(
                f"Cached index has {len(retriever.doc_ids)} docs but corpus has "
                f"{len(doc_ids)}. Rebuilding."
            )
            retriever = DenseRetriever(embedder=embedder, name=name)
            retriever.build_index(doc_ids, documents)
            retriever.save_index(cache_dir)
    else:
        retriever.build_index(doc_ids, documents)
        retriever.save_index(cache_dir)

    return retriever


def build_retriever(method: str, config: dict, doc_ids: list[str], documents: list[str]):
    """Factory: build and index a retriever by method name."""
    emb_key = config.get("_embedding_key", "openai_large")
    emb_config = config["embedding_models"].get(emb_key)
    chunk_strategy = config["chunking"]["strategy"]

    if method == "bm25":
        from src.retrieval.bm25_retriever import BM25Retriever

        retriever = BM25Retriever(k1=config["bm25"]["k1"], b=config["bm25"]["b"])
        retriever.build_index(doc_ids, documents)

    elif method == "dense":
        from src.retrieval.base import create_embedder

        embedder = create_embedder(emb_config)
        retriever = _get_or_build_dense(
            embedder,
            emb_key,
            chunk_strategy,
            doc_ids,
            documents,
            name=f"dense_{emb_config['model']}",
        )

    elif method == "hybrid":
        from src.retrieval.base import create_embedder
        from src.retrieval.bm25_retriever import BM25Retriever
        from src.retrieval.hybrid_retriever import HybridRetriever

        embedder = create_embedder(emb_config)
        bm25 = BM25Retriever(k1=config["bm25"]["k1"], b=config["bm25"]["b"])
        bm25.build_index(doc_ids, documents)
        dense = _get_or_build_dense(
            embedder,
            emb_key,
            chunk_strategy,
            doc_ids,
            documents,
            name=f"dense_{emb_config['model']}",
        )
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
@click.option(
    "--reranker-max-length",
    default=None,
    type=int,
    help="Override the reranker's max_length without editing YAML. "
    "Use this to vary cross-encoder context window across runs.",
)
@click.option(
    "--reranker-batch-size",
    default=None,
    type=int,
    help="Override the reranker's batch_size without editing YAML.",
)
@click.option("--top-k", default=5, type=int, help="Number of documents to retrieve")
@click.option("--subset", default=None, help="Specific subset (FinQA/ConvFinQA/TAT-DQA)")
@click.option("--max-queries", default=None, type=int, help="Limit queries (for testing)")
@click.option("--output-name", default=None, help="Custom output filename")
def main(
    method: str,
    config_path: str | None,
    emb_key: str,
    reranker_key: str,
    reranker_max_length: int | None,
    reranker_batch_size: int | None,
    top_k: int,
    subset: str | None,
    max_queries: int | None,
    output_name: str | None,
):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    cfg["_embedding_key"] = emb_key

    # Fail loud before any expensive load: a typo'd reranker key used to fall
    # through silently to NoReranker (caused a 1228 s no-op H100 run).
    if reranker_key != "none" and reranker_key not in cfg["rerankers"]:
        available = ", ".join(sorted(cfg["rerankers"].keys())) or "(none defined)"
        raise click.UsageError(
            f"--reranker={reranker_key!r} is not defined in configs/default.yaml "
            f"under 'rerankers'. Available keys: {available}. "
            "Pass --reranker bge with --reranker-max-length / --reranker-batch-size "
            "to vary the operating point without editing YAML."
        )

    logger.info("Loading T²-RAGBench...")
    subsets = [subset] if subset else cfg["dataset"]["subsets"]
    data = load_t2ragbench(subsets=subsets)

    doc_ids = list(data.corpus.keys())
    documents = [data.corpus[did].text for did in doc_ids]

    chunk_strategy = cfg["chunking"]["strategy"]
    chunk_ids, chunk_texts, chunk_to_doc = chunk_corpus(
        doc_ids,
        documents,
        strategy=chunk_strategy,
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
    )

    logger.info(f"Building retriever: {method} (embedding: {emb_key})")
    with Timer() as index_timer:
        retriever = build_retriever(method, cfg, chunk_ids, chunk_texts)
    logger.info(f"Index built in {index_timer.elapsed:.1f}s")

    if reranker_key == "none":
        from src.reranking.reranker import NoReranker

        reranker = NoReranker()
        effective_reranker_max_length: int | None = None
    else:
        from src.reranking.reranker import create_reranker

        rcfg = dict(cfg["rerankers"][reranker_key])
        if reranker_max_length is not None:
            rcfg["max_length"] = reranker_max_length
        if reranker_batch_size is not None:
            rcfg["batch_size"] = reranker_batch_size
        reranker = create_reranker(rcfg)
        rcfg.get("top_n", top_k)
        effective_reranker_max_length = rcfg.get("max_length")

    qa_items = data.qa_items[:max_queries] if max_queries else data.qa_items
    queries = [qa.question for qa in qa_items]
    relevant_ids = []
    for qa in qa_items:
        if chunk_strategy == "whole_doc":
            relevant_ids.append({qa.context_id})
        else:
            relevant_chunks = {cid for cid, did in chunk_to_doc.items() if did == qa.context_id}
            relevant_ids.append(relevant_chunks)

    logger.info(f"Retrieving for {len(queries)} queries (top_k={top_k})...")
    all_retrieved = []
    latencies = []

    for query in tqdm(queries, desc=f"Retrieving [{method}]"):
        with Timer() as t:
            candidates = retriever.retrieve(query, top_k=max(top_k * 3, 20))
            results = reranker.rerank(query, candidates, top_k=top_k)
        all_retrieved.append(results)
        latencies.append(t.elapsed_ms)

    k_values = cfg["evaluation"]["k_values"]
    retrieval_metrics = compute_retrieval_metrics(all_retrieved, relevant_ids, k_values)

    per_query = []
    for i, (qa, retrieved) in enumerate(zip(qa_items, all_retrieved, strict=True)):
        per_query.append(
            {
                **compute_per_query_retrieval(retrieved, relevant_ids[i], k_values),
                "query_id": qa.id,
                "subset": qa.subset,
                "latency_ms": latencies[i],
                "retrieved_ids": [r.doc_id for r in retrieved],
            }
        )

    method_label = method
    if reranker_key != "none":
        method_label += f"+{reranker_key}"

    emb_model_name = (
        cfg["embedding_models"].get(emb_key, {}).get("model")
        if emb_key in cfg.get("embedding_models", {})
        else None
    )
    # Must mirror the cache-suffix logic in _get_or_build_dense so the SHA-256
    # resolves to the same on-disk index file.
    _max_seq = None
    if hasattr(retriever, "embedder") and hasattr(retriever.embedder, "max_seq_length"):
        _max_seq = retriever.embedder.max_seq_length
    elif hasattr(retriever, "dense_retriever") and hasattr(retriever.dense_retriever, "embedder"):
        _emb = retriever.dense_retriever.embedder
        if hasattr(_emb, "max_seq_length"):
            _max_seq = _emb.max_seq_length
    _index_suffix = f"_seq{_max_seq}" if _max_seq else ""
    index_path = INDEX_CACHE_ROOT / f"{emb_key}_{chunk_strategy}{_index_suffix}"
    provenance = collect_provenance(
        embedding_model=emb_model_name,
        index_path=index_path,
    )

    embedder_max_seq = _max_seq
    candidate_pool_size = max(top_k * 3, 20)

    # rglob, not just index.faiss: ColBERT indices have multiple shard files.
    index_size_mb = 0.0
    if index_path.exists():
        with contextlib.suppress(OSError):
            index_size_mb = sum(p.stat().st_size for p in index_path.rglob("*") if p.is_file()) / (
                1024 * 1024
            )

    result = ExperimentResult(
        method=method_label,
        config={
            "method": method,
            "embedding": emb_key,
            "reranker": reranker_key,
            "top_k": top_k,
            "chunking": chunk_strategy,
            "subset": subset or "all",
            "seed": cfg["seed"],
            "embedder_max_seq_length": embedder_max_seq,
            "reranker_max_length": effective_reranker_max_length,
            "candidate_pool_size": candidate_pool_size,
            "provenance": provenance,
            # LLM calls that fell back (HyDE: raw query, MultiQuery: no variants, ...).
            "llm_failures": getattr(retriever, "llm_failures", None),
        },
        retrieval_metrics=retrieval_metrics,
        per_query_results=per_query,
        wall_clock_seconds=sum(latencies) / 1000,
        index_time_seconds=index_timer.elapsed,
        index_size_mb=index_size_mb,
        num_queries=len(queries),
        avg_latency_ms=float(sum(latencies) / len(latencies)) if latencies else 0,
    )

    fname = output_name or f"{method_label}_{emb_key}_{chunk_strategy}"
    output_path = RESULTS_DIR / f"{fname}.json"
    result.save(output_path)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Method: {method_label}")
    logger.info(f"Embedding: {emb_key} | Reranker: {reranker_key}")
    logger.info(f"Queries: {result.num_queries} | Avg latency: {result.avg_latency_ms:.1f}ms")
    logger.info(f"Index time: {result.index_time_seconds:.1f}s")
    logger.info(f"{'=' * 60}")
    for metric, value in sorted(retrieval_metrics.items()):
        logger.info(f"  {metric}: {value:.4f}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
