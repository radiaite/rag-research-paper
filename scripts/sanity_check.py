"""Sanity check: download data, run BM25 on 100 queries, verify pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import chunk_corpus
from src.data_loader import load_t2ragbench, save_corpus_texts, save_queries
from src.evaluation.generation_metrics import exact_match, number_match, token_f1
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.retrieval.bm25_retriever import BM25Retriever
from src.utils.common import Timer, get_logger, set_seed

logger = get_logger(__name__)


def main():
    set_seed(42)
    MAX_QUERIES = 100

    # Step 1: Load dataset
    logger.info("=" * 60)
    logger.info("STEP 1: Loading T²-RAGBench dataset...")
    logger.info("=" * 60)

    with Timer() as t:
        data = load_t2ragbench()
    logger.info(f"Dataset loaded in {t.elapsed:.1f}s")
    logger.info(data.summary())

    # Step 2: Save processed files
    logger.info("\nSTEP 2: Saving processed corpus and queries...")
    save_corpus_texts(data)
    save_queries(data)

    # Step 3: Prepare corpus
    doc_ids = list(data.corpus.keys())
    documents = [data.corpus[did].text for did in doc_ids]
    logger.info(f"\nCorpus: {len(doc_ids)} unique documents")
    logger.info(f"Avg doc length: {sum(len(d) for d in documents) / len(documents):.0f} chars")

    # Step 4: Chunk (whole document mode)
    chunk_ids, chunk_texts, chunk_to_doc = chunk_corpus(
        doc_ids, documents, strategy="whole_doc"
    )

    # Step 5: Build BM25 index
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Building BM25 index...")
    logger.info("=" * 60)

    bm25 = BM25Retriever(k1=1.2, b=0.75)
    with Timer() as t:
        bm25.build_index(chunk_ids, chunk_texts)
    logger.info(f"BM25 index built in {t.elapsed:.2f}s")

    # Step 6: Run retrieval on 100 queries
    logger.info("\n" + "=" * 60)
    logger.info(f"STEP 4: Running BM25 retrieval on {MAX_QUERIES} queries...")
    logger.info("=" * 60)

    qa_items = data.qa_items[:MAX_QUERIES]
    queries = [qa.question for qa in qa_items]
    relevant_ids = [{qa.context_id} for qa in qa_items]

    all_retrieved = []
    latencies = []
    for query in queries:
        with Timer() as t:
            results = bm25.retrieve(query, top_k=20)
        all_retrieved.append(results)
        latencies.append(t.elapsed_ms)

    avg_latency = sum(latencies) / len(latencies)
    logger.info(f"Avg retrieval latency: {avg_latency:.1f}ms/query")

    # Step 7: Compute metrics
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Computing retrieval metrics...")
    logger.info("=" * 60)

    metrics = compute_retrieval_metrics(all_retrieved, relevant_ids, k_values=[1, 3, 5, 10, 20])

    for name, value in sorted(metrics.items()):
        logger.info(f"  {name}: {value:.4f}")

    # Step 8: Oracle baseline check
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Oracle baseline verification...")
    logger.info("=" * 60)

    # Create fake "oracle" retrieval where gold doc is always rank 1
    from src.utils.common import RetrievedDoc
    oracle_retrieved = []
    for qa in qa_items:
        oracle_retrieved.append([RetrievedDoc(
            doc_id=qa.context_id,
            text=qa.gold_context,
            score=1.0,
            rank=0,
            method="oracle",
        )])

    oracle_metrics = compute_retrieval_metrics(oracle_retrieved, relevant_ids, k_values=[1, 3, 5])
    for name, value in sorted(oracle_metrics.items()):
        logger.info(f"  {name}: {value:.4f}")

    assert oracle_metrics["recall@1"] == 1.0, "Oracle should have perfect recall!"
    logger.info("  ✓ Oracle baseline verified (perfect scores)")

    # Step 9: Test generation metrics
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Testing generation metrics...")
    logger.info("=" * 60)

    test_cases = [
        ("206588.0", "206588.0", "exact match"),
        ("206588", "206588.0", "number match"),
        ("$206,588.00", "206588.0", "formatted number"),
        ("wrong answer", "206588.0", "wrong answer"),
    ]

    for pred, gold, desc in test_cases:
        nm = number_match(pred, gold)
        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)
        logger.info(f"  [{desc}] pred='{pred}' gold='{gold}' → NM={nm:.1f} EM={em:.1f} F1={f1:.2f}")

    # Step 10: Per-subset breakdown
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Per-subset breakdown...")
    logger.info("=" * 60)

    for subset_name in ["FinQA", "ConvFinQA", "TAT-DQA"]:
        subset_indices = [i for i, qa in enumerate(qa_items) if qa.subset == subset_name]
        if not subset_indices:
            continue
        subset_retrieved = [all_retrieved[i] for i in subset_indices]
        subset_relevant = [relevant_ids[i] for i in subset_indices]
        subset_metrics = compute_retrieval_metrics(
            subset_retrieved, subset_relevant, k_values=[3, 5, 10]
        )
        logger.info(f"\n  {subset_name} ({len(subset_indices)} queries):")
        for name in ["recall@3", "recall@5", "recall@10", "mrr@3"]:
            if name in subset_metrics:
                logger.info(f"    {name}: {subset_metrics[name]:.4f}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Dataset: {data.num_queries} queries, {data.num_documents} documents")
    logger.info(f"BM25 Recall@5: {metrics.get('recall@5', 0):.4f}")
    logger.info(f"BM25 MRR@3: {metrics.get('mrr@3', 0):.4f}")
    logger.info("Pipeline is working correctly. Ready for full experiments.")


if __name__ == "__main__":
    main()
