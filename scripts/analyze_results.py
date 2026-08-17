"""Analyze experiment results and generate tables/figures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.common import RESULTS_DIR, get_logger

logger = get_logger(__name__)


def load_all_results(results_dir: Path | None = None) -> list[dict]:
    """Load all experiment result JSON files."""
    results_dir = results_dir or RESULTS_DIR
    results = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def main_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Build the main method comparison table."""
    rows = []
    for r in results:
        row = {
            "Method": r["method"],
            "Embedding": r["config"].get("embedding", "-"),
            "Reranker": r["config"].get("reranker", "none"),
        }
        # Add retrieval metrics
        for metric in ["recall@1", "recall@3", "recall@5", "recall@10", "recall@20",
                       "mrr@3", "mrr@5", "ndcg@5", "ndcg@10", "map"]:
            row[metric] = r["retrieval_metrics"].get(metric, None)
        # Add efficiency metrics
        row["avg_latency_ms"] = r.get("avg_latency_ms", None)
        row["index_time_s"] = r.get("index_time_seconds", None)
        row["num_queries"] = r.get("num_queries", None)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def per_subset_table(results: list[dict]) -> pd.DataFrame:
    """Build per-subset breakdown table from per_query_results."""
    rows = []
    for r in results:
        method = r["method"]
        pq = r.get("per_query_results", [])
        if not pq:
            continue

        df_pq = pd.DataFrame(pq)
        for subset in df_pq["subset"].unique():
            subset_df = df_pq[df_pq["subset"] == subset]
            row = {
                "Method": method,
                "Subset": subset,
                "N": len(subset_df),
            }
            for metric in ["recall@3", "recall@5", "recall@10", "mrr@3", "ndcg@10"]:
                if metric in subset_df.columns:
                    row[metric] = subset_df[metric].mean()
            row["avg_latency_ms"] = (
                subset_df["latency_ms"].mean()
                if "latency_ms" in subset_df.columns
                else None
            )
            rows.append(row)

    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, title: str = "") -> None:
    """Pretty-print a DataFrame as a markdown table."""
    if title:
        print(f"\n## {title}\n")

    # Format floats
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    formatted = df.copy()
    for col in float_cols:
        if "latency" in col or "time" in col:
            formatted[col] = formatted[col].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        else:
            formatted[col] = formatted[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")

    print(formatted.to_markdown(index=False))


if __name__ == "__main__":
    results = load_all_results()

    if not results:
        print("No results found in data/results/. Run experiments first.")
        sys.exit(0)

    print(f"Found {len(results)} experiment result(s).\n")

    # Main comparison
    main_df = main_comparison_table(results)
    print_table(main_df, "Main Comparison Table")

    # Per-subset breakdown
    subset_df = per_subset_table(results)
    if not subset_df.empty:
        print_table(subset_df, "Per-Subset Breakdown")
