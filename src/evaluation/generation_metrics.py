"""Generation evaluation metrics: Number Match, EM, F1, ROUGE-L, BERTScore."""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison (SQuAD-style)."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def extract_number(text: str) -> float | None:
    """Extract the primary number from a text string."""
    text = text.strip()
    # Remove common prefixes/suffixes
    text = text.replace("$", "").replace("%", "").replace(",", "")
    text = text.replace("(", "-").replace(")", "")
    # Try to extract a float
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _relative_close(a: float, b: float, epsilon: float = 1e-2) -> bool:
    """Check if two numbers are within relative tolerance."""
    if b == 0:
        return abs(a) < epsilon
    return abs(a - b) / abs(b) < epsilon


def number_match(prediction: str, gold: str, epsilon: float = 1e-2) -> float:
    """Number Match metric from T²-RAGBench (relative tolerance, scale-invariant).

    Handles percentage/decimal mismatches: e.g., pred=53 vs gold=0.53 (53%)
    by testing multiple scale factors {1, 100, 0.01}.

    Returns 1.0 if numbers match within relative tolerance, 0.0 otherwise.
    """
    pred_num = extract_number(prediction)
    gold_num = extract_number(gold)

    if pred_num is None or gold_num is None:
        return 0.0

    # Direct comparison
    if _relative_close(pred_num, gold_num, epsilon):
        return 1.0

    # Scale-invariant: try ×100 and ÷100 for percentage/decimal mismatch
    for scale in [100.0, 0.01]:
        if _relative_close(pred_num * scale, gold_num, epsilon):
            return 1.0
        if _relative_close(pred_num, gold_num * scale, epsilon):
            return 1.0

    return 0.0


def exact_match(prediction: str, gold: str) -> float:
    """Exact match after normalization."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 score (SQuAD-style)."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F1 score."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(gold, prediction)
    return scores["rougeL"].fmeasure


def compute_bertscore(predictions: list[str], golds: list[str]) -> list[float]:
    """BERTScore F1 for a batch of predictions."""
    from bert_score import score as bert_score
    _, _, f1 = bert_score(predictions, golds, lang="en", verbose=False)
    return f1.tolist()


def compute_generation_metrics(
    predictions: list[str],
    golds: list[str],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute all generation metrics averaged over samples.

    Args:
        predictions: Predicted answers.
        golds: Gold answers.
        metrics: Which metrics to compute. Default: all.

    Returns:
        Dict of metric_name -> averaged value.
    """
    if metrics is None:
        metrics = ["number_match", "exact_match", "f1", "rouge_l", "bertscore"]

    results: dict[str, float] = {}

    if "number_match" in metrics:
        scores = [number_match(p, g) for p, g in zip(predictions, golds)]
        results["number_match"] = float(np.mean(scores))

    if "exact_match" in metrics:
        scores = [exact_match(p, g) for p, g in zip(predictions, golds)]
        results["exact_match"] = float(np.mean(scores))

    if "f1" in metrics:
        scores = [token_f1(p, g) for p, g in zip(predictions, golds)]
        results["f1"] = float(np.mean(scores))

    if "rouge_l" in metrics:
        scores = [compute_rouge_l(p, g) for p, g in zip(predictions, golds)]
        results["rouge_l"] = float(np.mean(scores))

    if "bertscore" in metrics:
        scores = compute_bertscore(predictions, golds)
        results["bertscore"] = float(np.mean(scores))

    return results


def compute_per_sample_generation(
    prediction: str, gold: str
) -> dict[str, float]:
    """Compute generation metrics for a single sample (no BERTScore)."""
    return {
        "number_match": number_match(prediction, gold),
        "exact_match": exact_match(prediction, gold),
        "f1": token_f1(prediction, gold),
        "rouge_l": compute_rouge_l(prediction, gold),
    }
