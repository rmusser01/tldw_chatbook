# Tests/RAG_Eval/test_metrics.py
"""Known-answer tests for the ported retrieval metrics (always-on; pure)."""
import math
import pytest
from tldw_chatbook.RAG_Search.eval.metrics import (
    evaluate_retrieval, evaluate_retrieval_batch, f1_at_k, mrr,
    ndcg_at_k, precision_at_k, recall_at_k,
)


def test_precision_at_k_hand_computed():
    # 2 of top-3 relevant
    assert precision_at_k(["a", "x", "b"], ["a", "b", "c"], k=3) == pytest.approx(2 / 3)

def test_recall_at_k_hand_computed():
    # 2 of 3 relevant found in top-3
    assert recall_at_k(["a", "x", "b"], ["a", "b", "c"], k=3) == pytest.approx(2 / 3)

def test_mrr_first_relevant_at_rank_2():
    assert mrr(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)

def test_ndcg_at_k_hand_computed():
    # relevant at ranks 1 and 3 of k=3, 2 relevant total:
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], ["a", "b"], k=3) == pytest.approx(expected)

def test_f1_is_harmonic_mean_of_p_and_r():
    p = precision_at_k(["a", "x"], ["a", "b", "c"], k=2)   # 0.5
    r = recall_at_k(["a", "x"], ["a", "b", "c"], k=2)      # 1/3
    assert f1_at_k(["a", "x"], ["a", "b", "c"], k=2) == pytest.approx(2 * p * r / (p + r))

def test_boundaries_empty_retrieved_and_empty_relevant():
    assert precision_at_k([], ["a"], k=5) == 0.0
    assert recall_at_k([], ["a"], k=5) == 0.0
    assert mrr([], ["a"]) == 0.0
    # Empty relevant: whatever convention the server module uses, pin it —
    # read the ported bodies and assert the exact behavior (0.0 expected).
    assert recall_at_k(["a"], [], k=5) == 0.0

def test_evaluate_retrieval_k_below_one_raises():
    with pytest.raises(ValueError):
        evaluate_retrieval(["a"], ["a"], k=0)

def test_batch_averages_across_queries():
    batch = [(["a"], ["a"]), (["x"], ["a"])]   # P@1 = 1.0 and 0.0
    out = evaluate_retrieval_batch(batch, k=1)
    assert out["precision"] == pytest.approx(0.5)
