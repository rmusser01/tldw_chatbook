"""B4: the semantic-similarity model is constructed once per process.

Regression guard for the eval-runner hot path: ``calculate_semantic_similarity``
used to construct a fresh ``SentenceTransformer("all-MiniLM-L6-v2")`` for every
scored sample that was not an exact string match. With the process singleton in
place, the model is built exactly once and reused for the rest of the process.
"""

import sys
import threading
import types

from tldw_chatbook.Evals import eval_runner


class _FakeST:
    """Counting stand-in for sentence_transformers.SentenceTransformer."""

    instances = 0

    def __init__(self, *args, **kwargs):
        type(self).instances += 1

    def encode(self, texts, **kwargs):
        return [[float(len(t))] for t in texts]


def _install_fake_st(monkeypatch):
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    _FakeST.instances = 0


def test_model_constructed_once(monkeypatch):
    eval_runner._reset_semantic_model_for_tests()
    _install_fake_st(monkeypatch)
    try:
        calc = eval_runner.MetricsCalculator.calculate_semantic_similarity
        calc("hello world", "hello there")
        calc("another", "pair")
        assert _FakeST.instances == 1
    finally:
        eval_runner._reset_semantic_model_for_tests()


def test_model_constructed_once_under_concurrent_calls(monkeypatch):
    eval_runner._reset_semantic_model_for_tests()
    _install_fake_st(monkeypatch)
    try:
        calc = eval_runner.MetricsCalculator.calculate_semantic_similarity
        barrier = threading.Barrier(4)
        results = []

        def _call(index: int) -> None:
            barrier.wait()
            results.append(calc(f"prediction-{index}", f"expected-{index}"))

        threads = [
            threading.Thread(target=_call, args=(i,)) for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 4
        assert _FakeST.instances == 1
    finally:
        eval_runner._reset_semantic_model_for_tests()


def test_explicit_model_argument_still_honored(monkeypatch):
    eval_runner._reset_semantic_model_for_tests()
    _install_fake_st(monkeypatch)
    try:
        sentinel = _FakeST()
        calc = eval_runner.MetricsCalculator.calculate_semantic_similarity
        score = calc("abc", "abcd", embedding_model=sentinel)
        assert 0.0 <= score <= 1.0
        # The caller-supplied model is used; no singleton construction happened.
        assert _FakeST.instances == 1
    finally:
        eval_runner._reset_semantic_model_for_tests()
