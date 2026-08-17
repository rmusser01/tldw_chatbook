"""The LOCAL reranking strategy: `cross_encoder` (TASK-16965 Task 1).

`cross_encoder` has been a name without an implementation since task-3170 P0
(`config_profiles.py`'s standing comment; Hybrid Full ships `pointwise` as a
stopgap). TASK-16965 exists to answer whether reranking helps retrieval here,
and a LOCAL cross-encoder is the only strategy the gated instrument can see:
deterministic, unpriced, offline. This module pins the strategy's contract so
the measurement (Task 2) is measuring something honest.

**No model is ever loaded here.** Every test seeds the module-level model
cache with a stub exposing `.predict(pairs) -> list[float]`, and an autouse
fixture makes the real `sentence_transformers` import raise, so a cache miss
fails loudly instead of downloading 90 MB inside a unit run. The real model is
Task 2's business.

What the strategy must honour, inherited from `BaseReranker` and pinned below:

* the `RerankOutcome` contract -- results plus THIS call's `failed`/`total`,
  never per-instance state (TASK-3502 AC#4);
* the per-row `scored` flag -- a row no model scored keeps its original score
  and is NOT stamped with `rerank_score`, the production "this is no longer a
  similarity" marker (TASK-3502 note-b);
* the result cache and `top_k_to_rerank` window, so only the window is scored
  and the tail is handed back untouched;
* degradation instead of exceptions: reranking must never fail a search.

And what it must NOT do: reach a provider. It resolves no credential and never
calls `chat_api_call` -- that is what makes AC#5 (no spend, no network) fall
out of the implementation rather than out of a promise.
"""

import ast
import inspect

import pytest

from tldw_chatbook.RAG_Search import reranker as reranker_module
from tldw_chatbook.RAG_Search.reranker import (
    DEFAULT_CROSS_ENCODER_MODEL,
    CrossEncoderReranker,
    RerankingConfig,
    create_reranker,
    create_reranker_from_config,
)
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult

STUB_MODEL = "stub-org/stub-cross-encoder"


class _StubModelDown(RuntimeError):
    """Raised by the stub model; never a real load or a real network call."""


class _StubCrossEncoder:
    """Stands in for `sentence_transformers.CrossEncoder`.

    Scores are keyed by DOCUMENT so an assertion about ordering cannot be
    satisfied by the reranker handing the model its rows in a lucky order.
    """

    def __init__(self, scores: dict[str, float], fail: bool = False):
        self._scores = scores
        self._fail = fail
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):
        self.calls.append([tuple(pair) for pair in pairs])
        if self._fail:
            raise _StubModelDown("stub cross-encoder is down")
        return [self._scores[document] for _query, document in pairs]


@pytest.fixture(autouse=True)
def _no_real_model_loads(monkeypatch):
    """A cache miss must FAIL, not download. Nothing here loads a real model."""

    def _refuse():
        raise AssertionError(
            "a unit test tried to import/load a real cross-encoder model"
        )

    monkeypatch.setattr(reranker_module, "_import_cross_encoder_class", _refuse)
    yield


def _install(monkeypatch, stub: _StubCrossEncoder, name: str = STUB_MODEL) -> None:
    """Seed the module-level model cache, exercising the real cache-hit path."""
    monkeypatch.setitem(reranker_module._CROSS_ENCODER_MODELS, name, stub)


def _rows(*specs: tuple[str, float, str]) -> list[SearchResult]:
    return [
        SearchResult(id=rid, score=score, document=document, metadata={"src": rid})
        for rid, score, document in specs
    ]


def _config(**kwargs) -> RerankingConfig:
    kwargs.setdefault("model_name", STUB_MODEL)
    return RerankingConfig(strategy="cross_encoder", **kwargs)


@pytest.mark.asyncio
async def test_cross_encoder_reorders_by_model_score(monkeypatch):
    """The model's scores decide the order, and scored rows carry the stamp."""
    stub = _StubCrossEncoder({"doc-a": -8.0, "doc-b": 9.0, "doc-c": 0.0})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config())
    results = _rows(("a", 0.9, "doc-a"), ("b", 0.5, "doc-b"), ("c", 0.1, "doc-c"))

    outcome = await reranker.rerank("what is a cross-encoder", results)

    # The model saw (query, document) pairs, in the caller's order, once.
    assert stub.calls == [
        [
            ("what is a cross-encoder", "doc-a"),
            ("what is a cross-encoder", "doc-b"),
            ("what is a cross-encoder", "doc-c"),
        ]
    ]

    # ...and the ORIGINAL retrieval order (a, b, c) is not what comes back.
    assert [r.id for r in outcome.results] == ["b", "c", "a"]
    assert all("rerank_score" in r.metadata for r in outcome.results)
    # A higher model score must not become a lower rerank_score.
    by_id = {r.id: r.metadata["rerank_score"] for r in outcome.results}
    assert by_id["b"] > by_id["c"] > by_id["a"]
    assert (outcome.failed, outcome.total) == (0, 3)
    assert outcome.degraded is False
    # The caller's own objects are never mutated (the service caches by
    # reference -- TASK-3502 AC#3).
    assert [r.id for r in results] == ["a", "b", "c"]
    assert all("rerank_score" not in r.metadata for r in results)


@pytest.mark.asyncio
async def test_cross_encoder_needs_no_credential(monkeypatch):
    """No key, no provider call -- the whole point of the local strategy."""
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "COHERE_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    called: list[tuple] = []

    def _explode(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("the cross-encoder strategy called a chat provider")

    monkeypatch.setattr(reranker_module, "chat_api_call", _explode)

    stub = _StubCrossEncoder({"doc-a": 1.0, "doc-b": 2.0})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config())
    outcome = await reranker.rerank(
        "q", _rows(("a", 0.4, "doc-a"), ("b", 0.3, "doc-b"))
    )

    assert called == []
    assert (outcome.failed, outcome.total) == (0, 2)
    assert [r.id for r in outcome.results] == ["b", "a"]
    # Static half: the strategy's own CODE never names the provider seam.
    # Parsed, not grepped -- the class docstring mentions `chat_api_call` to
    # say it does not use it, and a substring check cannot tell the two apart.
    tree = ast.parse(inspect.getsource(CrossEncoderReranker))
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert "chat_api_call" not in referenced


@pytest.mark.asyncio
async def test_cross_encoder_honours_top_k_to_rerank(monkeypatch):
    """Only the window is scored; the tail keeps its order and its kind."""
    stub = _StubCrossEncoder({"doc-a": 0.0, "doc-b": 5.0})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config(top_k_to_rerank=2))
    results = _rows(
        ("a", 0.9, "doc-a"),
        ("b", 0.8, "doc-b"),
        ("c", 0.7, "doc-c"),
        ("d", 0.6, "doc-d"),
        ("e", 0.5, "doc-e"),
    )

    outcome = await reranker.rerank("q", results)

    # The model was handed the window only -- never the tail's documents.
    assert len(stub.calls) == 1
    assert [document for _q, document in stub.calls[0]] == ["doc-a", "doc-b"]

    assert [r.id for r in outcome.results] == ["b", "a", "c", "d", "e"]
    assert (outcome.failed, outcome.total) == (0, 2)
    tail = outcome.results[2:]
    assert [r.id for r in tail] == ["c", "d", "e"]
    assert all("rerank_score" not in r.metadata for r in tail)
    assert [r.score for r in tail] == [0.7, 0.6, 0.5]


@pytest.mark.asyncio
async def test_cross_encoder_model_failure_degrades_like_the_others(monkeypatch):
    """A raising model degrades the call; it never fails the search."""
    stub = _StubCrossEncoder({"doc-a": 1.0, "doc-b": 2.0}, fail=True)
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config())
    results = _rows(("a", 0.9, "doc-a"), ("b", 0.5, "doc-b"))

    outcome = await reranker.rerank("q", results)

    assert [r.id for r in outcome.results] == ["a", "b"]
    assert (outcome.failed, outcome.total) == (2, 2)
    assert outcome.degraded is True
    # note-b: nothing scored these rows, so nothing may claim they were.
    assert all("rerank_score" not in r.metadata for r in outcome.results)
    assert [r.score for r in outcome.results] == [0.9, 0.5]


def test_create_reranker_dispatches_cross_encoder():
    """The factory builds it, and the model name resolves to a cross-encoder."""
    reranker = create_reranker("cross_encoder")
    assert isinstance(reranker, CrossEncoderReranker)
    assert isinstance(
        create_reranker_from_config(RerankingConfig(strategy="cross_encoder")),
        CrossEncoderReranker,
    )

    # RerankingConfig.model_name defaults to an LLM ("gpt-3.5-turbo") because
    # the three shipped strategies are LLM-driven. A local strategy cannot
    # load that; it falls back to the measured default.
    assert RerankingConfig().model_name == "gpt-3.5-turbo"
    assert reranker.model_name == DEFAULT_CROSS_ENCODER_MODEL
    assert (
        create_reranker("cross_encoder", model_name="gpt-4o-mini").model_name
        == DEFAULT_CROSS_ENCODER_MODEL
    )
    # ...but a real cross-encoder repo id is honoured verbatim.
    assert (
        create_reranker("cross_encoder", model_name="BAAI/bge-reranker-base").model_name
        == "BAAI/bge-reranker-base"
    )


@pytest.mark.asyncio
async def test_provider_shaped_fields_are_no_ops(monkeypatch):
    """`max_retries` and `include_reasoning` are PROVIDER concepts.

    They belong to `_call_llm`'s retry loop and to a prompt that asks a model
    to explain itself. A local cross-encoder has neither, so state explicitly
    that setting them changes nothing rather than leaving it implied.
    """
    scores = {"doc-a": -1.0, "doc-b": 3.0}
    plain_stub = _StubCrossEncoder(dict(scores))
    _install(monkeypatch, plain_stub)
    plain = await CrossEncoderReranker(
        _config(max_retries=0, include_reasoning=False)
    ).rerank("q", _rows(("a", 0.9, "doc-a"), ("b", 0.5, "doc-b")))

    provider_shaped_stub = _StubCrossEncoder(dict(scores))
    _install(monkeypatch, provider_shaped_stub)
    provider_shaped = await CrossEncoderReranker(
        _config(max_retries=5, include_reasoning=True)
    ).rerank("q", _rows(("a", 0.9, "doc-a"), ("b", 0.5, "doc-b")))

    assert [r.id for r in provider_shaped.results] == [r.id for r in plain.results]
    assert [r.metadata for r in provider_shaped.results] == [
        r.metadata for r in plain.results
    ]
    assert [r.score for r in provider_shaped.results] == [
        r.score for r in plain.results
    ]
    assert len(provider_shaped_stub.calls) == len(plain_stub.calls) == 1
    # include_reasoning asked for nothing to explain: no reasoning is stamped.
    assert all("reasoning" not in r.metadata for r in provider_shaped.results)

    # max_retries does not buy a second attempt at a failing model either.
    failing_stub = _StubCrossEncoder(dict(scores), fail=True)
    _install(monkeypatch, failing_stub)
    degraded = await CrossEncoderReranker(_config(max_retries=5)).rerank(
        "q", _rows(("a", 0.9, "doc-a"), ("b", 0.5, "doc-b"))
    )
    assert len(failing_stub.calls) == 1
    assert (degraded.failed, degraded.total) == (2, 2)
