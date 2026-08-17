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
import time

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

#: Captured at import time, BEFORE the autouse fixture below replaces it with
#: a refusal. The one test that exercises the optional-dependency guard needs
#: the real function; it never reaches an import, because it makes
#: `optional_deps.check_dependency` report the package missing first.
_REAL_IMPORTER = reranker_module._import_cross_encoder_class


class _StubModelDown(RuntimeError):
    """Raised by the stub model; never a real load or a real network call."""


class _StubCrossEncoder:
    """Stands in for `sentence_transformers.CrossEncoder`.

    Scores are keyed by DOCUMENT so an assertion about ordering cannot be
    satisfied by the reranker handing the model its rows in a lucky order.
    ``batch_sizes`` records what the strategy forwarded, because the real
    `CrossEncoder.predict` takes ``batch_size`` and silently defaults to 32
    when nobody passes one -- a config field that never arrives is
    indistinguishable from one that does, without recording it.
    """

    def __init__(
        self, scores: dict[str, float], fail: bool = False, sleep_s: float = 0.0
    ):
        self._scores = scores
        self._fail = fail
        self._sleep_s = sleep_s
        self.calls: list[list[tuple[str, str]]] = []
        self.batch_sizes: list[object] = []

    def predict(self, pairs, batch_size=None, **kwargs):
        self.calls.append([tuple(pair) for pair in pairs])
        self.batch_sizes.append(batch_size)
        if self._sleep_s:
            time.sleep(self._sleep_s)
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


# ---------------------------------------------------------------------------
# The cache (Qodo PR-1775 finding 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_replays_an_identical_window(monkeypatch):
    """The cache still IS a cache: an identical repeat does not re-run the model."""
    stub = _StubCrossEncoder({"doc-a": -8.0, "doc-b": 9.0})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config())
    rows = [("a", 0.9, "doc-a"), ("b", 0.5, "doc-b")]

    first = await reranker.rerank("q", _rows(*rows))
    second = await reranker.rerank("q", _rows(*rows))

    assert len(stub.calls) == 1, "the second identical call re-ran the model"
    assert [r.id for r in second.results] == [r.id for r in first.results] == ["b", "a"]
    assert [r.score for r in second.results] == [r.score for r in first.results]
    assert (second.failed, second.total) == (0, 2)


@pytest.mark.asyncio
async def test_cache_does_not_misassign_scores_to_a_reordered_window(monkeypatch):
    """The SAME ids in a DIFFERENT order must not inherit the first order's scores.

    The cached values are positional -- ``_apply_scores`` zips them onto the
    rows it is handed -- so a key that ignores order hands row 0's score to
    whatever happens to be sitting at index 0 the second time. Here the model
    says ``doc-b`` wins by a mile; a retrieval that returns (b, a) instead of
    (a, b) must still put b first, not inherit a's low score.
    """
    stub = _StubCrossEncoder({"doc-a": -8.0, "doc-b": 9.0})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config())

    first = await reranker.rerank("q", _rows(("a", 0.9, "doc-a"), ("b", 0.5, "doc-b")))
    assert [r.id for r in first.results] == ["b", "a"]

    # Same query, same ids, same documents -- the retrieval returned them in
    # the other order (a re-search, a different fusion, a changed tie-break).
    second = await reranker.rerank("q", _rows(("b", 0.5, "doc-b"), ("a", 0.9, "doc-a")))

    assert [r.id for r in second.results] == ["b", "a"], (
        "the reordered window was scored with the FIRST window's positional "
        "scores: b (the model's clear winner) was demoted because it moved to "
        "index 0"
    )
    assert len(stub.calls) == 2, (
        "a differently-ordered window is a different scoring problem; it must "
        "miss the cache rather than replay positions"
    )


@pytest.mark.asyncio
async def test_cache_does_not_blend_stale_retrieval_scores(monkeypatch):
    """`combine_original_score` must blend against THIS call's retrieval scores.

    The cached `RerankingResult`s carry the first call's ``original_score``.
    The default config blends 30% of it into the final score, so replaying
    them scores the second search with the first search's retrieval numbers.
    """
    stub = _StubCrossEncoder({"doc-a": 0.0, "doc-b": 10.0})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config())
    assert reranker.config.combine_original_score is True
    assert reranker.config.original_score_weight == 0.3

    await reranker.rerank("q", _rows(("a", 0.9, "doc-a"), ("b", 0.1, "doc-b")))

    # Same query, same ids, same documents, same order -- but retrieval now
    # scores both rows at 0.0 (a different index, a different search mode).
    second = await reranker.rerank("q", _rows(("a", 0.0, "doc-a"), ("b", 0.0, "doc-b")))

    by_id = {r.id: r.score for r in second.results}
    # a: 0.3 * 0.0 + 0.7 * 0.0 (normalised min) == 0.0, not 0.3 * 0.9 == 0.27.
    assert by_id["a"] == pytest.approx(0.0), (
        f"row a blended a stale retrieval score: {by_id['a']!r}"
    )
    # b: 0.3 * 0.0 + 0.7 * 1.0 (normalised max) == 0.7, not 0.73.
    assert by_id["b"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_cache_keys_on_the_document_text(monkeypatch):
    """An id whose text changed is a different scoring problem, not a hit."""
    stub = _StubCrossEncoder({"doc-a": 1.0, "doc-a-rewritten": 9.0, "doc-b": 5.0})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config())

    await reranker.rerank("q", _rows(("a", 0.5, "doc-a"), ("b", 0.5, "doc-b")))
    second = await reranker.rerank(
        "q", _rows(("a", 0.5, "doc-a-rewritten"), ("b", 0.5, "doc-b"))
    )

    assert len(stub.calls) == 2
    assert [r.id for r in second.results] == ["a", "b"]


@pytest.mark.asyncio
async def test_cache_hit_does_not_mutate_the_cached_entry(monkeypatch):
    """A replay must not hand `_apply_scores` the cache's own objects to rewrite."""
    stub = _StubCrossEncoder({"doc-a": -8.0, "doc-b": 9.0})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config())
    rows = [("a", 0.9, "doc-a"), ("b", 0.5, "doc-b")]

    await reranker.rerank("q", _rows(*rows))
    cached = list(reranker._cache.values())[0]
    ranks_before = [(r.original_rank, r.new_rank) for r in cached]

    await reranker.rerank("q", _rows(*rows))

    assert [(r.original_rank, r.new_rank) for r in cached] == ranks_before


@pytest.mark.asyncio
async def test_cache_can_be_switched_off(monkeypatch):
    """`cache_results=False` means every call reaches the model."""
    stub = _StubCrossEncoder({"doc-a": 1.0, "doc-b": 2.0})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config(cache_results=False))
    rows = [("a", 0.9, "doc-a"), ("b", 0.5, "doc-b")]
    await reranker.rerank("q", _rows(*rows))
    await reranker.rerank("q", _rows(*rows))

    assert len(stub.calls) == 2


# ---------------------------------------------------------------------------
# Honoured config (Qodo PR-1775 findings 7 and 8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_size_reaches_the_model(monkeypatch):
    """`RerankingConfig.batch_size` is forwarded, not left to the model's 32."""
    stub = _StubCrossEncoder({f"doc-{i}": float(i) for i in range(6)})
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config(batch_size=3))
    await reranker.rerank(
        "q", _rows(*[(f"r{i}", 0.5, f"doc-{i}") for i in range(6)])
    )

    assert stub.batch_sizes == [3]


@pytest.mark.asyncio
async def test_a_nonsense_batch_size_still_scores(monkeypatch):
    """A zero/negative batch size floors to 1 rather than raising at the model."""
    stub = _StubCrossEncoder({"doc-a": 1.0, "doc-b": 2.0})
    _install(monkeypatch, stub)

    outcome = await CrossEncoderReranker(_config(batch_size=0)).rerank(
        "q", _rows(("a", 0.9, "doc-a"), ("b", 0.5, "doc-b"))
    )

    assert stub.batch_sizes == [1]
    assert (outcome.failed, outcome.total) == (0, 2)


@pytest.mark.asyncio
async def test_timeout_seconds_bounds_the_model_call(monkeypatch):
    """A model that overruns `timeout_seconds` degrades; it never holds a search.

    The executor thread cannot be cancelled -- it runs to completion and its
    result is discarded -- but the AWAIT is bounded, which is the part a
    search's latency depends on.
    """
    stub = _StubCrossEncoder({"doc-a": 1.0, "doc-b": 9.0}, sleep_s=0.4)
    _install(monkeypatch, stub)

    reranker = CrossEncoderReranker(_config(timeout_seconds=0.02))
    started = time.perf_counter()
    outcome = await reranker.rerank(
        "q", _rows(("a", 0.9, "doc-a"), ("b", 0.5, "doc-b"))
    )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.35, f"the await was not bounded by the timeout ({elapsed:.3f}s)"
    assert (outcome.failed, outcome.total) == (2, 2)
    assert outcome.degraded is True
    assert [r.id for r in outcome.results] == ["a", "b"]
    # note-b: a timed-out row was never scored, so it may not claim it was.
    assert all("rerank_score" not in r.metadata for r in outcome.results)
    # A timeout is transient: it must not be cached as a result.
    assert reranker._cache == {}


@pytest.mark.asyncio
async def test_a_generous_timeout_does_not_interfere(monkeypatch):
    """The bound is a bound, not a throttle."""
    stub = _StubCrossEncoder({"doc-a": 1.0, "doc-b": 9.0}, sleep_s=0.02)
    _install(monkeypatch, stub)

    outcome = await CrossEncoderReranker(_config(timeout_seconds=30.0)).rerank(
        "q", _rows(("a", 0.9, "doc-a"), ("b", 0.5, "doc-b"))
    )

    assert (outcome.failed, outcome.total) == (0, 2)
    assert [r.id for r in outcome.results] == ["b", "a"]


# ---------------------------------------------------------------------------
# Model resolution and loading (Qodo PR-1775 findings 2, 3 and 6)
# ---------------------------------------------------------------------------


def test_a_local_model_path_is_not_replaced_by_the_default(tmp_path):
    """An existing filesystem path is a path, whatever it is named.

    The resolver's contract says local paths are used verbatim. A bare
    directory (`/tmp/x/my-reranker`, or a relative `models`) contains no
    forward slash after `partition`, and a Windows path contains none at
    all, so the chat-model heuristic would have swallowed both.
    """
    from tldw_chatbook.RAG_Search.reranker import _resolve_cross_encoder_model_name

    local_dir = tmp_path / "my-reranker"
    local_dir.mkdir()

    assert _resolve_cross_encoder_model_name(str(local_dir)) == str(local_dir)
    # ...including one whose basename has no separator at all once resolved.
    assert (
        _resolve_cross_encoder_model_name(r"C:\models\bge-base")
        == r"C:\models\bge-base"
    ), "a Windows path was silently replaced with the default artifact"
    assert _resolve_cross_encoder_model_name("./models/local") == "./models/local"
    assert _resolve_cross_encoder_model_name("~/models/local") == "~/models/local"

    # The chat-model fallback still fires for names that are not paths.
    assert _resolve_cross_encoder_model_name("gpt-4o-mini") == DEFAULT_CROSS_ENCODER_MODEL
    assert _resolve_cross_encoder_model_name("openai/gpt-4o") == DEFAULT_CROSS_ENCODER_MODEL
    assert (
        _resolve_cross_encoder_model_name("BAAI/bge-reranker-base")
        == "BAAI/bge-reranker-base"
    )


def test_the_model_is_loaded_offline_only(monkeypatch):
    """Production must not reach a model hub: the docs promise no network."""
    from tldw_chatbook.RAG_Search.reranker import _load_cross_encoder

    recorded: list[tuple[tuple, dict]] = []

    class _Recorder:
        def __init__(self, *args, **kwargs):
            recorded.append((args, kwargs))

    monkeypatch.setattr(reranker_module, "_CROSS_ENCODER_MODELS", {})
    monkeypatch.setattr(
        reranker_module, "_import_cross_encoder_class", lambda: _Recorder
    )

    _load_cross_encoder("stub-org/stub-cross-encoder")

    assert len(recorded) == 1
    args, kwargs = recorded[0]
    assert args == ("stub-org/stub-cross-encoder",)
    assert kwargs.get("local_files_only") is True, (
        "an uncached first use would silently fetch from the hub"
    )


def test_the_optional_import_is_guarded_by_optional_deps(monkeypatch):
    """`sentence_transformers` is an `embeddings_rag` extra, not a hard import."""
    from tldw_chatbook.Utils import optional_deps

    monkeypatch.setattr(reranker_module, "_import_cross_encoder_class", _REAL_IMPORTER)
    monkeypatch.setattr(optional_deps, "check_dependency", lambda *a, **k: False)

    with pytest.raises(ImportError) as excinfo:
        reranker_module._import_cross_encoder_class()

    assert "embeddings_rag" in str(excinfo.value)
