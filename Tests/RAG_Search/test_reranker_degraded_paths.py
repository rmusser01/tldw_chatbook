"""Degraded-path honesty for the LLM reranker (TASK-3502: AC#3, AC#4, note-b).

TASK-3170 made a reranking-enabled profile actually construct and RUN a
reranker (a double-strategy TypeError had meant reranking silently never
activated). Its re-review left three residuals, all about what the reranker
CLAIMS when its provider calls fail. This module pins all three:

* **AC#3 -- copy-not-mutate through the REAL Pairwise/Listwise strategies.**
  `_tag_first_result` (enhanced_rag_service_v2.py) exists because
  `RAGService.search()`'s cache holds `SearchResult` objects BY REFERENCE, so
  an in-place tag write poisons the cache for up to `cache_ttl`. Pointwise had
  that pinned; Pairwise/Listwise did not, and their copy semantics differ --
  neither builds copies at all, they hand BACK the caller's own objects (and,
  on Listwise's failure paths, the caller's own LIST). That makes the
  no-mutation contract load-bearing in a different way than for Pointwise,
  which is exactly why the residual was filed.
* **note-(b) -- the "| reranked" over-claim.** A partly-failed pointwise run
  used to stamp `metadata["rerank_score"] = <the ORIGINAL score>` on rows
  whose scoring call had failed, so a 14/15-failed rerank rendered
  " | reranked" on fourteen rows that were never rescored. Failed rows now
  keep their original score AND their original score kind.
* **AC#4 -- the counter race.** The per-call failure counts used to live in
  `BaseReranker.last_rerank_failures/_total`, instance state on a reranker
  the service holds as a singleton, read by the disclosure site AFTER
  `rerank()` returned. They are now RETURNED (`RerankOutcome`), so one
  search's failures can no longer be attributed to another search's tag.

**No live provider calls.** Every test here fakes `chat_api_call` -- the
reranker's single provider seam (imported at `RAG_Search/reranker.py`
module level, invoked through `run_in_executor` in `_call_llm_impl`) -- so
the whole real path (`rerank` -> `_call_llm` -> `_call_llm_impl` -> the
fake) executes. The fake BINDS what it is handed against the real
`chat_api_call` signature (TASK-17065): the previous one declared the
caller's own mis-ordered positional list and therefore agreed with the bug.
No credential is planted any more -- the reranker resolves none, so nothing
gates the call before the seam.
"""

import inspect
from typing import Callable, List

import pytest

from tldw_chatbook.Chat.Chat_Functions import chat_api_call as real_chat_api_call
from tldw_chatbook.RAG_Search import reranker as reranker_module
from tldw_chatbook.RAG_Search.reranker import (
    ListwiseReranker,
    PairwiseReranker,
    PointwiseReranker,
    RerankingConfig,
)
from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig, get_profile_manager
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import (
    EnhancedRAGServiceV2,
    _tag_first_result,
)
from tldw_chatbook.RAG_Search.simplified.vector_store import SearchResult


class _FakeProviderDown(RuntimeError):
    """Raised by the fake provider seam; never a real network failure."""


def _install_fake_provider(
    monkeypatch, responder: Callable[[list], str] | None = None
) -> List[dict]:
    """Fake the reranker's ONLY provider seam, BOUND to the real signature.

    `responder` receives the messages payload and returns the raw model
    string (or raises, to simulate a failing provider call); it defaults to
    a fixed valid score.

    Everything the reranker passes is run through
    `inspect.signature(chat_api_call).bind(...)`, so a mis-ordered positional
    call or a mis-named keyword raises HERE. The fake this replaces declared
    the caller's OWN (wrong) positional list -- `(api_key, messages_payload,
    provider, model, temp, maxp)` -- so it agreed with the bug, and a green
    suite could not see that the reranker reached zero of the 29 providers it
    offers (TASK-17065). No fake at this seam may be written that way again.
    No credential is planted either: the reranker resolves none, so nothing
    gates the call before the seam.

    Returns each call's `BoundArguments.arguments` -- what was ACTUALLY
    passed, defaults NOT applied, so "sent no credential at all" stays
    distinguishable from "sent `api_key=None`".
    """
    signature = inspect.signature(real_chat_api_call)
    landings: List[dict] = []

    def fake_chat_api_call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        landings.append(dict(bound.arguments))
        payload = bound.arguments["messages_payload"]
        return responder(payload) if responder is not None else '{"score": 0.5}'

    monkeypatch.setattr(reranker_module, "chat_api_call", fake_chat_api_call)
    return landings


def _always_fails(_messages) -> str:
    raise _FakeProviderDown("provider unavailable (fake)")


def _fail_for_titles(titles, score: float = 0.9) -> Callable[[list], str]:
    """Fail the scoring call for these `doc_title`s; score the rest."""
    wanted = set(titles)

    def responder(messages) -> str:
        prompt = messages[-1]["content"]
        for title in wanted:
            if f"Title: {title}\n" in prompt:
                raise _FakeProviderDown(f"provider unavailable for {title} (fake)")
        return '{"score": %s}' % score

    return responder


def _results(n: int) -> List[SearchResult]:
    return [
        SearchResult(
            id=f"doc-{i}",
            score=round(0.9 - 0.1 * i, 3),
            document=f"body of document number {i}",
            metadata={"doc_title": f"doc-{i}", "source_type": "media"},
        )
        for i in range(n)
    ]


def _snapshot(results):
    return [(r.id, r.score, r.document, dict(r.metadata)) for r in results]


def _degraded_config(strategy: str) -> RerankingConfig:
    # retry_on_failure=False: the retry path sleeps 1s then 2s per failed
    # call, which would make a fully-degraded run take minutes.
    return RerankingConfig(strategy=strategy, top_k_to_rerank=5, retry_on_failure=False)


# --------------------------------------------------------------------------
# AC#3 -- copy-not-mutate, at the reranker seam
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["pairwise", "listwise"])
@pytest.mark.asyncio
async def test_degraded_rerank_never_touches_the_callers_cached_objects(
    strategy, monkeypatch
):
    """The AC#3 residual, at the seam: a fully degraded Pairwise/Listwise run
    against a list the caller's cache is holding BY REFERENCE, followed by the
    `_tag_first_result` disclosure the service applies. The cached objects and
    the cached list must both come out byte-identical, and the tag must be on
    the RETURNED list only."""
    cls = {"pairwise": PairwiseReranker, "listwise": ListwiseReranker}[strategy]
    reranker = cls(_degraded_config(strategy))
    _install_fake_provider(monkeypatch, _always_fails)

    cached_list = _results(4)
    cache = {"quokka marsupial": cached_list}  # simulates rag_service's cache
    before = _snapshot(cached_list)
    identities = [id(r) for r in cached_list]

    outcome = await reranker.rerank("quokka marsupial", cache["quokka marsupial"])

    assert outcome.failed > 0, "the fake provider must have failed every call"

    # Neither strategy COPIES: the rows coming back ARE the caller's (cached)
    # objects, and Listwise's failure paths hand back the caller's LIST
    # itself. That is precisely why the disclosure must build a new list with
    # a new first element -- an in-place tag write here lands in the cache.
    assert {id(r) for r in outcome.results} <= set(identities)

    tagged = _tag_first_result(
        outcome.results,
        "reranking_degraded",
        f"{outcome.failed}/{outcome.total} scorings failed",
    )

    assert tagged[0].metadata["reranking_degraded"] == (
        f"{outcome.failed}/{outcome.total} scorings failed"
    )
    assert _snapshot(cache["quokka marsupial"]) == before, (
        f"{cls.__name__} mutated the caller's (cached) SearchResult objects"
    )
    assert [id(r) for r in cache["quokka marsupial"]] == identities, (
        f"{cls.__name__} reordered the caller's own list in place"
    )
    assert not any(
        "reranking_degraded" in r.metadata for r in cache["quokka marsupial"]
    ), "the disclosure tag leaked onto a cached object"


@pytest.mark.parametrize("strategy", ["pairwise", "listwise"])
@pytest.mark.asyncio
async def test_degraded_tag_does_not_poison_the_search_cache(
    strategy, tmp_path, monkeypatch
):
    """The same contract end-to-end, through the real service with its base
    search cache ON: a degraded reranked search tags the disclosure, and a
    later identical search with reranking OFF (served from that cache) must
    not see the stale tag."""
    service = _make_v2_service(tmp_path, strategy, enable_cache=True)
    assert service.reranker is not None
    _install_fake_provider(monkeypatch, _always_fails)

    await service.index_batch_optimized(_quokka_platypus_docs())
    query = "quokka marsupial"

    degraded = await service.search(
        query, top_k=5, search_type="semantic", include_citations=False
    )
    assert degraded[0].metadata.get("reranking_degraded"), (
        f"{strategy} degradation must be disclosed"
    )

    clean = await service.search(
        query,
        top_k=5,
        search_type="semantic",
        include_citations=False,
        rerank=False,
    )
    assert not clean[0].metadata.get("reranking_degraded"), (
        "stale reranking_degraded tag leaked forward via a cached SearchResult"
    )


# --------------------------------------------------------------------------
# note-(b) -- no "| reranked" on rows that were never rescored
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_pointwise_rows_do_not_claim_the_reranked_score_kind(monkeypatch):
    """3 of 5 scoring calls fail. The two SCORED rows carry the reranker
    marker and stop being banded as a similarity; the three FAILED rows keep
    the score they arrived with AND its kind -- rendering " | reranked" over a
    row no model ever looked at is the over-claim this closes."""
    from tldw_chatbook.Library.library_rag_score_kinds import (
        library_rag_result_score_kind,
    )
    from tldw_chatbook.Library.library_rag_state import library_rag_score_suffix

    reranker = PointwiseReranker(_degraded_config("pointwise"))
    _install_fake_provider(
        monkeypatch, _fail_for_titles(["doc-0", "doc-2", "doc-4"], score=0.95)
    )

    originals = {r.id: r for r in _results(5)}
    outcome = await reranker.rerank("quokka marsupial", list(originals.values()))

    assert (outcome.failed, outcome.total) == (3, 5)
    by_id = {r.id: r for r in outcome.results}
    assert set(by_id) == set(originals)

    for failed_id in ("doc-0", "doc-2", "doc-4"):
        row = by_id[failed_id]
        assert "rerank_score" not in row.metadata, (
            f"{failed_id} was never rescored but claims the reranker marker"
        )
        assert row.score == originals[failed_id].score
        kind, vector_score = library_rag_result_score_kind(row.metadata)
        assert kind == "vector_similarity"
        suffix = library_rag_score_suffix(
            row.score, score_kind=kind, vector_score=vector_score
        )
        assert suffix != " | reranked"
        assert suffix.startswith(" | match:")

    for scored_id in ("doc-1", "doc-3"):
        row = by_id[scored_id]
        assert row.metadata["rerank_score"] == 0.95
        kind, vector_score = library_rag_result_score_kind(row.metadata)
        assert kind == "reranker"
        assert (
            library_rag_score_suffix(
                row.score, score_kind=kind, vector_score=vector_score
            )
            == " | reranked"
        )


@pytest.mark.asyncio
async def test_failed_pointwise_rows_keep_their_kind_through_the_cache_hit(monkeypatch):
    """The reranker caches its per-result `RerankingResult`s, so a cache HIT
    re-applies them to a fresh results list. The "was this row actually
    scored?" fact has to survive that round trip, or the second identical
    search re-acquires the over-claim."""
    reranker = PointwiseReranker(_degraded_config("pointwise"))
    calls = _install_fake_provider(monkeypatch, _fail_for_titles(["doc-0"], score=0.8))

    first = await reranker.rerank("q", _results(3))
    assert first.failed == 1
    # The PROVIDER-call log, not `len(self._cache)`: a miss rewrites the same
    # key for the same query+ids, so the cache's LENGTH is identical either
    # way and cannot tell a hit from a miss (final-review F3).
    calls_before = len(calls)
    assert calls_before, "the first pass must really have called the fake provider"

    second = await reranker.rerank("q", _results(3))
    assert len(calls) == calls_before, (
        "expected a cache HIT: the second identical rerank must not re-call "
        f"the provider (extra calls: {calls[calls_before:]})"
    )

    by_id = {r.id: r for r in second.results}
    assert "rerank_score" not in by_id["doc-0"].metadata
    assert by_id["doc-1"].metadata["rerank_score"] == 0.8


# --------------------------------------------------------------------------
# AC#4 -- per-call counts, scoped by return value
# --------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["pointwise", "pairwise", "listwise"])
@pytest.mark.asyncio
async def test_rerank_returns_its_own_failure_counts(strategy, monkeypatch):
    """Every strategy reports its outcome in its RETURN value."""
    cls = {
        "pointwise": PointwiseReranker,
        "pairwise": PairwiseReranker,
        "listwise": ListwiseReranker,
    }[strategy]
    reranker = cls(_degraded_config(strategy))
    _install_fake_provider(monkeypatch, _always_fails)

    outcome = await reranker.rerank("q", _results(3))

    assert [r.id for r in outcome.results] == ["doc-0", "doc-1", "doc-2"]
    assert outcome.total > 0
    assert outcome.failed == outcome.total
    assert outcome.degraded is True


@pytest.mark.asyncio
async def test_reranker_keeps_no_cross_call_failure_state(monkeypatch):
    """The structural half of AC#4: there is no shared instance state left for
    a concurrent search to corrupt. The counts a caller reads are the ones
    that call returned -- not a field another coroutine/thread may have
    overwritten between `rerank()` returning and the disclosure being built."""
    reranker = PointwiseReranker(_degraded_config("pointwise"))
    _install_fake_provider(monkeypatch, _always_fails)

    await reranker.rerank("q", _results(2))

    for leaked in ("last_rerank_failures", "last_rerank_total"):
        assert not hasattr(reranker, leaked), (
            f"{leaked} is cross-call instance state on a shared singleton"
        )
    assert not hasattr(PointwiseReranker, "_record_rerank_outcome")

    pairwise = PairwiseReranker(_degraded_config("pairwise"))
    _install_fake_provider(monkeypatch, _always_fails)
    await pairwise.rerank("q", _results(3))
    for leaked in ("_pairwise_comparisons_failed", "_pairwise_comparisons_total"):
        assert not hasattr(pairwise, leaked), (
            f"{leaked} is cross-call instance state on a shared singleton"
        )


@pytest.mark.asyncio
async def test_disclosure_survives_a_concurrent_write_in_its_own_window(
    tmp_path, monkeypatch
):
    """A concurrent search finishing INSIDE this search's window used to
    rewrite the shared counters before this search's tag was built, so the
    disclosure described the other search. A thread-scheduling race cannot be
    asserted deterministically, so the interfering write is landed exactly
    where a second `rerank()` completing would land it (after this call has
    finished counting, before its caller builds the tag). With the counts
    returned rather than stored, that write is inert."""
    service = _make_v2_service(tmp_path, "pointwise", enable_cache=False)
    reranker = service.reranker
    _install_fake_provider(monkeypatch, _always_fails)
    await service.index_batch_optimized(_quokka_platypus_docs())

    real_log = reranker._log_reranking_metrics

    def _log_and_interfere(reranking_results):
        # What a concurrent, fully-successful second search's bookkeeping
        # used to do to this call's disclosure.
        reranker.last_rerank_failures = 0
        reranker.last_rerank_total = 0
        return real_log(reranking_results)

    monkeypatch.setattr(reranker, "_log_reranking_metrics", _log_and_interfere)

    results = await service.search(
        "quokka marsupial", top_k=5, search_type="semantic", include_citations=False
    )

    assert results[0].metadata.get("reranking_degraded") == "2/2 scorings failed", (
        "the disclosure described another call's counts"
    )


# --------------------------------------------------------------------------
# service scaffolding (mock embeddings, in-memory store -- offline)
# --------------------------------------------------------------------------


def _quokka_platypus_docs():
    return [
        {
            "id": "doc-1",
            "content": "The quokka is a small marsupial found on Rottnest Island.",
            "title": "Quokka Facts",
        },
        {
            "id": "doc-2",
            "content": "Platypuses are egg-laying mammals native to eastern Australia.",
            "title": "Platypus Facts",
        },
    ]


def _make_v2_service(tmp_path, strategy: str, enable_cache: bool):
    """An `EnhancedRAGServiceV2` whose profile actually carries a reranking
    config of `strategy` -- the construction path `test_reranker_construction`
    pins, parametrized by strategy."""
    manager = get_profile_manager(profiles_dir=tmp_path)

    rag_cfg = RAGConfig()
    rag_cfg.embedding.model = "mock"  # deterministic bag-of-words, offline
    rag_cfg.embedding.device = "cpu"
    rag_cfg.vector_store.type = "memory"
    rag_cfg.vector_store.persist_directory = None
    rag_cfg.chunking.chunk_size = 60
    rag_cfg.chunking.chunk_overlap = 10
    rag_cfg.search.enable_cache = enable_cache

    profile = ProfileConfig(
        name=f"test_rerank_{strategy}_cache_{enable_cache}",
        description=f"test profile with {strategy} reranking enabled",
        profile_type="balanced",
        rag_config=rag_cfg,
        reranking_config=_degraded_config(strategy),
    )
    manager.save_profile(profile)

    return EnhancedRAGServiceV2(
        config=profile.id,
        profile_manager=manager,
        enable_parent_retrieval=False,
        enable_reranking=True,
        enable_parallel_processing=False,
    )


# ---------------------------------------------------------------------------
# The seam contract itself (Qodo PR-1751 finding 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reranker_dispatch_binding_against_the_real_chat_api_call_signature(
    monkeypatch,
):
    """PROOF OF REPAIR (TASK-17065) -- the CORRECT binding, observed live.

    This test was written red-on-repair by Qodo PR-1751 finding 3: it used to
    assert the BROKEN landing (the credential in `api_endpoint`, every later
    argument displaced by one) reconstructed from the caller's source. The
    repair flips it, and this is the arc's proof.

    Two things changed beyond the assertions. It now drives the REAL caller
    (`_call_llm_impl`) instead of re-typing its argument list, so it cannot
    drift from the code it guards; and the seam fake binds through
    `inspect.signature(chat_api_call).bind(...)`, so a future positional
    mis-order raises instead of being agreed with.
    """
    config = RerankingConfig(
        model_provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.25,
        max_tokens=128,
    )
    reranker = PointwiseReranker(config)
    landings = _install_fake_provider(monkeypatch)

    await reranker._call_llm_impl("score this document")

    assert len(landings) == 1
    landed = landings[0]
    # The three the config owns land where the signature says they belong.
    assert landed["api_endpoint"] == "openai"
    assert landed["model"] == "gpt-4o-mini"
    assert landed["temp"] == 0.25
    assert landed["max_tokens"] == 128
    assert landed["messages_payload"][-1] == {
        "role": "user",
        "content": "score this document",
    }
    # No argument carries a credential: the reranker resolves none, and each
    # `chat_with_<provider>` handler resolves its own from the normalised
    # config path (CLAUDE.md's documented precedence) -- which is what every
    # other `chat_api_call` caller in this repo already relies on.
    assert "api_key" not in landed
    # And nothing lands in the parameters the broken positional call filled
    # by displacement.
    assert "system_message" not in landed
    assert "streaming" not in landed


def test_reranker_does_not_read_a_settings_table(monkeypatch):
    """AC#2: the phantom `self._settings["API"]` read is GONE, not repaired.

    `BaseReranker.__init__` used to do `self._settings = load_settings()` and
    `_call_llm_impl` read `self._settings["API"]["<p>_api_key"]` -- a table
    `load_settings()` never builds, so three of its four provider branches
    read `None` for every user, always. The fix deletes the read rather than
    re-pointing it: credential resolution is not the reranker's job.
    """
    assert not hasattr(reranker_module, "load_settings"), (
        "reranker.py must not import load_settings -- a settings read here is "
        "the divergence that produced TASK-17065"
    )

    def _explode():
        raise AssertionError("BaseReranker must not call load_settings()")

    monkeypatch.setattr(reranker_module, "load_settings", _explode, raising=False)

    reranker = PointwiseReranker(RerankingConfig())

    assert not hasattr(reranker, "_settings")


@pytest.mark.parametrize(
    "provider",
    [
        # keyless locals -- AC#5: never rejected for a key they do not need
        "ollama",
        "llama_cpp",
        "vllm",
        "koboldcpp",
        "mlx_lm",
        # remotes whose handlers resolve their own credential -- AC#4
        "openai",
        "anthropic",
        "groq",
        "deepseek",
    ],
)
@pytest.mark.asyncio
async def test_every_sampled_provider_reaches_the_seam_without_a_credential_gate(
    monkeypatch, provider
):
    """AC#4/AC#5: the reranker dispatches; it does not judge credentials.

    Sampled across the 29 rows of `API_CALL_HANDLERS` the picker enumerates:
    the five keyless locals TASK-17065 names, plus the four remotes the old
    hand-rolled `if/elif` claimed to cover. Every one used to die before the
    seam with `No API key found for provider: X` (three of them because the
    table they read does not exist); every one now arrives at `chat_api_call`
    with its own name in `api_endpoint`.
    """
    reranker = PointwiseReranker(
        RerankingConfig(model_provider=provider, model_name="the-model")
    )
    landings = _install_fake_provider(monkeypatch)

    response = await reranker._call_llm_impl("score this document")

    assert response == '{"score": 0.5}'
    assert [landed["api_endpoint"] for landed in landings] == [provider]
    assert landings[0]["model"] == "the-model"
    assert "api_key" not in landings[0]
