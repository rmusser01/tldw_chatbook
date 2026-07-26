import os
from pathlib import Path

import pytest
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig


def _wire(monkeypatch, tmp_path):
    """Wire a real ConfigProfileManager plus fakes for the active-profile
    pointer and the task-639 first-run-import-done marker.

    `ptr["v"]` is the pointer ([rag.service].profile); `ptr["marker"]` is the
    durable first-run marker ([rag.service].first_run_import_done, see
    `_first_run_import_done`/`_mark_first_run_import_done`). Both are backed
    by the SAME fake `save_setting_to_cli_config`, routed by `key` exactly
    like the real TOML writes would be -- this catches a real bug class
    (e.g. one write silently clobbering the other's tracked value) that a
    single shared dict/lambda would have masked.
    """
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    ptr = {"v": None, "marker": False}
    monkeypatch.setattr(ac, "_active_profile_id", lambda: ptr["v"] or "hybrid_basic", raising=False)
    monkeypatch.setattr(ac, "_first_run_import_done", lambda: ptr["marker"], raising=False)

    def _fake_save_setting(section, key, value):
        if section == "rag.service" and key == "profile":
            ptr["v"] = value
        elif section == "rag.service" and key == "first_run_import_done":
            ptr["marker"] = value
        return True

    monkeypatch.setattr(ac, "save_setting_to_cli_config", _fake_save_setting, raising=False)
    monkeypatch.setattr(ac, "reset_shared_rag_service", lambda: None, raising=False)
    # task-635: ensure_imported_profile() now only imports when there is
    # genuine hand-set [AppRAGSearchConfig.rag.*] material (see
    # _has_legacy_rag_config_material). Default every test to the
    # no-legacy-material baseline (deterministic, not incidentally-empty via
    # the real isolated test config.toml) -- tests that need to exercise the
    # "legacy upgrader" path call `_wire_legacy_rag_config` themselves,
    # afterwards, to override this with real content.
    _wire_legacy_rag_config(monkeypatch, {})
    return mgr, ptr


def test_first_run_creates_imported_profile_and_sets_active(monkeypatch, tmp_path):
    """A legacy upgrader (has hand-set [AppRAGSearchConfig.rag.*] material)
    gets the first-run "Imported settings" profile created and activated."""
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {"search": {"default_top_k": 10}})
    new_id = ensure_imported_profile()
    assert new_id is not None
    imported = mgr.get_profile(new_id)
    assert imported is not None and imported.read_only is False
    assert ptr["v"] == new_id  # set active
    assert ptr["marker"] is True  # task-639: first-run-import-done marker recorded
    # Idempotent: a second call is a no-op (no duplicate).
    assert ensure_imported_profile() is None


def test_imported_fingerprint_matches_sp1_adoption(monkeypatch, tmp_path):
    """Cross-SP invariant: the imported profile's fingerprint == the fingerprint
    SP1 would adopt the legacy 'default' collection under (both from the same
    first-run resolved config), so an upgraded user keeps their index.

    The pre-import fingerprint MUST be captured before ensure_imported_profile()
    runs — once it runs, the active pointer is repointed at the new profile, so
    resolving "the active config" afterwards just re-reads the profile we're
    trying to verify (tautological). The `_wire` fixture's active pointer
    starts at the builtin "hybrid_basic" profile (embedding.model
    "all-MiniLM-L6-v2", chunk_size 384), which is deliberately NOT the same as
    a bare RAGConfig() default (embedding.model "mxbai-embed-large-v1",
    chunk_size 400) — so a snapshot that silently fell back to bare defaults
    would fingerprint differently and this test would catch it.
    """
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import fingerprint_collection
    mgr, ptr = _wire(monkeypatch, tmp_path)
    # task-635: fingerprint continuity is only meaningful for a legacy
    # upgrader (a fresh install has no pre-profile collection to preserve).
    _wire_legacy_rag_config(monkeypatch, {"search": {"default_top_k": 10}})
    # Capture what SP1 would adopt the legacy collection under, resolved via
    # the ORIGINAL active pointer, BEFORE ensure_imported_profile() mutates it.
    pre_fp = fingerprint_collection(resolve_active_rag_config())
    new_id = ensure_imported_profile()
    imported_fp = fingerprint_collection(mgr.get_profile(new_id).rag_config)
    assert imported_fp == pre_fp


@pytest.mark.parametrize(
    "env_var,env_value,attr_path",
    [
        ("RAG_EMBEDDING_MODEL", "distinctive-env-model", "embedding.model"),
        ("RAG_CHUNK_SIZE", "999", "chunking.chunk_size"),
    ],
)
def test_imported_fingerprint_matches_sp1_adoption_with_env_override(
    monkeypatch, tmp_path, env_var, env_value, attr_path
):
    """Cross-SP invariant, env-divergence case: the same guarantee as
    test_imported_fingerprint_matches_sp1_adoption but with a fingerprint-
    affecting env var set (RAG_EMBEDDING_MODEL / RAG_CHUNK_SIZE), the one case
    that could actually orphan an index -- if the imported snapshot dropped
    the env-applied layer, the imported profile's fingerprint would silently
    diverge from what SP1 adopted the legacy collection under.

    Also asserts the snapshot's resolved value equals the env value directly:
    this documents that env IS captured into the imported profile (the import
    reflects resolve_active_rag_config(), not a bare base profile)."""
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import fingerprint_collection
    mgr, ptr = _wire(monkeypatch, tmp_path)
    # task-635: fingerprint continuity is only meaningful for a legacy
    # upgrader (a fresh install has no pre-profile collection to preserve).
    _wire_legacy_rag_config(monkeypatch, {"search": {"default_top_k": 10}})
    monkeypatch.setenv(env_var, env_value)
    # Capture BEFORE ensure_imported_profile() repoints the active pointer --
    # same ordering rationale as test_imported_fingerprint_matches_sp1_adoption.
    pre_fp = fingerprint_collection(resolve_active_rag_config())
    new_id = ensure_imported_profile()
    imported_cfg = mgr.get_profile(new_id).rag_config
    assert fingerprint_collection(imported_cfg) == pre_fp
    resolved = imported_cfg
    for part in attr_path.split("."):
        resolved = getattr(resolved, part)
    expected = int(env_value) if attr_path == "chunking.chunk_size" else env_value
    assert resolved == expected


def test_ensure_imported_profile_heals_half_done_first_run(monkeypatch, tmp_path):
    """If a prior first run persisted the imported profile but crashed before
    (or otherwise failed to) flip the active pointer to it, the guard must not
    be "does the profile exist" alone — it must also heal the pointer, since
    otherwise the profile is created-but-never-activated forever."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)
    half_done = ProfileConfig(id=ac._IMPORTED_ID,
                              name="Imported settings",
                              description="Captured from your existing RAG configuration on first run.",
                              profile_type="custom",
                              rag_config=ac.resolve_active_rag_config())
    mgr.save_profile(half_done)
    assert ptr["v"] is None  # pointer was never flipped -- simulates the half-done crash
    assert ptr["marker"] is False  # task-639: no marker either -- the new gate condition

    result = ac.ensure_imported_profile()

    assert result is None  # idempotent: no new profile id returned
    assert [p for p in mgr.list_profiles() if p == ac._IMPORTED_ID] == [ac._IMPORTED_ID]  # no duplicate created
    assert ptr["v"] == ac._IMPORTED_ID  # healed: pointer now activates the existing profile
    assert ptr["marker"] is True  # task-639: marker recorded, so this can't recur


def test_ensure_imported_profile_swallows_save_failure(monkeypatch, tmp_path):
    """Exception-safety: any failure while creating the imported profile must
    be swallowed (logged, not raised) so it can never block RAG service
    creation, and must not leave a half-activated pointer behind."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)
    # task-635: exercise the real save_profile attempt, which only happens
    # for a legacy upgrader (fresh installs short-circuit before ever
    # calling save_profile, see test_fresh_user_* below).
    _wire_legacy_rag_config(monkeypatch, {"search": {"default_top_k": 10}})

    def _boom(profile):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mgr, "save_profile", _boom)

    result = ac.ensure_imported_profile()

    assert result is None
    assert mgr.get_profile(ac._IMPORTED_ID) is None
    assert ptr["v"] is None  # never activated a profile that failed to save
    assert ptr["marker"] is False  # task-639: no activation happened, so no marker either


# --- Task 6: wiring into get_shared_rag_service --------------------------


@pytest.fixture
def _reset_first_run_wiring():
    """Isolate the module-level once-flag + shared-service singleton around
    each wiring test so they can't leak into each other or into unrelated
    RAG tests that call get_shared_rag_service()."""
    import tldw_chatbook.RAG_Search.ingestion_indexing as ii
    ii._first_run_import_attempted = False
    ii.reset_shared_rag_service()
    yield ii
    ii._first_run_import_attempted = False
    ii.reset_shared_rag_service()


def test_get_shared_rag_service_calls_first_run_import_at_most_once(
    monkeypatch, _reset_first_run_wiring
):
    """Task-6 wiring: get_shared_rag_service must attempt the first-run
    "Imported settings" capture exactly once per process, no matter how many
    times it (or callers of it) run.

    The real service is pre-injected via set_shared_rag_service so this
    exercises get_shared_rag_service's fast path (no real RAGService build),
    isolating the assertion to the wiring/once-guard behavior itself.
    """
    ii = _reset_first_run_wiring
    import tldw_chatbook.RAG_Search.simplified.active_config as ac

    calls = []
    monkeypatch.setattr(ac, "ensure_imported_profile", lambda: calls.append(1) or None)
    # No test-mode skip to bypass anymore (task-519 removed the
    # PYTEST_CURRENT_TEST guard): the _reset_first_run_wiring fixture already
    # resets _first_run_import_attempted so this exercises the real path.

    fake_service = object()
    ii.set_shared_rag_service(fake_service)

    assert ii.get_shared_rag_service() is fake_service
    assert ii.get_shared_rag_service() is fake_service
    assert ii.get_shared_rag_service() is fake_service

    assert len(calls) == 1


def test_first_run_import_runs_before_shared_service_lock_is_held(
    monkeypatch, _reset_first_run_wiring
):
    """Deadlock regression guard (task-6 hazard #1): ensure_imported_profile
    can call set_active_profile -> reset_shared_rag_service, which
    re-acquires the module-level, non-reentrant _shared_service_lock. If the
    wiring call were moved inside `with _shared_service_lock:`, this would
    self-deadlock in production.

    Proven directly (not just "the test completed"): the fake
    ensure_imported_profile tries to acquire the SAME lock object with a
    short timeout. If the wiring call happened while the lock was already
    held, the acquire would time out and this test would fail fast (2s)
    instead of hanging the suite.
    """
    ii = _reset_first_run_wiring
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    import tldw_chatbook.RAG_Search.simplified as simplified_pkg

    acquired = []

    def _fake_ensure_imported_profile():
        got = ii._shared_service_lock.acquire(timeout=2)
        if got:
            ii._shared_service_lock.release()
        acquired.append(got)

    monkeypatch.setattr(ac, "ensure_imported_profile", _fake_ensure_imported_profile)
    # No test-mode skip to bypass anymore (task-519 removed the
    # PYTEST_CURRENT_TEST guard): the _reset_first_run_wiring fixture already
    # resets _first_run_import_attempted so this exercises the real path.
    # Avoid a real (possibly slow/dependency-gated) RAGService build -- this
    # test is only exercising the lock-acquisition ordering, not construction.
    monkeypatch.setattr(simplified_pkg, "create_rag_service", lambda **kwargs: object())
    # resolve_active_rag_config() is evaluated eagerly (as a call argument)
    # inside the lock block, BEFORE the faked create_rag_service runs, so it
    # must be faked too -- otherwise it hits the real _manager() ->
    # ConfigProfileManager(), whose __init__ does a real, silent
    # (exist_ok=True) mkdir() of ~/.local/share/tldw_cli/.../rag_profiles on
    # whatever machine runs this test. This test only cares about lock
    # ordering, not config resolution, so a trivial stand-in is fine.
    monkeypatch.setattr(ac, "resolve_active_rag_config", lambda **kwargs: object())

    ii.get_shared_rag_service()  # no service pre-injected: exercises the real fast-path+lock flow up to construction

    assert acquired == [True]  # lock was free (not self-deadlocked) when the wiring call ran


# --- Task-495: merge legacy query-time keys into the first-run snapshot --


def _wire_legacy_rag_config(monkeypatch, rag_section):
    """Monkeypatch active_config.get_cli_setting so
    get_cli_setting("AppRAGSearchConfig", "rag", {}) resolves to `rag_section`
    (shaped like the raw [AppRAGSearchConfig.rag.<subsection>].<key> TOML
    tree), independent of whatever the test-bootstrap config on disk has."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac

    def _fake_get_cli_setting(section, key=None, default=None):
        if section == "AppRAGSearchConfig" and key == "rag":
            return rag_section
        return default

    monkeypatch.setattr(ac, "get_cli_setting", _fake_get_cli_setting, raising=False)


def test_imported_profile_preserves_hand_set_legacy_query_time_keys(monkeypatch, tmp_path):
    """AC #1: hand-set legacy query-time keys (top_k, score_threshold,
    citations, reranking) from [AppRAGSearchConfig.rag.search] /
    [AppRAGSearchConfig.rag.processor] survive into the imported profile
    instead of being silently discarded."""
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {
        "search": {"default_top_k": 25, "score_threshold": 0.42, "include_citations": False},
        "processor": {"enable_reranking": True, "reranker_model": "cross-encoder/legacy", "reranker_top_k": 7},
    })

    new_id = ensure_imported_profile()

    imported_profile = mgr.get_profile(new_id)
    imported = imported_profile.rag_config
    assert imported.search.default_top_k == 25
    assert imported.search.score_threshold == 0.42
    assert imported.search.include_citations is False
    assert imported.search.enable_reranking is True
    assert imported.search.reranker_model == "cross-encoder/legacy"
    assert imported.search.reranker_top_k == 7
    # PR #874 Qodo finding 2: rag_factory.create_rag_service() decides
    # reranking enablement from `profile.reranking_config is not None`
    # (rag_factory.py:65), NOT `rag_config.search.enable_reranking` -- that
    # field only mirrors for UI display (see settings_rag_profile_adapter.py
    # apply_defaults_to_profile). A legacy user with reranking enabled must
    # therefore also get a populated `reranking_config`, or reranking stays
    # silently OFF for them after import despite `search.enable_reranking`
    # being True above.
    assert imported_profile.reranking_config is not None
    assert imported_profile.reranking_config.model_name == "cross-encoder/legacy"
    assert imported_profile.reranking_config.top_k_to_rerank == 7


def test_imported_profile_reranking_config_absent_when_legacy_reranking_unset(monkeypatch, tmp_path):
    """No legacy `enable_reranking` key set -> no `reranking_config` is
    fabricated (mirrors today's "nothing to merge" behavior for the other
    legacy keys)."""
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {
        "search": {"default_top_k": 25},
    })

    new_id = ensure_imported_profile()

    assert mgr.get_profile(new_id).reranking_config is None


def test_imported_profile_reranking_config_keeps_model_default_when_legacy_model_blank(monkeypatch, tmp_path):
    """Legacy reranking enabled but no `reranker_model` hand-set -> the
    fabricated `RerankingConfig`'s `model_name` keeps its own default rather
    than being stomped with an empty/missing value (same "blank means leave
    alone" convention as `apply_defaults_to_profile`)."""
    from tldw_chatbook.RAG_Search.reranker import RerankingConfig
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {
        "processor": {"enable_reranking": True, "reranker_top_k": 12},
    })

    new_id = ensure_imported_profile()

    imported_profile = mgr.get_profile(new_id)
    assert imported_profile.reranking_config is not None
    assert imported_profile.reranking_config.model_name == RerankingConfig().model_name
    assert imported_profile.reranking_config.top_k_to_rerank == 12


def test_imported_profile_fingerprint_invariant_with_legacy_query_keys_set(monkeypatch, tmp_path):
    """AC #2 (verbatim): with those same legacy query-time keys set, the
    imported profile's fingerprint still equals the fingerprint of the
    unmodified built-in base (SP1's adopted legacy-collection fingerprint) --
    merging query-time-only fields must never move the fingerprint."""
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import fingerprint_collection
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {
        "search": {"default_top_k": 25, "score_threshold": 0.42, "include_citations": False},
        "processor": {"enable_reranking": True, "reranker_model": "cross-encoder/legacy", "reranker_top_k": 7},
    })
    # Captured BEFORE ensure_imported_profile() repoints the active pointer --
    # same ordering rationale as test_imported_fingerprint_matches_sp1_adoption.
    pre_fp = fingerprint_collection(resolve_active_rag_config())

    new_id = ensure_imported_profile()

    imported_fp = fingerprint_collection(mgr.get_profile(new_id).rag_config)
    assert imported_fp == pre_fp


def test_imported_profile_unchanged_when_no_legacy_query_time_keys_set(monkeypatch, tmp_path):
    """A legacy upgrader with SOME hand-set [AppRAGSearchConfig.rag.*]
    material (so import still happens, task-635), but none of it in the
    query-time allow-list -> the imported snapshot is byte-equal to today's
    plain resolve_active_rag_config() capture (no regression when there's
    nothing to merge). Uses a non-query-time legacy key (embedding) purely
    as a "genuine legacy user" presence signal -- it is never merged, so it
    must not affect the captured snapshot either."""
    from dataclasses import asdict
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {"embedding": {"model": "all-MiniLM-L6-v2"}})
    expected = resolve_active_rag_config()

    new_id = ensure_imported_profile()

    imported = mgr.get_profile(new_id).rag_config
    assert asdict(imported) == asdict(expected)


def test_fresh_user_no_legacy_rag_config_stays_on_default_builtin(monkeypatch, tmp_path):
    """task-635: a truly fresh install (no [AppRAGSearchConfig.rag.*]
    material at all) must NOT get an auto-created + auto-activated
    "Imported settings" profile. There is no legacy pre-profile collection
    to preserve continuity for, so ensure_imported_profile() is a no-op and
    the active pointer is never written."""
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, _IMPORTED_ID
    mgr, ptr = _wire(monkeypatch, tmp_path)
    # _wire() already defaults to no legacy material; assert that
    # explicitly so this test documents the exact scenario it covers.
    _wire_legacy_rag_config(monkeypatch, {})

    result = ensure_imported_profile()

    assert result is None
    assert mgr.get_profile(_IMPORTED_ID) is None
    assert ptr["v"] is None  # pointer never written -- stays on the default builtin


def test_fresh_user_unreadable_legacy_section_stays_on_default_builtin(monkeypatch, tmp_path):
    """Exception-safety parity with _hand_set_legacy_query_time_keys: if the
    legacy section can't be read at all (e.g. a non-dict value under
    [AppRAGSearchConfig.rag], or get_cli_setting raising), that must be
    treated the same as "no legacy material" -- never as a reason to import
    anyway -- so a fresh user is never surprised by a profile creation
    triggered by a config-read failure."""
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, _IMPORTED_ID
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, "not-a-dict")

    result = ensure_imported_profile()

    assert result is None
    assert mgr.get_profile(_IMPORTED_ID) is None
    assert ptr["v"] is None


def test_imported_profile_does_not_merge_legacy_index_determining_keys(monkeypatch, tmp_path):
    """Legacy embedding/chunking/distance_metric keys must NEVER be merged
    (that would move the fingerprint and orphan the legacy collection) --
    only the allow-listed query-time keys are eligible, and the fingerprint
    stays equal to the unmodified built-in base's even when those
    index-determining legacy keys are hand-set."""
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import fingerprint_collection
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {
        "embedding": {"model": "legacy-embedding-model", "max_length": 9999},
        "chunking": {"chunk_size": 1, "chunk_overlap": 0},
        "vector_store": {"distance_metric": "l2"},
        "search": {"default_top_k": 25},
    })
    pre_fp = fingerprint_collection(resolve_active_rag_config())

    new_id = ensure_imported_profile()

    imported = mgr.get_profile(new_id).rag_config
    assert imported.embedding.model != "legacy-embedding-model"
    assert imported.chunking.chunk_size != 1
    assert imported.vector_store.distance_metric != "l2"
    assert imported.search.default_top_k == 25  # query-time key still merged
    assert fingerprint_collection(imported) == pre_fp


# --- Task-639: healing must not undo a deliberate profile switch, and must --
# --- never delete a profile on a content guess (review: marker-based fix) --


def test_ensure_imported_profile_does_not_reflip_deliberate_switch(monkeypatch, tmp_path):
    """AC #1: once a user has deliberately activated a DIFFERENT profile
    (neither the default builtin nor imported_settings), a later
    ensure_imported_profile() call (e.g. the next process's first RAG touch)
    must leave that pointer alone -- the old healing branch treated "pointer
    != imported_settings" as proof of a half-done first run, which silently
    undid a real user choice on every subsequent launch. The fresh import
    above already recorded the first-run-import-done marker, so the healing
    branch's "not marker" gate is skipped entirely on the later call."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {"search": {"default_top_k": 10}})

    new_id = ac.ensure_imported_profile()
    assert ptr["v"] == new_id  # sanity: import activated it, as before
    assert ptr["marker"] is True

    # The user later, deliberately, switches to a different builtin.
    ptr["v"] = "fast_search"

    result = ac.ensure_imported_profile()

    assert result is None  # idempotent, no duplicate profile
    assert ptr["v"] == "fast_search"  # NOT flipped back to imported_settings


def test_ensure_imported_profile_still_heals_when_pointer_is_default(monkeypatch, tmp_path):
    """AC #3 (restated for task-639's narrower gate): the healing branch must
    still fire for the one case it exists for -- no marker recorded yet AND
    the pointer never having been successfully written by anyone, i.e. it
    still names the default builtin. This is the same scenario
    test_ensure_imported_profile_heals_half_done_first_run covers; kept here
    too as an explicit task-639 lock on the new gating condition itself."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)
    half_done = ProfileConfig(id=ac._IMPORTED_ID,
                              name="Imported settings",
                              description="Captured from your existing RAG configuration on first run.",
                              profile_type="custom",
                              rag_config=ac.resolve_active_rag_config())
    mgr.save_profile(half_done)
    assert ptr["v"] is None  # pointer still resolves to the default builtin
    assert ptr["marker"] is False

    result = ac.ensure_imported_profile()

    assert result is None
    assert ptr["v"] == ac._IMPORTED_ID  # healed
    assert ptr["marker"] is True  # recorded, so this can never recur


def test_ensure_imported_profile_marker_present_never_touches_pointer_even_back_to_default(monkeypatch, tmp_path):
    """Closes the previously-disclosed gap: once the marker is set, the
    pointer is NEVER touched again -- including a user switching all the way
    back to the default builtin itself, which is otherwise indistinguishable
    from "never written" using the pointer alone. With the marker recording
    that the activation was deliberate, that ambiguity no longer matters:
    the healing branch is skipped entirely regardless of what the pointer
    currently names."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)
    imported = _pre635_damage_snapshot(mgr, ac)
    mgr.save_profile(imported)
    ptr["marker"] = True  # a prior run already recorded a deliberate activation
    ptr["v"] = None  # the user has since switched all the way back to the default builtin

    result = ac.ensure_imported_profile()

    assert result is None
    assert ptr["v"] is None  # NOT flipped back to imported_settings -- stays on the default
    assert mgr.get_profile(ac._IMPORTED_ID) is not None  # untouched either way


def test_fresh_user_leaves_marker_unset(monkeypatch, tmp_path):
    """A truly fresh install (task-635 no-op path) never creates the imported
    profile, so there is nothing to mark done either -- the marker stays
    absent. This must not, by itself, trigger any different behavior on a
    later call (still a no-op, still no marker)."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)

    result = ac.ensure_imported_profile()

    assert result is None
    assert ptr["marker"] is False
    assert ptr["v"] is None
    # Idempotent, still no marker.
    assert ac.ensure_imported_profile() is None
    assert ptr["marker"] is False


def _pre635_damage_snapshot(mgr, ac):
    """A profile shaped exactly like what the pre-635 always-import bug would
    have created+activated for a truly fresh install: a snapshot of the
    default builtin (the active pointer a fresh user starts with), no legacy
    merge, no reranking. Also stands in generically for "an imported_settings
    profile that predates the first-run-import-done marker" in the tests
    below, whether or not its content happens to look default-equivalent --
    the marker-based fix no longer cares which."""
    return ProfileConfig(
        id=ac._IMPORTED_ID,
        name="Imported settings",
        description="Snapshot of your active RAG profile (plus any RAG_* env "
                    "overrides) captured on first run; edit freely.",
        profile_type="custom",
        rag_config=ac.resolve_active_rag_config(),  # pointer is still default here
    )


def test_ensure_imported_profile_never_deletes_settings_screen_customization(monkeypatch, tmp_path):
    """Critical (task-639 review, reviewer-reproduced): the Settings screen
    editor (apply_defaults_to_profile, settings_rag_profile_adapter.py:150-
    167) can hand-tune SearchConfig fields the prior fingerprint + allow-list
    damage heuristic never checked at all -- e.g. `search.hybrid_alpha`
    (default 0.7), `default_search_mode`, `citation_style`,
    `snippet_max_chars`, `max_context_size`, `fts_top_k`, `vector_top_k`, or
    `embedding.batch_size`. None of those are index-determining (the
    fingerprint still matched the default) and none were in the old
    allow-list, so a profile with ONLY one of these fields customized still
    looked "provably safe to delete" and was PERMANENTLY DESTROYED. No finite
    content allow-list can close this for good -- there is always another
    field it doesn't know about -- so the fix must never delete a profile
    based on comparing its content at all, no matter what differs (or
    doesn't). "Survives every subsequent ensure call": exercised 3x."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)
    customized = _pre635_damage_snapshot(mgr, ac)
    customized.rag_config.search.hybrid_alpha = 0.95  # real hand-set tuning, default is 0.7
    mgr.save_profile(customized)
    ptr["v"] = ac._IMPORTED_ID  # already active; no marker (predates the marker fix)
    assert ptr["marker"] is False

    for _ in range(3):
        result = ac.ensure_imported_profile()
        assert result is None

    imported = mgr.get_profile(ac._IMPORTED_ID)
    assert imported is not None  # NEVER deleted
    assert imported.rag_config.search.hybrid_alpha == 0.95  # customization intact
    assert ptr["v"] == ac._IMPORTED_ID  # left active
    assert ptr["marker"] is True  # adopted as settled after the first call


def test_ensure_imported_profile_adopts_preexisting_imported_pointer_without_deleting(monkeypatch, tmp_path):
    """task-639 AC #2 (revised per review): a config whose pointer is ALREADY
    imported_settings with no marker yet -- whether a pre-635 damage artifact
    or a genuine, deliberately-kept import from before the marker existed,
    indistinguishable from config contents alone (see the Critical finding
    above) -- is NEVER deleted, even when its content happens to look
    identical to the default builtin. It is simply adopted as settled: the
    marker is set, and both the profile and the pointer are left completely
    untouched."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)
    imported = _pre635_damage_snapshot(mgr, ac)  # content-identical to the default builtin
    mgr.save_profile(imported)
    ptr["v"] = ac._IMPORTED_ID  # already active, no marker (pre-marker world)
    assert ptr["marker"] is False

    result = ac.ensure_imported_profile()

    assert result is None
    assert ptr["v"] == ac._IMPORTED_ID  # untouched
    assert mgr.get_profile(ac._IMPORTED_ID) is not None  # NOT deleted
    assert ptr["marker"] is True  # adopted as settled -- never re-evaluated again

    # Idempotent: a further call changes nothing further.
    assert ac.ensure_imported_profile() is None
    assert ptr["v"] == ac._IMPORTED_ID
    assert mgr.get_profile(ac._IMPORTED_ID) is not None


# --- Task-639 review round 3: don't mark "done" on an unconfirmed pointer --
# --- write (set_active_profile swallows a failed write) --------------------


def _wire_pointer_write_fails(monkeypatch, ac, ptr):
    """Make the pointer write ([rag.service].profile) always fail (return
    False, as save_setting_to_cli_config does on an I/O failure -- see its
    real implementation) while the marker write
    ([rag.service].first_run_import_done) keeps succeeding normally. Models
    set_active_profile()'s documented failure mode: it swallows a failed
    write (logs a warning, leaves the pointer untouched) instead of raising."""

    def _fake_save_setting(section, key, value):
        if section == "rag.service" and key == "profile":
            return False  # simulated failure -- ptr["v"] deliberately NOT updated
        if section == "rag.service" and key == "first_run_import_done":
            ptr["marker"] = value
            return True
        return True

    monkeypatch.setattr(ac, "save_setting_to_cli_config", _fake_save_setting, raising=False)


def _wire_pointer_write_succeeds(monkeypatch, ac, ptr):
    """Restore normal (successful) writes for both keys -- the same routing
    _wire()'s own fake uses, exposed here so a test can flip a previously-
    failing pointer write back to succeeding mid-test."""

    def _fake_save_setting(section, key, value):
        if section == "rag.service" and key == "profile":
            ptr["v"] = value
        elif section == "rag.service" and key == "first_run_import_done":
            ptr["marker"] = value
        return True

    monkeypatch.setattr(ac, "save_setting_to_cli_config", _fake_save_setting, raising=False)


def test_ensure_imported_profile_does_not_mark_when_healing_pointer_write_fails(monkeypatch, tmp_path):
    """Important (task-639 review round 3, reviewer-reproduced): the healing
    branch used to call _mark_first_run_import_done() unconditionally right
    after set_active_profile(_IMPORTED_ID), but set_active_profile() itself
    swallows a failed pointer write (logs a warning and returns without
    resetting the shared service -- see its docstring). If the pointer write
    fails while the marker write succeeds, the user would be permanently
    parked on the default profile with NO future retry: the marker is
    exactly what stops this branch from ever running again. Must instead
    only mark once _active_profile_id() is confirmed to read back as
    _IMPORTED_ID, and a later call (once the write starts succeeding) must
    retry the activation rather than being permanently blocked."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)
    half_done = ProfileConfig(id=ac._IMPORTED_ID,
                              name="Imported settings",
                              description="Captured from your existing RAG configuration on first run.",
                              profile_type="custom",
                              rag_config=ac.resolve_active_rag_config())
    mgr.save_profile(half_done)
    assert ptr["v"] is None  # pointer still resolves to the default builtin

    _wire_pointer_write_fails(monkeypatch, ac, ptr)

    result = ac.ensure_imported_profile()

    assert result is None
    assert ptr["v"] is None  # pointer write failed -- never actually activated
    assert ptr["marker"] is False  # MUST NOT be set: the activation never actually happened

    # A later call, once the pointer write starts succeeding again, must
    # RETRY the activation -- not be permanently blocked by a marker that
    # was never legitimately earned.
    _wire_pointer_write_succeeds(monkeypatch, ac, ptr)

    result2 = ac.ensure_imported_profile()

    assert result2 is None
    assert ptr["v"] == ac._IMPORTED_ID  # retried and succeeded this time
    assert ptr["marker"] is True


def test_ensure_imported_profile_does_not_mark_when_fresh_import_pointer_write_fails(monkeypatch, tmp_path):
    """Mirrors the fix at the fresh-import call site (task-639 review round
    3): if set_active_profile() fails right after a brand-new "Imported
    settings" profile is created, the marker must not be set either --
    otherwise the next process's first RAG touch would see `existing is not
    None`, the marker already True, and skip retrying the failed activation
    forever, even though the pointer was never actually updated."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {"search": {"default_top_k": 10}})
    _wire_pointer_write_fails(monkeypatch, ac, ptr)

    new_id = ac.ensure_imported_profile()

    assert new_id == ac._IMPORTED_ID  # the profile WAS created
    assert ptr["v"] is None  # but the pointer write failed -- never activated
    assert ptr["marker"] is False  # MUST NOT be set

    # A later call must retry activation via the healing branch (the profile
    # already exists now), not skip it because of a falsely-set marker.
    _wire_pointer_write_succeeds(monkeypatch, ac, ptr)

    result2 = ac.ensure_imported_profile()

    assert result2 is None  # profile already exists -- this is the healing path now
    assert ptr["v"] == ac._IMPORTED_ID
    assert ptr["marker"] is True


# --- task-640 item 3: real on-disk config.toml integration coverage for
# _has_legacy_rag_config_material(). Every test above exercises it only
# against `_wire_legacy_rag_config`'s monkeypatched `get_cli_setting` fake --
# these tests instead write a REAL config.toml to the (already test-isolated,
# via the autouse `isolate_test_environment` fixture in Tests/conftest.py)
# TLDW_CONFIG_PATH and call the function with NO get_cli_setting patch at
# all, so the real tomllib-parsing path (tldw_chatbook.config.
# load_cli_config_and_ensure_existence -> get_cli_setting) is what's under
# test, closing the gap between the unit-level fakes and reality. ---


def _write_config_toml(text: str) -> None:
    """Write real TOML content to this test's isolated TLDW_CONFIG_PATH.

    `isolate_test_environment` (Tests/conftest.py, autouse) already points
    TLDW_CONFIG_PATH at a fresh, per-test tmp_path location before any test
    body runs -- writing directly to that path (rather than introducing a
    second override) is what makes this a real round-trip through
    `tldw_chatbook.config`'s actual load/cache path: `_get_effective_config_
    path()` resolves the env var, and the module-level config cache is keyed
    by that resolved path, so each test's unique tmp_path guarantees a real
    fresh parse rather than a stale cached config from a previous test.
    """
    import tldw_chatbook.config as config_module

    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    # Belt-and-braces: force a reload even if some earlier-in-process code
    # already touched this exact path (shouldn't happen given tmp_path's
    # per-test uniqueness, but this keeps the test honest either way rather
    # than silently passing against a stale in-memory cache).
    config_module.load_cli_config_and_ensure_existence(force_reload=True)


def test_has_legacy_rag_config_material_true_against_a_real_config_toml_with_legacy_section():
    """task-640 item 3: a real config.toml with a hand-set legacy
    [AppRAGSearchConfig.rag.search] section (the task-635 "upgrading from
    before the profile system existed" signal) must be detected via the
    REAL tomllib-parsing path, no get_cli_setting monkeypatch involved."""
    from tldw_chatbook.RAG_Search.simplified.active_config import (
        _has_legacy_rag_config_material,
    )

    _write_config_toml(
        """
[AppRAGSearchConfig.rag.search]
default_top_k = 25
score_threshold = 0.42
"""
    )

    assert _has_legacy_rag_config_material() is True


def test_has_legacy_rag_config_material_false_against_a_fresh_install_config_toml():
    """task-640 item 3: a real config.toml with NO [AppRAGSearchConfig.rag]
    section at all (the genuine fresh-install case) must resolve to False
    via the real parsing path -- ensure_imported_profile() must leave a
    brand-new user on the default builtin profile rather than auto-creating
    'Imported settings' underneath them."""
    from tldw_chatbook.RAG_Search.simplified.active_config import (
        _has_legacy_rag_config_material,
    )

    _write_config_toml(
        """
[rag.service]
profile = "hybrid_basic"
"""
    )

    assert _has_legacy_rag_config_material() is False


def test_has_legacy_rag_config_material_false_when_the_legacy_section_is_present_but_empty():
    """An [AppRAGSearchConfig.rag] table that exists in the TOML but has no
    sub-keys at all (e.g. left behind by a partial hand-edit) must still
    read as "nothing to preserve" -- matches the function's documented
    contract (non-empty dict required), verified here against the real
    parser rather than a hand-built fake dict."""
    from tldw_chatbook.RAG_Search.simplified.active_config import (
        _has_legacy_rag_config_material,
    )

    _write_config_toml(
        """
[AppRAGSearchConfig]
"""
    )

    assert _has_legacy_rag_config_material() is False
