import pytest
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig


def _wire(monkeypatch, tmp_path):
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    ptr = {"v": None}
    monkeypatch.setattr(ac, "_active_profile_id", lambda: ptr["v"] or "hybrid_basic", raising=False)
    monkeypatch.setattr(ac, "save_setting_to_cli_config",
                        lambda s, k, v: ptr.update(v=v) or True, raising=False)
    monkeypatch.setattr(ac, "reset_shared_rag_service", lambda: None, raising=False)
    return mgr, ptr


def test_first_run_creates_imported_profile_and_sets_active(monkeypatch, tmp_path):
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    mgr, ptr = _wire(monkeypatch, tmp_path)
    new_id = ensure_imported_profile()
    assert new_id is not None
    imported = mgr.get_profile(new_id)
    assert imported is not None and imported.read_only is False
    assert ptr["v"] == new_id  # set active
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

    result = ac.ensure_imported_profile()

    assert result is None  # idempotent: no new profile id returned
    assert [p for p in mgr.list_profiles() if p == ac._IMPORTED_ID] == [ac._IMPORTED_ID]  # no duplicate created
    assert ptr["v"] == ac._IMPORTED_ID  # healed: pointer now activates the existing profile


def test_ensure_imported_profile_swallows_save_failure(monkeypatch, tmp_path):
    """Exception-safety: any failure while creating the imported profile must
    be swallowed (logged, not raised) so it can never block RAG service
    creation, and must not leave a half-activated pointer behind."""
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    mgr, ptr = _wire(monkeypatch, tmp_path)

    def _boom(profile):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mgr, "save_profile", _boom)

    result = ac.ensure_imported_profile()

    assert result is None
    assert mgr.get_profile(ac._IMPORTED_ID) is None
    assert ptr["v"] is None  # never activated a profile that failed to save


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

    imported = mgr.get_profile(new_id).rag_config
    assert imported.search.default_top_k == 25
    assert imported.search.score_threshold == 0.42
    assert imported.search.include_citations is False
    assert imported.search.enable_reranking is True
    assert imported.search.reranker_model == "cross-encoder/legacy"
    assert imported.search.reranker_top_k == 7


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


def test_imported_profile_unchanged_when_no_legacy_keys_set(monkeypatch, tmp_path):
    """No hand-set legacy keys -> the imported snapshot is byte-equal to
    today's plain resolve_active_rag_config() capture (no regression when
    there's nothing to merge)."""
    from dataclasses import asdict
    from tldw_chatbook.RAG_Search.simplified.active_config import ensure_imported_profile, resolve_active_rag_config
    mgr, ptr = _wire(monkeypatch, tmp_path)
    _wire_legacy_rag_config(monkeypatch, {})
    expected = resolve_active_rag_config()

    new_id = ensure_imported_profile()

    imported = mgr.get_profile(new_id).rag_config
    assert asdict(imported) == asdict(expected)


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
