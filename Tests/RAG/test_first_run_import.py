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
