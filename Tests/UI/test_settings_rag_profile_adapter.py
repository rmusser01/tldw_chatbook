import dataclasses

import pytest
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, SearchConfig, VectorStoreConfig


@pytest.fixture
def wired(tmp_path, monkeypatch):
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    state = {"active": "hybrid_basic"}
    monkeypatch.setattr(ac, "_active_profile_id", lambda: state["active"], raising=False)
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as ad
    monkeypatch.setattr(ad, "_manager", lambda: mgr, raising=False)
    monkeypatch.setattr(ad, "_active_profile_id", lambda: state["active"], raising=False)
    return mgr, state


def _user_profile(mgr, state, **search_over):
    p = mgr.clone_profile("hybrid_basic", "My RAG")
    for k, v in search_over.items():
        setattr(p.rag_config.search, k, v)
    mgr.save_profile(p)
    state["active"] = p.id
    return p


def test_load_reads_the_active_profile(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import load_rag_defaults_from_active_profile
    mgr, state = wired
    _user_profile(mgr, state, default_top_k=42, hybrid_alpha=0.25)
    d = load_rag_defaults_from_active_profile()
    assert d.default_top_k == 42
    assert d.hybrid_alpha == 0.25


def test_save_writes_the_active_profile_file_not_config(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile, save_rag_defaults_to_active_profile)
    mgr, state = wired
    p = _user_profile(mgr, state)
    d = load_rag_defaults_from_active_profile()
    # SettingsLibraryRagDefaults is frozen (see settings_library_rag_defaults.py);
    # the codebase's own convention for producing a modified copy in tests is
    # dataclasses.replace(...) (see Tests/UI/test_settings_library_rag_defaults.py).
    d = dataclasses.replace(d, default_top_k=77)
    ok, reason = save_rag_defaults_to_active_profile(d)
    assert ok and reason == ""
    # Reload from disk via a FRESH manager over the same dir:
    mgr2 = ConfigProfileManager(profiles_dir=mgr.profiles_dir)
    assert mgr2.get_profile(p.id).rag_config.search.default_top_k == 77


def test_save_refuses_builtin_active(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile, save_rag_defaults_to_active_profile)
    mgr, state = wired      # active = hybrid_basic (builtin)
    d = load_rag_defaults_from_active_profile()
    ok, reason = save_rag_defaults_to_active_profile(d)
    assert not ok and reason == "builtin"


def test_active_profile_info(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import active_profile_info
    mgr, state = wired
    info = active_profile_info()
    assert info == {"id": "hybrid_basic", "name": "Hybrid Basic", "read_only": True}
