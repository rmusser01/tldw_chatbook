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


def test_list_profiles_grouped_separates_builtin_and_user_name_sorted(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import list_profiles_grouped
    mgr, state = wired
    user = _user_profile(mgr, state)  # clones hybrid_basic -> "My RAG", sets it active
    state["active"] = user.id

    grouped = list_profiles_grouped()

    builtin_ids = {p["id"] for p in grouped["builtin"]}
    user_entries = {(p["id"], p["name"]) for p in grouped["user"]}
    assert "hybrid_basic" in builtin_ids
    assert (user.id, "My RAG") in user_entries
    # user clone must not leak into the builtin group (and vice versa)
    assert user.id not in builtin_ids
    assert "hybrid_basic" not in {p["id"] for p in grouped["user"]}
    assert grouped["active_id"] == user.id
    assert [p["name"] for p in grouped["builtin"]] == sorted(
        p["name"] for p in grouped["builtin"]
    )
    assert [p["name"] for p in grouped["user"]] == sorted(
        p["name"] for p in grouped["user"]
    )


def test_activate_profile_rejects_unsafe_id_without_raising(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import activate_profile
    ok, reason = activate_profile("../evil")
    assert ok is False
    assert reason


def test_activate_profile_flips_the_pointer_on_a_valid_user_id(wired, monkeypatch):
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as ad
    mgr, state = wired
    user = _user_profile(mgr, state)
    state["active"] = "hybrid_basic"  # something else is active right now

    # Bound as a module global on the adapter (see settings_rag_profile_adapter.py
    # module-seam comment) precisely so this fake, which just flips the
    # test's pointer state, can stand in for the real SP2b config write.
    monkeypatch.setattr(
        ad, "set_active_profile", lambda pid: state.__setitem__("active", pid)
    )

    ok, reason = ad.activate_profile(user.id)

    assert ok is True
    assert reason == ""
    assert state["active"] == user.id


def test_delete_user_profile_refuses_builtin_without_raising(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import delete_user_profile
    ok, reason = delete_user_profile("hybrid_basic")
    assert ok is False
    assert reason


def test_delete_user_profile_removes_a_user_profile(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import delete_user_profile
    mgr, state = wired
    user = _user_profile(mgr, state)

    ok, reason = delete_user_profile(user.id)

    assert ok is True
    assert reason == ""
    assert mgr.get_profile(user.id) is None


def test_clone_profile_as_creates_a_new_writable_profile(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import clone_profile_as
    mgr, state = wired

    ok, new_id = clone_profile_as("hybrid_basic", "My Copy")

    assert ok is True
    clone = mgr.get_profile(new_id)
    assert clone is not None
    assert clone.name == "My Copy"
    assert clone.read_only is False
    # writable: saving further edits to it must not raise
    mgr.save_profile(clone)


def test_clone_profile_as_reports_unknown_source_without_raising(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import clone_profile_as
    ok, reason = clone_profile_as("does-not-exist", "Copy")
    assert ok is False
    assert reason


def test_rename_user_profile_updates_the_display_name(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import rename_user_profile
    mgr, state = wired
    user = _user_profile(mgr, state)

    ok, reason = rename_user_profile(user.id, "Renamed RAG")

    assert ok is True
    assert reason == ""
    assert mgr.get_profile(user.id).name == "Renamed RAG"


def test_rename_user_profile_refuses_builtin_without_raising(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import rename_user_profile
    ok, reason = rename_user_profile("hybrid_basic", "Nope")
    assert ok is False
    assert reason
