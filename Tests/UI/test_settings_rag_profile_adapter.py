import dataclasses

import pytest
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig
from tldw_chatbook.RAG_Search.reranker import RerankingConfig
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


# --- Task-2 review Finding 3: the SP2a manager does raw file I/O, so
# clone/rename/delete must convert OSError (not just ValueError) into
# (False, reason) -- these wrappers run inside a thread @work with Textual's
# default exit_on_error=True, so an uncaught OSError crashes the whole app. ---


def test_clone_profile_as_converts_os_error_without_raising(wired, monkeypatch):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import clone_profile_as
    mgr, state = wired

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mgr, "clone_profile", _raise)

    ok, reason = clone_profile_as("hybrid_basic", "My Copy")

    assert ok is False
    assert "disk full" in reason


def test_rename_user_profile_converts_os_error_without_raising(wired, monkeypatch):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import rename_user_profile
    mgr, state = wired
    user = _user_profile(mgr, state)

    def _raise(*args, **kwargs):
        raise PermissionError("no write access")

    monkeypatch.setattr(mgr, "rename_profile", _raise)

    ok, reason = rename_user_profile(user.id, "Renamed RAG")

    assert ok is False
    assert "no write access" in reason


def test_delete_user_profile_converts_os_error_without_raising(wired, monkeypatch):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import delete_user_profile
    mgr, state = wired
    user = _user_profile(mgr, state)

    def _raise(*args, **kwargs):
        raise OSError("device busy")

    monkeypatch.setattr(mgr, "delete_profile", _raise)

    ok, reason = delete_user_profile(user.id)

    assert ok is False
    assert "device busy" in reason


# --- Task 3: extended editor fields (embedding/chunking/vector-store/reranking)
# + rerank presence semantics. ---


def test_load_round_trips_the_extended_fields_from_a_distinctive_profile(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
    )
    mgr, state = wired
    p = _user_profile(mgr, state)
    p.rag_config.embedding.model = "BAAI/bge-large-en-v1.5"
    p.rag_config.embedding.device = "cuda"
    p.rag_config.embedding.batch_size = 8
    p.rag_config.embedding.max_length = 1024
    p.rag_config.chunking.chunk_size = 777
    p.rag_config.chunking.chunk_overlap = 99
    p.rag_config.chunking.chunking_method = "sentences"
    p.rag_config.vector_store.distance_metric = "l2"
    p.reranking_config = RerankingConfig(model_name="my-reranker", top_k_to_rerank=13)
    p.rag_config.search.enable_reranking = True
    mgr.save_profile(p)

    d = load_rag_defaults_from_active_profile()

    assert d.embedding_model == "BAAI/bge-large-en-v1.5"
    assert d.embedding_device == "cuda"
    assert d.embedding_batch_size == 8
    assert d.embedding_max_length == 1024
    assert d.chunk_size == 777
    assert d.chunk_overlap == 99
    assert d.chunking_method == "sentences"
    assert d.distance_metric == "l2"
    assert d.enable_reranking is True
    assert d.reranker_model == "my-reranker"
    assert d.reranker_top_k == 13


def test_load_reports_reranking_disabled_when_reranking_config_is_none(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
    )
    mgr, state = wired
    p = _user_profile(mgr, state)
    assert p.reranking_config is None

    d = load_rag_defaults_from_active_profile()

    assert d.enable_reranking is False
    assert d.reranker_model == ""


def test_save_with_reranking_enabled_creates_a_reranking_config_on_reload(wired):
    """Rerank presence semantics: the service reads
    ``profile.reranking_config is not None`` (rag_factory.py) to decide
    whether reranking is on -- so the UI toggle must control the PRESENCE of
    ``reranking_config``, not just a flag somewhere."""
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        save_rag_defaults_to_active_profile,
    )
    mgr, state = wired
    p = _user_profile(mgr, state)
    assert p.reranking_config is None

    d = load_rag_defaults_from_active_profile()
    d = dataclasses.replace(
        d, enable_reranking=True, reranker_model="cross-encoder-x", reranker_top_k=7
    )
    ok, reason = save_rag_defaults_to_active_profile(d)
    assert ok and reason == ""

    mgr2 = ConfigProfileManager(profiles_dir=mgr.profiles_dir)
    reloaded = mgr2.get_profile(p.id)
    assert reloaded.reranking_config is not None
    assert reloaded.reranking_config.model_name == "cross-encoder-x"
    assert reloaded.reranking_config.top_k_to_rerank == 7
    assert reloaded.rag_config.search.enable_reranking is True


def test_save_with_reranking_disabled_clears_an_existing_reranking_config(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        save_rag_defaults_to_active_profile,
    )
    mgr, state = wired
    p = _user_profile(mgr, state)
    p.reranking_config = RerankingConfig(model_name="existing", top_k_to_rerank=9)
    p.rag_config.search.enable_reranking = True
    mgr.save_profile(p)

    d = load_rag_defaults_from_active_profile()
    assert d.enable_reranking is True
    d = dataclasses.replace(d, enable_reranking=False)
    ok, reason = save_rag_defaults_to_active_profile(d)
    assert ok and reason == ""

    mgr2 = ConfigProfileManager(profiles_dir=mgr.profiles_dir)
    reloaded = mgr2.get_profile(p.id)
    assert reloaded.reranking_config is None
    assert reloaded.rag_config.search.enable_reranking is False


def test_save_with_reranking_enabled_and_blank_model_leaves_the_default_model_name(
    wired,
):
    """An empty ``reranker_model`` means "use the default" -- applying it must
    never stomp ``RerankingConfig``'s own default ``model_name`` with an empty
    string."""
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        save_rag_defaults_to_active_profile,
    )
    mgr, state = wired
    p = _user_profile(mgr, state)

    d = load_rag_defaults_from_active_profile()
    assert d.reranker_model == ""
    d = dataclasses.replace(d, enable_reranking=True)
    ok, reason = save_rag_defaults_to_active_profile(d)
    assert ok and reason == ""

    mgr2 = ConfigProfileManager(profiles_dir=mgr.profiles_dir)
    reloaded = mgr2.get_profile(p.id)
    assert reloaded.reranking_config.model_name == RerankingConfig().model_name


def test_rerank_toggle_reaches_the_real_service_enable_reranking_flag(
    wired, monkeypatch
):
    """Inert-toggle integration: proves the UI toggle actually threads through
    to ``create_rag_service``'s ``EnhancedRAGServiceV2.enable_reranking`` --
    not just a config field nobody reads."""
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        save_rag_defaults_to_active_profile,
    )
    mgr, state = wired
    p = _user_profile(mgr, state)

    d = load_rag_defaults_from_active_profile()
    d = dataclasses.replace(d, enable_reranking=True)
    ok, reason = save_rag_defaults_to_active_profile(d)
    assert ok and reason == ""

    import tldw_chatbook.RAG_Search.simplified.rag_factory as rag_factory
    # Same seam as the `wired` fixture (module-level `get_profile_manager`
    # lookup) but bound in rag_factory's own namespace -- it imports the
    # symbol directly (`from ..config_profiles import get_profile_manager`),
    # so patching `config_profiles.get_profile_manager` would not be seen here.
    monkeypatch.setattr(rag_factory, "get_profile_manager", lambda: mgr)

    from tldw_chatbook.RAG_Search.simplified.config import create_config_for_testing

    service = rag_factory.create_rag_service(
        profile_name=p.id, config=create_config_for_testing()
    )

    assert service.enable_reranking is True


def test_rerank_toggle_off_reaches_the_real_service_enable_reranking_flag(
    wired, monkeypatch
):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        save_rag_defaults_to_active_profile,
    )
    mgr, state = wired
    p = _user_profile(mgr, state)
    p.reranking_config = RerankingConfig()
    mgr.save_profile(p)

    d = load_rag_defaults_from_active_profile()
    d = dataclasses.replace(d, enable_reranking=False)
    ok, reason = save_rag_defaults_to_active_profile(d)
    assert ok and reason == ""

    import tldw_chatbook.RAG_Search.simplified.rag_factory as rag_factory
    monkeypatch.setattr(rag_factory, "get_profile_manager", lambda: mgr)

    from tldw_chatbook.RAG_Search.simplified.config import create_config_for_testing

    service = rag_factory.create_rag_service(
        profile_name=p.id, config=create_config_for_testing()
    )

    assert service.enable_reranking is False


def test_validate_full_config_reports_a_real_ragconfig_violation(wired):
    """`RAGConfig.validate()` has zero callers today; `validate_full_config`
    is the first. Pick a violation it genuinely reports (chunk_overlap must
    be less than chunk_size) and confirm it surfaces."""
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        validate_full_config,
    )
    mgr, state = wired
    _user_profile(mgr, state)
    d = load_rag_defaults_from_active_profile()
    d = dataclasses.replace(d, chunk_size=100, chunk_overlap=100)

    errors = validate_full_config(d)

    assert any("chunk_overlap" in e and "chunk_size" in e for e in errors)


def test_validate_full_config_accepts_a_valid_config(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        validate_full_config,
    )
    mgr, state = wired
    _user_profile(mgr, state)
    d = load_rag_defaults_from_active_profile()

    errors = validate_full_config(d)

    assert errors == []


def test_validate_full_config_flags_reranker_top_k_below_one(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        validate_full_config,
    )
    mgr, state = wired
    _user_profile(mgr, state)
    d = load_rag_defaults_from_active_profile()
    d = dataclasses.replace(d, enable_reranking=True, reranker_top_k=0)

    errors = validate_full_config(d)

    assert any("top-k" in e.lower() for e in errors)


def test_validate_full_config_warns_when_reranker_top_k_exceeds_default_top_k(wired):
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        validate_full_config,
    )
    mgr, state = wired
    _user_profile(mgr, state)
    d = load_rag_defaults_from_active_profile()
    d = dataclasses.replace(
        d, enable_reranking=True, default_top_k=5, reranker_top_k=50
    )

    errors = validate_full_config(d)

    assert any("top-k" in e.lower() and "exceed" in e.lower() for e in errors)


def test_validate_full_config_never_mutates_the_cached_active_profile(wired):
    """`validate_full_config` must scratch-copy the active profile -- never
    mutate the live cached object the manager hands back to every other
    caller."""
    from tldw_chatbook.UI.Screens.settings_rag_profile_adapter import (
        load_rag_defaults_from_active_profile,
        validate_full_config,
    )
    mgr, state = wired
    p = _user_profile(mgr, state)
    assert p.reranking_config is None
    original_chunk_size = p.rag_config.chunking.chunk_size

    d = load_rag_defaults_from_active_profile()
    d = dataclasses.replace(
        d, chunk_size=original_chunk_size + 1, enable_reranking=True
    )

    validate_full_config(d)

    assert mgr.get_profile(p.id).rag_config.chunking.chunk_size == original_chunk_size
    assert mgr.get_profile(p.id).reranking_config is None
