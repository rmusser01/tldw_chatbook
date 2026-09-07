import pytest
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig
from tldw_chatbook.config import get_cli_setting


@pytest.mark.parametrize(
    "value, expected",
    [
        ("plain", "plain"),
        ("semantic", "semantic"),
        ("hybrid", "hybrid"),
        ("future-mode", "semantic"),
        ("SEMANTIC", "semantic"),
        (None, "semantic"),
        (1, "semantic"),
    ],
)
def test_normalize_rag_search_mode_accepts_only_known_exact_values(value, expected):
    from tldw_chatbook.RAG_Search.simplified.active_config import (
        normalize_rag_search_mode,
    )

    assert normalize_rag_search_mode(value) == expected


@pytest.fixture
def active(tmp_path, monkeypatch):
    """A ConfigProfileManager over a temp dir + a helper to set the active pointer."""
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "profiles")
    # Point resolve_active_rag_config at this manager + a chosen active id via monkeypatch.
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "_manager", lambda: mgr, raising=False)
    state = {"active": "hybrid_basic"}
    monkeypatch.setattr(ac, "_active_profile_id", lambda: state["active"], raising=False)
    return mgr, state


def test_resolves_active_profiles_rag_config(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom A", description="d", profile_type="custom",
                      rag_config=RAGConfig(embedding=EmbeddingConfig(model="modelA"),
                                           chunking=ChunkingConfig(chunk_size=321),
                                           vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id
    cfg = resolve_active_rag_config()
    assert cfg.embedding.model == "modelA"
    assert cfg.chunking.chunk_size == 321


def test_env_overrides_win_over_profile(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom B", description="d", profile_type="custom",
                      rag_config=RAGConfig(embedding=EmbeddingConfig(model="profile-model"),
                                           vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-model")
    assert resolve_active_rag_config().embedding.model == "env-model"


def test_returns_deep_copy_not_the_profile_object(active):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    cfg = resolve_active_rag_config()
    cfg.chunking.chunk_size = 99999
    assert mgr.get_profile(state["active"]).rag_config.chunking.chunk_size != 99999


def test_env_overrides_chunking_and_search_and_pipeline_settings(active, monkeypatch):
    """These env reads exist in the legacy from_settings body (RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP, RAG_TOP_K, RAG_SEARCH_MODE, RAG_DEFAULT_PIPELINE) but were
    not covered in the task brief's _apply_env_overrides sketch. They must still
    be honored so no override behavior regresses."""
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom C", description="d", profile_type="custom",
                      rag_config=RAGConfig(vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id

    monkeypatch.setenv("RAG_CHUNK_SIZE", "555")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "77")
    monkeypatch.setenv("RAG_TOP_K", "13")
    monkeypatch.setenv("RAG_SEARCH_MODE", "hybrid")
    monkeypatch.setenv("RAG_DEFAULT_PIPELINE", "custom_pipeline")

    cfg = resolve_active_rag_config()
    assert cfg.chunking.chunk_size == 555
    assert cfg.chunking.chunk_overlap == 77
    assert cfg.search.default_top_k == 13
    assert cfg.search.default_search_mode == "hybrid"
    assert cfg.pipeline.default_pipeline == "custom_pipeline"


def test_override_persist_dir_and_embedding_model_args_win_over_env(active, monkeypatch, tmp_path):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="Custom D", description="d", profile_type="custom",
                      rag_config=RAGConfig(vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id

    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "env-model")
    monkeypatch.setenv("RAG_PERSIST_DIR", str(tmp_path / "env-dir"))

    override_dir = tmp_path / "explicit-dir"
    cfg = resolve_active_rag_config(override_embedding_model="explicit-model",
                                     override_persist_dir=override_dir)
    assert cfg.embedding.model == "explicit-model"
    assert str(cfg.vector_store.persist_directory) == str(override_dir)


def test_hybrid_alpha_comes_from_active_profile(active, monkeypatch):
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, SearchConfig, VectorStoreConfig
    from tldw_chatbook.RAG_Search.fusion import resolve_hybrid_alpha
    mgr, state = active
    p = ProfileConfig(name="Alpha", description="d", profile_type="custom",
                      rag_config=RAGConfig(search=SearchConfig(hybrid_alpha=0.33),
                                           vector_store=VectorStoreConfig(type="memory")))
    mgr.save_profile(p); state["active"] = p.id
    assert resolve_hybrid_alpha() == pytest.approx(0.33)  # explicit=None -> active profile


def test_set_active_profile_writes_pointer_and_resets_service(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import set_active_profile
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    writes = {}
    monkeypatch.setattr(ac, "save_setting_to_cli_config",
                        lambda section, key, value: writes.update({(section, key): value}) or True,
                        raising=False)
    reset = {"called": False}
    monkeypatch.setattr(ac, "reset_shared_rag_service",
                        lambda: reset.update(called=True), raising=False)
    set_active_profile("my_profile")
    # CORRECTED assertion: save_setting_to_cli_config(section, key, value) nests
    # via the section arg, so the pointer write is section="rag.service",
    # key="profile" (NOT section="rag", key="service.profile" as the brief's
    # first draft had it) -- that shape is what the read path
    # (get_cli_setting("rag", "service", {}).get("profile")) actually resolves.
    assert writes.get(("rag.service", "profile")) == "my_profile"
    assert reset["called"] is True


@pytest.mark.parametrize(
    "stored_type,env_value,expected_type",
    [
        ("chroma", "memory", "memory"),
        ("memory", "chroma", "chroma"),
    ],
)
def test_env_vector_store_override_wins_over_profile(
    active, monkeypatch, stored_type, env_value, expected_type
):
    """RAG_VECTOR_STORE must be applied by _apply_env_overrides itself.

    resolve_active_rag_config() deep-copies an already-constructed profile's
    RAGConfig -- copy.deepcopy does NOT re-run VectorStoreConfig.__post_init__
    -- so a claim that __post_init__ "honors" the env var here is false; the
    env layer must apply it explicitly, in both directions (memory<->chroma).
    """
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="VS", description="d", profile_type="custom",
                      rag_config=RAGConfig(vector_store=VectorStoreConfig(type=stored_type)))
    mgr.save_profile(p); state["active"] = p.id
    monkeypatch.setenv("RAG_VECTOR_STORE", env_value)
    assert resolve_active_rag_config().vector_store.type == expected_type


def test_no_env_vector_store_leaves_profile_type_untouched(active, monkeypatch):
    from tldw_chatbook.RAG_Search.simplified.active_config import resolve_active_rag_config
    mgr, state = active
    p = ProfileConfig(name="VS2", description="d", profile_type="custom",
                      rag_config=RAGConfig(vector_store=VectorStoreConfig(type="chroma")))
    mgr.save_profile(p); state["active"] = p.id
    monkeypatch.delenv("RAG_VECTOR_STORE", raising=False)
    assert resolve_active_rag_config().vector_store.type == "chroma"


def test_set_active_profile_rejects_bad_pointer_write(monkeypatch):
    """Finding #6: a failed pointer write must not silently reset the shared
    service -- there's no point rebuilding it when the pointer didn't move."""
    from tldw_chatbook.RAG_Search.simplified.active_config import set_active_profile
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    monkeypatch.setattr(ac, "save_setting_to_cli_config",
                        lambda section, key, value: False, raising=False)
    reset = {"called": False}
    monkeypatch.setattr(ac, "reset_shared_rag_service",
                        lambda: reset.update(called=True), raising=False)
    set_active_profile("some_profile")
    assert reset["called"] is False


@pytest.mark.parametrize("bad_id", ["", None, "../x", "not a slug!", "Has Spaces"])
def test_set_active_profile_rejects_invalid_profile_id(bad_id, monkeypatch):
    """Finding #3: profile_id must be validated before any write/reset."""
    from tldw_chatbook.RAG_Search.simplified.active_config import set_active_profile
    import tldw_chatbook.RAG_Search.simplified.active_config as ac
    writes = []
    resets = []
    monkeypatch.setattr(ac, "save_setting_to_cli_config",
                        lambda *a, **k: writes.append(a) or True, raising=False)
    monkeypatch.setattr(ac, "reset_shared_rag_service",
                        lambda: resets.append(1), raising=False)
    with pytest.raises(ValueError):
        set_active_profile(bad_id)
    assert writes == []
    assert resets == []


def test_set_active_profile_round_trip_write_matches_read(monkeypatch, tmp_path):
    """Real (non-mock) proof: the pointer set_active_profile() writes is exactly
    what the resolver's read path (_active_profile_id() ->
    get_cli_setting("rag", "service", {}).get("profile")) reads back. Uses a
    real temp TOML file via TLDW_CONFIG_PATH so this exercises an actual
    write-then-read round trip, not mocks."""
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    from tldw_chatbook.RAG_Search.simplified.active_config import set_active_profile
    set_active_profile("some_profile_xyz")
    assert get_cli_setting("rag", "service", {}).get("profile") == "some_profile_xyz"


# --------------------------------------------------------------------------
# The search-depth-only resolution (B3 / TASK-15020).
#
# `resolve_active_rag_config()` builds the WHOLE config, and its env layer
# probes the embedding device -- which imports torch (measured: ~0.9s and a
# torch import on the first call in a warm app process). The Library Search/RAG
# panel's display state is rebuilt on every render and is documented as a path
# that "never imports torch (or any other optional Search/RAG dependency)"
# (`library_screen._library_rag_panel_state`), so the window's depth default
# cannot be read through the full resolution. `resolve_active_rag_top_k()` is
# the narrow read for it: same profile, same RAG_TOP_K override rule (one
# definition, shared with `_apply_env_overrides`), no device probe.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stored_mode, env_mode, expected",
    [
        ("plain", None, "plain"),
        ("semantic", None, "semantic"),
        ("hybrid", None, "hybrid"),
        ("future-mode", None, "semantic"),
        ("plain", "hybrid", "hybrid"),
        ("plain", "future-mode", "semantic"),
    ],
)
def test_search_mode_only_resolution_agrees_with_full_config(
    active, monkeypatch, stored_mode, env_mode, expected
):
    from tldw_chatbook.RAG_Search.simplified.active_config import (
        resolve_active_rag_config,
        resolve_active_rag_search_mode,
    )
    from tldw_chatbook.RAG_Search.simplified.config import SearchConfig

    mgr, state = active
    profile = ProfileConfig(
        name="Mode profile",
        description="d",
        profile_type="custom",
        rag_config=RAGConfig(
            search=SearchConfig(default_search_mode=stored_mode),
            vector_store=VectorStoreConfig(type="memory"),
        ),
    )
    mgr.save_profile(profile)
    state["active"] = profile.id
    monkeypatch.delenv("RAG_SEARCH_MODE", raising=False)
    if env_mode is not None:
        monkeypatch.setenv("RAG_SEARCH_MODE", env_mode)

    assert resolve_active_rag_search_mode() == expected
    assert resolve_active_rag_config().search.default_search_mode == expected


@pytest.mark.parametrize(
    "profile_top_k, env_top_k, expected",
    [
        (15, None, 15),
        (15, "13", 13),
        (5, None, 5),
        (20, "1", 1),
    ],
)
def test_top_k_only_resolution_agrees_with_the_full_config_resolution(
    active, monkeypatch, profile_top_k, env_top_k, expected
):
    """The coupling pin: two readers of one number must never disagree."""
    from tldw_chatbook.RAG_Search.simplified.active_config import (
        resolve_active_rag_config,
        resolve_active_rag_top_k,
    )
    from tldw_chatbook.RAG_Search.simplified.config import SearchConfig

    mgr, state = active
    p = ProfileConfig(
        name="Depth profile",
        description="d",
        profile_type="custom",
        rag_config=RAGConfig(
            search=SearchConfig(default_top_k=profile_top_k),
            vector_store=VectorStoreConfig(type="memory"),
        ),
    )
    mgr.save_profile(p)
    state["active"] = p.id
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    if env_top_k is not None:
        monkeypatch.setenv("RAG_TOP_K", env_top_k)

    assert resolve_active_rag_top_k() == expected
    assert resolve_active_rag_config().search.default_top_k == expected


def test_top_k_only_resolution_reports_the_shipped_default_profile_depth(active):
    """The number B3's live check reads off the screen: the default profile
    (`hybrid_basic`, the pointer's own fallback) ships `default_top_k = 15`.
    """
    from tldw_chatbook.RAG_Search.simplified.active_config import (
        DEFAULT_PROFILE,
        resolve_active_rag_top_k,
    )

    _mgr, state = active
    state["active"] = DEFAULT_PROFILE

    assert resolve_active_rag_top_k() == 15


def test_top_k_only_resolution_does_not_import_torch():
    """The load-bearing property, checked the only way it can be: in a fresh
    interpreter. A UI render path calls this; `resolve_active_rag_config()`'s
    `device == "auto"` branch imports torch, so a "simplification" back onto
    the full resolution would put a ~1s optional-dependency import on the
    Library panel's render. Subprocess, because `torch` is already in
    `sys.modules` for most of this suite.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    import tldw_chatbook

    repo_root = Path(tldw_chatbook.__file__).resolve().parent.parent
    script = (
        "import sys\n"
        "from tldw_chatbook.RAG_Search.simplified.active_config import "
        "resolve_active_rag_top_k\n"
        "resolve_active_rag_top_k()\n"
        "print('TORCH_IMPORTED' if 'torch' in sys.modules else 'NO_TORCH')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().splitlines()[-1] == "NO_TORCH"


def test_search_mode_only_resolution_does_not_import_torch():
    """The narrow mode resolver must remain safe for render-time use."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import tldw_chatbook

    repo_root = Path(tldw_chatbook.__file__).resolve().parent.parent
    script = (
        "import sys\n"
        "from tldw_chatbook.RAG_Search.simplified.active_config import "
        "resolve_active_rag_search_mode\n"
        "resolve_active_rag_search_mode()\n"
        "print('TORCH_IMPORTED' if 'torch' in sys.modules else 'NO_TORCH')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip().splitlines()[-1] == "NO_TORCH"
