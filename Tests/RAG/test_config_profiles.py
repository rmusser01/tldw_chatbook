import json
import pytest
from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager, ProfileConfig
from tldw_chatbook.RAG_Search.simplified.config import (
    RAGConfig, EmbeddingConfig, ChunkingConfig, VectorStoreConfig,
)


def _profile():
    rag = RAGConfig(
        embedding=EmbeddingConfig(model="round-trip-model"),
        chunking=ChunkingConfig(chunk_size=333, chunk_overlap=77),
        # Pin vector_store to "memory" so persist_directory stays None. With
        # the "auto" default, VectorStoreConfig.__post_init__ resolves to
        # "chroma" (and a real PosixPath persist_directory) whenever the
        # embeddings_rag optional deps happen to be installed, which breaks
        # json.dumps below for reasons unrelated to what this test checks.
        vector_store=VectorStoreConfig(type="memory"),
    )
    return ProfileConfig(name="RT", description="d", profile_type="custom", rag_config=rag)


def test_round_trip_reconstructs_nested_dataclasses():
    p = _profile()
    restored = ProfileConfig.from_dict(json.loads(json.dumps(p.to_dict())))
    # These attribute accesses raise AttributeError today (sub-configs are dicts).
    assert isinstance(restored.rag_config, RAGConfig)
    assert isinstance(restored.rag_config.embedding, EmbeddingConfig)
    assert isinstance(restored.rag_config.chunking, ChunkingConfig)
    assert restored.rag_config.embedding.model == "round-trip-model"
    assert restored.rag_config.chunking.chunk_size == 333
    assert restored.rag_config.chunking.chunk_overlap == 77


def _mgr(tmp_path):
    return ConfigProfileManager(profiles_dir=tmp_path / "profiles")


def test_builtins_apply_declared_settings(tmp_path):
    m = _mgr(tmp_path)
    fast = m.get_profile("fast_search")
    assert fast.rag_config.chunking.chunk_size == 256      # was silently 400 (dead attr)
    assert fast.rag_config.chunking.chunk_overlap == 32
    assert fast.rag_config.search.default_top_k == 5

    # Builtins are meaningfully differentiated, not all default:
    sizes = {name: m.get_profile(name).rag_config.chunking.chunk_size
             for name in ("fast_search", "high_accuracy", "long_context")}
    assert len(set(sizes.values())) == 3, sizes


def test_builtins_use_valid_search_mode_and_store_type(tmp_path):
    m = _mgr(tmp_path)
    valid_modes = {"plain", "semantic", "hybrid"}
    valid_types = {"chroma", "memory", "auto"}
    for name in m.list_profiles():
        p = m.get_profile(name)
        assert p.rag_config.search.default_search_mode in valid_modes, name
        assert p.rag_config.vector_store.type in valid_types, name
    # bm25_only was keyword-only -> plain (value-mapped, not "keyword")
    assert m.get_profile("bm25_only").rag_config.search.default_search_mode == "plain"


def test_validate_profile_reads_real_fields(tmp_path):
    m = _mgr(tmp_path)
    bad = m.get_profile("hybrid_basic")
    # Force an overlap >= size on the REAL fields; validate must catch it.
    bad.rag_config.chunking.chunk_overlap = bad.rag_config.chunking.chunk_size + 1
    warnings = m.validate_profile(bad)
    assert any("overlap" in w.lower() for w in warnings)


def test_high_accuracy_has_no_stray_method_attr(tmp_path):
    m = _mgr(tmp_path)
    chunking = m.get_profile("high_accuracy").rag_config.chunking
    # '.method' is a typo of the real field 'chunking_method' -> must not exist as a stray attr
    assert "method" not in vars(chunking)
    assert chunking.chunking_method in {"words", "sentences", "paragraphs"}


def test_builtins_are_read_only_with_ids(tmp_path):
    m = _mgr(tmp_path)
    hb = m.get_profile("hybrid_basic")
    assert hb.read_only is True
    assert hb.id == "hybrid_basic"


def test_profileconfig_id_backfilled_and_round_trips():
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig, _slugify
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, VectorStoreConfig
    # Pin vector_store to "memory" (see _profile() above): with the "auto"
    # default, VectorStoreConfig.__post_init__ resolves to "chroma" (and a
    # real PosixPath persist_directory) whenever the embeddings_rag optional
    # deps happen to be installed, which breaks json.dumps below for reasons
    # unrelated to what this test checks (id/read_only round-tripping).
    p = ProfileConfig(name="My Cool Profile", description="d",
                      profile_type="custom",
                      rag_config=RAGConfig(vector_store=VectorStoreConfig(type="memory")))
    assert p.id == _slugify("My Cool Profile")  # auto-derived when not given
    import json
    restored = ProfileConfig.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored.id == p.id
    assert restored.read_only is False
    # Old files with no id/read_only keys still load (backfill):
    legacy = {k: v for k, v in p.to_dict().items() if k not in ("id", "read_only")}
    legacy_restored = ProfileConfig.from_dict(legacy)
    assert legacy_restored.id == _slugify("My Cool Profile")
    assert legacy_restored.read_only is False


import json as _json


def test_user_profile_saved_as_own_file(tmp_path):
    m = _mgr(tmp_path)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    p = ProfileConfig(name="Sales RAG", description="d", profile_type="custom",
                      rag_config=RAGConfig())
    m._save_one(p)  # (Task 5 adds save_profile; Task 4 uses _save_one directly)
    assert (tmp_path / "profiles" / f"{p.id}.json").exists()
    # A fresh manager over the same dir loads it back, correctly:
    m2 = _mgr(tmp_path)
    loaded = m2.get_profile(p.id)
    assert loaded is not None and loaded.read_only is False
    assert isinstance(loaded.rag_config, RAGConfig)


def test_legacy_blob_migrated_to_per_file(tmp_path):
    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    legacy = ProfileConfig(name="Legacy One", description="d",
                           profile_type="custom", rag_config=RAGConfig())
    (pdir / "custom_profiles.json").write_text(
        _json.dumps({"profiles": [legacy.to_dict()]}, default=str))
    m = _mgr(tmp_path)  # construction triggers load+migrate
    assert m.get_profile(legacy.id) is not None
    assert (pdir / f"{legacy.id}.json").exists()
    assert (pdir / "custom_profiles.json.migrated").exists()
    assert not (pdir / "custom_profiles.json").exists()
    # Idempotent: a second manager doesn't choke on the already-migrated blob.
    _mgr(tmp_path)


def test_cannot_mutate_builtins(tmp_path):
    m = _mgr(tmp_path)
    with pytest.raises(ValueError):
        m.delete_profile("hybrid_basic")
    with pytest.raises(ValueError):
        m.rename_profile("hybrid_basic", "Nope")
    with pytest.raises(ValueError):
        m.save_profile(m.get_profile("hybrid_basic"))  # read_only


def test_clone_builtin_creates_writable_copy(tmp_path):
    m = _mgr(tmp_path)
    clone = m.clone_profile("high_accuracy", "My Accuracy")
    assert clone.read_only is False
    assert clone.id != "high_accuracy"
    assert clone.rag_config.chunking.chunk_size == m.get_profile("high_accuracy").rag_config.chunking.chunk_size
    assert (tmp_path / "profiles" / f"{clone.id}.json").exists()
    # Editing the clone does not touch the builtin:
    clone.rag_config.chunking.chunk_size = 111
    m.save_profile(clone)
    assert m.get_profile("high_accuracy").rag_config.chunking.chunk_size != 111


def test_rename_keeps_id_and_file(tmp_path):
    m = _mgr(tmp_path)
    c = m.clone_profile("hybrid_basic", "Before")
    old_id = c.id
    renamed = m.rename_profile(c.id, "After")
    assert renamed.id == old_id                 # id stable across rename
    assert renamed.name == "After"
    assert (tmp_path / "profiles" / f"{old_id}.json").exists()


def test_delete_removes_file_and_entry(tmp_path):
    m = _mgr(tmp_path)
    c = m.clone_profile("hybrid_basic", "Temp")
    assert m.delete_profile(c.id) is True
    assert m.get_profile(c.id) is None
    assert not (tmp_path / "profiles" / f"{c.id}.json").exists()


def test_create_custom_profile_is_id_consistent(tmp_path):
    # Carry-forward fix from Task 4 review: create_custom_profile used to key
    # _profiles by name.lower().replace(" ", "_") while profile.id is
    # _slugify(name) (which collapses ALL non-alnum, not just spaces), so the
    # in-memory key could diverge from the id-based filename. It must now go
    # through save_profile() so the same key (profile.id) is used both in
    # memory and on disk.
    m = _mgr(tmp_path)
    created = m.create_custom_profile("My Custom! Profile", base_profile="balanced")
    assert m.get_profile(created.id) is created
    assert (tmp_path / "profiles" / f"{created.id}.json").exists()


def test_save_profile_rejects_id_collision_with_readonly_builtin(tmp_path):
    # Review finding (Critical): a builtin's id is _slugify(display_name), so
    # a brand-new non-read-only ProfileConfig named "High Accuracy" collides
    # with the builtin id "high_accuracy". save_profile only checked
    # `profile.read_only` on the INCOMING object, so it would silently
    # overwrite (and persist) the builtin, making it deletable/mutable.
    m = _mgr(tmp_path)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    builtin = m.get_profile("high_accuracy")
    assert builtin.read_only is True

    colliding = ProfileConfig(
        name="High Accuracy", description="d", profile_type="custom",
        rag_config=RAGConfig(),
    )
    assert colliding.id == "high_accuracy"
    assert colliding.read_only is False

    with pytest.raises(ValueError):
        m.save_profile(colliding)

    # The builtin must be untouched: same object, still read-only.
    assert m.get_profile("high_accuracy") is builtin
    assert m.get_profile("high_accuracy").read_only is True
    assert not (tmp_path / "profiles" / "high_accuracy.json").exists()


def test_create_custom_profile_uniquifies_id_matching_builtin_name(tmp_path):
    # Review finding (Critical), friendly-path companion: creating a custom
    # profile whose display name matches a builtin's must not raise and must
    # not collide with the builtin id -- it should get a uniquified id.
    m = _mgr(tmp_path)
    created = m.create_custom_profile("High Accuracy", base_profile="balanced")
    assert created.id != "high_accuracy"
    assert m.get_profile(created.id) is created

    builtin = m.get_profile("high_accuracy")
    assert builtin.read_only is True
    assert builtin.id == "high_accuracy"


def test_clone_tags_are_independent_of_source(tmp_path):
    # Review finding (Important): ProfileConfig.to_dict() did `"tags": self.tags`
    # (no copy), so a clone's tags list was the SAME list object as the
    # source's. Mutating the clone's tags corrupted the source builtin.
    m = _mgr(tmp_path)
    clone = m.clone_profile("high_accuracy", "My Accuracy")
    clone.tags.append("X")
    assert "X" not in m.get_profile("high_accuracy").tags


def test_legacy_blob_migration_does_not_shadow_readonly_builtin(tmp_path):
    # Review finding (Important): _migrate_legacy_blob wrote
    # self._profiles[profile.id] = profile directly, bypassing
    # save_profile's read-only collision guard. A legacy custom_profiles.json
    # blob containing a user profile named "High Accuracy" migrates to
    # high_accuracy.json and would overwrite the read-only builtin -- making
    # it deletable and persisting the overwrite on every future boot.
    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    legacy_user_profile = ProfileConfig(
        name="High Accuracy", description="user's own", profile_type="custom",
        rag_config=RAGConfig(),
    )
    (pdir / "custom_profiles.json").write_text(
        _json.dumps({"profiles": [legacy_user_profile.to_dict()]}, default=str))

    m = _mgr(tmp_path)  # construction triggers load+migrate

    builtin = m.get_profile("high_accuracy")
    assert builtin.read_only is True
    # Real builtin config, not the migrated user profile's defaults:
    assert builtin.rag_config.embedding.model == "BAAI/bge-large-en-v1.5"

    with pytest.raises(ValueError):
        m.delete_profile("high_accuracy")

    # The migrated user profile must be present under a DIFFERENT, writable id.
    others = [p for pid, p in m._profiles.items()
              if pid != "high_accuracy" and p.name == "High Accuracy"]
    assert len(others) == 1
    assert others[0].read_only is False
    assert others[0].id != "high_accuracy"
    assert (pdir / f"{others[0].id}.json").exists()


def test_hand_placed_file_named_like_builtin_id_is_self_healed(tmp_path):
    # Companion path for the same review finding: a hand-placed (or
    # otherwise stray) <builtin_id>.json file loaded directly via
    # _load_custom_profiles must not shadow the builtin either -- it must be
    # reassigned to a unique id and the colliding file removed from disk.
    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    colliding = ProfileConfig(
        id="high_accuracy", name="High Accuracy", description="hand placed",
        profile_type="custom", rag_config=RAGConfig(), read_only=False,
    )
    (pdir / "high_accuracy.json").write_text(
        _json.dumps(colliding.to_dict(), default=str))

    m = _mgr(tmp_path)

    builtin = m.get_profile("high_accuracy")
    assert builtin.read_only is True
    assert builtin.rag_config.embedding.model == "BAAI/bge-large-en-v1.5"

    with pytest.raises(ValueError):
        m.delete_profile("high_accuracy")

    others = [p for pid, p in m._profiles.items()
              if pid != "high_accuracy" and p.name == "High Accuracy"]
    assert len(others) == 1
    assert others[0].read_only is False
    new_id = others[0].id
    assert new_id != "high_accuracy"
    assert (pdir / f"{new_id}.json").exists()
    assert not (pdir / "high_accuracy.json").exists()


def test_create_custom_profile_returns_fully_built_rag_config(tmp_path):
    # Review finding (Important): create_custom_profile built rag_config via
    # RAGConfig(**asdict(base.rag_config)) -- the RAGConfig(**dict)
    # anti-pattern that leaves sub-configs (chunking, embedding, ...) as raw
    # dicts instead of dataclass instances, so
    # .rag_config.chunking.chunk_size raised AttributeError.
    m = _mgr(tmp_path)
    created = m.create_custom_profile("X", base_profile="balanced")
    assert isinstance(created.rag_config.chunking.chunk_size, int)
    assert (created.rag_config.chunking.chunk_size
            == m.get_profile("balanced").rag_config.chunking.chunk_size)


def test_save_reload_round_trips_explicit_persist_directory(tmp_path):
    # Exercises the Path/default=str save+reload branch for
    # persist_directory regardless of whether the embeddings_rag optional
    # deps are installed, by pinning vector_store explicitly instead of
    # relying on auto-resolution.
    from pathlib import Path
    m = _mgr(tmp_path)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, VectorStoreConfig
    persist_dir = tmp_path / "chroma_store"
    p = ProfileConfig(
        name="Chroma Explicit", description="d", profile_type="custom",
        rag_config=RAGConfig(
            vector_store=VectorStoreConfig(type="chroma", persist_directory=persist_dir)
        ),
    )
    m.save_profile(p)

    m2 = _mgr(tmp_path)
    reloaded = m2.get_profile(p.id)
    assert isinstance(reloaded.rag_config.vector_store.persist_directory, Path)
    assert reloaded.rag_config.vector_store.persist_directory == persist_dir


def test_save_profile_allows_user_over_user_same_id(tmp_path):
    # Only READ-ONLY collisions are rejected; two distinct user profiles that
    # happen to share an id may overwrite one another via save_profile.
    m = _mgr(tmp_path)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
    first = ProfileConfig(id="shared_id", name="First", description="d",
                          profile_type="custom", rag_config=RAGConfig(), read_only=False)
    second = ProfileConfig(id="shared_id", name="Second", description="d",
                           profile_type="custom", rag_config=RAGConfig(), read_only=False)
    m.save_profile(first)
    m.save_profile(second)  # must not raise
    assert m.get_profile("shared_id").name == "Second"
