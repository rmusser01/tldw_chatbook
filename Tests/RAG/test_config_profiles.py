import json
import json as _json

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


def test_legacy_blob_migration_isolates_per_entry_failures(tmp_path):
    # Review finding (Important): the whole `for pdata in ...` loop in
    # _migrate_legacy_blob was one try/except. If ProfileConfig.from_dict
    # raised on one entry (e.g. an old blob with a renamed/unknown
    # rag_config sub-key -> TypeError from `EmbeddingConfig(**data)`), the
    # WHOLE migration aborted: the blob was never renamed, and every OTHER
    # valid custom profile in the same blob silently vanished on upgrade.
    # Per-entry isolation must let the valid entries still migrate, and the
    # blob must still be renamed even though one entry failed.
    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig

    bad_entry = {
        "name": "Broken",
        "description": "d",
        "profile_type": "custom",
        # Unknown/renamed key -> EmbeddingConfig(**data) raises TypeError
        # inside RAGConfig.from_dict, which ProfileConfig.from_dict propagates.
        "rag_config": {"embedding": {"nonexistent_field_xyz": True}},
    }
    valid = ProfileConfig(
        name="Still Good", description="d", profile_type="custom",
        rag_config=RAGConfig(),
    )
    (pdir / "custom_profiles.json").write_text(
        _json.dumps({"profiles": [bad_entry, valid.to_dict()]}, default=str))

    m = _mgr(tmp_path)  # construction must not raise / must not abort entirely

    # The valid profile (listed AFTER the bad one) still migrated + is loadable.
    assert m.get_profile(valid.id) is not None
    assert (pdir / f"{valid.id}.json").exists()

    # The blob is renamed regardless of the partial failure.
    assert (pdir / "custom_profiles.json.migrated").exists()
    assert not (pdir / "custom_profiles.json").exists()


def test_cannot_mutate_builtins(tmp_path):
    m = _mgr(tmp_path)
    with pytest.raises(ValueError):
        m.delete_profile("hybrid_basic")
    with pytest.raises(ValueError):
        m.rename_profile("hybrid_basic", "Nope")
    with pytest.raises(ValueError):
        m.save_profile(m.get_profile("hybrid_basic"))  # read_only


def test_builtin_readonly_flag_cannot_be_flipped_to_bypass_guards(tmp_path):
    # Review finding (Important): delete_profile/rename_profile/save_profile
    # guarded on `profile.read_only`, checked on the SHARED ProfileConfig
    # instance that get_profile() returns. A caller could flip
    # `read_only = False` on that shared object and then delete/rename/save
    # a builtin -- save_profile's `existing is not profile` collision check
    # is bypassed entirely since it IS the same object. The guard must
    # additionally be keyed off an immutable, un-flippable record of which
    # ids are builtins (captured once at construction), not the mutable flag.
    m = _mgr(tmp_path)
    b = m.get_profile("hybrid_basic")
    b.read_only = False  # attacker flips the mutable flag on the shared object

    with pytest.raises(ValueError):
        m.delete_profile("hybrid_basic")
    with pytest.raises(ValueError):
        m.rename_profile("hybrid_basic", "X")
    with pytest.raises(ValueError):
        m.save_profile(b)

    # The builtin is intact: same object, unrenamed, still on the manager.
    assert m.get_profile("hybrid_basic") is b
    assert m.get_profile("hybrid_basic").name == "Hybrid Basic"
    assert not (tmp_path / "profiles" / "hybrid_basic.json").exists()


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


def test_load_does_not_clobber_unrelated_profile_on_disk_during_self_heal(tmp_path, monkeypatch):
    # Regression test (SP2a review): _load_custom_profiles's self-heal branch
    # reassigns a builtin-colliding profile's id via _unique_id, which only
    # checks self._profiles -- the in-memory dict being built INCREMENTALLY
    # during the glob loop. It does not check on-disk files. So if the glob
    # visits the collider file (high_accuracy.json, shadowing the read-only
    # builtin) BEFORE an unrelated, legitimate high_accuracy_2.json is loaded,
    # _unique_id returns "high_accuracy_2" (not yet in _profiles) and
    # _save_one overwrites the unrelated profile's file, silently destroying
    # it. Path.glob order is filesystem-dependent, so we force the
    # collider-first order here to make the failure deterministic; the fix
    # (_unique_id_reserving_disk) must be safe under either order.
    from pathlib import Path
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, EmbeddingConfig

    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)

    collider = ProfileConfig(
        id="high_accuracy", name="Collider", description="hand placed",
        profile_type="custom", read_only=False,
        rag_config=RAGConfig(embedding=EmbeddingConfig(model="collider-marker-model")),
    )
    (pdir / "high_accuracy.json").write_text(
        _json.dumps(collider.to_dict(), default=str))

    legit = ProfileConfig(
        id="high_accuracy_2", name="Legit Other Profile", description="unrelated",
        profile_type="custom", read_only=False,
        rag_config=RAGConfig(embedding=EmbeddingConfig(model="legit-other-marker-model")),
    )
    (pdir / "high_accuracy_2.json").write_text(
        _json.dumps(legit.to_dict(), default=str))

    # Force the collider-first glob order (the order under which the bug
    # manifests) so this test doesn't depend on filesystem-specific readdir
    # ordering. The fix must be safe regardless of order.
    real_glob = Path.glob

    def ordered_glob(self, pattern):
        results = list(real_glob(self, pattern))
        results.sort(key=lambda p: (p.name != "high_accuracy.json", p.name))
        return iter(results)

    monkeypatch.setattr(Path, "glob", ordered_glob)

    m = _mgr(tmp_path)

    # The read-only builtin is untouched.
    builtin = m.get_profile("high_accuracy")
    assert builtin.read_only is True
    assert builtin.rag_config.embedding.model == "BAAI/bge-large-en-v1.5"

    # The collider must be reassigned off BOTH "high_accuracy" (the builtin)
    # and "high_accuracy_2" (already claimed on disk by the unrelated
    # profile) -- landing on a third id.
    reassigned = [
        p for p in m._profiles.values()
        if p.rag_config.embedding.model == "collider-marker-model"
    ]
    assert len(reassigned) == 1
    assert reassigned[0].id == "high_accuracy_3"
    assert (pdir / "high_accuracy_3.json").exists()
    assert not (pdir / "high_accuracy.json").exists()

    # The unrelated legit profile must survive intact -- in memory...
    legit_loaded = m.get_profile("high_accuracy_2")
    assert legit_loaded is not None
    assert legit_loaded.name == "Legit Other Profile"
    assert legit_loaded.rag_config.embedding.model == "legit-other-marker-model"

    # ...and on disk (never overwritten by the collider's content).
    on_disk = json.loads((pdir / "high_accuracy_2.json").read_text())
    assert on_disk["name"] == "Legit Other Profile"
    assert on_disk["rag_config"]["embedding"]["model"] == "legit-other-marker-model"


def _force_glob_order(monkeypatch, first_name):
    # Same technique as the Addendum-2 collider test above: Path.glob order
    # is filesystem-dependent, so force a deterministic order by sorting the
    # real results with `first_name` pinned first.
    from pathlib import Path

    real_glob = Path.glob

    def ordered_glob(self, pattern):
        results = list(real_glob(self, pattern))
        results.sort(key=lambda p: (p.name != first_name, p.name))
        return iter(results)

    monkeypatch.setattr(Path, "glob", ordered_glob)


def test_load_does_not_clobber_sibling_file_that_slugifies_to_same_id(tmp_path, monkeypatch):
    # Regression test (SP2a review, second clobber hole): the self-heal in
    # _load_custom_profiles fires _save_one whenever canonical_path != path,
    # but the OLD code only reassigns/uniquifies the id inside the
    # `if desired_id in self._profiles:` branch -- a MEMORY-only check. If
    # the canonical file for `desired_id` belongs to a DIFFERENT,
    # not-yet-loaded profile (two distinct filenames that slugify to the
    # same id, e.g. "foo-bar.json" and "foo_bar.json" both -> "foo_bar"),
    # that on-disk collision is invisible to the check, and the self-heal
    # overwrites the other file's content before it's ever loaded --
    # silently and permanently destroying it.
    #
    # This test forces the vulnerable glob order (the non-canonical stem
    # "foo-bar.json" visited BEFORE the canonical "foo_bar.json") so the
    # failure is deterministic under the old code.
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, EmbeddingConfig

    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)

    profile_a = ProfileConfig(
        id="foo-bar", name="Profile A", description="dash-named",
        profile_type="custom", read_only=False,
        rag_config=RAGConfig(embedding=EmbeddingConfig(model="profile-a-marker-model")),
    )
    (pdir / "foo-bar.json").write_text(_json.dumps(profile_a.to_dict(), default=str))

    profile_b = ProfileConfig(
        id="foo_bar", name="Profile B", description="underscore-named",
        profile_type="custom", read_only=False,
        rag_config=RAGConfig(embedding=EmbeddingConfig(model="profile-b-marker-model")),
    )
    (pdir / "foo_bar.json").write_text(_json.dumps(profile_b.to_dict(), default=str))

    _force_glob_order(monkeypatch, "foo-bar.json")

    m = _mgr(tmp_path)

    loaded = list(m._profiles.values())
    a_matches = [p for p in loaded if p.rag_config.embedding.model == "profile-a-marker-model"]
    b_matches = [p for p in loaded if p.rag_config.embedding.model == "profile-b-marker-model"]

    # Both distinctive profiles must survive, each exactly once, under
    # different ids.
    assert len(a_matches) == 1, "Profile A missing or duplicated after self-heal"
    assert len(b_matches) == 1, "Profile B missing or duplicated after self-heal (clobbered)"
    assert a_matches[0].id != b_matches[0].id

    # Neither on-disk file was overwritten with the other's content.
    on_disk_names = {p.name for p in pdir.glob("*.json")}
    disk_models = {
        json.loads(p.read_text())["rag_config"]["embedding"]["model"]
        for p in pdir.glob("*.json")
    }
    assert "profile-a-marker-model" in disk_models
    assert "profile-b-marker-model" in disk_models
    assert len(on_disk_names) == 2


def test_load_does_not_clobber_sibling_file_reverse_glob_order(tmp_path, monkeypatch):
    # Same two colliding-slug files as above, but with the glob order
    # reversed (canonical "foo_bar.json" visited first). The fix must hold
    # regardless of which file the filesystem happens to yield first.
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, EmbeddingConfig

    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)

    profile_a = ProfileConfig(
        id="foo-bar", name="Profile A", description="dash-named",
        profile_type="custom", read_only=False,
        rag_config=RAGConfig(embedding=EmbeddingConfig(model="profile-a-marker-model")),
    )
    (pdir / "foo-bar.json").write_text(_json.dumps(profile_a.to_dict(), default=str))

    profile_b = ProfileConfig(
        id="foo_bar", name="Profile B", description="underscore-named",
        profile_type="custom", read_only=False,
        rag_config=RAGConfig(embedding=EmbeddingConfig(model="profile-b-marker-model")),
    )
    (pdir / "foo_bar.json").write_text(_json.dumps(profile_b.to_dict(), default=str))

    _force_glob_order(monkeypatch, "foo_bar.json")

    m = _mgr(tmp_path)

    loaded = list(m._profiles.values())
    a_matches = [p for p in loaded if p.rag_config.embedding.model == "profile-a-marker-model"]
    b_matches = [p for p in loaded if p.rag_config.embedding.model == "profile-b-marker-model"]

    assert len(a_matches) == 1, "Profile A missing or duplicated after self-heal"
    assert len(b_matches) == 1, "Profile B missing or duplicated after self-heal (clobbered)"
    assert a_matches[0].id != b_matches[0].id

    disk_models = {
        json.loads(p.read_text())["rag_config"]["embedding"]["model"]
        for p in pdir.glob("*.json")
    }
    assert "profile-a-marker-model" in disk_models
    assert "profile-b-marker-model" in disk_models


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


def test_profile_path_rejects_non_slug_id(tmp_path):
    # Review finding (Important), defense-in-depth: _profile_path built
    # `profiles_dir / f"{id}.json"` directly from whatever id it was given.
    # An id that is not already a bare slug (e.g. containing "..", "/", or
    # an absolute path) must be rejected outright so every file operation
    # keyed by id (_save_one, delete_profile, migration) can never escape
    # profiles_dir.
    m = _mgr(tmp_path)
    with pytest.raises(ValueError):
        m._profile_path("../x")
    with pytest.raises(ValueError):
        m._profile_path("../../tmp/pwned")
    with pytest.raises(ValueError):
        m._profile_path("/etc/passwd")
    with pytest.raises(ValueError):
        m._profile_path("Not A Slug")


def test_hand_edited_id_path_traversal_is_neutralized_by_filename_authority(tmp_path):
    # Review finding (Important): ProfileConfig.from_dict trusts the JSON's
    # internal "id" field verbatim (__post_init__ only backfills when it's
    # falsy). A hand-edited profile file with a crafted internal id like
    # "../../pwned" was registered in self._profiles under THAT unsafe key,
    # so any later save/delete keyed by id could write/unlink outside
    # profiles_dir. The FILENAME (stem) must be authoritative on load, never
    # the JSON's internal id.
    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig

    evil = ProfileConfig(
        id="../../pwned", name="Evil", description="d",
        profile_type="custom", rag_config=RAGConfig(), read_only=False,
    )
    (pdir / "evil.json").write_text(_json.dumps(evil.to_dict(), default=str))

    m = _mgr(tmp_path)

    # Registered under the SAFE stem-derived id, never the traversal id.
    assert m.get_profile("evil") is not None
    assert m.get_profile("../../pwned") is None
    assert "../../pwned" not in m._profiles
    assert m.get_profile("evil").id == "evil"

    # No file was ever written outside profiles_dir.
    assert not (tmp_path / "pwned.json").exists()
    assert not (tmp_path.parent / "pwned.json").exists()

    # And the safe id is fully usable through normal CRUD:
    assert m.delete_profile("evil") is True
    assert not (pdir / "evil.json").exists()


def test_load_uses_filename_stem_as_authoritative_id(tmp_path):
    # Review finding (Important), companion case: even a non-malicious
    # divergence between a profile file's name and its internal JSON id
    # (e.g. from manual editing, or copying a file to a new name) must not
    # be trusted -- the on-disk stem is the id, full stop. Otherwise files
    # get orphaned and delete_profile(<stem>) doesn't stick.
    pdir = tmp_path / "profiles"
    pdir.mkdir(parents=True)
    from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig
    from tldw_chatbook.RAG_Search.simplified.config import RAGConfig

    mismatched = ProfileConfig(
        id="b", name="Mismatched", description="d",
        profile_type="custom", rag_config=RAGConfig(), read_only=False,
    )
    (pdir / "a.json").write_text(_json.dumps(mismatched.to_dict(), default=str))

    m = _mgr(tmp_path)
    assert m.get_profile("a") is not None
    assert m.get_profile("a").id == "a"
    assert m.get_profile("b") is None

    assert m.delete_profile("a") is True
    assert m.get_profile("a") is None
    assert not (pdir / "a.json").exists()

    # Fresh manager: it must NOT reappear (e.g. from a leftover/duplicate file).
    m2 = _mgr(tmp_path)
    assert m2.get_profile("a") is None
    assert m2.get_profile("b") is None
