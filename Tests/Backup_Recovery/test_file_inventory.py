"""Real-file recovery inventory, metadata and optional-content evidence."""

import os

from tldw_chatbook.Backup_Recovery.file_inventory import inventory_tree


def test_nested_inventory_serializes_windows_relative_paths(tmp_path, monkeypatch):
    from pathlib import Path, PureWindowsPath

    from tldw_chatbook.Backup_Recovery import file_inventory

    root = tmp_path / "tree"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "content").write_bytes(b"retained")

    def platform_path(value):
        return Path(value) if Path(value).is_absolute() else PureWindowsPath(value)

    monkeypatch.setattr(file_inventory, "Path", platform_path)
    items = file_inventory.inventory_tree(root, owner="assets", external=False)
    assert [item.metadata.relative_path for item in items] == ["", "nested", "nested/content"]


def test_empty_directory_is_an_inventory_item(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    items = inventory_tree(tmp_path, owner="test.files", external=True)
    assert any(
        item.path == empty and item.status == "included_directory" for item in items
    )


def test_directory_metadata_is_versioned_and_preserves_mode_and_mtime(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir(mode=0o750)
    os.utime(empty, ns=(1700000000000000000, 1700000000123456789))
    items = inventory_tree(tmp_path, owner="test.files", external=True)
    entry = next(item for item in items if item.path == empty)
    metadata = getattr(entry, "metadata", None)
    assert metadata is not None
    assert metadata.version == 1
    assert metadata.mode == 0o750
    assert metadata.mtime_ns == empty.stat().st_mtime_ns
    assert metadata.relative_path == "empty"
    assert metadata.parent_id == entry.dependencies[0]
    assert metadata.root_id == items[0].logical_id
    assert metadata.kind == "directory"
    assert metadata.policy == "external"


import ctypes
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery.inventory import classify_entries, discover
from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
    DiscoverySelections,
    StorageItem,
)
from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration


def test_nested_empty_tree_metadata_composes_and_scope_ignores_times_modes(tmp_path):
    root = tmp_path / "root"
    (root / "one" / "two").mkdir(parents=True)
    config = tmp_path / "config.toml"
    config.write_text("")
    context = DiscoveryContext(config, "p")
    cfg = {DISCOVERY_CONTEXT_KEY: context}
    declaration = _RawDeclaration("test.files")

    def inventory():
        return (
            StorageItem("config", "profile:p:config", config, "included", ()),
        ) + declaration._tree(cfg, root)

    before = classify_entries(inventory())
    assert before.complete, before.issues
    for item in inventory()[1:]:
        assert item.metadata.root_id.startswith("profile:p:test.files:")
        assert (
            item.metadata.parent_id is None
            or item.metadata.parent_id in item.dependencies
        )
        assert item.metadata.policy == "private"
    os.chmod(root / "one", 0o750)
    os.utime(root / "one", ns=(1, 2))
    assert classify_entries(inventory()).scope_digest == before.scope_digest


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "privilege"])
def test_unsupported_objects_are_not_included(tmp_path, kind):
    target = tmp_path / "target"
    target.write_bytes(b"owned")
    unsafe = tmp_path / "unsafe"
    if kind == "symlink":
        unsafe.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, unsafe)
    elif kind == "fifo":
        os.mkfifo(unsafe)
    else:
        unsafe.mkdir()
        unsafe.chmod(0o1700)
        assert unsafe.stat().st_mode & 0o1000
    entries = inventory_tree(tmp_path, owner="test.files", external=True)
    assert next(item for item in entries if item.path == unsafe).status == "unsupported"


def test_real_host_xattr_is_reported_unsupported(tmp_path):
    target = tmp_path / "asset"
    target.write_bytes(b"owned")
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        libc.setxattr.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
        ]
        assert (
            libc.setxattr(os.fsencode(target), b"org.chatbook.test", b"x", 1, 0, 0) == 0
        )
    else:
        os.setxattr(target, "user.chatbook", b"x")
    assert (
        next(
            item
            for item in inventory_tree(tmp_path, owner="test.files", external=True)
            if item.path == target
        ).status
        == "unsupported"
    )


def test_real_host_acl_is_reported_unsupported(tmp_path):
    target = tmp_path / "asset"
    target.write_bytes(b"owned")
    assert sys.platform == "darwin", (
        "This test qualifies the current native Darwin ACL policy only"
    )
    result = subprocess.run(
        ["/bin/chmod", "+a", "everyone allow read", str(target)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        next(
            item
            for item in inventory_tree(tmp_path, owner="test.files", external=True)
            if item.path == target
        ).status
        == "unsupported"
    )


def test_missing_explicit_external_root_is_unavailable(tmp_path):
    assert (
        inventory_tree(tmp_path / "missing", owner="external.files", external=True)[
            0
        ].status
        == "unavailable"
    )


def test_linked_ancestor_does_not_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"kept")
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    entries = inventory_tree(link / "secret", owner="test.files", external=True)
    assert entries[0].status == "unavailable"


def test_case_collisions_refuse_without_silently_omitting(tmp_path):
    # A case-sensitive pair is not available on every host; a literal backslash
    # still proves the cross-platform relative namespace rejects unsafe names.
    (tmp_path / "bad\\name").write_bytes(b"data")
    assert (
        inventory_tree(tmp_path, owner="test.files", external=True)[0].status
        == "unavailable"
    )


def test_explicit_external_selection_and_option_scope(tmp_path):
    config = tmp_path / "profile.toml"
    config.write_text(f'[paths]\ndata_dir = "{tmp_path / "data"}"\n')
    external = tmp_path / "external"
    external.mkdir()
    (external / "custom").write_bytes(b"asset")
    before = discover((config,))
    selected = discover(
        (config,), selections=DiscoverySelections(external_roots=(external,))
    )
    assert not any(item.path == external for item in before.items)
    assert any(
        item.path == external / "custom" and item.status == "included"
        for item in selected.items
    )
    assert before.scope_digest != selected.scope_digest
    assert not selected.complete  # Other unresolved real owner cohorts still block.
    missing = discover(
        (config,),
        selections=DiscoverySelections(external_roots=(tmp_path / "missing",)),
    )
    assert any(
        item.owner == "external.files" and item.status == "unavailable"
        for item in missing.items
    )


def test_selection_and_metadata_are_immutable_and_strict():
    with pytest.raises(ValueError):
        DiscoverySelections(external_roots=[])
    with pytest.raises(ValueError):
        DiscoverySelections(diagnostics=1)
    with pytest.raises(FrozenInstanceError):
        DiscoverySelections().diagnostics = True
    with pytest.raises(ValueError):
        discover((), selections={"external_roots": ()})


@pytest.mark.parametrize("package", ["TTS", "Persona_Visual", "Skills_Interop"])
def test_owner_package_import_is_inert_in_fresh_process(tmp_path, package):
    script = f"""
import sys
class BlockRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {{"tldw_chatbook.config", "tldw_chatbook.Utils.optional_deps", "torch", "httpx", "tldw_chatbook.DB.ChaChaNotes_DB"}}:
            raise AssertionError("runtime_import:" + fullname)
sys.meta_path.insert(0, BlockRuntime())
import importlib
importlib.import_module("tldw_chatbook.{package}")
"""
    env = dict(
        os.environ,
        HOME=str(tmp_path / "home"),
        TLDW_CONFIG_PATH=str(tmp_path / "absent.toml"),
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "home").exists()


def test_tts_current_profile_store_has_installed_capture_adapter(tmp_path):
    from tldw_chatbook.TTS.profile_schema import open_profile_store

    source = tmp_path / "profiles.db"
    connection = open_profile_store(source)
    try:
        catalog = tuple(
            row[0]
            for row in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
    finally:
        connection.close()
    from tldw_chatbook.TTS.recovery import recovery_adapters

    assert any(a.owner_id == "tts.profile_store" for a in recovery_adapters())
    adapter = next(a for a in recovery_adapters() if a.owner_id == "tts.profile_store")
    assert adapter.schema_policy().schema_sql == ((4, catalog),)
    assert adapter.validate(source) == ()


def test_tts_capture_preserves_real_profile_reference_bytes(tmp_path, monkeypatch):
    import hashlib
    import sqlite3
    from contextlib import closing
    from threading import Event

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.TTS.profile_schema import open_profile_store
    from tldw_chatbook.TTS.recovery import recovery_adapters

    source = tmp_path / "profiles.db"
    with closing(open_profile_store(source)) as connection:
        connection.execute(
            "INSERT INTO tts_generation_profiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "p",
                "Voice",
                "voice",
                "openai",
                "tts-1",
                "alloy",
                "wav",
                1.0,
                "{}",
                1,
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ),
        )
        payload = b"private-reference-bytes"
        connection.execute(
            "INSERT INTO tts_profile_clone_references VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "p",
                "r",
                payload,
                "Private voice transcript",
                hashlib.sha256(payload).hexdigest(),
                len(payload),
                1,
                8000,
                1,
                "pcm_s16le",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                None,
                None,
            ),
        )
        connection.commit()
    with closing(sqlite3.connect(source)) as connection:
        before = tuple(connection.iterdump())
    adapter = next(a for a in recovery_adapters() if a.owner_id == "tts.profile_store")
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    candidate = stage / "tts.db"
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            adapter.capture(
                StorageItem(
                    adapter.owner_id,
                    "profile:p:tts.profile_store",
                    source,
                    "included",
                    (),
                ),
                candidate,
                Event(),
            )
            adapter.relocate(candidate, {"ignored": tmp_path / "not-created"})
    with closing(sqlite3.connect(candidate)) as connection:
        assert tuple(connection.iterdump()) == before
        assert connection.execute(
            "SELECT wav_bytes, reference_text FROM tts_profile_clone_references"
        ).fetchone() == (payload, "Private voice transcript")
    assert adapter.activation_required
    assert not (tmp_path / "not-created").exists()
    with closing(sqlite3.connect(candidate)) as connection:
        connection.execute(
            "UPDATE tts_profile_clone_references SET sha256=?", ("0" * 64,)
        )
        connection.commit()
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            assert adapter.validate(candidate) == ("tts_reference_digest_mismatch",)


def test_config_mapping_keys_are_installed_selectors_and_prose_is_untouched(tmp_path):
    from tldw_chatbook.Backup_Recovery.config_adapter import (
        managed_secret_locations,
        remap_config_locations,
    )

    source = {
        "database": {"media_db_path": "/old/db"},
        "api_settings": {"openai": {"api_key": "encrypted:opaque"}},
        "notes": {"template": "My /old/db prose"},
        "unknown": {"path": "/old/db"},
    }
    remapped = remap_config_locations(
        source, {"database.media_db_path": tmp_path / "new.db"}
    )
    assert remapped["database"]["media_db_path"] == str(tmp_path / "new.db")
    assert source["database"]["media_db_path"] == "/old/db"
    assert remapped["notes"] == source["notes"]
    assert remapped["unknown"] == source["unknown"]
    assert managed_secret_locations(source) == (("api_settings", "openai", "api_key"),)
    assert remapped["api_settings"] == source["api_settings"]
    with pytest.raises(ValueError):
        remap_config_locations(source, {"/old/db": tmp_path / "new"})
    with pytest.raises(ValueError):
        remap_config_locations(
            source, {"paths.data_dir": tmp_path / "a", "Paths.data_dir": tmp_path / "b"}
        )


def test_current_history_config_captures_without_secret_decryption(
    tmp_path, monkeypatch
):
    from threading import Event

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters

    source = tmp_path / "custom.toml"
    source.write_text('[API]\nopenai_api_key = "encrypted:opaque"\n')
    history = source.with_suffix(".toml.bak")
    history.write_bytes(b"corrupt-but-recoverable-history")
    snapshot = tmp_path / "config_backup_20260908_120000.toml"
    snapshot.write_text("old_settings = true")
    config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(source, "p")}
    all_adapters = {a.owner_id: a for a in recovery_adapters()}
    adapters = (all_adapters["config"], all_adapters["config.history"])
    current = adapters[0].discover(config)[0]
    histories = adapters[1].discover(config)
    assert current.logical_id == "profile:p:config"
    assert current.dependencies == ()
    assert {item.path for item in histories} == {history, snapshot}
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            adapters[0].capture(current, stage / "config", Event())
    assert (stage / "config").read_bytes() == source.read_bytes()


def test_file_factories_return_actual_custom_assets_and_pending_participants(tmp_path):
    from tldw_chatbook.Backup_Recovery.config_adapter import (
        recovery_adapters as config_adapters,
    )
    from tldw_chatbook.Persona_Visual.recovery import (
        recovery_adapters as visual_adapters,
    )
    from tldw_chatbook.Skills_Interop.recovery import (
        recovery_adapters as skills_adapters,
    )

    data = tmp_path / "data" / "Ada"
    (data / "skills" / "skills" / "custom").mkdir(parents=True)
    skill = data / "skills" / "skills" / "custom" / "SKILL.md"
    skill.write_text("custom definition")
    (data / "persona_visual" / "packs").mkdir(parents=True)
    artwork = data / "persona_visual" / "packs" / "art.png"
    artwork.write_bytes(b"custom artwork")
    (data / "chunking_templates").mkdir()
    template = data / "chunking_templates" / "custom.json"
    template.write_text('{"custom":true}')
    selector = tmp_path / "config.toml"
    selector.write_text("")
    config = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Ada"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    entries = tuple(
        item
        for adapter in (*config_adapters(), *skills_adapters(), *visual_adapters())
        for item in adapter.discover(config)
    )
    for path in (skill, artwork, template):
        assert any(item.path == path and item.status == "included" for item in entries)
    assert any(
        item.logical_id.endswith(":participant_pending")
        and item.status == "unsupported"
        for item in entries
    )
    assert not classify_entries(entries).complete


def test_installed_model_selection_includes_exact_dependency_closure(tmp_path):
    import hashlib

    from Tests.Model_Artifacts.test_service import descriptor
    from tldw_chatbook.Model_Artifacts.recovery import recovery_adapters
    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactFile,
        ArtifactRef,
        ModelArtifactService,
    )

    data = tmp_path / "data" / "Ada"
    root = data / "models" / "managed"
    store = ModelArtifactService(root)
    parent = ArtifactRef("parent", "1", "int8")
    dep = ArtifactRef("dependency", "1", "int8")
    for ref, model_id, deps in (
        (dep, "dep-model", ()),
        (parent, "selected-model", (dep,)),
    ):
        directory = tmp_path / ref.artifact_id
        directory.mkdir()
        payload = b"model-bytes-" + ref.artifact_id.encode()
        (directory / "model.onnx").write_bytes(payload)
        desc = descriptor(
            reference=ref,
            model_id=model_id,
            dependencies=deps,
            files=(
                ArtifactFile(
                    "model.onnx", len(payload), hashlib.sha256(payload).hexdigest()
                ),
            ),
        )
        store.install(desc, directory)
    selector = tmp_path / "config.toml"
    selector.write_text("")
    base = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Ada"},
    }
    adapter = recovery_adapters()[0]
    base[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(selector, "p")
    unselected = adapter.discover(base)
    assert all(
        item.status == "intentionally_excluded"
        for item in unselected
        if item.path and item.path.name == "model.onnx"
    )
    base[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(
        selector, "p", DiscoverySelections(model_ids=("selected-model",))
    )
    selected = adapter.discover(base)
    included = {item.path for item in selected if item.status == "included"}
    assert {
        store.artifact_path(parent) / "model.onnx",
        store.artifact_path(dep) / "model.onnx",
    } <= included
    assert not any(item.logical_id.endswith(":participant_pending") for item in selected)
    assert not [
        item
        for item in selected
        if item.status == "unsupported"
        and not item.logical_id.endswith(":participant_pending")
    ]
    by_path = {item.path: item for item in selected}
    parent_manifest = by_path[store.artifact_path(parent) / "manifest.json"]
    assert (
        by_path[store.artifact_path(parent) / "model.onnx"].logical_id
        in parent_manifest.dependencies
    )
    assert (
        by_path[store.artifact_path(dep) / "manifest.json"].logical_id
        in parent_manifest.dependencies
    )
    candidates = {
        item.logical_id: item.path for item in selected if item.status == "included"
    }
    assert (
        adapter.validate_dependencies(parent_manifest, parent_manifest.path, candidates)
        == ()
    )
    dependency_manifest = by_path[store.artifact_path(dep) / "manifest.json"]
    assert (
        adapter.validate_dependencies(
            dependency_manifest, dependency_manifest.path, candidates
        )
        == ()
    )
    candidates.pop(dependency_manifest.logical_id)
    assert adapter.validate_dependencies(
        parent_manifest, parent_manifest.path, candidates
    ) == ("dependency_unavailable",)
    # Only int8 is installed; downloadable alternatives have no required bytes.
    assert not any(
        item.status in {"unavailable", "missing_required"} for item in selected
    )
    (store.artifact_path(dep) / "model.onnx").write_bytes(b"corrupted-dependency")
    corrupted = adapter.discover(base)
    assert (
        next(
            item
            for item in corrupted
            if item.path == store.artifact_path(dep) / "model.onnx"
        ).status
        == "unsupported"
    )


def test_streamed_digest_count_limit_and_cancel(tmp_path):
    import hashlib
    from threading import Event

    from tldw_chatbook.Backup_Recovery.storage_admission import _digest_recovery_file

    payload = b"payload" * 400_000
    source = tmp_path / "large.bin"
    source.write_bytes(payload)
    assert _digest_recovery_file(
        "models.artifacts", source, max_bytes=len(payload)
    ) == (len(payload), hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="definition_byte_limit"):
        _digest_recovery_file("models.artifacts", source, max_bytes=len(payload) - 1)
    cancel = Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        _digest_recovery_file(
            "models.artifacts", source, max_bytes=len(payload), cancel=cancel
        )
    with pytest.raises(RuntimeError):
        _digest_recovery_file("unregistered", source, max_bytes=len(payload))


@pytest.mark.parametrize("existing", [False, True])
def test_planned_output_exclusion_requires_positive_absence(tmp_path, existing):
    config = tmp_path / "profile.toml"
    config.write_text(f'[paths]\ndata_dir = "{tmp_path / "data"}"\n')
    root = tmp_path / "new-output"
    if existing:
        root.mkdir()
    result = discover(
        (config,), selections=DiscoverySelections(planned_output_root=root)
    )
    if existing:
        assert "config_discovery_failure" in result.issues
        assert not any(item.owner == "recovery.output" for item in result.items)
    else:
        assert any(
            item.path == root and item.status == "intentionally_excluded"
            for item in result.items
        )
    assert root.exists() is existing


def test_new_output_inside_baseline_refuses(tmp_path):
    config = tmp_path / "profile.toml"
    data = tmp_path / "data" / "Ada"
    data.mkdir(parents=True)
    config.write_text(
        f'[general]\nusers_name="Ada"\n[paths]\ndata_dir="{data.parent}"\n'
    )
    result = discover(
        (config,), selections=DiscoverySelections(planned_output_root=data / "new")
    )
    assert "config_discovery_failure" in result.issues
    assert not (data / "new").exists()


@pytest.mark.parametrize("damage", [None, "missing", "digest", "link"])
def test_actual_persona_publication_dependencies_and_missing_retained_assets(
    tmp_path, damage
):
    from Tests.Persona_Visual.test_persona_visual_publication import _snapshot
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.Persona_Visual.publication import publish_persona_visual
    from tldw_chatbook.Persona_Visual.recovery import recovery_adapters
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

    data = tmp_path / "data" / "Ada"
    data.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    database = tmp_path / "core.db"
    db = CharactersRAGDB(database, "recovery")
    try:
        publish_persona_visual(
            PersonaVisualRepository(db),
            _snapshot(source),
            source_root=source,
            profile_root=data,
            authority_guard=lambda: True,
        )
        asset = (
            data
            / db.get_connection()
            .execute("SELECT storage_relpath FROM persona_visual_assets")
            .fetchone()[0]
        )
    finally:
        db.close()
    if damage == "missing":
        asset.unlink()
    elif damage == "digest":
        asset.write_bytes(b"changed")
    elif damage == "link":
        asset.unlink()
        asset.symlink_to(source / "idle.png")
    selector = tmp_path / "profile.toml"
    selector.write_text("")
    config = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Ada"},
        "database": {"chachanotes_db_path": str(database)},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapter = recovery_adapters()[0]
    entries = adapter.discover(config)
    found = next(item for item in entries if item.path == asset)
    assert (
        found.status
        == {
            None: "included",
            "missing": "missing_required",
            "digest": "unsupported",
            "link": "unsupported",
        }[damage]
    )
    root = next(item for item in entries if item.path == data / "persona_visual")
    assert root.logical_id == "profile:p:persona.assets"
    assert found.logical_id in root.dependencies
    assert "profile:p:db.chachanotes.primary" in root.dependencies
    core = next(
        a for a in core_adapters() if a.owner_id == "db.chachanotes.primary"
    ).discover(config)[0]
    assert "profile:p:persona.assets" in core.dependencies
    assert "profile:p:persona.visual_identity" not in core.dependencies
    if damage is None:
        available = {
            entry.logical_id: entry.path for entry in entries if entry.path is not None
        }
        available[core.logical_id] = database
        available["profile:p:config"] = selector
        candidates = {key: available[key] for key in root.dependencies}
        assert adapter.validate_dependencies(root, root.path, candidates) == ()
        undeclared = replace(
            root,
            dependencies=tuple(
                key for key in root.dependencies if key != core.logical_id
            ),
        )
        assert adapter.validate_dependencies(undeclared, root.path, candidates) == (
            "dependency_unavailable",
        )
    wrong_profile = replace(root, logical_id="profile:other:persona.assets")
    assert adapter.validate_dependencies(
        wrong_profile, asset, {"profile:p:db.chachanotes.primary": database}
    ) == ("dependency_unavailable",)


def test_tts_reference_cohort_requires_actual_matching_physical_identity(tmp_path):
    from dataclasses import replace

    from tldw_chatbook.Backup_Recovery.inventory import _merge_chachanotes_cohort
    from tldw_chatbook.TTS.profile_schema import open_profile_store
    from tldw_chatbook.TTS.recovery import recovery_adapters

    source = tmp_path / "tts.db"
    open_profile_store(source).close()
    selector = tmp_path / "profile.toml"
    selector.write_text("")
    config = {
        "database": {"tts_profiles_db_path": str(source)},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapters = [
        a
        for a in recovery_adapters()
        if a.owner_id in {"tts.profile_store", "tts.references"}
    ]
    items = tuple(item for adapter in adapters for item in adapter.discover(config))
    merged, issues = _merge_chachanotes_cohort(items, cohort="tts")
    assert issues == ()
    assert merged[0].shared_group == merged[1].shared_group
    assert (
        "profile:p:tts.profile_store"
        in next(item for item in items if item.owner == "tts.references").dependencies
    )
    other = tmp_path / "other.db"
    open_profile_store(other).close()
    mismatched = (items[0], replace(items[1], path=other))
    assert _merge_chachanotes_cohort(mismatched, cohort="tts")[1] == (
        "shared_identity_mismatch",
    )


def test_saved_assets_baseline_temporary_and_diagnostics_opt_in(tmp_path):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters

    data = tmp_path / "data" / "Ada"
    saved = data / "generated_images" / "saved" / "saved.png"
    temp = data / "generated_images" / "temp" / "temp.png"
    video = data / "generated_videos" / "message" / "video.mp4"
    for path in (saved, temp, video):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"asset")
    log = data / "custom.log"
    log.write_text("diagnostic")
    selector = tmp_path / "profile.toml"
    selector.write_text("")
    config = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Ada"},
        "logging": {"log_filename": "custom.log"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapters = [
        a
        for a in recovery_adapters()
        if a.owner_id in {"generation.assets", "diagnostics.logs"}
    ]

    def entries():
        return tuple(item for adapter in adapters for item in adapter.discover(config))

    default = entries()
    assert next(item for item in default if item.path == saved).status == "included"
    assert (
        next(item for item in default if item.path == temp).status
        == "intentionally_excluded"
    )
    assert (
        next(item for item in default if item.path == log).status
        == "intentionally_excluded"
    )
    assert not any(item.path == video for item in default)
    config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(
        selector, "p", DiscoverySelections(temporary_media=True, diagnostics=True)
    )
    selected = entries()
    assert all(
        next(item for item in selected if item.path == path).status == "included"
        for path in (saved, temp, video, log)
    )


@pytest.fixture
def installed_model(tmp_path):
    import hashlib

    from Tests.Model_Artifacts.test_service import descriptor
    from tldw_chatbook.Model_Artifacts.recovery import recovery_adapters
    from tldw_chatbook.Model_Artifacts.service import ArtifactFile, ModelArtifactService

    store = ModelArtifactService(tmp_path / "data" / "Ada" / "models" / "managed")
    source = tmp_path / "source"
    source.mkdir()
    payload = b"actual model payload"
    (source / "model.onnx").write_bytes(payload)
    desc = descriptor(
        files=(
            ArtifactFile(
                "model.onnx", len(payload), hashlib.sha256(payload).hexdigest()
            ),
        )
    )
    store.install(desc, source)
    selector = tmp_path / "config.toml"
    selector.write_text("")
    config = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Ada"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(
            selector, "p", DiscoverySelections(model_ids=(desc.model_id,))
        ),
    }
    return store, desc, config, recovery_adapters()[0]


@pytest.mark.parametrize(
    "damage",
    ["missing", "link", "duplicate_manifest", "pending_download", "unknown_sibling"],
)
def test_installed_model_states_refuse_without_silent_omission(
    installed_model, tmp_path, damage
):
    store, desc, config, adapter = installed_model
    directory = store.artifact_path(desc.reference)
    path = directory / "model.onnx"
    if damage == "missing":
        path.unlink()
    elif damage == "link":
        path.unlink()
        path.symlink_to(tmp_path / "source" / "model.onnx")
    elif damage == "duplicate_manifest":
        path = directory / "manifest.json"
        path.write_text(path.read_text().replace("{", '{"schema_version":1,', 1))
    elif damage == "pending_download":
        stage = store._download_stage_for(desc, create=True)
        path = stage.marker
        assert path.is_file()
    else:
        path = directory.parents[4] / "unknown-model-state"
        path.write_bytes(b"durable unknown")
    before = path.read_bytes() if path.exists() else None
    entries = adapter.discover(config)
    assert any(
        item.path == path and item.status in {"unsupported", "missing_required"}
        for item in entries
    )
    assert not classify_entries(entries).complete
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize(
    "damage", ["corrupt", "missing", "wrong_profile", "changed_ref"]
)
def test_selected_model_capture_uses_checked_source_and_preserves_recipe(
    installed_model, tmp_path, monkeypatch, damage
):
    from threading import Event

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.admission import _local

    store, desc, config, adapter = installed_model
    entries = adapter.discover(config)
    directory = store.artifact_path(desc.reference)
    authority = application_authority(tmp_path, directory, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope(
            (directory / "model.onnx", directory / "manifest.json"), stage
        ):
            for name in ("model.onnx", "manifest.json"):
                item = next(item for item in entries if item.path == directory / name)
                adapter.capture(item, stage / name, Event())
                assert (stage / name).read_bytes() == (directory / name).read_bytes()
                assert _local.capture_scope.resources == []

            manifest = next(
                item for item in entries if item.path == directory / "manifest.json"
            )
            candidates = {
                item.logical_id: stage / item.path.name
                for item in entries
                if item.path in {directory / "manifest.json", directory / "model.onnx"}
            }
            assert (
                adapter.validate_dependencies(
                    manifest, stage / "manifest.json", candidates
                )
                == ()
            )
            if damage == "corrupt":
                (stage / "model.onnx").write_bytes(b"corrupt staged model")
                expected = "model_payload_mismatch"
            elif damage == "missing":
                candidates = {
                    key: path
                    for key, path in candidates.items()
                    if path.name != "model.onnx"
                }
                expected = "dependency_unavailable"
            elif damage == "wrong_profile":
                candidates = {
                    key.replace("profile:p:", "profile:other:"): path
                    for key, path in candidates.items()
                }
                expected = "dependency_unavailable"
            else:
                import json

                raw = json.loads((stage / "manifest.json").read_text())
                raw["descriptor"]["reference"]["revision"] = "changed"
                (stage / "manifest.json").write_text(json.dumps(raw))
                expected = "model_identity_mismatch"
            assert adapter.validate_dependencies(
                manifest, stage / "manifest.json", candidates
            ) == (expected,)


def test_custom_kokoro_voice_root_is_baseline(tmp_path):
    from tldw_chatbook.TTS.recovery import recovery_adapters

    voice = tmp_path / "custom" / "blend.pt"
    voice.parent.mkdir()
    voice.write_bytes(b"inert blend")
    selector = tmp_path / "profile.toml"
    selector.write_text("")
    config = {
        "app_tts": {"KOKORO_VOICE_BLENDS_DIR": str(voice.parent)},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    entries = recovery_adapters()[0].discover(config)
    assert next(item for item in entries if item.path == voice).status == "included"


def test_chat_attachment_adapter_uses_exact_core_blob_cohort(tmp_path, monkeypatch):
    import sqlite3
    from contextlib import closing
    from threading import Event

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
    from tldw_chatbook.Backup_Recovery.inventory import _merge_chachanotes_cohort
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.recovery_core import core_adapters

    source = tmp_path / "core.db"
    db = CharactersRAGDB(source, "recovery")
    try:
        conversation = db.add_conversation({"title": "captured attachments"})
        message = db.add_message(
            {"conversation_id": conversation, "sender": "user", "content": "attachment"}
        )
        db.set_message_attachments(
            message,
            [
                {
                    "position": 1,
                    "data": b"\x00attachment\xff",
                    "mime_type": "application/octet-stream",
                    "display_name": "evidence.bin",
                }
            ],
        )
    finally:
        db.close()
    selector = tmp_path / "profile.toml"
    selector.write_text("")
    config = {
        "database": {"chachanotes_db_path": str(source)},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapter = next(a for a in recovery_adapters() if a.owner_id == "chat.attachments")
    attachment = adapter.discover(config)[0]
    core = core_adapters()[0].discover(config)[0]
    assert _merge_chachanotes_cohort((core, attachment))[1] == ()
    assert adapter.validate(source) == ()
    assert _merge_chachanotes_cohort(
        (core, replace(attachment, shared_group="forged"))
    )[1]

    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            adapter.capture(attachment, stage / "core.db", Event())
    with closing(sqlite3.connect(stage / "core.db")) as captured:
        assert (
            captured.execute("SELECT data FROM message_attachments").fetchone()[0]
            == b"\x00attachment\xff"
        )


@pytest.mark.parametrize("preview", ["preview.png", "missing.png", "../escape.png"])
def test_visual_identity_owned_preview_and_exact_staged_references(
    tmp_path, preview, monkeypatch
):
    import hashlib

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.Persona_Visual.recovery import recovery_adapters

    data = tmp_path / "data" / "Ada"
    root = data / "visual_identities"
    root.mkdir(parents=True)
    payload = b"retained manual asset"
    (root / "portrait.png").write_bytes(payload)
    (root / "preview.png").write_bytes(b"retained preview")
    source = tmp_path / "core.db"
    db = CharactersRAGDB(source, "fixture")
    try:
        actor = db.add_character_card({"name": "Manual identity"})
        VisualIdentityRepository(db).activate_pack(
            pack={
                "title": "manual",
                "default_expression_key": "neutral",
                "source_kind": "manual",
            },
            manifest={},
            assets=[
                {
                    "expression_key": "neutral",
                    "original_expression_key": "neutral",
                    "source_filename": "portrait.png",
                    "storage_relpath": "portrait.png",
                    "content_type": "image/png",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "width": 1,
                    "height": 1,
                    "preview_relpath": preview,
                }
            ],
            actor_kind="character",
            actor_id=actor,
        )
    finally:
        db.close()
    selector = tmp_path / "config.toml"
    selector.write_text("")
    config = {
        "paths": {"data_dir": str(tmp_path / "data")},
        "general": {"users_name": "Ada"},
        "database": {"chachanotes_db_path": str(source)},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapter = next(
        a for a in recovery_adapters() if a.owner_id == "persona.visual_identity"
    )
    entries = adapter.discover(config)
    item = next(item for item in entries if item.path == root)
    if preview == "preview.png":
        core_key = "profile:p:db.chachanotes.primary"
        assert core_key in item.dependencies
        available = {
            entry.logical_id: entry.path for entry in entries if entry.path is not None
        }
        available[core_key] = source
        available["profile:p:config"] = selector
        candidates = {key: available[key] for key in item.dependencies}
        assert adapter.validate_dependencies(item, root, candidates) == ()
        peer_reads = []
        original_references = type(adapter)._references

        def observe_references(self, peer):
            peer_reads.append(peer)
            return original_references(self, peer)

        monkeypatch.setattr(type(adapter), "_references", observe_references)
        undeclared = replace(
            item,
            dependencies=tuple(key for key in item.dependencies if key != core_key),
        )
        assert adapter.validate_dependencies(undeclared, root, candidates) == (
            "dependency_unavailable",
        )
        assert peer_reads == []
        key = next(
            entry.logical_id for entry in entries if entry.path == root / preview
        )
        assert key in item.dependencies
        candidates.pop(key)
        assert adapter.validate_dependencies(item, root, candidates) == (
            "dependency_unavailable",
        )
    elif preview == "missing.png":
        assert any(
            entry.path == root / preview and entry.status == "missing_required"
            for entry in entries
        )
    else:
        assert any(
            entry.logical_id.endswith(":references_unavailable") for entry in entries
        )


def test_selected_tree_never_traverses_siblings_or_expands_empty_selection(tmp_path):
    from tldw_chatbook.Backup_Recovery.file_inventory import _inventory_tree

    selected = tmp_path / "needed" / "asset"
    selected.parent.mkdir()
    selected.write_bytes(b"referenced")
    (tmp_path / "unrelated").symlink_to(tmp_path / "missing", target_is_directory=True)
    items = _inventory_tree(
        tmp_path,
        owner="test.files",
        external=False,
        selected_paths=frozenset({"needed/asset"}),
    )
    assert {item.path for item in items} == {tmp_path, selected.parent, selected}
    assert all(item.status in {"included", "included_directory"} for item in items)
    assert (
        _inventory_tree(
            tmp_path, owner="test.files", external=False, selected_paths=frozenset()
        )
        == ()
    )
    with pytest.raises(ValueError):
        _inventory_tree(
            tmp_path,
            owner="test.files",
            external=False,
            selected_paths=frozenset({"needed/a", "Needed/b"}),
        )


def test_actual_builtin_asset_reference_uses_only_installed_package_bytes(tmp_path):
    import hashlib

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
    from tldw_chatbook.Persona_Visual.recovery import recovery_adapters

    package_root = Path(__file__).parents[2] / "tldw_chatbook" / "assets"
    path = package_root / "characters" / "samira" / "Sammy.png"
    payload = path.read_bytes()
    source = tmp_path / "core.db"
    db = CharactersRAGDB(source, "fixture")
    try:
        actor = db.add_character_card({"name": "Builtin identity"})
        VisualIdentityRepository(db).activate_pack(
            pack={
                "title": "builtin",
                "default_expression_key": "neutral",
                "source_kind": "builtin",
            },
            manifest={},
            assets=[
                {
                    "expression_key": "neutral",
                    "original_expression_key": "neutral",
                    "source_filename": path.name,
                    "storage_relpath": path.relative_to(package_root).as_posix(),
                    "content_type": "image/png",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "width": 1,
                    "height": 1,
                }
            ],
            actor_kind="character",
            actor_id=actor,
        )
    finally:
        db.close()
    selector = tmp_path / "profile.toml"
    selector.write_text("")
    config = {
        "database": {"chachanotes_db_path": str(source)},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapters = {a.owner_id: a for a in recovery_adapters()}
    entries = adapters["persona.visual_identity_builtin"].discover(config)
    assert {item.path for item in entries if item.status == "included"} == {path}
    assert all(
        item.path is None or item.path == path or item.path in path.parents
        for item in entries
    )
    assert (
        "profile:p:persona.visual_identity_builtin"
        in core_adapters()[0].discover(config)[0].dependencies
    )
    assert (
        "profile:p:persona.visual_identity"
        not in core_adapters()[0].discover(config)[0].dependencies
    )
    adapter = adapters["persona.visual_identity_builtin"]
    root = next(
        item
        for item in entries
        if item.logical_id == "profile:p:persona.visual_identity_builtin"
    )
    core_key = "profile:p:db.chachanotes.primary"
    assert core_key in root.dependencies
    available = {
        entry.logical_id: entry.path for entry in entries if entry.path is not None
    }
    available[core_key] = source
    available["profile:p:config"] = selector
    candidates = {key: available[key] for key in root.dependencies}
    assert adapter.validate_dependencies(root, root.path, candidates) == ()
    undeclared = replace(
        root, dependencies=tuple(key for key in root.dependencies if key != core_key)
    )
    assert adapter.validate_dependencies(undeclared, root.path, candidates) == (
        "dependency_unavailable",
    )
    owned = adapters["persona.visual_identity"].discover(config)
    assert not any(item.status == "missing_required" for item in owned)


def test_actual_host_file_flags_refuse_without_clearing_them(tmp_path):
    import stat

    assert sys.platform == "darwin", "This case qualifies the current Darwin host only"
    path = tmp_path / "flagged"
    path.write_bytes(b"flagged bytes")
    os.chflags(path, stat.UF_HIDDEN)
    try:
        assert path.stat().st_flags & stat.UF_HIDDEN
        item = inventory_tree(path, owner="test.files", external=True)[0]
        assert item.status == "unsupported"
        assert path.stat().st_flags & stat.UF_HIDDEN
    finally:
        os.chflags(path, 0)


def test_planned_output_cannot_claim_an_absent_installed_owner_path(tmp_path):
    from tldw_chatbook.Backup_Recovery.inventory import _planned_output_exclusion

    reserved = tmp_path / "note_templates.json"
    context = DiscoveryContext(
        tmp_path / "profile.toml",
        "p",
        DiscoverySelections(planned_output_root=reserved),
    )
    declared = (
        StorageItem(
            "notes.templates", "profile:p:notes.templates", reserved, "unused", ()
        ),
    )
    with pytest.raises(ValueError, match="output_overlaps_baseline"):
        _planned_output_exclusion(context, declared, tmp_path / "data")
    assert not reserved.exists()


@pytest.mark.asyncio
async def test_remaining_installed_preferences_history_and_chatbooks_are_baseline(
    tmp_path, monkeypatch
):
    from threading import Event

    from Tests.Backup_Recovery.test_core_owners import application_authority
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
    from tldw_chatbook.Chat.prompt_history import PromptHistory
    from tldw_chatbook.Chatbooks.local_chatbook_service import LocalChatbookService

    data = tmp_path / "data" / "Ada"
    data.mkdir(parents=True)
    selector = tmp_path / "profile" / "custom.toml"
    selector.parent.mkdir()
    selector.write_text("")
    history = data / "prompt_history.jsonl"
    await PromptHistory(history).append("private retained prompt")
    state = selector.parent / "ui_state.toml"
    state.write_text('[sidebar]\nsearch_query="retained"\n')
    emojis = selector.parent / "recent_emojis.json"
    emojis.write_text('{"recent":["hello"]}')
    theme = selector.parent / "themes" / "retained.toml"
    theme.parent.mkdir()
    theme.write_text('[theme]\nname="retained"')
    archives = data / "chatbooks"
    archives.mkdir()
    archive = archives / "retained.zip"
    archive.write_bytes(b"opaque retained content export")
    prompts = tmp_path / "custom-db" / "prompts.db"
    prompts.parent.mkdir()
    registry = prompts.with_name("tldw_chatbook_chatbooks.json")
    service = LocalChatbookService(
        {"Prompts": str(prompts), "ChaChaNotes": str(tmp_path / "other.db")}
    )
    await service.create_chatbook(name="retained", file_path=archive)
    config = {
        "paths": {"data_dir": str(data.parent)},
        "general": {"users_name": "Ada"},
        "database": {"prompts_db_path": str(prompts)},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapters = {adapter.owner_id: adapter for adapter in recovery_adapters()}
    expected = {
        "chat.prompt_history": history,
        "ui.state": state,
        "ui.emoji_recents": emojis,
        "ui.themes": theme,
        "chatbooks.registry": registry,
        "chatbooks.archives": archive,
    }
    assert expected.keys() <= adapters.keys()
    for owner, path in expected.items():
        entries = adapters[owner].discover(config)
        assert next(item for item in entries if item.path == path).status == "included"
        pending = any(
            item.logical_id.endswith(":participant_pending") for item in entries
        )
        assert not pending
    # The default archive directory owns every retained ordinary export, even
    # a backup-looking extension; registry external destinations remain inert.
    backup_named = archives / "ordinary-content.tldw-backup.zip"
    backup_named.write_bytes(b"retained ordinary content")
    assert (
        next(
            i
            for i in adapters["chatbooks.archives"].discover(config)
            if i.path == backup_named
        ).status
        == "included"
    )
    await service.create_chatbook(
        name="external", file_path=tmp_path / "outside" / "missing.zip"
    )
    assert (
        len([i for i in adapters["chatbooks.registry"].discover(config) if i.path]) == 1
    )
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    for owner, path in expected.items():
        control = tmp_path / ("authority-" + owner)
        control.mkdir()
        authority = application_authority(control, path, monkeypatch)
        with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
            with session.capture_scope((path,), stage):
                adapter = adapters[owner]
                item = next(i for i in adapter.discover(config) if i.path == path)
                adapter.capture(item, stage / owner, Event())
                assert (stage / owner).read_bytes() == path.read_bytes()
    registry_item = next(
        i for i in adapters["chatbooks.registry"].discover(config) if i.path == registry
    )
    archive_item = next(
        i for i in adapters["chatbooks.archives"].discover(config) if i.path == archive
    )
    assert set(registry_item.dependencies) == {
        "profile:p:config",
        "profile:p:db.prompts.primary",
        archive_item.logical_id,
    }


def test_chatbook_scratch_has_exact_producers_cleanup_and_unknown_sibling_refusal(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters
    from tldw_chatbook.Chatbooks import chatbook_creator, chatbook_importer

    data = tmp_path / "data" / "Ada"
    data.mkdir(parents=True)
    selector = tmp_path / "config.toml"
    selector.write_text("")
    monkeypatch.setattr(chatbook_creator, "get_user_data_dir", lambda: data)
    monkeypatch.setattr(chatbook_importer, "get_user_data_dir", lambda: data)
    creator = chatbook_creator.ChatbookCreator({})
    importer = chatbook_importer.ChatbookImporter({})
    assert creator.temp_dir == data / "temp" / "chatbooks"
    assert importer.temp_dir == data / "temp" / "imports"
    output = tmp_path / "content.zip"
    success, _, _ = creator.create_chatbook("empty", "", {}, output)
    assert success
    assert not tuple(creator.temp_dir.iterdir())
    manifest, error = importer.preview_chatbook(output)
    assert manifest is not None and error is None
    assert not tuple(importer.temp_dir.iterdir())
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"invalid archive")
    assert importer.preview_chatbook(broken)[0] is None
    assert not tuple(importer.temp_dir.iterdir())
    config = {
        "paths": {"data_dir": str(data.parent)},
        "general": {"users_name": "Ada"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapter = next(
        a for a in recovery_adapters() if a.owner_id == "runtime.chatbook_scratch"
    )
    # Stale per-run inputs are disposable; unrelated bytes at this common parent
    # have no source-purpose proof and must not inherit either exclusion.
    for root in (creator.temp_dir, importer.temp_dir):
        (root / "stale-run").mkdir()
        (root / "stale-run" / "manifest.json").write_text("{}")
    unrelated = data / "temp" / "retained-project"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("durable user bytes")
    entries = adapter.discover(config)
    assert entries[0].path == data / "temp"
    assert entries[0].status == "included_directory"
    assert entries[0].metadata is not None
    for item in entries[1:]:
        expected = (
            "unsupported"
            if item.path.is_relative_to(unrelated)
            else "intentionally_excluded"
        )
        assert item.status == expected
    assert (unrelated / "notes.txt").read_text() == "durable user bytes"
    # The installed scratch selector is a directory, not an arbitrary file with
    # the same name. Preserve refusal for a wrong-kind replacement.
    import shutil

    shutil.rmtree(importer.temp_dir)
    importer.temp_dir.write_bytes(b"not an installed scratch directory")
    assert (
        next(i for i in adapter.discover(config) if i.path == importer.temp_dir).status
        == "unsupported"
    )


def test_instance_lock_is_exact_process_exclusion_and_preserves_link_refusal(tmp_path):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters

    data = tmp_path / "Ada"
    data.mkdir()
    selector = tmp_path / "config.toml"
    selector.write_text("")
    config = {
        "paths": {"data_dir": str(tmp_path)},
        "general": {"users_name": "Ada"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapter = next(
        a for a in recovery_adapters() if a.owner_id == "runtime.instance_lock"
    )
    lock = data / ".instance.lock"
    lock.write_text("pid=123\nsince=456\n")
    assert adapter.discover(config)[0].status == "intentionally_excluded"
    lock.unlink()
    lock.symlink_to(selector)
    assert adapter.discover(config)[0].status == "unsupported"
    assert selector.read_text() == ""


@pytest.mark.parametrize(
    "owner,relative,kind",
    [
        ("generation.assets", "generated_images/temp", "directory"),
        ("generation.assets", "generated_videos", "directory"),
        ("cache.model_catalog", "model_catalog_cache.json", "file"),
        ("diagnostics.logs", "custom.log", "file"),
    ],
)
@pytest.mark.parametrize("damage", ["valid", "wrong_kind", "link", "metadata"])
def test_optional_exclusions_preserve_wrong_kind_link_and_metadata(
    tmp_path, owner, relative, kind, damage
):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters

    data = tmp_path / "data" / "Ada"
    target = data / relative
    target.parent.mkdir(parents=True)
    selector = tmp_path / "profile.toml"
    selector.write_text("")
    config = {
        "paths": {"data_dir": str(data.parent)},
        "general": {"users_name": "Ada"},
        "logging": {"log_filename": "custom.log"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    if damage == "link":
        target.symlink_to(selector)
    elif (kind == "directory") != (damage == "wrong_kind"):
        target.mkdir()
    else:
        target.write_bytes(b"local bytes")
    if damage == "metadata":
        # Actual ordinary owned inode with unsupported privilege bits, which
        # the shared pinned metadata observer must detect even when excluded.
        target.chmod(target.stat().st_mode | 0o1000)
    adapter = next(a for a in recovery_adapters() if a.owner_id == owner)
    item = next(i for i in adapter.discover(config) if i.path == target)
    assert item.status == (
        "intentionally_excluded" if damage == "valid" else "unsupported"
    )


@pytest.mark.parametrize("damage", ["link", "fifo", "hardlink", "xattr"])
def test_optional_exclusions_preserve_unsafe_generated_temp_children(tmp_path, damage):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters

    data = tmp_path / "data" / "Ada"
    target = data / "generated_images" / "temp" / "unsafe"
    target.parent.mkdir(parents=True)
    selector = tmp_path / "profile.toml"
    selector.write_text("")
    if damage == "link":
        target.symlink_to(selector)
    elif damage == "fifo":
        os.mkfifo(target)
    elif damage == "hardlink":
        os.link(selector, target)
    else:
        target.write_bytes(b"bytes")
        assert sys.platform == "darwin", "Actual host metadata qualification"
        libc = ctypes.CDLL(None, use_errno=True)
        libc.setxattr.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
        ]
        assert (
            libc.setxattr(os.fsencode(target), b"org.chatbook.review", b"x", 1, 0, 0)
            == 0
        )
    config = {
        "paths": {"data_dir": str(data.parent)},
        "general": {"users_name": "Ada"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapter = next(a for a in recovery_adapters() if a.owner_id == "generation.assets")
    item = next(i for i in adapter.discover(config) if i.path == target)
    assert item.status == "unsupported"


def test_optional_exclusions_root_checks_do_not_traverse_unrelated_children(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters

    data = tmp_path / "data" / "Ada"
    roots = {
        data / "generated_videos",
        data / "model_catalog_cache.json",
        data / "custom.log",
    }
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        (root / "unrelated").mkdir()
        os.mkfifo(root / "unrelated" / "pipe")
    identities = {(p.stat().st_dev, p.stat().st_ino) for p in roots}
    real_scandir = os.scandir

    def observe_scandir(path):
        info = os.fstat(path) if isinstance(path, int) else os.stat(path)
        assert (info.st_dev, info.st_ino) not in identities, (
            "excluded/wrong-kind root was traversed"
        )
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", observe_scandir)
    selector = tmp_path / "config.toml"
    selector.write_text("")
    config = {
        "paths": {"data_dir": str(data.parent)},
        "general": {"users_name": "Ada"},
        "logging": {"log_filename": "custom.log"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    entries = tuple(
        i
        for a in recovery_adapters()
        if a.owner_id
        in {"generation.assets", "cache.model_catalog", "diagnostics.logs"}
        for i in a.discover(config)
    )
    assert (
        next(i for i in entries if i.path == data / "generated_videos").status
        == "intentionally_excluded"
    )
    assert all(
        next(i for i in entries if i.path == path).status == "unsupported"
        for path in roots
        if path.name != "generated_videos"
    )
    assert not any(i.path and i.path.name == "unrelated" for i in entries)


@pytest.mark.parametrize(
    "owner,relative",
    [
        ("generation.assets", "generated_videos"),
        ("cache.model_catalog", "model_catalog_cache.json"),
        ("diagnostics.logs", "custom.log"),
    ],
)
def test_optional_exclusions_preserve_unavailable_unobserved_parent(
    tmp_path, owner, relative
):
    from tldw_chatbook.Backup_Recovery.config_adapter import recovery_adapters

    data = tmp_path / "absent-parent" / "Ada"
    selector = tmp_path / "profile.toml"
    selector.write_text("")
    config = {
        "paths": {"data_dir": str(data.parent)},
        "general": {"users_name": "Ada"},
        "logging": {"log_filename": "custom.log"},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p"),
    }
    adapter = next(a for a in recovery_adapters() if a.owner_id == owner)
    item = next(i for i in adapter.discover(config) if i.path == data / relative)
    assert item.status == "unavailable"
    assert not data.exists()


def test_checked_root_observation_preserves_empty_selection_contract(tmp_path):
    from tldw_chatbook.Backup_Recovery.file_inventory import (
        _inventory_root,
        _inventory_tree,
    )

    root = tmp_path / "root"
    root.mkdir()
    (root / "untouched").mkdir()
    item = _inventory_root(root, owner="test.files", external=False)
    assert item.status == "included_directory"
    assert item.metadata.kind == "directory"
    assert item.metadata.relative_path == ""
    assert (
        _inventory_tree(
            root, owner="test.files", external=False, selected_paths=frozenset()
        )
        == ()
    )
    with pytest.raises(ValueError, match="invalid_root_inspection"):
        _inventory_tree(
            root,
            owner="test.files",
            external=False,
            root_only=True,
            selected_paths=frozenset(),
        )
