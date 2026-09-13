"""Restore dependencies use logical topology, never original absolute paths."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_file_inventory import (
    installed_model as installed_model,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414
)
from Tests.Backup_Recovery.test_recovered_media import media as media  # noqa: PLC0414
from Tests.Backup_Recovery.test_recovered_media import retain
from tldw_chatbook.Backup_Recovery.models import (
    DISCOVERY_CONTEXT_KEY,
    DiscoveryContext,
)


def staged_entries(entries, stage):
    stage.mkdir(mode=0o700)
    candidates, topology = {}, {}
    for index, item in enumerate(entries):
        if item.metadata is None:
            continue
        meta = item.metadata
        topology[item.logical_id] = (
            meta.root_id,
            meta.parent_id,
            meta.relative_path,
            meta.kind,
        )
        if item.status == "included":
            candidate = stage / str(index)
            shutil.copyfile(item.path, candidate)
            candidate.chmod(0o600)
            candidates[item.logical_id] = candidate
        elif item.status == "included_directory":
            candidates[item.logical_id] = stage
    return candidates, topology


@pytest.mark.parametrize(
    "damage", [None, "missing", "corrupt", "wrong_root", "undeclared"]
)
def test_model_restore_uses_archived_topology(installed_model, tmp_path, damage):
    _store, _desc, config, adapter = installed_model
    entries = adapter.discover(config)
    candidates, topology = staged_entries(entries, tmp_path / "stage")
    item = next(row for row in entries if row.path and row.path.name == "manifest.json")
    payload = next(row for row in entries if row.path and row.path.name == "model.onnx")
    item = replace(item, path=Path("/different-machine/manifest.json"))
    if damage == "missing":
        candidates.pop(payload.logical_id)
    elif damage == "corrupt":
        candidates[payload.logical_id].write_bytes(b"changed")
    elif damage == "wrong_root":
        old = topology[payload.logical_id]
        topology[payload.logical_id] = ("foreign", *old[1:])
    elif damage == "undeclared":
        item = replace(
            item,
            dependencies=tuple(
                key for key in item.dependencies if key != payload.logical_id
            ),
        )
    method = getattr(
        adapter, "validate_restore_dependencies", adapter.validate_dependencies
    )
    if hasattr(adapter, "validate_restore_dependencies"):
        issues = method(
            item, candidates[item.logical_id], candidates, topology=topology
        )
    else:
        issues = method(item, candidates[item.logical_id], candidates)
    assert bool(issues) == (damage is not None), issues


@pytest.mark.parametrize(
    "damage", [None, "missing", "corrupt", "wrong_root", "undeclared"]
)
def test_recovered_restore_catalog_and_raw_roles(media, tmp_path, damage):
    from tldw_chatbook.Backup_Recovery.recovered_media import _RecoveredAdapter

    store, source = media
    asset = retain(store, source)
    adapter = _RecoveredAdapter(store.root)
    selector = tmp_path / "config.toml"
    selector.write_text("")
    entries = adapter.discover({DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p")})
    candidates, topology = staged_entries(entries, tmp_path / "stage")
    item = next(
        row for row in entries if row.path and row.path.name == "catalog.sqlite3"
    )
    payload = next(
        row for row in entries if row.path and row.path.name == asset + ".payload"
    )
    item = replace(item, path=Path("/other-machine/catalog.sqlite3"))
    if damage == "missing":
        candidates.pop(payload.logical_id)
    elif damage == "corrupt":
        candidates[payload.logical_id].write_bytes(b"changed")
    elif damage == "wrong_root":
        old = topology[payload.logical_id]
        topology[payload.logical_id] = ("foreign", *old[1:])
    elif damage == "undeclared":
        item = replace(
            item,
            dependencies=tuple(
                key for key in item.dependencies if key != payload.logical_id
            ),
        )
    method = getattr(
        adapter, "validate_restore_dependencies", adapter.validate_dependencies
    )
    if hasattr(adapter, "validate_restore_dependencies"):
        issues = method(
            item, candidates[item.logical_id], candidates, topology=topology
        )
    else:
        issues = method(item, candidates[item.logical_id], candidates)
    assert bool(issues) == (damage is not None), issues
    if damage is None:
        assert adapter.restore_role(payload) == "file"
        assert adapter.validate_restore(payload, candidates[payload.logical_id]) == ()
        adapter.relocate_restore(payload, candidates[payload.logical_id], {})


@pytest.mark.parametrize("damage", [None, "missing", "corrupt", "wrong_root"])
def test_persona_directory_restore_validates_real_core_assets(tmp_path, damage):
    from Tests.Persona_Visual.test_persona_visual_publication import _snapshot
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
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
        asset_path = (
            data
            / db.get_connection()
            .execute("SELECT storage_relpath FROM persona_visual_assets LIMIT 1")
            .fetchone()[0]
        )
    finally:
        db.close()
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
    candidates, topology = staged_entries(entries, tmp_path / "stage")
    item = next(row for row in entries if row.path == data / "persona_visual")
    payload = next(row for row in entries if row.path == asset_path)
    candidates["profile:p:db.chachanotes.primary"] = database
    item = replace(item, path=Path("/different-machine/persona_visual"))
    if damage == "missing":
        candidates.pop(payload.logical_id)
    elif damage == "corrupt":
        candidates[payload.logical_id].write_bytes(b"changed")
    elif damage == "wrong_root":
        old = topology[payload.logical_id]
        topology[payload.logical_id] = ("foreign", *old[1:])
    method = getattr(
        adapter, "validate_restore_dependencies", adapter.validate_dependencies
    )
    if hasattr(adapter, "validate_restore_dependencies"):
        issues = method(
            item, candidates[item.logical_id], candidates, topology=topology
        )
    else:
        issues = method(item, candidates[item.logical_id], candidates)
    assert bool(issues) == (damage is not None), issues


def test_model_recipe_with_omitted_payload_stays_inert(installed_model, tmp_path):
    _store, _desc, config, adapter = installed_model
    entries = adapter.discover(config)
    candidates, topology = staged_entries(entries, tmp_path / "stage")
    item = next(row for row in entries if row.path and row.path.name == "manifest.json")
    payload = next(row for row in entries if row.path and row.path.name == "model.onnx")
    candidates.pop(payload.logical_id)
    item = replace(
        item,
        path=Path("/other-machine/manifest.json"),
        dependencies=tuple(
            key for key in item.dependencies if key != payload.logical_id
        ),
    )
    assert (
        adapter.validate_restore_dependencies(
            item, candidates[item.logical_id], candidates, topology=topology
        )
        == ()
    )


def test_recovered_role_rejects_unknown_archive_layout(media, tmp_path):
    from tldw_chatbook.Backup_Recovery.recovered_media import _RecoveredAdapter

    store, source = media
    retain(store, source)
    adapter = _RecoveredAdapter(store.root)
    selector = tmp_path / "config.toml"
    selector.write_text("")
    entries = adapter.discover({DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "p")})
    item = next(
        row for row in entries if row.path and row.path.name == "catalog.sqlite3"
    )
    malformed = replace(
        item, metadata=replace(item.metadata, relative_path="nested/catalog.sqlite3")
    )
    with pytest.raises(ValueError, match="invalid_recovered_role"):
        adapter.restore_role(malformed)
