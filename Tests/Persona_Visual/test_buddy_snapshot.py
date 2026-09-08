"""Validated immutable Buddy sources and native expression extraction."""

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from Tests.Persona_Visual.test_persona_visual_importer import (
    _archive_payloads,
    _canonical,
    _replace_declared_payload,
    _write_archive,
)
from tldw_chatbook.Character_Chat.expression_set_io import resolve_local_expression_set
from tldw_chatbook.Persona_Visual.importer import PersonaVisualImportError


def test_archive_snapshot_is_validated_and_pins_exact_source(tmp_path):
    from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive

    path = _write_archive(tmp_path / "buddy.zip")
    snapshot = read_buddy_archive(path)
    assert snapshot.title == "Imported operator states"
    assert snapshot.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(snapshot.manifest_json)["states"]["idle"]
    assert (
        snapshot.assets[0].data
        == _archive_payloads()["assets/persona_visuals/idle.png"]
    )
    assert snapshot.artwork == {
        "version": 1,
        "creator": None,
        "license": None,
        "source_url": None,
        "notices": "",
    }
    assert snapshot.is_current()
    assert str(tmp_path) not in repr(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.title = "changed"
    path.write_bytes(path.read_bytes())
    assert not snapshot.is_current()


@pytest.mark.parametrize("damage", ["checksum", "undeclared", "traversal", "manifest"])
def test_invalid_native_archive_is_rejected_by_snapshot_and_expression_route(
    tmp_path, damage
):
    from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive

    payloads = _archive_payloads()
    if damage == "checksum":
        payloads["assets/persona_visuals/idle.png"] += b"changed"
    elif damage == "undeclared":
        payloads["assets/extra.png"] = payloads["assets/persona_visuals/idle.png"]
    elif damage == "traversal":
        payloads["../idle.png"] = payloads["assets/persona_visuals/idle.png"]
    else:
        pack = json.loads(payloads["metadata/pack.json"])
        pack["pack"]["visual_manifest"]["animations"]["idle-loop"]["frames"][0][
            "asset_id"
        ] = "absent"
        _replace_declared_payload(payloads, "metadata/pack.json", _canonical(pack))
    path = _write_archive(tmp_path / "bad.zip", payloads)
    with pytest.raises(PersonaVisualImportError):
        read_buddy_archive(path)
    result = resolve_local_expression_set([path])
    assert not result.images
    assert result.skipped


def test_native_expression_route_reads_pack_manifest_before_generic_limits(
    tmp_path, monkeypatch
):
    import tldw_chatbook.Character_Chat.expression_set_io as expression_io

    path = _write_archive(tmp_path / "native.zip")
    monkeypatch.setattr(expression_io, "MAX_MEMBER_BYTES", 1)
    monkeypatch.setattr(expression_io, "MAX_ZIP_MEMBERS", 1)
    result = resolve_local_expression_set([path])
    assert set(result.images) == {"idle", "thinking", "speaking", "error"}
    assert not result.skipped


def test_archive_projects_only_public_known_artwork(tmp_path):
    from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive

    payloads = _archive_payloads()
    pack = json.loads(payloads["metadata/pack.json"])
    pack["pack"].update(
        creator="Artist",
        license="Original terms",
        notices="Notice\n" * 80,
        source_url="https://example.org/art",
        source_id="/private/identity",
    )
    _replace_declared_payload(payloads, "metadata/pack.json", _canonical(pack))
    snapshot = read_buddy_archive(_write_archive(tmp_path / "credited.zip", payloads))
    assert snapshot.artwork["creator"] == "Artist"
    assert snapshot.artwork["license"] == "Original terms"
    assert snapshot.artwork["notices"] == "Notice\n" * 80
    assert "source_id" not in snapshot.artwork


@pytest.fixture
def saved_source(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
    from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive

    source = read_buddy_archive(_write_archive(tmp_path / "native.zip"))
    asset = source.assets[0]
    root = tmp_path / "profile"
    root.mkdir()
    (root / "idle.png").write_bytes(asset.data)
    db = CharactersRAGDB(tmp_path / "saved.db", client_id="buddy-snapshot")
    repository = PersonaVisualRepository(db)
    repository.activate_new_pack(
        persona_id="buddy",
        title="Saved Buddy",
        description="",
        source_kind="manual",
        source_context={"source_id": "private", "license": "Artist terms"},
        manifest=json.loads(source.manifest_json),
        manifest_storage_relpath="manifest.json",
        assets=[
            {
                "asset_key": asset.metadata.asset_key,
                "role": "frame",
                "storage_relpath": "idle.png",
                "mime_type": "image/png",
                "bytes": len(asset.data),
                "sha256": asset.metadata.sha256,
                "width": 4,
                "height": 5,
                "frame_count": 1,
                "duration_ms": None,
            }
        ],
        expected_persona_revision=1,
        authority_guard=lambda: True,
    )
    yield repository, root
    db.close_connection()


@pytest.mark.parametrize(
    "change", ["bytes", "title", "binding", "source_context", "storage"]
)
def test_saved_guard_catches_source_mutations(saved_source, change):
    from tldw_chatbook.Persona_Visual.snapshot import read_saved_buddy

    repository, root = saved_source
    snapshot = read_saved_buddy(repository, "buddy", root)
    assert snapshot.is_current()
    assert snapshot.artwork["license"] == "Artist terms"
    assert snapshot.artwork["creator"] is None
    assert "private" not in json.dumps(dict(snapshot.artwork))
    if change == "bytes":
        (root / "idle.png").write_bytes(b"broken")
    else:
        sql = {
            "title": "UPDATE persona_visual_packs SET title = 'Changed'",
            "binding": "UPDATE persona_visual_bindings SET version = version + 1",
            "source_context": "UPDATE persona_visual_packs SET source_context_json = '{}'",
            "storage": "UPDATE persona_visual_assets SET storage_relpath = 'missing.png'",
        }[change]
        with repository.db.transaction() as cursor:
            cursor.execute(sql)
    assert not snapshot.is_current()


def test_saved_guard_fails_when_binding_is_deleted(saved_source):
    from tldw_chatbook.Persona_Visual.snapshot import read_saved_buddy

    repository, root = saved_source
    snapshot = read_saved_buddy(repository, "buddy", root)
    with repository.db.transaction() as cursor:
        cursor.execute("DELETE FROM persona_visual_bindings")
    assert not snapshot.is_current()
    with pytest.raises(PersonaVisualImportError):
        read_saved_buddy(repository, "buddy", root)


def test_finished_collection_archives_read_through_real_native_route():
    import os

    from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive

    configured_root = os.environ.get("TLDW_BUDDY_COLLECTION")
    if not configured_root:
        pytest.skip("Set TLDW_BUDDY_COLLECTION for the optional collection probe")
    root = Path(configured_root)
    paths = sorted(
        path
        for path in root.glob("*/*.tldw-persona-vpack")
        if "scaffold" not in str(path)
    )
    assert paths, "The configured collection must contain finished Buddy archives"
    import zipfile

    for path in paths:
        snapshot = read_buddy_archive(path)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("metadata/pack.json"))["pack"][
                "visual_manifest"
            ]
            assert json.loads(snapshot.manifest_json) == manifest
            assert len(snapshot.assets) == len(
                json.loads(archive.read("metadata/assets.json"))["assets"]
            )
        result = resolve_local_expression_set([path])
        assert set(result.images) == {"idle", "thinking", "speaking", "error"}, (
            path.name,
            result,
        )
        assert not result.skipped


def test_native_missing_required_member_never_falls_back_to_generic(tmp_path):
    payloads = _archive_payloads()
    del payloads["metadata/pack.json"]
    payloads["idle.png"] = payloads["assets/persona_visuals/idle.png"]
    result = resolve_local_expression_set(
        [_write_archive(tmp_path / "broken.zip", payloads)]
    )
    assert not result.images
    assert result.skipped


def test_snapshot_artwork_is_detached_and_immutable(tmp_path):
    from dataclasses import replace

    from tldw_chatbook.Persona_Visual.snapshot import read_buddy_archive

    source = read_buddy_archive(_write_archive(tmp_path / "immutable.zip"))
    artwork = dict(source.artwork)
    snapshot = replace(source, artwork=artwork)
    artwork["creator"] = "Changed outside review"
    assert snapshot.artwork["creator"] is None
    with pytest.raises(TypeError):
        snapshot.artwork["license"] = "Invented terms"
