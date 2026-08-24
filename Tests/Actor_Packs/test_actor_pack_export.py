"""Consistent local actor snapshots for Actor Pack export."""

from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from tldw_chatbook.Actor_Packs.export import (
    ActorPackExportError,
    ActorPackExportService,
    ActorPackExportSnapshot,
    write_actor_pack_archive,
)
from tldw_chatbook.Actor_Packs.contracts import (
    ZIP_COMPRESSION,
    ZIP_CREATE_SYSTEM,
    ZIP_EXTERNAL_ATTR,
    ZIP_TIMESTAMP,
)
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.Character_Chat.visual_identity import (
    SAMIRA_MANIFEST_SCHEMA_ID,
    compute_pack_content_sha256,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

from .conftest import PNG_1X1, canonical_json


PORTABLE_UUID = "123e4567-e89b-42d3-a456-426614174000"


def _png(color: str = "red") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _persona_visual_manifest() -> dict[str, object]:
    states = {
        state: {"animation_id": "idle"}
        for state in ("idle", "listening", "thinking", "speaking", "error")
    }
    return {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": states,
        "animations": {
            "idle": {
                "frames": [{"asset_id": "idle"}],
                "preview_asset_id": "idle",
            }
        },
        "state_catalog": {},
        "fallbacks": {},
        "authored_triggers": [],
    }


def _shared_visual_manifest(data: bytes, storage_key: str) -> dict[str, object]:
    asset: dict[str, object] = {
        "expression_key": "neutral",
        "original_label": "neutral",
        "display_label": "Neutral",
        "storage_relpath": storage_key,
        "content_type": "image/png",
        "bytes": len(data),
        "width": 8,
        "height": 8,
        "sha256": hashlib.sha256(data).hexdigest(),
        "is_animated": False,
        "frame_count": 1,
        "duration_ms": None,
    }
    manifest: dict[str, object] = {
        "schema_id": SAMIRA_MANIFEST_SCHEMA_ID,
        "pack_id": "user.export.pack",
        "title": "Export reactions",
        "license": "MIT",
        "default_expression_key": "neutral",
        "assets": [asset],
    }
    manifest["pack_content_sha256"] = compute_pack_content_sha256(manifest)
    return manifest


def _activate_shared_visual(
    database: CharactersRAGDB,
    *,
    actor_kind: str,
    actor_id: int | str,
    data: bytes,
    storage_key: str,
) -> dict[str, object]:
    manifest = _shared_visual_manifest(data, storage_key)
    asset = dict(manifest["assets"][0])  # type: ignore[index]
    asset["original_expression_key"] = asset.pop("original_label")
    asset["source_filename"] = "neutral.png"
    return VisualIdentityRepository(database).activate_pack(
        pack={
            "title": "Export reactions",
            "default_expression_key": "neutral",
            "source_kind": "manual",
            "source_context": {"provenance": "local-authoring"},
        },
        manifest=manifest,
        assets=[asset],
        actor_kind=actor_kind,
        actor_id=actor_id,
    )


@pytest.fixture
def export_components(tmp_path: Path):
    database = CharactersRAGDB(tmp_path / "actors.db", client_id="actor-pack-export")
    repository = ActorPackRepository(
        database, uuid_factory=lambda: uuid.UUID(PORTABLE_UUID)
    )
    local_service = LocalCharacterPersonaService(
        database, persona_store_path=tmp_path / "personas.json"
    )
    export = ActorPackExportService(database, local_service, repository)
    yield export, repository, local_service, database
    database.close_connection()


def test_unregistered_local_character_gets_durable_identity_and_snapshot(
    export_components,
) -> None:
    export, repository, local_service, database = export_components
    character_id = database.add_character_card(
        {"name": "Portable Guide", "description": "Local", "image": _png()}
    )

    snapshot = export.capture_snapshot("character", str(character_id), source="local")

    assert type(snapshot) is ActorPackExportSnapshot
    assert snapshot.actor_kind == "character"
    assert snapshot.portable_uuid == PORTABLE_UUID
    assert snapshot.actor_revision >= 1
    assert snapshot.portrait_sha256
    assert repository.get_identity("character", character_id) is not None
    assert local_service.get_character(character_id)["name"] == "Portable Guide"
    assert "Portable Guide" not in repr(snapshot)
    assert "local_actor_id=" not in repr(snapshot)
    assert "image" not in snapshot.actor_payload.decode("utf-8")


def test_eligibility_validates_without_assigning_portable_identity(
    export_components,
) -> None:
    export, repository, _local_service, database = export_components
    character_id = database.add_character_card(
        {"name": "Eligible", "description": "Local", "image": _png()}
    )

    eligibility = export.capture_eligibility(
        "character", str(character_id), source="local"
    )

    assert eligibility.actor_kind == "character"
    assert eligibility.local_actor_id == str(character_id)
    assert eligibility.actor_revision >= 1
    assert repository.get_identity("character", character_id) is None


def test_invalid_eligibility_does_not_assign_portable_identity(
    export_components,
) -> None:
    export, repository, _local_service, database = export_components
    character_id = database.add_character_card(
        {"name": "Broken", "image": b"not-an-image"}
    )

    with pytest.raises(ActorPackExportError, match="actor_pack_portrait_invalid"):
        export.capture_eligibility("character", str(character_id), source="local")

    assert repository.get_identity("character", character_id) is None


def test_server_source_is_rejected_before_identity_assignment(
    export_components,
) -> None:
    export, repository, _local_service, database = export_components
    character_id = database.add_character_card({"name": "Server", "image": _png()})

    with pytest.raises(ActorPackExportError, match="actor_pack_source_not_local"):
        export.capture_snapshot("character", str(character_id), source="server")

    assert repository.get_identity("character", character_id) is None


def test_invalid_portrait_is_rejected_before_identity_assignment(
    export_components,
) -> None:
    export, repository, _local_service, database = export_components
    character_id = database.add_character_card(
        {"name": "Broken", "image": b"not-an-image"}
    )

    with pytest.raises(ActorPackExportError, match="actor_pack_portrait_invalid"):
        export.capture_snapshot("character", str(character_id), source="local")

    assert repository.get_identity("character", character_id) is None


def test_missing_portrait_has_stable_portrait_category_before_assignment(
    export_components,
) -> None:
    export, repository, _local_service, database = export_components
    character_id = database.add_character_card({"name": "No Portrait"})

    with pytest.raises(ActorPackExportError, match="actor_pack_portrait_invalid"):
        export.capture_snapshot("character", str(character_id), source="local")

    assert repository.get_identity("character", character_id) is None


def test_inactive_local_persona_is_eligible_and_uses_linked_portrait(
    export_components,
) -> None:
    export, repository, local_service, database = export_components
    character_id = database.add_character_card(
        {"name": "Portrait", "image": _png("blue")}
    )
    persona = local_service.create_persona_profile(
        {
            "id": "persona-inactive",
            "name": "Inactive Persona",
            "character_card_id": character_id,
            "is_active": False,
        }
    )

    snapshot = export.capture_snapshot("persona", persona["id"], source="local")

    assert snapshot.actor_kind == "persona"
    assert snapshot.actor_revision == 1
    assert snapshot.portable_uuid == PORTABLE_UUID
    assert snapshot.portrait_sha256 == hashlib.sha256(_png("blue")).hexdigest()
    assert repository.get_identity("persona", persona["id"]) is not None


def test_post_assignment_actor_change_refuses_mixed_snapshot_but_keeps_uuid(
    export_components,
) -> None:
    export, repository, _local_service, database = export_components
    character_id = database.add_character_card({"name": "Before", "image": _png()})

    def mutate_after_assignment(phase: str) -> None:
        if phase == "identity_assigned":
            current = database.get_character_card_by_id(character_id)
            assert current is not None
            database.update_character_card(
                character_id, {"name": "After"}, int(current["version"])
            )

    with pytest.raises(
        ActorPackExportError, match="actor_pack_export_authority_changed"
    ):
        export.capture_snapshot(
            "character",
            str(character_id),
            source="local",
            phase_hook=mutate_after_assignment,
        )

    assert repository.get_identity("character", character_id) is not None


def test_existing_identity_is_reused_and_missing_actor_is_path_free(
    export_components,
) -> None:
    export, repository, _local_service, database = export_components
    character_id = database.add_character_card({"name": "Existing", "image": _png()})
    original = repository.assign_identity("character", character_id)

    snapshot = export.capture_snapshot("character", str(character_id), source="local")

    assert snapshot.portable_uuid == original.portable_uuid
    with pytest.raises(ActorPackExportError) as caught:
        export.capture_snapshot("character", "999999", source="local")
    assert caught.value.category == "actor_pack_actor_unavailable"
    assert "999999" not in str(caught.value)


def test_deleted_persona_is_not_exportable(export_components) -> None:
    export, repository, local_service, database = export_components
    character_id = database.add_character_card({"name": "Portrait", "image": _png()})
    persona = local_service.create_persona_profile(
        {
            "id": "persona-deleted",
            "name": "Deleted Persona",
            "character_card_id": character_id,
        }
    )
    local_service.delete_persona_profile(
        persona["id"], expected_version=int(persona["version"])
    )

    with pytest.raises(ActorPackExportError, match="actor_pack_actor_unavailable"):
        export.capture_snapshot("persona", persona["id"], source="local")

    assert repository.get_identity("persona", persona["id"]) is None


def test_persona_visual_section_is_self_contained(
    export_components, tmp_path: Path
) -> None:
    _export, repository, local_service, database = export_components
    portrait_id = database.add_character_card(
        {"name": "Portrait", "image": _png("blue")}
    )
    persona = local_service.create_persona_profile(
        {
            "id": "persona-visual-export",
            "name": "Visual Persona",
            "character_card_id": portrait_id,
        }
    )
    visual_bytes = _png("green")
    storage_key = "persona_visual/export/idle.png"
    asset_path = tmp_path / storage_key
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(visual_bytes)
    visual_repository = PersonaVisualRepository(database)
    active = visual_repository.activate_new_pack(
        persona_id=persona["id"],
        title="Runtime states",
        source_context={"license": "CC0-1.0", "provenance": "local"},
        manifest=_persona_visual_manifest(),
        manifest_storage_relpath="persona_visual/export/manifest.json",
        assets=[
            {
                "asset_key": "idle",
                "role": "frame",
                "storage_relpath": storage_key,
                "mime_type": "image/png",
                "bytes": len(visual_bytes),
                "sha256": hashlib.sha256(visual_bytes).hexdigest(),
                "width": 8,
                "height": 8,
                "frame_count": 1,
                "duration_ms": None,
            }
        ],
        expected_persona_revision=int(persona["version"]),
        authority_guard=lambda: True,
    )
    export = ActorPackExportService(
        database,
        local_service,
        repository,
        persona_visual_repository=visual_repository,
        visual_identity_repository=VisualIdentityRepository(database),
        profile_root=tmp_path,
    )

    snapshot = export.capture_snapshot("persona", persona["id"], source="local")

    assert len(snapshot.sections) == 1
    section = snapshot.sections[0]
    assert section.kind == "persona-runtime"
    assert section.graph_identity == active.identity
    assert section.license == "CC0-1.0"
    assert section.provenance == "local"
    assert section.manifest_path == "persona-runtime/manifest.json"
    assert section.assets[0].path == "persona-runtime/assets/asset-0001.png"
    assert section.assets[0].data == visual_bytes
    assert storage_key not in repr(snapshot)

    shared_bytes = _png("yellow")
    shared_storage = "persona/shared/neutral.png"
    shared_path = tmp_path / "visual_identities" / shared_storage
    shared_path.parent.mkdir(parents=True)
    shared_path.write_bytes(shared_bytes)
    _activate_shared_visual(
        database,
        actor_kind="persona",
        actor_id=persona["id"],
        data=shared_bytes,
        storage_key=shared_storage,
    )
    both = export.capture_snapshot("persona", persona["id"], source="local")
    assert tuple(item.kind for item in both.sections) == (
        "shared-visual-identity",
        "persona-runtime",
    )
    archive_bytes = io.BytesIO()
    write_actor_pack_archive(both, archive_bytes)
    assert storage_key.encode() not in archive_bytes.getvalue()
    assert shared_storage.encode() not in archive_bytes.getvalue()
    with zipfile.ZipFile(io.BytesIO(archive_bytes.getvalue())) as archive:
        assert archive.namelist() == [
            "actor-pack.json",
            "actor/actor.json",
            "actor/portrait.png",
            "persona-runtime/assets/asset-0001.png",
            "persona-runtime/manifest.json",
            "shared-visual-identity/assets/asset-0001.png",
            "shared-visual-identity/manifest.json",
        ]
        root = json.loads(archive.read("actor-pack.json"))
        assert [item["kind"] for item in root["sections"]] == [
            "shared-visual-identity",
            "persona-runtime",
        ]

    def change_binding(phase: str) -> None:
        if phase == "visuals_loaded":
            database.get_connection().execute(
                "UPDATE persona_visual_bindings SET version = version + 1 WHERE id = ?",
                (active.binding.id,),
            )
            database.get_connection().commit()

    with pytest.raises(
        ActorPackExportError, match="actor_pack_export_authority_changed"
    ):
        export.capture_snapshot(
            "persona",
            persona["id"],
            source="local",
            phase_hook=change_binding,
        )

    asset_path.unlink()
    with pytest.raises(
        ActorPackExportError, match="actor_pack_export_asset_unavailable"
    ):
        export.capture_snapshot("persona", persona["id"], source="local")


def test_character_shared_visual_section_remaps_private_storage(
    export_components, tmp_path: Path
) -> None:
    _export, repository, local_service, database = export_components
    character_id = database.add_character_card(
        {"name": "Shared Visual", "image": _png("blue")}
    )
    visual_bytes = _png("yellow")
    storage_key = "visual_identity/private/neutral.png"
    asset_path = tmp_path / "visual_identities" / storage_key
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(visual_bytes)
    active = _activate_shared_visual(
        database,
        actor_kind="character",
        actor_id=character_id,
        data=visual_bytes,
        storage_key=storage_key,
    )
    export = ActorPackExportService(
        database,
        local_service,
        repository,
        visual_identity_repository=VisualIdentityRepository(database),
        profile_root=tmp_path,
    )

    snapshot = export.capture_snapshot("character", str(character_id), source="local")

    assert len(snapshot.sections) == 1
    section = snapshot.sections[0]
    assert section.kind == "shared-visual-identity"
    assert section.graph_identity[4] == active["version"]["id"]
    assert section.license == "MIT"
    assert section.provenance == "local-authoring"
    assert section.assets[0].path == "shared-visual-identity/assets/asset-0001.png"
    exported_manifest = json.loads(section.manifest_bytes)
    assert exported_manifest["assets"][0]["storage_relpath"] == section.assets[0].path
    assert storage_key not in section.manifest_bytes.decode()
    assert storage_key not in repr(snapshot)

    replacement = _png("purple")

    def replace_after_materialization(phase: str) -> None:
        if phase == "visuals_loaded":
            asset_path.write_bytes(replacement)

    with pytest.raises(
        ActorPackExportError, match="actor_pack_export_asset_unavailable"
    ):
        export.capture_snapshot(
            "character",
            str(character_id),
            source="local",
            phase_hook=replace_after_materialization,
        )

    asset_path.write_bytes(visual_bytes)

    def substitute_same_bytes(phase: str) -> None:
        if phase == "visuals_loaded":
            replacement_path = asset_path.with_name("replacement.png")
            replacement_path.write_bytes(visual_bytes)
            os.replace(replacement_path, asset_path)

    with pytest.raises(
        ActorPackExportError, match="actor_pack_export_authority_changed"
    ):
        export.capture_snapshot(
            "character",
            str(character_id),
            source="local",
            phase_hook=substitute_same_bytes,
        )

    database.get_connection().execute(
        "UPDATE visual_identity_pack_versions SET manifest_json = '{}' WHERE id = ?",
        (active["version"]["id"],),
    )
    database.get_connection().commit()
    with pytest.raises(ActorPackExportError, match="actor_pack_export_visual_invalid"):
        export.capture_snapshot("character", str(character_id), source="local")


def test_archive_bytes_are_deterministic_and_independently_readable(
    export_components,
) -> None:
    export, _repository, _local_service, database = export_components
    character_id = database.add_character_card(
        {"name": "Deterministic", "image": _png()}
    )
    snapshot = export.capture_snapshot("character", str(character_id), source="local")
    first = io.BytesIO()
    second = io.BytesIO()

    first_digest = write_actor_pack_archive(snapshot, first)
    second_digest = write_actor_pack_archive(snapshot, second)

    assert first.getvalue() == second.getvalue()
    assert first_digest == hashlib.sha256(first.getvalue()).hexdigest()
    assert second_digest == first_digest
    with zipfile.ZipFile(io.BytesIO(first.getvalue())) as archive:
        assert archive.namelist() == [
            "actor-pack.json",
            "actor/actor.json",
            "actor/portrait.png",
        ]
        for info in archive.infolist():
            assert info.compress_type == ZIP_COMPRESSION
            assert info.date_time == ZIP_TIMESTAMP
            assert info.create_system == ZIP_CREATE_SYSTEM
            assert info.external_attr == ZIP_EXTERNAL_ATTR
        root = json.loads(archive.read("actor-pack.json"))
        assert "actor-pack.json" not in {item["path"] for item in root["files"]}
        declared = {
            item["path"]: (item["bytes"], item["sha256"]) for item in root["files"]
        }
        for name in ("actor/actor.json", "actor/portrait.png"):
            data = archive.read(name)
            assert declared[name] == (len(data), hashlib.sha256(data).hexdigest())
        root_without_digest = dict(root)
        digest = root_without_digest.pop("content_digest")
        canonical_root = json.dumps(
            root_without_digest,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
        assert digest == hashlib.sha256(canonical_root).hexdigest()


@pytest.mark.parametrize("actor_kind", ("character", "persona"))
def test_archive_matches_independent_golden_bytes(
    actor_kind: str,
) -> None:
    payload = canonical_json(
        {
            "schema": "tldw.actor/v1",
            "actor_kind": actor_kind,
            "portable_uuid": PORTABLE_UUID,
            "data": {"name": "Golden"},
        }
    )
    snapshot = ActorPackExportSnapshot(
        actor_kind=actor_kind,
        actor_revision=1,
        portable_uuid=PORTABLE_UUID,
        identity_version=1,
        portrait_name="portrait.png",
        portrait_sha256=hashlib.sha256(PNG_1X1).hexdigest(),
        local_actor_id="private-local-id",
        actor_payload=payload,
        portrait_bytes=PNG_1X1,
    )
    output = io.BytesIO()

    write_actor_pack_archive(snapshot, output)

    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "export-golden"
        / f"minimal-{actor_kind}.tldw-actor-pack"
    )
    golden = fixture.read_bytes()
    assert output.getvalue() == golden
    assert b"private-local-id" not in golden
    with zipfile.ZipFile(io.BytesIO(golden)) as archive:
        root = json.loads(archive.read("actor-pack.json"))
        assert root["actor"]["kind"] == actor_kind
        assert set(archive.namelist()) == {
            "actor-pack.json",
            "actor/actor.json",
            "actor/portrait.png",
        }
        assert {item["path"] for item in root["files"]} == {
            "actor/actor.json",
            "actor/portrait.png",
        }


def test_archive_validation_failure_keeps_assigned_uuid_and_writes_nothing(
    export_components,
) -> None:
    export, repository, _local_service, database = export_components
    character_id = database.add_character_card(
        {"name": "Invalid archive", "image": _png()}
    )
    snapshot = export.capture_snapshot("character", str(character_id), source="local")
    invalid = replace(snapshot, actor_payload=b"not canonical actor json")
    output = io.BytesIO()

    with pytest.raises(ActorPackExportError, match="actor_pack_export_archive_failed"):
        write_actor_pack_archive(invalid, output)

    assert output.getvalue() == b""
    assert repository.get_identity("character", character_id) is not None
