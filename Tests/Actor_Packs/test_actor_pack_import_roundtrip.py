"""Independent stdlib Actor Pack section fixtures exercised through activation."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.activation import ActorPackActivationService
from tldw_chatbook.Actor_Packs.importer import ActorPackImportService
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository

from .conftest import (
    PNG_1X1,
    PORTABLE_UUID,
    canonical_json,
    file_descriptor,
    with_content_digest,
)


def _shared_section() -> tuple[dict[str, object], dict[str, bytes]]:
    asset_path = "shared-visual-identity/assets/asset-0001.png"
    digest = hashlib.sha256(PNG_1X1).hexdigest()
    asset = {
        "expression_key": "neutral",
        "original_label": "neutral",
        "display_label": "Neutral",
        "storage_relpath": asset_path,
        "content_type": "image/png",
        "bytes": len(PNG_1X1),
        "width": 1,
        "height": 1,
        "sha256": digest,
        "is_animated": False,
        "frame_count": 1,
        "duration_ms": None,
    }
    content = {
        "schema_id": "tldw.visual_identity_pack/v1",
        "pack_id": "actor.pack.independent",
        "default_expression_key": "neutral",
        "license": "MIT",
        "assets": [
            {
                "expression_key": "neutral",
                "original_label": "neutral",
                "relative_filename": asset_path,
                "content_type": "image/png",
                "bytes": len(PNG_1X1),
                "width": 1,
                "height": 1,
                "sha256": digest,
            }
        ],
    }
    manifest = {
        "schema_id": "tldw.visual_identity_pack/v1",
        "pack_id": "actor.pack.independent",
        "title": "Independent shared visual",
        "license": "MIT",
        "default_expression_key": "neutral",
        "assets": [asset],
        "pack_content_sha256": hashlib.sha256(canonical_json(content)).hexdigest(),
    }
    return (
        {
            "kind": "shared-visual-identity",
            "manifest": "shared-visual-identity/manifest.json",
        },
        {
            "shared-visual-identity/manifest.json": canonical_json(manifest),
            asset_path: PNG_1X1,
        },
    )


def _runtime_section() -> tuple[dict[str, object], dict[str, bytes]]:
    manifest = {
        "renderer_type": "sprite_frames",
        "manifest_version": 1,
        "states": {
            state: {"animation_id": "idle"}
            for state in ("idle", "listening", "thinking", "speaking", "error")
        },
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
    return (
        {
            "kind": "persona-runtime",
            "manifest": "persona-runtime/manifest.json",
        },
        {
            "persona-runtime/manifest.json": canonical_json(manifest),
            "persona-runtime/assets/asset-0001.png": PNG_1X1,
        },
    )


def _write_independent_archive(
    path: Path, actor_kind: str, section_names: tuple[str, ...]
) -> Path:
    actor = canonical_json(
        {
            "schema": "tldw.actor/v1",
            "actor_kind": actor_kind,
            "portable_uuid": PORTABLE_UUID,
            "data": {"name": "Independent Golden"},
        }
    )
    files = {"actor/actor.json": actor, "actor/portrait.png": PNG_1X1}
    section_records: list[dict[str, object]] = []
    for name in section_names:
        section, section_files = (
            _shared_section() if name == "shared" else _runtime_section()
        )
        section_records.append(section)
        files.update(section_files)
    features = [
        feature
        for name, feature in (
            ("shared", "shared-visual-identity/v1"),
            ("runtime", "persona-runtime/sprite-frames-v1"),
        )
        if name in section_names
    ]
    manifest = with_content_digest(
        {
            "schema": "tldw.actor-pack/v1",
            "actor": {
                "kind": actor_kind,
                "portable_uuid": PORTABLE_UUID,
                "payload": "actor/actor.json",
                "portrait": "actor/portrait.png",
            },
            "sections": section_records,
            "producer": {"name": "independent-test", "version": "1"},
            "license": {"value": "test-only"},
            "provenance": {"source": "stdlib-oracle"},
            "required_features": features,
            "files": [
                file_descriptor(name, data) for name, data in sorted(files.items())
            ],
        }
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("actor-pack.json", canonical_json(manifest))
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    return path.resolve()


@pytest.mark.parametrize(
    ("actor_kind", "sections"),
    [
        ("character", ("shared",)),
        ("persona", ("runtime",)),
        ("persona", ("shared", "runtime")),
    ],
)
def test_independent_section_archives_activate_complete_local_graphs(
    tmp_path: Path, actor_kind: str, sections: tuple[str, ...]
) -> None:
    db = CharactersRAGDB(tmp_path / "roundtrip.db", client_id="import-roundtrip")
    repository = ActorPackRepository(db)
    local_service = LocalCharacterPersonaService(
        db, persona_store_path=tmp_path / "personas.json"
    )
    importer = ActorPackImportService(
        repository,
        staging_root=tmp_path / "staging",
        profile_root=tmp_path,
        local_service=local_service,
    )
    activation = ActorPackActivationService(
        db,
        local_service,
        repository,
        PersonaActorPackCoordinator(repository, local_service),
        importer,
    )
    archive = _write_independent_archive(
        tmp_path / "independent.tldw-actor-pack", actor_kind, sections
    )

    result = activation.activate(importer.inspect_archive(archive), "create_new")

    shared = VisualIdentityRepository(db).get_active_actor_pack(
        actor_kind, result.local_actor_id
    )
    runtime = (
        PersonaVisualRepository(db).get_active_persona_pack(result.local_actor_id)
        if actor_kind == "persona"
        else None
    )
    assert (shared is not None) is ("shared" in sections)
    assert (runtime is not None) is ("runtime" in sections)
    db.close_connection()
