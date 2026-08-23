from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.activation import (
    ActorPackActivationError,
    ActorPackActivationService,
)
from tldw_chatbook.Actor_Packs.contracts import (
    canonical_json_bytes,
    canonicalize_actor_payload,
)
from tldw_chatbook.Actor_Packs.export import (
    ActorPackExportFile,
    ActorPackExportSection,
    ActorPackExportService,
    ActorPackExportSnapshot,
    write_actor_pack_archive,
)
from tldw_chatbook.Actor_Packs.importer import (
    ActorPackImportError,
    ActorPackImportService,
)
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
from tldw_chatbook.Character_Chat.visual_identity import (
    SAMIRA_MANIFEST_SCHEMA_ID,
    compute_pack_content_sha256,
)

from .conftest import PNG_1X1


FIXTURES = Path(__file__).parent / "fixtures" / "export-golden"
PORTABLE_UUID = "123e4567-e89b-42d3-a456-426614174000"


def _shared_visual_section() -> ActorPackExportSection:
    digest = hashlib.sha256(PNG_1X1).hexdigest()
    asset_path = "shared-visual-identity/assets/asset-0001.png"
    manifest = {
        "schema_id": SAMIRA_MANIFEST_SCHEMA_ID,
        "pack_id": "actor.pack.import.test",
        "title": "Imported reactions",
        "license": "MIT",
        "default_expression_key": "neutral",
        "assets": [
            {
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
        ],
    }
    manifest["pack_content_sha256"] = compute_pack_content_sha256(manifest)
    return ActorPackExportSection(
        kind="shared-visual-identity",
        manifest_path="shared-visual-identity/manifest.json",
        graph_identity=(1,),
        license="MIT",
        provenance="test",
        manifest_bytes=canonical_json_bytes(manifest),
        assets=(ActorPackExportFile(asset_path, digest, PNG_1X1),),
    )


def _persona_runtime_section() -> ActorPackExportSection:
    digest = hashlib.sha256(PNG_1X1).hexdigest()
    asset_path = "persona-runtime/assets/asset-0001.png"
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
    return ActorPackExportSection(
        kind="persona-runtime",
        manifest_path="persona-runtime/manifest.json",
        graph_identity=(1,),
        license="CC0-1.0",
        provenance="test",
        manifest_bytes=canonical_json_bytes(manifest),
        assets=(ActorPackExportFile(asset_path, digest, PNG_1X1),),
    )


def _write_archive(
    path: Path,
    *,
    actor_kind: str,
    portable_uuid: str,
    sections: tuple[ActorPackExportSection, ...],
) -> Path:
    digest = hashlib.sha256(PNG_1X1).hexdigest()
    with path.open("wb+") as sink:
        write_actor_pack_archive(
            ActorPackExportSnapshot(
                actor_kind=actor_kind,
                actor_revision=1,
                portable_uuid=portable_uuid,
                identity_version=1,
                portrait_name="portrait.png",
                portrait_sha256=digest,
                local_actor_id="source",
                actor_payload=canonicalize_actor_payload(
                    actor_kind, portable_uuid, {"name": "Visual import"}
                ),
                portrait_bytes=PNG_1X1,
                sections=sections,
            ),
            sink,
        )
    return path.resolve()


@pytest.fixture
def activation_components(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "activation.db", client_id="actor-pack-activate")
    repository = ActorPackRepository(db)
    local_service = LocalCharacterPersonaService(
        db,
        persona_store_path=tmp_path / "personas.json",
    )
    coordinator = PersonaActorPackCoordinator(repository, local_service)
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
        coordinator,
        importer,
    )
    yield activation, importer, repository, local_service, db
    db.close_connection()


def test_create_new_character_preserves_incoming_uuid(activation_components) -> None:
    activation, importer, repository, _local_service, db = activation_components
    review = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )

    result = activation.activate(review, "create_new")

    actor = db.get_character_card_by_id(int(result.local_actor_id))
    assert actor is not None
    assert actor["name"] == "Golden"
    assert actor["image"].startswith(b"\x89PNG")
    identity = repository.get_identity("character", int(result.local_actor_id))
    assert identity is not None
    assert identity.portable_uuid == review.portable_uuid
    assert identity.source_portable_uuid is None
    assert not (importer._staging_root / review._candidate_name).exists()


def test_create_copy_character_gets_fresh_uuid_and_source_provenance(
    activation_components,
) -> None:
    activation, importer, repository, _local_service, _db = activation_components
    first = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )
    original = activation.activate(first, "create_new")
    second = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )

    copied = activation.activate(second, "create_copy")

    assert copied.portable_uuid != original.portable_uuid
    identity = repository.get_identity("character", int(copied.local_actor_id))
    assert identity is not None
    assert identity.source_portable_uuid == original.portable_uuid


def test_create_new_persona_preserves_incoming_uuid(activation_components) -> None:
    activation, importer, repository, local_service, db = activation_components
    review = importer.inspect_archive(
        (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
    )

    result = activation.activate(review, "create_new")

    profile = local_service.get_persona_profile(result.local_actor_id)
    assert profile["name"] == "Golden"
    identity = repository.get_identity("persona", result.local_actor_id)
    assert identity is not None
    assert identity.portable_uuid == review.portable_uuid
    assert type(profile.get("character_card_id")) is int
    portrait_actor = db.get_character_card_by_id(profile["character_card_id"])
    assert portrait_actor is not None
    assert portrait_actor["image"] == PNG_1X1
    snapshot = ActorPackExportService(db, local_service, repository).capture_snapshot(
        "persona", result.local_actor_id, source="local"
    )
    assert snapshot.portrait_bytes == PNG_1X1


def test_cancel_after_section_publication_removes_owned_orphans(
    activation_components, tmp_path: Path
) -> None:
    activation, importer, repository, _local_service, _db = activation_components
    archive = _write_archive(
        tmp_path / "cancel-section.tldw-actor-pack",
        actor_kind="character",
        portable_uuid=PORTABLE_UUID,
        sections=(_shared_visual_section(),),
    )
    review = importer.inspect_archive(archive)
    publication_root = (
        importer._profile_root
        / "visual_identities"
        / "actor_packs"
        / review.content_digest
    )

    with pytest.raises(ActorPackActivationError) as raised:
        activation.activate(
            review,
            "create_new",
            cancel_requested=lambda: (
                publication_root.exists() and any(publication_root.iterdir())
            ),
        )

    assert raised.value.category == "actor_pack_import_cancelled"
    assert not publication_root.exists() or list(publication_root.iterdir()) == []
    assert repository.get_identity_by_portable_uuid(PORTABLE_UUID) is None


def test_partial_immutable_publication_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Actor_Packs import activation as activation_module

    target = tmp_path / "asset.png"
    real_write = activation_module.os.write
    calls = 0

    def fail_after_partial(descriptor: int, data: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, memoryview(data)[:1])
        raise OSError("injected")

    monkeypatch.setattr(activation_module.os, "write", fail_after_partial)

    with pytest.raises(OSError):
        activation_module._publish_immutable(
            target, PNG_1X1, hashlib.sha256(PNG_1X1).hexdigest()
        )

    assert not target.exists()


def test_cancellation_before_commit_leaves_no_actor_or_identity(
    activation_components,
) -> None:
    activation, importer, repository, _local_service, db = activation_components
    review = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(ActorPackActivationError) as raised:
        activation.activate(review, "create_new", cancel_requested=cancelled.is_set)

    assert raised.value.category == "actor_pack_import_cancelled"
    assert all(
        card["name"] != "Golden"
        for card in db.list_character_cards(limit=100, offset=0)
    )
    assert repository.get_identity_by_portable_uuid(review.portable_uuid) is None


def test_staged_inode_substitution_prevents_activation(activation_components) -> None:
    activation, importer, repository, _local_service, _db = activation_components
    review = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )
    candidate = importer._staging_root / review._candidate_name
    portrait = candidate / "actor" / "portrait.png"
    replacement = candidate / "actor" / "replacement.png"
    replacement.write_bytes(portrait.read_bytes())
    replacement.replace(portrait)

    with pytest.raises(ActorPackActivationError) as raised:
        activation.activate(review, "create_new")

    assert raised.value.category == "actor_pack_import_review_stale"
    assert repository.get_identity_by_portable_uuid(review.portable_uuid) is None


def test_action_must_be_offered_by_the_exact_review(activation_components) -> None:
    activation, importer, _repository, _local_service, _db = activation_components
    review = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )

    with pytest.raises(ActorPackActivationError) as raised:
        activation.activate(review, "update_existing")

    assert raised.value.category == "actor_pack_import_action_invalid"


def test_update_character_changes_only_present_portable_fields(
    activation_components,
) -> None:
    activation, importer, repository, _local_service, db = activation_components
    original_review = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )
    original = activation.activate(original_review, "create_new")
    actor_id = int(original.local_actor_id)
    actor = db.get_character_card_by_id(actor_id)
    assert actor is not None
    db.update_character_card(
        actor_id,
        {"description": "Keep this local description"},
        expected_version=actor["version"],
    )
    update_review = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )

    updated = activation.activate(update_review, "update_existing")

    assert updated.local_actor_id == str(actor_id)
    current = db.get_character_card_by_id(actor_id)
    assert current is not None
    assert current["name"] == "Golden"
    assert current["description"] == "Keep this local description"
    assert current["version"] == actor["version"] + 2
    identity = repository.get_identity("character", actor_id)
    assert identity is not None
    assert identity.portable_uuid == original.portable_uuid


def test_update_persona_changes_only_present_portable_fields(
    activation_components,
) -> None:
    activation, importer, repository, local_service, db = activation_components
    original_review = importer.inspect_archive(
        (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
    )
    original = activation.activate(original_review, "create_new")
    current = local_service.get_persona_profile(original.local_actor_id)
    local_service.update_persona_profile(
        original.local_actor_id,
        {"description": "Keep this local description"},
        expected_version=current["version"],
    )
    portrait_id = current["character_card_id"]
    portrait_actor = db.get_character_card_by_id(portrait_id)
    assert portrait_actor is not None
    db.update_character_card(
        portrait_id,
        {"image": b"locally replaced portrait"},
        expected_version=portrait_actor["version"],
    )
    update_review = importer.inspect_archive(
        (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
    )

    updated = activation.activate(update_review, "update_existing")

    assert updated.local_actor_id == original.local_actor_id
    profile = local_service.get_persona_profile(original.local_actor_id)
    assert profile["name"] == "Golden"
    assert profile["description"] == "Keep this local description"
    assert profile["version"] == current["version"] + 2
    identity = repository.get_identity("persona", original.local_actor_id)
    assert identity is not None
    assert identity.portable_uuid == original.portable_uuid
    assert db.get_character_card_by_id(portrait_id)["image"] == PNG_1X1


def test_persona_review_becomes_stale_when_portrait_authority_changes(
    activation_components,
) -> None:
    activation, importer, _repository, local_service, db = activation_components
    created = activation.activate(
        importer.inspect_archive(
            (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
        ),
        "create_new",
    )
    review = importer.inspect_archive(
        (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
    )
    profile = local_service.get_persona_profile(created.local_actor_id)
    portrait_id = profile["character_card_id"]
    portrait_actor = db.get_character_card_by_id(portrait_id)
    assert portrait_actor is not None
    db.update_character_card(
        portrait_id,
        {"image": b"concurrent portrait"},
        expected_version=portrait_actor["version"],
    )

    with pytest.raises(ActorPackActivationError) as raised:
        activation.activate(review, "update_existing")

    assert raised.value.category == "actor_pack_import_review_stale"


def test_update_persona_rebinds_shared_portrait_without_mutating_character(
    activation_components,
) -> None:
    activation, importer, repository, local_service, db = activation_components
    shared_id = db.add_character_card(
        {"name": "Shared portrait", "image": b"shared portrait bytes"}
    )
    assert shared_id is not None
    persona = local_service.create_persona_profile(
        {
            "id": "local-persona-shared",
            "name": "Existing Persona",
            "character_card_id": shared_id,
        }
    )
    with db.transaction(immediate=True):
        repository._assign_identity_in_transaction(
            "persona",
            persona["id"],
            portable_uuid=PORTABLE_UUID,
        )
    review = importer.inspect_archive(
        (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
    )

    activation.activate(review, "update_existing")

    assert db.get_character_card_by_id(shared_id)["image"] == b"shared portrait bytes"
    updated = local_service.get_persona_profile(persona["id"])
    assert updated["character_card_id"] != shared_id
    owned = db.get_character_card_by_id(updated["character_card_id"])
    assert owned["image"] == PNG_1X1
    assert owned["extensions"]["actor_pack_persona_portrait_owner"] == persona["id"]


def test_persona_portrait_anchor_uses_normal_sequential_character_id(
    activation_components,
) -> None:
    activation, importer, _repository, _local_service, db = activation_components
    result = activation.activate(
        importer.inspect_archive(
            (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
        ),
        "create_new",
    )
    profile = activation.local_service.get_persona_profile(result.local_actor_id)
    portrait_id = profile["character_card_id"]

    following_id = db.add_character_card({"name": "Following Character"})

    assert portrait_id < 1_000
    assert following_id == portrait_id + 1
    assert portrait_id not in {
        card["id"] for card in db.list_character_cards(limit=1_000)
    }
    assert portrait_id not in {
        card["id"] for card in db.list_character_cards_page(limit=1_000, offset=0)
    }
    assert portrait_id not in {
        card["id"] for card in db.search_character_cards("Golden")
    }
    with pytest.raises(InputError):
        db.add_character_card(
            {
                "name": "Spoofed internal Character",
                "extensions": {"actor_pack_persona_portrait_owner": "spoofed"},
            }
        )
    with pytest.raises(InputError):
        db.update_character_card(
            following_id,
            {"extensions": {"actor_pack_persona_portrait_owner": "spoofed"}},
            expected_version=1,
        )


def test_persona_portrait_plans_reserve_distinct_sequential_ids(
    activation_components,
) -> None:
    activation, _importer, _repository, _local_service, db = activation_components

    first = activation._persona_portrait_plan(
        {"id": "local-persona-first"}, PNG_1X1, "First"
    )
    second = activation._persona_portrait_plan(
        {"id": "local-persona-second"}, PNG_1X1, "Second"
    )
    following_id = db.add_character_card({"name": "After reservations"})

    assert second.character_id == first.character_id + 1
    assert following_id == second.character_id + 1


def test_character_activation_binds_included_shared_visual_identity(
    activation_components, tmp_path: Path
) -> None:
    activation, importer, _repository, _local_service, db = activation_components
    digest = hashlib.sha256(PNG_1X1).hexdigest()
    asset_path = "shared-visual-identity/assets/asset-0001.png"
    manifest = {
        "schema_id": SAMIRA_MANIFEST_SCHEMA_ID,
        "pack_id": "actor.pack.import.test",
        "title": "Imported reactions",
        "license": "MIT",
        "default_expression_key": "neutral",
        "assets": [
            {
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
        ],
    }
    manifest["pack_content_sha256"] = compute_pack_content_sha256(manifest)
    section = ActorPackExportSection(
        kind="shared-visual-identity",
        manifest_path="shared-visual-identity/manifest.json",
        graph_identity=(1,),
        license="MIT",
        provenance="test",
        manifest_bytes=canonical_json_bytes(manifest),
        assets=(ActorPackExportFile(asset_path, digest, PNG_1X1),),
    )
    actor_payload = canonicalize_actor_payload(
        "character",
        "223e4567-e89b-42d3-a456-426614174000",
        {"name": "Visual import"},
    )
    archive = (tmp_path / "shared.tldw-actor-pack").resolve()
    with archive.open("wb+") as sink:
        write_actor_pack_archive(
            ActorPackExportSnapshot(
                actor_kind="character",
                actor_revision=1,
                portable_uuid="223e4567-e89b-42d3-a456-426614174000",
                identity_version=1,
                portrait_name="portrait.png",
                portrait_sha256=digest,
                local_actor_id="source",
                actor_payload=actor_payload,
                portrait_bytes=PNG_1X1,
                sections=(section,),
            ),
            sink,
        )

    review = importer.inspect_archive(archive)
    result = activation.activate(review, "create_new")

    graph = VisualIdentityRepository(db).get_active_actor_pack(
        "character", int(result.local_actor_id)
    )
    assert graph is not None
    assert graph["pack"]["title"] == "Imported reactions"
    storage = graph["assets"][0]["storage_relpath"]
    assert (tmp_path / "visual_identities" / storage).read_bytes() == PNG_1X1


def test_persona_activation_commits_included_runtime_visual_with_identity(
    activation_components, tmp_path: Path
) -> None:
    activation, importer, _repository, _local_service, db = activation_components
    digest = hashlib.sha256(PNG_1X1).hexdigest()
    asset_path = "persona-runtime/assets/asset-0001.png"
    states = {
        state: {"animation_id": "idle"}
        for state in ("idle", "listening", "thinking", "speaking", "error")
    }
    manifest = {
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
    section = ActorPackExportSection(
        kind="persona-runtime",
        manifest_path="persona-runtime/manifest.json",
        graph_identity=(1,),
        license="CC0-1.0",
        provenance="test",
        manifest_bytes=canonical_json_bytes(manifest),
        assets=(ActorPackExportFile(asset_path, digest, PNG_1X1),),
    )
    portable_uuid = "323e4567-e89b-42d3-a456-426614174000"
    archive = (tmp_path / "persona-runtime.tldw-actor-pack").resolve()
    with archive.open("wb+") as sink:
        write_actor_pack_archive(
            ActorPackExportSnapshot(
                actor_kind="persona",
                actor_revision=1,
                portable_uuid=portable_uuid,
                identity_version=1,
                portrait_name="portrait.png",
                portrait_sha256=digest,
                local_actor_id="source",
                actor_payload=canonicalize_actor_payload(
                    "persona", portable_uuid, {"name": "Visual Persona"}
                ),
                portrait_bytes=PNG_1X1,
                sections=(section,),
            ),
            sink,
        )

    review = importer.inspect_archive(archive)
    result = activation.activate(review, "create_new")

    graph = PersonaVisualRepository(db).get_active_persona_pack(result.local_actor_id)
    assert graph is not None
    assert graph.identity.persona_revision == 1
    assert graph.assets[0].asset_key == "idle"
    storage = db.execute_query(
        "SELECT storage_relpath FROM persona_visual_assets WHERE id = ?",
        (graph.assets[0].id,),
    ).fetchone()[0]
    assert (tmp_path / storage).read_bytes() == PNG_1X1


def test_character_update_preserves_omitted_shared_visual_byte_for_byte(
    activation_components, tmp_path: Path
) -> None:
    activation, importer, _repository, _local_service, db = activation_components
    archive = _write_archive(
        tmp_path / "character-with-visual.tldw-actor-pack",
        actor_kind="character",
        portable_uuid=PORTABLE_UUID,
        sections=(_shared_visual_section(),),
    )
    created = activation.activate(importer.inspect_archive(archive), "create_new")
    visual_repository = VisualIdentityRepository(db)
    before = visual_repository.get_active_actor_pack(
        "character", int(created.local_actor_id)
    )
    assert before is not None

    review = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )
    activation.activate(review, "update_existing")

    assert (
        visual_repository.get_active_actor_pack(
            "character", int(created.local_actor_id)
        )
        == before
    )


def test_persona_update_preserves_omitted_runtime_visual_byte_for_byte(
    activation_components, tmp_path: Path
) -> None:
    activation, importer, _repository, _local_service, db = activation_components
    archive = _write_archive(
        tmp_path / "persona-with-visual.tldw-actor-pack",
        actor_kind="persona",
        portable_uuid=PORTABLE_UUID,
        sections=(_persona_runtime_section(),),
    )
    created = activation.activate(importer.inspect_archive(archive), "create_new")
    visual_repository = PersonaVisualRepository(db)
    before = visual_repository.get_active_persona_pack(created.local_actor_id)
    assert before is not None

    review = importer.inspect_archive(
        (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
    )
    activation.activate(review, "update_existing")

    assert visual_repository.get_active_persona_pack(created.local_actor_id) == before


def test_character_visual_failure_rolls_back_actor_identity_and_visual_rows(
    activation_components, tmp_path: Path, monkeypatch
) -> None:
    activation, importer, repository, _local_service, db = activation_components
    archive = _write_archive(
        tmp_path / "character-rollback.tldw-actor-pack",
        actor_kind="character",
        portable_uuid=PORTABLE_UUID,
        sections=(_shared_visual_section(),),
    )
    review = importer.inspect_archive(archive)

    def fail_visual(**_kwargs) -> None:
        raise ValueError("private failure")

    monkeypatch.setattr(
        activation.visual_identity_repository, "activate_pack", fail_visual
    )
    with pytest.raises(ActorPackActivationError) as raised:
        activation.activate(review, "create_new")

    assert raised.value.category == "actor_pack_import_activation_failed"
    assert repository.get_identity_by_portable_uuid(PORTABLE_UUID) is None
    assert all(
        card["name"] != "Visual import"
        for card in db.list_character_cards(limit=100, offset=0)
    )
    for table in (
        "visual_identity_packs",
        "visual_identity_pack_versions",
        "visual_identity_assets",
        "visual_identity_bindings",
    ):
        assert db.execute_query(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_persona_visual_failure_compensates_profile_identity_and_visual_rows(
    activation_components, tmp_path: Path, monkeypatch
) -> None:
    activation, importer, repository, local_service, db = activation_components
    archive = _write_archive(
        tmp_path / "persona-rollback.tldw-actor-pack",
        actor_kind="persona",
        portable_uuid=PORTABLE_UUID,
        sections=(_persona_runtime_section(),),
    )
    review = importer.inspect_archive(archive)

    def fail_visual(**_kwargs) -> None:
        raise ValueError("private failure")

    monkeypatch.setattr(
        activation.persona_visual_repository,
        "_activate_new_pack_in_transaction",
        fail_visual,
    )
    with pytest.raises(ActorPackActivationError) as raised:
        activation.activate(review, "create_new")

    assert raised.value.category == "actor_pack_import_activation_failed"
    assert repository.get_identity_by_portable_uuid(PORTABLE_UUID) is None
    assert local_service.list_persona_profiles() == []
    assert repository.list_persona_intents() == ()
    for table in (
        "persona_visual_packs",
        "persona_visual_pack_versions",
        "persona_visual_assets",
        "persona_visual_bindings",
    ):
        assert db.execute_query(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_review_becomes_stale_when_shared_visual_authority_changes(
    activation_components, tmp_path: Path
) -> None:
    activation, importer, _repository, _local_service, db = activation_components
    archive = _write_archive(
        tmp_path / "character-stale-visual.tldw-actor-pack",
        actor_kind="character",
        portable_uuid=PORTABLE_UUID,
        sections=(_shared_visual_section(),),
    )
    created = activation.activate(importer.inspect_archive(archive), "create_new")
    review = importer.inspect_archive(
        (FIXTURES / "minimal-character.tldw-actor-pack").resolve()
    )
    db.execute_query(
        """
        UPDATE visual_identity_bindings
           SET version = version + 1
         WHERE actor_kind = 'character' AND actor_id = ? AND status = 'active'
        """,
        (created.local_actor_id,),
    )

    with pytest.raises(ActorPackActivationError) as raised:
        activation.activate(review, "update_existing")

    assert raised.value.category == "actor_pack_import_review_stale"
    assert db.get_character_card_by_id(int(created.local_actor_id))["name"] == (
        "Visual import"
    )


def test_import_rejects_shared_visual_manifest_dimension_mismatch(
    activation_components, tmp_path: Path
) -> None:
    _activation, importer, _repository, _local_service, _db = activation_components
    section = _shared_visual_section()
    manifest = json.loads(section.manifest_bytes)
    manifest["assets"][0]["width"] = 2
    manifest["pack_content_sha256"] = compute_pack_content_sha256(manifest)
    malformed = replace(section, manifest_bytes=canonical_json_bytes(manifest))
    archive = _write_archive(
        tmp_path / "bad-shared-visual.tldw-actor-pack",
        actor_kind="character",
        portable_uuid=PORTABLE_UUID,
        sections=(malformed,),
    )

    with pytest.raises(ActorPackImportError) as raised:
        importer.inspect_archive(archive)

    assert raised.value.category == "actor_pack_import_invalid"


def test_import_rejects_persona_runtime_without_required_state(
    activation_components, tmp_path: Path
) -> None:
    _activation, importer, _repository, _local_service, _db = activation_components
    section = _persona_runtime_section()
    manifest = json.loads(section.manifest_bytes)
    del manifest["states"]["error"]
    malformed = replace(section, manifest_bytes=canonical_json_bytes(manifest))
    archive = _write_archive(
        tmp_path / "bad-persona-runtime.tldw-actor-pack",
        actor_kind="persona",
        portable_uuid=PORTABLE_UUID,
        sections=(malformed,),
    )

    with pytest.raises(ActorPackImportError) as raised:
        importer.inspect_archive(archive)

    assert raised.value.category == "actor_pack_import_invalid"


def test_persona_update_replaces_present_shared_visual_and_preserves_runtime(
    activation_components, tmp_path: Path
) -> None:
    activation, importer, _repository, _local_service, db = activation_components
    initial = _write_archive(
        tmp_path / "persona-both-sections.tldw-actor-pack",
        actor_kind="persona",
        portable_uuid=PORTABLE_UUID,
        sections=(_shared_visual_section(), _persona_runtime_section()),
    )
    created = activation.activate(importer.inspect_archive(initial), "create_new")
    shared_repository = VisualIdentityRepository(db)
    runtime_repository = PersonaVisualRepository(db)
    shared_before = shared_repository.get_active_actor_pack(
        "persona", created.local_actor_id
    )
    runtime_before = runtime_repository.get_active_persona_pack(created.local_actor_id)
    assert shared_before is not None and runtime_before is not None
    shared_only = _write_archive(
        tmp_path / "persona-shared-update.tldw-actor-pack",
        actor_kind="persona",
        portable_uuid=PORTABLE_UUID,
        sections=(_shared_visual_section(),),
    )

    activation.activate(importer.inspect_archive(shared_only), "update_existing")

    shared_after = shared_repository.get_active_actor_pack(
        "persona", created.local_actor_id
    )
    assert shared_after is not None
    assert shared_after["pack"]["id"] != shared_before["pack"]["id"]
    assert (
        runtime_repository.get_active_persona_pack(created.local_actor_id)
        == runtime_before
    )


def test_persona_ambiguous_post_commit_keeps_identity_and_visual_graph(
    activation_components, tmp_path: Path, monkeypatch
) -> None:
    activation, importer, repository, local_service, db = activation_components
    archive = _write_archive(
        tmp_path / "persona-ambiguous-commit.tldw-actor-pack",
        actor_kind="persona",
        portable_uuid=PORTABLE_UUID,
        sections=(_persona_runtime_section(),),
    )
    commit = repository.commit_persona_intent

    def commit_then_raise(intent_id: str, **kwargs):
        commit(intent_id, **kwargs)
        raise ValueError("ambiguous commit")

    monkeypatch.setattr(repository, "commit_persona_intent", commit_then_raise)
    result = activation.activate(importer.inspect_archive(archive), "create_new")

    assert repository.get_identity("persona", result.local_actor_id) is not None
    assert local_service.get_persona_profile(result.local_actor_id)["name"] == (
        "Visual import"
    )
    assert (
        PersonaVisualRepository(db).get_active_persona_pack(result.local_actor_id)
        is not None
    )
    assert repository.list_persona_intents() == ()
