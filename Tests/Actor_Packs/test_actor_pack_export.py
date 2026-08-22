"""Consistent local actor snapshots for Actor Pack export."""

from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

import pytest
from PIL import Image

from tldw_chatbook.Actor_Packs.export import (
    ActorPackExportError,
    ActorPackExportService,
    ActorPackExportSnapshot,
)
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


PORTABLE_UUID = "123e4567-e89b-42d3-a456-426614174000"


def _png(color: str = "red") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


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
