from __future__ import annotations

import threading
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.activation import (
    ActorPackActivationError,
    ActorPackActivationService,
)
from tldw_chatbook.Actor_Packs.importer import ActorPackImportService
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


FIXTURES = Path(__file__).parent / "fixtures" / "export-golden"


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
    activation, importer, repository, local_service, _db = activation_components
    review = importer.inspect_archive(
        (FIXTURES / "minimal-persona.tldw-actor-pack").resolve()
    )

    result = activation.activate(review, "create_new")

    profile = local_service.get_persona_profile(result.local_actor_id)
    assert profile["name"] == "Golden"
    identity = repository.get_identity("persona", result.local_actor_id)
    assert identity is not None
    assert identity.portable_uuid == review.portable_uuid


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
    activation, importer, repository, local_service, _db = activation_components
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
