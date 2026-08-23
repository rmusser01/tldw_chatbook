"""Pack-ready local Character and Persona creation coverage."""

from __future__ import annotations

import hashlib
import inspect
import io
import threading
import uuid
from pathlib import Path

import pytest
from PIL import Image

from tldw_chatbook.Actor_Packs.creation import (
    ActorPackCreationError,
    ActorPackCreationService,
)
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


UUID_A = "123e4567-e89b-42d3-a456-426614174000"
UUID_B = "223e4567-e89b-42d3-a456-426614174000"


def _png(color: str = "red") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def components(tmp_path: Path):
    database = CharactersRAGDB(tmp_path / "actors.db", client_id="creation")
    values = iter((uuid.UUID(UUID_A), uuid.UUID(UUID_B)))
    repository = ActorPackRepository(database, uuid_factory=lambda: next(values))
    local_service = LocalCharacterPersonaService(
        database, persona_store_path=tmp_path / "personas.json"
    )
    coordinator = PersonaActorPackCoordinator(repository, local_service)
    creation = ActorPackCreationService(database, repository, coordinator)
    yield creation, repository, coordinator, local_service, database
    database.close_connection()


def _character_count(database: CharactersRAGDB, name: str) -> int:
    row = database.execute_query(
        "SELECT COUNT(*) FROM character_cards WHERE name = ?", (name,)
    ).fetchone()
    assert row is not None
    return int(row[0])


def test_character_and_registry_commit_in_one_transaction(components) -> None:
    creation, repository, _coordinator, _service, database = components
    result = creation.create_character(
        {"name": "Pack Hero", "description": "Portable"},
        portrait_name="portrait.png",
        portrait_bytes=_png(),
    )

    assert result.actor_kind == "character"
    assert result.portable_uuid == UUID_A
    assert result.local_actor_id.isdigit()
    character = database.get_character_card_by_id(int(result.local_actor_id))
    assert character is not None and character["image"] == _png()
    assert repository.get_identity("character", int(result.local_actor_id)) is not None
    assert _character_count(database, "Pack Hero") == 1


@pytest.mark.parametrize(
    ("portrait_name", "portrait_bytes"),
    [
        ("portrait.png", b"not-an-image"),
        ("portrait.exe", _png()),
        ("portrait.png", b""),
    ],
)
def test_character_portrait_validation_precedes_mutation(
    components, portrait_name: str, portrait_bytes: bytes
) -> None:
    creation, repository, _coordinator, _service, database = components
    with pytest.raises(ActorPackCreationError, match="actor_pack_portrait_invalid"):
        creation.create_character(
            {"name": "Bad Portrait"},
            portrait_name=portrait_name,
            portrait_bytes=portrait_bytes,
        )
    assert _character_count(database, "Bad Portrait") == 0
    assert repository.get_identity("persona", "Bad Portrait") is None


def test_uuid_collision_rolls_back_character_row(components) -> None:
    creation, repository, _coordinator, _service, database = components
    repository.assign_identity("persona", "existing")
    repository._uuid_factory = lambda: uuid.UUID(UUID_A)

    with pytest.raises(ActorPackCreationError, match="actor_pack_creation_failed"):
        creation.create_character(
            {"name": "Collision"},
            portrait_name="portrait.png",
            portrait_bytes=_png(),
        )
    assert _character_count(database, "Collision") == 0


def test_character_cancel_and_stale_authority_leave_no_rows(components) -> None:
    creation, _repository, _coordinator, _service, database = components
    with pytest.raises(ActorPackCreationError, match="actor_pack_creation_cancelled"):
        creation.create_character(
            {"name": "Cancelled"},
            portrait_name="portrait.png",
            portrait_bytes=_png(),
            cancel_requested=lambda: True,
        )
    with pytest.raises(
        ActorPackCreationError, match="actor_pack_creation_authority_changed"
    ):
        creation.create_character(
            {"name": "Stale"},
            portrait_name="portrait.png",
            portrait_bytes=_png(),
            authority_guard=lambda: False,
        )
    assert _character_count(database, "Cancelled") == 0
    assert _character_count(database, "Stale") == 0


def test_character_authority_change_after_insert_rolls_back_both_rows(
    components,
) -> None:
    creation, repository, _coordinator, _service, database = components
    current = True

    def change_authority(phase: str) -> None:
        nonlocal current
        if phase == "character_inserted":
            current = False

    with pytest.raises(
        ActorPackCreationError, match="actor_pack_creation_authority_changed"
    ):
        creation.create_character(
            {"name": "Stale after insert"},
            portrait_name="portrait.png",
            portrait_bytes=_png(),
            authority_guard=lambda: current,
            phase_hook=change_authority,
        )
    assert _character_count(database, "Stale after insert") == 0
    assert repository.get_identity("character", 1) is None


def test_character_creation_rejects_caller_owned_transaction(components) -> None:
    creation, _repository, _coordinator, _service, database = components
    connection = database.get_connection()
    connection.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(ActorPackCreationError, match="actor_pack_creation_failed"):
            creation.create_character(
                {"name": "Borrowed"},
                portrait_name="portrait.png",
                portrait_bytes=_png(),
            )
        assert connection.in_transaction
        assert _character_count(database, "Borrowed") == 0
    finally:
        connection.rollback()


def test_duplicate_submit_is_rejected_while_first_operation_owns_service(
    components,
) -> None:
    creation, _repository, _coordinator, _service, _database = components
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def pause(phase: str) -> None:
        if phase == "validated":
            entered.set()
            release.wait(timeout=3)

    def create_first() -> None:
        try:
            creation.create_character(
                {"name": "First"},
                portrait_name="portrait.png",
                portrait_bytes=_png(),
                phase_hook=pause,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=create_first)
    thread.start()
    assert entered.wait(timeout=2)
    try:
        with pytest.raises(
            ActorPackCreationError, match="actor_pack_creation_in_progress"
        ):
            creation.create_character(
                {"name": "Second"},
                portrait_name="portrait.png",
                portrait_bytes=_png(),
            )
    finally:
        release.set()
        thread.join(timeout=4)
    assert errors == []


def test_persona_creation_revalidates_exact_linked_character_portrait(
    components,
) -> None:
    creation, repository, _coordinator, service, database = components
    portrait = _png("blue")
    character_id = database.add_character_card({"name": "Portrait", "image": portrait})
    assert character_id is not None
    character = database.get_character_card_by_id(character_id)
    assert character is not None

    result = creation.create_persona(
        {"id": "guide", "name": "Guide", "character_card_id": character_id},
        source="local",
        expected_portrait_revision=int(character["version"]),
        expected_portrait_sha256=hashlib.sha256(portrait).hexdigest(),
    )

    assert result.actor_kind == "persona"
    assert result.local_actor_id == "guide"
    assert result.portable_uuid == UUID_A
    assert service.get_persona_profile("guide")["character_card_id"] == character_id
    assert repository.get_identity("persona", "guide") is not None


def test_server_persona_requires_save_local_copy(components) -> None:
    creation, repository, _coordinator, _service, _database = components
    with pytest.raises(ActorPackCreationError) as error:
        creation.create_persona(
            {"id": "guide", "name": "Guide", "character_card_id": 7},
            source="server",
            expected_portrait_revision=1,
            expected_portrait_sha256="a" * 64,
        )
    assert error.value.user_message == "Save a local copy first"
    assert repository.get_identity("persona", "guide") is None


def test_stale_portrait_after_json_prepare_never_commits_persona(components) -> None:
    creation, repository, coordinator, _service, database = components
    portrait = _png("green")
    character_id = database.add_character_card({"name": "Portrait", "image": portrait})
    assert character_id is not None
    character = database.get_character_card_by_id(character_id)
    assert character is not None

    def mutate(phase: str) -> None:
        if phase == "prepared":
            database.execute_query(
                "UPDATE character_cards SET image = ?, version = version + 1 WHERE id = ?",
                (_png("black"), character_id),
            )
            database.get_connection().commit()

    with pytest.raises(
        ActorPackCreationError, match="actor_pack_creation_authority_changed"
    ):
        creation.create_persona(
            {"id": "guide", "name": "Guide", "character_card_id": character_id},
            source="local",
            expected_portrait_revision=int(character["version"]),
            expected_portrait_sha256=hashlib.sha256(portrait).hexdigest(),
            phase_hook=mutate,
        )
    assert repository.get_identity("persona", "guide") is None
    assert coordinator.blocked_intent_ids == ()


def test_foundation_writes_no_archives_visuals_or_chats(
    components, tmp_path: Path
) -> None:
    creation, _repository, _coordinator, _service, database = components
    creation.create_character(
        {"name": "Scoped"},
        portrait_name="portrait.png",
        portrait_bytes=_png(),
    )

    assert list(tmp_path.rglob("*.zip")) == []
    for table in (
        "persona_visual_packs",
        "persona_visual_pack_versions",
        "persona_visual_bindings",
        "visual_identity_packs",
        "visual_identity_pack_versions",
        "visual_identity_bindings",
        "conversations",
    ):
        row = database.execute_query(f"SELECT COUNT(*) FROM {table}").fetchone()
        assert row is not None and int(row[0]) == 0


def test_app_owns_creation_only_after_recovery_and_before_surfaces() -> None:
    from tldw_chatbook.app import TldwCli

    wiring = inspect.getsource(TldwCli._wire_character_persona_services)
    recovery = wiring.index(".recover()")
    creation = wiring.index("ActorPackCreationService(")
    scope = wiring.index("CharacterPersonaScopeService(")
    assert recovery < creation < scope


def test_app_wires_live_actor_pack_services_when_the_database_opens(
    monkeypatch, tmp_path: Path
) -> None:
    """TASK-20970 review: the *open-database* half of the new wiring branch.

    The sibling above asserts source-text ordering, which a behaviour-only
    regression walks straight past: nulling all three services at the end of
    the ``else`` branch leaves that assertion (and every other test in
    `Tests/Actor_Packs`, `Tests/App`, `Tests/UI/test_actor_pack_creation_
    workflow.py`) green, because `TASK-20970`'s own coverage only pins the
    *degraded* side and the shared app factory patches
    `get_chachanotes_db_lazy` to `None` for every app-building test in the
    repo. Nothing behavioural asserted that a working database still yields
    working Actor Pack services. This does.
    """

    from unittest.mock import Mock

    from tldw_chatbook import app as app_module

    database = CharactersRAGDB(tmp_path / "wiring.db", client_id="wiring")
    try:
        monkeypatch.setattr(
            app_module.ServerCharacterPersonaService,
            "from_server_context_provider",
            Mock(return_value=Mock()),
        )
        monkeypatch.setattr(
            app_module.ServerChatDictionaryService,
            "from_server_context_provider",
            Mock(return_value=Mock()),
        )

        fake_app = Mock()
        fake_app.chachanotes_db = database
        fake_app.service_policy_enforcer = object()
        fake_app.server_context_provider = object()

        app_module.TldwCli._wire_character_persona_services(fake_app)

        assert isinstance(fake_app.actor_pack_repository, ActorPackRepository)
        assert fake_app.actor_pack_repository.db is database
        assert isinstance(
            fake_app.persona_actor_pack_coordinator, PersonaActorPackCoordinator
        )
        assert (
            fake_app.persona_actor_pack_coordinator.repository
            is fake_app.actor_pack_repository
        )
        assert isinstance(
            fake_app.actor_pack_creation_service, ActorPackCreationService
        )
        assert fake_app.actor_pack_creation_service.database is database
        assert (
            fake_app.actor_pack_creation_service.repository
            is fake_app.actor_pack_repository
        )
        assert (
            fake_app.actor_pack_creation_service.persona_coordinator
            is fake_app.persona_actor_pack_coordinator
        )
        # Recovery ran and found nothing to reconcile: no fixed category is
        # recorded, so the degraded/blocked/failed states stay distinguishable.
        assert fake_app.actor_pack_recovery_error is None
    finally:
        database.close_connection()
