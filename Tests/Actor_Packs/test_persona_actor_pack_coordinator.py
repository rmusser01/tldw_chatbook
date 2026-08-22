"""Real-file/real-SQLite Persona Actor Pack coordination coverage."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.persona_coordinator import (
    PersonaActorPackCoordinator,
    PersonaActorPackCoordinatorError,
)
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


UUID_A = "123e4567-e89b-42d3-a456-426614174000"


@pytest.fixture
def components(tmp_path: Path):
    database = CharactersRAGDB(tmp_path / "actors.db", client_id="coordinator")
    repository = ActorPackRepository(database, uuid_factory=lambda: uuid.UUID(UUID_A))
    store = tmp_path / "personas.json"
    service = LocalCharacterPersonaService(database, persona_store_path=store)
    coordinator = PersonaActorPackCoordinator(repository, service)
    yield coordinator, repository, service, store
    database.close_connection()


def _profile(persona_id: str = "guide") -> dict[str, object]:
    return {
        "id": persona_id,
        "name": "Guide",
        "version": 1,
        "deleted": False,
        "is_active": True,
        "character_card_id": 7,
    }


def _stored_profile(store: Path, persona_id: str = "guide") -> dict[str, object] | None:
    if not store.exists():
        return None
    payload = json.loads(store.read_text(encoding="utf-8"))
    return next(
        (item for item in payload["profiles"] if item.get("id") == persona_id),
        None,
    )


def test_create_orders_prepared_json_commit_and_cleanup(components) -> None:
    coordinator, repository, _service, store = components
    observed: list[str] = []

    def observe(phase: str) -> None:
        observed.append(phase)
        intents = repository.list_persona_intents()
        identity = repository.get_identity("persona", "guide")
        if phase == "prepared":
            assert len(intents) == 1 and intents[0].state == "prepared"
            assert _stored_profile(store) is None
            assert identity is None
        elif phase == "profile_replaced":
            assert len(intents) == 1 and intents[0].state == "prepared"
            assert _stored_profile(store) == _profile()
            assert identity is None
        elif phase == "committed":
            assert len(intents) == 1 and intents[0].state == "committed"
            assert _stored_profile(store) == _profile()
            assert identity is not None and identity.portable_uuid == UUID_A

    result = coordinator.create_persona(
        _profile(), portable_uuid=UUID_A, phase_hook=observe
    )

    assert observed == ["prepared", "profile_replaced", "committed"]
    assert result.identity.portable_uuid == UUID_A
    assert result.cleanup_pending is False
    assert repository.list_persona_intents() == ()
    assert coordinator.blocked_intent_ids == ()


@pytest.mark.parametrize("cancel_phase", ["prepared", "profile_replaced"])
def test_cancellation_compensates_owned_work(components, cancel_phase: str) -> None:
    coordinator, repository, service, store = components
    cancelled = False

    def observe(phase: str) -> None:
        nonlocal cancelled
        if phase == cancel_phase:
            cancelled = True

    with pytest.raises(
        PersonaActorPackCoordinatorError, match="actor_pack_creation_cancelled"
    ):
        coordinator.create_persona(
            _profile(),
            portable_uuid=UUID_A,
            phase_hook=observe,
            cancel_requested=lambda: cancelled,
        )

    assert repository.list_persona_intents() == ()
    assert repository.get_identity("persona", "guide") is None
    assert _stored_profile(store) is None
    with pytest.raises(ValueError, match="not_found"):
        service.get_persona_profile("guide")


def test_database_failure_compensates_profile_and_intent(
    components, monkeypatch
) -> None:
    coordinator, repository, _service, store = components

    def fail(_intent_id: str):
        raise ValueError("/Users/private/database")

    monkeypatch.setattr(repository, "commit_persona_intent", fail)
    with pytest.raises(
        PersonaActorPackCoordinatorError, match="actor_pack_creation_failed"
    ) as error:
        coordinator.create_persona(_profile(), portable_uuid=UUID_A)
    assert "private" not in repr(error.value)
    assert repository.list_persona_intents() == ()
    assert repository.get_identity("persona", "guide") is None
    assert _stored_profile(store) is None


def test_cleanup_failure_preserves_committed_success_for_recovery(
    components, monkeypatch
) -> None:
    coordinator, repository, _service, store = components
    cleanup = repository.cleanup_persona_intent

    def fail(*_args, **_kwargs):
        raise ValueError("cleanup failed")

    monkeypatch.setattr(repository, "cleanup_persona_intent", fail)
    result = coordinator.create_persona(_profile(), portable_uuid=UUID_A)
    assert result.cleanup_pending is True
    assert repository.get_identity("persona", "guide") == result.identity
    assert _stored_profile(store) == _profile()
    assert repository.list_persona_intents()[0].state == "committed"

    monkeypatch.setattr(repository, "cleanup_persona_intent", cleanup)
    recovery = coordinator.recover()
    assert recovery.cleaned == 1
    assert recovery.compensated == 0
    assert recovery.quarantined == 0
    assert repository.list_persona_intents() == ()


def test_ambiguous_post_commit_error_preserves_committed_success(
    components, monkeypatch
) -> None:
    coordinator, repository, _service, store = components
    commit = repository.commit_persona_intent

    def commit_then_raise(intent_id: str, **kwargs):
        commit(intent_id, **kwargs)
        raise ValueError("ambiguous commit")

    monkeypatch.setattr(repository, "commit_persona_intent", commit_then_raise)
    result = coordinator.create_persona(_profile(), portable_uuid=UUID_A)

    assert result.identity.portable_uuid == UUID_A
    assert _stored_profile(store) == _profile()
    assert repository.get_identity("persona", "guide") == result.identity
    assert repository.list_persona_intents() == ()


def test_external_store_change_before_sqlite_commit_quarantines_without_guessing(
    components,
) -> None:
    coordinator, repository, _service, store = components

    def replace_after_apply(phase: str) -> None:
        if phase != "profile_replaced":
            return
        payload = json.loads(store.read_text(encoding="utf-8"))
        payload["profiles"][0]["name"] = "External"
        store.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    with pytest.raises(
        PersonaActorPackCoordinatorError, match="actor_pack_creation_blocked"
    ):
        coordinator.create_persona(
            _profile(), portable_uuid=UUID_A, phase_hook=replace_after_apply
        )

    assert repository.get_identity("persona", "guide") is None
    retained = repository.list_persona_intents()
    assert len(retained) == 1 and retained[0].state == "quarantined"
    assert _stored_profile(store)["name"] == "External"


def test_recovery_cleans_prepared_old_json_old_sqlite(components) -> None:
    coordinator, repository, service, _store = components
    plan = service._actor_pack_plan_persona_profile(_profile(), operation="create")
    repository.prepare_persona_intent(
        persona_id="guide",
        operation="create",
        old_profile=None,
        new_profile=_profile(),
        old_store_sha256=plan.old_store_sha256,
        new_store_sha256=plan.new_store_sha256,
        old_registry_uuid=None,
        new_registry_uuid=UUID_A,
        intent_id="a" * 32,
    )

    result = coordinator.recover()
    assert (result.cleaned, result.compensated, result.quarantined) == (1, 0, 0)
    assert repository.list_persona_intents() == ()


def test_recovery_compensates_prepared_new_json_old_sqlite(components) -> None:
    coordinator, repository, service, store = components
    plan = service._actor_pack_plan_persona_profile(_profile(), operation="create")
    repository.prepare_persona_intent(
        persona_id="guide",
        operation="create",
        old_profile=None,
        new_profile=_profile(),
        old_store_sha256=plan.old_store_sha256,
        new_store_sha256=plan.new_store_sha256,
        old_registry_uuid=None,
        new_registry_uuid=UUID_A,
        intent_id="b" * 32,
    )
    service._actor_pack_apply_persona_plan(plan)

    result = coordinator.recover()
    assert (result.cleaned, result.compensated, result.quarantined) == (1, 1, 0)
    assert _stored_profile(store) is None
    assert repository.get_identity("persona", "guide") is None


def test_recovery_retains_committed_new_json_new_sqlite(components) -> None:
    coordinator, repository, service, store = components
    plan = service._actor_pack_plan_persona_profile(_profile(), operation="create")
    intent = repository.prepare_persona_intent(
        persona_id="guide",
        operation="create",
        old_profile=None,
        new_profile=_profile(),
        old_store_sha256=plan.old_store_sha256,
        new_store_sha256=plan.new_store_sha256,
        old_registry_uuid=None,
        new_registry_uuid=UUID_A,
        intent_id="c" * 32,
    )
    service._actor_pack_apply_persona_plan(plan)
    repository.commit_persona_intent(intent.intent_id)

    result = coordinator.recover()
    assert (result.cleaned, result.compensated, result.quarantined) == (1, 0, 0)
    assert _stored_profile(store) == _profile()
    assert repository.get_identity("persona", "guide") is not None


@pytest.mark.parametrize(
    ("store_state", "identity_state"), [("old", "new"), ("new", "new")]
)
def test_recovery_quarantines_prepared_third_authority(
    components, store_state: str, identity_state: str
) -> None:
    coordinator, repository, service, store = components
    plan = service._actor_pack_plan_persona_profile(_profile(), operation="create")
    intent = repository.prepare_persona_intent(
        persona_id="guide",
        operation="create",
        old_profile=None,
        new_profile=_profile(),
        old_store_sha256=plan.old_store_sha256,
        new_store_sha256=plan.new_store_sha256,
        old_registry_uuid=None,
        new_registry_uuid=UUID_A,
        intent_id="d" * 32,
    )
    if store_state == "new":
        service._actor_pack_apply_persona_plan(plan)
    if identity_state == "new":
        repository.assign_identity("persona", "guide")

    result = coordinator.recover()
    assert (result.cleaned, result.compensated, result.quarantined) == (0, 0, 1)
    retained = repository.list_persona_intents()
    assert retained[0].intent_id == intent.intent_id
    assert retained[0].state == "quarantined"
    assert coordinator.blocked_intent_ids == (intent.intent_id,)
    assert (_stored_profile(store) is not None) is (store_state == "new")


def test_unknown_store_digest_is_quarantined_without_guessing(components) -> None:
    coordinator, repository, service, store = components
    plan = service._actor_pack_plan_persona_profile(_profile(), operation="create")
    repository.prepare_persona_intent(
        persona_id="guide",
        operation="create",
        old_profile=None,
        new_profile=_profile(),
        old_store_sha256=plan.old_store_sha256,
        new_store_sha256=plan.new_store_sha256,
        old_registry_uuid=None,
        new_registry_uuid=UUID_A,
        intent_id="e" * 32,
    )
    store.write_text('{"profiles":[],"external":true}', encoding="utf-8")

    result = coordinator.recover()
    assert result.quarantined == 1
    assert store.read_text(encoding="utf-8") == '{"profiles":[],"external":true}'


def test_store_plan_preserves_unrelated_sections(tmp_path: Path) -> None:
    store = tmp_path / "personas.json"
    payload = {
        "profiles": [],
        "exemplars": [],
        "character_exemplars": [],
        "chat_settings": {},
        "chat_greeting_selections": {},
        "chat_presets": [],
        "character_memories": [],
        "future_section": {"kept": [1, 2, 3]},
    }
    store.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    service = LocalCharacterPersonaService(None, persona_store_path=store)
    plan = service._actor_pack_plan_persona_profile(_profile(), operation="create")
    service._actor_pack_apply_persona_plan(plan)

    assert json.loads(store.read_text(encoding="utf-8"))["future_section"] == {
        "kept": [1, 2, 3]
    }
