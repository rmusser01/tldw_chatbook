"""Crash-safe coordination between Persona JSON and portable identity rows."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)

from .repository import (
    ActorPackRepository,
    ActorPackRepositoryError,
    PersonaActorPackIntent,
    PortableActorIdentity,
)


class PersonaActorPackCoordinatorError(ValueError):
    """One fixed-category cross-store coordination failure."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class PersonaActorPackCreationResult:
    """Committed Persona portable identity and cleanup status."""

    identity: PortableActorIdentity
    cleanup_pending: bool


@dataclass(frozen=True, slots=True)
class PersonaActorPackRecoveryResult:
    """Bounded startup recovery summary without profile content."""

    cleaned: int
    compensated: int
    quarantined: int
    blocked_intent_ids: tuple[str, ...]


class PersonaActorPackCoordinator:
    """Coordinate one Persona mutation without pretending SQLite owns JSON."""

    def __init__(
        self,
        repository: ActorPackRepository,
        local_service: LocalCharacterPersonaService,
    ) -> None:
        self.repository = repository
        self.local_service = local_service
        self._blocked_intent_ids: tuple[str, ...] = ()

    @property
    def blocked_intent_ids(self) -> tuple[str, ...]:
        """Return opaque quarantined intent identifiers."""

        return self._blocked_intent_ids

    def create_persona(
        self,
        profile: Mapping[str, Any],
        *,
        portable_uuid: str,
        operation: str = "create",
        source_portable_uuid: str | None = None,
        cancel_requested: Callable[[], bool] = lambda: False,
        phase_hook: Callable[[str], None] | None = None,
    ) -> PersonaActorPackCreationResult:
        """Create/update one pack-ready Persona with compensating rollback."""

        if self._blocked_intent_ids:
            raise PersonaActorPackCoordinatorError("actor_pack_creation_blocked")
        try:
            plan = self.local_service._actor_pack_plan_persona_profile(
                profile, operation=operation
            )
            old_profile = (
                None
                if plan.old_profile_json is None
                else json.loads(plan.old_profile_json)
            )
            new_profile = json.loads(plan.new_profile_json)
            if operation == "update":
                current = self.repository.get_identity("persona", plan.persona_id)
                if current is None or current.portable_uuid != portable_uuid:
                    raise PersonaActorPackCoordinatorError(
                        "actor_pack_creation_authority_changed"
                    )
                old_registry_uuid = current.portable_uuid
            elif operation == "copy":
                old_registry_uuid = source_portable_uuid
            else:
                old_registry_uuid = None
            self._raise_if_cancelled(cancel_requested)
            intent = self.repository.prepare_persona_intent(
                persona_id=plan.persona_id,
                operation=operation,
                old_profile=old_profile,
                new_profile=new_profile,
                old_store_sha256=plan.old_store_sha256,
                new_store_sha256=plan.new_store_sha256,
                old_registry_uuid=old_registry_uuid,
                new_registry_uuid=portable_uuid,
            )
        except PersonaActorPackCoordinatorError:
            raise
        except (ActorPackRepositoryError, TypeError, ValueError, UnicodeError):
            raise PersonaActorPackCoordinatorError(
                "actor_pack_creation_failed"
            ) from None

        applied = False
        committed = False
        try:
            self._phase(phase_hook, "prepared")
            self._raise_if_cancelled(cancel_requested)
            self.local_service._actor_pack_apply_persona_plan(plan)
            applied = True
            self._phase(phase_hook, "profile_replaced")
            self._raise_if_cancelled(cancel_requested)
            identity, _ = self.repository.commit_persona_intent(
                intent.intent_id,
                authority_guard=lambda: (
                    self.local_service._actor_pack_store_state(
                        old_store_sha256=intent.old_store_sha256,
                        new_store_sha256=intent.new_store_sha256,
                    )
                    == "new"
                ),
            )
            committed = True
            self._phase(phase_hook, "committed")
        except BaseException as exc:
            if not committed:
                identity = self._committed_identity_if_present(intent)
                committed = identity is not None
            if not committed:
                self._compensate_failed_creation(intent, applied=applied)
            if isinstance(exc, PersonaActorPackCoordinatorError):
                raise
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if not committed:
                raise PersonaActorPackCoordinatorError(
                    "actor_pack_creation_failed"
                ) from None

        try:
            self.repository.cleanup_persona_intent(
                intent.intent_id, expected_state="committed"
            )
        except (ActorPackRepositoryError, ValueError):
            return PersonaActorPackCreationResult(identity, cleanup_pending=True)
        return PersonaActorPackCreationResult(identity, cleanup_pending=False)

    def recover(self) -> PersonaActorPackRecoveryResult:
        """Reconcile all intents before affected application surfaces mount."""

        cleaned = compensated = quarantined = 0
        blocked: list[str] = []
        try:
            intents = self.repository.list_persona_intents()
        except ActorPackRepositoryError:
            raise PersonaActorPackCoordinatorError(
                "actor_pack_recovery_failed"
            ) from None
        for intent in intents:
            if intent.state == "quarantined":
                blocked.append(intent.intent_id)
                quarantined += 1
                continue
            try:
                store_state = self.local_service._actor_pack_store_state(
                    old_store_sha256=intent.old_store_sha256,
                    new_store_sha256=intent.new_store_sha256,
                )
                identity = self.repository.get_identity("persona", intent.persona_id)
                identity_old = _identity_is_old(intent, identity)
                identity_new = (
                    identity is not None
                    and identity.portable_uuid == intent.new_registry_uuid
                )
                if intent.state == "prepared" and store_state == "old" and identity_old:
                    self.repository.cleanup_persona_intent(
                        intent.intent_id, expected_state="prepared"
                    )
                    cleaned += 1
                    continue
                if (
                    intent.state == "prepared"
                    and store_state == "new"
                    and identity_old
                    and not identity_new
                ):
                    self._compensate_intent(intent)
                    self.repository.cleanup_persona_intent(
                        intent.intent_id, expected_state="prepared"
                    )
                    compensated += 1
                    cleaned += 1
                    continue
                if (
                    intent.state == "committed"
                    and store_state == "new"
                    and identity_new
                ):
                    self.repository.cleanup_persona_intent(
                        intent.intent_id, expected_state="committed"
                    )
                    cleaned += 1
                    continue
                retained = self.repository.quarantine_persona_intent(
                    intent.intent_id, "authority_mismatch"
                )
                blocked.append(retained.intent_id)
                quarantined += 1
            except (ActorPackRepositoryError, TypeError, ValueError, UnicodeError):
                try:
                    retained = self.repository.quarantine_persona_intent(
                        intent.intent_id, "recovery_failed"
                    )
                except ActorPackRepositoryError:
                    raise PersonaActorPackCoordinatorError(
                        "actor_pack_recovery_failed"
                    ) from None
                blocked.append(retained.intent_id)
                quarantined += 1
        self._blocked_intent_ids = tuple(sorted(set(blocked)))
        return PersonaActorPackRecoveryResult(
            cleaned,
            compensated,
            quarantined,
            self._blocked_intent_ids,
        )

    @staticmethod
    def _phase(hook: Callable[[str], None] | None, phase: str) -> None:
        if hook is not None:
            hook(phase)

    @staticmethod
    def _raise_if_cancelled(cancel_requested: Callable[[], bool]) -> None:
        if cancel_requested():
            raise PersonaActorPackCoordinatorError("actor_pack_creation_cancelled")

    def _compensate_failed_creation(
        self, intent: PersonaActorPackIntent, *, applied: bool
    ) -> None:
        try:
            if applied:
                self._compensate_intent(intent)
            self.repository.cleanup_persona_intent(
                intent.intent_id, expected_state="prepared"
            )
        except (ActorPackRepositoryError, TypeError, ValueError, UnicodeError):
            try:
                retained = self.repository.quarantine_persona_intent(
                    intent.intent_id, "compensation_failed"
                )
                self._blocked_intent_ids = (retained.intent_id,)
            except ActorPackRepositoryError:
                self._blocked_intent_ids = (intent.intent_id,)
            raise PersonaActorPackCoordinatorError(
                "actor_pack_creation_blocked"
            ) from None

    def _compensate_intent(self, intent: PersonaActorPackIntent) -> None:
        self.local_service._actor_pack_compensate_persona_profile(
            persona_id=intent.persona_id,
            old_profile_json=intent.old_profile_json,
            new_profile_json=intent.new_profile_json,
            old_store_sha256=intent.old_store_sha256,
            new_store_sha256=intent.new_store_sha256,
        )

    def _committed_identity_if_present(
        self, intent: PersonaActorPackIntent
    ) -> PortableActorIdentity | None:
        """Resolve only the exact converged state after an ambiguous DB error."""

        try:
            stored = next(
                (
                    candidate
                    for candidate in self.repository.list_persona_intents()
                    if candidate.intent_id == intent.intent_id
                ),
                None,
            )
            identity = self.repository.get_identity("persona", intent.persona_id)
            store_state = self.local_service._actor_pack_store_state(
                old_store_sha256=intent.old_store_sha256,
                new_store_sha256=intent.new_store_sha256,
            )
        except (ActorPackRepositoryError, TypeError, ValueError, UnicodeError):
            return None
        if (
            stored is None
            or stored.state != "committed"
            or identity is None
            or identity.portable_uuid != intent.new_registry_uuid
            or store_state != "new"
        ):
            return None
        return identity


def _identity_is_old(
    intent: PersonaActorPackIntent, identity: PortableActorIdentity | None
) -> bool:
    if intent.operation in {"create", "copy"}:
        return identity is None
    return (
        identity is not None
        and intent.old_registry_uuid is not None
        and identity.portable_uuid == intent.old_registry_uuid
    )
