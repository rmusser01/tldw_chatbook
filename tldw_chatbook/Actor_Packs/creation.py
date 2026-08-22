"""Atomic local Character and coordinated Persona creation for Actor Packs."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_chatbook.tldw_api.character_persona_schemas import (
    CharacterCreateRequest,
    LocalPersonaProfileCreate,
)

from .contracts import ActorPackValidationError, validate_actor_portrait
from .persona_coordinator import (
    PersonaActorPackCoordinator,
    PersonaActorPackCoordinatorError,
)
from .repository import ActorPackRepository, ActorPackRepositoryError


class ActorPackCreationError(ValueError):
    """One stable, path-free Actor Pack creation failure."""

    def __init__(self, category: str, *, user_message: str | None = None) -> None:
        self.category = category
        self.user_message = user_message
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackCreationResult:
    """The local actor and stable portable identity created together."""

    actor_kind: str
    local_actor_id: str
    portable_uuid: str


class ActorPackCreationService:
    """Admit one pack-ready local actor creation operation at a time."""

    def __init__(
        self,
        database: CharactersRAGDB,
        repository: ActorPackRepository,
        persona_coordinator: PersonaActorPackCoordinator,
    ) -> None:
        self.database = database
        self.repository = repository
        self.persona_coordinator = persona_coordinator
        self._operation_lock = threading.Lock()

    def create_character(
        self,
        request_data: Mapping[str, Any],
        *,
        portrait_name: str,
        portrait_bytes: bytes,
        cancel_requested: Callable[[], bool] = lambda: False,
        authority_guard: Callable[[], bool] = lambda: True,
        phase_hook: Callable[[str], None] | None = None,
    ) -> ActorPackCreationResult:
        """Validate and commit one Character plus UUID in one transaction.

        Args:
            request_data: Character fields accepted by the local card schema.
            portrait_name: Filename used to validate the portrait media type.
            portrait_bytes: Complete bounded portrait payload.
            cancel_requested: Returns whether the owning UI operation was cancelled.
            authority_guard: Returns whether the captured creation authority is current.
            phase_hook: Optional test/diagnostic hook called at stable phase boundaries.

        Returns:
            The committed local Character id and portable UUID.

        Raises:
            ActorPackCreationError: If validation, authority, or persistence fails.
        """

        self._begin_operation()
        try:
            try:
                request = CharacterCreateRequest.model_validate(request_data)
                card = request.model_dump(exclude_none=True, mode="json")
            except (ValidationError, TypeError, ValueError):
                raise ActorPackCreationError("actor_pack_creation_invalid") from None
            try:
                validate_actor_portrait(portrait_name, portrait_bytes)
            except ActorPackValidationError:
                raise ActorPackCreationError("actor_pack_portrait_invalid") from None
            card.pop("image_base64", None)
            card["image"] = portrait_bytes
            self._phase(phase_hook, "validated")
            self._require_current(cancel_requested, authority_guard)
            self._require_no_transaction()
            try:
                with self.database.transaction(immediate=True) as cursor:
                    self._require_current(cancel_requested, authority_guard)
                    character_id = self.database._insert_character_card_in_transaction(
                        cursor, card, require_outermost=True
                    )
                    self._phase(phase_hook, "character_inserted")
                    self._require_current(cancel_requested, authority_guard)
                    identity = self.repository._assign_identity_in_transaction(
                        "character", character_id
                    )
            except ActorPackCreationError:
                raise
            except (
                ActorPackRepositoryError,
                CharactersRAGDBError,
                sqlite3.Error,
                TypeError,
                ValueError,
            ):
                raise ActorPackCreationError("actor_pack_creation_failed") from None
            self._phase(phase_hook, "committed")
            return ActorPackCreationResult(
                "character", str(character_id), identity.portable_uuid
            )
        finally:
            self._operation_lock.release()

    def create_persona(
        self,
        request_data: Mapping[str, Any],
        *,
        source: str,
        expected_portrait_revision: int,
        expected_portrait_sha256: str,
        cancel_requested: Callable[[], bool] = lambda: False,
        authority_guard: Callable[[], bool] = lambda: True,
        phase_hook: Callable[[str], None] | None = None,
    ) -> ActorPackCreationResult:
        """Coordinate one local Persona and UUID with linked-portrait authority.

        Args:
            request_data: Persona fields accepted by the local profile schema.
            source: Actor source; only ``local`` is eligible.
            expected_portrait_revision: Captured Character portrait revision.
            expected_portrait_sha256: Captured Character portrait content digest.
            cancel_requested: Returns whether the owning UI operation was cancelled.
            authority_guard: Returns whether the captured creation authority is current.
            phase_hook: Optional test/diagnostic hook called at stable phase boundaries.

        Returns:
            The committed local Persona id and portable UUID.

        Raises:
            ActorPackCreationError: If validation, authority, or persistence fails.
        """

        self._begin_operation()
        try:
            if type(source) is not str or source != "local":
                raise ActorPackCreationError(
                    "actor_pack_source_not_local",
                    user_message="Save a local copy first",
                )
            try:
                request = LocalPersonaProfileCreate.model_validate(request_data)
                payload = request.model_dump(exclude_none=True, mode="json")
                persona_id = payload.get("id") or f"local-persona-{uuid.uuid4().hex}"
                character_id = payload.get("character_card_id")
                if type(character_id) is not int or character_id < 1:
                    raise ValueError
                if (
                    type(expected_portrait_revision) is not int
                    or expected_portrait_revision < 1
                    or type(expected_portrait_sha256) is not str
                    or len(expected_portrait_sha256) != 64
                    or any(
                        char not in "0123456789abcdef"
                        for char in expected_portrait_sha256
                    )
                ):
                    raise ValueError
                now = datetime.now(timezone.utc).isoformat()
                payload.update(
                    {
                        "id": str(persona_id),
                        "created_at": now,
                        "last_modified": now,
                        "version": 1,
                        "deleted": False,
                    }
                )
            except (ValidationError, TypeError, ValueError):
                raise ActorPackCreationError("actor_pack_creation_invalid") from None
            try:
                portrait_name, portrait = self._portrait_snapshot(character_id)
                validate_actor_portrait(portrait_name, portrait)
            except (ActorPackValidationError, CharactersRAGDBError, ValueError):
                raise ActorPackCreationError("actor_pack_portrait_invalid") from None
            if not self._portrait_matches(
                character_id,
                expected_portrait_revision,
                expected_portrait_sha256,
            ):
                raise ActorPackCreationError("actor_pack_creation_authority_changed")
            try:
                portable_uuid = self.repository._new_portable_uuid()
            except (ActorPackRepositoryError, TypeError, ValueError):
                raise ActorPackCreationError("actor_pack_creation_failed") from None

            def combined_authority() -> bool:
                return self._guard_is_current(
                    authority_guard
                ) and self._portrait_matches(
                    character_id,
                    expected_portrait_revision,
                    expected_portrait_sha256,
                )

            try:
                result = self.persona_coordinator.create_persona(
                    payload,
                    portable_uuid=portable_uuid,
                    cancel_requested=cancel_requested,
                    authority_guard=combined_authority,
                    phase_hook=phase_hook,
                )
            except PersonaActorPackCoordinatorError as exc:
                categories = {
                    "actor_pack_creation_authority_changed",
                    "actor_pack_creation_blocked",
                    "actor_pack_creation_cancelled",
                }
                category = (
                    exc.category
                    if exc.category in categories
                    else "actor_pack_creation_failed"
                )
                raise ActorPackCreationError(category) from None
            return ActorPackCreationResult(
                "persona", str(persona_id), result.identity.portable_uuid
            )
        finally:
            self._operation_lock.release()

    def _portrait_snapshot(self, character_id: int) -> tuple[str, bytes]:
        character = self.database.get_character_card_by_id(character_id)
        if character is None or type(character.get("image")) is not bytes:
            raise ValueError
        portrait = bytes(character["image"])
        return _portrait_name(portrait), portrait

    def _portrait_matches(self, character_id: int, revision: int, sha256: str) -> bool:
        try:
            character = self.database.get_character_card_by_id(character_id)
            if character is None or type(character.get("image")) is not bytes:
                return False
            return (
                type(character.get("version")) is int
                and character["version"] == revision
                and hashlib.sha256(character["image"]).hexdigest() == sha256
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return False

    def _begin_operation(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise ActorPackCreationError("actor_pack_creation_in_progress")

    def _require_no_transaction(self) -> None:
        connection = self.database.get_connection()
        if connection.in_transaction or getattr(
            self.database._local, "transaction_depth", 0
        ):
            raise ActorPackCreationError("actor_pack_creation_failed")

    @classmethod
    def _require_current(
        cls,
        cancel_requested: Callable[[], bool],
        authority_guard: Callable[[], bool],
    ) -> None:
        if cancel_requested():
            raise ActorPackCreationError("actor_pack_creation_cancelled")
        if not cls._guard_is_current(authority_guard):
            raise ActorPackCreationError("actor_pack_creation_authority_changed")

    @staticmethod
    def _guard_is_current(authority_guard: Callable[[], bool]) -> bool:
        try:
            return authority_guard() is True
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return False

    @staticmethod
    def _phase(hook: Callable[[str], None] | None, phase: str) -> None:
        if hook is not None:
            hook(phase)


def _portrait_name(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "portrait.png"
    if data.startswith(b"\xff\xd8\xff"):
        return "portrait.jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "portrait.gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "portrait.webp"
    raise ValueError("actor_pack_portrait_invalid")


__all__ = [
    "ActorPackCreationError",
    "ActorPackCreationResult",
    "ActorPackCreationService",
]
