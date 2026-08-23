"""Transactional activation of one still-current Actor Pack review."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

from .importer import (
    ActorPackImportError,
    ActorPackImportReview,
    ActorPackImportService,
)
from .contracts import canonical_json_bytes
from .persona_coordinator import (
    PersonaActorPackCoordinator,
    PersonaActorPackCoordinatorError,
)
from .repository import ActorPackRepository, ActorPackRepositoryError


class ActorPackActivationError(ValueError):
    """One fixed, path-free activation failure."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackActivationResult:
    """Path-free committed actor identity."""

    actor_kind: str
    local_actor_id: str
    portable_uuid: str
    cleanup_pending: bool = False


class ActorPackActivationService:
    """Consume immutable reviews at the existing Character/Persona boundaries."""

    def __init__(
        self,
        db: CharactersRAGDB,
        local_service: LocalCharacterPersonaService,
        repository: ActorPackRepository,
        persona_coordinator: PersonaActorPackCoordinator,
        importer: ActorPackImportService,
    ) -> None:
        if (
            not isinstance(db, CharactersRAGDB)
            or not isinstance(local_service, LocalCharacterPersonaService)
            or not isinstance(repository, ActorPackRepository)
            or not isinstance(persona_coordinator, PersonaActorPackCoordinator)
            or not isinstance(importer, ActorPackImportService)
        ):
            raise ActorPackActivationError("actor_pack_import_activation_invalid")
        self.db = db
        self.local_service = local_service
        self.repository = repository
        self.persona_coordinator = persona_coordinator
        self.importer = importer

    def activate(
        self,
        review: ActorPackImportReview,
        action: str,
        *,
        cancel_requested: Callable[[], bool] = lambda: False,
    ) -> ActorPackActivationResult:
        """Activate one action offered by the exact review."""

        if (
            type(review) is not ActorPackImportReview
            or type(action) is not str
            or action not in review.allowed_actions
        ):
            raise ActorPackActivationError("actor_pack_import_action_invalid")
        self._raise_if_cancelled(cancel_requested)
        try:
            material = self.importer._activation_material(review)
            if review.sections:
                raise ActorPackActivationError(
                    "actor_pack_import_section_activation_failed"
                )
            if review.actor_kind == "character":
                result = self._activate_character(
                    review,
                    action,
                    dict(material.actor_fields),
                    material.portrait,
                    cancel_requested,
                )
            else:
                result = self._activate_persona(
                    review,
                    action,
                    dict(material.actor_fields),
                    cancel_requested,
                )
        except ActorPackActivationError:
            raise
        except ActorPackImportError as exc:
            raise ActorPackActivationError(exc.category) from None
        except PersonaActorPackCoordinatorError as exc:
            category = (
                "actor_pack_import_cancelled"
                if exc.category == "actor_pack_creation_cancelled"
                else "actor_pack_import_activation_failed"
            )
            raise ActorPackActivationError(category) from None
        except (ActorPackRepositoryError, sqlite3.Error, TypeError, ValueError):
            raise ActorPackActivationError(
                "actor_pack_import_activation_failed"
            ) from None
        try:
            self.importer.cleanup_review(review)
        except ActorPackImportError:
            return ActorPackActivationResult(
                result.actor_kind,
                result.local_actor_id,
                result.portable_uuid,
                cleanup_pending=True,
            )
        return result

    def _activate_character(
        self,
        review: ActorPackImportReview,
        action: str,
        fields: dict[str, Any],
        portrait: bytes,
        cancel_requested: Callable[[], bool],
    ) -> ActorPackActivationResult:
        if action == "update_existing":
            if (
                review._matched_local_actor_id is None
                or review._matched_actor_version is None
            ):
                raise ActorPackActivationError("actor_pack_import_review_stale")
            character_id = int(review._matched_local_actor_id)
            fields["image"] = portrait
            with self.db.transaction(immediate=True) as cursor:
                self._raise_if_cancelled(cancel_requested)
                self.importer.revalidate_review(review)
                self.db._update_character_card_in_transaction(
                    cursor,
                    character_id,
                    fields,
                    expected_version=review._matched_actor_version,
                    require_outermost=True,
                )
            return ActorPackActivationResult(
                "character", character_id.__str__(), review.portable_uuid
            )
        assigned = (
            review.portable_uuid
            if action == "create_new"
            else self.repository._new_portable_uuid()
        )
        source_uuid = review.portable_uuid if action == "create_copy" else None
        fields["image"] = portrait
        with self.db.transaction(immediate=True) as cursor:
            self._raise_if_cancelled(cancel_requested)
            self.importer.revalidate_review(review)
            if action == "create_copy":
                fields["name"] = self._copy_character_name(
                    cursor, str(fields["name"])
                )
            character_id = self.db._insert_character_card_in_transaction(
                cursor,
                fields,
                require_outermost=True,
            )
            identity = self.repository._assign_identity_in_transaction(
                "character",
                character_id,
                source_portable_uuid=source_uuid,
                portable_uuid=assigned,
            )
        return ActorPackActivationResult(
            "character", str(character_id), identity.portable_uuid
        )

    def _activate_persona(
        self,
        review: ActorPackImportReview,
        action: str,
        fields: dict[str, Any],
        cancel_requested: Callable[[], bool],
    ) -> ActorPackActivationResult:
        if action == "update_existing":
            if (
                review._matched_local_actor_id is None
                or review._matched_actor_version is None
            ):
                raise ActorPackActivationError("actor_pack_import_review_stale")
            persona_id = review._matched_local_actor_id
            current = dict(self.local_service._find_persona_profile(persona_id))
            current.update(fields)
            current["last_modified"] = self.local_service._now()
            current["version"] = review._matched_actor_version + 1
            new_authority = (
                current["version"],
                hashlib.sha256(canonical_json_bytes(current)).hexdigest(),
            )
            self.importer.revalidate_review(review)
            self._raise_if_cancelled(cancel_requested)
            committed = self.persona_coordinator.create_persona(
                current,
                portable_uuid=review.portable_uuid,
                operation="update",
                cancel_requested=cancel_requested,
                authority_guard=lambda: self._review_is_current(
                    review,
                    alternate_actor_authorities=(new_authority,),
                ),
            )
            return ActorPackActivationResult(
                "persona",
                persona_id,
                committed.identity.portable_uuid,
                cleanup_pending=committed.cleanup_pending,
            )
        self.importer.revalidate_review(review)
        self._raise_if_cancelled(cancel_requested)
        persona_id = f"local-persona-{uuid4().hex}"
        now = self.local_service._now()
        profile = {
            **fields,
            "id": persona_id,
            "created_at": now,
            "last_modified": now,
            "version": 1,
            "deleted": False,
        }
        portable_uuid = (
            review.portable_uuid
            if action == "create_new"
            else self.repository._new_portable_uuid()
        )
        committed = self.persona_coordinator.create_persona(
            profile,
            portable_uuid=portable_uuid,
            operation="create" if action == "create_new" else "copy",
            source_portable_uuid=(
                review.portable_uuid if action == "create_copy" else None
            ),
            cancel_requested=cancel_requested,
            authority_guard=lambda: self._review_is_current(review),
        )
        return ActorPackActivationResult(
            "persona",
            persona_id,
            committed.identity.portable_uuid,
            cleanup_pending=committed.cleanup_pending,
        )

    def _review_is_current(
        self,
        review: ActorPackImportReview,
        *,
        alternate_actor_authorities: tuple[tuple[int, str], ...] = (),
    ) -> bool:
        try:
            self.importer.revalidate_review(
                review,
                alternate_actor_authorities=alternate_actor_authorities,
            )
        except ActorPackImportError:
            return False
        return True

    @staticmethod
    def _copy_character_name(cursor: sqlite3.Cursor, name: str) -> str:
        candidate = f"{name} (Copy)"
        suffix = 2
        while cursor.execute(
            "SELECT 1 FROM character_cards WHERE name = ? AND deleted = 0",
            (candidate,),
        ).fetchone():
            candidate = f"{name} (Copy {suffix})"
            suffix += 1
        return candidate

    @staticmethod
    def _raise_if_cancelled(checker: Callable[[], bool]) -> None:
        try:
            cancelled = checker()
        except Exception:
            cancelled = True
        if cancelled is True:
            raise ActorPackActivationError("actor_pack_import_cancelled")


__all__ = [
    "ActorPackActivationError",
    "ActorPackActivationResult",
    "ActorPackActivationService",
]
