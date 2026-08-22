"""Consistent local actor snapshots for deterministic Actor Pack export."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tldw_chatbook.Character_Chat.local_character_persona_service import (
    LocalCharacterPersonaService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

from .contracts import (
    ActorPackValidationError,
    canonicalize_actor_payload,
    validate_actor_portrait,
)
from .repository import (
    ActorPackRepository,
    ActorPackRepositoryError,
    PortableActorIdentity,
)


_VALIDATION_UUID = "123e4567-e89b-42d3-a456-426614174000"


class ActorPackExportError(ValueError):
    """One stable, path-free Actor Pack export failure."""

    def __init__(self, category: str, *, user_message: str | None = None) -> None:
        self.category = category
        self.user_message = user_message
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class ActorPackExportSnapshot:
    """Immutable actor/portrait/portable-identity export authority."""

    actor_kind: str
    actor_revision: int
    portable_uuid: str
    identity_version: int
    portrait_name: str
    portrait_sha256: str
    local_actor_id: str = field(repr=False)
    actor_payload: bytes = field(repr=False)
    portrait_bytes: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class ActorPackExportResult:
    """Path-free result reserved for archive/publication phases."""

    archive_sha256: str
    committed: bool
    durability: str


class ActorPackExportService:
    """Capture one exact local actor before archive materialization."""

    def __init__(
        self,
        database: CharactersRAGDB,
        local_service: LocalCharacterPersonaService,
        repository: ActorPackRepository,
    ) -> None:
        self.database = database
        self.local_service = local_service
        self.repository = repository

    def capture_snapshot(
        self,
        actor_kind: str,
        local_actor_id: str,
        *,
        source: str,
        phase_hook: Callable[[str], None] | None = None,
    ) -> ActorPackExportSnapshot:
        """Validate, assign portable identity, and freeze a reread snapshot."""

        if type(source) is not str or source != "local":
            raise ActorPackExportError(
                "actor_pack_source_not_local",
                user_message="Save a local copy first",
            )
        actor_id = _actor_id(actor_kind, local_actor_id)
        initial, initial_portrait = self._read_candidate(actor_kind, actor_id)
        self._validate_candidate(actor_kind, initial, initial_portrait)
        try:
            identity = self.repository.assign_identity(
                actor_kind, actor_id, source=source
            )
        except ActorPackRepositoryError as exc:
            raise ActorPackExportError("actor_pack_export_failed") from exc
        if phase_hook is not None:
            phase_hook("identity_assigned")
        current, current_portrait = self._read_candidate(actor_kind, actor_id)
        self._validate_candidate(actor_kind, current, current_portrait)
        if _candidate_digest(
            actor_kind, initial, initial_portrait
        ) != _candidate_digest(actor_kind, current, current_portrait):
            raise ActorPackExportError("actor_pack_export_authority_changed")
        return _snapshot(actor_kind, actor_id, current, current_portrait, identity)

    def _read_candidate(
        self, actor_kind: str, actor_id: int | str
    ) -> tuple[dict[str, Any], bytes]:
        try:
            if actor_kind == "character":
                actor = dict(self.local_service.get_character(int(actor_id)))
            else:
                actor = dict(self.local_service.get_persona_profile(str(actor_id)))
        except (CharactersRAGDBError, KeyError, TypeError, ValueError):
            raise ActorPackExportError("actor_pack_actor_unavailable") from None
        if actor_kind == "character":
            portrait = actor.get("image")
        else:
            character_id = actor.get("character_card_id")
            linked = (
                self.database.get_character_card_by_id(character_id)
                if type(character_id) is int and character_id > 0
                else None
            )
            portrait = None if linked is None else linked.get("image")
        if type(portrait) is not bytes:
            raise ActorPackExportError("actor_pack_portrait_invalid")
        return actor, portrait

    @staticmethod
    def _validate_candidate(
        actor_kind: str, actor: dict[str, Any], portrait: bytes
    ) -> None:
        name = _portrait_name(portrait)
        try:
            validate_actor_portrait(name, portrait)
            canonicalize_actor_payload(actor_kind, _VALIDATION_UUID, actor)
        except ActorPackValidationError as exc:
            category = (
                "actor_pack_portrait_invalid"
                if exc.category == "actor_pack_portrait_invalid"
                else "actor_pack_actor_invalid"
            )
            raise ActorPackExportError(category) from None


def _actor_id(actor_kind: object, value: object) -> int | str:
    if actor_kind == "character":
        if type(value) is not str or not value.isdigit() or int(value) < 1:
            raise ActorPackExportError("actor_pack_actor_unavailable")
        return int(value)
    if actor_kind == "persona":
        if type(value) is not str or not value:
            raise ActorPackExportError("actor_pack_actor_unavailable")
        return value
    raise ActorPackExportError("actor_pack_actor_unavailable")


def _portrait_name(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "portrait.png"
    if data.startswith(b"\xff\xd8\xff"):
        return "portrait.jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "portrait.gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "portrait.webp"
    return "portrait.invalid"


def _candidate_digest(actor_kind: str, actor: dict[str, Any], portrait: bytes) -> str:
    payload = canonicalize_actor_payload(actor_kind, _VALIDATION_UUID, actor)
    return hashlib.sha256(payload + hashlib.sha256(portrait).digest()).hexdigest()


def _snapshot(
    actor_kind: str,
    actor_id: int | str,
    actor: dict[str, Any],
    portrait: bytes,
    identity: PortableActorIdentity,
) -> ActorPackExportSnapshot:
    revision = actor.get("version", 1)
    if type(revision) is not int or revision < 1:
        raise ActorPackExportError("actor_pack_actor_invalid")
    return ActorPackExportSnapshot(
        actor_kind=actor_kind,
        actor_revision=revision,
        portable_uuid=identity.portable_uuid,
        identity_version=identity.version,
        portrait_name=_portrait_name(portrait),
        portrait_sha256=hashlib.sha256(portrait).hexdigest(),
        local_actor_id=str(actor_id),
        actor_payload=canonicalize_actor_payload(
            actor_kind, identity.portable_uuid, actor
        ),
        portrait_bytes=portrait,
    )
