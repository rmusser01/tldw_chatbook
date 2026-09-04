"""Authority-safe contracts for character-conversation navigation."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.character_conversation_search import (
        CharacterConversationSearchRepository,
    )

_IDENTITY_VERSION = 1
_IDENTITY_TEXT_MAX_BYTES = 256
_MAX_SQLITE_INTEGER = 2**63 - 1


def _validated_identity_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    if value != value.strip() or not value:
        raise ValueError(f"{field_name} must be nonblank canonical text")
    if len(value.encode("utf-8")) > _IDENTITY_TEXT_MAX_BYTES:
        raise ValueError(f"{field_name} must not exceed 256 UTF-8 bytes")
    return value


class UnavailableCharacterReason(StrEnum):
    """Why an unresolved character-backed conversation cannot be activated."""

    MISSING_CARD = "missing_card"
    DELETED_CARD = "deleted_card"
    MISSING_CHARACTER_AUTHORITY_LINK = "missing_character_authority_link"
    AMBIGUOUS_LEGACY_LINK = "ambiguous_legacy_link"


@dataclass(frozen=True)
class ResolvedLocalCharacterKey:
    """A character identifier scoped by one local Data Profile authority."""

    data_authority_id: str
    character_id: int

    def __post_init__(self) -> None:
        _validated_identity_text(self.data_authority_id, "data_authority_id")
        if isinstance(self.character_id, bool) or not isinstance(self.character_id, int):
            raise TypeError("character_id must be an integer")
        if not 1 <= self.character_id <= _MAX_SQLITE_INTEGER:
            raise ValueError("character_id is outside SQLite's positive integer range")


@dataclass(frozen=True)
class UnresolvedConversationKey:
    """A non-activatable conversation identity with unresolved character provenance."""

    data_authority_id: str
    conversation_id: str

    def __post_init__(self) -> None:
        _validated_identity_text(self.data_authority_id, "data_authority_id")
        _validated_identity_text(self.conversation_id, "conversation_id")


@dataclass(frozen=True)
class LocalCharacterConversationTarget:
    """An exact activatable local character conversation."""

    character: ResolvedLocalCharacterKey
    conversation_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.character, ResolvedLocalCharacterKey):
            raise TypeError("character must be a ResolvedLocalCharacterKey")
        _validated_identity_text(self.conversation_id, "conversation_id")


CharacterConversationKey: TypeAlias = (
    ResolvedLocalCharacterKey | UnresolvedConversationKey
)


def serialize_character_conversation_key(
    key: CharacterConversationKey,
) -> dict[str, object]:
    """Serialize a character-conversation key as a closed versioned union."""

    if isinstance(key, ResolvedLocalCharacterKey):
        return {
            "version": _IDENTITY_VERSION,
            "tag": "resolved_local_character",
            "data_authority_id": key.data_authority_id,
            "character_id": key.character_id,
        }
    if isinstance(key, UnresolvedConversationKey):
        return {
            "version": _IDENTITY_VERSION,
            "tag": "unresolved_conversation",
            "data_authority_id": key.data_authority_id,
            "conversation_id": key.conversation_id,
        }
    raise TypeError("unsupported character-conversation key")


def deserialize_character_conversation_key(
    payload: Mapping[str, Any],
) -> CharacterConversationKey:
    """Deserialize the exact supported key version and tag."""

    if payload.get("version") != _IDENTITY_VERSION:
        raise ValueError("unsupported character-conversation identity version")
    tag = payload.get("tag")
    if tag == "resolved_local_character":
        return ResolvedLocalCharacterKey(
            payload.get("data_authority_id"),  # type: ignore[arg-type]
            payload.get("character_id"),  # type: ignore[arg-type]
        )
    if tag == "unresolved_conversation":
        return UnresolvedConversationKey(
            payload.get("data_authority_id"),  # type: ignore[arg-type]
            payload.get("conversation_id"),  # type: ignore[arg-type]
        )
    raise ValueError("unsupported character-conversation identity tag")


def _row_key(
    key: CharacterConversationKey,
    conversation_id: str | None = None,
) -> str:
    payload = serialize_character_conversation_key(key)
    if conversation_id is not None:
        payload["conversation_id"] = _validated_identity_text(
            conversation_id, "conversation_id"
        )
    return f"{payload['tag']}:" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


@dataclass(frozen=True)
class CharacterConversationCursor:
    last_modified: str
    conversation_id: str


@dataclass(frozen=True)
class CharacterConversationRow:
    row_key: str
    target: LocalCharacterConversationTarget | None
    unresolved: UnresolvedConversationKey | None
    unavailable_reason: UnavailableCharacterReason | None
    character_label: str
    title: str
    last_modified: str
    is_current: bool
    selected_excerpt: str

    def __post_init__(self) -> None:
        if (self.target is None) == (self.unresolved is None):
            raise ValueError("row must be exactly one of resolved or unresolved")
        if self.target is not None:
            if not isinstance(self.target, LocalCharacterConversationTarget):
                raise TypeError("target must be a LocalCharacterConversationTarget")
            if self.unavailable_reason is not None:
                raise ValueError("resolved rows cannot have an unavailable reason")
        elif not isinstance(self.unresolved, UnresolvedConversationKey):
            raise TypeError("unresolved must be an UnresolvedConversationKey")
        elif not isinstance(self.unavailable_reason, UnavailableCharacterReason):
            raise TypeError("unresolved rows require an unavailable reason")

    @classmethod
    def resolved(
        cls,
        target: LocalCharacterConversationTarget,
        *,
        character_label: str,
        title: str,
        last_modified: str,
        is_current: bool = False,
        selected_excerpt: str = "",
    ) -> CharacterConversationRow:
        """Build a resolved row with its collision-safe stable key."""

        return cls(
            row_key=_row_key(target.character, target.conversation_id),
            target=target,
            unresolved=None,
            unavailable_reason=None,
            character_label=character_label,
            title=title,
            last_modified=last_modified,
            is_current=is_current,
            selected_excerpt=selected_excerpt,
        )

    @classmethod
    def unavailable(
        cls,
        key: UnresolvedConversationKey,
        *,
        reason: UnavailableCharacterReason,
        character_label: str,
        title: str,
        last_modified: str,
        is_current: bool = False,
        selected_excerpt: str = "",
    ) -> CharacterConversationRow:
        """Build an unavailable row while keeping diagnosis outside identity."""

        return cls(
            row_key=_row_key(key),
            target=None,
            unresolved=key,
            unavailable_reason=reason,
            character_label=character_label,
            title=title,
            last_modified=last_modified,
            is_current=is_current,
            selected_excerpt=selected_excerpt,
        )


@dataclass(frozen=True)
class CharacterConversationGroup:
    key: CharacterConversationKey
    character_label: str
    rows: tuple[CharacterConversationRow, ...]
    total: int
    is_current: bool


@dataclass(frozen=True)
class EligibleConversationDocument:
    target: LocalCharacterConversationTarget
    title: str
    body: str
    source_revision: int
    eligibility_digest: str


@dataclass(frozen=True)
class CharacterConversationPage:
    rows: tuple[CharacterConversationRow, ...]
    total: int
    next_cursor: CharacterConversationCursor | None
    data_revision: int


class CharacterKeywordIndexStatus(StrEnum):
    ABSENT = "absent"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class CharacterRepairCandidate:
    key: ResolvedLocalCharacterKey
    display_name: str
    version: int


@dataclass(frozen=True)
class CharacterRepairRequest:
    unresolved: UnresolvedConversationKey
    replacement: ResolvedLocalCharacterKey
    expected_conversation_version: int


class CharacterRepairResult(StrEnum):
    APPLIED = "applied"
    STALE_VERSION = "stale_version"
    NOT_FOUND = "not_found"
    INVALID_CANDIDATE = "invalid_candidate"


class CharacterConversationNavigationService:
    """Application façade for authority-safe character-conversation navigation."""

    def __init__(
        self,
        database: CharactersRAGDB,
        *,
        current_character: ResolvedLocalCharacterKey | None = None,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        from tldw_chatbook.DB.character_conversation_search import (
            CharacterConversationSearchRepository,
        )

        self._repository: CharacterConversationSearchRepository = (
            CharacterConversationSearchRepository(
                database,
                current_character=current_character,
                progress_callback=progress_callback,
            )
        )

    def recent_groups(
        self, *, group_limit: int = 4, row_limit: int = 5
    ) -> tuple[CharacterConversationGroup, ...]:
        """Return bounded section-first recent groups."""

        return self._repository.recent_groups(
            group_limit=group_limit, row_limit=row_limit
        )

    def keyword_search(
        self, query: str, *, offset: int = 0, limit: int = 50
    ) -> CharacterConversationPage:
        """Return a bounded page from the ready Keyword generation."""

        return self._repository.keyword_search(query, offset=offset, limit=limit)

    def page_for_character(
        self,
        key: ResolvedLocalCharacterKey,
        *,
        cursor: CharacterConversationCursor | None = None,
        limit: int = 20,
    ) -> CharacterConversationPage:
        """Return a stable keyset page for one exact character."""

        return self._repository.page_for_character(key, cursor=cursor, limit=limit)

    def repair_candidates(
        self, key: UnresolvedConversationKey
    ) -> tuple[CharacterRepairCandidate, ...]:
        """Return live same-authority repair choices."""

        return self._repository.repair_candidates(key)

    def repair(self, request: CharacterRepairRequest) -> CharacterRepairResult:
        """Compare-and-set one unresolved conversation's character identity."""

        return self._repository.repair(request)

    def ensure_keyword_index(self) -> CharacterKeywordIndexStatus:
        """Build or report the local Keyword generation when explicitly called."""

        return self._repository.ensure_keyword_index()

    def keyword_index_status(self) -> CharacterKeywordIndexStatus:
        """Return the current Keyword generation status."""

        return self._repository.keyword_index_status()
