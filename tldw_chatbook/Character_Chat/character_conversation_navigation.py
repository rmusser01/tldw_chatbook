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
    """A character identifier scoped by one local Data Profile authority.

    Attributes:
        data_authority_id: Exact nonblank canonical text, at most 256 UTF-8 bytes.
        character_id: Positive SQLite integer ID (1..2**63-1), never a bool.

    Raises:
        TypeError: An identity field has the wrong type.
        ValueError: Text is blank/noncanonical/overlong or the ID is out of range.
    """

    data_authority_id: str
    character_id: int

    def __post_init__(self) -> None:
        _validated_identity_text(self.data_authority_id, "data_authority_id")
        if isinstance(self.character_id, bool) or not isinstance(
            self.character_id, int
        ):
            raise TypeError("character_id must be an integer")
        if not 1 <= self.character_id <= _MAX_SQLITE_INTEGER:
            raise ValueError("character_id is outside SQLite's positive integer range")


@dataclass(frozen=True)
class UnresolvedConversationKey:
    """A non-activatable conversation identity with unresolved provenance.

    Attributes:
        data_authority_id: Exact local Data Profile authority.
        conversation_id: Exact conversation ID, not historical character evidence.

    Raises:
        TypeError: Either field is not text.
        ValueError: Either field is blank, not stripped, or over 256 UTF-8 bytes.
    """

    data_authority_id: str
    conversation_id: str

    def __post_init__(self) -> None:
        _validated_identity_text(self.data_authority_id, "data_authority_id")
        _validated_identity_text(self.conversation_id, "conversation_id")


@dataclass(frozen=True)
class LocalCharacterConversationTarget:
    """An exact activatable local character conversation.

    Attributes:
        character: Validated resolved authority and card identity.
        conversation_id: Canonical nonblank ID, at most 256 UTF-8 bytes.

    Raises:
        TypeError: Character is not a resolved key or conversation ID is not text.
        ValueError: Conversation ID is blank, noncanonical, or overlong.
    """

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
    """Serialize a character-conversation key as a closed versioned union.

    Args:
        key: Validated resolved or unresolved identity.

    Returns:
        A JSON-compatible mapping containing only the supported version/tag fields.

    Raises:
        TypeError: Key is not a supported identity class.
    """

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
    """Deserialize the exact supported key version and tag.

    Args:
        payload: Mapping containing exactly the fields for the supported tag.

    Returns:
        A validated resolved or unresolved identity, preserving case exactly.

    Raises:
        TypeError: An identity field has the wrong type.
        ValueError: Version, tag, fields, canonical text, or integer bounds fail.
    """

    if payload.get("version") != _IDENTITY_VERSION:
        raise ValueError("unsupported character-conversation identity version")
    tag = payload.get("tag")
    if tag == "resolved_local_character":
        if set(payload) != {
            "version",
            "tag",
            "data_authority_id",
            "character_id",
        }:
            raise ValueError("invalid character-conversation identity fields")
        return ResolvedLocalCharacterKey(
            payload.get("data_authority_id"),  # type: ignore[arg-type]
            payload.get("character_id"),  # type: ignore[arg-type]
        )
    if tag == "unresolved_conversation":
        if set(payload) != {
            "version",
            "tag",
            "data_authority_id",
            "conversation_id",
        }:
            raise ValueError("invalid character-conversation identity fields")
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
    """Complete descending date-order boundary returned by character browsing.

    Attributes:
        last_modified: Source modification timestamp, compared as SQLite text.
        created_at: Source creation timestamp, the second tie-breaker.
        conversation_id: Exact final ID tie-breaker. Reuse returned cursors for
            the same character; restart after mutation to see changed rows.
    """

    last_modified: str
    created_at: str
    conversation_id: str


@dataclass(frozen=True)
class CharacterConversationRow:
    """Presentation data with exactly one resolved target or unresolved key.

    Attributes:
        row_key: Collision-safe stable key; labels never establish identity.
        target: Activatable exact target, or None for an unavailable row.
        unresolved: Non-activatable identity, or None for a resolved row.
        unavailable_reason: Required diagnosis only for an unresolved row.
        character_label: Display-only current or historical label.
        title: Display-only conversation title.
        last_modified: Source modification timestamp for ordering.
        created_at: Source creation timestamp for date-order ties.
        is_current: Whether the row belongs to the captured current character.
        selected_excerpt: Display excerpt, not an activation or authority token.

    Raises:
        TypeError: Target, unresolved key, or reason has the wrong type.
        ValueError: Resolved/unresolved alternatives or reason are inconsistent.
    """

    row_key: str
    target: LocalCharacterConversationTarget | None
    unresolved: UnresolvedConversationKey | None
    unavailable_reason: UnavailableCharacterReason | None
    character_label: str
    title: str
    last_modified: str
    created_at: str
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
        created_at: str,
        is_current: bool = False,
        selected_excerpt: str = "",
    ) -> CharacterConversationRow:
        """Build a resolved row with its collision-safe stable key.

        Args:
            target: Validated exact local character conversation.
            character_label: Display-only card name.
            title: Display-only conversation title.
            last_modified: Source modification timestamp.
            created_at: Source creation timestamp for complete date ordering.
            is_current: Whether this is the current character.
            selected_excerpt: Optional display text from validated content.

        Returns:
            A resolved row with no unresolved key or unavailable reason.
        """

        return cls(
            row_key=_row_key(target.character, target.conversation_id),
            target=target,
            unresolved=None,
            unavailable_reason=None,
            character_label=character_label,
            title=title,
            last_modified=last_modified,
            created_at=created_at,
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
        created_at: str,
        is_current: bool = False,
        selected_excerpt: str = "",
    ) -> CharacterConversationRow:
        """Build an unavailable row while keeping diagnosis outside identity.

        Args:
            key: Exact unresolved local conversation identity.
            reason: Typed unavailable diagnosis, never part of identity.
            character_label: Historical display label only.
            title: Display-only conversation title.
            last_modified: Source modification timestamp.
            created_at: Source creation timestamp for complete date ordering.
            is_current: Whether presentation marks this row current.
            selected_excerpt: Optional display excerpt.

        Returns:
            A non-activatable row with no resolved target.

        Raises:
            TypeError: Reason is not an UnavailableCharacterReason.
        """

        return cls(
            row_key=_row_key(key),
            target=None,
            unresolved=key,
            unavailable_reason=reason,
            character_label=character_label,
            title=title,
            last_modified=last_modified,
            created_at=created_at,
            is_current=is_current,
            selected_excerpt=selected_excerpt,
        )


@dataclass(frozen=True)
class CharacterConversationGroup:
    """Bounded recent section with an exact unbounded total.

    Attributes:
        key: Resolved character key or an unresolved section anchor.
        character_label: Display-only section label.
        rows: Bounded newest-first presentation rows.
        total: All matching conversations, not just the visible rows.
        is_current: Whether the group represents the current character.
    """

    key: CharacterConversationKey
    character_label: str
    rows: tuple[CharacterConversationRow, ...]
    total: int
    is_current: bool


@dataclass(frozen=True)
class EligibleConversationDocument:
    """Canonical selected-branch content from one authoritative SQLite snapshot.

    Attributes:
        target: Exact eligible local character conversation.
        title: Searchable conversation title.
        body: Joined eligible visible user/assistant messages only.
        source_revision: Live revision read in the projection's connection.
        eligibility_digest: Digest of the policy, title and selected messages.
    """

    target: LocalCharacterConversationTarget
    title: str
    body: str
    source_revision: int
    eligibility_digest: str


@dataclass(frozen=True)
class CharacterKeywordSnapshot:
    """Captured identity and completion time of the queried ready corpus.

    Attributes:
        generation_id: Exact ready generation queried.
        policy_version: Eligibility policy under which the corpus was built.
        source_revision: Captured corpus revision, possibly older than the live
            page data_revision while an unchanged eligible snapshot is retained.
        completed_at: Valid completion timestamp of that generation.
    """

    generation_id: str
    policy_version: int
    source_revision: int
    completed_at: str


@dataclass(frozen=True)
class CharacterConversationPage:
    """A bounded read result and its separate live and corpus revision fences.

    Attributes:
        rows: Validated presentation rows.
        total: Matching eligible total before paging.
        next_cursor: Next character date-keyset boundary, otherwise None.
        data_revision: Live source revision for preview/activation revalidation.
        keyword_status: Queried Keyword availability, or None for date browsing.
        keyword_snapshot: Captured ready corpus identity; never substitute its
            older source_revision for the page's live data_revision.
    """

    rows: tuple[CharacterConversationRow, ...]
    total: int
    next_cursor: CharacterConversationCursor | None
    data_revision: int
    keyword_status: CharacterKeywordIndexStatus | None = None
    keyword_snapshot: CharacterKeywordSnapshot | None = None


class CharacterKeywordIndexStatus(StrEnum):
    """Typed availability: absent, active building, ready, or failed."""

    ABSENT = "absent"
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class CharacterRepairCandidate:
    """One explicitly selectable live card, never selected by its name.

    Attributes:
        key: Exact same-authority resolved identity.
        display_name: Presentation label only.
        version: Card version captured when this choice was read.
    """

    key: ResolvedLocalCharacterKey
    display_name: str
    version: int


@dataclass(frozen=True)
class CharacterRepairPage:
    """One bounded repair-choice snapshot, never an implicit complete list.

    Attributes:
        candidates: At most the requested limit of same-authority live cards.
        total: All live choices at this read snapshot, before offset/limit.
        next_offset: Offset for another page, or None when exhausted. Restart
            enumeration after source mutations; pages do not retain a snapshot.
    """

    candidates: tuple[CharacterRepairCandidate, ...]
    total: int
    next_offset: int | None


@dataclass(frozen=True)
class CharacterRepairRequest:
    """An explicitly confirmed repair request; the repository revalidates it.

    Attributes:
        unresolved: Exact conversation requiring repair.
        replacement: Explicitly chosen same-authority live card identity.
        expected_conversation_version: Captured version for compare-and-set.
    """

    unresolved: UnresolvedConversationKey
    replacement: ResolvedLocalCharacterKey
    expected_conversation_version: int


class CharacterRepairResult(StrEnum):
    """Repair outcome: applied, stale version, missing chat, or invalid choice."""

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
        """Bind navigation to one local Data Profile without starting work.

        Args:
            database: Authoritative local conversation database.
            current_character: Optional current card; foreign authority is ignored.
            progress_callback: Optional processed-count callback for explicitly
                requested Keyword builds. Call those builds from an owning worker.
        """

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
        """Return section-first recent groups, including the current live card.

        Args:
            group_limit: Integer section bound 1..4 (booleans rejected).
            row_limit: Integer rows-per-section bound 1..5 (booleans rejected).

        Returns:
            Bounded groups with exact totals and complete descending date order.
            An unavailable section is reserved when space permits.

        Raises:
            ValueError: Either bound is outside its supported range."""

        return self._repository.recent_groups(
            group_limit=group_limit, row_limit=row_limit
        )

    def keyword_search(
        self,
        query: str,
        *,
        character: ResolvedLocalCharacterKey | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> CharacterConversationPage:
        """Search a retained ready corpus and revalidate against live source state.

        Args:
            query: Literal phrase text; blank or nontext input returns no rows.
            character: Optional exact character filter; foreign keys return empty.
            offset: Nonnegative integer result offset, not a date cursor.
            limit: Integer result bound 1..50 (booleans rejected).

        Returns:
            Relevance-ordered rows, exact eligible total and live data_revision.
            keyword_snapshot identifies the possibly older retained ready corpus.
            Changed/ineligible candidates are excluded before paging; a concurrent
            source change retries once, then fails closed with an absent page.

        Raises:
            ValueError: Offset or limit is invalid."""

        return self._repository.keyword_search(
            query, character=character, offset=offset, limit=limit
        )

    def page_for_character(
        self,
        key: ResolvedLocalCharacterKey,
        *,
        cursor: CharacterConversationCursor | None = None,
        limit: int = 20,
    ) -> CharacterConversationPage:
        """Return a complete descending date-keyset page for one exact character.

        Args:
            key: Resolved local character identity; foreign keys return empty.
            cursor: Previous page's complete (last_modified, created_at, ID)
                boundary for this character, or None to start/restart.
            limit: Integer result bound 1..20 (booleans rejected).

        Returns:
            Rows, total, next cursor and live revision from one SQLite snapshot.
            Restart after mutations to discover rows moved before the boundary.

        Raises:
            ValueError: Limit is invalid."""

        return self._repository.page_for_character(key, cursor=cursor, limit=limit)

    def unavailable_page(
        self,
        *,
        offset: int = 0,
        limit: int = 20,
        query: str = "",
    ) -> CharacterConversationPage:
        """Return a bounded metadata-filtered page of unresolved local chats.

        Args:
            offset: Nonnegative integer row offset (booleans rejected).
            limit: Integer result bound 1..50 (booleans rejected).
            query: Text of at most 200 characters before stripping whitespace.
                Empty/whitespace text selects ordinary date browsing; LIKE
                metacharacters remain literal, never SQL or FTS instructions.

        Returns:
            Newest-first unresolved rows, exact filtered total and live revision.
            Continue with offset plus row count while below total; no keyset
            cursor is returned. Restart after source mutations.

        Raises:
            TypeError: Query is not text; objects are never coerced.
            ValueError: Query is overlong, or offset/limit is invalid."""

        return self._repository.unavailable_page(
            offset=offset,
            limit=limit,
            query=query,
        )

    def repair_candidates(
        self, key: UnresolvedConversationKey, *, offset: int = 0, limit: int = 20
    ) -> CharacterRepairPage:
        """Return a bounded page of live same-authority repair choices.

        Args:
            key: Unresolved conversation identity, revalidated on each read.
            offset: Nonnegative offset, initially zero, then page.next_offset.
            limit: Integer page size 1..50; defaults to 20.

        Returns:
            Candidates, exact total and explicit continuation. SQLite NOCASE
            name then ID gives deterministic order. Restart after mutations.
            Foreign or no-longer-unresolved keys return an empty page.

        Raises:
            ValueError: Offset or limit is invalid, including booleans.
        """

        return self._repository.repair_candidates(key, offset=offset, limit=limit)

    def refresh_unresolved_evidence(
        self, key: UnresolvedConversationKey
    ) -> tuple[int, str] | None:
        """Read current unresolved evidence and its compare-and-set version.

        Args:
            key: Exact unresolved identity from the selected local authority.

        Returns:
            (conversation version, historical identity display text), or None
            if foreign, missing, deleted, nonlocal, or now resolved. Historical
            text is evidence only and must never select a replacement card."""

        return self._repository.refresh_unresolved_evidence(key)

    def validated_preview_messages(
        self,
        target: LocalCharacterConversationTarget,
        *,
        data_revision: int,
        limit: int = 200,
    ) -> tuple[dict[str, Any], ...] | None:
        """Read a transcript only while its exact live identity remains valid.

        Args:
            target: Exact resolved local character conversation.
            data_revision: Live page revision, not a retained corpus revision.
            limit: Integer message bound 1..200 (booleans rejected).

        Returns:
            Oldest-first nondeleted message dictionaries, or None if revision,
            authority, card or conversation eligibility changed. This is a
            transcript preview, not the selected-branch Keyword document.

        Raises:
            ValueError: Limit is invalid."""

        return self._repository.validated_preview_messages(
            target, data_revision=data_revision, limit=limit
        )

    def repair(self, request: CharacterRepairRequest) -> CharacterRepairResult:
        """Compare-and-set one exact unresolved local conversation identity.

        Args:
            request: Explicitly confirmed replacement and conversation version.

        Returns:
            APPLIED after source update and derived-document invalidation;
            STALE_VERSION, NOT_FOUND or INVALID_CANDIDATE without a repair when
            validation fails. A resolved conversation cannot be repaired again."""

        return self._repository.repair(request)

    def ensure_keyword_index(self) -> CharacterKeywordIndexStatus:
        """Explicitly activate and build or maintain the local Keyword corpus.

        Returns:
            READY after complete fenced promotion/maintenance, BUILDING when
            another lease owns a build, or FAILED on a handled build/maintenance
            failure. Prior ready snapshots survive failed replacement builds.
            Construction alone never starts work; call from an owning worker."""

        return self._repository.ensure_keyword_index()

    def keyword_index_status(self) -> CharacterKeywordIndexStatus:
        """Read maintenance status for the current source revision and policy.

        Returns:
            Current status without starting work. ABSENT/FAILED can coexist with
            an older ready snapshot that keyword_search can still safely query."""

        return self._repository.keyword_index_status()

    def reconcile_keyword_index(self) -> CharacterKeywordIndexStatus:
        """Reconcile an already activated ready corpus against authoritative SQLite.

        Returns:
            ABSENT when dormant or without a compatible ready generation;
            otherwise the fenced reconciliation result. This never activates a
            dormant index or publishes partially processed content."""

        return self._repository.reconcile_keyword_index()
