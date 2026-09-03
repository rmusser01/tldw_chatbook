"""Pure result contracts for the Console session switcher (Ctrl+K)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from enum import Enum
from typing import Any, Iterable, Literal, Sequence
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    ReverseKey,
    console_conversation_status_detail,
    format_console_relative_age,
)

CONSOLE_SWITCHER_RESULT_LIMIT = 20
CONSOLE_SWITCHER_PAGE_LIMIT = 50


class SwitcherMode(str, Enum):
    """The two bounded views owned by the Ctrl+K switcher."""

    ACTIVE = "active"
    HISTORY = "history"


class ActivityGroup(str, Enum):
    """Ordered Active groups; enum values are persistence-free UI identity."""

    WAITING_FOR_YOU = "waiting_for_you"
    WORKING = "working"
    NEW_RESULTS = "new_results"
    CURRENT = "current"
    OTHER_OPEN = "other_open"


class SwitcherTargetKind(str, Enum):
    """Explicit local destinations supported by Phase 1."""

    NATIVE_SESSION = "console_native_session"
    PERSISTED_CONVERSATION = "console_persisted_conversation"


class CapturedReceipt(BaseModel):
    """Exact immutable receipt evidence captured by one switcher result."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    activity_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r".*\S.*",
    )
    status: Literal["done", "failed", "stuck", "stopped", "cancelled"]


@dataclass(frozen=True)
class ConsoleSwitcherTarget:
    """One explicit activation destination with frozen receipt evidence."""

    kind: SwitcherTargetKind
    profile_authority: str
    authority_token: str
    session_id: str | None
    conversation_id: str | None
    scope_type: str | None
    workspace_id: str | None
    receipts: tuple[CapturedReceipt, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_authority or not self.authority_token:
            raise ValueError("Switcher target authority is required.")
        if self.kind is SwitcherTargetKind.NATIVE_SESSION and not self.session_id:
            raise ValueError("Native switcher targets require session_id.")
        if (
            self.kind is SwitcherTargetKind.PERSISTED_CONVERSATION
            and not self.conversation_id
        ):
            raise ValueError("Persisted switcher targets require conversation_id.")


@dataclass(frozen=True)
class ConsoleSwitcherActivitySignal:
    """Content-free controller contribution to an Active subject."""

    source_key: str
    state: str
    session_id: str | None = None
    conversation_id: str | None = None
    occurred_at: str | datetime | None = None


@dataclass(frozen=True)
class ConsoleSwitcherHistoryPage:
    """One bounded persisted-conversation History page."""

    entries: tuple["ConsoleSwitcherEntry", ...]
    offset: int
    limit: int
    total: int
    error: str = ""

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.entries) < self.total


@dataclass(frozen=True)
class ConsoleSwitcherHistoryQuery:
    """Storage-facing plan for one semantic History query."""

    text_query: str
    workspace_terms: tuple[str, ...] = ()
    can_match: bool = True


@dataclass(frozen=True)
class ConsoleSwitcherEntry:
    """One selectable result in the Console session switcher."""

    row_key: str
    title: str
    subtitle: str
    native_session_id: str | None
    conversation_id: str | None
    scope_type: str
    workspace_id: str | None
    is_active: bool
    section: str = "open"
    state_label: str = ""
    openable: bool = True
    target: ConsoleSwitcherTarget | None = None
    group: ActivityGroup | None = None
    activity_state: str = ""
    latest_at: datetime | None = None
    starred: bool = False
    multiplicity: int = 0
    workspace_label: str = ""
    lifecycle: str = ""

    @property
    def stable_result_key(self) -> str:
        """Canonical immutable identity used by widget/focus maps."""
        return self.row_key


@dataclass(frozen=True)
class UnavailableSessionNotice:
    """Receipt-keyed action for a vanished session or saved conversation."""

    stable_result_key: str
    profile_authority: str
    authority_token: str
    group: ActivityGroup
    latest_at: datetime | None
    receipts: tuple[CapturedReceipt, ...]
    primary_status: str
    session_id: str | None = None
    conversation_id: str | None = None
    all_statuses: tuple[str, ...] = ()
    title: str = "Session unavailable"

    def __post_init__(self) -> None:
        if (
            not self.profile_authority
            or not self.authority_token
            or not self.receipts
            or bool(self.session_id) == bool(self.conversation_id)
        ):
            raise ValueError(
                "Unavailable activity notices require authority, destination, "
                "and receipts."
            )

    @property
    def subtitle(self) -> str:
        extra = len(self.receipts) - 1
        return f"{self.primary_status}{f' · +{extra}' if extra else ''} · Mark seen"

    @property
    def state_label(self) -> str:
        return self.primary_status.upper()

    @property
    def openable(self) -> bool:
        return False


ConsoleSwitcherActiveResult = ConsoleSwitcherEntry | UnavailableSessionNotice


_RUN_MARKER_LABELS = {
    "◆": "APPROVAL",
    "●": "RUNNING",
    "✗": "FAILED · UNSEEN",
    "✓": "FINISHED · UNSEEN",
    "◈": "SUB-AGENT · UNSEEN",
    "[!]": "APPROVAL",
    "[*]": "RUNNING",
    "[X]": "FAILED · UNSEEN",
    "[x]": "FINISHED · UNSEEN",
    "[s]": "SUB-AGENT · UNSEEN",
}


def _state_label(row: ConsoleConversationBrowserInputRow) -> str:
    parts: list[str] = []
    if not row.openable:
        parts.append("UNAVAILABLE")
    if row.selected:
        parts.append("CURRENT")
    marker_label = _RUN_MARKER_LABELS.get(str(row.run_marker or "").strip())
    if marker_label:
        parts.append(marker_label)
    if row.native_session_id and not marker_label:
        parts.append("OPEN AGENT")
    if row.queued_count > 0:
        parts.append(f"{row.queued_count} QUEUED")
    if not row.native_session_id:
        parts.append("saved chat")
    return " · ".join(parts)


def _matches_filter(row: ConsoleConversationBrowserInputRow, token: str) -> bool:
    if token.startswith("workspace:"):
        value = token.partition(":")[2]
        return bool(value) and value in str(row.workspace_label or "").lower()
    if not token.startswith("is:"):
        return False
    value = token.partition(":")[2]
    state = _state_label(row).lower()
    predicates = {
        "open": bool(row.native_session_id),
        "saved": not row.native_session_id,
        "current": bool(row.selected),
        "running": "running" in state,
        "approval": "approval" in state,
        "queued": row.queued_count > 0,
        "failed": "failed" in state,
        "finished": "finished" in state,
        "unavailable": not row.openable,
    }
    return predicates.get(value, False)


def _matches(row: ConsoleConversationBrowserInputRow, tokens: list[str]) -> bool:
    """Return whether every token matches the row's searchable text.

    ``ConsoleConversationBrowserInputRow`` is an unvalidated dataclass and
    rows are assembled by several different builders, so ``title``,
    ``workspace_label``, and ``status`` are coerced through ``str(... or "")``
    before joining -- a ``None`` in any field must not raise ``TypeError``.

    The haystack includes BOTH the raw ``status`` and its friendly detail
    (``console_conversation_status_detail``): TASK-356 made that friendly label
    the one shown in the subtitle, so a query for the visible word ("saved")
    must match even though the persisted status is still "in-progress"; the raw
    token stays searchable for back-compat.

    Args:
        row: Candidate browser input row.
        tokens: Lowercased query tokens that must all match.

    Returns:
        True if every token is a substring of the row's joined text.
    """
    haystack = " ".join(
        str(part or "")
        for part in (
            row.title,
            row.workspace_label,
            row.status,
            console_conversation_status_detail(row.status),
            _state_label(row),
        )
    ).lower()
    return all(
        _matches_filter(row, token)
        if token.startswith(("is:", "workspace:"))
        else token in haystack
        for token in tokens
    )


def build_console_switcher_entries(
    rows: Iterable[ConsoleConversationBrowserInputRow],
    *,
    query: str = "",
    limit: int = CONSOLE_SWITCHER_RESULT_LIMIT,
    now: datetime | None = None,
) -> tuple[ConsoleSwitcherEntry, ...]:
    """Build deduped, recent-first switcher results for a query.

    Args:
        rows: Browser input rows from the chat screen row builders.
        query: Whitespace-separated tokens; every token must match the row's
            title, workspace label, or status (case-insensitive substring).
        limit: Maximum number of entries returned.
        now: Reference time used to derive a recency label from ``updated_sort``
            when a row carries no precomputed ``updated_label``; defaults to the
            current UTC time.

    Returns:
        Up to ``limit`` entries, active row first, then most recent.
    """
    tokens = [token for token in query.lower().split() if token]
    seen: set[str] = set()
    deduped: list[ConsoleConversationBrowserInputRow] = []
    for row in rows:
        key = str(row.row_key or "")
        if not key or key in seen:
            continue
        seen.add(key)
        if tokens and not _matches(row, tokens):
            continue
        deduped.append(row)

    deduped.sort(
        key=lambda row: (
            not bool(row.native_session_id),
            not row.selected,
            ReverseKey(str(row.updated_sort or "")),
            row.title.casefold(),
            row.row_key,
        )
    )
    reference_now = now or datetime.now(timezone.utc)
    entries = []
    for row in deduped[: max(0, int(limit))]:
        # TASK-356: one state vocabulary across surfaces ("saved chat", not
        # the raw "in-progress"), and always show recency — deriving it from
        # updated_sort when the row carries no precomputed age label (the
        # switcher's input rows usually don't, unlike the rail's).
        status_detail = console_conversation_status_detail(row.status)
        recency = str(row.updated_label or "").strip() or format_console_relative_age(
            str(row.updated_sort or ""), now=reference_now
        )
        state_label = _state_label(row)
        subtitle = " · ".join(
            part
            for part in (row.workspace_label, state_label or status_detail, recency)
            if str(part or "").strip()
        )
        entries.append(
            ConsoleSwitcherEntry(
                row_key=str(row.row_key),
                title=str(row.title or "Untitled conversation"),
                subtitle=subtitle,
                native_session_id=row.native_session_id,
                conversation_id=row.conversation_id,
                scope_type=str(row.scope_type or ""),
                workspace_id=row.workspace_id,
                is_active=bool(row.selected),
                section="open" if row.native_session_id else "saved",
                state_label=state_label,
                openable=bool(row.openable),
            )
        )
    return tuple(entries)


_GROUP_ORDER = {
    ActivityGroup.WAITING_FOR_YOU: 0,
    ActivityGroup.WORKING: 1,
    ActivityGroup.NEW_RESULTS: 2,
    ActivityGroup.CURRENT: 3,
    ActivityGroup.OTHER_OPEN: 4,
}
_GROUP_LABELS = {
    ActivityGroup.WAITING_FOR_YOU: "WAITING FOR YOU",
    ActivityGroup.WORKING: "WORKING",
    ActivityGroup.NEW_RESULTS: "NEW RESULTS",
    ActivityGroup.CURRENT: "CURRENT",
    ActivityGroup.OTHER_OPEN: "OTHER OPEN",
}
_STATE_GROUP = {
    "human-input": ActivityGroup.WAITING_FOR_YOU,
    "approval": ActivityGroup.WAITING_FOR_YOU,
    "stuck": ActivityGroup.WAITING_FOR_YOU,
    "failed": ActivityGroup.WAITING_FOR_YOU,
    "error": ActivityGroup.WAITING_FOR_YOU,
    "stopped": ActivityGroup.WAITING_FOR_YOU,
    "paused": ActivityGroup.WAITING_FOR_YOU,
    "blocked": ActivityGroup.WAITING_FOR_YOU,
    "running": ActivityGroup.WORKING,
    "streaming": ActivityGroup.WORKING,
    "validating": ActivityGroup.WORKING,
    "retrying": ActivityGroup.WORKING,
    "checking_citations": ActivityGroup.WORKING,
    "queued": ActivityGroup.WORKING,
    "cancelled": ActivityGroup.NEW_RESULTS,
    "done": ActivityGroup.NEW_RESULTS,
    "completed": ActivityGroup.NEW_RESULTS,
    "succeeded": ActivityGroup.NEW_RESULTS,
    "current": ActivityGroup.CURRENT,
    "other-open": ActivityGroup.OTHER_OPEN,
}
_STATE_RANK = {
    state: rank
    for rank, state in enumerate(
        reversed(
            (
                "human-input",
                "approval",
                "stuck",
                "failed",
                "error",
                "stopped",
                "paused",
                "blocked",
                "running",
                "streaming",
                "validating",
                "retrying",
                "checking_citations",
                "queued",
                "cancelled",
                "done",
                "completed",
                "succeeded",
                "current",
                "other-open",
            )
        ),
        start=1,
    )
}
_STATE_COPY = {
    "human-input": "INPUT NEEDED",
    "approval": "APPROVAL",
    "stuck": "STUCK · UNSEEN",
    "failed": "FAILED · UNSEEN",
    "error": "FAILED · UNSEEN",
    "stopped": "STOPPED · UNSEEN",
    "paused": "PAUSED",
    "blocked": "INPUT NEEDED",
    "running": "RUNNING",
    "streaming": "RUNNING",
    "validating": "STARTING",
    "retrying": "RETRYING",
    "checking_citations": "CHECKING",
    "queued": "QUEUED",
    "cancelled": "CANCELLED · UNSEEN",
    "done": "FINISHED · UNSEEN",
    "completed": "FINISHED · UNSEEN",
    "succeeded": "FINISHED · UNSEEN",
    "current": "CURRENT",
    "other-open": "OPEN AGENT",
}


def _parse_instant(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_console_switcher_instant(
    value: str | datetime | None,
) -> datetime | None:
    """Public safe timestamp parser shared by bounded History adapters."""
    return _parse_instant(value)


def _normalized_state(value: str) -> str:
    state = str(value or "").strip().lower().replace(" ", "-")
    aliases = {
        "needs-approval": "approval",
        "finished-ok": "done",
        "finished-failed": "failed",
        "subagent-unseen": "done",
        "checking-citations": "checking_citations",
    }
    return aliases.get(state, state)


def _state_from_marker(marker: str) -> str | None:
    copy = _RUN_MARKER_LABELS.get(str(marker or "").strip(), "").lower()
    if "approval" in copy:
        return "approval"
    if "running" in copy:
        return "running"
    if "failed" in copy:
        return "failed"
    if "finished" in copy or "sub-agent" in copy:
        return "done"
    return None


@dataclass(frozen=True)
class _ActiveContribution:
    subject_key: str
    target_key: str
    source_key: str
    state: str
    group: ActivityGroup
    occurred_at: datetime | None
    session_id: str | None
    conversation_id: str | None
    row: ConsoleConversationBrowserInputRow | None
    receipt: CapturedReceipt | None = None
    source_kind: str = "shell"


def _subject_key(
    profile_authority: str,
    *,
    session_id: str | None,
    conversation_id: str | None,
) -> str:
    if conversation_id:
        return f"conversation:{profile_authority}:{conversation_id}"
    return f"session:{profile_authority}:{session_id}"


def _contribution_sort_key(contribution: _ActiveContribution) -> tuple[Any, ...]:
    instant = contribution.occurred_at
    return (
        _GROUP_ORDER[contribution.group],
        instant is None,
        -(instant.timestamp() if instant is not None else 0.0),
        -_STATE_RANK.get(contribution.state, 0),
        contribution.source_key,
    )


def _result_sort_key(result: ConsoleSwitcherActiveResult) -> tuple[Any, ...]:
    instant = result.latest_at
    starred = bool(getattr(result, "starred", False))
    return (
        _GROUP_ORDER[result.group],
        not starred,
        instant is None,
        -(instant.timestamp() if instant is not None else 0.0),
        result.title.casefold(),
        result.stable_result_key,
    )


def build_console_active_results(
    rows: Iterable[ConsoleConversationBrowserInputRow],
    *,
    receipts: Sequence[Any] = (),
    controller_signals: Sequence[ConsoleSwitcherActivitySignal] = (),
    profile_authority: str,
    authority_token: str,
    now: datetime | None = None,
) -> tuple[ConsoleSwitcherActiveResult, ...]:
    """Aggregate open shell, controller, and receipt signals by subject.

    The function consumes only normalized presentation metadata.  It never
    accepts or inspects transcript/message bodies.

    Args:
        rows: Safe local conversation and open-session presentation rows.
        receipts: Loosely sourced unseen receipt projections to validate.
        controller_signals: Content-free live controller activity signals.
        profile_authority: Profile identity owning every projected result.
        authority_token: App-lifetime token fencing stale result actions.
        now: Optional deterministic clock used for relative-age labels.

    Returns:
        Canonical Active results ordered by action priority and recency.
    """
    profile = str(profile_authority or "").strip()
    token = str(authority_token or "").strip()
    if not profile or not token:
        return ()
    reference_now = now or datetime.now(timezone.utc)
    row_tuple = tuple(rows)
    rows_by_session = {
        str(row.native_session_id): row for row in row_tuple if row.native_session_id
    }
    rows_by_conversation: dict[str, list[ConsoleConversationBrowserInputRow]] = {}
    for row in row_tuple:
        if row.conversation_id:
            rows_by_conversation.setdefault(str(row.conversation_id), []).append(row)
    open_sessions = frozenset(rows_by_session)
    contributions: list[_ActiveContribution] = []

    def add(
        *,
        state: str,
        source_key: str,
        session_id: str | None,
        conversation_id: str | None,
        occurred_at: str | datetime | None,
        row: ConsoleConversationBrowserInputRow | None,
        receipt: CapturedReceipt | None = None,
        source_kind: str,
    ) -> None:
        normalized = _normalized_state(state)
        group = _STATE_GROUP.get(normalized)
        if group is None or not (session_id or conversation_id):
            return
        subject = _subject_key(
            profile,
            session_id=session_id,
            conversation_id=conversation_id,
        )
        target_key = (
            f"native:{session_id}" if session_id else f"conversation:{conversation_id}"
        )
        contributions.append(
            _ActiveContribution(
                subject_key=subject,
                target_key=target_key,
                source_key=source_key,
                state=normalized,
                group=group,
                occurred_at=_parse_instant(occurred_at),
                session_id=session_id,
                conversation_id=conversation_id,
                row=row,
                receipt=receipt,
                source_kind=source_kind,
            )
        )

    for row in row_tuple:
        session_id = str(row.native_session_id or "").strip() or None
        if session_id is None:
            continue
        conversation_id = str(row.conversation_id or "").strip() or None
        add(
            state="current" if row.selected else "other-open",
            source_key=f"shell:{session_id}",
            session_id=session_id,
            conversation_id=conversation_id,
            occurred_at=row.updated_sort,
            row=row,
            source_kind="shell",
        )
        marker_state = _state_from_marker(row.run_marker)
        if marker_state:
            add(
                state=marker_state,
                source_key=f"controller:native:{session_id}:{marker_state}",
                session_id=session_id,
                conversation_id=conversation_id,
                occurred_at=row.updated_sort,
                row=row,
                source_kind="controller",
            )
        if row.queued_count > 0:
            add(
                state="queued",
                source_key=f"controller:native:{session_id}:queued",
                session_id=session_id,
                conversation_id=conversation_id,
                occurred_at=row.updated_sort,
                row=row,
                source_kind="controller",
            )

    for signal in controller_signals:
        session_id = str(signal.session_id or "").strip() or None
        row = rows_by_session.get(session_id or "")
        conversation_id = (
            str(signal.conversation_id or "").strip()
            or str(getattr(row, "conversation_id", "") or "").strip()
            or None
        )
        add(
            state=signal.state,
            source_key=str(signal.source_key or ""),
            session_id=session_id if session_id in open_sessions else None,
            conversation_id=conversation_id,
            occurred_at=signal.occurred_at,
            row=row,
            source_kind="controller",
        )

    unavailable: dict[
        tuple[str, str], list[tuple[CapturedReceipt, datetime | None]]
    ] = {}
    for raw in receipts:
        try:
            captured = CapturedReceipt.model_validate(
                {
                    "activity_id": getattr(raw, "activity_id"),
                    "status": getattr(raw, "status"),
                }
            )
        except (AttributeError, ValidationError):
            continue
        session_id = str(getattr(raw, "session_id", "") or "").strip() or None
        conversation_id = str(getattr(raw, "conversation_id", "") or "").strip() or None
        instant = _parse_instant(getattr(raw, "created_at", None))
        row = rows_by_session.get(session_id or "")
        if row is not None:
            conversation_id = conversation_id or row.conversation_id
        if session_id and session_id not in open_sessions and not conversation_id:
            unavailable.setdefault(("session", session_id), []).append(
                (captured, instant)
            )
            continue
        metadata_rows = rows_by_conversation.get(conversation_id or "", [])
        metadata_row = row or (metadata_rows[0] if metadata_rows else None)
        exact_session = session_id if session_id in open_sessions else None
        if (
            exact_session is None
            and conversation_id
            and (
                metadata_row is None
                or not bool(getattr(metadata_row, "openable", True))
            )
        ):
            unavailable.setdefault(("conversation", conversation_id), []).append(
                (captured, instant)
            )
            continue
        add(
            state=captured.status,
            source_key=f"receipt:{captured.activity_id}",
            session_id=exact_session,
            conversation_id=conversation_id,
            occurred_at=instant,
            row=metadata_row,
            receipt=captured,
            source_kind="receipt",
        )

    by_subject: dict[str, list[_ActiveContribution]] = {}
    for contribution in {item.source_key: item for item in contributions}.values():
        by_subject.setdefault(contribution.subject_key, []).append(contribution)

    results: list[ConsoleSwitcherActiveResult] = []
    for subject_key, subject_items in by_subject.items():
        by_target: dict[str, list[_ActiveContribution]] = {}
        for item in subject_items:
            by_target.setdefault(item.target_key, []).append(item)
        reduced: list[tuple[_ActiveContribution, tuple[CapturedReceipt, ...], int]] = []
        for target_items in by_target.values():
            winning_group = min(target_items, key=_contribution_sort_key).group
            eligible = [item for item in target_items if item.group is winning_group]
            winner = min(eligible, key=_contribution_sort_key)
            captured = tuple(
                sorted(
                    {
                        item.receipt.activity_id: item.receipt
                        for item in target_items
                        if item.receipt is not None
                    }.values(),
                    key=lambda receipt: receipt.activity_id,
                )
            )
            reduced.append((winner, captured, len(target_items)))

        def target_rank(
            value: tuple[_ActiveContribution, tuple[CapturedReceipt, ...], int],
        ) -> tuple[Any, ...]:
            winner, captured, _count = value
            instant = winner.occurred_at
            actionable_native = bool(
                winner.session_id
                and winner.group
                in {ActivityGroup.WAITING_FOR_YOU, ActivityGroup.WORKING}
            )
            destination_rank = 0 if actionable_native else 2 if captured else 3
            return (
                _GROUP_ORDER[winner.group],
                instant is None,
                -(instant.timestamp() if instant is not None else 0.0),
                destination_rank,
                winner.target_key,
            )

        winner, captured, _target_count = min(reduced, key=target_rank)
        subject_rows = [item.row for item in subject_items if item.row is not None]
        row = winner.row or (subject_rows[0] if subject_rows else None)
        conversation_id = winner.conversation_id
        session_id = winner.session_id
        kind = (
            SwitcherTargetKind.NATIVE_SESSION
            if session_id
            else SwitcherTargetKind.PERSISTED_CONVERSATION
        )
        target = ConsoleSwitcherTarget(
            kind=kind,
            profile_authority=profile,
            authority_token=token,
            session_id=session_id,
            conversation_id=conversation_id,
            scope_type=str(getattr(row, "scope_type", "") or "") or None,
            workspace_id=getattr(row, "workspace_id", None),
            receipts=captured,
        )
        source_copy = "CONSOLE TAB" if session_id else "SAVED CHAT"
        workspace_label = str(getattr(row, "workspace_label", "") or "")
        lifecycle = str(getattr(row, "status", "") or "")
        state_label = _STATE_COPY[winner.state]
        recency = str(getattr(row, "updated_label", "") or "").strip() or (
            format_console_relative_age(
                winner.occurred_at.isoformat(), now=reference_now
            )
            if winner.occurred_at is not None
            else ""
        )
        multiplicity = max(0, len({item.source_key for item in subject_items}) - 1)
        subtitle = " · ".join(
            part
            for part in (
                state_label,
                source_copy,
                workspace_label,
                console_conversation_status_detail(lifecycle) if lifecycle else "",
                recency,
                f"+{multiplicity}" if multiplicity else "",
            )
            if part
        )
        results.append(
            ConsoleSwitcherEntry(
                row_key=subject_key,
                title=str(getattr(row, "title", "") or "Conversation activity"),
                subtitle=subtitle,
                native_session_id=session_id,
                conversation_id=conversation_id,
                scope_type=str(getattr(row, "scope_type", "") or ""),
                workspace_id=getattr(row, "workspace_id", None),
                is_active=any(
                    bool(getattr(item.row, "selected", False)) for item in subject_items
                ),
                section=winner.group.value,
                state_label=state_label,
                openable=bool(getattr(row, "openable", True)),
                target=target,
                group=winner.group,
                activity_state=winner.state,
                latest_at=winner.occurred_at,
                starred=any(
                    bool(getattr(item.row, "starred", False)) for item in subject_items
                ),
                multiplicity=multiplicity,
                workspace_label=workspace_label,
                lifecycle=lifecycle,
            )
        )

    for (destination_kind, destination_id), receipt_rows in unavailable.items():
        receipts_only = tuple(
            sorted(
                (receipt for receipt, _instant in receipt_rows),
                key=lambda item: item.activity_id,
            )
        )
        states = tuple(sorted({receipt.status for receipt in receipts_only}))
        group = min((_STATE_GROUP[state] for state in states), key=_GROUP_ORDER.get)
        eligible = [
            (receipt, instant)
            for receipt, instant in receipt_rows
            if _STATE_GROUP[receipt.status] is group
        ]
        primary, instant = min(
            eligible,
            key=lambda item: (
                item[1] is None,
                -(item[1].timestamp() if item[1] is not None else 0.0),
                -_STATE_RANK[item[0].status],
                item[0].activity_id,
            ),
        )
        results.append(
            UnavailableSessionNotice(
                stable_result_key=(
                    f"unavailable-{destination_kind}:{profile}:{destination_id}"
                ),
                profile_authority=profile,
                authority_token=token,
                group=group,
                latest_at=instant,
                receipts=receipts_only,
                primary_status=primary.status,
                session_id=(destination_id if destination_kind == "session" else None),
                conversation_id=(
                    destination_id if destination_kind == "conversation" else None
                ),
                all_statuses=states,
                title=(
                    "Session unavailable"
                    if destination_kind == "session"
                    else "Conversation unavailable"
                ),
            )
        )
    return tuple(sorted(results, key=_result_sort_key))


def _normalized_active_query(query: str) -> list[str]:
    normalized = f" {str(query or '').casefold().strip()} "
    for phrase, replacement in (
        (" needs attention ", " is:waiting "),
        (" waiting on me ", " is:waiting "),
        (" new results ", " is:new "),
    ):
        normalized = normalized.replace(phrase, replacement)
    return [token for token in normalized.split() if token]


def plan_console_history_query(query: str) -> ConsoleSwitcherHistoryQuery:
    """Translate switcher semantics into bounded persisted-history filters.

    History contains saved destinations only. Semantic tokens that describe
    that invariant are consumed before title/message search; tokens describing
    live or unavailable activity cannot match. Workspace terms remain separate
    because their labels live in the Console workspace registry, not the
    conversation FTS index.

    Args:
        query: User-entered switcher query.

    Returns:
        A storage text query, workspace-label terms, and matchability flag.
    """
    text_terms: list[str] = []
    workspace_terms: list[str] = []
    history_false_terms = {
        "waiting",
        "working",
        "new",
        "current",
        "open",
        "unavailable",
        "running",
        "queued",
        "failed",
        "finished",
        "cancelled",
        "approval",
        "paused",
        "stuck",
        "stopped",
        "validating",
        "retrying",
    }
    for token in _normalized_active_query(query):
        if token.startswith("workspace:"):
            value = token.partition(":")[2]
            if not value:
                return ConsoleSwitcherHistoryQuery("", can_match=False)
            workspace_terms.append(value)
            continue
        if token.startswith("is:"):
            if token == "is:saved":
                continue
            return ConsoleSwitcherHistoryQuery("", can_match=False)
        if token == "saved":
            continue
        if token in history_false_terms:
            return ConsoleSwitcherHistoryQuery("", can_match=False)
        text_terms.append(token)
    return ConsoleSwitcherHistoryQuery(
        " ".join(text_terms),
        workspace_terms=tuple(workspace_terms),
    )


def _active_result_predicates(
    result: ConsoleSwitcherActiveResult,
) -> dict[str, bool]:
    """Return the shared semantic vocabulary for plain and ``is:`` terms."""
    group = result.group
    state = str(
        getattr(result, "activity_state", "") or getattr(result, "primary_status", "")
    )
    is_unavailable = isinstance(result, UnavailableSessionNotice)
    return {
        "waiting": group is ActivityGroup.WAITING_FOR_YOU,
        "working": group is ActivityGroup.WORKING,
        "new": group is ActivityGroup.NEW_RESULTS,
        # Current is destination identity, independent of the winning
        # activity group (a current tab may simultaneously be running).
        "current": bool(getattr(result, "is_active", False)),
        "open": bool(getattr(result, "native_session_id", None)) and not is_unavailable,
        # Persistence and openness overlap: a resumed saved conversation is
        # both saved and open, so neither alias hides the other identity.
        "saved": bool(getattr(result, "conversation_id", None)),
        "unavailable": is_unavailable,
        # User vocabulary treats both terms as routes to the Working group;
        # exact lifecycle states remain visible in labels and free text.
        "running": group is ActivityGroup.WORKING,
        "queued": group is ActivityGroup.WORKING,
        "failed": state in {"failed", "error"},
        "finished": state in {"done", "completed", "succeeded"},
        "cancelled": state == "cancelled",
        "approval": state == "approval",
        "paused": state in {"paused", "blocked"},
        "stuck": state == "stuck",
        "stopped": state == "stopped",
        "validating": state == "validating",
        "retrying": state == "retrying",
    }


def filter_console_active_results(
    results: Iterable[ConsoleSwitcherActiveResult], query: str
) -> tuple[ConsoleSwitcherActiveResult, ...]:
    """Filter safe normalized switcher metadata with deterministic aliases."""
    tokens = _normalized_active_query(query)
    if not tokens:
        return tuple(results)

    def matches(result: ConsoleSwitcherActiveResult) -> bool:
        group = result.group
        state = str(
            getattr(result, "activity_state", "")
            or getattr(result, "primary_status", "")
        )
        workspace = str(getattr(result, "workspace_label", "") or "").casefold()
        predicates = _active_result_predicates(result)
        haystack = " ".join(
            str(value or "")
            for value in (
                result.title,
                result.subtitle,
                result.state_label,
                state,
                _GROUP_LABELS[group],
                workspace,
                getattr(result, "lifecycle", ""),
                getattr(result, "session_id", ""),
                " ".join(getattr(result, "all_statuses", ())),
            )
        ).casefold()
        for token in tokens:
            if token.startswith("workspace:"):
                value = token.partition(":")[2]
                if not value or value not in workspace:
                    return False
                continue
            if token.startswith("is:"):
                value = token.partition(":")[2]
                if value not in predicates or not predicates[value]:
                    return False
                continue
            if token in predicates:
                if not predicates[token]:
                    return False
            elif token not in haystack:
                return False
        return True

    return tuple(result for result in results if matches(result))


def resolve_console_history_timezone(
    configured_name: object,
    *,
    system_timezone: tzinfo | None = None,
) -> tzinfo | None:
    """Resolve an explicit IANA zone; None delegates to host-local rules."""
    timezone_name = str(configured_name or "").strip()
    if timezone_name:
        try:
            return ZoneInfo(timezone_name)
        except (KeyError, ValueError):
            pass
    return system_timezone


def console_history_section(
    value: str | datetime | None,
    *,
    now: datetime,
    local_timezone: tzinfo | None,
) -> str:
    """Return the local-calendar section for one persisted timestamp."""
    instant = _parse_instant(value)
    if instant is None:
        return "Older"
    local_date = instant.astimezone(local_timezone).date()
    today = now.astimezone(local_timezone).date()
    if local_date >= today:
        return "Today"
    if local_date == today - timedelta(days=1):
        return "Yesterday"
    if local_date >= today - timedelta(days=7):
        return "Previous 7 days"
    return "Older"


def group_console_history_entries(
    entries: Iterable[ConsoleSwitcherEntry],
    *,
    now: datetime,
    local_timezone: tzinfo | None,
) -> tuple[ConsoleSwitcherEntry, ...]:
    """Apply pure fixed-order local-calendar sections to a bounded page."""
    order = {"Today": 0, "Yesterday": 1, "Previous 7 days": 2, "Older": 3}
    grouped = [
        ConsoleSwitcherEntry(
            **{
                **entry.__dict__,
                "section": console_history_section(
                    entry.latest_at,
                    now=now,
                    local_timezone=local_timezone,
                ),
            }
        )
        for entry in entries
    ]
    return tuple(
        sorted(
            grouped,
            key=lambda entry: (
                order[entry.section],
                entry.title.casefold(),
                entry.row_key,
            ),
        )
    )
