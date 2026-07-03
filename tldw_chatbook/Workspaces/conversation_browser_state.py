"""Pure display state for the Console grouped conversation browser."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import total_ordering
from typing import Iterable, Mapping

from .models import DEFAULT_WORKSPACE_ID


CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT = 75
CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT = 12


def _parse_browser_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def format_console_relative_age(value: str, *, now: datetime) -> str:
    """Return a compact age label such as ``2m``, ``1h``, ``3d`` for a timestamp.

    Args:
        value: ISO-8601-ish timestamp text; naive values are treated as UTC.
        now: Reference time for age calculation.

    Returns:
        Compact age label, or an empty string when the value is unparseable.
    """
    parsed = _parse_browser_timestamp(value)
    if parsed is None:
        return ""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    total_seconds = max(0.0, (now - parsed).total_seconds())
    minutes = int(total_seconds // 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    if days < 7:
        return f"{days}d"
    weeks = days // 7
    if days < 365:
        return f"{weeks}w"
    return f"{days // 365}y"


@dataclass(frozen=True)
class ConsoleConversationBrowserInputRow:
    """Input row used to build the grouped Console conversation browser.

    Attributes:
        row_key: Stable display identity for this row.
        conversation_id: Persisted conversation id, when the row can be resumed.
        native_session_id: Unsaved native session id, when the row is draft-only.
        title: Conversation title shown in the browser.
        scope_type: Conversation scope, such as ``"workspace"`` or ``"global"``.
        workspace_id: Workspace id associated with the row, if any.
        workspace_label: Workspace label shown to the user.
        status: Small status copy for the row.
        updated_label: Human-readable age/update label.
        selected: Whether this row represents the active conversation.
        starred: Whether this row is locally starred.
        star_enabled: Whether the UI may toggle a local star for this row.
        source_kind: Source of the row, such as persisted, native, or membership.
        starred_sort: Timestamp-like value used for starred section ordering.
        updated_sort: Timestamp-like value used for normal ordering.
    """

    row_key: str
    conversation_id: str | None
    native_session_id: str | None
    title: str
    scope_type: str
    workspace_id: str | None
    workspace_label: str
    status: str = ""
    updated_label: str = ""
    selected: bool = False
    starred: bool = False
    star_enabled: bool = True
    source_kind: str = "persisted"
    starred_sort: str = ""
    updated_sort: str = ""


@dataclass(frozen=True)
class ConsoleConversationBrowserRow:
    """Rendered row in a Console conversation browser section or group.

    Attributes:
        row_key: Stable display identity for this row.
        conversation_id: Persisted conversation id, when available.
        native_session_id: Unsaved native session id, when available.
        title: Conversation title shown in the browser.
        scope_type: Conversation scope.
        workspace_id: Workspace id associated with the row, if any.
        workspace_label: Workspace label shown to the user.
        status: Small status copy for the row.
        updated_label: Human-readable age/update label.
        selected: Whether this row represents the active conversation.
        starred: Whether this row is locally starred.
        star_enabled: Whether the UI may toggle a local star for this row.
        source_kind: Source of the row.
    """

    row_key: str
    conversation_id: str | None
    native_session_id: str | None
    title: str
    scope_type: str
    workspace_id: str | None
    workspace_label: str
    status: str
    updated_label: str
    selected: bool = False
    starred: bool = False
    star_enabled: bool = True
    source_kind: str = "persisted"


@dataclass(frozen=True)
class ConsoleConversationBrowserGroup:
    """Workspace group in the Console conversation browser.

    Attributes:
        group_id: Stable group id, normally prefixed with ``"workspace:"``.
        label: Workspace label shown in the browser.
        collapsed: Whether rows are hidden for the current render.
        rows: Visible rows after group capping.
        count: Total rows available in this group.
        hidden_count: Rows hidden by collapse or per-group capping.
        preference_collapsed: Persisted collapse preference before query expansion.
        empty_copy: Empty-state copy for the group.
    """

    group_id: str
    label: str
    collapsed: bool
    rows: tuple[ConsoleConversationBrowserRow, ...]
    count: int
    hidden_count: int = 0
    preference_collapsed: bool = False
    empty_copy: str = ""


@dataclass(frozen=True)
class ConsoleConversationBrowserSection:
    """Top-level section in the Console conversation browser.

    Attributes:
        section_id: Stable section id, such as ``"starred"``.
        label: Section label shown in the browser.
        collapsed: Whether this section is currently collapsed.
        rows: Visible rows for row-based sections.
        groups: Workspace groups for grouped sections.
        count: Total row count represented by the section.
        hidden_count: Rows hidden by collapse or row capping.
        empty_copy: Empty-state copy for the section.
    """

    section_id: str
    label: str
    collapsed: bool
    rows: tuple[ConsoleConversationBrowserRow, ...] = ()
    groups: tuple[ConsoleConversationBrowserGroup, ...] = ()
    count: int = 0
    hidden_count: int = 0
    empty_copy: str = ""


@dataclass(frozen=True)
class ConsoleConversationBrowserState:
    """Immutable grouped Console conversation browser display state.

    Attributes:
        query: Normalized active search query.
        sections: Ordered browser sections.
        selected_summary: Human-readable summary of the active row.
        status_copy: Search/result status copy.
        error_copy: Recoverable error copy for the browser.
        marks_available: Whether local star state is available.
        result_total_count: Total matched rows when known.
        result_limit: Caller fetch cap retained for diagnostics/UI state.
    """

    query: str
    sections: tuple[ConsoleConversationBrowserSection, ...]
    selected_summary: str
    status_copy: str = ""
    error_copy: str = ""
    marks_available: bool = True
    result_total_count: int | None = None
    result_limit: int = CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT


def build_console_conversation_browser_state(
    *,
    rows: Iterable[ConsoleConversationBrowserInputRow],
    active_workspace_id: str | None,
    group_collapse_preferences: Mapping[str, bool] | None = None,
    query: str = "",
    marks_available: bool = True,
    error_copy: str = "",
    result_total_count: int | None = None,
    result_limit: int = CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
    group_row_limit: int = CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT,
    now: datetime | None = None,
) -> ConsoleConversationBrowserState:
    """Build a deterministic grouped conversation browser snapshot.

    Args:
        rows: Input rows from the eventual controller layer. The builder does
            not query services or mutate UI preferences.
        active_workspace_id: Workspace id that should be expanded by default.
        group_collapse_preferences: Tri-state collapse preferences where a
            missing key means use the builder default.
        query: Search text matched against row title, workspace, status, and
            scope copy.
        marks_available: Whether local star state is available for rendering.
        error_copy: Optional scoped recovery copy.
        result_total_count: Total matched rows when the caller already knows a
            larger search result count.
        result_limit: Caller-level result cap retained on the resulting state.
        group_row_limit: Maximum visible rows per section or workspace group.
        now: Reference time for computing relative age labels. Defaults to now.

    Returns:
        Grouped, capped, immutable display state.
    """

    preferences = dict(group_collapse_preferences or {})
    normalized_query = str(query or "").strip().lower()
    query_active = bool(normalized_query)
    safe_result_limit = max(1, int(result_limit))
    safe_group_row_limit = max(0, int(group_row_limit))
    reference_now = now or datetime.now(timezone.utc)
    prepared_rows = tuple(_normalize_input_row(row, now=reference_now) for row in rows)
    filtered_rows = tuple(
        row for row in prepared_rows if not query_active or _row_matches(row, normalized_query)
    )

    starred_rows = _dedupe_rows(
        row for row in filtered_rows if row.starred and bool(row.conversation_id)
    )
    workspace_rows_by_group: dict[str, list[ConsoleConversationBrowserInputRow]] = {}
    workspace_labels: dict[str, str] = {}
    chat_rows: list[ConsoleConversationBrowserInputRow] = []

    for row in filtered_rows:
        if _belongs_to_chats(row):
            chat_rows.append(row)
            continue
        workspace_id = _text_or_none(row.workspace_id)
        if workspace_id is None:
            chat_rows.append(row)
            continue
        group_id = f"workspace:{workspace_id}"
        workspace_rows_by_group.setdefault(group_id, []).append(row)
        workspace_labels.setdefault(group_id, row.workspace_label or workspace_id)

    starred_section = _build_row_section(
        section_id="starred",
        label="Starred",
        rows=_sort_starred_rows(starred_rows),
        preference_collapsed=_resolve_collapsed(
            preferences,
            "section:starred",
            default_collapsed=False,
        ),
        query_active=query_active,
        group_row_limit=safe_group_row_limit,
        empty_copy="No starred conversations.",
    )
    workspace_groups = _build_workspace_groups(
        rows_by_group=workspace_rows_by_group,
        labels=workspace_labels,
        active_workspace_id=_text_or_none(active_workspace_id),
        preferences=preferences,
        query_active=query_active,
        group_row_limit=safe_group_row_limit,
    )
    workspaces_preference_collapsed = _resolve_collapsed(
        preferences,
        "section:workspaces",
        default_collapsed=False,
    )
    workspaces_collapsed = workspaces_preference_collapsed and not (
        query_active and bool(workspace_groups)
    )
    workspaces_section = ConsoleConversationBrowserSection(
        section_id="workspaces",
        label="Workspaces",
        collapsed=workspaces_collapsed,
        groups=workspace_groups,
        count=sum(group.count for group in workspace_groups),
        hidden_count=(
            sum(group.count for group in workspace_groups)
            if workspaces_collapsed
            else sum(group.hidden_count for group in workspace_groups)
        ),
        empty_copy="No workspace conversations.",
    )
    chat_input_rows = _sort_normal_rows(_dedupe_rows(chat_rows))
    chat_preference_collapsed = _resolve_collapsed(
        preferences,
        "section:chats",
        default_collapsed=not bool(chat_input_rows),
    )
    chats_section = _build_row_section(
        section_id="chats",
        label="Chats",
        rows=chat_input_rows,
        preference_collapsed=chat_preference_collapsed,
        query_active=query_active,
        group_row_limit=safe_group_row_limit,
        empty_copy="No chats.",
    )

    sections = (starred_section, workspaces_section, chats_section)
    effective_total_count = (
        _safe_non_negative_int(result_total_count)
        if result_total_count is not None
        else len(_dedupe_rows(filtered_rows))
    )
    status_copy = _build_status_copy(
        query_active=query_active,
        total_count=effective_total_count,
        displayed_count=_displayed_row_count(sections),
    )

    return ConsoleConversationBrowserState(
        query=normalized_query,
        sections=sections,
        selected_summary=_selected_summary(prepared_rows),
        status_copy=status_copy,
        error_copy=str(error_copy or ""),
        marks_available=bool(marks_available),
        result_total_count=effective_total_count if query_active else result_total_count,
        result_limit=safe_result_limit,
    )


def _normalize_input_row(
    row: ConsoleConversationBrowserInputRow, *, now: datetime
) -> ConsoleConversationBrowserInputRow:
    conversation_id = _text_or_none(row.conversation_id)
    native_session_id = _text_or_none(row.native_session_id)
    source_kind = str(row.source_kind or "persisted")
    star_enabled = bool(row.star_enabled and conversation_id)
    return ConsoleConversationBrowserInputRow(
        row_key=str(row.row_key or ""),
        conversation_id=conversation_id,
        native_session_id=native_session_id,
        title=str(row.title or ""),
        scope_type=str(row.scope_type or ""),
        workspace_id=_text_or_none(row.workspace_id),
        workspace_label=str(row.workspace_label or ""),
        status=str(row.status or ""),
        updated_label=(
            str(row.updated_label or "")
            or format_console_relative_age(str(row.updated_sort or ""), now=now)
        ),
        selected=bool(row.selected),
        starred=bool(row.starred),
        star_enabled=star_enabled,
        source_kind=source_kind,
        starred_sort=str(row.starred_sort or ""),
        updated_sort=str(row.updated_sort or ""),
    )


def _to_browser_row(row: ConsoleConversationBrowserInputRow) -> ConsoleConversationBrowserRow:
    return ConsoleConversationBrowserRow(
        row_key=row.row_key,
        conversation_id=row.conversation_id,
        native_session_id=row.native_session_id,
        title=row.title,
        scope_type=row.scope_type,
        workspace_id=row.workspace_id,
        workspace_label=row.workspace_label,
        status=row.status,
        updated_label=row.updated_label,
        selected=row.selected,
        starred=row.starred,
        star_enabled=row.star_enabled,
        source_kind=row.source_kind,
    )


def _build_row_section(
    *,
    section_id: str,
    label: str,
    rows: tuple[ConsoleConversationBrowserInputRow, ...],
    preference_collapsed: bool,
    query_active: bool,
    group_row_limit: int,
    empty_copy: str,
) -> ConsoleConversationBrowserSection:
    collapsed = preference_collapsed and not (query_active and bool(rows))
    visible_rows, hidden_count = _visible_rows(rows, collapsed, group_row_limit)
    return ConsoleConversationBrowserSection(
        section_id=section_id,
        label=label,
        collapsed=collapsed,
        rows=visible_rows,
        count=len(rows),
        hidden_count=hidden_count,
        empty_copy=empty_copy,
    )


def _build_workspace_groups(
    *,
    rows_by_group: Mapping[str, list[ConsoleConversationBrowserInputRow]],
    labels: Mapping[str, str],
    active_workspace_id: str | None,
    preferences: Mapping[str, bool],
    query_active: bool,
    group_row_limit: int,
) -> tuple[ConsoleConversationBrowserGroup, ...]:
    groups: list[tuple[str, str, str, tuple[ConsoleConversationBrowserInputRow, ...]]] = []
    for group_id, group_rows in rows_by_group.items():
        deduped_rows = _sort_normal_rows(_dedupe_rows(group_rows))
        latest_sort = max((row.updated_sort for row in deduped_rows), default="")
        groups.append((group_id, str(labels.get(group_id) or group_id), latest_sort, deduped_rows))

    groups.sort(key=lambda group: _workspace_group_sort_key(group, active_workspace_id))

    browser_groups: list[ConsoleConversationBrowserGroup] = []
    for group_id, label, _latest_sort, group_rows in groups:
        workspace_id = group_id.removeprefix("workspace:")
        default_collapsed = workspace_id != active_workspace_id
        preference_collapsed = _resolve_collapsed(
            preferences,
            group_id,
            default_collapsed=default_collapsed,
        )
        collapsed = preference_collapsed and not (query_active and bool(group_rows))
        visible_rows, hidden_count = _visible_rows(group_rows, collapsed, group_row_limit)
        browser_groups.append(
            ConsoleConversationBrowserGroup(
                group_id=group_id,
                label=label,
                collapsed=collapsed,
                rows=visible_rows,
                count=len(group_rows),
                hidden_count=hidden_count,
                preference_collapsed=preference_collapsed,
                empty_copy="No workspace conversations.",
            )
        )
    return tuple(browser_groups)


def _visible_rows(
    rows: tuple[ConsoleConversationBrowserInputRow, ...],
    collapsed: bool,
    group_row_limit: int,
) -> tuple[tuple[ConsoleConversationBrowserRow, ...], int]:
    if collapsed:
        return (), 0
    visible_input_rows = rows[:group_row_limit] if group_row_limit else ()
    hidden_count = max(0, len(rows) - len(visible_input_rows))
    return tuple(_to_browser_row(row) for row in visible_input_rows), hidden_count


def _dedupe_rows(
    rows: Iterable[ConsoleConversationBrowserInputRow],
) -> tuple[ConsoleConversationBrowserInputRow, ...]:
    seen: set[str] = set()
    deduped: list[ConsoleConversationBrowserInputRow] = []
    for row in rows:
        if row.row_key in seen:
            continue
        seen.add(row.row_key)
        deduped.append(row)
    return tuple(deduped)


def _sort_normal_rows(
    rows: tuple[ConsoleConversationBrowserInputRow, ...],
) -> tuple[ConsoleConversationBrowserInputRow, ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                not row.selected,
                ReverseKey(row.updated_sort),
                row.title.casefold(),
                row.row_key,
            ),
        )
    )


def _sort_starred_rows(
    rows: tuple[ConsoleConversationBrowserInputRow, ...],
) -> tuple[ConsoleConversationBrowserInputRow, ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                ReverseKey(row.starred_sort),
                ReverseKey(row.updated_sort),
                row.title.casefold(),
                row.row_key,
            ),
        )
    )


def _workspace_group_sort_key(
    group: tuple[str, str, str, tuple[ConsoleConversationBrowserInputRow, ...]],
    active_workspace_id: str | None,
) -> tuple[bool, "ReverseKey", str, str]:
    group_id, label, latest_sort, _rows = group
    is_active = group_id == f"workspace:{active_workspace_id}" if active_workspace_id else False
    return (not is_active, ReverseKey(latest_sort), label.casefold(), group_id)


def _row_matches(row: ConsoleConversationBrowserInputRow, normalized_query: str) -> bool:
    haystack = " ".join(
        (
            row.title,
            row.workspace_label,
            row.status,
            _scope_copy(row),
        )
    ).lower()
    return normalized_query in haystack


def _scope_copy(row: ConsoleConversationBrowserInputRow) -> str:
    if row.scope_type == "global":
        return "global chats"
    if row.workspace_id == DEFAULT_WORKSPACE_ID:
        return "default workspace chats"
    if row.workspace_id:
        return f"workspace {row.workspace_label}"
    return "chats"


def _belongs_to_chats(row: ConsoleConversationBrowserInputRow) -> bool:
    return row.scope_type == "global" or row.workspace_id in (None, DEFAULT_WORKSPACE_ID)


def _resolve_collapsed(
    preferences: Mapping[str, bool],
    key: str,
    *,
    default_collapsed: bool,
) -> bool:
    if key in preferences:
        return bool(preferences[key])
    return bool(default_collapsed)


def _selected_summary(rows: tuple[ConsoleConversationBrowserInputRow, ...]) -> str:
    selected = next((row for row in rows if row.selected), None)
    if selected is None:
        return ""
    if selected.title and selected.workspace_label:
        return f"{selected.title} - {selected.workspace_label}"
    return selected.title or selected.workspace_label


def _build_status_copy(
    *,
    query_active: bool,
    total_count: int,
    displayed_count: int,
) -> str:
    if not query_active:
        return ""
    match_label = "match" if total_count == 1 else "matches"
    status = f"{total_count} {match_label}"
    shown_count = min(total_count, displayed_count)
    if total_count > shown_count:
        status = f"{status}. Showing {shown_count} of {total_count}"
    return status


def _displayed_row_count(
    sections: tuple[ConsoleConversationBrowserSection, ...],
) -> int:
    total = 0
    for section in sections:
        if section.collapsed:
            continue
        total += len(section.rows)
        total += sum(len(group.rows) for group in section.groups)
    return total


def _text_or_none(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _safe_non_negative_int(value: int | None) -> int:
    return max(0, int(value or 0))


@total_ordering
@dataclass(frozen=True)
class ReverseKey:
    """Sort key wrapper that compares text values in descending order."""

    value: str

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ReverseKey):
            return NotImplemented
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReverseKey):
            return NotImplemented
        return self.value == other.value
