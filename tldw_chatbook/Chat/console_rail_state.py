"""Pure rail-state contracts for the native Console workbench."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

CONSOLE_RAIL_LEFT_DEFAULT_OPEN = True
CONSOLE_RAIL_RIGHT_DEFAULT_OPEN = False
CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS = 150
CONSOLE_RAIL_CONTEXT_LABEL = "Context"
CONSOLE_RAIL_INSPECTOR_LABEL = "Inspector"

_PERSISTENCE_PREFIX = "console_rail_state"
_INVALID_KEY_RUN_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TRUE_STRINGS = {"true", "yes", "1", "on"}
_FALSE_STRINGS = {"false", "no", "0", "off"}
_WORKSPACE_FALLBACK_LABELS = {
    "local",
    "default",
    "global",
    "no workspace",
    "no-workspace",
    "no_workspace",
}
_NEGATIVE_READINESS_TERMS = {
    "blocked",
    "missing source",
    "no results",
    "not available",
    "not staged",
    "unavailable",
}


@dataclass(frozen=True)
class ConsoleRailPreferences:
    """Persisted user preferences for Console side rail openness."""

    left_open: bool = CONSOLE_RAIL_LEFT_DEFAULT_OPEN
    right_open: bool = CONSOLE_RAIL_RIGHT_DEFAULT_OPEN


@dataclass(frozen=True)
class ConsoleRailPreferenceKey:
    """Primary and optional fallback persistence key for Console rail state."""

    workspace_id: str
    scope_id: str
    value: str
    fallback_value: str | None = None


@dataclass(frozen=True)
class ConsoleRailState:
    """Effective Console rail state after preferences and responsive rules."""

    left_open: bool
    right_open: bool
    preferred_left_open: bool
    preferred_right_open: bool
    left_label: str = CONSOLE_RAIL_CONTEXT_LABEL
    right_label: str = CONSOLE_RAIL_INSPECTOR_LABEL
    left_badge: str = ""
    right_badge: str = ""
    persistence_key: str = ""
    right_forced_collapsed: bool = False


def _sanitize_key_part(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    sanitized = _INVALID_KEY_RUN_RE.sub("_", text).strip("_")
    return sanitized or "global"


def _build_persistence_key(workspace_id: str, scope_id: str) -> str:
    return f"{_PERSISTENCE_PREFIX}:{workspace_id}:{scope_id}"


def build_console_rail_preference_key(
    *,
    workspace_id: Any = None,
    conversation_id: Any = None,
    session_id: Any = None,
) -> ConsoleRailPreferenceKey:
    """Build the deterministic persistence key for Console rail preferences.

    Args:
        workspace_id: Workspace scope value, or global when empty.
        conversation_id: Preferred conversation-specific scope value.
        session_id: Temporary session scope used when no conversation exists.

    Returns:
        Primary preference key, with a session fallback when both conversation
        and session scopes are available.
    """
    workspace_scope = _sanitize_key_part(workspace_id)
    conversation_scope = _sanitize_key_part(conversation_id) if conversation_id else ""
    session_scope = _sanitize_key_part(session_id) if session_id else ""

    if conversation_scope:
        fallback_value = (
            _build_persistence_key(workspace_scope, session_scope)
            if session_scope
            else None
        )
        return ConsoleRailPreferenceKey(
            workspace_id=workspace_scope,
            scope_id=conversation_scope,
            value=_build_persistence_key(workspace_scope, conversation_scope),
            fallback_value=fallback_value,
        )

    scope_id = session_scope or "global"
    return ConsoleRailPreferenceKey(
        workspace_id=workspace_scope,
        scope_id=scope_id,
        value=_build_persistence_key(workspace_scope, scope_id),
    )


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return fallback


def coerce_console_rail_preferences(raw: Any) -> ConsoleRailPreferences:
    """Normalize stored Console rail preferences.

    Args:
        raw: Dict-like stored value.

    Returns:
        Rail preferences with invalid or missing fields replaced by defaults.
    """
    defaults = ConsoleRailPreferences()
    if not isinstance(raw, Mapping):
        return defaults

    return ConsoleRailPreferences(
        left_open=_coerce_bool(raw.get("left_open"), defaults.left_open),
        right_open=_coerce_bool(raw.get("right_open"), defaults.right_open),
    )


def serialize_console_rail_preferences(
    preferences: ConsoleRailPreferences,
) -> dict[str, bool]:
    """Serialize Console rail preferences to the persistence shape."""
    return {
        "left_open": bool(preferences.left_open),
        "right_open": bool(preferences.right_open),
    }


def _coerce_non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_console_context_rail_badge(
    *,
    staged_source_count: Any = 0,
    staged_summary: Any = "",
    workspace_label: Any = "",
    session_label: Any = "",
) -> str:
    """Build the left rail badge from staged context and workspace state."""
    count = _coerce_non_negative_int(staged_source_count)
    if count > 0:
        return f"{count} staged"

    if _clean_text(staged_summary):
        return "staged"

    workspace_text = _clean_text(workspace_label)
    if workspace_text and workspace_text.lower() not in _WORKSPACE_FALLBACK_LABELS:
        return "workspace"

    if _clean_text(session_label):
        return "session"

    return ""


def _row_text_parts(row: Any) -> tuple[str, str, str, str]:
    return (
        _clean_text(getattr(row, "label", "")),
        _clean_text(getattr(row, "status", "")),
        _clean_text(getattr(row, "value", "")),
        _clean_text(getattr(row, "text", "")),
    )


def _normalized_status(value: Any) -> str:
    status = getattr(value, "value", value)
    return _clean_text(status).lower()


def _has_row_match(rows: tuple[Any, ...], candidates: set[str]) -> bool:
    for row in rows:
        combined = " ".join(part.lower() for part in _row_text_parts(row) if part)
        if any(candidate in combined for candidate in candidates):
            return True
    return False


def _has_row_readiness_match(rows: tuple[Any, ...], candidates: set[str]) -> bool:
    for row in rows:
        _, status, value, text = _row_text_parts(row)
        combined = " ".join(
            part.lower()
            for part in (status, value, text)
            if part
        )
        if not combined or any(term in combined for term in _NEGATIVE_READINESS_TERMS):
            continue
        if any(candidate in combined for candidate in candidates):
            return True
    return False


def build_console_inspector_rail_badge(
    *,
    run_status: Any = None,
    inspector_rows: tuple[Any, ...] = (),
    tool_count: Any = 0,
    approval_count: Any = 0,
    can_save_chatbook: bool = False,
) -> str:
    """Build the right rail badge from run, review, tool, and artifact state."""
    normalized_run_status = _normalized_status(run_status)
    if normalized_run_status == "failed" or _has_row_match(inspector_rows, {"failed"}):
        return "failed"

    if normalized_run_status == "blocked" or _has_row_match(
        inspector_rows,
        {"blocked"},
    ):
        return "blocked"

    approvals = _coerce_non_negative_int(approval_count)
    if approvals == 1:
        return "1 approval"
    if approvals > 1:
        return f"{approvals} approvals"

    if _coerce_non_negative_int(tool_count) > 0:
        return "tools"

    if can_save_chatbook or _has_row_readiness_match(
        inspector_rows,
        {"artifact", "chatbook"},
    ):
        return "artifact"

    if _has_row_readiness_match(inspector_rows, {"source", "rag", "staged"}):
        return "source"

    return ""


def build_console_rail_state(
    *,
    preference_key: ConsoleRailPreferenceKey,
    stored_preferences: Any = None,
    staged_source_count: Any = 0,
    staged_summary: Any = "",
    workspace_label: Any = "",
    session_label: Any = "",
    run_status: Any = None,
    inspector_rows: tuple[Any, ...] = (),
    tool_count: Any = 0,
    approval_count: Any = 0,
    can_save_chatbook: bool = False,
    available_columns: int | None = None,
) -> ConsoleRailState:
    """Build effective Console rail state without importing Textual."""
    preferences = coerce_console_rail_preferences(stored_preferences)
    right_forced_collapsed = (
        available_columns is not None
        and available_columns < CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS
    )

    return ConsoleRailState(
        left_open=preferences.left_open,
        right_open=False if right_forced_collapsed else preferences.right_open,
        preferred_left_open=preferences.left_open,
        preferred_right_open=preferences.right_open,
        left_badge=build_console_context_rail_badge(
            staged_source_count=staged_source_count,
            staged_summary=staged_summary,
            workspace_label=workspace_label,
            session_label=session_label,
        ),
        right_badge=build_console_inspector_rail_badge(
            run_status=run_status,
            inspector_rows=inspector_rows,
            tool_count=tool_count,
            approval_count=approval_count,
            can_save_chatbook=can_save_chatbook,
        ),
        persistence_key=preference_key.value,
        right_forced_collapsed=right_forced_collapsed,
    )
