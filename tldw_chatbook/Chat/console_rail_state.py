"""Pure rail-state contracts for the native Console workbench."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

from tldw_chatbook.Chat.console_glyphs import GLYPH_COLLAPSE_LEFT, GLYPH_COLLAPSED

CONSOLE_RAIL_LEFT_DEFAULT_OPEN = True
CONSOLE_RAIL_RIGHT_DEFAULT_OPEN = False
# Task-400: the "context" (staged sources) section moved from the left rail
# into the Inspector rail, so it is no longer a collapsible left-rail section.
CONSOLE_RAIL_SECTION_IDS = ("session", "model", "details", "agent")
CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS = 150
CONSOLE_RAIL_CONTEXT_LABEL = f"Context {GLYPH_COLLAPSED}"
CONSOLE_RAIL_INSPECTOR_LABEL = f"{GLYPH_COLLAPSE_LEFT} Inspector"

_PERSISTENCE_PREFIX = "console_rail_state"
_INVALID_KEY_RUN_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_TRUE_STRINGS = {"true", "yes", "1", "on"}
_FALSE_STRINGS = {"false", "no", "0", "off"}
_WORKSPACE_FALLBACK_LABELS = {
    "local",
    "default",
    "global",
    "no workspace",
    "no workspace selected",
    "no-workspace",
    "no_workspace",
    "workspace: default",
    "workspace: local default",
}
_INACTIVE_STAGED_SUMMARIES = {
    "no live work item is staged",
    "no staged work",
    "no sources attached",
}
_NEGATIVE_READINESS_TERMS = {
    "blocked",
    "missing source",
    "no results",
    "not available",
    "not requested",
    "not staged",
    "unavailable",
}
_POSITIVE_READINESS_TERMS = {
    "attached",
    "available",
    "ready",
    "retrieving",
    "staged",
}
_SETUP_BLOCKER_LABEL_TERMS = {"model", "provider"}
_SETUP_BLOCKER_READINESS_TERMS = {
    "blocked",
    "invalid",
    "missing",
    "unavailable",
    "unconfigured",
}


@dataclass(frozen=True)
class ConsoleRailPreferences:
    """Persisted user preferences for Console side rail openness."""

    left_open: bool = CONSOLE_RAIL_LEFT_DEFAULT_OPEN
    right_open: bool = CONSOLE_RAIL_RIGHT_DEFAULT_OPEN
    session_open: bool = True
    model_open: bool = True
    details_open: bool = False
    agent_open: bool = False


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
    session_open: bool = True
    model_open: bool = True
    details_open: bool = False
    agent_open: bool = False


def _sanitize_key_part(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    sanitized = _INVALID_KEY_RUN_RE.sub("_", text).strip("_")
    return sanitized or "global"


def _sanitize_optional_key_part(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return _sanitize_key_part(text)


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
    conversation_scope = _sanitize_optional_key_part(conversation_id)
    session_scope = _sanitize_optional_key_part(session_id)

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


def collect_prunable_console_rail_keys(
    stored_keys: Any,
    *,
    live_scope_ids: Any,
) -> list[str]:
    """Return stored rail-preference keys whose scope is no longer live.

    A key is prunable only when it matches the canonical
    ``console_rail_state:<workspace>:<scope>`` shape and its scope id is
    neither the reserved ``global`` scope nor present in ``live_scope_ids``.
    Unrecognized key shapes are always kept.

    Args:
        stored_keys: Iterable of stored config key strings (non-string
            entries are ignored). ``None`` is treated as empty.
        live_scope_ids: Iterable of live scope ids (conversation ids plus
            open session ids), matched after the module's key sanitization.
            ``None`` is treated as empty.

    Returns:
        The subset of ``stored_keys`` safe to delete, order-preserved.
    """
    live_sanitized = {
        sanitized
        for raw in (live_scope_ids or ())
        for sanitized in (_sanitize_optional_key_part(raw),)
        if sanitized
    }
    prunable: list[str] = []
    for key in stored_keys or ():
        if not isinstance(key, str):
            continue
        parts = key.split(":")
        if len(parts) != 3 or parts[0] != _PERSISTENCE_PREFIX:
            continue
        scope_id = parts[2]
        if scope_id == "global" or scope_id in live_sanitized:
            continue
        prunable.append(key)
    return prunable


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
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
        Legacy ``context_open`` keys (persisted before task-400 moved the
        staged-sources Context section into the Inspector rail) are ignored.
    """
    defaults = ConsoleRailPreferences()
    if not isinstance(raw, Mapping):
        return defaults

    return ConsoleRailPreferences(
        left_open=_coerce_bool(raw.get("left_open"), defaults.left_open),
        right_open=_coerce_bool(raw.get("right_open"), defaults.right_open),
        session_open=_coerce_bool(raw.get("session_open"), defaults.session_open),
        model_open=_coerce_bool(raw.get("model_open"), defaults.model_open),
        details_open=_coerce_bool(raw.get("details_open"), defaults.details_open),
        agent_open=_coerce_bool(raw.get("agent_open"), defaults.agent_open),
    )


def serialize_console_rail_preferences(
    preferences: ConsoleRailPreferences,
) -> dict[str, bool]:
    """Serialize Console rail preferences to the persistence shape.

    Args:
        preferences: Rail preferences to serialize.

    Returns:
        Persistence dict with the left/right rail flags and the four
        left-rail section flags (task-400 dropped ``context_open``).
    """
    return {
        "left_open": bool(preferences.left_open),
        "right_open": bool(preferences.right_open),
        "session_open": bool(preferences.session_open),
        "model_open": bool(preferences.model_open),
        "details_open": bool(preferences.details_open),
        "agent_open": bool(preferences.agent_open),
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


def _normalized_inactive_text(value: Any) -> str:
    return _clean_text(value).lower().rstrip(".")


def _has_active_staged_summary(value: Any) -> bool:
    normalized = _normalized_inactive_text(value)
    return bool(normalized) and normalized not in _INACTIVE_STAGED_SUMMARIES


def build_console_context_rail_badge(
    *,
    workspace_label: Any = "",
    session_label: Any = "",
) -> str:
    """Build the left rail badge from workspace/session state.

    Task-400: staged-context signals moved to the Inspector rail badge along
    with the staged-sources section itself, so the left badge summarizes only
    what the left rail actually holds (workspace + session context).

    Args:
        workspace_label: Active workspace display label; default/fallback
            labels are treated as no workspace.
        session_label: Active session title, when any.

    Returns:
        ``"workspace"``, ``"session"``, or ``""`` when neither applies.
    """
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


def _has_setup_blocker_row(rows: tuple[Any, ...]) -> bool:
    for row in rows:
        label, status, value, text = _row_text_parts(row)
        category = label.lower()
        if not any(term in category for term in _SETUP_BLOCKER_LABEL_TERMS):
            continue

        readiness = " ".join(part.lower() for part in (status, value, text) if part)
        if _contains_any_term(readiness, _SETUP_BLOCKER_READINESS_TERMS):
            return True

    return False


def _contains_any_term(text: str, terms: set[str]) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", text))
    return bool(tokens & terms)


def _has_row_readiness_match(rows: tuple[Any, ...], category_terms: set[str]) -> bool:
    for row in rows:
        label, status, value, text = _row_text_parts(row)
        category = label.lower()
        readiness = " ".join(part.lower() for part in (status, value, text) if part)
        if not readiness or any(
            term in readiness for term in _NEGATIVE_READINESS_TERMS
        ):
            continue
        if any(term in category for term in category_terms) and _contains_any_term(
            readiness, _POSITIVE_READINESS_TERMS
        ):
            return True
    return False


def build_console_inspector_rail_badge(
    *,
    run_status: Any = None,
    inspector_rows: tuple[Any, ...] = (),
    tool_count: Any = 0,
    approval_count: Any = 0,
    can_save_chatbook: bool = False,
    staged_source_count: Any = 0,
    staged_summary: Any = "",
) -> str:
    """Build the right rail badge from run, review, tool, and staged state.

    Task-400: the staged-sources Context section lives in the Inspector rail,
    so its "N staged"/"staged" badge surfaces here. Action-required signals
    (failed/setup/blocked/approvals/tools) keep precedence; staged context
    outranks the informational artifact/source readiness fallbacks.

    Args:
        run_status: Current Console run status value or enum.
        inspector_rows: Inspector display rows used for keyword matching.
        tool_count: Pending tool-call count.
        approval_count: Pending approval count.
        can_save_chatbook: Whether a Chatbook artifact save is available.
        staged_source_count: Number of staged sources for the next send.
        staged_summary: Staged-context summary line; inactive/legacy
            empty-state copy is ignored.

    Returns:
        The highest-precedence badge string, or ``""`` when nothing applies.
    """
    normalized_run_status = _normalized_status(run_status)
    if normalized_run_status == "failed" or _has_row_match(inspector_rows, {"failed"}):
        return "failed"

    if _has_setup_blocker_row(inspector_rows):
        return "setup"

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

    staged_count = _coerce_non_negative_int(staged_source_count)
    if staged_count > 0:
        return f"{staged_count} staged"

    if _has_active_staged_summary(staged_summary):
        return "staged"

    if can_save_chatbook or _has_row_readiness_match(
        inspector_rows,
        {"artifact", "chatbook"},
    ):
        return "artifact"

    if _has_row_readiness_match(inspector_rows, {"source", "rag"}):
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
    """Build effective Console rail state without importing Textual.

    Args:
        preference_key: Persistence key for the active workspace/scope.
        stored_preferences: Raw stored preference payload, if any (legacy
            ``context_open`` keys are ignored; task-400).
        staged_source_count: Staged-source count routed to the Inspector
            rail badge.
        staged_summary: Staged-context summary routed to the Inspector
            rail badge.
        workspace_label: Active workspace display label for the left badge.
        session_label: Active session title for the left badge.
        run_status: Current Console run status for the right badge.
        inspector_rows: Inspector display rows for right-badge matching.
        tool_count: Pending tool-call count for the right badge.
        approval_count: Pending approval count for the right badge.
        can_save_chatbook: Whether a Chatbook artifact save is available.
        available_columns: Current terminal width, when known, for the
            compact right-rail collapse rule.

    Returns:
        Effective rail state combining stored preferences, badges, and the
        responsive right-rail collapse.
    """
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
            workspace_label=workspace_label,
            session_label=session_label,
        ),
        right_badge=build_console_inspector_rail_badge(
            run_status=run_status,
            inspector_rows=inspector_rows,
            tool_count=tool_count,
            approval_count=approval_count,
            can_save_chatbook=can_save_chatbook,
            staged_source_count=staged_source_count,
            staged_summary=staged_summary,
        ),
        persistence_key=preference_key.value,
        right_forced_collapsed=right_forced_collapsed,
        session_open=preferences.session_open,
        model_open=preferences.model_open,
        details_open=preferences.details_open,
        agent_open=preferences.agent_open,
    )
