"""Pure rail-state contracts for the native Console workbench."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import re
from typing import Any

from tldw_chatbook.Chat.console_glyphs import GLYPH_COLLAPSE_LEFT, GLYPH_COLLAPSED

CONSOLE_RAIL_LEFT_DEFAULT_OPEN = True
CONSOLE_RAIL_RIGHT_DEFAULT_OPEN = False
# Task-400: the "context" (staged sources) section moved from the left rail
# into the Inspector rail, so it is no longer a collapsible left-rail section.
CONSOLE_RAIL_SECTION_IDS = (
    "session",
    "workspace",
    "conversations",
    "model",
    "details",
    "agent",
    "character",
)
CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS = 150
# TASK-2154.1/TASK-19639 (formerly TASK-18913): default Context stays open at exactly 100 columns,
# where the framed grid has 96 content columns and resolves as Context 30 +
# main outer 55 + the horizontal Inspector handle 11. The existing main
# min-width waiver permits that one-column yield; below 100, default Context
# is force-collapsed as a rendering override without rewriting preference.
# TASK-2154.2 (LY-11, ADR-043): eligible explicit opens below either compact
# threshold receive the same layout-minimum waiver. ``compact_override`` is
# only that layout authority, never persisted preference or explicit intent.
CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS = 100
# TASK-2154.1 (LY-08/LY-09): below this width even the collapsed-handle
# layout fills the terminal (left handle 13 + main 56 + right handle 11 +
# two borders + two horizontal-padding cells = 84). Below 84 the default
# layout is transcript-only: both handles hide and the main minimum is waived.
# Budget-eligible explicit rails may still render from their 70/74 floors
# through 83 via compact override while the handles remain hidden.
CONSOLE_SINGLE_PANE_COLUMNS = 84
#: task-18911: rail min-widths mirrored from ChatScreen's compose-time
#: styles (left 30, right 34), plus the floor a transcript needs to stay
#: usable. An explicitly-toggled-open rail is honored only while the
#: viewport can afford rail + this floor; below that budget the rendering
#: override wins no matter how the preference was set -- an honored rail at
#: phone width squeezed the transcript to ~14 cols (2026-08-19 mobile
#: audit). The floor is deliberately NOT the single-pane threshold: at
#: 84+ cols a honored rail + waived-min transcript still resolves fine.
CONSOLE_RAIL_LEFT_MIN_COLUMNS = 30
CONSOLE_RAIL_RIGHT_MIN_COLUMNS = 34
CONSOLE_RAIL_MAIN_USABLE_COLUMNS = 40
#: Width band where the Console may automatically reveal Inspector when its
#: standard-width readiness contract is satisfied. Exported so resize
#: deduplication and the UI eligibility check share exact boundaries.
CONSOLE_INSPECTOR_AUTO_OPEN_MIN_COLUMNS = 118
CONSOLE_INSPECTOR_AUTO_OPEN_MAX_COLUMNS = 128
CONSOLE_RAIL_CONTEXT_LABEL = f"Context {GLYPH_COLLAPSED}"
CONSOLE_RAIL_INSPECTOR_LABEL = f"{GLYPH_COLLAPSE_LEFT} Inspector"

#: TASK-2154.2 (ADR-043): payload key marking that ``left_open`` was set by
#: an explicit user toggle rather than riding along in a full-payload
#: serialize. Only the marker lets the narrow left-rail collapse rule yield
#: to explicit opens below 100 cols while keeping the LY-08 default; the
#: right rail needs no marker because its closed default is distinguishable
#: from an explicit ``right_open=True`` by value alone.
CONSOLE_RAIL_LEFT_OPEN_EXPLICIT_KEY = "left_open_explicit"

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
    workspace_open: bool = True
    conversations_open: bool = True
    model_open: bool = True
    details_open: bool = False
    agent_open: bool = False
    character_open: bool = True


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
    left_forced_collapsed: bool = False
    single_pane: bool = False
    # Layout-minimum-waiver authority for a rail rendered open in compact
    # geometry. Below a collapse threshold this covers an honored explicit
    # open; for Context it also covers the effective default open at exactly
    # 100 columns. It does not record persistence or explicit user intent.
    # resolve_console_rail_priority may also grant Inspector this authority
    # after an automatic open.
    right_compact_override: bool = False
    left_compact_override: bool = False
    compact_override: bool = False
    session_open: bool = True
    workspace_open: bool = True
    conversations_open: bool = True
    model_open: bool = True
    details_open: bool = False
    agent_open: bool = False
    character_open: bool = True


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


#: TASK-718: the single per-workspace layout scope. Rail section preferences
#: were previously keyed per workspace+conversation, which multiplied config
#: entries per chat and reset a user's section layout on every new
#: conversation (a toggle made moments earlier was gone after a workspace
#: switch round-trip). Layout is a workspace-level preference.
CONSOLE_RAIL_LAYOUT_SCOPE = "layout"
#: Legacy no-conversation scope kept readable as a one-time migration source.
_LEGACY_GLOBAL_SCOPE = "global"


def build_console_rail_preference_key(
    *,
    workspace_id: Any = None,
    conversation_id: Any = None,
    session_id: Any = None,
) -> ConsoleRailPreferenceKey:
    """Build the deterministic persistence key for Console rail preferences.

    Args:
        workspace_id: Workspace scope value, or global when empty.
        conversation_id: Accepted for API compatibility; no longer shapes the
            key (TASK-718 - preferences are per workspace).
        session_id: Accepted for API compatibility; no longer shapes the key.

    Returns:
        The per-workspace layout key, with the legacy ``:global`` key as the
        read-only migration fallback (adopted by the caller's fallback
        migration when the layout key has never been written).
    """
    del conversation_id, session_id
    workspace_scope = _sanitize_key_part(workspace_id)
    return ConsoleRailPreferenceKey(
        workspace_id=workspace_scope,
        scope_id=CONSOLE_RAIL_LAYOUT_SCOPE,
        value=_build_persistence_key(workspace_scope, CONSOLE_RAIL_LAYOUT_SCOPE),
        fallback_value=_build_persistence_key(workspace_scope, _LEGACY_GLOBAL_SCOPE),
    )


def collect_prunable_console_rail_keys(
    stored_keys: Any,
    *,
    live_scope_ids: Any,
) -> list[str]:
    """Return stored rail-preference keys whose scope is no longer live.

    TASK-718: preferences are per-workspace (``:layout`` scope) with the
    legacy ``:global`` scope kept as the migration source, so every other
    scoped key (per-conversation/per-session entries from the old scheme) is
    stale by definition and safe to delete. Unrecognized key shapes are
    always kept.

    Args:
        stored_keys: Iterable of stored config key strings (non-string
            entries are ignored). ``None`` is treated as empty.
        live_scope_ids: Accepted for API compatibility; conversation/session
            liveness no longer affects prunability.

    Returns:
        The subset of ``stored_keys`` safe to delete, order-preserved.
    """
    del live_scope_ids
    prunable: list[str] = []
    for key in stored_keys or ():
        if not isinstance(key, str):
            continue
        parts = key.split(":")
        if len(parts) != 3 or parts[0] != _PERSISTENCE_PREFIX:
            continue
        scope_id = parts[2]
        if scope_id in (CONSOLE_RAIL_LAYOUT_SCOPE, _LEGACY_GLOBAL_SCOPE):
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


def console_rail_left_open_explicit(stored_preferences: Any) -> bool:
    """Return whether a stored payload marks ``left_open`` as user-toggled.

    TASK-2154.2 (ADR-042): the marker is written alongside the toggle
    gesture (``ChatScreen._set_console_rail_preference``) and preserved
    across later unrelated writes; every read of left-rail explicitness
    goes through here so the write/read sides can never drift.

    Args:
        stored_preferences: Raw stored preference payload, if any.

    Returns:
        ``True`` only when the payload carries a truthy
        ``CONSOLE_RAIL_LEFT_OPEN_EXPLICIT_KEY``.
    """
    return isinstance(stored_preferences, Mapping) and _coerce_bool(
        stored_preferences.get(CONSOLE_RAIL_LEFT_OPEN_EXPLICIT_KEY), False
    )


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

    session_open = _coerce_bool(raw.get("session_open"), defaults.session_open)
    return ConsoleRailPreferences(
        left_open=_coerce_bool(raw.get("left_open"), defaults.left_open),
        right_open=_coerce_bool(raw.get("right_open"), defaults.right_open),
        session_open=session_open,
        workspace_open=_coerce_bool(raw.get("workspace_open"), session_open),
        conversations_open=_coerce_bool(raw.get("conversations_open"), session_open),
        model_open=_coerce_bool(raw.get("model_open"), defaults.model_open),
        details_open=_coerce_bool(raw.get("details_open"), defaults.details_open),
        agent_open=_coerce_bool(raw.get("agent_open"), defaults.agent_open),
        character_open=_coerce_bool(raw.get("character_open"), defaults.character_open),
    )


def serialize_console_rail_preferences(
    preferences: ConsoleRailPreferences,
) -> dict[str, bool]:
    """Serialize Console rail preferences to the persistence shape.

    Args:
        preferences: Rail preferences to serialize.

    Returns:
        Persistence dict with the left/right rail flags and the seven
        left-rail section flags. TASK-14810 split the former mixed Session
        body into Sessions, Workspaces, and Conversations while preserving
        the existing section-persistence model.
    """
    return {
        "left_open": bool(preferences.left_open),
        "right_open": bool(preferences.right_open),
        "session_open": bool(preferences.session_open),
        "workspace_open": bool(preferences.workspace_open),
        "conversations_open": bool(preferences.conversations_open),
        "model_open": bool(preferences.model_open),
        "details_open": bool(preferences.details_open),
        "agent_open": bool(preferences.agent_open),
        "character_open": bool(preferences.character_open),
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


def _inspector_priority_width(available_columns: int | None) -> bool:
    return (
        available_columns is not None
        and CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS
        <= available_columns
        < CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS
    )


def resolve_console_rail_priority(
    rail_state: ConsoleRailState,
    available_columns: int | None,
) -> ConsoleRailState:
    """Give Inspector effective priority when both compact rails are open.

    Args:
        rail_state: Effective rail state before compact priority resolution.
        available_columns: Current terminal width, when known.

    Returns:
        The original state outside the priority conflict, or an immutable copy
        with Context collapsed and Inspector granted compact override authority.
    """
    if not (
        _inspector_priority_width(available_columns)
        and rail_state.left_open
        and rail_state.right_open
    ):
        return rail_state
    return replace(
        rail_state,
        left_open=False,
        left_compact_override=False,
        right_compact_override=True,
        compact_override=True,
    )


def console_context_reveal_preferences(
    rail_state: ConsoleRailState,
    available_columns: int | None,
) -> dict[str, bool]:
    """Return minimal preference updates needed to reveal Context.

    Args:
        rail_state: Current effective rail state.
        available_columns: Current terminal width, when known.

    Returns:
        Context's open preference and, only during an effective compact
        Inspector conflict, Inspector's closed preference.
    """
    changes = {"left_open": True}
    if _inspector_priority_width(available_columns) and rail_state.right_open:
        changes["right_open"] = False
    return changes


def console_rail_width_band(available_columns: int | None) -> str:
    """Bucket a terminal width into the Console workspace layout band.

    TASK-2154.1: the resize hook rebuilds rail state only when the band
    actually changes, so the bucketing lives here next to the thresholds.

    Args:
        available_columns: Current terminal width, when known.

    Returns:
        A stable resize-deduplication key separating the single-pane, narrow,
        compact auto-open boundary, and standard-width bands.
    """
    if available_columns is None:
        return "standard"
    if available_columns < CONSOLE_SINGLE_PANE_COLUMNS:
        return "single-pane"
    if available_columns < CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS:
        return "narrow"
    if available_columns == CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS:
        return "exact-left-boundary"
    if available_columns < CONSOLE_INSPECTOR_AUTO_OPEN_MIN_COLUMNS:
        return "compact-before-auto-open"
    if available_columns < CONSOLE_INSPECTOR_AUTO_OPEN_MAX_COLUMNS + 1:
        return "compact-auto-open"
    if available_columns < CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS:
        return "compact-after-auto-open"
    return "standard"


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
            ``context_open`` keys are ignored; task-400). Beyond the coerced
            values, the ``left_open_explicit`` marker key matters: it marks
            an explicit user toggle of the left rail, which the narrow
            left-rail collapse rule honors (TASK-2154.2).
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
            compact right-rail collapse rule, the narrow left-rail collapse
            rule, and the single-pane fallback.

    Returns:
        Effective rail state combining stored preferences, badges, and the
        responsive rail-collapse/single-pane rules. The collapse rules are
        the default rendering only: ``*_compact_override`` grants the layout
        minimum waiver needed by honored explicit opens below a threshold
        and by effective Context openness at exactly 100 columns; it does not
        mark persisted preference or explicit intent.
    """
    preferences = coerce_console_rail_preferences(stored_preferences)
    # TASK-2154.2 (LY-11, ADR-043): the compact-collapse rules below are the
    # responsive default. Explicit opens are honored while the 70/74-column
    # usable-transcript budgets permit, and receive the layout-minimum waiver;
    # below those floors the rendering override wins without rewriting the
    # stored preference. The two rails detect "explicit" differently because
    # their defaults differ:
    # - Right (default closed): value-based. Default AND explicitly-stored
    #   ``right_open=False`` both keep the collapse, so the rendering AND
    #   the pending-launch auto-open suppression below the threshold are
    #   byte-identical to the pre-2154.2 behavior; only an explicit
    #   ``right_open=True`` yields.
    # - Left (default open): marker-based. The coerced value cannot tell
    #   "never toggled" (must keep the LY-08 force-collapse) apart from
    #   "explicitly opened below the threshold" (must be honored), because
    #   both coerce to ``left_open=True`` -- and plain key-presence in the
    #   stored mapping is useless because every write serializes the FULL
    #   payload, so any toggle would mark ``left_open`` as present. Only a
    #   dedicated marker written alongside the toggle gesture itself
    #   (``CONSOLE_RAIL_LEFT_OPEN_EXPLICIT_KEY``, set by
    #   ``ChatScreen._set_console_rail_preference``) records it. Legacy
    #   payloads lack the marker and keep the force-collapse default.
    explicit_left_open = console_rail_left_open_explicit(stored_preferences)
    # task-18911: an explicit toggle is honored only while the viewport can
    # afford rail + a usable transcript (rail min + main floor). Below that
    # budget the collapse is a rendering override the explicit marker
    # cannot buy its way past -- the stored preference is untouched, so
    # widening back past the budget restores the explicit rail.
    left_width_budget = (
        CONSOLE_RAIL_LEFT_MIN_COLUMNS + CONSOLE_RAIL_MAIN_USABLE_COLUMNS
    )
    right_width_budget = (
        CONSOLE_RAIL_RIGHT_MIN_COLUMNS + CONSOLE_RAIL_MAIN_USABLE_COLUMNS
    )
    right_forced_collapsed = (
        available_columns is not None
        and available_columns < CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS
        and (
            not preferences.right_open
            or available_columns < right_width_budget
        )
    )
    left_forced_collapsed = (
        available_columns is not None
        and available_columns < CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS
        and (
            not explicit_left_open
            or available_columns < left_width_budget
        )
    )
    single_pane = (
        available_columns is not None
        and available_columns < CONSOLE_SINGLE_PANE_COLUMNS
    )
    left_open = False if left_forced_collapsed else preferences.left_open
    right_open = False if right_forced_collapsed else preferences.right_open
    right_compact_override = (
        available_columns is not None
        and available_columns < CONSOLE_RAIL_RIGHT_COMPACT_COLLAPSE_COLUMNS
        and right_open
    )
    left_compact_override = (
        available_columns is not None
        and available_columns <= CONSOLE_RAIL_LEFT_COMPACT_COLLAPSE_COLUMNS
        and left_open
    )

    return ConsoleRailState(
        left_open=left_open,
        right_open=right_open,
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
        left_forced_collapsed=left_forced_collapsed,
        single_pane=single_pane,
        right_compact_override=right_compact_override,
        left_compact_override=left_compact_override,
        compact_override=right_compact_override or left_compact_override,
        session_open=preferences.session_open,
        workspace_open=preferences.workspace_open,
        conversations_open=preferences.conversations_open,
        model_open=preferences.model_open,
        details_open=preferences.details_open,
        agent_open=preferences.agent_open,
        character_open=preferences.character_open,
    )
