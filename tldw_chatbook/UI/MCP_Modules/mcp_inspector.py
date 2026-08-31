"""MCP Hub inspector: readiness explanation, actions, Advanced escape hatch."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping
from functools import partial
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Label,
    Select,
    Static,
    TextArea,
)

from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config
from tldw_chatbook.Library.library_rag_state import (
    LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX,
    library_rag_all_matches_weak,
)
from tldw_chatbook.Library.library_rag_score_kinds import library_rag_result_score_kind
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.hub_test_execution import ToolTestAdmissionPreview
from tldw_chatbook.MCP.local_control_service import MCPGovernanceDenied
from tldw_chatbook.MCP.local_runtime_delegate import (
    PERMISSION_STATE_UNRESOLVED_CLAUSE,
    RawToolCallRefusedError,
    capitalize_first,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.MCP.readiness import (
    REASON_LABELS,
    STATE_CSS_CLASSES,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    is_off_opt_in,
)
from tldw_chatbook.MCP.redaction import redact_mapping
from tldw_chatbook.MCP.unified_control_plane_service import MCPHubGateDeniedError
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import tool_state_kind
from tldw_chatbook.UI.MCP_Modules.mcp_schema_form import MCPSchemaForm, parse_schema

_TOOL_TEST_TEXT_LIMIT = 480
_TOOL_TEST_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret)"
    r"(\s*[:=]\s*)(\[redacted\]|'[^']*'|\"[^\"]*\"|[^\s,;}\]]+)"
)
_TOOL_TEST_BEARER = re.compile(r"(?i)\bbearer\s+(?:\[redacted\]|[^\s,;}\]]+)")
_TOOL_TEST_KEY_VALUE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")
_TOOL_TEST_PATH_START = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Za-z]:[\\/]"
    r"|\\\\(?:[?.][\\/]|[^\\/\s'\"<>]+[\\/][^\\/\s'\"<>]+)"
    r"|/)"
)
_TOOL_TEST_PATH_TRAILING_PUNCTUATION = ".,;:!?)]}"


def _http_url_end(text: str, path_start: int) -> int | None:
    """Return the end of an HTTP URL whose first slash looked path-like."""
    prefix = text[:path_start].lower()
    if not (prefix.endswith("http:") or prefix.endswith("https:")):
        return None
    match = re.match(r"[^\s'\"<>]+", text[path_start:])
    return path_start + len(match.group(0)) if match is not None else path_start


def _unquoted_tool_test_path_end(candidate: str) -> int:
    """Consume an ambiguous unquoted path until a structural boundary.

    Words after a spaced path cannot reliably be classified as prose or path
    components. Privacy wins that ambiguity: keep redacting until punctuation,
    a structured ``: `` separator, or an HTTP URL. A filename extension is not
    a boundary because it can name a directory. Callers already bound candidates
    at quotes, newlines, and field delimiters.
    """
    diagnostic = re.search(r":(?=\s)", candidate[2:])
    if diagnostic is not None:
        return 2 + diagnostic.start()
    tokens = list(re.finditer(r"\S+", candidate))
    if not tokens:
        return 0
    end = tokens[0].end()
    for token in tokens[1:]:
        token_text = token.group(0)
        unwrapped = token_text.lstrip("([{")
        content = unwrapped.rstrip(_TOOL_TEST_PATH_TRAILING_PUNCTUATION)
        trailing_punctuation = len(content) < len(unwrapped)
        normalized = content.lower()
        if normalized.startswith(("http://", "https://")):
            break
        if not content:
            break
        end = token.end() - (len(unwrapped) - len(content))
        if trailing_punctuation:
            break
    return end


def _redact_tool_test_paths(text: str) -> str:
    """Redact absolute filesystem paths without treating URLs or regexes as paths."""
    parts: list[str] = []
    cursor = 0
    while match := _TOOL_TEST_PATH_START.search(text, cursor):
        start = match.start()
        url_end = _http_url_end(text, start)
        if url_end is not None:
            parts.append(text[cursor:url_end])
            cursor = url_end
            continue
        replacement_start = start
        if text[max(cursor, start - 5) : start].lower() == "file:":
            replacement_start = start - 5
        parts.append(text[cursor:replacement_start])
        quote = text[start - 1] if start and text[start - 1] in {'"', "'"} else None
        if quote is not None:
            closing_quote = text.find(quote, match.end())
            end = closing_quote if closing_quote >= 0 else len(text)
        else:
            hard_end = len(text)
            for marker in "\r\n\t<>\"';":
                marker_at = text.find(marker, match.end())
                if marker_at >= 0:
                    hard_end = min(hard_end, marker_at)
            candidate = text[start:hard_end]
            end = start + _unquoted_tool_test_path_end(candidate)
            while end > start and text[end - 1] in _TOOL_TEST_PATH_TRAILING_PUNCTUATION:
                end -= 1
        parts.append("[path]")
        cursor = max(end, match.end())
    parts.append(text[cursor:])
    return "".join(parts)


def _safe_tool_test_text(value: object, *, limit: int = _TOOL_TEST_TEXT_LIMIT) -> str:
    """Return bounded, secret- and path-free text for Test Tool surfaces."""
    try:
        text = str(value)
    except Exception:
        text = "The service returned an unreadable error."
    text = _TOOL_TEST_SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[redacted]", text
    )
    text = _TOOL_TEST_BEARER.sub("Bearer [redacted]", text)
    text = _TOOL_TEST_KEY_VALUE.sub("[redacted]", text)
    text = _redact_tool_test_paths(text)
    text = text.strip()
    if len(text) > limit:
        text = f"{text[: limit - 1].rstrip()}…"
    return text


def _safe_exception_argument(value: object) -> object:
    """Sanitize nested exception data before its repr escapes path separators."""
    if isinstance(value, Mapping):
        return {
            _safe_tool_test_text(key): _safe_exception_argument(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_safe_exception_argument(item) for item in value)
    if isinstance(value, list):
        return [_safe_exception_argument(item) for item in value]
    if isinstance(value, str):
        return _safe_tool_test_text(value)
    return value


def _safe_exception_text(exc: BaseException) -> str:
    """Return bounded exception text without exposing mapping-shaped arguments."""
    args = getattr(exc, "args", ())
    if not any(isinstance(arg, Mapping) for arg in args):
        return _safe_tool_test_text(exc)
    try:
        safe_args = tuple(
            _safe_exception_argument(redact_mapping(arg))
            if isinstance(arg, Mapping)
            else arg
            for arg in args
        )
        rendered = str(safe_args[0]) if len(safe_args) == 1 else str(safe_args)
        return _safe_tool_test_text(rendered)
    except Exception:
        return "<error redacted>"


def _safe_diagnostic_message(prefix: str, exc: BaseException) -> str:
    """Build one bounded, redacted MCP diagnostic from an exception."""
    return _safe_tool_test_text(f"{prefix}: {_safe_exception_text(exc)}")


# Actions that have first-class UI in every source. Everything else renders
# disabled and points at the Advanced runner below (capability preserved).
_BASE_WIRED_ACTIONS = {
    HubAction.VIEW_DETAILS,
    HubAction.OPEN_TOOL_CATALOG,
    HubAction.OPEN_AUDIT,
}

# Local-profile lifecycle actions (Task 5): wired only for local-source
# snapshots, where MCPWorkbench._start_lifecycle() can actually run them
# against the typed T2 control-plane methods. Server-source servers are
# mutated on the server side (Advanced), not from this pane.
_LIFECYCLE_ACTIONS = {
    HubAction.CONNECT,
    HubAction.VALIDATE,
    HubAction.REFRESH_DISCOVERY,
}

# Task 6: editing a local profile's config (command/args/env) is now wired
# for local-source snapshots -- MCPWorkbench opens the MCPProfileForm
# pre-filled from the catalog record for that profile_id. Server-source
# servers are still edited on the server side (Advanced), not from this pane.
_CONFIG_ACTIONS = {HubAction.EDIT_CONFIG}


def _wired_actions(snapshot: ReadinessSnapshot | None) -> set[HubAction]:
    """Actions this inspector renders enabled for the given snapshot."""
    wired = set(_BASE_WIRED_ACTIONS)
    if snapshot is not None and snapshot.source == "local":
        wired |= _LIFECYCLE_ACTIONS
        wired |= _CONFIG_ACTIONS
    return wired


_ACTION_LABELS: dict[HubAction, str] = {
    HubAction.ADD_SERVER: "Add server",
    HubAction.EDIT_CONFIG: "Edit config",
    HubAction.OPEN_CREDENTIALS: "Open credentials",
    HubAction.CONNECT: "Connect",
    HubAction.REFRESH_DISCOVERY: "Refresh tools",
    HubAction.VALIDATE: "Check readiness",
    HubAction.VIEW_DETAILS: "View details",
    HubAction.OPEN_TOOL_CATALOG: "Open tool catalog",
    HubAction.OPEN_AUDIT: "Open audit",
}

# Tooltips for the actions that have first-class UI (see _wired_actions()).
# Every rendered action button must explain its outcome -- disabled buttons get
# a tooltip below; these cover the wired, enabled ones.
_WIRED_ACTION_TOOLTIPS: dict[HubAction, str] = {
    HubAction.VIEW_DETAILS: "Show this server's detail view in Servers mode.",
    HubAction.OPEN_TOOL_CATALOG: "Switch to Tools mode.",
    HubAction.OPEN_AUDIT: "Switch to Audit mode.",
    HubAction.CONNECT: "Connect to this server and discover its tools.",
    HubAction.VALIDATE: "Test the connection without changing the cached catalog.",
    HubAction.REFRESH_DISCOVERY: "Reconnect and refresh the tool/resource/prompt catalog.",
    HubAction.EDIT_CONFIG: "Edit this profile's command, args, and env.",
    # Task 2 (MCP Hub Phase 6): only ever rendered from a Findings-detail
    # remediation button (`show_finding()` below) -- there is no wired
    # OPEN_CREDENTIALS button in the readiness action list itself yet (see
    # `_wired_actions()`), so this entry exists purely for that reuse.
    HubAction.OPEN_CREDENTIALS: (
        "Open this server in Servers mode — credentials are managed in its config."
    ),
}

# Disabled-button tooltip for a lifecycle action on a server-source snapshot
# (managed server-side, not from this local-lifecycle pane).
_SERVER_MANAGED_TOOLTIP = "Managed on the server — use Advanced."
# I2 (MCP Hub Phase 6 finale, review): OPEN_CREDENTIALS is never wired (see
# `_wired_actions()` -- no credentials editor exists for either source), but
# a LOCAL profile's disabled button has an honest, actionable substitute:
# the Edit-config button right next to it edits the same env placeholders a
# credentials editor would. Server-source OPEN_CREDENTIALS (and everything
# else still unwired) falls through to the generic `_UNAVAILABLE_ACTION_
# TOOLTIP` below instead.
_OPEN_CREDENTIALS_LOCAL_TOOLTIP = "Edit the profile's env placeholders via Edit config."
# Disabled-button tooltip for every other still-unwired action. Deliberately
# makes no phase promise and points at no hidden pane -- the program-close
# decision (MCP Hub Phase 6) retired the "later phase" framing this used to
# carry; Advanced remains reachable on its own merits, not as this button's
# consolation prize.
_UNAVAILABLE_ACTION_TOOLTIP = "Not available from this panel."

_TEST_RUN_TOOLTIP = "Send these arguments to the tool and show the result."
_TEST_PREPARING_TEXT = "Preparing a current permission preview…"
_TEST_ASK_TEXT = "Approves this one invocation only. The approval does not persist."
_TEST_OFF_TEXT = "Blocked by Permissions. Change this tool from Off to run it."
_TEST_UNAVAILABLE_TEXT = "Unavailable. Try again, or review this tool in Permissions."

# Task 7: permission-explanation copy (spec-verbatim, binding) -- rendered
# into `#mcp-inspector-permission` by `_render_permission_container()`,
# shared by `show_tool()`'s `effective` keyword (Tools-mode selections
# append the block below the tool detail) and the standalone
# `show_permission()` (Permissions-mode matrix tool-row selections).
_ORIGIN_SENTENCES: dict[str, str] = {
    "tool_override": "From this tool's override.",
    "server_default": "Inherited from the server default.",
    "global_default": "Inherited from the global default.",
    "builtin_default": "Built-in tools default to allow.",
}
# Honest fallback when the service cannot explain a permission origin.
_UNKNOWN_ORIGIN_SENTENCE = f"{capitalize_first(PERMISSION_STATE_UNRESOLVED_CLAUSE)}."
_CONFIG_CHANGED_NOTICE = "Definition changed since you allowed it."
_RISK_FLOORED_NOTICE = (
    "High-risk tool — asks even though the inherited default is Allow."
)
_REALLOW_TOOLTIP = "Store the new definition hash and allow again."

# Task 3 (MCP Hub Phase 6): cascade provenance -- `show_permission()`'s
# `cascade` tuple, when given, replaces the single `_ORIGIN_SENTENCES`
# sentence above with three rungs (tool override / server default / global
# default), so a user can see the WHOLE precedence chain at once instead of
# just where the winning value came from. `_GOTO_PERMISSION_TOOLTIP` is
# shared by both "Change in Permissions" buttons below (Task 3) -- one
# copy, so the two call sites can never drift.
_GOTO_PERMISSION_TOOLTIP = "Switch to Permissions mode and select this tool's row."

# Task 6 (PR-T3), Route B: every other Advanced action reads or mutates
# control-plane config; `tool.execute` EXECUTES a tool -- and used to do it on
# a single press with no permission gate and no execution-log record.
# `UnifiedMCPControlPlaneService.execute_advanced_tool()` now enforces the
# gate's hard "Off" verdict and records the run; this pane supplies the
# per-run consent that gate's "ask" verdict requires, keyed to
# the exact payload it was shown for so an edited payload re-confirms.
_ADVANCED_EXECUTE_ACTION = "tool.execute"
# Fix Round C, Item 4: "Editing anything cancels" undersold what actually
# cancels the arm -- switching the object Advanced is showing
# (`set_service_context()`) or the section (`_load_advanced_section()`)
# disarms too, invisibly, same as an edit. Named all three, still short.
#
# Fix Round E, Item 2 (review of Fix Round C): that enumeration was STILL
# incomplete -- switching the ACTION (`on_select_changed()`) disarms too
# (`_run_advanced_action()`'s own docstring already named it: "switching
# action or editing the payload re-arms"), and hiding the panel
# (`_hide_advanced()`) disarms as well; neither was named. An enumeration
# that omits a real trigger reads as complete and is not, and a fourth or
# fifth one can always surface later. The house already has the correct
# formulation 46 lines above (`_TEST_RUN_ARMED_HINT`: "anything else
# cancels") -- adopted here instead of maintaining a list that cannot stay
# complete.
#
# Fix Round G, Items 1-2 (review of Fix Round E): "anything else cancels"
# was STILL not true -- collapsing this disclosure's own triangle
# (`_on_advanced_collapsible_toggled()`) and editing the payload
# (`#mcp-adv-payload`, no `TextArea.Changed` handler existed at all) both
# left a live arm untouched. Rather than extend the enumeration this
# comment already rejected once, both are now wired to disarm
# (`_on_advanced_collapsible_toggled()`, `_on_advanced_payload_changed()`),
# closing the two remaining gaps between what this sentence claims and
# what the code does.
#
_ADVANCED_EXECUTE_CONFIRM = (
    "Runs {tool} now — press Run Action again to confirm; anything else cancels."
)
# The refusal heading `show_tool_result()` gives a blocked test run (":2254"),
# reused verbatim so a refusal reads the same wherever it surfaces -- a
# refusal is not a failure, and must never render as "Action failed:".
_ADVANCED_BLOCKED_HEADING = "Blocked · not run"

# F-054: the nothing-selected header copy, shared by compose() and
# update_readiness(None) so the two can never drift. Contextual instead of
# a bare "Select an item to inspect." -- it says what picking a row gets
# you. Mode-agnostic on purpose: this pane serves Servers, Tools,
# Permissions, and Audit selections alike.
_EMPTY_STATE_COPY = (
    "Pick a server, tool, or entry to see what's wrong and what you can do."
)

# Task 3 (PR-T3): "a run that ran always says something" -- a completed
# tool test that can't be RENDERED (its panel moved on, or was closed)
# must still be heard about via a toast, even though `show_tool_result()`'s
# render itself stays dropped (I1; the protected stale-drop tests pin that
# silence -- a toast is a different surface).


def _toast(text: str) -> str:
    """Escape a `notify()`-bound message before Rich's markup interpreter
    sees it.

    A small, separate copy of `mcp_workbench.py`'s own `_toast()` --
    `mcp_workbench.py` already imports FROM this module (`_ORIGIN_
    SENTENCES`, `MCPInspector`); importing back the other way would create
    the exact import cycle PR-T2 shipped a real regression from.
    """
    return escape_markup(text)


def _stale_result_toast_text(tool_name: str) -> str:
    """Toast copy for a Test Tool result that arrived but isn't shown --
    covers BOTH `show_tool_result()` silent-drop guards (a different tool
    now selected/nothing selected, or the same tool's panel was closed) --
    deliberately reason-agnostic (this function only ever sees `tool_name`,
    not WHY the render was dropped), so it never asserts a specific cause
    it can't verify.
    """
    return _safe_tool_test_text(
        f"{tool_name} finished running, but its result isn't shown here."
    )


def _cascade_rungs(
    cascade: tuple[str | None, str | None, str],
    effective: EffectiveToolState | None = None,
) -> list[Static]:
    """Build the three provenance-rung Statics for `show_permission()`'s
    `cascade` tuple: `(tool_entry_state, server_default, global_default)`,
    the raw STORE values straight off the permission-store payload for one
    tool -- the SAME values `MCPWorkbench._build_permission_rows()` already
    derives per tool (`PermRow.cycle_current`) and per server
    (`server_cycle_current`), just packaged as one tuple per tool instead of
    split across matrix rows.

    Precedence mirrors `resolve_effective_state()`: the first non-`None`
    rung, tool -> server -> global, is the WINNING rung -- prefixed `▸ ` and
    colored by its resolved verdict (`tool_state_kind()`, the T1 kind->class
    helper, via the existing `.mcp-status-{ready|warning|error}` classes);
    the other two rungs are dimmed (`.mcp-status-muted`, defined in this
    widget's own `BUNDLED_CSS` below -- no such class existed in the shared
    bundle yet). `global_default` is never `None` (a permission store always
    resolves SOME global default), so a winner always exists.

    A set rung's value carries a trailing `" •"` marker -- whether or not it
    wins -- the same "an explicit override is set at this level" convention
    `_build_permission_rows()` already uses for a server row's own label; an
    unset rung renders `"—"`. The global rung never carries the marker (it
    has no parent level to override).

    Critical review fix (MCP Hub Phase 6): the winning rung used to color
    and label itself from a SYNTHETIC `EffectiveToolState` built straight
    off the raw, pre-downgrade cascade value -- a rug-pulled tool (stale
    `definition_hash`) or a risk-floored inherited `allow` rendered
    "▸ Tool override: Allow •" READY-green even though the REAL resolved
    `effective.state` (passed in here, the same value `resolve_
    effective_state()` already downgraded to `"ask"`) reads "Ask" one line
    above. `effective`, when given (every `show_permission()` call site
    now passes its own already-resolved state), overrides the winning
    rung's color to the downgraded WARNING kind and appends the same
    `⚠`/`⚑` marker `format_tool_state_label()` bakes into the matrix's own
    State column -- replacing, not stacking on top of, the plain `" •"`
    override marker, mirroring that helper's own marker precedence. The
    raw stored label itself (e.g. "Allow") is kept as-is: this block's
    whole purpose is showing what is actually STORED at each level, not
    the resolved verdict the origin sentence already stated. `None`
    (the default -- direct/legacy callers with no `EffectiveToolState` to
    hand) skips this and falls back to the plain `tool_state_kind()` color,
    exactly the pre-fix behavior.
    """
    tool_state, server_state, global_state = cascade
    if tool_state is not None:
        winner = "tool"
    elif server_state is not None:
        winner = "server"
    else:
        winner = "global"
    downgraded = effective is not None and (
        effective.config_changed or effective.risk_floored
    )
    downgrade_marker = (
        "⚠" if (effective is not None and effective.config_changed) else "⚑"
    )
    rungs = (
        ("tool", "Tool override", tool_state, "tool_override"),
        ("server", "Server default", server_state, "server_default"),
        ("global", "Global default", global_state, "global_default"),
    )
    widgets: list[Static] = []
    for key, label, state, origin in rungs:
        is_winner = key == winner
        if state is None:
            value_text = "—"
        elif key == "global":
            # The global rung has no parent level to override -- never
            # marked, even though it always carries a concrete state.
            value_text = EffectiveToolState(state=state, origin=origin).ui_label
        elif is_winner and downgraded:
            # Marker replaces the bare override bullet -- same precedence
            # as `format_tool_state_label()`'s own config_changed/
            # risk_floored branches ahead of its plain-override one.
            value_text = f"{EffectiveToolState(state=state, origin=origin).ui_label} {downgrade_marker}"
        else:
            value_text = f"{EffectiveToolState(state=state, origin=origin).ui_label} •"
        prefix = "▸ " if is_winner else ""
        classes = "ds-field-row"
        if is_winner:
            assert state is not None, "the winning rung always has a concrete state"
            kind = (
                "warning"
                if downgraded
                else tool_state_kind(EffectiveToolState(state=state, origin=origin))
            )
            classes += f" mcp-status-{kind}"
        else:
            classes += " mcp-status-muted"
        widgets.append(
            Static(
                f"{prefix}{label}: {value_text}",
                id=f"mcp-inspector-permission-cascade-{key}",
                classes=classes,
                markup=False,
            )
        )
    return widgets


def format_duration_ms(duration_ms: int) -> str:
    """Format a duration in milliseconds for a status line or detail dump.

    Mirrors `library_ingest_state._format_elapsed()`'s granularity at the
    minute tier (integer minutes/seconds, "{m}m {s}s") but adds a finer,
    millisecond-aware tier below it -- a Test Tool run or a Hub tool
    execution is typically sub-second, where a bare "0s" would say nothing
    useful:
      - under 1000ms: "{n}ms"
      - under 60s: "{s:.1f}s" (one decimal)
      - 60s or more: "{m}m {s}s" (integer minutes/seconds)

    Module-level and public (T7, MCP Hub Phase 5) -- was `_format_duration_
    ms`, private to this module and used only by `show_tool_result()`'s
    status line below. `mcp_audit_mode.py`'s Duration column and
    `show_audit_entry()`'s pretty-printed detail (both this same module's
    Audit-mode support) need the identical formatting, so this is now the
    shared home rather than a duplicate copy -- `mcp_audit_mode.py` has no
    dependents of its own to make the natural home instead, mirroring the
    rationale `mcp_permissions_mode.format_tool_state_label()` documents
    for the same "no dependents -> natural shared home" choice.
    """
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    total_seconds = duration_ms / 1000
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    minutes, seconds = divmod(int(round(total_seconds)), 60)
    return f"{minutes}m {seconds}s"


def _duration_segment(duration_ms: float | None) -> str:
    """`" · <formatted>"` when `duration_ms` is known, `""` when it isn't.

    RAG-51 (PR-5 task 5) fix: `show_tool_result()`'s failed and legacy-text
    branches used to call `format_duration_ms(duration_ms)` unconditionally
    -- reachable with `duration_ms=None` now that the keyword defaults to
    `None` (RAG-49), which would `TypeError` inside `format_duration_ms()`'s
    own `duration_ms < 1000` comparison. Mirrors `_summarize_tool_result()`'s
    own `if duration_ms is not None` guard for the structured shape's
    segments list, just packaged for the `"Failed{seg}"`/`"OK{seg}"` prefix
    style those two branches use instead of a segments list.
    `format_duration_ms()` itself stays int-only (its only other caller,
    `mcp_audit_mode.py`'s Duration column, always has a real duration) --
    smaller to guard the two call sites than to widen that contract.
    """
    if duration_ms is None:
        return ""
    return f" · {format_duration_ms(duration_ms)}"


# RAG-49 (PR-5 task 4): the "Raw response" Collapsible's body is capped so a
# large tool result never blows out the Test Tool panel's layout -- the old
# 500-char cap (a flattened single-line excerpt) is retired on this path in
# favor of a much larger, still-bounded pretty-printed dump.
_RAW_BODY_CHAR_CAP = 20_000


def _format_raw_body(raw: str) -> str:
    """Cap a pretty-printed raw-response string at `_RAW_BODY_CHAR_CAP`
    chars, appending an honest truncation note when it had to cut.

    Pure and independently unit-testable (exercised end-to-end via
    `show_tool_result()`'s own raw-body rendering below).
    """
    if len(raw) <= _RAW_BODY_CHAR_CAP:
        return raw
    total = len(raw)
    return (
        raw[:_RAW_BODY_CHAR_CAP]
        + f"\n… truncated (showing {_RAW_BODY_CHAR_CAP} of {total} chars)"
    )


def _is_tool_error_shape(result: object) -> bool:
    """Whether `result` is the MCP/tools.py:326 tool-returned-error shape:
    a list of exactly one element, itself a mapping whose only key is
    `"error"`.

    Distinct from an infrastructure-level failure (`ok=False`): the HUB
    CALL succeeded here, but the TOOL's own logical result reports an
    error. A mapping with an `"error"` key ALONGSIDE other keys, or a list
    of more than one element, does not match -- only the exact single-key
    shape does.
    """
    if not isinstance(result, list) or len(result) != 1:
        return False
    item = result[0]
    if not isinstance(item, Mapping):
        return False
    return list(item.keys()) == ["error"]


class _ScoredRow:
    """Minimal score-provenance shim for the shared weak-match predicate.

    `library_rag_all_matches_weak()` is typed for `LibraryRagResultRow`
    but at runtime reads score provenance through duck typing rather than a
    hard dependency on the Library dataclass. This lets MCP tool result rows
    (plain `Mapping`s) feed the same, canonical all-weak check the Library
    evidence list uses without copying its logic.
    """

    __slots__ = ("score", "score_kind", "vector_score")

    def __init__(
        self, score: object, score_kind: str, vector_score: float | None
    ) -> None:
        # Defensive coercion: anything that isn't a real number (or is a
        # bool -- `isinstance(True, int)` is True in Python) is treated
        # as unscored rather than risking a `<` comparison against a
        # non-numeric value inside `library_rag_all_matches_weak()`.
        try:
            self.score = (
                score
                if (
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                )
                else None
            )
        except OverflowError:
            self.score = None
        self.score_kind = score_kind
        self.vector_score = vector_score


def _extract_scored_rows(rows: list) -> list[_ScoredRow] | None:
    """Whether `rows` is shaped like a scored result list (RAG-search's
    row shape), and if so, the `.score`-bearing shim rows for
    `library_rag_all_matches_weak()`.

    Global Constraints: "Band vocabulary is gated on result shape." --
    `_summarize_tool_result()` is generic across every MCP tool (a
    `list_characters` result must never grow match-band vocabulary), so
    this only recognizes a result as scored when EVERY row is a mapping
    carrying a `"score"` key (value may legitimately be `None` --
    keyword-mode rows post PR-T3 task 1). One row missing the key at all
    (a different tool's row shape) means "not a scored result" -- no
    signal is fabricated from a shape this wasn't designed for.

    Returns:
        `None` when `rows` isn't uniformly scored-shaped (including the
        empty-list case, handled separately by the caller); otherwise
        score-provenance shim list.
    """
    if not rows:
        return None
    scored_rows: list[_ScoredRow] = []
    for row in rows:
        if not isinstance(row, Mapping) or "score" not in row:
            return None
        score_kind, vector_score = library_rag_result_score_kind(
            row.get("metadata"), row
        )
        scored_rows.append(_ScoredRow(row.get("score"), score_kind, vector_score))
    return scored_rows


def _summarize_tool_result(
    *, ok: bool, duration_ms: float | None, source: str | None, result: object
) -> tuple[str, str | None]:
    """Build the Test Tool result's status-line segments and (optional)
    quiet interpretation line, from the structured pieces `show_tool_
    result()` was given -- pure and unit-testable without any UI harness.

    Returns:
        `(status_line, interpretation)` -- `interpretation` is `None` when
        there is nothing further to say (a non-list result, or a
        non-empty, non-error-shaped list whose rows aren't uniformly
        scored, or whose scores aren't all weak -- the count segment
        alone is the whole story there).
    """
    segments = ["OK" if ok else "Failed"]
    if source:
        segments.append(str(source))
    if duration_ms is not None:
        segments.append(format_duration_ms(duration_ms))
    interpretation: str | None = None
    if isinstance(result, list):
        if _is_tool_error_shape(result):
            segments.append("tool returned an error")
            interpretation = str(result[0]["error"])
        elif not result:
            segments.append("0 results")
            interpretation = "The tool ran and returned no results."
        else:
            count = len(result)
            segments.append(f"{count} result" + ("s" if count != 1 else ""))
            # F1: a result that "found" nothing useful must say so instead
            # of reporting a bare `OK · N results` that implies success.
            # Only rows carrying a numeric score participate -- an
            # unrelated tool's rows (no "score" key at all) are left
            # exactly as they render today.
            scored_rows = _extract_scored_rows(result)
            if scored_rows is not None and library_rag_all_matches_weak(scored_rows):
                interpretation = LIBRARY_RAG_ALL_WEAK_COVERAGE_PREFIX
    return " · ".join(segments), interpretation


def audit_entry_detail_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Project one execution-log entry into its metadata-only display schema.

    Args:
        entry: Raw execution-log fields to normalize for the inspector.

    Returns:
        A metadata-only payload safe for the execution-detail display.
    """

    server_key = str(entry.get("server_key") or "")
    tool_name = str(entry.get("tool_name") or "")
    return {
        "ts": entry.get("ts"),
        "tool": f"{server_key}::{tool_name}",
        "initiator": entry.get("initiator"),
        "decision": entry.get("decision"),
        "ok": entry.get("ok"),
        "status": entry.get("status"),
        "duration": format_duration_ms(int(entry.get("duration_ms") or 0)),
        "error_category": entry.get("error_category"),
        "exception_type": entry.get("exception_type"),
        "status_code": entry.get("status_code"),
        "argument_names": entry.get("argument_names") or [],
        "unknown_argument_count": int(entry.get("unknown_argument_count") or 0),
        "result_type": entry.get("result_type"),
        "result_size": int(entry.get("result_size") or 0),
    }


def _finding_text(finding: Mapping[str, Any], *keys: str) -> str:
    """Defensive raw-dict read for one finding field (T8, MCP Hub
    Phase 5) -- mirrors `hub_tool_catalog.server_tools_from_inventory()`'s
    own tolerant-of-missing-keys style: a finding comes straight off the
    wire (a server-side product, versioned independently), so every field
    is optional. Tries each key in order and returns the first non-blank
    value found, `"—"` when none match -- `mcp_audit_mode.py`'s own
    `_finding_field()` does the identical single-key version for the
    Findings table; this module accepts multiple key aliases since the
    exact remediation field name isn't pinned down by the wire contract
    yet (see `_finding_remediation()` below).
    """
    for key in keys:
        value = finding.get(key)
        if value not in (None, ""):
            return str(value)
    return "—"


def _finding_remediation(finding: Mapping[str, Any]) -> str | None:
    """The finding's suggested-remediation text, or `None` when absent.

    Unlike `_finding_text()`'s columns (always rendered, `"—"` fallback),
    remediation is shown only "when present" per the spec (task-8-brief.md)
    -- an absent remediation is not an error, most findings simply won't
    carry one. Two key aliases are checked defensively (`"remediation"`
    and `"suggested_remediation"`) since the real wire field name isn't
    pinned down by any client/schema in this codebase yet.
    """
    for key in ("remediation", "suggested_remediation"):
        value = finding.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _is_blank(value: Any) -> bool:
    """Whether a Select value means "nothing selected".

    NOTE: `Select.BLANK` is not a real Select sentinel in this Textual
    version (8.2.7) - it resolves to `Widget.BLANK` (`False`) via MRO,
    distinct from the actual blank marker `Select.NULL`. We use
    `Select.BLANK` as the value of our own synthetic placeholder option (so
    its custom label isn't replaced by the dim default prompt text), but
    `set_options()` can reset a Select's value to `Select.NULL` (the real
    no-selection sentinel used when `allow_blank=True`), so both must be
    treated as "no selection" here. See mcp_rail.py for the precedent.
    """
    return value is Select.BLANK or value is Select.NULL


def _render_section_payload(section: str, payload: Any) -> str:
    """Render one Advanced-pane section's raw payload for `#mcp-adv-content`.

    Task 5 (MCP Hub Phase 6): replaces the import of `unified_mcp_sections.
    render_unified_mcp_section()` -- that module (and its one bespoke text
    formatter per section: overview/inventory/external_servers/governance/
    advanced) is deleted along with the legacy `UnifiedMCPPanel` it was
    written for. Advanced is an opt-in escape hatch now (see the reveal
    button below), not the primary view a user is expected to read prose
    summaries in -- a single generic JSON dump of whatever the service
    returned is enough, and doesn't require maintaining a second per-section
    renderer in lockstep with the service's payload shapes.

    `#mcp-adv-content` is always mounted with `markup=False` (both at
    initial `compose()` and in `_build_advanced_collapsible()`'s reveal-time
    build below), so this function only needs to produce a plain string --
    it does not itself need to escape Rich markup.

    `section` is accepted but not otherwise used -- kept in the signature
    so a future caller can prefix a section header without changing every
    call site again. `default=str` covers any payload value that isn't
    natively JSON-serializable (an enum, a raw exception, ...); the broad
    `Exception` fallback covers anything `json.dumps()` can raise on a
    payload this code doesn't control -- not just `TypeError` (a raw
    non-Mapping object), but also e.g. `ValueError` (a circular reference)
    or `OverflowError` (an out-of-range float) -- which should not happen
    for a service-returned dict but must never crash the Advanced pane
    either way.
    """
    try:
        return json.dumps(payload, indent=2, sort_keys=True, default=str)
    except Exception:
        return str(payload)


class MCPInspector(Vertical):
    """Right-pane inspector: what is selected, why, what can I do."""

    # F-056: Escape closes the Test Tool panel when it's open (same path as
    # its Close button); no-op otherwise.
    BINDINGS = [Binding("escape", "close_test_panel", "Close test panel", show=False)]

    BUNDLED_CSS = """
    MCPInspector {
        width: 3fr;
        min-width: 28;
        height: 100%;
        min-height: 0;
    }
    /* F-054: let the empty-state/badge line WRAP at narrow widths instead
    of clipping mid-word -- the shared `.ds-status-badge` rule pins
    `height: 1`. This override covers the bare test harnesses that never
    load the app bundle; the REAL app gets the identical rule from
    _agentic_terminal.tcss (app-tier CSS beats widget DEFAULT_CSS on ties
    in this Textual version, so the bundle carries its own copy -- the
    established lockstep pattern documented there). */
    #mcp-inspector-state {
        width: 1fr;
        height: auto;
        min-height: 1;
    }
    #mcp-inspector-actions {
        /* Vertical defaults to height: 1fr, which would make this empty-by-
        default container greedily claim half the remaining space (splitting
        it with #mcp-adv-scroll below) even with zero or few action buttons
        mounted. Size it to its actual content instead. */
        height: auto;
        min-height: 0;
    }
    /* T6: the tool-detail container, shown above the Advanced collapsible
    when a Tools-mode row is selected. Hidden by default (display: none) --
    show_tool()'s own display toggle is what reveals it, so a fresh mount
    (or a selection cleared back to None) never shows an empty box. */
    #mcp-inspector-tool {
        height: auto;
        min-height: 0;
        display: none;
    }
    /* T7: the permission-explanation container, shared by `show_tool()`'s
    `effective` keyword (mounted below `#mcp-inspector-tool`, Tools-mode
    selections) and the standalone `show_permission()` (Permissions-mode
    matrix tool-row selections). Same hidden-by-default discipline as
    `#mcp-inspector-tool` above -- `_render_permission_container()`'s own
    display toggle is what reveals it. */
    #mcp-inspector-permission {
        height: auto;
        min-height: 0;
        display: none;
    }
    /* T7 (MCP Hub Phase 5): the audit-entry detail container, populated by
    `show_audit_entry()` -- Audit mode's own row-selection entry point.
    Same hidden-by-default discipline as `#mcp-inspector-tool`/
    `#mcp-inspector-permission` above -- `show_audit_entry()`'s own display
    toggle is what reveals it. */
    #mcp-inspector-audit {
        height: auto;
        min-height: 0;
        display: none;
    }
    /* T8 (MCP Hub Phase 5): the finding-detail container, populated by
    `show_finding()` -- Audit mode's Findings sub-view row-selection entry
    point. Same hidden-by-default discipline as `#mcp-inspector-audit`
    above -- `show_finding()`'s own display toggle is what reveals it. */
    #mcp-inspector-finding {
        height: auto;
        min-height: 0;
        display: none;
    }
    /* Bounded, not `height: 1fr`/`auto` -- the pretty-printed JSON detail
    (arguments/result excerpt included) can run long; a fixed height with
    scroll keeps the rest of the inspector's layout stable regardless of
    payload size. Mirrors `#mcp-adv-payload`'s own bounded-height precedent
    just below. */
    #mcp-inspector-audit-scroll {
        height: 12;
        min-height: 6;
    }
    /* T12: the Advanced block moved from a direct-child VerticalScroll to a
    Collapsible's body. Give the Collapsible itself the 1fr the scroll used
    to claim directly (so it still fills the remaining pane height when
    expanded) and drop back to auto when collapsed (Contents is display:
    none then -- reserving 1fr of empty space below the title bar would
    waste most of the pane).

    Fix Round K (live walkthrough of PR #1385): the T12 comment here used
    to claim the auto-height Contents "mostly falls back to intrinsic
    sizing, but VerticalScroll still scrolls on overflow regardless, so
    nothing breaks -- exact geometry polish is T13's job." REFUTED LIVE:
    with a real section loaded (Inventory's ~200-row JSON), the Action
    select, the Payload editor, and the Run Action button were
    unreachable at ANY terminal height (reproduced at 300 rows; wheel and
    PageDown both bottom out mid-JSON) -- the very hatch this branch
    spent rounds making truthful could not be operated. Tests never saw
    it because every fake service returns a few-line payload. Measured
    with a bundled-CSS probe: `1fr` on this chain resolves without
    subtracting the rows ABOVE the collapsible inside the pane, so
    #mcp-adv-scroll's region hung 3+ rows past the screen bottom and its
    max scroll could never bring the tail rows into view
    (`pilot.click("#mcp-adv-run")` -> OutOfBounds). A `> Contents
    { height: 1fr }` bridge did not even apply (styles.height stayed
    auto). The robust fix drops the fr chain entirely: the collapsible is
    auto-height and the scroll caps itself (`max-height`), so the box
    always fits on screen whole and scrolls its overflow inside a real
    viewport -- probe click goes green, and live the runner is reachable
    again (End key / wheel, then arm-confirm verified). T13 never ran;
    this is that geometry debt, paid where it bit. */
    #mcp-adv-collapsible {
        height: auto;
        min-height: 0;
    }
    #mcp-adv-collapsible.-collapsed {
        height: auto;
    }
    #mcp-adv-scroll {
        height: auto;
        max-height: 24;
        min-height: 0;
    }
    #mcp-adv-payload {
        height: 6;
        min-height: 3;
    }
    Button.mcp-inspector-action {
        width: 100%;
        height: 1;
        min-height: 1;
        border: none;
        /* A3: Button defaults BOTH text-align and content-align to center
        (see Textual's own Button.DEFAULT_CSS -- the same lesson already
        documented on Button.mcp-rail-row in MCPRail.BUNDLED_CSS and
        Button.mcp-callout in _agentic_terminal.tcss) -- without this, the
        action stack (and the lone Cancel button during an in-flight
        lifecycle op) renders each label centered in its full-width row
        instead of left-aligned. */
        text-align: left;
        content-align: left middle;
    }
    /* A2: the generic `Button:disabled` rule (_buttons.tcss) stacks 50%
    opacity on top of `$text-disabled` on a dark surface -- combined with
    `.console-action-secondary`'s own colors that renders as functionally
    invisible, not just "disabled-looking". Win back full opacity and use a
    dim-but-still-readable color instead.
    NOTE: uses the raw `$text-muted`/`$surface` tokens (not the project's
    `$ds-text-muted`/`$ds-surface-raised` aliases) deliberately -- those
    aliases are only defined once the app-wide tcss bundle is loaded, and
    this widget's own unit tests (test_mcp_inspector.py) mount it without
    that bundle. `$ds-text-muted` and `$ds-surface-raised` currently alias
    to exactly these two raw tokens (see css/core/_variables.tcss), so this
    is not a visual compromise. */
    Button.mcp-inspector-action:disabled {
        opacity: 100%;
        background: $surface;
        color: $text-muted;
        text-style: none;
    }
    Button.mcp-inspector-action:disabled:hover {
        opacity: 100%;
        background: $surface;
        color: $text-muted;
        text-style: none;
    }
    /* Task 3 (MCP Hub Phase 6): dims a non-winning cascade-provenance rung
    (`_cascade_rungs()` above) -- the shared bundle
    (`css/tldw_cli_modular.tcss`) defines `.mcp-status-ready/-warning/-error/
    -info` (Task 1) but no `-muted` variant yet. Scoped here rather than
    added to the bundle for the same reason as the disabled-button rule just
    above: this widget's own unit tests (test_mcp_inspector.py) mount it
    without that bundle, and the raw `$text-muted` token (not the `$ds-text-
    muted` alias) is what those two currently resolve to anyway (see
    `css/core/_variables.tcss`) -- not a visual compromise. */
    .mcp-status-muted {
        color: $text-muted;
    }
    /* RAG-49 (PR-5 task 4): the Test Tool result's quiet interpretation
    line (empty/error/unusual-shape explanations) -- dimmed like the
    cascade's non-winning rungs above, distinct from the bold status-line
    Static it sits below. Toggled `display` per-render by show_tool_
    result(); always mounted (never conditionally composed) per the
    program's PR-2 always-mounted-widget lesson. */
    .mcp-inspector-result-note {
        color: $text-muted;
        height: auto;
        min-height: 0;
    }
    /* The "Raw response" Collapsible: always mounted (display: none until
    show_tool_result() has raw content), collapsed by default. Bounded,
    not `1fr`/`auto`-greedy -- mirrors #mcp-adv-collapsible's own T12
    caveat just above: nested inside #mcp-inspector-tool's auto-height
    container here (not a direct MCPInspector child competing for the
    pane's own 1fr budget), so a plain auto height is enough to avoid
    reserving empty space while collapsed. */
    #mcp-inspector-test-result-raw {
        height: auto;
        min-height: 0;
    }
    /* The raw JSON dump itself can run long (up to the 20,000-char cap) --
    a bounded, scrollable region keeps an expanded Collapsible from
    dominating the rest of the inspector pane, mirroring #mcp-inspector-
    audit-scroll's identical bounded-pretty-printed-JSON precedent above. */
    #mcp-inspector-test-result-raw-scroll {
        height: 12;
        min-height: 6;
    }
    """

    class HubActionRequested(Message, namespace="mcp_inspector"):
        def __init__(self, action: HubAction, server_key: str | None) -> None:
            super().__init__()
            self.action = action
            self.server_key = server_key

    class CancelRequested(Message, namespace="mcp_inspector"):
        """Posted when the user clicks Cancel on an in-flight (CHECKING)
        lifecycle operation. `MCPWorkbench` owns the actual worker and
        cancels it -- this pane only knows which server the button belongs
        to."""

        def __init__(self, server_key: str) -> None:
            super().__init__()
            self.server_key = server_key

    class ToolTestRequested(Message, namespace="mcp_inspector"):
        """One click bound to the immutable preview currently rendered."""

        def __init__(
            self,
            server_key: str,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            preview_nonce: str,
            intent: str,
        ) -> None:
            super().__init__()
            self.server_key = server_key
            self.tool_name = tool_name
            self.arguments = arguments
            self.preview_nonce = preview_nonce
            self.intent = intent

    class ToolTestPreviewRequested(Message, namespace="mcp_inspector"):
        """Ask the Workbench to prepare one current service preview."""

        def __init__(self, server_key: str, tool_name: str) -> None:
            super().__init__()
            self.server_key = server_key
            self.tool_name = tool_name

    class ToolTestPreviewRevocationRequested(Message, namespace="mcp_inspector"):
        """Best-effort revocation request for a preview leaving the panel."""

        def __init__(self, preview_nonce: str) -> None:
            super().__init__()
            self.preview_nonce = preview_nonce

    class ReallowRequested(Message, namespace="mcp_inspector"):
        """Posted when the user presses Re-allow on a `config_changed`-
        downgraded tool's permission block (`#mcp-inspector-reallow`, only
        ever mounted for that downgrade -- see
        `_render_permission_container()`). `MCPWorkbench` resolves the
        live `HubTool` and calls `set_tool_state(..., "allow", tool=tool)`
        (T4), which stores the tool's CURRENT definition hash and clears
        the rug-pull downgrade -- then resyncs the Permissions matrix (its
        ⚠ marker clears)."""

        def __init__(self, server_key: str, tool_name: str) -> None:
            super().__init__()
            self.server_key = server_key
            self.tool_name = tool_name

    class AuditOpenToolRequested(Message, namespace="mcp_inspector"):
        """Posted when the user presses "Open tool" (`#mcp-audit-open-tool`)
        on an execution-log entry's detail view (`show_audit_entry()`).
        `MCPWorkbench` resolves `(server_key, tool_name)` against
        `_last_hub_tools` -- a tool that has since dropped out of the
        catalog is a warning toast, never a crash; a resolved tool switches
        to Tools mode and selects its row."""

        def __init__(self, server_key: str, tool_name: str) -> None:
            super().__init__()
            self.server_key = server_key
            self.tool_name = tool_name

    class AuditAdjustPermissionRequested(Message, namespace="mcp_inspector"):
        """Posted when the user presses "Adjust permission"
        (`#mcp-audit-adjust-permission`) on an execution-log entry's detail
        view. Same resolve-then-route contract as `AuditOpenToolRequested`,
        but switches to Permissions mode and moves the matrix cursor to the
        tool's row instead."""

        def __init__(self, server_key: str, tool_name: str) -> None:
            super().__init__()
            self.server_key = server_key
            self.tool_name = tool_name

    class ChangeInPermissionsRequested(Message, namespace="mcp_inspector"):
        """Posted by either "Change in Permissions" button (Task 3, MCP Hub
        Phase 6): the Tools-mode permission block's own button
        (`#mcp-inspector-goto-permission`, `_render_permission_container()`'s
        `show_goto_button` path -- rendered only for `show_tool()`'s
        combined call, never the standalone Permissions-mode
        `show_permission()`) and the Test Tool panel's blocked/ask button
        (`#mcp-inspector-goto-permission-test`, shown by previews and blocked
        outcomes).

        Both route through `MCPWorkbench._goto_permission_row()` -- the SAME
        shared helper the audit drill's `AuditAdjustPermissionRequested`
        already uses: one implementation, three callers, no duplicated
        mode-switch-plus-matrix-row-selection logic."""

        def __init__(self, server_key: str, tool_name: str) -> None:
            super().__init__()
            self.server_key = server_key
            self.tool_name = tool_name

    def __init__(self, **kwargs: Any) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"ds-inspector {classes}".strip(), **kwargs)
        self._snapshot: ReadinessSnapshot | None = None
        self._service: Any = None
        self._sections: list[tuple[str, str]] = [("Overview", "overview")]
        self._action_templates: dict[str, str] = {}
        # T12: the object the Advanced pane's content currently describes --
        # defaults match set_service_context()'s own default source="local"
        # so the label composed here (before any set_service_context() call)
        # agrees with what a fresh mount would show.
        self._advanced_source: str = "local"
        self._advanced_target_label: str | None = None
        # Task 5 (MCP Hub Phase 6): whether the Advanced collapsible has
        # ever been composed/mounted this session -- gates both `compose()`
        # (mount-time: renders the Collapsible when a persisted
        # `mcp.hub_state.advanced_visible` opt-in is True, the reveal
        # Button otherwise) and `set_service_context()` (skips touching
        # `#mcp-adv-*` widgets while they don't exist; see that method).
        # Set for real in `compose()`/`_reveal_advanced()`; False here is
        # just the pre-mount default so an out-of-order `set_service_
        # context()` call (there is none today, but nothing prevents one)
        # degrades to "state recorded, DOM untouched" rather than crashing.
        self._advanced_visible: bool = False
        # T12 review fix: the collapsed state this widget last knew about --
        # set to the constructed value in compose(), updated only by the
        # Toggled handler. Used to drop the spurious mount-time Toggled that
        # `Collapsible(collapsed=False)` posts (see the handler); default
        # True matches Collapsible's own reactive default.
        self._advanced_last_collapsed: bool = True
        # Task 6 (PR-T3): the `(action, payload)` the Advanced runner's Run
        # Action button is currently armed to execute, or None when unarmed.
        # Only the executing descriptor (`tool.execute`) ever sets it -- see
        # `_run_advanced_action()`. Keyed on the payload, not just the
        # action, so an edited payload re-arms instead of running under a
        # confirm the user gave for different arguments.
        self._advanced_confirm_key: tuple[str, str] | None = None
        # Task 4: serializes `update_readiness()`'s remove+mount cycle. Two
        # calls awaited concurrently (a worker-driven refresh interleaved
        # with a pump-driven one) previously could both be mid-flight at
        # once even though each call itself awaits remove/mount in order --
        # this lock ensures the second call's whole body only starts once the
        # first has fully finished, so the last writer's buttons win exactly
        # once instead of racing into `DuplicateIds`.
        self._refresh_lock = asyncio.Lock()
        # T6: the `HubTool` `#mcp-inspector-tool` currently describes, or
        # `None` when hidden. Used by the Test Tool panel handlers below to
        # know which tool a Run press is testing without re-querying the
        # workbench.
        self._current_tool: HubTool | None = None
        # Task 7: the `HubTool` `#mcp-inspector-permission` currently
        # describes, or `None` when hidden -- set by
        # `_render_permission_container()`, the single writer for that
        # container. Read by the Re-allow button's press handler (below) to
        # know which tool's `(server_key, tool_name)` to post in
        # `ReallowRequested` without re-querying the workbench.
        self._current_permission_tool: HubTool | None = None
        # T7 (MCP Hub Phase 5): the raw execution-log entry dict
        # `#mcp-inspector-audit` currently describes, or `None` when
        # hidden -- set by `show_audit_entry()`, the single writer. Read by
        # the "Open tool"/"Adjust permission" button press handlers below
        # to know which `(server_key, tool_name)` to post without
        # re-querying the workbench.
        self._current_audit_entry: dict[str, Any] | None = None
        # T8 (MCP Hub Phase 5): the raw finding dict `#mcp-inspector-
        # finding` currently describes, or `None` when hidden -- set by
        # `show_finding()`, the single writer. No action buttons read this
        # back (unlike `_current_audit_entry` above) -- the finding detail
        # is read-only this phase (no client-side fix actions).
        self._current_finding: dict[str, Any] | None = None
        # Task 2 (MCP Hub Phase 6): the finding's owning server key, as
        # resolved by the caller (`MCPWorkbench.on_mcp_audit_mode_finding_
        # selected()`) and threaded through `show_finding()`'s `server_key`
        # keyword -- read by the finding-detail action buttons' press
        # handler below (`#mcp-finding-action-*`) to know what to post in
        # `HubActionRequested`. `None` when the caller couldn't resolve one
        # (nothing derivable from the finding, nothing selected in the
        # rail).
        self._current_finding_server_key: str | None = None
        # The only execution authority this renderer retains is the immutable,
        # metadata-only preview the service issued for the visible panel.
        self._test_preview: ToolTestAdmissionPreview | None = None

    def _advanced_object_label(self) -> str:
        """Compute the "Showing: <object>" text for `#mcp-adv-object`.

        UX-inputs: label the object the Advanced content describes, so its
        section dumps -- which can legitimately describe a different object
        than the selected server -- never get mistaken for facts about the
        currently-selected row.
        """
        if self._advanced_source == "server":
            target = self._advanced_target_label or "(none selected)"
            return f"Showing: server {target}"
        return "Showing: Local control plane"

    def compose(self) -> ComposeResult:
        yield Static("Inspector", classes="destination-section")
        yield Static(
            _EMPTY_STATE_COPY,
            id="mcp-inspector-state",
            classes="ds-status-badge",
            markup=False,
        )
        yield Static(
            "", id="mcp-inspector-message", classes="ds-field-row", markup=False
        )
        yield Vertical(id="mcp-inspector-actions")
        # T6: tool-detail container, populated by show_tool() -- hidden
        # (display: none, see BUNDLED_CSS) until a Tools-mode row is
        # selected.
        yield Vertical(id="mcp-inspector-tool")
        # T7: permission-explanation container, populated by
        # `_render_permission_container()` (via `show_tool()`'s `effective`
        # keyword or the standalone `show_permission()`) -- hidden (display:
        # none, see BUNDLED_CSS) until a permission context is supplied.
        yield Vertical(id="mcp-inspector-permission")
        # T7 (MCP Hub Phase 5): audit-entry detail container, populated by
        # `show_audit_entry()` -- hidden (display: none, see BUNDLED_CSS)
        # until an Audit-mode row is selected.
        yield Vertical(id="mcp-inspector-audit")
        # T8 (MCP Hub Phase 5): finding-detail container, populated by
        # `show_finding()` -- hidden (display: none, see BUNDLED_CSS) until
        # an Audit-mode Findings-table row is selected.
        yield Vertical(id="mcp-inspector-finding")
        # Task 5 (MCP Hub Phase 6): the Advanced (legacy control plane)
        # runner is opt-in now -- `mcp.hub_state.advanced_visible` (default
        # False) gates whether the Collapsible composes at all. False (the
        # common case, and every fresh install) renders just the toggle
        # Button; pressing it flips the setting and mounts the SAME widget
        # tree this branch would have composed, via `_reveal_advanced()`
        # below. True (a user who has already opted in during a previous
        # session) composes the collapsible immediately, exactly as every
        # phase before this one did.
        # F-053: the toggle is a REVERSIBLE two-way control -- it is always
        # rendered (as "Hide advanced" while the runner is visible) instead
        # of being removed on reveal, and hiding persists
        # `advanced_visible=False`. Whatever the user last explicitly chose
        # is what future visits compose, so neither direction is a one-way
        # door.
        # `get_cli_setting` reads the real user config in a bare test App;
        # tests monkeypatch this module's `get_cli_setting` name for
        # determinism (see test_mcp_inspector.py).
        self._advanced_visible = bool(
            get_cli_setting("mcp.hub_state", "advanced_visible", False)
        )
        yield Button(
            "Hide advanced" if self._advanced_visible else "Advanced…",
            id="mcp-inspector-advanced-reveal",
            classes="console-action-subdued",
            compact=True,
            tooltip=(
                "Hide the legacy control-plane action runner."
                if self._advanced_visible
                else "Show the legacy control-plane action runner."
            ),
        )
        if self._advanced_visible:
            yield self._build_advanced_collapsible()

    def _build_advanced_collapsible(self, *, force_open: bool = False) -> Collapsible:
        """Construct the Advanced (legacy control plane) Collapsible tree.

        Shared by `compose()` (`advanced_visible=True` at mount) and
        `_reveal_advanced()` (the opt-in reveal Button's handler) -- the
        exact same widget tree either way, built directly (not via
        `compose()`'s `with Collapsible(...): yield ...` context-manager
        idiom, which only works from inside a `compose()` generator) so
        there is exactly one place this tree can drift from itself.

        T12: default collapsed unless the user has previously opened it --
        per-user GLOBAL preference (Console rail section-preference
        precedent), NOT per-server.

        `force_open` (Task 6 review fold, MCP Hub Phase 6): `True` only
        from `_reveal_advanced()`'s explicit opt-in path. A fresh install
        has never persisted `advanced_open` at all, so honoring the
        persisted value unconditionally would land the panel COLLAPSED
        the very moment the user pressed "Advanced..." asking to see it --
        they just asked, so open it regardless of whatever was last
        persisted (or never persisted). `compose()`'s mount-time path
        passes nothing and keeps pure persistence semantics: whatever the
        user set last session stands, with no forcing.
        """
        persisted_open = bool(get_cli_setting("mcp.hub_state", "advanced_open", False))
        open_state = True if force_open else persisted_open
        self._advanced_last_collapsed = not open_state
        return Collapsible(
            Static(self._advanced_object_label(), id="mcp-adv-object", markup=False),
            VerticalScroll(
                Label("Section", classes="form-label"),
                Select(
                    self._sections,
                    id="mcp-adv-section-select",
                    allow_blank=False,
                    value=self._sections[0][1],
                ),
                Static("", id="mcp-adv-content", classes="ds-field-row", markup=False),
                Label("Action", classes="form-label"),
                Select(
                    [("No actions available", Select.BLANK)],
                    id="mcp-adv-action-select",
                    value=Select.BLANK,
                ),
                # Task 4: guidance shown only while the section above has zero
                # runnable action descriptors (see `_refresh_advanced_actions`),
                # so a user landing on e.g. Overview isn't left staring at a
                # disabled "No actions available" select with no next step.
                Static(
                    "", id="mcp-adv-empty-hint", classes="ds-field-row", markup=False
                ),
                Label("Payload (JSON)", classes="form-label"),
                TextArea("{}", id="mcp-adv-payload"),
                Button(
                    "Run Action",
                    id="mcp-adv-run",
                    classes="console-action-primary",
                    compact=True,
                    tooltip="Run the selected legacy control-plane action with this JSON payload.",
                ),
                Static("", id="mcp-adv-result", classes="ds-field-row", markup=False),
                id="mcp-adv-scroll",
            ),
            title="Advanced (legacy control plane)",
            collapsed=not open_state,
            id="mcp-adv-collapsible",
        )

    async def _toggle_advanced(self) -> None:
        """Dispatch the Advanced toggle (F-053) to the right direction.

        The button's press handler disables it synchronously before
        scheduling this worker, so at most one toggle is ever in flight;
        each direction re-enables the button as its last step.
        """
        if self._advanced_visible:
            await self._hide_advanced()
        else:
            await self._reveal_advanced()

    async def _reveal_advanced(self) -> None:
        """Opt into the Advanced control-plane runner (Task 5, MCP Hub Phase 6).

        One direction of the F-053 toggle. Persists
        `mcp.hub_state.advanced_visible=True` (thread-offloaded, same
        pattern as `_persist_advanced_open()` below) so a future mount
        composes the collapsible directly -- an explicit user choice, now
        reversible via `_hide_advanced()` (the same control), so opting in
        is no longer a one-way door.

        Semantics for the button itself (F-053): it is RELABELLED to
        "Hide advanced" (and re-enabled) rather than removed -- it is the
        hide path as well as the reveal path.

        `set_service_context()` may already have been called while
        Advanced was hidden (the workbench rebinds on every `reload()`/
        source switch/selection change, unconditionally) -- that call
        already recorded `self._service`/`self._sections`/
        `self._advanced_source`/`self._advanced_target_label` even though
        it could not touch the (not yet mounted) `#mcp-adv-*` widgets. Once
        the collapsible exists, replay `set_service_context()` with that
        same recorded state so the freshly mounted widgets bind to
        whatever object is actually selected right now instead of opening
        blank -- reusing that method's own population logic rather than
        duplicating it here.

        Task 6 review fold: the collapsible mounts EXPANDED
        (`_build_advanced_collapsible(force_open=True)`) regardless of
        whatever `advanced_open` was last persisted -- the user just
        pressed this button asking to see the panel, so a fresh install's
        never-persisted (False) default must not land it collapsed.
        `advanced_open=True` is persisted alongside `advanced_visible` via
        `_persist_advanced_open()` (the same helper the disclosure's own
        Toggled handler uses) so a future mount also opens directly. That
        handler's own mount-echo guard (below) will NOT do this write
        itself here: `_build_advanced_collapsible()` sets
        `_advanced_last_collapsed` to match the forced (already-expanded)
        state before construction, so the Toggled the reactive fires on
        construction reads as a no-op echo and is dropped -- this call is
        the only place the persist happens.
        """
        if self._advanced_visible:
            return
        self._advanced_visible = True
        try:
            await asyncio.to_thread(
                save_setting_to_cli_config, "mcp.hub_state", "advanced_visible", True
            )
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message(
                    "MCP advanced-visible preference save failed", exc
                ),
            )
        await self._persist_advanced_open(True)
        await self.mount(self._build_advanced_collapsible(force_open=True))
        toggle = self.query_one("#mcp-inspector-advanced-reveal", Button)
        toggle.label = "Hide advanced"
        toggle.tooltip = "Hide the legacy control-plane action runner."
        toggle.disabled = False
        self.set_service_context(
            self._service,
            self._sections,
            source=self._advanced_source,
            target_label=self._advanced_target_label,
        )

    async def _hide_advanced(self) -> None:
        """Hide the Advanced control-plane runner again (F-053).

        The reverse direction of the toggle, mirroring
        `_reveal_advanced()`: persists `advanced_visible=False` (explicit
        user choice, honored by future mounts), removes the collapsible,
        and flips the toggle back to its "Advanced…" reveal state
        (re-enabled). The `advanced_open` disclosure preference is left
        untouched -- collapsing-vs-expanded is a separate, already
        reversible choice the Collapsible's own triangle owns.

        Fix Round G, Item 6: this panel's real-output policy, stated once.
        `collapsible.remove()` below is a genuine TEARDOWN -- the whole
        widget tree (including `#mcp-adv-result`'s run output/refusal
        text) is destroyed, and `_reveal_advanced()` rebuilds a fresh one
        from scratch (`_build_advanced_collapsible()`, a blank `Static("",
        id="mcp-adv-result", ...)`) on the next reveal. Output does not,
        and structurally cannot, survive a hide/reveal cycle. Every OTHER
        interaction with this pane -- section change, rebind, action
        switch, payload edit, collapse/expand of the disclosure triangle
        (Fix Round E, Item 1; Fix Round G, Items 1-2) -- keeps the widget
        tree mounted and therefore preserves real output, clearing only a
        LIVE confirm arm/sentence (`_was_armed`-gated at each of those
        sites). The rule: destroying the DOM subtree destroys its
        content, by construction; anything short of that destruction
        preserves it, by choice.
        """
        if not self._advanced_visible:
            return
        self._advanced_visible = False
        # Fix Round A, Item 3: hiding the panel is the user backing out of
        # the legacy runner entirely -- disarm immediately rather than
        # waiting for the next reveal's `set_service_context()` replay
        # (`_reveal_advanced()` below) to do it. Defense in depth: that
        # replay already covers the reveal path on its own, but a stale arm
        # should not linger in memory for the whole time the panel is
        # hidden either.
        self._advanced_confirm_key = None
        try:
            await asyncio.to_thread(
                save_setting_to_cli_config, "mcp.hub_state", "advanced_visible", False
            )
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message(
                    "MCP advanced-visible preference save failed", exc
                ),
            )
        try:
            collapsible = self.query_one("#mcp-adv-collapsible", Collapsible)
        except NoMatches:
            pass
        else:
            await collapsible.remove()
        toggle = self.query_one("#mcp-inspector-advanced-reveal", Button)
        toggle.label = "Advanced…"
        toggle.tooltip = "Show the legacy control-plane action runner."
        toggle.disabled = False

    # -- T12: Advanced disclosure open/collapsed persistence -----------------

    @on(Collapsible.Toggled, "#mcp-adv-collapsible")
    def _on_advanced_collapsible_toggled(self, event: Collapsible.Toggled) -> None:
        event.stop()
        collapsed = event.collapsible.collapsed
        # Mount-echo guard (review fix): `Collapsible.collapsed` is
        # `reactive(True, init=False)`, so constructing the widget
        # already-expanded (`collapsed=False` differs from the reactive's
        # own True default) fires the watcher during construction and posts
        # ONE Toggled with zero user interaction. The same quirk is
        # documented at library_screen.py's
        # `sync_library_ingest_advanced_open`, where the handler is a
        # harmless in-memory sync -- here it would be a real disk write
        # (TOML read-modify-write) on every mount whenever the preference
        # is open. Drop any event that merely re-asserts the state we
        # already track; real toggles always change it.
        if collapsed == self._advanced_last_collapsed:
            return
        self._advanced_last_collapsed = collapsed
        # Fix Round G, Item 1 (review of Fix Round E): a real collapse/
        # expand of this disclosure's own triangle used to be invisible to
        # the arm entirely -- `_ADVANCED_EXECUTE_CONFIRM` promises "anything
        # else cancels", but this handler only ever persisted the open/
        # collapsed preference. Live-verified: arm `tool.execute`, collapse,
        # then expand -- the arm survived, so ONE press ran the tool with no
        # confirm ever shown for that viewing. The Collapsible's own child
        # widgets (`#mcp-adv-result` included) stay mounted across a
        # collapse -- only their CSS display toggles (see `_build_advanced_
        # collapsible()`'s `.-collapsed` rule) -- so this is the same
        # "attention moved" trigger `_hide_advanced()`'s own clear (:1226)
        # already treats as arm-cancelling, just a lighter-touch version of
        # it (tuck away, not tear down; see that method's own docstring for
        # the teardown-vs-preserve rule this class follows). Same `_was_
        # armed` gate every other disarm site in this class uses: a real
        # run result/refusal sitting in `#mcp-adv-result` while UNARMED
        # must survive a collapse/expand exactly as it survives a section
        # change or a rebind (Fix Round E, Item 1) -- only a LIVE confirm
        # sentence is ever cleared away.
        _was_armed = self._advanced_confirm_key is not None
        self._advanced_confirm_key = None
        if _was_armed:
            self.query_one("#mcp-adv-result", Static).update("")
        self.run_worker(
            self._persist_advanced_open(not collapsed),
            group="mcp-adv-open",
            exclusive=True,
        )

    async def _persist_advanced_open(self, open_state: bool) -> None:
        """Persist the Advanced disclosure's open/collapsed state.

        Thread-offloaded exactly like `MCPWorkbench._save_builtin_flag()`
        (`mcp_workbench.py`, Task 10 precedent) -- `save_setting_to_cli_config`
        does a blocking TOML read-modify-write. Unlike that handler, this one
        has no UI to resync afterward (the Collapsible already reflects its
        own reactive `collapsed` state) and doesn't reach into `self.app` at
        all: `save_setting_to_cli_config` is a free function, not something
        that needs app-specific wiring, so there is nothing here that a bare
        test App would be missing (contrast `_action_allowed()`'s
        getattr-tolerant read of `self.app.require_ui_action_allowed`, which
        DOES need that idiom because it's an app-specific seam). Failures are
        logged and swallowed rather than surfaced via `self.app.notify()`
        (unlike `_save_builtin_flag`): this is a low-stakes UI preference that
        silently reverts to its default on next launch, not worth alarming
        the user over.
        """
        try:
            await asyncio.to_thread(
                save_setting_to_cli_config, "mcp.hub_state", "advanced_open", open_state
            )
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message(
                    "MCP advanced-open preference save failed", exc
                ),
            )

    # -- readiness block -----------------------------------------------------

    async def update_readiness(self, snapshot: ReadinessSnapshot | None) -> None:
        """Rebuild the action-button list for the given snapshot.

        Awaited end to end (remove, then mount) within a single call so
        Textual's per-widget message pump cannot interleave two rebuilds:
        selecting a second server before the first's `remove_children()`
        has actually pruned its buttons from the DOM previously raced a
        `mount()` of a same-id button (`view_details` is in almost every
        reason's action set) into `DuplicateIds`, crashing the whole app.
        Mirrors the fix in `console_session_switcher_modal.py`
        (`_refresh_results`) for the same bug class.

        Task 4: the whole body additionally runs under `self._refresh_lock`
        so two concurrently-awaited calls (e.g. a worker-driven refresh
        racing a pump-driven one) can't interleave their remove/mount
        cycles -- the second call's remove_children()/mount_all() only
        begins once the first has fully completed.
        """
        async with self._refresh_lock:
            self._snapshot = snapshot
            state = self.query_one("#mcp-inspector-state", Static)
            # RAG-50: `show_tool()` owns `state.display` (hidden while tool
            # detail is shown, restored on `show_tool(None)`) -- this method
            # only ever touches `state`'s CONTENT/CSS class below and must
            # keep it that way. A server-selection sync
            # (`MCPWorkbench._sync_children()`) can legitimately fire while
            # Tools mode has a tool displayed (e.g. a background readiness
            # refresh with no server selected, or a stale-selection race);
            # if this method ever starts writing `state.display` too, guard
            # it on "a detail container is currently displayed"
            # (`not self._current_tool`) so it can't resurrect the badge
            # over populated detail.
            message = self.query_one("#mcp-inspector-message", Static)
            actions = self.query_one("#mcp-inspector-actions", Vertical)
            await actions.remove_children()
            # Task 11: this Static persists across snapshots (unlike the
            # rail's rows, which are recomposed fresh) -- drop whatever
            # status class the previous snapshot left behind before
            # possibly adding the new one, so two selections in a row never
            # leave a stale color class stacked alongside the current one.
            for css_class in STATE_CSS_CLASSES.values():
                state.remove_class(css_class)
            if snapshot is None:
                state.update(_EMPTY_STATE_COPY)
                message.update("")
                return
            state.add_class(STATE_CSS_CLASSES[snapshot.state])
            state.update(f"{snapshot.badge_text()}  {snapshot.label}")
            # A5: lead with the humanized *reason*, not a repeat of the
            # canvas's own snapshot.message -- the inspector should add
            # "why", which the canvas detail view doesn't say, rather than
            # mirror what it already shows. A3a: never leak the internal
            # ReasonCode value (e.g. "runtime_unavailable") into user-facing
            # copy.
            reason = snapshot.primary_reason
            if snapshot.state is ReadinessState.CHECKING:
                # as_checking() clears reasons but leaves tool_count from the
                # underlying snapshot alone -- lead with its own working
                # message instead of falling through to a stale "Ready"/reason
                # line that would contradict the "Checking" badge above.
                why_line = snapshot.message
            elif is_off_opt_in(snapshot):
                # task-2240: an off-by-choice built-in is not "not
                # configured" -- with the lone-row preselect this is the
                # first inspector content a new user sees, so the why-line
                # explains the opt-in (and where the fix lives) instead of
                # filing a setup defect.
                why_line = (
                    "Why · Off — enable it to let MCP clients use chatbook's tools."
                )
            elif reason is not None:
                why_line = f"Why · {REASON_LABELS[reason]}"
            elif snapshot.tool_count is not None:
                why_line = f"Why · Ready — {snapshot.tool_count} tools available"
            else:
                why_line = "Why · Ready"
            message.update(why_line)
            if snapshot.state is ReadinessState.CHECKING:
                # T5: an in-flight lifecycle action replaces the action set
                # with a single Cancel button -- nothing else is actionable
                # on this server until the worker finishes or is cancelled.
                cancel_button = Button(
                    "Cancel",
                    id="mcp-inspector-cancel",
                    classes="mcp-inspector-action console-action-secondary",
                    compact=True,
                    tooltip="Cancel the in-flight operation.",
                )
                await actions.mount_all([cancel_button])
                return
            wired = _wired_actions(snapshot)
            buttons = []
            for action in snapshot.allowed_actions:
                button = Button(
                    _ACTION_LABELS[action],
                    id=f"mcp-inspector-action-{action.value}",
                    classes="mcp-inspector-action console-action-secondary",
                    compact=True,
                )
                if action not in wired:
                    button.disabled = True
                    if (
                        action in (_LIFECYCLE_ACTIONS | _CONFIG_ACTIONS)
                        and snapshot.source != "local"
                    ):
                        button.tooltip = _SERVER_MANAGED_TOOLTIP
                    elif (
                        action is HubAction.OPEN_CREDENTIALS
                        and snapshot.source == "local"
                    ):
                        button.tooltip = _OPEN_CREDENTIALS_LOCAL_TOOLTIP
                    else:
                        button.tooltip = _UNAVAILABLE_ACTION_TOOLTIP
                else:
                    button.tooltip = _WIRED_ACTION_TOOLTIPS.get(
                        action, _ACTION_LABELS[action]
                    )
                buttons.append(button)
            if buttons:
                await actions.mount_all(buttons)

    # -- T6: tool detail view + Test Tool runner ------------------------------

    def _any_detail_displayed(self) -> bool:
        """True while any of the four detail views has content on screen.

        Returns:
            True when a tool, permission row, audit entry, or finding is
            currently displayed; False when every detail view is cleared.
        """
        return (
            self._current_tool is not None
            or self._current_permission_tool is not None
            or self._current_audit_entry is not None
            or self._current_finding is not None
        )

    def _sync_state_badge_display(self) -> None:
        """task-2270: single owner of `#mcp-inspector-state`'s DISPLAY.

        The badge (empty-state copy, or the selected server's readiness --
        whatever `update_readiness()` last wrote) shows exactly when NO
        detail view is displayed. RAG-50 fixed this for Tools mode only,
        with `show_tool()` writing `state.display` directly; the other
        three detail views (`show_permission()` via
        `_render_permission_container()`, `show_audit_entry()`,
        `show_finding()`) never touched it and left the "Pick a server,
        tool, or entry…" badge stacked above fully populated detail.
        Every view's show/clear path now funnels through this method, so
        clearing ONE view cannot resurrect the badge while another still
        shows detail, and `update_readiness()` stays content-only (it can
        never force the badge back over displayed detail in any mode).
        """
        self.query_one(
            "#mcp-inspector-state", Static
        ).display = not self._any_detail_displayed()

    async def show_tool(
        self, tool: HubTool | None, *, effective: EffectiveToolState | None = None
    ) -> None:
        """Rebuild `#mcp-inspector-tool` for the given tool, or hide it.

        Awaited end to end (remove, then mount) within a single call, under
        the SAME `_refresh_lock` `update_readiness()` uses -- two selections
        in a row (a Tools-mode row click arriving while the previous
        selection's tool-detail refresh is still settling) must never
        interleave their remove/mount cycles into `DuplicateIds`, exactly
        the P0 class `update_readiness()` was already hardened against (see
        its own docstring). Any previously-open Test Tool panel is
        implicitly discarded by `remove_children()` below -- selecting a
        different tool (or clearing the selection) always starts fresh.

        Task 7: `effective`, when given (Tools-mode's own call site --
        `MCPWorkbench.on_mcp_tools_mode_tool_selected()`), appends the
        permission-explanation block below the tool detail via
        `_render_permission_container()`, folded into this SAME locked
        pass rather than re-entering `_refresh_lock` a second time (it is
        not reentrant). `None` (every pre-Task-7 call site, and a cleared
        selection) hides `#mcp-inspector-permission` instead of leaving a
        previous tool's permission facts on screen.
        """
        async with self._refresh_lock:
            self._current_tool = tool
            old_nonce = self.clear_test_preview()
            if old_nonce:
                self.post_message(self.ToolTestPreviewRevocationRequested(old_nonce))
            container = self.query_one("#mcp-inspector-tool", Vertical)
            await container.remove_children()
            # RAG-50 / task-2270: the empty-state badge's DISPLAY is owned
            # by `_sync_state_badge_display()` (badge shows exactly when NO
            # detail view is displayed -- RAG-50 fixed Tools mode alone and
            # left the other three views stacking the badge over populated
            # detail). The sync here runs BEFORE any await so a paint
            # between this method's awaits can never show badge + detail
            # together; the branch-final `_render_permission_container()`
            # call syncs again after `_current_permission_tool` settles.
            # Content stays `update_readiness()`'s job -- restoring
            # visibility just reveals whatever it last wrote (or the
            # compose()-time `_EMPTY_STATE_COPY` if it never ran).
            self._sync_state_badge_display()
            if tool is None:
                container.display = False
                await self._render_permission_container(None, None)
                return
            container.display = True
            widgets: list[Any] = [
                Static(
                    f"{tool.name} — {tool.server_label}",
                    id="mcp-inspector-tool-name",
                    classes="ds-field-row",
                    markup=False,
                ),
                Static(
                    tool.description,
                    id="mcp-inspector-tool-description",
                    classes="ds-field-row",
                    markup=False,
                ),
                Static(
                    f"Tags: {', '.join(tool.tags) if tool.tags else '—'}",
                    id="mcp-inspector-tool-tags",
                    classes="ds-field-row",
                    markup=False,
                ),
                Static(
                    "Parameters: form"
                    if parse_schema(tool.input_schema) is not None
                    else "Parameters: raw JSON",
                    id="mcp-inspector-tool-schema",
                    classes="ds-field-row",
                    markup=False,
                ),
            ]
            if tool.stale:
                widgets.append(
                    Static(
                        "Stale — not currently connected.",
                        id="mcp-inspector-tool-stale",
                        classes="ds-field-row",
                        markup=False,
                    )
                )
            if tool.executable:
                widgets.append(
                    Button(
                        "Test Tool",
                        id="mcp-inspector-test-tool",
                        classes="console-action-primary",
                        compact=True,
                        tooltip="Run this tool with test arguments.",
                    )
                )
            else:
                phase_note = "Server-source tools are display-only."
                # Keep this UI-only identity check lightweight. Importing
                # the raw-shell provider here pulls its executor and input-
                # validation graph into the generic inspector, creating a
                # Chat/Library import cycle during test and app startup.
                if tool.tool_id == "local:__local__::shell_exec":
                    phase_note = (
                        "Policy only — raw shell commands run from Console "
                        "under its separate approval flow."
                    )
                elif tool.source != "server":
                    phase_note = "Tool testing is unavailable from this policy view."
                widgets.append(
                    Static(
                        phase_note,
                        id="mcp-inspector-tool-phase-note",
                        classes="ds-field-row",
                        markup=False,
                    )
                )
            await container.mount_all(widgets)
            # Task 3 (MCP Hub Phase 6): `show_goto_button=True` -- Tools-
            # mode's own combined call gets the "Change in Permissions" jump
            # button; the standalone `show_permission()` below does not
            # (jumping to the Permissions-mode row you're already looking at
            # would be a no-op affordance). Never passes `cascade` -- that
            # wiring is `show_permission()`-only per the brief, so this path
            # keeps rendering the plain origin sentence.
            await self._render_permission_container(
                tool, effective, show_goto_button=True
            )

    async def _render_permission_container(
        self,
        tool: HubTool | None,
        effective: EffectiveToolState | None,
        *,
        cascade: tuple[str | None, str | None, str] | None = None,
        show_goto_button: bool = False,
    ) -> None:
        """Rebuild `#mcp-inspector-permission` for one tool's resolved
        permission state, or hide it.

        LOCK-FREE by design: both callers (`show_tool()`, `show_permission()`
        below) already hold `_refresh_lock` when they call this -- folding
        the permission-block update into their SAME locked pass instead of
        re-entering the (non-reentrant) `asyncio.Lock` a second time.

        `tool is None or effective is None` means "nothing to explain" --
        covers an outright cleared selection (`show_tool(None)`) and a tool
        selected with no permission context supplied (`show_tool(tool)`,
        the plain T6 call shape every pre-Task-7 call site still uses).

        Task 3 (MCP Hub Phase 6): `cascade`, when given (`show_permission()`
        only), renders the three provenance rungs (`_cascade_rungs()`)
        instead of the single `_ORIGIN_SENTENCES` sentence; `None` (every
        other call shape, and a `show_permission()` caller with nothing to
        report) keeps the old sentence. `show_goto_button` mounts the
        "Change in Permissions" jump button (`#mcp-inspector-goto-
        permission`) -- `show_tool()`'s own call site only; see its
        docstring.
        """
        container = self.query_one("#mcp-inspector-permission", Vertical)
        await container.remove_children()
        if tool is None or effective is None:
            container.display = False
            self._current_permission_tool = None
            # task-2270: restore the badge -- unless another detail view
            # (tool/audit/finding) still shows content.
            self._sync_state_badge_display()
            return
        container.display = True
        self._current_permission_tool = tool
        # task-2270: a Permissions-matrix row selection hides the badge
        # exactly like Tools mode does; synced before the mounts below so
        # no paint frame shows badge + populated detail together.
        self._sync_state_badge_display()
        widgets: list[Any] = [
            # UX batch item 8: identity line first, mirroring
            # `show_tool()`'s own `#mcp-inspector-tool-name` -- this block
            # is ALSO the standalone entry point (Permissions-mode matrix
            # row selection, `show_permission()` below), which never mounts
            # `#mcp-inspector-tool` at all, so without this the permission
            # explanation would render with no indication of WHICH tool it
            # describes.
            Static(
                f"{tool.name} — {tool.server_label}",
                id="mcp-inspector-permission-tool",
                classes="ds-field-row",
                markup=False,
            ),
            # Task 1 (MCP Hub Phase 6): a non-cell Static -- prefer the
            # existing `.mcp-status-*` CSS classes (`css/tldw_cli_modular.
            # tcss`) over `mcp_permissions_mode.state_text()`'s Rich-style
            # mechanism, which exists only because a DataTable cell can't
            # carry a CSS class at all. `tool_state_kind()` returns exactly
            # `"ready"|"warning"|"error"` for a real `EffectiveToolState`
            # (the `"muted"` fallback never fires here -- `state` is always
            # one of `allow|ask|deny`), so `mcp-status-{kind}` always
            # resolves to one of the three classes the bundle defines.
            #
            # Fix Round J (PR #1385) found the stacked contradiction here:
            # a gate_error verdict rendered "Permission: Off" (a confident
            # configuration claim) one line above "Permission state could
            # not be resolved." (an admission we don't know it), and this
            # site grew its own origin-aware label branch. task-2870 moved
            # that ownership INTO `EffectiveToolState.ui_label` (gate_error
            # -> "Unknown") so the matrix/State-column renderers tell the
            # same truth -- the branch here collapsed back to the plain
            # read, and round J's test now pins the behavior through
            # `ui_label` like every other surface. The `error` status
            # class is KEPT on purpose: the color encodes the EFFECT
            # (fail-closed, the tool will not run), which is true; only
            # the causal label lied.
            Static(
                f"Permission: {effective.ui_label}",
                id="mcp-inspector-permission-state",
                classes=f"ds-field-row mcp-status-{tool_state_kind(effective)}",
                markup=False,
            ),
        ]
        if cascade is not None:
            widgets.extend(_cascade_rungs(cascade, effective))
        else:
            widgets.append(
                Static(
                    _ORIGIN_SENTENCES.get(effective.origin, _UNKNOWN_ORIGIN_SENTENCE),
                    id="mcp-inspector-permission-origin",
                    classes="ds-field-row",
                    markup=False,
                )
            )
        if effective.config_changed:
            widgets.append(
                Static(
                    _CONFIG_CHANGED_NOTICE,
                    id="mcp-inspector-permission-notice",
                    classes="ds-field-row",
                    markup=False,
                )
            )
            widgets.append(
                Button(
                    "Re-allow",
                    id="mcp-inspector-reallow",
                    classes="console-action-primary",
                    compact=True,
                    tooltip=_REALLOW_TOOLTIP,
                )
            )
        elif effective.risk_floored:
            widgets.append(
                Static(
                    _RISK_FLOORED_NOTICE,
                    id="mcp-inspector-permission-notice",
                    classes="ds-field-row",
                    markup=False,
                )
            )
        if show_goto_button:
            widgets.append(
                Button(
                    "Change in Permissions",
                    id="mcp-inspector-goto-permission",
                    classes="console-action-secondary",
                    compact=True,
                    tooltip=_GOTO_PERMISSION_TOOLTIP,
                )
            )
        await container.mount_all(widgets)

    async def show_permission(
        self,
        tool: HubTool,
        effective: EffectiveToolState,
        *,
        cascade: tuple[str | None, str | None, str] | None = None,
    ) -> None:
        """Render `#mcp-inspector-permission` standalone -- Permissions-mode's
        matrix tool-row selection entry point
        (`MCPWorkbench.on_mcp_permissions_mode_row_selected()`).

        Unlike `show_tool()`'s `effective` keyword, this never touches
        `#mcp-inspector-tool` -- the full tool-detail-plus-Test-Tool block
        is Tools-mode's own selection surface; a Permissions-mode row
        selection only explains the permission rule. Same `_refresh_lock`
        discipline as `show_tool()` -- two selections back to back must not
        interleave their remove/mount cycles into `DuplicateIds` (mandatory
        regression, mirrors
        `test_second_show_tool_back_to_back_does_not_duplicate_ids`).

        Task 3 (MCP Hub Phase 6): `cascade` is the raw
        `(tool_entry_state, server_default, global_default)` tuple the
        workbench already derives per tool while building the Permissions
        matrix (`MCPWorkbench._build_permission_rows()`) -- `None` (the
        default) falls back to the pre-Task-3 single origin sentence.
        """
        async with self._refresh_lock:
            await self._render_permission_container(tool, effective, cascade=cascade)

    async def show_audit_entry(self, entry: dict[str, Any] | None) -> None:
        """Render `#mcp-inspector-audit` for one execution-log entry, or hide it.

        Audit mode's own row-selection entry point
        (`MCPWorkbench.on_mcp_audit_mode_entry_selected()`) -- standalone,
        same `_refresh_lock` discipline as `show_permission()` above (never
        folded into `show_tool()`'s single locked pass, since an audit-entry
        selection never touches `#mcp-inspector-tool`/`#mcp-inspector-
        permission`). `entry=None` (a stale/out-of-range selection, or a
        mode switch via `MCPWorkbench._clear_tool_view()`) hides the
        container instead of leaving a previous entry's facts on screen.

        The detail is a ``json.dumps(indent=2)`` view of the execution log's
        metadata-only public schema. It contains categories, types, counts,
        and registered argument names, never argument values, result excerpts,
        or exception text.
        """
        async with self._refresh_lock:
            container = self.query_one("#mcp-inspector-audit", Vertical)
            await container.remove_children()
            if entry is None:
                container.display = False
                self._current_audit_entry = None
                # task-2270: restore the badge unless another detail view
                # still shows content (e.g. a finding alongside this entry).
                self._sync_state_badge_display()
                return
            container.display = True
            self._current_audit_entry = entry
            self._sync_state_badge_display()  # task-2270: hide over detail
            server_key = str(entry.get("server_key") or "")
            tool_name = str(entry.get("tool_name") or "")
            detail_payload = audit_entry_detail_payload(entry)
            detail_text = json.dumps(detail_payload, indent=2, default=str)
            widgets: list[Any] = [
                Static(
                    f"{tool_name} — {server_key}"
                    if (tool_name or server_key)
                    else "Execution detail",
                    id="mcp-inspector-audit-name",
                    classes="ds-field-row",
                    markup=False,
                ),
                VerticalScroll(
                    Static(detail_text, id="mcp-inspector-audit-detail", markup=False),
                    id="mcp-inspector-audit-scroll",
                ),
                Button(
                    "Open tool",
                    id="mcp-audit-open-tool",
                    classes="console-action-secondary",
                    compact=True,
                    tooltip="Switch to Tools mode and select this tool.",
                ),
                Button(
                    "Adjust permission",
                    id="mcp-audit-adjust-permission",
                    classes="console-action-secondary",
                    compact=True,
                    tooltip="Switch to Permissions mode and select this tool's row.",
                ),
            ]
            await container.mount_all(widgets)

    async def show_finding(
        self, finding: dict[str, Any] | None, *, server_key: str | None = None
    ) -> None:
        """Render `#mcp-inspector-finding` for one Audit-mode Findings-table
        row, or hide it (T8, MCP Hub Phase 5).

        Findings-mode's own row-selection entry point (`MCPWorkbench.
        on_mcp_audit_mode_finding_selected()`) -- standalone, same
        `_refresh_lock` discipline as `show_permission()`/`show_audit_
        entry()` (two selections back to back must not interleave their
        remove/mount cycles into `DuplicateIds`). `finding=None` (a
        stale/out-of-range selection, or a mode switch via `MCPWorkbench.
        _clear_tool_view()`) hides the container instead of leaving a
        previous finding's facts on screen.

        Severity/type/message, plus a suggested-remediation line only when
        the raw payload actually carries one, plus -- Task 2 (MCP Hub
        Phase 6) -- one Button per `remediation_actions(finding)` HubAction
        (ids `#mcp-finding-action-<action>`, reusing `_ACTION_LABELS`/
        `_WIRED_ACTION_TOOLTIPS`, tooltipped). Each posts the EXISTING
        `HubActionRequested` message with `server_key` -- the finding's
        owning server as resolved by the CALLER (target-level when
        derivable from the finding itself, else the selected rail server;
        `None` when neither is available) -- read back by the button press
        handler in `on_button_pressed()` below via `_current_finding_
        server_key`. `markup=False` throughout -- finding fields are
        server-derived free text that must never be interpreted as Rich
        markup.

        New Minor 3 (MCP Hub Phase 6 finale, review): `server_key=None`
        means the CALLER already tried both resolution paths (the finding's
        own target-level id, then the selected rail server) and neither
        worked -- every remediation button would post `HubActionRequested`
        with no server to act on, and `on_mcp_inspector_hub_action_
        requested()` silently drops every one of those (each branch guards
        on a truthy `event.server_key`). Rendering the buttons anyway would
        just be dead chrome, so this renders one explanatory note instead
        and skips the button loop entirely.
        """
        # Task 2: local import -- `mcp_audit_mode.py` imports `format_
        # duration_ms` from THIS module at its own top level, so importing
        # `remediation_actions` back from it at module level here would be
        # a circular import. Deferred to call time, by which point both
        # modules have already finished loading.
        from tldw_chatbook.UI.MCP_Modules.mcp_audit_mode import remediation_actions

        async with self._refresh_lock:
            container = self.query_one("#mcp-inspector-finding", Vertical)
            await container.remove_children()
            if finding is None:
                container.display = False
                self._current_finding = None
                self._current_finding_server_key = None
                # task-2270: restore the badge unless another detail view
                # still shows content (e.g. the audit entry beside this).
                self._sync_state_badge_display()
                return
            container.display = True
            self._current_finding = finding
            self._current_finding_server_key = server_key
            self._sync_state_badge_display()  # task-2270: hide over detail
            severity = _finding_text(finding, "severity")
            finding_type = _finding_text(finding, "finding_type", "type")
            message = _finding_text(finding, "message")
            widgets: list[Any] = [
                Static(
                    f"Finding — {severity}",
                    id="mcp-inspector-finding-name",
                    classes="ds-field-row",
                    markup=False,
                ),
                Static(
                    f"Type: {finding_type}",
                    id="mcp-inspector-finding-type",
                    classes="ds-field-row",
                    markup=False,
                ),
                Static(
                    message,
                    id="mcp-inspector-finding-message",
                    classes="ds-field-row",
                    markup=False,
                ),
            ]
            remediation = _finding_remediation(finding)
            if remediation:
                widgets.append(
                    Static(
                        f"Suggested remediation: {remediation}",
                        id="mcp-inspector-finding-remediation",
                        classes="ds-field-row",
                        markup=False,
                    )
                )
            if server_key is None:
                widgets.append(
                    Static(
                        "No server context — select a server first.",
                        id="mcp-inspector-finding-no-context",
                        classes="ds-field-row",
                        markup=False,
                    )
                )
            else:
                for action in remediation_actions(finding):
                    widgets.append(
                        Button(
                            _ACTION_LABELS[action],
                            id=f"mcp-finding-action-{action.value}",
                            classes="console-action-secondary",
                            compact=True,
                            tooltip=_WIRED_ACTION_TOOLTIPS.get(
                                action, _ACTION_LABELS[action]
                            ),
                        )
                    )
            await container.mount_all(widgets)

    async def _mount_test_tool_panel(self) -> None:
        """Mount the schema-driven form + Run/Close/result panel, once.

        Guarded against a double mount (two `Test Tool` presses queued
        before the first handler's `disabled = True` takes effect -- the
        same message-pump race the profile-save-form double-submit fix
        documents in mcp_workbench.py) by checking for the panel first.
        """
        tool = self._current_tool
        if tool is None:
            return
        container = self.query_one("#mcp-inspector-tool", Vertical)
        try:
            container.query_one("#mcp-inspector-test-panel")
            return  # already open
        except NoMatches:
            pass
        panel = Vertical(
            MCPSchemaForm(schema=tool.input_schema, id="mcp-inspector-test-form"),
            Static(
                _TEST_PREPARING_TEXT,
                id="mcp-inspector-test-preview",
                classes="ds-field-row",
                markup=False,
            ),
            Button(
                "Preparing…",
                id="mcp-inspector-test-run",
                classes="console-action-primary",
                compact=True,
                tooltip=_TEST_RUN_TOOLTIP,
                disabled=True,
            ),
            self._build_test_retry_button(),
            Button(
                "Close",
                id="mcp-inspector-test-close",
                classes="console-action-secondary",
                compact=True,
                tooltip="Close this test form without running the tool.",
            ),
            Static(
                "", id="mcp-inspector-test-result", classes="ds-field-row", markup=False
            ),
            # RAG-49 (PR-5 task 4): the quiet interpretation line (empty/
            # error/unusual-shape explanations) -- a sibling of the summary
            # Static above, not appended into it. Always mounted, hidden
            # (`display = False`) until `show_tool_result()` has something
            # to say; the same always-mounted discipline as the raw
            # Collapsible below (never conditionally composed).
            self._build_test_result_note_static(),
            # RAG-49 (PR-5 task 4): the collapsed "Raw response" Collapsible
            # -- always mounted (never conditionally composed, the
            # program's PR-2 lesson), hidden (`display = False`) until
            # `show_tool_result()` has a raw body to show.
            self._build_test_result_raw_collapsible(),
            # The Test Tool panel's own "Change in Permissions" jump button
            # is mounted once and toggled from previews/results. A distinct id
            # from the Tools-mode permission block's own button
            # (`#mcp-inspector-goto-permission`) -- both can be mounted at
            # once (this same tool selected, its permission block shown
            # below the detail, AND this panel open), and `query_one`
            # requires a unique id across the whole subtree.
            self._build_test_goto_permission_button(),
            id="mcp-inspector-test-panel",
        )
        await container.mount(panel)
        self.post_message(self.ToolTestPreviewRequested(tool.server_key, tool.name))
        # F-056: opening the panel moves keyboard focus into it -- the
        # schema form's first control when there is one (a raw-JSON
        # TextArea, an enum Select, a Checkbox, or a scalar Input), the
        # Close button otherwise. `call_after_refresh` so this lands after
        # Textual's own mount-time focus settling instead of racing it.
        #
        # task-2740: two stacked defects made this a live CRASHER, not a
        # focus nit -- the query omitted `Checkbox`, and `DOMQuery.first()`
        # RAISES `NoMatches` on an empty result (it never returns None, so
        # the old `is None` fallback to Close was dead code and the
        # "otherwise" above never happened). Any tool whose schema renders
        # no Input/Select/TextArea -- an all-boolean schema, or the empty
        # `properties` the real built-in `list_characters` ships -- blew up
        # this mount worker (default `exit_on_error`) and took the app
        # down. The truthiness guard below is the check `DOMQuery`
        # actually supports; the Close fallback is now reachable, and the
        # zero-control regression test pins it.
        controls = panel.query("Input, Select, TextArea, Checkbox")
        first_control = (
            controls.first()
            if controls
            else panel.query_one("#mcp-inspector-test-close", Button)
        )
        self.call_after_refresh(first_control.focus)

    async def action_close_test_panel(self) -> None:
        """F-056: Escape -- close the Test Tool panel exactly like its
        Close button; no-op when no panel is open."""
        if self.query("#mcp-inspector-test-panel"):
            await self._close_test_tool_panel()

    @staticmethod
    def _build_test_goto_permission_button() -> Button:
        button = Button(
            "Change in Permissions",
            id="mcp-inspector-goto-permission-test",
            classes="console-action-secondary",
            compact=True,
            tooltip=_GOTO_PERMISSION_TOOLTIP,
        )
        button.display = False
        return button

    @staticmethod
    def _build_test_retry_button() -> Button:
        """Build the stable, normally-hidden transient preview retry action."""
        button = Button(
            "Retry preview",
            id="mcp-inspector-test-retry",
            classes="console-action-secondary",
            compact=True,
            tooltip=(
                "Request a fresh permission preview without changing these arguments."
            ),
        )
        button.display = False
        return button

    @staticmethod
    def _build_test_result_note_static() -> Static:
        """RAG-49 (PR-5 task 4): the Test Tool result's quiet interpretation
        line -- always mounted (never conditionally composed), hidden until
        `show_tool_result()` has something to say (an empty-result or
        tool-error-shape explanation)."""
        widget = Static(
            "",
            id="mcp-inspector-test-result-note",
            classes="mcp-inspector-result-note",
            markup=False,
        )
        widget.display = False
        return widget

    @staticmethod
    def _build_test_result_raw_collapsible() -> Collapsible:
        """RAG-49 (PR-5 task 4): the "Raw response" Collapsible -- ALWAYS
        mounted (never conditionally composed, the program's PR-2 lesson:
        conditional composition breeds invisible-widget bugs), hidden via
        `display = False` until `show_tool_result()` has raw content to
        show. Collapsed by default; the body Static inside is `markup=
        False` -- tool output is untrusted (the builtin branch executes
        in-process code)."""
        collapsible = Collapsible(
            VerticalScroll(
                Static(
                    "",
                    id="mcp-inspector-test-result-raw-body",
                    markup=False,
                ),
                id="mcp-inspector-test-result-raw-scroll",
            ),
            title="Raw response",
            collapsed=True,
            id="mcp-inspector-test-result-raw",
        )
        collapsible.display = False
        return collapsible

    @property
    def current_tool(self) -> HubTool | None:
        """The `HubTool` `#mcp-inspector-tool` currently describes, or `None`.

        Read-only accessor for `MCPWorkbench.open_test_for_selected_tool()`
        (the `t` keybinding's entry point, mcp_screen.py) to check whether
        there's anything to test before dispatching -- mirrors how every
        other cross-widget read here goes through a public method rather
        than reaching into `_current_tool` directly.
        """
        return self._current_tool

    async def open_test_panel(self) -> str:
        """Open the Test Tool panel for the currently selected tool, via the
        SAME path the Test Tool button's own press handler uses
        (`on_button_pressed`'s `mcp-inspector-test-tool` branch: disable the
        button synchronously, then `_mount_test_tool_panel()`) -- the `t`
        keybinding's entry point never duplicates that mount logic.

        Returns one of three statuses so the caller
        (`MCPWorkbench.open_test_for_selected_tool()`) can tell "nothing
        selected" apart from "a tool IS selected but isn't executable yet"
        (server-source, Phase 4) -- `show_tool()` never renders a `Test
        Tool` button for the latter, so there is nothing this keybinding
        could open for one either, but the two cases warrant different
        copy (see that caller):
          - `"opened"`: the panel was mounted (or was already open).
          - `"no_tool"`: nothing is selected in the inspector.
          - `"not_executable"`: a tool is selected but can't be tested yet.
        """
        tool = self._current_tool
        if tool is None:
            return "no_tool"
        if not tool.executable:
            return "not_executable"
        try:
            self.query_one("#mcp-inspector-test-tool", Button).disabled = True
        except NoMatches:
            pass
        await self._mount_test_tool_panel()
        return "opened"

    async def _close_test_tool_panel(self) -> None:
        nonce = self.clear_test_preview()
        if nonce:
            self.post_message(self.ToolTestPreviewRevocationRequested(nonce))
        try:
            panel = self.query_one("#mcp-inspector-test-panel", Vertical)
        except NoMatches:
            pass
        else:
            await panel.remove()
        try:
            self.query_one("#mcp-inspector-test-tool", Button).disabled = False
        except NoMatches:
            pass

    @property
    def current_permission_tool(self) -> HubTool | None:
        """The tool `#mcp-inspector-permission` is currently explaining, or
        `None` when nothing is shown there.

        Minor 3: lets `MCPWorkbench` check, after a Space-cycle resyncs the
        Permissions-mode matrix, whether the already-open permission block
        belongs to the SAME tool that was just cycled -- so it can refresh
        that block too (`_render_permission_container()` is otherwise only
        re-entered by a fresh selection or the re-allow handler)."""
        return self._current_permission_tool

    def show_test_preview(self, preview: ToolTestAdmissionPreview) -> None:
        """Render an immutable service preview only for the visible exact tool."""
        tool = self._current_tool
        if (
            tool is None
            or preview.server_key != tool.server_key
            or preview.tool_name != tool.name
            or not self.query("#mcp-inspector-test-panel")
        ):
            return
        self._test_preview = preview
        try:
            button = self.query_one("#mcp-inspector-test-run", Button)
            status = self.query_one("#mcp-inspector-test-preview", Static)
            goto = self.query_one("#mcp-inspector-goto-permission-test", Button)
            retry = self.query_one("#mcp-inspector-test-retry", Button)
        except NoMatches:
            return
        retry.display = False
        retry.disabled = True
        gate = preview.rendered_gate
        if gate == "allow":
            button.label = "Run"
            button.tooltip = _TEST_RUN_TOOLTIP
            button.disabled = False
            status.update("Ready. Runs once with the current arguments.")
            goto.display = False
        elif gate == "ask":
            button.label = "Approve & run once"
            button.tooltip = (
                "Approve this one invocation; the permission does not persist."
            )
            button.disabled = False
            status.update(_TEST_ASK_TEXT)
            goto.display = True
        elif gate == "off":
            button.label = "Blocked"
            button.disabled = True
            status.update(_TEST_OFF_TEXT)
            goto.display = True
        else:
            button.label = "Unavailable"
            button.disabled = True
            status.update(_TEST_UNAVAILABLE_TEXT)
            goto.display = True
            retry.display = True
            retry.disabled = False

    def clear_test_preview(self) -> str | None:
        """Drop the rendered preview and return its nonce for revocation."""
        preview = self._test_preview
        self._test_preview = None
        return preview.nonce if preview is not None else None

    def show_test_preparing(self) -> None:
        """Fail closed while a service preview is being prepared."""
        self.clear_test_preview()
        self._set_test_unavailable("Preparing…", _TEST_PREPARING_TEXT, retry=False)

    def show_test_unavailable(self, reason: str | None = None) -> None:
        """Fail closed with bounded recovery copy when previewing fails."""
        message = _TEST_UNAVAILABLE_TEXT
        if reason:
            safe_reason = _safe_tool_test_text(reason, limit=240)
            message = f"Unavailable. {safe_reason} Try again."
        self._set_test_unavailable("Unavailable", message, retry=True)

    def show_test_active(self, active: bool) -> None:
        """Render service-owned active state without becoming its authority."""
        if not active:
            return
        self._set_test_unavailable(
            "Running…",
            "A test for this tool is already active. Wait for it to finish.",
            retry=False,
        )

    def _set_test_unavailable(
        self, label: str, message: str, *, retry: bool = False
    ) -> None:
        try:
            button = self.query_one("#mcp-inspector-test-run", Button)
            status = self.query_one("#mcp-inspector-test-preview", Static)
            retry_button = self.query_one("#mcp-inspector-test-retry", Button)
        except NoMatches:
            return
        button.label = label
        button.disabled = True
        status.update(message)
        retry_button.display = retry
        retry_button.disabled = not retry

    def _handle_test_run(self) -> None:
        """Handle a Run press: collect arguments and dispatch a test run.

        Task 3 (PR-T3): both early returns below used to be silent -- a
        stray Run press reaching this method with no tool selected, or
        with the panel's own widgets not (yet, or no longer) mounted,
        produced no run AND no explanation. Both are defensive guards
        (the Run button only exists inside the panel `show_tool()` mounts
        for `self._current_tool`, so neither should be reachable via
        normal UI interaction), but "defensive" and "silent" don't have to
        mean the same thing -- a toast costs nothing and closes the last
        gap between "Run pressed" and "nothing visible happened".

        Review fix (Minor #6): the `tool is None` toast reuses
        `MCPWorkbench.open_test_for_selected_tool()`'s own house sentence
        for the identical "nothing selected" situation
        (`mcp_workbench.py`'s `"Select a tool in Tools mode first."`)
        rather than inventing a weaker synonym -- verb-first, matching the
        plan's register, and consistent copy for the same fact wherever it
        surfaces. Not imported (would reach into `mcp_workbench.py`, the
        wrong import direction -- see this module's own `_toast()`
        docstring); a small literal duplicate instead, same as `_toast()`.
        """
        tool = self._current_tool
        if tool is None:
            self.app.notify(
                _toast("Select a tool in Tools mode first."), severity="warning"
            )
            return
        try:
            form = self.query_one("#mcp-inspector-test-form", MCPSchemaForm)
            result_widget = self.query_one("#mcp-inspector-test-result", Static)
            run_button = self.query_one("#mcp-inspector-test-run", Button)
        except NoMatches:
            self.app.notify(
                _toast(
                    f"{tool.name}: the test panel isn't ready — reopen it and try again."
                ),
                severity="warning",
            )
            return
        preview = self._test_preview
        if preview is None:
            self.show_test_unavailable("No current permission preview is available.")
            return
        try:
            arguments = form.collect_arguments()
        except ValueError as exc:
            # F4 (PR-T3 task 3): this used to be the one result write in
            # this module with NO status prefix at all -- every other
            # write here leads with "OK"/"Failed"/"Blocked · not run"; a
            # bare exception message read as if the whole panel were
            # broken rather than "fix your input and press Run again".
            result_widget.update(f"Failed\n{_safe_tool_test_text(exc)}")
            return
        run_button.disabled = True
        intent = "approve_once" if preview.rendered_gate == "ask" else "run"
        self.post_message(
            self.ToolTestRequested(
                tool.server_key,
                tool.name,
                arguments,
                preview_nonce=preview.nonce,
                intent=intent,
            )
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        """Keep form edits local; the service canonicalizes current arguments."""
        event.stop()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Keep form edits local; the preview binds identity, not form state."""
        event.stop()

    @on(TextArea.Changed, "#mcp-schema-raw")
    def _on_test_form_raw_payload_changed(self, event: TextArea.Changed) -> None:
        """Keep raw form edits local until the one-click request is posted."""
        event.stop()

    def show_tool_result(
        self,
        *,
        server_key: str,
        tool_name: str,
        ok: bool,
        text: str | None = None,
        duration_ms: float | None = None,
        result: object = None,
        source: str | None = None,
        raw: str | None = None,
        blocked: bool = False,
        admission_changed: bool = False,
        decision_note: str | None = None,
        show_permission_jump: bool = True,
    ) -> None:
        """Render one Test Tool run's outcome, and re-enable Run.

        Tolerant of the panel having been closed (or a different tool
        selected) while the run was in flight -- `MCPWorkbench` posts this
        purely as "here's what happened", with no guarantee the panel this
        result belongs to is still on screen.

        I1: `(server_key, tool_name)` must match `self._current_tool`'s own
        fields -- a slow tool A's result arriving after the user has already
        switched the inspector to tool B's panel must never render under B
        (and must never re-enable B's Run button, which has nothing to do
        with A's completion). A mismatched result is dropped silently
        (debug-logged only) -- render-wise, it belongs to a panel that is no
        longer showing, and the protected stale-drop tests
        (`Tests/UI/test_mcp_inspector.py`) pin that silence deliberately.
        Task 3 (PR-T3): the render drop is NOT the whole story though -- the
        run itself really did complete, so this (and the sibling `NoMatches`
        drop just below, the panel-closed-but-same-tool case) now ALSO
        fires a toast naming the tool. A toast is a different surface from
        the dropped render; it does not touch the protected pin.

        `show_permission_jump` (Task 3, F4): the "Change in Permissions"
        jump button is only meaningful for the ONE Hub Tool Permissions
        matrix `blocked=True` already covers (the deny-gate short-circuit,
        Task 5) -- a refusal from a DIFFERENT permission system entirely
        (e.g. runtime governance's `PermissionError`, reclassified to
        `blocked=True` by `MCPWorkbench._run_tool_test()`) has no matching
        Hub Permissions row to jump to, so that call site passes `False`.
        Defaults `True` to reproduce every pre-Task-3 `blocked=True` call
        site's behavior unchanged.

        `blocked` (Task 5, UX batch item 5): True for the permissions
        deny-gate's synthetic result -- the call never reached the tool at
        all, so the status line reads "Blocked · not run" instead of
        routing through the ok/duration_ms failure template ("Failed ·
        0ms"), which would misleadingly imply an attempted, timed run.
        `ok`/`duration_ms` are still accepted (the deny-gate call site
        passes its usual `ok=False, duration_ms=0`) but ignored for the
        status line when `blocked` is True.

        RAG-49 (PR-5 task 4): two ways to describe a successful (`ok=True`,
        not blocked) run's content now coexist, kept keyword-only so Task 5
        can add one more optional kwarg (`decision_note`) cleanly:
          - `text` given (legacy call shape, still used by several
            pre-existing callers/tests): rendered inline exactly as before
            -- `f"{status_line}\\n{text}"`, no structured summary, no
            interpretation line, no raw Collapsible content.
          - `text` omitted: the NEW structured shape -- `result`/`source`
            feed `_summarize_tool_result()` to build the status line (`OK ·
            <source> · <duration> · N results`, singular/empty/error-shape
            variants) plus an optional quiet interpretation line, and `raw`
            (a pre-formatted, already-redacted JSON string) fills the
            collapsed "Raw response" Collapsible, capped via
            `_format_raw_body()`.
        Failed (`ok=False`) and blocked paths are UNCHANGED either way --
        they always use `text` (or `""` when absent) exactly as before,
        with no structured summary, interpretation, or raw Collapsible
        content -- "Failure/blocked paths keep their existing rendering".

        `decision_note` names the service-owned permission decision the run
        dispatched under. It shares the `#mcp-inspector-test-result-note`
        Static with the structured shape's own quiet `interpretation` line
        above -- when both are present (a structured OK run with something
        to interpret) they stack, `decision_note` first, one per line; when
        only one is present (every failed/blocked/legacy-text run, or a
        structured run with nothing to interpret) that one renders alone.
        `None` (the default -- every pre-existing call site) reproduces the
        exact pre-Task-5 behavior: the note shows only `interpretation`, or
        nothing.
        """
        current = self._current_tool
        if (
            current is None
            or current.server_key != server_key
            or current.name != tool_name
        ):
            logger.debug(
                f"MCPInspector: dropping stale tool result for "
                f"server_key={server_key!r} tool_name={tool_name!r} "
                f"(current tool is "
                f"{(current.server_key, current.name) if current else None!r})"
            )
            # Task 3 (PR-T3): the RENDER stays dropped (protected pin --
            # this belongs to a panel that is no longer showing, and
            # showing it under a different tool's panel would be wrong,
            # not just late) but the run genuinely completed, so the user
            # hears about it via a toast instead of nothing at all.
            self.app.notify(_toast(_stale_result_toast_text(tool_name)))
            return
        try:
            result_widget = self.query_one("#mcp-inspector-test-result", Static)
        except NoMatches:
            # Same tool, but its Test Tool panel isn't mounted (e.g. Close
            # was pressed while this run was still in flight) -- nothing to
            # render into, but the run still completed.
            self.app.notify(_toast(_stale_result_toast_text(tool_name)))
            return

        interpretation: str | None = None
        if admission_changed:
            result_widget.update(
                f"Changed · not run\n{_safe_tool_test_text(text or '')}"
            )
        elif blocked:
            result_widget.update(
                f"{_ADVANCED_BLOCKED_HEADING}\n{_safe_tool_test_text(text or '')}"
            )
        elif not ok:
            status_line = f"Failed{_duration_segment(duration_ms)}"
            result_widget.update(f"{status_line}\n{_safe_tool_test_text(text or '')}")
        elif text is not None:
            # Legacy call shape: a pre-formatted body string, rendered
            # inline exactly as `show_tool_result()` always has -- no
            # structured summary, no interpretation, no raw Collapsible.
            status_line = f"OK{_duration_segment(duration_ms)}"
            result_widget.update(f"{status_line}\n{text}")
        else:
            status_line, interpretation = _summarize_tool_result(
                ok=True,
                duration_ms=duration_ms,
                source=source,
                result=result,
            )
            result_widget.update(status_line)

        try:
            note_widget = self.query_one("#mcp-inspector-test-result-note", Static)
        except NoMatches:
            pass
        else:
            # RAG-51 (task 5): `decision_note` and `interpretation` are
            # independent facts (why the run dispatched vs. what the result
            # means) that share this one Static -- stack them, decision_note
            # first, when both are present; either alone renders bare; empty
            # hides the widget exactly like the pre-Task-5 `interpretation`-
            # only contract did.
            note_text = "\n".join(
                line for line in (decision_note, interpretation) if line
            )
            note_widget.update(note_text)
            note_widget.display = bool(note_text)

        try:
            raw_collapsible = self.query_one(
                "#mcp-inspector-test-result-raw", Collapsible
            )
            raw_body_widget = raw_collapsible.query_one(
                "#mcp-inspector-test-result-raw-body", Static
            )
        except NoMatches:
            pass
        else:
            if raw:
                raw_body_widget.update(_format_raw_body(raw))
                raw_collapsible.display = True
            else:
                raw_body_widget.update("")
                raw_collapsible.display = False

        try:
            self.query_one(
                "#mcp-inspector-test-run", Button
            ).disabled = admission_changed
        except NoMatches:
            pass
        # Task 3 (MCP Hub Phase 6): `blocked=True` is the deny-gate's
        # synthetic result (this method's own docstring) -- reveal the jump
        # button there; any other outcome (a real, non-blocked run) hides it,
        # covering prepared Ask executions too.
        # Task 3 (PR-T3): `show_permission_jump=False` further suppresses it
        # for a `blocked=True` result that has nothing to do with the Hub
        # Permissions matrix (see this method's own docstring).
        try:
            goto_button = self.query_one("#mcp-inspector-goto-permission-test", Button)
        except NoMatches:
            pass
        else:
            goto_button.display = blocked and show_permission_jump

    # -- advanced escape hatch -----------------------------------------------

    def set_service_context(
        self,
        service: Any,
        sections: list[tuple[str, str]],
        *,
        source: str = "local",
        target_label: str | None = None,
    ) -> None:
        """Bind the Advanced pane to a service context (initial mount, or a
        rebind on a workbench source/target switch).

        `source`/`target_label` drive `#mcp-adv-object`'s "Showing: ..."
        label. The section resets to `sections[0]` and `#mcp-adv-content` is
        blanked SYNCHRONOUSLY (not just once the reload worker below
        resolves) so a rebind can never leave a previous object's rendered
        dump on screen, even for one frame (UX-inputs acceptance: "reopening
        never shows a previous object's facts").

        Task 5 (MCP Hub Phase 6): the workbench calls this UNCONDITIONALLY
        on every `reload()`/source switch/selection change -- including
        while Advanced is still hidden behind the opt-in reveal Button (see
        `compose()`), when none of the `#mcp-adv-*` widgets this method
        used to assume exist have been mounted at all. The context fields
        (`_service`/`_sections`/`_advanced_source`/`_advanced_target_label`)
        are always recorded regardless -- `_reveal_advanced()` replays this
        same call once the collapsible is mounted, so the widgets end up
        bound to whatever was last recorded rather than opening blank.
        """
        self._service = service
        self._sections = sections or [("Overview", "overview")]
        self._advanced_source = source
        self._advanced_target_label = target_label
        # Fix Round A, Item 3: a rebind means the user's attention moved to
        # a (possibly different) object -- any confirm armed for a
        # PREVIOUS `set_service_context()` call must not silently satisfy a
        # first press against whatever is showing now, even when the new
        # object's `tool.execute` template happens to render byte-identical
        # JSON to what was just armed (two default templates that are the
        # same text is entirely plausible, not a contrived edge case).
        # Cleared unconditionally, before the `_advanced_visible` early
        # return, since a rebind while hidden must not leave a stale arm
        # for the next reveal to (re-)inherit either.
        #
        # Fix Round E, Item 1: capture whether an arm existed BEFORE
        # clearing it -- the result-pane blank a few lines down must be
        # conditional on this (see that blank's own comment for why).
        _was_armed = self._advanced_confirm_key is not None
        self._advanced_confirm_key = None
        if not self._advanced_visible:
            return
        self.query_one("#mcp-adv-object", Static).update(self._advanced_object_label())
        self.query_one("#mcp-adv-content", Static).update("")
        # Fix Round C, Item 4: the confirm sentence
        # (`_ADVANCED_EXECUTE_CONFIRM`) renders into `#mcp-adv-result`, not
        # `#mcp-adv-content` -- blanking only the latter left "Runs <tool>
        # now — press Run Action again to confirm." on screen after a
        # rebind that had just disarmed it, so the very next press silently
        # re-arms and re-renders the identical string: the button reads as
        # dead for one press. Blank both on disarm.
        #
        # Fix Round E, Item 1 (review of Fix Round C): that blank used to
        # run UNCONDITIONALLY, which also erases genuine RUN OUTPUT sitting
        # in this same widget -- a completed `tool.execute` result, or a
        # "Blocked · not run" refusal -- the instant the user switches
        # section or object to go act on it (re-reading it then means
        # running the tool again). The confirm sentence only ever occupies
        # this pane while an arm is live, so only blank when one WAS live
        # (`_was_armed`, captured above before the clear) -- output left
        # behind while unarmed is never the stale confirm sentence, and
        # must survive the rebind.
        if _was_armed:
            self.query_one("#mcp-adv-result", Static).update("")
        section_select = self.query_one("#mcp-adv-section-select", Select)
        with section_select.prevent(Select.Changed):
            section_select.set_options(self._sections)
            section_select.value = self._sections[0][1]
        self._refresh_advanced_actions()
        # Task 5: a CALLABLE, not a pre-created coroutine -- `exclusive=True`
        # cancels any not-yet-started worker in this group, and a cancelled
        # pre-created coroutine that never ran emits a noisy "coroutine was
        # never awaited" RuntimeWarning. This matters on the reveal path,
        # where this schedule and the freshly-mounted section Select's own
        # mount-echo Changed (whose handler schedules the same group) land
        # back to back; a callable the worker never invoked leaks nothing.
        self.run_worker(
            partial(self._load_advanced_section, self._sections[0][1]),
            group="mcp-adv-section",
            exclusive=True,
        )

    def _refresh_advanced_actions(self) -> None:
        action_select = self.query_one("#mcp-adv-action-select", Select)
        payload = self.query_one("#mcp-adv-payload", TextArea)
        run_button = self.query_one("#mcp-adv-run", Button)
        # Legacy-parity: keep the current action selected across a section
        # switch when it's still offered by the new section's descriptor
        # set, instead of always resetting to the new section's first
        # option.
        previous_value = None if _is_blank(action_select.value) else action_select.value
        descriptors = []
        if self._service is not None:
            loader = getattr(self._service, "available_actions", None)
            if callable(loader):
                descriptors = [d for d in (loader() or []) if self._action_allowed(d)]
        self._action_templates = {
            str(d["name"]): str(d.get("payload_template") or "{}") for d in descriptors
        }
        hint = self.query_one("#mcp-adv-empty-hint", Static)
        # Fix Round I, Item 2: `payload.text = ...` below is a PROGRAMMATIC
        # rewrite, not a user edit -- `payload.prevent(TextArea.Changed)`
        # (mirroring `action_select`'s own `Select.Changed` prevent, an
        # instance-scoped guard that does nothing for a DIFFERENT widget's
        # messages, hence the separate call here) stops it from reaching
        # `_on_advanced_payload_changed()` at all, so a confirm armed
        # during THIS call's caller's own `await` (`_load_advanced_
        # section()` awaits `self._service.load_section()` before calling
        # this method) can never be silently disarmed by this rewrite --
        # see that handler's own docstring for the race this closes.
        with action_select.prevent(Select.Changed), payload.prevent(TextArea.Changed):
            if not descriptors:
                action_select.set_options([("No actions available", Select.BLANK)])
                action_select.value = Select.BLANK
                action_select.disabled = True
                run_button.disabled = True
                # Legacy behavior: a section with nothing to run resets the
                # payload editor rather than leaving a stale template behind.
                payload.text = "{}"
                hint.update(
                    "No actions for this section. Select External Servers or "
                    "Inventory to see runnable actions."
                )
                hint.display = True
                return
            options = [(str(d["label"]), str(d["name"])) for d in descriptors]
            option_values = [value for _, value in options]
            selected = (
                previous_value if previous_value in option_values else options[0][1]
            )
            action_select.set_options(options)
            action_select.value = selected
            action_select.disabled = False
            run_button.disabled = False
            payload.text = self._action_templates.get(selected, "{}")
            hint.display = False

    def _action_allowed(self, descriptor: dict[str, Any]) -> bool:
        """Mirror the legacy panel's policy gate; permissive only when seams absent.

        Two distinct cases:
        - Seams absent (no callable gate/override): permissive by design -
          this is the test-fake/degraded path where policy enforcement isn't
          wired up at all.
        - Seams present but the gate call raises: fail closed. A runtime
          error must never silently expose an action that policy might
          forbid, so we log and deny rather than swallow and allow.
        """
        gate = getattr(self.app, "require_ui_action_allowed", None)
        override = getattr(self._service, "runtime_state_override", None)
        if not callable(gate) or not callable(override):
            return True
        action_id = str(descriptor.get("action_id") or "")
        try:
            decision = gate(action_id=action_id, runtime_state_override=override())
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message(
                    "MCPInspector policy gate raised; failing closed", exc
                ),
            )
            return False
        return bool(getattr(decision, "allowed", True))

    async def _load_advanced_section(self, section: str) -> None:
        # Fix Round A, Item 3: a section change is an attention-moved
        # transition too -- called both from `set_service_context()`'s own
        # initial section load (already cleared there; redundant but
        # harmless here) and from `on_select_changed()` when the user picks
        # a DIFFERENT section directly, which `set_service_context()` never
        # sees. Same rug-pull-adjacent reasoning as that clear: a section's
        # `tool.execute` template can render byte-identical JSON to another
        # section's, and a stale arm from before the switch must not
        # silently satisfy this section's first press.
        # Fix Round E, Item 1: capture whether an arm existed BEFORE
        # clearing it, same as `set_service_context()`'s matching capture --
        # the blank below must not fire for a section change that finds
        # nothing armed.
        _was_armed = self._advanced_confirm_key is not None
        self._advanced_confirm_key = None
        # Fix Round C, Item 4: same reasoning as `set_service_context()`'s
        # own blank -- the confirm sentence lives in `#mcp-adv-result`, not
        # `#mcp-adv-content`, so clearing the arm without blanking this too
        # would leave a stale "press Run Action again to confirm" on screen
        # describing an arm that no longer exists. This method only ever
        # runs while Advanced is visible (`set_service_context()`'s own
        # guard, and `on_select_changed()`'s section-select can't fire
        # unmounted), so `#mcp-adv-result` is always present to blank.
        #
        # Fix Round E, Item 1 (review of Fix Round C): that blank used to be
        # unconditional, which erased genuine run output/refusal text on
        # every section change, armed or not -- see the identical fix and
        # fuller reasoning on `set_service_context()`'s own blank above.
        # Conditional on `_was_armed` for the same reason.
        if _was_armed:
            self.query_one("#mcp-adv-result", Static).update("")
        if self._service is None:
            return
        payload = await self._service.load_section(section)
        self.query_one("#mcp-adv-content", Static).update(
            _render_section_payload(section, payload)
        )
        # C2: available_actions() is section-dependent (mirrors the legacy
        # panel, unified_mcp_panel.py). `load_section()` above is what
        # actually moves the service's notion of "current section" forward,
        # so re-derive the action list only now that it reflects the section
        # this call just loaded -- otherwise governance/inventory/advanced
        # actions stay permanently unreachable after the first section.
        self._refresh_advanced_actions()

    def on_select_changed(self, event: Select.Changed) -> None:
        select_id = event.select.id or ""
        if select_id == "mcp-adv-section-select":
            event.stop()
            # Callable, not coroutine -- same rationale as
            # `set_service_context()`'s own schedule for this group.
            self.run_worker(
                partial(self._load_advanced_section, str(event.value)),
                group="mcp-adv-section",
                exclusive=True,
            )
        elif select_id == "mcp-adv-action-select":
            event.stop()
            # Fix Round E, Item 2: switching the action is a FOURTH trigger
            # that disarms a pending `tool.execute` confirm --
            # `_run_advanced_action()`'s own docstring already named it
            # ("switching action or editing the payload re-arms"), but this
            # handler never actually cleared `_advanced_confirm_key` on an
            # action switch, so a stale arm (and its rendered confirm
            # sentence) survived it: pressing Run then executed the NEWLY
            # selected action immediately, no confirm, under a sentence
            # still promising one. Section membership: `tool.execute` only
            # shares its `inventory` section with `resource.read` and
            # `prompt.get` (both reads) -- the destructive actions live in
            # other sections, which a section change already disarms (see
            # `_load_advanced_section()`).
            #
            # Fix Round G, Item 3 (review of Fix Round E): the closing
            # sentence here used to call this "a truthfulness defect on the
            # confirm text, not a path to an unconfirmed destructive
            # action." That is FALSE, and was verified false by mutation
            # (drop this clear, keep the `_was_armed` blank below): arm
            # `tool.execute`, switch to `resource.read`, switch BACK to
            # `tool.execute` -- `_refresh_advanced_actions()`/this handler
            # regenerate `tool.execute`'s payload template from the same
            # fixed string every time, so the round trip reproduces
            # BYTE-IDENTICAL JSON. Without this clear (and, as of Fix Round
            # G, Item 2, ALSO without `_on_advanced_payload_changed()`'s
            # independent clear below -- see that method's own docstring:
            # its cascade now covers for a dropped clear here on every
            # reachable action switch), the stale (never-cleared)
            # `_advanced_confirm_key` from the FIRST arm still equals the
            # confirm key `_run_advanced_action()` recomputes for the
            # second `tool.execute` selection, so its `!=` comparison is
            # False, the "arm and return" branch is skipped, and the very
            # next press runs `tool.execute` -- which executes arbitrary
            # built-in tools from raw JSON -- with NO confirm and a
            # completely blank pane. This clear is a genuine, independently
            # worth-keeping safety boundary (belt-and-braces with Item 2's
            # cascade, not superseded by it -- a future change to either
            # side could silently remove the OTHER'S coverage), not merely
            # cosmetic; see `test_action_switch_clears_the_confirm_key_
            # synchronously` (isolates THIS clear specifically) and
            # `test_action_switch_round_trip_does_not_execute_tool_execute_
            # unconfirmed` (the combined, genuine-UI-reachable consequence)
            # in test_mcp_inspector.py.
            _was_armed = self._advanced_confirm_key is not None
            self._advanced_confirm_key = None
            if _was_armed:
                self.query_one("#mcp-adv-result", Static).update("")
            if not _is_blank(event.value):
                # Fix Round I, Item 2: programmatic rewrite, not a user
                # edit -- see `_refresh_advanced_actions()`'s matching
                # `payload.prevent(...)` for the full rationale (this call
                # site's own clear two lines up already runs synchronously
                # right before this write with no `await` between them, so
                # in practice this was already a no-op; the prevent makes
                # that mechanical instead of order-dependent, the same
                # guarantee `_refresh_advanced_actions()` now carries).
                payload = self.query_one("#mcp-adv-payload", TextArea)
                with payload.prevent(TextArea.Changed):
                    payload.text = self._action_templates.get(str(event.value), "{}")
        elif select_id.startswith("mcp-schema-field-"):
            # The preview binds tool identity and gate; current arguments are
            # collected and canonicalized by the service on activation.
            event.stop()

    @on(TextArea.Changed, "#mcp-adv-payload")
    def _on_advanced_payload_changed(self, event: TextArea.Changed) -> None:
        """A genuine user edit to the Advanced payload disarms a pending
        `tool.execute` confirm. Item 2 (PR-T3 fix round G):
        `_run_advanced_action()`'s own docstring has always promised
        "switching action or editing the payload re-arms" -- the
        action-switch half was implemented (Fix Round E, Item 2,
        `on_select_changed()` above); the payload-edit half had NO handler
        at all (there was no `TextArea.Changed` listener anywhere in this
        module), so a `tool.execute` arm survived a payload edit with the
        STALE confirm sentence still on screen -- naming the OLD tool and
        promising a confirm the very next press would not give
        (`_run_advanced_action()`'s own `confirm_key` mismatch already
        made that outcome SAFE -- an edited payload always re-arms instead
        of running, never the reverse -- just not TRUTHFUL about what the
        pane was about to do).

        Fix Round I, Item 2 (review of Fix Round G): this docstring used to
        claim every programmatic `payload.text = ...` write "always runs
        AFTER that call site's own clear, so `_was_armed` is already False
        there and this is a no-op." True for `on_select_changed()`'s
        action-switch branch and `set_service_context()` (both clear, then
        write, with no `await` between the two) -- FALSE for
        `_load_advanced_section()`: its own clear runs BEFORE `await
        self._service.load_section(section)`, and `_refresh_advanced_
        actions()`'s payload write happens AFTER that await, a real yield
        point during which a Run Action press can arm a NEW confirm
        against whatever the OLD section still shows on screen. When that
        race landed, this handler saw the LATE, post-await write as an
        "edit" and silently disarmed a confirm the user never touched --
        the button reverting to "Run" and the confirm sentence vanishing
        through no action of theirs, direction-wise fail-safe (never an
        extra execution) but reintroducing the "button reads as dead for
        one press" symptom Fix Round G, Item 2 existed to eliminate, from
        the opposite side (a background load cancelling the user's arm,
        instead of an edit failing to cancel it).

        Fixed at the SOURCE rather than by inspecting who's calling: every
        programmatic write to `#mcp-adv-payload` (`_refresh_advanced_
        actions()`'s two `payload.text = ...` assignments, and
        `on_select_changed()`'s own) is now wrapped in
        `payload.prevent(TextArea.Changed)`, so `TextArea.load_text()`
        (what the `.text` setter calls, and what a real keystroke's
        `Edit()` also posts through) never even posts the message for a
        programmatic write. This handler now fires ONLY for a genuine user
        edit, unconditionally -- a mechanical guarantee, not an accounting
        of call-site ordering a future call site could silently violate
        again.

        Same `_was_armed` gate as every other disarm site in this class:
        only touch `#mcp-adv-result` when an arm was actually live, so
        real run output/a refusal sitting there while UNARMED survives
        ordinary typing. Deliberately does not touch `event.text_area`
        itself -- clearing state and updating a DIFFERENT widget never
        fights the user's cursor or selection in this one.
        """
        event.stop()
        _was_armed = self._advanced_confirm_key is not None
        self._advanced_confirm_key = None
        if _was_armed:
            self.query_one("#mcp-adv-result", Static).update("")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "mcp-inspector-advanced-reveal":
            event.stop()
            # Task 6 review fold: check-then-disable synchronously, BEFORE
            # scheduling. Textual's message pump runs this (synchronous,
            # non-`async def`) handler to full completion before it even
            # looks at the next queued message, so a second `Pressed`
            # already queued for this same button (the message-pump race
            # documented on mcp_workbench.py's profile-save Save button)
            # sees `disabled=True` here and bails before re-scheduling --
            # without the check, `exclusive=True` would let it CANCEL
            # worker A mid-save (`_reveal_advanced()`'s
            # `self._advanced_visible = True` already landed, but the
            # button relabel + collapsible mount never ran) and leave a
            # dead-looking button no further press could ever recover,
            # since every future call into `_reveal_advanced()` now
            # short-circuits on that same flag. F-053: each direction of
            # the toggle re-enables the button as its last step.
            if event.button.disabled:
                return
            event.button.disabled = True
            # A CALLABLE, not a pre-created coroutine -- same rationale as
            # `set_service_context()`'s own schedule for `mcp-adv-section`.
            self.run_worker(
                partial(self._toggle_advanced), group="mcp-adv-reveal", exclusive=True
            )
            return
        if button_id == "mcp-adv-run":
            event.stop()
            self.run_worker(
                self._run_advanced_action(), group="mcp-adv-run", exclusive=True
            )
            return
        if button_id == "mcp-inspector-cancel":
            event.stop()
            if self._snapshot is not None:
                self.post_message(self.CancelRequested(self._snapshot.server_key))
            return
        if button_id.startswith("mcp-inspector-action-"):
            event.stop()
            action = HubAction(button_id.removeprefix("mcp-inspector-action-"))
            server_key = self._snapshot.server_key if self._snapshot else None
            self.post_message(self.HubActionRequested(action, server_key))
            return
        if button_id == "mcp-inspector-test-tool":
            event.stop()
            # Disable synchronously (before dispatching the mount worker) so
            # a second Pressed already queued for this same button (the
            # message-pump race documented on mcp_workbench.py's
            # profile-save Save button) sees it disabled -- the panel's own
            # existence check in `_mount_test_tool_panel()` is the second
            # line of defense for the window before this takes effect.
            event.button.disabled = True
            self.run_worker(
                self._mount_test_tool_panel(),
                group="mcp-inspector-test-panel",
                exclusive=True,
            )
            return
        if button_id == "mcp-inspector-test-run":
            event.stop()
            self._handle_test_run()
            return
        if button_id == "mcp-inspector-test-retry":
            event.stop()
            tool = self._current_tool
            if tool is None:
                return
            nonce = self.clear_test_preview()
            if nonce:
                self.post_message(self.ToolTestPreviewRevocationRequested(nonce))
            self.show_test_preparing()
            self.post_message(self.ToolTestPreviewRequested(tool.server_key, tool.name))
            return
        if button_id == "mcp-inspector-test-close":
            event.stop()
            self.run_worker(
                self._close_test_tool_panel(),
                group="mcp-inspector-test-panel",
                exclusive=True,
            )
            return
        if button_id == "mcp-inspector-reallow":
            event.stop()
            tool = self._current_permission_tool
            if tool is not None:
                self.post_message(self.ReallowRequested(tool.server_key, tool.name))
            return
        if button_id == "mcp-inspector-goto-permission":
            # Task 3: the Tools-mode permission block's own jump button --
            # `_current_permission_tool` is the SAME tool `show_tool()`'s
            # `effective` block is currently describing (set by
            # `_render_permission_container()`, this button's own mount
            # site).
            event.stop()
            tool = self._current_permission_tool
            if tool is not None:
                self.post_message(
                    self.ChangeInPermissionsRequested(tool.server_key, tool.name)
                )
            return
        if button_id == "mcp-inspector-goto-permission-test":
            # Task 3: the Test Tool panel's own jump button -- always
            # describes `_current_tool` (the panel only ever exists for that
            # tool; see `_mount_test_tool_panel()`).
            event.stop()
            tool = self._current_tool
            if tool is not None:
                self.post_message(
                    self.ChangeInPermissionsRequested(tool.server_key, tool.name)
                )
            return
        if button_id == "mcp-audit-open-tool":
            event.stop()
            entry = self._current_audit_entry
            if entry is not None:
                self.post_message(
                    self.AuditOpenToolRequested(
                        str(entry.get("server_key") or ""),
                        str(entry.get("tool_name") or ""),
                    )
                )
            return
        if button_id == "mcp-audit-adjust-permission":
            event.stop()
            entry = self._current_audit_entry
            if entry is not None:
                self.post_message(
                    self.AuditAdjustPermissionRequested(
                        str(entry.get("server_key") or ""),
                        str(entry.get("tool_name") or ""),
                    )
                )
            return
        if button_id.startswith("mcp-finding-action-"):
            event.stop()
            action = HubAction(button_id.removeprefix("mcp-finding-action-"))
            self.post_message(
                self.HubActionRequested(action, self._current_finding_server_key)
            )
            return

    async def _run_advanced_action(self) -> None:
        """Run the selected Advanced action, confirming the executing one.

        Task 6 (PR-T3), Route B: `tool.execute` is the only descriptor here
        that runs a tool rather than reading or writing control-plane
        config, and the Hub's permission gate resolves nearly every tool to
        "ask" from this pane (`gate_tool_test_by_key()` cannot verify an
        `allow` without a live definition to hash, so it collapses one to
        "ask"). "Ask" needs a per-run approval, and this legacy panel has
        no confirm control of its own -- so the Run Action button IS the
        arm: the first press states what will run and arms; the second
        runs it. The arm is keyed to `(action, payload)`, so switching
        action or editing the payload re-arms rather than executing
        something the user never read.

        A REFUSAL -- the gate's hard "Off" refusal from
        `execute_advanced_tool()` (`MCPHubGateDeniedError`), the in-process
        runtime-governance profile's own denial (`local_control_service.
        MCPGovernanceDenied`), or a raw `tools/call` refused by the
        `runtime.request`/`runtime.batch` pre-dispatch scan
        (`RawToolCallRefusedError`) -- renders under `show_tool_result()`'s
        "Blocked · not run" heading. Nothing ran, so the generic "Action
        failed:" dump (which reads as an attempted, crashed call) would
        misdescribe it.

        Item 2 (PR-T3 fix round D): this used to match the bare
        `PermissionError` base class, which is too broad the same way Fix
        Round B narrowed the Test Tool runner's OWN classifier
        (`mcp_workbench._is_permission_refusal()`) -- a `PermissionError` a
        BUILT-IN TOOL'S OWN body raises (this pane can execute arbitrary
        built-in tools via raw JSON) would misrender as a refusal that
        never reached the tool, when the tool ran and that IS its failure.
        Narrowed to the three TYPED refusals above instead (matching
        `_is_permission_refusal()`'s own type-based precedent); an
        untyped/tool-body `PermissionError` now falls through to the
        generic `except Exception` branch below, same as any other crash.
        The three typed exceptions are each imported from where they are
        DEFINED (`local_control_service`, `unified_control_plane_service`,
        `local_runtime_delegate`), not reused from `mcp_workbench.py`'s own
        classifier -- `mcp_workbench.py` imports FROM this module, so the
        reverse import would be circular.

        Item 6 (PR-T3 fix round F): this tuple is DELIBERATELY narrower
        than it could be, and DIFFERENT from `mcp_workbench.
        _is_permission_refusal()`'s own set (`MCPGovernanceDenied,
        MCPServerSourceDisplayOnlyError`) -- the two independently encode
        refusal-type knowledge and nothing besides this comment (and its
        twin over there) ties them together, so a fifth typed refusal
        added later has both to update. Not a defect today:
        `MCPServerSourceDisplayOnlyError` is excluded here because it is
        unreachable from every action this runner can dispatch --
        `execute_advanced_tool()` (the `tool.execute` action) hardcodes
        `BUILTIN_SERVER_KEY` when it calls `execute_hub_tool()`, so the
        server-source branch that raises it can never fire from this path,
        and `runtime.request`/`runtime.batch` never call `execute_hub_
        tool()` at all. Do not merge the two sets into one: each is
        correct for its own surface, and the asymmetry is what's true, not
        an oversight -- see `_is_permission_refusal()`'s own comment for
        why `MCPHubGateDeniedError`/`RawToolCallRefusedError` are excluded
        there.
        """
        result_widget = self.query_one("#mcp-adv-result", Static)
        action_select = self.query_one("#mcp-adv-action-select", Select)
        if self._service is None or _is_blank(action_select.value):
            return
        action_name = str(action_select.value)
        raw = self.query_one("#mcp-adv-payload", TextArea).text or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._advanced_confirm_key = None
            result_widget.update(f"Invalid JSON payload: {exc}")
            return
        if action_name == _ADVANCED_EXECUTE_ACTION:
            # `default=str` for the same reason the result dump below uses
            # it: this is arbitrary user JSON, and an un-dumpable payload
            # must re-arm, not raise out of the Run button's worker.
            confirm_key = (
                action_name,
                json.dumps(payload, sort_keys=True, default=str),
            )
            if self._advanced_confirm_key != confirm_key:
                self._advanced_confirm_key = confirm_key
                tool_label = (
                    str(payload.get("tool_name") or "").strip()
                    if isinstance(payload, Mapping)
                    else ""
                )
                result_widget.update(
                    _ADVANCED_EXECUTE_CONFIRM.format(tool=tool_label or "this tool")
                )
                return
        self._advanced_confirm_key = None
        try:
            result = await self._service.run_action(action_name, payload)
        except (
            MCPGovernanceDenied,
            MCPHubGateDeniedError,
            RawToolCallRefusedError,
        ) as exc:  # a refusal is not a failure
            result_widget.update(f"{_ADVANCED_BLOCKED_HEADING}\n{exc}")
            return
        except Exception as exc:  # surface, never crash the inspector -- a
            # tool-body `PermissionError` (not one of the three typed
            # refusals above) lands here too, same as any other crash.
            result_widget.update(f"Action failed: {exc}")
            return
        if isinstance(result, dict):
            result = redact_mapping(result)
        result_widget.update(json.dumps(result, default=str, indent=1)[:2000])
