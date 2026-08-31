# tldw_chatbook/UI/MCP_Modules/mcp_workbench.py
"""MCP Hub workbench: rail + mode canvases + inspector assembly."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger
from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.css.query import QueryError
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import ContentSwitcher
from textual.worker import Worker

from tldw_chatbook.Agents.builtin_tool_gate import (
    LOCAL_TOOLS_DEFAULT_ENABLED,
    builtin_permission_rows,
    tool_gate_breadcrumb,
)

# task-24458: the workspace tool-execution providers are deferred to their
# runtime use sites here for the same reason as in
# `Chat/console_chat_controller.py`. This module is reached by the SCREEN
# PRE-IMPORTER, so a module-scope import puts `Tools.workspace_tool_executor`
# and ~9 further modules into the pre-import payload -- work that belongs to
# the moment a workspace tool actually runs, not to every start. Every
# annotation in this module is a string (`from __future__ import annotations`
# above), so no type reference here evaluates at runtime.
from tldw_chatbook.config import (
    coerce_bool_setting,
    get_cli_setting,
    save_setting_to_cli_config,
)
from tldw_chatbook.MCP.hub_tool_catalog import (
    HubTool,
    builtin_tools_from_inventory,
    local_tools_from_record,
    server_tools_from_inventory,
)
from tldw_chatbook.MCP.hub_test_execution import (
    LocalHubExecutionOutcome,
    ToolTestAdmissionBlocked,
    ToolTestAdmissionPreview,
    ToolTestAdmissionStale,
)
from tldw_chatbook.MCP.local_control_service import MCPGovernanceDenied
from tldw_chatbook.MCP.local_runtime_delegate import PERMISSION_STATE_UNRESOLVED_CLAUSE
from tldw_chatbook.MCP.local_server_tools import resolve_server_workspace_root
from tldw_chatbook.MCP.mcp_import import ImportCandidate
from tldw_chatbook.MCP.permission_store import (
    BUILTIN_DEFAULT_STATE,
    BUILTIN_TOOL_SERVER_KEY,
    DEFAULT_GLOBAL,
    STORE_STATES,
    EffectiveToolState,
)
from tldw_chatbook.MCP.readiness import (
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
    as_checking,
    builtin_readiness,
    is_off_opt_in,
    local_profile_readiness,
    server_external_record_readiness,
    server_target_readiness,
)
from tldw_chatbook.MCP.redaction import redact_args, redact_mapping
from tldw_chatbook.MCP.unified_control_plane_service import (
    MCPServerSourceDisplayOnlyError,
)
from tldw_chatbook.UI.MCP_Modules.mcp_audit_mode import MCPAuditMode
from tldw_chatbook.UI.MCP_Modules.mcp_inspector import (
    _safe_diagnostic_message,
    _safe_exception_text,
    _safe_tool_test_text,
    MCPInspector,
)
from tldw_chatbook.UI.MCP_Modules.mcp_permissions_mode import (
    MCPPermissionsMode,
    PermRow,
    format_tool_state_label,
)
from tldw_chatbook.UI.MCP_Modules.mcp_profile_form import MCPImportPanel, MCPProfileForm
from tldw_chatbook.UI.MCP_Modules.mcp_rail import MCPRail
from tldw_chatbook.UI.MCP_Modules.mcp_server_mutations import MCPServerMutationsPanel
from tldw_chatbook.UI.MCP_Modules.mcp_servers_mode import MCPServersMode
from tldw_chatbook.UI.MCP_Modules.mcp_tools_mode import MCPToolsMode
from tldw_chatbook.Utils.path_validation import is_safe_path, validate_path

# Sentinel distinguishing "key absent from a restore blob" from "key present
# with value None" -- see `_apply_view_state()`'s scope_ref handling.
_UNSET: Any = object()
_TOOL_TEST_ACTIVE_POLL_SECONDS = 0.3
_TOOL_TEST_BLOCKED_UNKNOWN_TEXT = f"Blocked — {PERMISSION_STATE_UNRESOLVED_CLAUSE}."


def _target_id_from_server_key(key: str | None) -> str | None:
    """Parse a server-target id out of a `"server:<id>"` or
    `"server:<id>/<sub>"` key (a target row directly, or an external-record
    row beneath it), or `None` for anything else (a `local:`/`builtin:` key,
    an empty string, or `None` itself).

    Shared by `MCPWorkbench._selected_target_id()` (parses the workbench's
    OWN rail/table selection) and `_refresh_server_discovery()` (New Minor
    2, MCP Hub Phase 6 finale -- parses an arbitrary triggering event's
    `server_key`, which need not match whatever is currently selected).
    """
    if not key or not key.startswith("server:"):
        return None
    remainder = key.split(":", 1)[1]
    return remainder.split("/", 1)[0] if remainder else None


# A pasted/imported mcpServers config JSON. 1MB comfortably covers even a
# large hand-authored config while catching anything clearly not one --
# mirrors attachment_core.MAX_ATTACHMENT_BYTES's constant style (a fixed
# cap, not a config knob, since this is a hard sanity limit not a user
# preference).
MAX_MCP_IMPORT_FILE_BYTES = 1024 * 1024  # 1MB cap for an imported config JSON


def _mcp_import_home() -> str:
    """Return the user-selected import containment root."""

    return os.path.expanduser("~")


def _hub_lifecycle_timeout_seconds() -> float:
    """Read `[mcp] hub_lifecycle_timeout_seconds`, falling back to 45s.

    Mirrors `UnifiedMCPControlPlaneService._lifecycle_timeout()`
    (unified_control_plane_service.py) so the CHECKING-state time bound
    shown here can never drift from the value that service actually
    enforces the timeout with -- including the malformed-config fallback:
    a non-numeric value (e.g. a user typo like "soon" in config.toml) must
    not raise out of the render path, it should just fall back to 45s same
    as the enforcement side does.
    """
    try:
        return float(get_cli_setting("mcp", "hub_lifecycle_timeout_seconds", 45))
    except (TypeError, ValueError):
        return 45.0


def _toast(text: str) -> str:
    """Escape a `notify()`-bound message before Rich's markup interpreter sees it.

    The local/service-backed stores keep profile ids, exception text, and
    import summaries RAW on purpose -- Static surfaces already render with
    `markup=False`, and inline callouts escape at their own render time --
    but `app.notify()` DOES interpret Rich markup in its message, so any of
    that stored text reaching a toast unescaped could inject markup/styling
    (e.g. a profile id of `"[red]x[/red]"`). This is the one additional
    layer that needs its own guard.
    """
    return escape_markup(text)


def _cycled_ui_label(state: str | None) -> str:
    """The mutation-echo word for a just-cycled TOOL-row state (Task 3, MCP
    Hub Phase 6) -- `"Inherit"` for `None` (`cycle_ui_state()`'s own Inherit
    rung), otherwise the same `EffectiveToolState.ui_label` word every other
    state-word surface in this module uses (`"Allow"|"Ask"|"Off"`). A plain
    `EffectiveToolState` can't represent Inherit at all (its `state` field
    is a bare `str`, not `str | None`), so that one case is spelled out
    directly rather than routed through it.
    """
    if state is None:
        return "Inherit"
    return EffectiveToolState(state=state, origin="tool_override").ui_label


# Task 3 (built-in permissions UI, TASK-627): the Permissions matrix's
# built-in-tool section label -- deliberately distinct from "tldw_chatbook"
# (`hub_tool_catalog.builtin_tools_from_inventory()`'s label for the
# built-in MCP *server*, key `"builtin:tldw_chatbook"`). That is a
# different execution path (an MCP-exposed wrapper) from the in-process
# agent-runtime built-ins (`BUILTIN_TOOL_SERVER_KEY`, `"agent:builtin"`)
# this section renders -- Constraint 3 requires the UI never let a user
# mistake one for the other.
_BUILTIN_SECTION_LABEL = "Built-in (agent runtime)"


def _resolve_raw_shell_state():
    """Late-bound `resolve_raw_shell_state` (task-24458 deferral)."""

    from tldw_chatbook.Agents.raw_shell_tool_provider import (
        resolve_raw_shell_state,
    )

    return resolve_raw_shell_state


def _is_raw_shell_tool(server_key: str, tool_name: str | None) -> bool:
    """Return whether one policy identity is the unsafe host shell."""

    from tldw_chatbook.Agents.raw_shell_tool_provider import (
        RAW_SHELL_SERVER_KEY,
        RAW_SHELL_TOOL_NAME,
    )

    return server_key == RAW_SHELL_SERVER_KEY and tool_name == RAW_SHELL_TOOL_NAME


def _project_raw_shell_store_state(state: str | None) -> str | None:
    """Map a generic stored rung to raw shell's Ask/Off-only policy."""

    if state is None:
        return None
    return "deny" if state == "deny" else "ask"


def _is_permission_refusal(exc: BaseException) -> bool:
    """Whether `exc` is a REFUSAL to run the tool -- the call never
    reached the tool at all -- rather than an actual run failure.

    F4 (PR-T3 task 3): two refusals used to propagate through
    `_run_tool_test()`'s generic `except Exception` and render as `Failed
    · Nms`, indistinguishable from a genuine tool crash or timeout:

      - `MCPGovernanceDenied`, raised by
        `local_control_service.LocalMCPControlService._require_runtime_
        governance_allowed()` when the in-process runtime-governance
        profile denies the action -- a DIFFERENT permission system from
        the Hub's own prepared Allow/Ask/Off admission.
      - `MCPServerSourceDisplayOnlyError`, raised by
        `unified_control_plane_service.UnifiedMCPControlPlaneService.
        execute_hub_tool()` for a server-source key -- structurally can't
        run from this path at all.

    A refusal is not a failure: it must read as `Blocked · not run`
    (`MCPInspector.show_tool_result(blocked=True, ...)`), not `Failed`,
    which would misleadingly imply an attempted, timed run.

    task-2537 + task-2539 (PR-T3 fix round B, item 3): this used to match
    `isinstance(exc, PermissionError)` (the bare base class) OR an EXACT
    string match against a `ValueError`'s text -- two independent
    defects:

      - too broad: ANY `PermissionError`, including one a tool's own
        `execute()` body raises for an unrelated reason (e.g. a genuine
        OS EACCES), would misrender as `Blocked · not run` -- claiming
        the call never reached the tool when it actually did, and the
        tool itself is what failed.
      - prose-dependent: the matched `ValueError` string was pinned only
        where it's RENDERED, never at its raise site -- an unrelated
        reword there would have silently reverted this classifier's own
        fix with a fully green suite.

    Both call sites now raise a DEDICATED subclass instead of the bare
    base class (`MCPGovernanceDenied(PermissionError)`,
    `MCPServerSourceDisplayOnlyError(ValueError)`) and this matches those
    TYPES, not the base classes and not the message text -- a tool-body
    `PermissionError` (not `MCPGovernanceDenied`) now falls through to the
    ordinary `Failed` path, and a reword of the display-only message no
    longer has any bearing on classification.

    Item 6 (PR-T3 fix round F): this set is DELIBERATELY narrower than
    `mcp_inspector._run_advanced_action()`'s own typed-refusal tuple
    (`MCPGovernanceDenied, MCPHubGateDeniedError, RawToolCallRefusedError`)
    -- the two lists encode refusal-type knowledge independently and
    nothing besides this comment (and its twin over there) ties them
    together, so a fifth typed refusal added later has both to update.
    Not a defect today: `MCPHubGateDeniedError` is raised only by
    `execute_advanced_tool()`. `RawToolCallRefusedError` has TWO raise
    sites -- `_refuse_raw_tool_call()` (same file as `execute_advanced_
    tool()`) AND `LocalMCPRuntimeDelegate.request()`'s own `tools/call`
    branch (`local_runtime_delegate.py`), the durable backstop that module
    documents at length (Fix Round H, PR-T3 review, Item 4: corrected --
    this used to say "raised only by `execute_advanced_tool()`/
    `_refuse_raw_tool_call()`", missing the delegate's own raise site).
    Both types are reachable only from the Advanced runner's `tool.
    execute`/`runtime.request`/`runtime.batch` actions and the runtime
    delegate's own protocol surface -- never from this prepared Test Tool
    path, which never calls into `LocalMCPRuntimeDelegate.
    request()` at all (Test Tool execution goes through `execute_hub_
    tool()` -> `LocalMCPControlService.execute_tool()` ->
    `LocalMCPRuntimeDelegate.execute_tool()`, not the raw protocol
    surface). Do not merge the two sets into one: each is correct for its
    own surface, and the asymmetry is what's true, not an oversight -- see
    `_run_advanced_action()`'s own comment for why
    `MCPServerSourceDisplayOnlyError` is excluded there.
    """
    return isinstance(exc, (MCPGovernanceDenied, MCPServerSourceDisplayOnlyError))


MCP_HUB_MODES: dict[str, dict[str, str]] = {
    "servers": {"label": "Servers", "button_id": "mcp-mode-servers", "placeholder": ""},
    # T5: Tools mode now hosts the real `MCPToolsMode` canvas (see compose())
    # -- "placeholder" is unused for it, same as "servers" above, kept "" for
    # shape parity with the other MCP_HUB_MODES entries.
    "tools": {"label": "Tools", "button_id": "mcp-mode-tools", "placeholder": ""},
    # T6: Permissions mode now hosts the real `MCPPermissionsMode` canvas
    # (see compose()) -- "placeholder" is unused for it, same as
    # "servers"/"tools" above, kept "" for shape parity with the remaining
    # MCP_HUB_MODES entries.
    "permissions": {
        "label": "Permissions",
        "button_id": "mcp-mode-permissions",
        "placeholder": "",
    },
    # T7 (MCP Hub Phase 5): Audit mode now hosts the real `MCPAuditMode`
    # canvas (see compose()) -- "placeholder" is unused for it, same as
    # "servers"/"tools"/"permissions" above, kept "" for shape parity. This
    # was the last MCP_HUB_MODES entry still on the generic phase-placeholder
    # path; compose()'s placeholder-rendering loop is gone along with it.
    "audit": {"label": "Audit", "button_id": "mcp-mode-audit", "placeholder": ""},
}

_LEGACY_SECTIONS = [
    ("Overview", "overview"),
    ("Inventory", "inventory"),
    ("External Servers", "external_servers"),
    ("Governance", "governance"),
    ("Advanced", "advanced"),
]

# F-057: terminal-width threshold (cols) below which `#mcp-hub-grid` gets
# the `.mcp-compact` class and the triad rebalances toward the canvas (see
# BUNDLED_CSS and its _agentic_terminal.tcss mirror).
_COMPACT_WIDTH = 120

# T5: local-profile lifecycle actions this workbench can dispatch, keyed by
# the short verb used throughout `_in_flight_action`/notifications. Maps to
# the typed T2 methods on `UnifiedMCPControlPlaneService` -- each raises with
# a user-ready message on failure and records its own attempt state, so the
# wrapper below must not re-record anything, just surface the result.
_LIFECYCLE_METHOD_NAMES: dict[str, str] = {
    "connect": "connect_local_profile",
    "test": "test_local_profile",
    "refresh": "refresh_local_profile",
    "disconnect": "disconnect_local_profile",
}

# Verb map from the inspector's HubActionRequested action to the lifecycle
# verb keys above -- only these three ever originate from the readiness
# action buttons (disconnect is a detail-view-only action, wired in T7).
_HUB_ACTION_TO_LIFECYCLE_VERB: dict[HubAction, str] = {
    HubAction.CONNECT: "connect",
    HubAction.VALIDATE: "test",
    HubAction.REFRESH_DISCOVERY: "refresh",
}

# Past-tense verb used in the success notification, e.g. "docs: connected — 3 tools."
_LIFECYCLE_PAST_TENSE: dict[str, str] = {
    "connect": "connected",
    "test": "checked",
    "refresh": "refreshed",
    "disconnect": "disconnected",
}

# T9: success-notify copy per server-mutation action name. A generic
# "<last segment> saved." fallback would read as "Create saved."/"Delete
# saved." for the slot actions -- ambiguous about *what* was created or
# deleted -- so every wired action gets its own sentence instead.
_SERVER_MUTATION_MESSAGES: dict[str, str] = {
    "external_server.create": "External server created.",
    "external_server.update": "External server updated.",
    "external_server.slot.create": "Credential slot added.",
    "external_server.slot.secret.set": "Secret set.",
    "external_server.slot.secret.clear": "Secret cleared.",
    "external_server.slot.delete": "Credential slot deleted.",
}


def _import_summary(succeeded: list[str], failed: list[tuple[str, str]]) -> str:
    """One notify-ready sentence covering a whole import batch.

    Every candidate is attempted regardless of an earlier failure (T8: "a
    failing save produces the summary notify without aborting the rest") --
    this renders whatever mix of successes/failures resulted into a single
    toast instead of one per candidate.
    """
    parts: list[str] = []
    if succeeded:
        parts.append(f"Imported {len(succeeded)}: {', '.join(succeeded)}.")
    if failed:
        failed_desc = ", ".join(
            f"{profile_id} ({error})" for profile_id, error in failed
        )
        parts.append(f"Failed {len(failed)}: {failed_desc}.")
    return " ".join(parts) if parts else "Nothing to import."


def _import_severity(succeeded: list[str], failed: list[tuple[str, str]]) -> str:
    if failed and not succeeded:
        return "error"
    if failed:
        return "warning"
    return "information"


def _redact_external_server_record(record: Any) -> Any:
    """Redact one external-server record before it can reach the legacy
    Advanced renderer (frozen `render_external_servers_section()` in
    unified_mcp_sections.py).

    That renderer keys local records by "name", which local profile dicts
    never have (they use "profile_id"), so its `item.get(key) or item`
    fallback prints the FULL RAW DICT per entry -- CLI args and env values
    included -- whenever the key doesn't match. Non-Mapping records (already
    a shape the renderer can't consume sensibly) pass through unchanged.
    """
    if not isinstance(record, Mapping):
        return record
    record = dict(record)
    args = record.get("args")
    if isinstance(args, (list, tuple)):
        # redact_args handles `--api-key VALUE` / `key=value` CLI-arg shapes
        # that redact_mapping's generic key-based redaction below doesn't
        # reach (args is a plain list of strings, not a mapping).
        record["args"] = redact_args([str(a) for a in args])
    return redact_mapping(record)


def _redact_external_servers_list(records: Any) -> Any:
    if not isinstance(records, list):
        return records
    return [_redact_external_server_record(r) for r in records]


class _AdvancedSectionShim:
    """Shields the inspector's legacy Advanced pane from two local-source gaps.

    `UnifiedMCPControlPlaneService.load_section()` returns a dict for every
    section except local-source "external_servers", which comes back as a
    bare list (mirroring `LocalMCPControlService.get_external_servers()`).
    The renderers in `unified_mcp_sections.py` all assume a Mapping, and
    `MCPInspector.set_service_context()`/`on_select_changed()` schedule the
    section load as a worker with Textual's default `exit_on_error=True` —
    that one shape mismatch (or a raised exception) there would crash the
    whole app, not just the Advanced pane. Normalize and fail closed here
    instead, at the integration seam this task owns, without touching
    mcp_inspector.py.

    Second gap: `render_external_servers_section()` (also frozen) keys
    records by "name", which local profile dicts never have -- its fallback
    then prints each FULL RAW DICT, secrets included (CLI args, env values).
    Records are redacted here, at this same seam, before the renderer ever
    sees them -- on both the bare-list local path and any dict payload that
    already carries an "external_servers" list (the server-source shape).
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    def __getattr__(self, name: str) -> Any:
        return getattr(self._service, name)

    async def load_section(self, section: str | None = None) -> dict[str, Any]:
        try:
            payload = await self._service.load_section(section)
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message(
                    "MCP workbench advanced section load failed", exc
                ),
            )
            return {
                "source": "local",
                "section": section or "overview",
                "error": _safe_exception_text(exc),
            }
        if isinstance(payload, dict):
            if isinstance(payload.get("external_servers"), list):
                payload = dict(payload)
                payload["external_servers"] = _redact_external_servers_list(
                    payload["external_servers"]
                )
            return payload
        if isinstance(payload, list) and section == "external_servers":
            # UnifiedMCPControlPlaneService.load_section() only returns a
            # bare list for the local-source "external_servers" section
            # (LocalMCPControlService.get_external_servers()); every other
            # section already comes back as a dict. render_external_servers_section
            # reads this key as a list.
            return {
                "source": "local",
                "section": section,
                "external_servers": _redact_external_servers_list(payload),
            }
        return {"source": "local", "section": section or "overview"}


class MCPWorkbench(Container):
    """Assembles the MCP Hub. Read-only over the control-plane service."""

    #: Whether a `reload()` is in flight. Distinct from `_reloading`, which is
    #: internal write-ordering state: this one is the UI-facing third state
    #: (loading / empty / populated) so an in-flight load is never rendered as
    #: "nothing here" -- the same distinction TASK-1020 drew for Watchlists.
    is_loading: reactive[bool] = reactive(False)

    class ModeChanged(Message, namespace="mcp_workbench"):
        """Posted by `set_mode()` whenever the active mode actually changes,
        so the hosting screen can keep its mode-chip highlight in sync.
        `set_mode` is the single emission point: it covers every path that
        changes the mode without going through `MCPScreen._activate_mode()`
        (a click or keybinding) -- state restore and inspector hub actions
        ("Open tool catalog"/"Open audit") alike. The screen's chip sync is
        idempotent, so the redundant notification on the _activate_mode
        path is harmless."""

        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

    BUNDLED_CSS = """
    MCPWorkbench {
        width: 100%;
        height: 1fr;
        min-height: 0;
    }
    #mcp-hub-grid {
        width: 100%;
        height: 100%;
        min-height: 0;
    }
    #mcp-hub-canvas {
        width: 5fr;
        min-width: 38;
        height: 100%;
        min-height: 0;
    }
    /* F-057: below ~120 cols (`.mcp-compact` on #mcp-hub-grid, toggled by
    `on_resize`) the triad rebalances toward the canvas so the servers
    table keeps its primary columns in-viewport; the rail/inspector take
    narrower shares + min-widths (their content wraps/truncates honestly --
    see #mcp-inspector-state's wrap override and MCPRail's width-aware row
    truncation budget). Bare-harness copy; the REAL app gets the identical
    rules from _agentic_terminal.tcss (app-tier CSS beats widget
    DEFAULT_CSS on ties in this Textual version -- the established lockstep
    pattern documented there). */
    #mcp-hub-grid.mcp-compact #mcp-hub-rail {
        width: 2fr;
        min-width: 16;
    }
    #mcp-hub-grid.mcp-compact #mcp-hub-canvas {
        width: 7fr;
        min-width: 30;
    }
    #mcp-hub-grid.mcp-compact #mcp-hub-inspector {
        width: 2fr;
        min-width: 20;
    }
    """

    def __init__(self, app_instance: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._app_instance = app_instance
        self._active_mode = "servers"
        #: Mode requested before the deferred canvases mounted (task-2901);
        #: replayed by `_mount_deferred_canvases`.
        self._pending_deferred_mode: str | None = None
        self._source = "local"
        self._selected_server_key: str | None = None
        # F-054: one-shot gate for `_preselect_single_problem_on_load()` --
        # True once the first load has had its chance to pre-select, so a
        # later resync can never re-hijack a selection the user cleared.
        self._did_initial_preselect: bool = False
        self._scope: str = "personal"
        self._scope_ref: str | None = None
        self._snapshots: list[ReadinessSnapshot] = []
        # T6: raw local-profile catalog records keyed by profile_id, kept in
        # sync with `_snapshots` by `_collect_snapshots()` -- readiness
        # snapshots don't carry every field the add/edit form needs
        # (profile_id/command/args/env_placeholders/env_literals), so
        # `show_form()` on an EDIT_CONFIG hub action looks the record up
        # here instead.
        self._catalog_records: dict[str, dict[str, Any]] = {}
        # T6: True while a profile-save worker is in flight. Mirrors
        # `_start_lifecycle()`'s synchronous-registration pattern: set in the
        # (sync) SubmitRequested handler before the worker is dispatched and
        # cleared in the worker's `finally`, so a second Save arriving in the
        # same pump window reliably observes it. Without this, dispatching
        # every submit through `run_worker(..., exclusive=True)` let a second
        # click CANCEL the in-flight save mid-write.
        self._profile_save_in_flight: bool = False
        # T8: True while an mcpServers-import apply worker is in flight. Same
        # synchronous-registration guard as `_profile_save_in_flight`: set
        # before the worker is dispatched, cleared in its `finally`, so a
        # second Import click during a slow batch can't dispatch a second
        # overlapping apply worker.
        self._profile_import_in_flight: bool = False
        # T7: True while a profile-delete worker is in flight. Same
        # synchronous-registration guard as `_profile_save_in_flight` above:
        # set before the worker is dispatched, cleared in its `finally`, so
        # a second DeleteConfirmed arriving in the same pump window (e.g. a
        # double-click on "Confirm delete" before the button unmounts)
        # cannot cancel the in-flight delete mid-write.
        self._profile_delete_in_flight: bool = False
        # T9: True while an external-server-record mutation worker
        # (external_server.create/update or a credential-slot action) is in
        # flight. Same synchronous-registration guard as the local-profile
        # flags above: set before dispatch, cleared in the worker's
        # `finally`.
        self._server_mutation_in_flight: bool = False
        # T9: whether the active server target's scope permits
        # `external_server.*` mutations (team/org/system-admin only --
        # `service.available_actions()` returns `[]` for "personal"). Reset
        # to False for local source and recomputed for server source in
        # `_collect_snapshots()` (whenever a target's external-servers
        # section is loaded, which pins the service context's
        # `selected_section` to "external_servers" -- see that method's
        # docstring for why this is read directly off `available_actions()`
        # instead of a synthetic select-then-restore round trip) and again,
        # cheaply, on scope changes (`on_mcp_rail_scope_changed`).
        self._server_mutations_available: bool = False
        # T5: in-flight local-profile lifecycle operations, keyed by
        # server_key ("local:<profile_id>"). While a key is present here,
        # `_snapshot_for_display()`/`_sync_children()` render that server as
        # CHECKING (see `as_checking()`) regardless of its last-known
        # readiness, and the inspector shows a Cancel button instead of the
        # normal action set.
        self._in_flight: dict[str, Worker] = {}
        # The lifecycle verb ("connect"/"test"/"refresh"/"disconnect") each
        # in-flight key is running, so the CHECKING badge's message and the
        # eventual notification can say what's actually happening.
        self._in_flight_action: dict[str, str] = {}
        # T12 review fix: identity of the object the inspector's Advanced
        # pane was last rebound to (see `_rebind_inspector_advanced_context`).
        # None until the first rebind so the mount-time reload() always
        # binds.
        self._advanced_rebind_key: tuple[Any, ...] | None = None
        self._pending_view_state: dict[str, Any] | None = None
        # T7: `_sync_children()` now flows through `MCPServersMode.show_detail()`,
        # which (since T7 added the detail toolbar) performs its own awaited
        # remove_children()/mount_all() cycle on `#mcp-detail-toolbar` --
        # real suspension points a concurrently *running* `_sync_children()`
        # call can interleave with. `_start_lifecycle()` deliberately fires
        # two independent workers (the immediate "mcp-lifecycle-sync" resync
        # for the optimistic CHECKING badge, and "mcp-lifecycle"'s own
        # finally-triggered resync once the lifecycle call completes) --
        # different worker groups, so Textual's `exclusive=True` cancellation
        # within one group does not serialize them against each other. This
        # lock does: a second `_sync_children()` call simply waits for the
        # first's remove+mount cycle to finish (harmless -- it repaints with
        # whatever the latest state is once it acquires the lock) instead of
        # racing it and raising DuplicateIds.
        #
        # DELIBERATELY BROAD (T7 review adjudication): review proposed
        # narrowing this to a toolbar-local lock per the widget-local
        # `MCPInspector._refresh_lock` convention; kept broad because beyond
        # the DuplicateIds fix it also guarantees whole-triad consistency --
        # rail, overview table, and inspector always render from the same
        # snapshot generation within one locked pass, so a torn state where
        # the rail shows refresh A while the table shows refresh B cannot
        # occur -- and it serializes the DataTable clear()/add_row() cycle in
        # `update_overview()` against concurrent workers. Relationship to
        # `MCPInspector._refresh_lock`: that one is the inspector's own
        # widget-local guard for its external callers (worker-driven Advanced
        # refreshes racing pump-driven ones); on the `_sync_children()` path
        # it is always acquired AFTER this lock (via `update_readiness()`
        # inside the locked block), and no code path acquires them in the
        # opposite order -- same ordering everywhere, no AB-BA deadlock.
        self._sync_children_lock = asyncio.Lock()
        # Guards the post-mount restore race: `on_mount` awaits `reload()`
        # inline, but a caller (e.g. the destination screen) can call
        # `set_initial_view_state()` while that reload is still in flight.
        # Without this flag, the reload's own `_sync_children()` and a
        # concurrently scheduled restore worker can race to write
        # `_source`/`_selected_server_key`/`_scope`/`_scope_ref` last.
        self._reloading: bool = False
        # T6: the cross-server `HubTool` catalog `_sync_tools_mode()` most
        # recently derived (via `_collect_hub_tools()`) -- `HubTool.tool_id`
        # (the Tools-mode DataTable's row key) is looked up against this
        # cache when `MCPToolsMode.ToolSelected` arrives, rather than
        # re-deriving the whole catalog on every selection.
        self._last_hub_tools: list[HubTool] = []
        # Presentation generation only. Service preview consumption and its
        # active registry remain the execution authority.
        self._tool_test_generation: int = 0
        # Presentation bookkeeping only. The service registry remains the
        # authority; this copy exists because Textual unmounts descendants
        # before the parent's ``on_unmount`` can query the inspector.
        self._tool_test_preview_nonce: str | None = None
        # Reclamation must outlive the cancelled Textual worker which minted
        # the nonce. Retain each cleanup until its result has been observed.
        self._tool_test_reclaim_tasks: set[asyncio.Task[None]] = set()
        # T7: the batch `EffectiveToolState` resolution `_sync_permissions_
        # mode()` most recently computed (via `service.effective_tool_
        # states()`), keyed the same as that method's own return value --
        # `_effective_for_display()` prefers this cache over a second,
        # redundant per-tool resolution when explaining a tool's permission
        # rule (Tools-mode's `show_tool(tool, effective=...)` and
        # Permissions-mode's own `show_permission()`).
        self._last_effective_states: dict[tuple[str, str], EffectiveToolState] = {}
        # Task 3 (MCP Hub Phase 6): the raw per-tool cascade tuple
        # (`tool_entry_state`, `server_default`, `global_default`) --
        # `_build_permission_rows()`'s SAME raw STORE reads that already
        # produce `PermRow.cycle_current`/`server_cycle_current`/
        # `global_state`, just packaged one tuple per tool instead of split
        # across matrix rows. `_cascade_for_tool()` reads this to thread
        # `show_permission(..., cascade=...)` -- mirrors
        # `_last_effective_states`'s own "computed once per
        # `_sync_permissions_mode()` pass, reused rather than re-derived"
        # precedent immediately above.
        self._last_cascade: dict[
            tuple[str, str], tuple[str | None, str | None, str]
        ] = {}
        # Fix 1 (PR #906 review, post-TASK-627): the per-BUILT-IN-tool
        # `EffectiveToolState` `_builtin_permission_matrix_rows()` most
        # recently resolved (via `resolve_builtin_state`, not the MCP
        # `effective_tool_states()` batch above) -- keyed like
        # `_last_cascade` immediately above (server_key half is always
        # `BUILTIN_TOOL_SERVER_KEY` here, kept as a tuple for the same
        # lookup shape). `on_mcp_permissions_mode_row_selected()` reads
        # this to route a built-in `"tool"` row to the inspector's
        # permission view -- built-ins are NEVER in `_last_hub_tools`/
        # `effective` (Constraint 1/5), so without a cache of their own
        # that handler had no state to show and fell through to
        # `show_tool(None)`, blanking the inspector instead of explaining
        # the row.
        self._last_builtin_effective: dict[tuple[str, str], EffectiveToolState] = {}
        # T7 (MCP Hub Phase 5): the full (unfiltered) execution-log record
        # list `_sync_audit_mode()` most recently pushed into `MCPAuditMode`
        # -- `MCPAuditMode.EntrySelected.index` (a position in THAT SAME
        # list) is looked up against this cache when the event arrives,
        # rather than re-reading the log a second time.
        self._last_audit_entries: list[dict[str, Any]] = []
        # T8 (MCP Hub Phase 5): the full findings list `_sync_audit_mode()`
        # most recently pushed into `MCPAuditMode` -- `MCPAuditMode.
        # FindingSelected.index` is looked up against this cache when the
        # event arrives, mirroring `_last_audit_entries`/`EntrySelected`
        # immediately above.
        self._last_audit_findings: list[dict[str, Any]] = []
        # T8: this pass's server-source Audit-mode Findings fetch
        # (`_load_server_findings()`), cached by `(source, target)` --
        # mirrors `_governance_profiles_cache`/`_governance_profiles_
        # cache_key` (T11) exactly: STATIC server-side data, unaffected by
        # anything a client-side interaction can do, so repeated
        # `_sync_children()` passes under the same identity reuse this
        # instead of re-awaiting `load_section("advanced")` every time.
        # `_UNSET` seeds the key so the very first fetch always "changes"
        # it and fetches once. `_sync_audit_mode()` has no standalone
        # caller (unlike `_sync_permissions_mode()`'s three), so unlike
        # `_server_governance_profiles()` there is no separate `refresh`
        # gate here -- every call is part of a full `_sync_children()`
        # pass already.
        self._findings_cache: list[dict[str, Any]] | None = None
        self._findings_cache_key: tuple[str, str | None] | Any = _UNSET
        # T11: this pass's server-source governance-listing fetch
        # (`_load_server_governance_profiles()`), cached by `(source,
        # target)` identity -- that data is STATIC server-side profile
        # config, unaffected by anything a permission-matrix interaction
        # (Space-press cycle, kill-switch toggle, re-allow) can do, so a
        # standalone `_sync_permissions_mode()` resync from any of those
        # must reuse this instead of re-awaiting `load_section
        # ("governance")` on every keypress. `_UNSET` seeds the key so the
        # very first fetch (whatever source/target that turns out to be)
        # always "changes" it and fetches once; see `_server_governance_
        # profiles()` for the full cache/refresh contract.
        self._governance_profiles_cache: list[dict[str, Any]] | None = None
        self._governance_profiles_cache_key: tuple[str, str | None] | Any = _UNSET

    @property
    def active_mode(self) -> str:
        return self._active_mode

    @property
    def app_instance(self) -> Any:
        if self._app_instance is not None:
            return self._app_instance
        try:
            return self.app
        except Exception:
            return None

    def _service(self) -> Any:
        return getattr(self.app_instance, "unified_mcp_service", None)

    def compose(self) -> ComposeResult:
        with Horizontal(id="mcp-hub-grid", classes="destination-workbench"):
            yield MCPRail(
                source=self._source,
                snapshots=[],
                selected_server_key=None,
                scope_options=[("Personal", "personal")],
                scope_value=self._scope,
                scope_ref_options=[],
                scope_ref_value=self._scope_ref,
                id="mcp-hub-rail",
                classes="destination-workbench-pane",
            )
            with ContentSwitcher(
                initial="mcp-mode-canvas-servers",
                id="mcp-hub-canvas",
                classes="destination-workbench-pane",
            ):
                yield MCPServersMode(id="mcp-mode-canvas-servers")
                # task-2901: Tools/Permissions/Audit (T5/T6/T7 canvases —
                # every `MCP_HUB_MODES` entry is a real canvas) arrive
                # hidden behind the ContentSwitcher and are NOT composed
                # here. They mount as `_reload_guarded`'s first step
                # (`_mount_deferred_canvases`), before `reload()` pushes
                # data into them — off the click→paint critical path with
                # the load pipeline's ordering intact.
            yield MCPInspector(
                id="mcp-hub-inspector", classes="destination-workbench-pane"
            )

    def on_mount(self) -> None:
        """Mount now, load after (TASK-1320).
        This is deliberately SYNCHRONOUS. Textual awaits a widget's `on_mount`
        as part of mounting, and the app awaits the whole mount inside its own
        `NavigateToScreen` handler -- so awaiting the service here awaited it on
        the App's message pump, and the entire app stopped handling clicks,
        keys and further navigation until it answered. Against a configured but
        unreachable server that window was minutes, which users reported as the
        app freezing when they clicked into a screen.

        Scheduling `reload()` instead lets the canvas mount immediately in a
        loading state. `_reloading` already existed to guard a restore racing an
        in-flight reload, so a reload outliving mount is a case this widget was
        already built for -- it is simply the normal case now.
        """
        self.is_loading = True
        # Claim the reload SYNCHRONOUSLY, before yielding to the event loop.
        # `set_initial_view_state()` treats `_reloading` as "a reload owns the
        # restore, stash it for the end" -- and it is called by the destination
        # screen during this same mount. When `reload()` was awaited inline the
        # flag was already set by the time that happened; deferring the load
        # left it False, so the screen started its own restore worker and
        # applied the saved mode before the mode chips existed to hear
        # `ModeChanged`. Setting it here preserves the original ordering: the
        # restore is stashed and consumed at the end of `reload()`.
        self._reloading = True
        # `call_after_refresh`, not a bare `run_worker`: a widget's `on_mount`
        # fires once IT is mounted, before the children `compose()` yielded have
        # finished mounting. The old `await reload()` happened to be safe
        # because each await let the pending child mounts drain first; a worker
        # started here does not, and `_sync_children()` then tries to mount into
        # a canvas that does not exist yet ("Can't mount widget(s) before
        # Vertical(id='mcp-perm-server-profiles') is mounted"). Deferring one
        # refresh puts the load after the subtree has settled.
        self.call_after_refresh(self._start_initial_load)
        # F-057: set the initial compact-mode class once the first layout
        # gives the grid a real width (`on_resize` keeps it current after).
        self.call_after_refresh(self._sync_compact_class)

    async def on_unmount(self) -> None:
        """Invalidate preview work and revoke the visible nonce best effort."""
        self._tool_test_generation += 1
        nonce = self._tool_test_preview_nonce
        self._tool_test_preview_nonce = None
        try:
            inspector_nonce = self.query_one(MCPInspector).clear_test_preview()
        except Exception:
            inspector_nonce = None
        nonce = nonce or inspector_nonce
        await self._revoke_test_nonce(nonce)

    def on_resize(self) -> None:
        """F-057: keep the compact-mode class in step with the grid's width."""
        self._sync_compact_class()

    def _sync_compact_class(self) -> None:
        """Toggle `.mcp-compact` on `#mcp-hub-grid` below ~120 cols (F-057).

        The class drives the triad-rebalancing rules in BUNDLED_CSS (and
        their _agentic_terminal.tcss mirror): narrower rail/inspector
        shares so the canvas keeps its primary columns in-viewport. Width
        0 (pre-layout) means "not compact" -- the full triad renders first
        and the first real resize corrects it.
        """
        try:
            grid = self.query_one("#mcp-hub-grid")
        except Exception:
            # Pre-compose (or a torn-down subtree during unmount) -- nothing
            # to toggle yet; the post-mount call_after_refresh covers it.
            return
        width = self.size.width
        grid.set_class(0 < width < _COMPACT_WIDTH, "mcp-compact")

    def _start_initial_load(self) -> None:
        """Kick off the mount-time reload once the subtree is mounted."""
        # Re-assert the spinner now that the subtree has definitely settled.
        # `watch_is_loading` runs when `on_mount` sets the flag, and the canvas
        # happens to be queryable by then -- but that watcher tolerates a miss
        # rather than retrying, so re-applying here keeps a future change to
        # mount ordering from silently costing the loading state.
        self.watch_is_loading(self.is_loading)
        self.run_worker(
            self._reload_guarded(),
            group="mcp_workbench_reload",
            # A load failure is a broken destination, never a dead app. Textual
            # defaults this to True, so moving mount work into a worker would
            # otherwise turn any error `reload()` does not itself catch into an
            # app exit -- a failure mode that did not exist while the load ran
            # inside `on_mount`.
            exit_on_error=False,
            exclusive=True,
        )

    async def _mount_deferred_canvases(self) -> None:
        """Mount the three hidden mode canvases after first paint (task-2901).

        Runs as `_reload_guarded`'s first step, so every data push inside
        `reload()` (and everything after it) sees the full canvas set —
        the pipeline's unguarded `query_one(MCP*Mode)` sites stay valid by
        ordering, not by luck. A mode request that lands in the narrow
        pre-mount window is stashed by `set_mode` and replayed here.
        Idempotent for re-entered loads.
        """
        try:
            switcher = self.query_one(ContentSwitcher)
        except QueryError:
            return
        if not self.query(MCPToolsMode):
            tools = MCPToolsMode(id="mcp-mode-canvas-tools")
            permissions = MCPPermissionsMode(id="mcp-mode-canvas-permissions")
            audit = MCPAuditMode(id="mcp-mode-canvas-audit")
            # ContentSwitcher hides children from its `current` WATCHER, which
            # only fires on change — late-mounted children arrive visible, and
            # all three canvases briefly stacking pushed the current one's
            # content off-screen. Hide explicitly; `current = <mode>` shows
            # the right one when a mode change lands.
            for canvas in (tools, permissions, audit):
                canvas.display = False
            await switcher.mount(tools, permissions, audit)
        pending = self._pending_deferred_mode
        if pending is not None:
            self._pending_deferred_mode = None
            self.set_mode(pending)

    async def _reload_guarded(self) -> None:
        """Run the mount-time reload without letting a failure strand the UI."""
        try:
            await self._mount_deferred_canvases()
            if not self.query(MCPToolsMode):
                # Textual 8.2.8 can deliver the first after-refresh callback
                # before this widget's composed ContentSwitcher is queryable.
                # In that case _mount_deferred_canvases() deliberately returns;
                # queue one message-pump turn instead of letting reload() query
                # canvases that do not exist yet. The current exclusive worker
                # completes before this callback starts its replacement.
                self.call_later(self._start_initial_load)
                return
            await self.reload()
        except Exception as exc:
            # `reload()` clears `is_loading` in its own `finally`, but only for
            # paths that reach it; anything raised earlier would otherwise leave
            # the canvas spinning forever, telling the user data is coming when
            # nothing is.
            self.is_loading = False
            self._reloading = False
            # Do not attach Loguru's implicit traceback here: exception text can
            # contain credentials or local paths before our diagnostic boundary.
            logger.error(
                "MCP workbench initial load failed "
                "(source={}, scope={}, scope_ref={}, server_key={}, mode={}, "
                "exception_category={}).",
                self._source,
                self._scope,
                self._scope_ref,
                self._selected_server_key,
                self.active_mode,
                type(exc).__name__,
            )
            try:
                self.app.notify(
                    "Couldn't load MCP data. Use Refresh to try again.",
                    severity="error",
                )
            except Exception:
                pass

    def watch_is_loading(self, loading: bool) -> None:
        """Show the spinner over the canvas only, leaving the rail usable."""
        try:
            self.query_one("#mcp-hub-canvas").loading = loading
        except Exception:
            # Called before the canvas exists (the reactive is set in
            # `on_mount`, ahead of the first refresh) -- nothing to show yet.
            pass

    # -- data loading ---------------------------------------------------------

    async def reload(self) -> None:
        self._reloading = True
        # Raised here, not only in `on_mount`: `reload()` always CLEARS this in
        # its `finally`, so every caller must also raise it or a direct reload
        # (MCPScreen's manual refresh calls this straight) would fetch with no
        # spinner and then clear a flag it never set.
        self.is_loading = True
        try:
            service = self._service()
            if service is not None:
                try:
                    context = await service.load_context()
                    self._source = context.selected_source or "local"
                    if (
                        self._source == "server"
                        and context.selected_active_server_id
                        and self._selected_server_key is None
                    ):
                        self._selected_server_key = (
                            f"server:{context.selected_active_server_id}"
                        )
                    if context.selected_scope is not None:
                        self._scope = context.selected_scope
                    if context.selected_scope_ref is not None:
                        self._scope_ref = context.selected_scope_ref
                except Exception as exc:
                    logger.warning(
                        "{}",
                        _safe_diagnostic_message(
                            "MCP workbench context load failed", exc
                        ),
                    )
            self._snapshots = await self._collect_snapshots()
            self._preselect_single_problem_on_load()
            await self._sync_children()
            self._rebind_inspector_advanced_context(service)
        finally:
            self._reloading = False
            self.is_loading = False
        # Consume any view state that arrived while this reload was in
        # flight (see `set_initial_view_state()`), so it is applied exactly
        # once and always after this reload's own `_sync_children()`.
        await self._consume_pending_view_state()

    def _preselect_single_problem_on_load(self) -> None:
        """Pre-select on the workbench's first load (F-054, task-2240).

        When nothing is selected and exactly one server needs attention,
        the inspector should open on what's wrong and what you can do
        instead of dead space. task-2240: a LONE rail row is pre-selected
        the same way even when it isn't a "problem" -- the fresh-install
        rail is exactly one row (the off/opt-in built-in, which the
        problem rule below deliberately excludes), and its inspector
        detail (what it is, why it's off, the Enable affordance) is
        informational, not alarmist. Guarded to run at most once per mount
        (`_did_initial_preselect`) so a later resync (lifecycle
        completions, background refreshes) can never re-hijack a selection
        the user deliberately cleared -- and a restored view state
        (`_consume_pending_view_state()`, applied after this reload's sync)
        still wins over the heuristic, since that is explicit user state.

        "Problem" mirrors the Servers-mode recovery-callout definition:
        any state other than READY/CHECKING, excluding the off/opt-in
        built-in (`is_off_opt_in`, F-051) -- an off-by-choice server is not
        a problem to land on.
        """
        if self._did_initial_preselect or self._selected_server_key is not None:
            return
        self._did_initial_preselect = True
        problems = [
            snap
            for snap in self._snapshots
            if snap.state not in (ReadinessState.READY, ReadinessState.CHECKING)
            and not is_off_opt_in(snap)
        ]
        if len(problems) == 1:
            self._selected_server_key = problems[0].server_key
        elif len(self._snapshots) == 1:
            # task-2240: the lone rail row (fresh install's off/opt-in
            # built-in) is worth landing on too -- see the docstring.
            self._selected_server_key = self._snapshots[0].server_key

    def _selected_target_id(self) -> str | None:
        """The server-target id implied by `_selected_server_key`.

        Handles both a target row directly selected ("server:main") and an
        external-record row beneath it ("server:main/docs") -- both drill
        into the same target's external-servers listing. Thin wrapper over
        the module-level `_target_id_from_server_key()` (shared with
        `_refresh_server_discovery()`'s own, independent server_key).
        """
        return _target_id_from_server_key(self._selected_server_key)

    def _active_service_target_id(self) -> str | None:
        """The target id server-source operations would actually run against.

        UI selection wins when present, but `run_action`'s server branch
        resolves its target from the SERVICE context
        (`_require_active_server_target()` reads
        `context.selected_active_server_id`), not from the workbench's local
        selection -- and the two genuinely diverge: Add-server is only ever
        reachable from the overview, where `_selected_server_key` is None
        while the service still remembers the last-activated target. Falling
        back to the service context here keeps everything derived from this
        id (external-record loading, the post-create drill, the Add-server
        tooltip's target naming) consistent with where a mutation would
        really land.
        """
        target_id = self._selected_target_id()
        if target_id is not None:
            return target_id
        service = self._service()
        context = getattr(service, "context", None) if service is not None else None
        active = getattr(context, "selected_active_server_id", None)
        return str(active) if active else None

    def _active_target_label(self) -> str | None:
        """Human label for `_active_service_target_id()`'s target, or None.

        Prefers the target store's configured label; falls back to the raw
        id so the Add-server tooltip can always name a resolvable target.
        """
        target_id = self._active_service_target_id()
        if target_id is None:
            return None
        target_store = getattr(self._service(), "target_store", None)
        if target_store is not None:
            try:
                for target in target_store.list_targets():
                    if str(getattr(target, "server_id", "")) == target_id:
                        label = getattr(target, "label", None)
                        return str(label) if label else target_id
            except Exception as exc:
                logger.warning(
                    "{}",
                    _safe_diagnostic_message("MCP target label lookup failed", exc),
                )
        return target_id

    def _rebind_inspector_advanced_context(self, service: Any) -> None:
        """Push the current source/target into the inspector's Advanced pane.

        T12 (UX-inputs #1): "rebind or reset the section content whenever
        the selection changes so reopening never shows a previous object's
        facts; and label the object the content describes." Calling
        `set_service_context()` again resets the Advanced section back to
        its first entry and reloads it against the (possibly new) service
        context -- the same full rebind `reload()` already did on mount --
        so this is called from every place that changes which object the
        service context refers to: `reload()` itself, `_switch_source()`,
        and `_select_server_key()`.

        Review fix: rebinding is deduplicated on the OBJECT's identity, not
        on every call -- the UX-inputs text requires a rebind on selection
        CHANGE, and e.g. reclicking the already-selected rail row (or a
        no-op reload) is not a change; unconditionally rebinding there wiped
        the user's Advanced browsing state (section snapping back to
        Overview). Mirrors the C1 ScopeChanged dedup precedent in this
        file. The Advanced object is the local control plane (local source,
        regardless of which row is selected) or the active server target
        (server source) -- so the key is the source plus, for server source,
        the active target id/label; the service's identity is included so a
        swapped-in service (e.g. None -> real) always rebinds.
        """
        target_label = self._active_target_label()
        if self._source == "server":
            identity: Any = (self._active_service_target_id(), target_label)
        else:
            identity = None
        key = (id(service) if service is not None else None, self._source, identity)
        if key == self._advanced_rebind_key:
            return
        self._advanced_rebind_key = key
        self.query_one(MCPInspector).set_service_context(
            _AdvancedSectionShim(service) if service is not None else None,
            _LEGACY_SECTIONS,
            source=self._source,
            target_label=target_label,
        )

    @staticmethod
    def _is_external_record_key(server_key: str | None) -> bool:
        if not server_key or not server_key.startswith("server:"):
            return False
        remainder = server_key.split(":", 1)[1]
        return "/" in remainder

    def _compute_server_mutations_available(self, service: Any) -> bool:
        """Whether `external_server.*` mutation actions are usable right now.

        `available_actions()` only returns the `external_server.*` set when
        the service context's `selected_section` is "external_servers"
        (mirrors the legacy Advanced panel/inspector -- see
        mcp_inspector.py's `_load_advanced_section` C2 comment). Rather than
        issuing a synthetic `select_section("external_servers")` +
        `available_actions()` + restore-previous-section round trip purely
        to answer this question, this piggybacks on the read that
        `_collect_snapshots()` already performs for real, functional reasons
        whenever a server target is selected: loading that target's
        external-servers section (to render its record rows) pins
        `selected_section` to "external_servers" as a side effect of real
        navigation, so `available_actions()` called right after is accurate
        with no extra round trip and no context left mutated beyond what the
        UI was already doing. When no target is ACTIVE at all
        (`_active_service_target_id()` is None -- neither a UI selection nor
        a service-remembered target), that load never ran and
        `selected_section` may be stale -- this then reads as unavailable,
        which happens to also be the honest answer: without an active
        target, `external_server.create` has nowhere to attach anyway (and
        the Add-server button additionally carries its own no-target gate,
        see `MCPServersMode._update_add_server_button()`).
        """
        if service is None:
            return False
        loader = getattr(service, "available_actions", None)
        if not callable(loader):
            return False
        try:
            actions = loader() or []
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message("MCP available_actions check failed", exc),
            )
            return False
        return any(
            isinstance(a, Mapping) and a.get("name") == "external_server.create"
            for a in actions
        )

    async def _collect_snapshots(self) -> list[ReadinessSnapshot]:
        snapshots: list[ReadinessSnapshot] = []
        service = self._service()
        if self._source == "local":
            self._server_mutations_available = False
            snapshots.append(
                builtin_readiness(
                    enabled=bool(get_cli_setting("mcp", "enabled", False)),
                    expose_tools=bool(get_cli_setting("mcp", "expose_tools", True)),
                    expose_resources=bool(
                        get_cli_setting("mcp", "expose_resources", True)
                    ),
                    expose_prompts=bool(get_cli_setting("mcp", "expose_prompts", True)),
                )
            )
            if service is not None:
                # T5: `local_external_catalog()` (T2) additionally attaches
                # each record's persisted `runtime_state` (last connect/test/
                # refresh attempt), which `local_profile_readiness()` uses to
                # surface a specific failure reason instead of the generic
                # "not currently connected". Fall back to the Phase 1 path
                # for any service that doesn't expose it yet (older fakes).
                catalog_loader = getattr(service, "local_external_catalog", None)
                try:
                    if callable(catalog_loader):
                        records = await catalog_loader()
                    else:
                        records = await service.load_section("external_servers")
                except Exception as exc:
                    logger.warning(
                        "{}",
                        _safe_diagnostic_message(
                            "MCP local profile listing failed", exc
                        ),
                    )
                    records = []
                if isinstance(records, list):  # local source returns a bare list
                    snapshots.extend(local_profile_readiness(r) for r in records)
                    self._catalog_records = {
                        str(r.get("profile_id")): dict(r)
                        for r in records
                        if isinstance(r, Mapping) and r.get("profile_id")
                    }
        else:
            target_store = getattr(service, "target_store", None)
            if target_store is not None:
                snapshots.extend(
                    server_target_readiness(t) for t in target_store.list_targets()
                )
            # T9: with an ACTIVE target (a target/external-record row
            # selected in the UI, or -- review fix -- the service context's
            # remembered active target when nothing is visibly selected),
            # also load and append that target's external-server records --
            # they appear in rail/table beneath the target, keyed
            # "server:<target>/<ext>". Gating on the service's notion of
            # active (not just the UI selection) is what lets a freshly
            # created record show up immediately: Add-server runs from the
            # overview with no selection at all.
            target_id = self._active_service_target_id()
            if service is not None and target_id is not None:
                try:
                    payload = await service.load_section("external_servers")
                except Exception as exc:
                    logger.warning(
                        "{}",
                        _safe_diagnostic_message(
                            "MCP external server listing failed", exc
                        ),
                    )
                    payload = None
                records = (
                    payload.get("external_servers")
                    if isinstance(payload, Mapping)
                    else None
                )
                if isinstance(records, list):
                    snapshots.extend(
                        server_external_record_readiness(r, server_id=target_id)
                        for r in records
                        if isinstance(r, Mapping)
                    )
            self._server_mutations_available = self._compute_server_mutations_available(
                service
            )
        return snapshots

    def _snapshot_for(self, server_key: str | None) -> ReadinessSnapshot | None:
        if server_key is None:
            return None
        for snap in self._snapshots:
            if snap.server_key == server_key:
                return snap
        return None

    def _display_snapshot(self, snapshot: ReadinessSnapshot) -> ReadinessSnapshot:
        """Overlay the in-flight CHECKING state onto a snapshot for rendering.

        `self._snapshots` itself always holds the last *derived* readiness
        (from `_collect_snapshots()`) so a cancelled or failed lifecycle
        action has something correct to fall back to -- this wraps it with
        `as_checking()` only for display, purely based on whether the key is
        currently in `self._in_flight`.

        T7 (P3 UX batch): the CHECKING message ("Working — <action>…") had
        no indication of how long a stuck lifecycle op might sit there --
        this appends a time bound, e.g. "connect (up to 45s)", read straight
        from the same `[mcp] hub_lifecycle_timeout_seconds` setting
        `UnifiedMCPControlPlaneService._lifecycle_timeout()` already uses to
        actually enforce the timeout (unified_control_plane_service.py), so
        the copy can never drift from the real bound without touching
        readiness.py's `as_checking()` itself.
        """
        if snapshot.server_key in self._in_flight:
            action = self._in_flight_action.get(snapshot.server_key, "update")
            timeout_seconds = _hub_lifecycle_timeout_seconds()
            bounded_action = f"{action} (up to {int(round(timeout_seconds))}s)"
            return as_checking(snapshot, bounded_action)
        return snapshot

    def _snapshot_for_display(self, server_key: str | None) -> ReadinessSnapshot | None:
        snapshot = self._snapshot_for(server_key)
        return None if snapshot is None else self._display_snapshot(snapshot)

    async def _sync_children(self) -> None:
        """Push current state into the rail/canvas/inspector children.

        Awaited end to end -- `MCPInspector.update_readiness()` must fully
        finish its remove+mount cycle (see mcp_inspector.py) before this
        coroutine returns, so that Textual's message pump cannot dequeue a
        second selection event and start another `_sync_children()` call
        while the first's inspector refresh is still settling
        (`DuplicateIds` regression; see test_mcp_inspector.py).

        Wrapped in `_sync_children_lock` (T7): that guards the *pump*
        ordering above, but `_start_lifecycle()` also fires this method from
        two independent worker groups (the immediate optimistic-CHECKING
        resync and the lifecycle wrapper's own completion resync), which can
        genuinely run concurrently as separate asyncio tasks -- the lock
        serializes those too, so two overlapping calls' remove+mount cycles
        (now real suspension points, since `MCPServersMode.show_detail()`
        rebuilds the detail toolbar) can't interleave on the same widgets.

        T10: `_collect_hub_tools()` and `_resolve_effective_states()` are
        each called EXACTLY ONCE per pass, right here, and threaded into
        both `_sync_tools_mode()` (the State column) and
        `_sync_permissions_mode()` (the matrix rows) -- both modes render
        the SAME tool list against the SAME effective-state resolution for
        one pass, so there is no reason to pay for a second full store
        load (plus any mark/audit side effects `effective_tool_states()`
        performs) back-to-back. The three STANDALONE callers of
        `_sync_permissions_mode()` (Space-press cycle, kill-switch toggle,
        re-allow) deliberately do NOT go through this path -- they call it
        with no `effective` argument, so it resolves fresh (see that
        method's docstring for why that's correct there: each of those
        handlers just mutated the store itself).
        """
        async with self._sync_children_lock:
            # Lifecycle and restore workers may request a sync during the same
            # Textual-floor pre-mount window as the initial load. Establish the
            # deferred-canvas invariant at this shared boundary, under the lock
            # that already serializes every sync pass. If even the parent
            # switcher is not queryable yet, the initial-load retry will paint
            # the current state on the next message-pump turn.
            await self._mount_deferred_canvases()
            if not self.query(MCPToolsMode):
                return
            display_snapshots = [
                self._display_snapshot(snap) for snap in self._snapshots
            ]
            rail = self.query_one(MCPRail)
            rail.sync_state(
                source=self._source,
                snapshots=display_snapshots,
                selected_server_key=self._selected_server_key,
                scope_options=[("Personal", "personal")],
                scope_value=self._scope,
                scope_ref_options=[],
                scope_ref_value=self._scope_ref,
            )
            canvas = self.query_one(MCPServersMode)
            await canvas.update_overview(
                display_snapshots,
                source=self._source,
                mutations_available=self._server_mutations_available,
                mutation_target_label=self._active_target_label(),
            )
            selected = self._snapshot_for_display(self._selected_server_key)
            await self._show_selected_detail(canvas, selected)
            await self.query_one(MCPInspector).update_readiness(selected)
            tools = self._collect_hub_tools()
            self._last_hub_tools = tools
            effective = self._resolve_effective_states(tools)
            await self._sync_tools_mode(tools, effective)
            await self._sync_permissions_mode(effective, refresh_governance=True)
            await self._sync_audit_mode()

    async def _sync_audit_mode(self) -> None:
        """Push the current execution-log window AND Findings into `MCPAuditMode`.

        Mirrors `_sync_tools_mode()`/`_sync_permissions_mode()`: runs on
        every `_sync_children()` pass (the ContentSwitcher never unmounts
        the inactive canvas, only hides it) so switching INTO Audit mode
        never shows a stale window from before the last background resync.

        Task 5 (PR-T3, F3): the entries half is split out into
        `_sync_audit_log_entries()` so `_run_tool_test()`'s `finally` can
        resync JUST that half after a completed run -- without also
        re-fetching the server-source Findings sub-view below, a separate,
        source-scoped concern unrelated to "did my run just land in the
        audit trail" (and one a Test Tool run can never affect: server-
        source tools are display-only, `execute_hub_tool()` refuses them
        before any run reaches here).
        """
        await self._sync_audit_log_entries()

        # T8 (MCP Hub Phase 5): Findings sub-view -- server source only.
        service = self._service()
        findings = await self._server_findings(service)
        self._last_audit_findings = findings or []
        await self.query_one(MCPAuditMode).update_findings(
            findings, source=self._source
        )

    async def _sync_audit_log_entries(self) -> None:
        """Push the current execution-log window into `MCPAuditMode`.

        Task 5 (PR-T3, F3): split out of `_sync_audit_mode()` so a
        completed tool run (`_run_tool_test()`'s `finally`) can resync
        JUST this half -- the JSONL log already has the new row by the
        time `test_hub_tool()` returns (it records before returning/
        raising), so re-reading it here is all a completed run needs to
        show up in the Audit table without pressing `r`.

        `service.execution_log` is a PROPERTY on
        `UnifiedMCPControlPlaneService` that can itself raise (see that
        property's own N1 lesson, unified_control_plane_service.py) -- the
        access is guarded by the SAME try/except as the `read_recent()`
        call itself, not just a `getattr(..., None)` (which only catches an
        `AttributeError`, not an arbitrary raise from inside a property
        getter). No log configured (a service too old to expose the
        property, a fake in older tests, or the property itself resolving
        to `None` -- e.g. no local store yet) renders an empty window
        rather than raising out of the caller.
        """
        service = self._service()
        entries: list[dict[str, Any]] = []
        log = None
        if service is not None:
            try:
                log = service.execution_log
            except Exception as exc:
                logger.warning(
                    "{}",
                    _safe_diagnostic_message("MCP execution log access failed", exc),
                )
                log = None
        if log is not None:
            try:
                entries = log.read_recent(200)
            except Exception as exc:
                logger.warning(
                    "{}", _safe_diagnostic_message("MCP execution log read failed", exc)
                )
                entries = []
        self._last_audit_entries = entries
        await self.query_one(MCPAuditMode).update_entries(entries)

    async def _server_findings(self, service: Any) -> list[dict[str, Any]] | None:
        """T8: this pass's server-source Audit-mode Findings listing,
        fetched at most once per `(source, target)` identity -- mirrors
        `_server_governance_profiles()` (T11) exactly, minus that method's
        `refresh` gate (see `_findings_cache_key`'s own docstring in
        `__init__` for why: `_sync_audit_mode()`, this method's only
        caller, has no standalone invocation path the way `_sync_
        permissions_mode()` does).
        """
        key = (self._source, self._active_service_target_id())
        if key != self._findings_cache_key:
            self._findings_cache = await self._load_server_findings(service)
            self._findings_cache_key = key
        return self._findings_cache

    async def _load_server_findings(self, service: Any) -> list[dict[str, Any]] | None:
        """T8: the server-source Audit-mode Findings fetch's data.

        Only ever fetched under the server source -- local/builtin never
        call `load_section("advanced")` for this at all. Guarded the same
        fail-soft way as `_load_server_governance_profiles()`: any
        exception (no active target, a backend error, a service too old to
        expose the section) -> `None` -> `MCPAuditMode.update_findings()`
        renders the fetch-failure retry hint rather than raising out of
        `_sync_children()`.

        `governance_audit_findings` is an ENVELOPE dict (`{"items": [...]}
        `, `_envelope_payload()`'s own shape in `MCP/server_unified_
        service.py`'s `get_advanced()` ~:392), not a bare list -- mirrors
        `unified_mcp_sections.render_advanced_section()`'s own extraction
        (`(payload.get("governance_audit_findings") or {}).get("items")`).
        A malformed-but-present response (not a Mapping, a
        `governance_audit_findings` that isn't a Mapping, or an `items`
        that isn't a list) still counts as a successful fetch -- `[]`, same
        as `_load_server_governance_profiles()`'s own malformed-but-present
        contract, so the Findings table renders its "no findings" empty
        copy rather than the fetch-failure one.
        """
        if self._source != "server" or service is None:
            return None
        loader = getattr(service, "load_section", None)
        if not callable(loader):
            return None
        try:
            advanced_payload = await loader("advanced")
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message("MCP audit findings fetch failed", exc),
            )
            return None
        if not isinstance(advanced_payload, Mapping):
            return []
        findings_envelope = advanced_payload.get("governance_audit_findings")
        if not isinstance(findings_envelope, Mapping):
            return []
        raw_items = findings_envelope.get("items")
        return raw_items if isinstance(raw_items, list) else []

    async def _sync_tools_mode(
        self, tools: list[HubTool], states: dict[tuple[str, str], EffectiveToolState]
    ) -> None:
        """Push the current cross-server tool catalog into `MCPToolsMode`.

        T5: runs on every `_sync_children()` pass (mirrors `MCPServersMode`'s
        own unconditional resync above) rather than only while Tools mode is
        the active canvas -- the ContentSwitcher hides the inactive canvas
        via CSS but never unmounts it, so keeping its data current the whole
        time means switching TO Tools mode never shows stale rows from
        before the last background resync.

        T10: `tools` (from `_collect_hub_tools()`) and `states` (from
        `_resolve_effective_states()`) are now both resolved ONCE by the
        caller (`_sync_children()`) and passed in, rather than this method
        deriving them itself -- `_sync_permissions_mode()` needs the exact
        same two values this same pass, and resolving them twice meant two
        full store loads (plus any mark/audit side effects
        `effective_tool_states()` performs) back-to-back for identical
        input.
        """
        diagnosis = None if tools else self._empty_tools_diagnosis()
        canvas = self.query_one(MCPToolsMode)
        enabled, workspace_root = self._local_tools_config_values()
        canvas.update_local_config(
            enabled=enabled,
            workspace_root=workspace_root,
            visible=self._source == "local",
        )
        await canvas.update_tools(tools, empty_diagnosis=diagnosis, states=states)

    @staticmethod
    def _local_tools_config_values() -> tuple[bool, str]:
        """Read the persisted values rendered by Tools mode."""
        enabled = coerce_bool_setting(
            get_cli_setting(
                "console", "local_tools_enabled", LOCAL_TOOLS_DEFAULT_ENABLED
            ),
            LOCAL_TOOLS_DEFAULT_ENABLED,
        )
        raw_root = get_cli_setting("console", "workspace_root", "")
        workspace_root = raw_root.strip() if isinstance(raw_root, str) else ""
        return enabled, workspace_root

    def _refresh_local_tools_controls(self) -> MCPToolsMode | None:
        """Reset Tools-mode controls to persisted config after a failed save."""
        if not self.query(MCPToolsMode):
            return None
        canvas = self.query_one(MCPToolsMode)
        enabled, workspace_root = self._local_tools_config_values()
        canvas.update_local_config(
            enabled=enabled,
            workspace_root=workspace_root,
            visible=self._source == "local",
        )
        return canvas

    def _collect_hub_tools(self) -> list[HubTool]:
        """Derive the current source's cross-server `HubTool` catalog.

        Local source: every local profile's discovered tools
        (`self._catalog_records`, populated by `_collect_snapshots()` --
        reused here, not re-fetched), the built-in server's inventory
        (`service.local_service.get_inventory()`, guarded by getattr since
        test fakes and a still-initializing service may not expose it),
        and the workspace, web, and Watchlists agent tool set
        (`_local_agent_hub_tools()`, task-2838 -- keyed `local:__local__`):
        exact shared descriptor identities are executable, while Console-only
        rows remain visible but non-executable.

        Server source: each external-server record's own embedded `tools`
        list (when the backend includes one -- `ReadinessSnapshot.detail
        ["raw"]`, already loaded by `_collect_snapshots()`'s external-
        servers-section read for the active target; see
        `server_external_record_readiness()`), keyed by that record's own
        composite id so its tools group under the SAME server_key
        (`"server:<target>/<record>"`) the Servers-mode rail/table already
        use for it. Bare server TARGETS (the tldw_server connection itself)
        carry no tools of their own -- only the external records beneath
        them do, mirroring how a local profile (not the built-in server) is
        the tool-bearing entity on the local side.
        """
        tools: list[HubTool] = []
        if self._source == "local":
            for record in self._catalog_records.values():
                tools.extend(local_tools_from_record(record))
            service = self._service()
            local_service = (
                getattr(service, "local_service", None) if service is not None else None
            )
            get_inventory = getattr(local_service, "get_inventory", None)
            if callable(get_inventory):
                try:
                    inventory = get_inventory()
                except Exception as exc:
                    logger.warning(
                        "{}",
                        _safe_diagnostic_message(
                            "MCP built-in inventory read failed", exc
                        ),
                    )
                    inventory = None
                if isinstance(inventory, Mapping):
                    tools.extend(builtin_tools_from_inventory(inventory))
            # task-2838: the workspace, web, and Watchlists agent tool set
            # is a first-class Hub catalog source too -- same shared
            # permission store the Console gates on, resolved by the same
            # `effective_tool_states()` pass as every other row.
            tools.extend(self._local_agent_hub_tools())
            # TASK-22510: policy discoverability is not runtime authority.
            # Keep the raw host shell visible even while locked, unarmed,
            # disabled by the local-tools master switch, or absent from the
            # model's live schema catalog. The Tools view itself cannot run
            # it; Console owns the separate command-visible approval path.
            raw_shell_tool = self._raw_shell_hub_tool()
            if raw_shell_tool is not None:
                tools.append(raw_shell_tool)
        else:
            for snap in self._snapshots:
                if snap.source != "server" or not self._is_external_record_key(
                    snap.server_key
                ):
                    continue
                raw = (snap.detail or {}).get("raw")
                if isinstance(raw, Mapping):
                    remainder = snap.server_key.split(":", 1)[1]
                    tools.extend(
                        server_tools_from_inventory(
                            raw, target_id=remainder, target_label=snap.label
                        )
                    )
        return tools

    def _local_agent_hub_tools(self) -> list[HubTool]:
        """Collect service-owned local projection plus local Virtual CLI.

        The control-plane service owns the full workspace/web/Watchlists
        inspection catalog and exact executable projection. This widget keeps
        only the separately governed, always non-executable Virtual CLI view.
        """
        tools: list[HubTool] = []
        service = self._service()
        local_hub_tools = getattr(service, "local_hub_tools", None)
        if callable(local_hub_tools):
            try:
                tools.extend(local_hub_tools())
            except Exception as exc:  # noqa: BLE001 -- catalog view is fail-soft
                logger.warning(
                    "MCP local agent tool catalog unavailable "
                    f"(exception_type={type(exc).__name__})"
                )

        enabled = coerce_bool_setting(
            get_cli_setting(
                "console", "local_tools_enabled", LOCAL_TOOLS_DEFAULT_ENABLED
            ),
            LOCAL_TOOLS_DEFAULT_ENABLED,
        )
        if not enabled:
            return tools
        try:
            root = resolve_server_workspace_root()
            from tldw_chatbook.Agents.virtual_cli_provider import (
                VirtualCliProvider,
            )

            provider = VirtualCliProvider(workspace_root=root)
            tools.extend(replace(hub, executable=False) for hub in provider.hub_tools())
        except Exception as exc:  # noqa: BLE001 -- catalog view must never break the hub
            logger.warning(
                "MCP Virtual CLI catalog unavailable "
                f"(exception_type={type(exc).__name__})"
            )
        return tools

    def _raw_shell_hub_tool(self) -> HubTool | None:
        """Project raw-shell policy when this app owns the required runtime."""

        runtime = getattr(self.app_instance, "raw_cli_runtime", None)
        if runtime is None:
            # Compatibility/fail-soft boundary: lightweight embedders and
            # test harnesses may mount the generic MCP workbench without the
            # app-owned raw CLI subsystem. Chatbook itself always creates
            # that runtime before screens compose; absence means the feature
            # does not exist here, not a misleading persistent "Locked" row.
            return None
        try:
            permitted = runtime.permitted is True
            armed = permitted and runtime.armed is True
        except Exception:  # noqa: BLE001 -- a broken runtime must read locked
            permitted = False
            armed = False

        if not permitted:
            availability = (
                "Locked — the persistent Raw CLI unlock is Off. Models cannot "
                "use this tool."
            )
        elif not armed:
            availability = (
                "Unlocked, not armed — re-arm Raw CLI in Privacy & Security "
                "for this Chatbook launch before models can use it."
            )
        else:
            availability = (
                "Armed — available to models, but each command still requires "
                "visible approval unless this Console session has temporary "
                "authority."
            )
        warning = (
            "DANGER: This runs a real host shell with the full authority of "
            "the OS user and is not workspace confined. Permission is Ask or "
            "Off only; a stored Allow value is treated as Ask."
        )
        from tldw_chatbook.Agents.raw_shell_tool_provider import (
            RawShellToolProvider,
        )

        return replace(
            RawShellToolProvider.hub_tool(),
            description=f"{availability}\n\n{warning}",
            executable=False,
        )

    def _empty_tools_diagnosis(self) -> tuple[str, str]:
        """Diagnose why the Tools mode catalog is currently empty.

        Mirrors the design spec's three-bucket empty-state model for LOCAL
        source: no servers at all -> add one; servers exist but none have
        ever connected/discovered (every relevant snapshot is still
        `NEEDS_SETUP`) -> connect or refresh; otherwise (servers have
        connected/discovered but genuinely returned zero tools) -> refresh
        again. "Relevant" excludes the built-in server under local source
        (it's always present and isn't something the user "configured").

        UX item 10 (Task 2, MCP Hub Phase 6): SERVER source instead gets
        ONE fixed diagnosis regardless of which of those three reasons
        actually applies. The local-source buckets above all end up
        pointing at Servers mode's per-server CONNECT/REFRESH_DISCOVERY
        lifecycle actions -- but those are disabled for server-source
        snapshots in the inspector (`_wired_actions()` only wires them for
        local source), so routing there would just point the user at more
        disabled buttons. Its own "refresh" action instead routes to the
        cache-invalidating resync (`_refresh_server_discovery()`, see
        `on_mcp_tools_mode_empty_action_requested()`) rather than a bare
        mode switch.

        task-3240 (SECONDARY/partial discoverability breadcrumb): whenever
        one or more `[tools]`/`[console]` registration gates are off, that
        is APPENDED to whichever message above applies. Honestly partial,
        not a full breadcrumb: this method only ever runs when the whole
        Tools-mode catalog is empty (see the caller, `_sync_tools_mode()`),
        so it stays silent whenever ANY unrelated local tool source (an MCP
        server, a connected local profile) already produced tools -- the
        PRIMARY breadcrumb (the Permissions matrix's always-visible legend,
        `_sync_permissions_mode()`) has no such blind spot.
        """
        if self._source == "server":
            message, action = (
                "No tools visible from this server — refresh or check the server.",
                "refresh",
            )
        else:
            relevant = [snap for snap in self._snapshots if snap.source == self._source]
            if not relevant:
                message, action = (
                    "No servers configured — add one to see its tools.",
                    "add_server",
                )
            elif all(snap.state is ReadinessState.NEEDS_SETUP for snap in relevant):
                message, action = (
                    "No tools discovered yet — connect or refresh a server.",
                    "connect",
                )
            else:
                message, action = (
                    "No tools found — try refreshing a server's discovery.",
                    "refresh",
                )
        breadcrumb = tool_gate_breadcrumb()
        if breadcrumb:
            message = f"{message} {breadcrumb}"
        return (message, action)

    # -- T6: Permissions mode (matrix, kill switch, policy preview) -----------

    def _resolve_effective_states(
        self, tools: list[HubTool]
    ) -> dict[tuple[str, str], EffectiveToolState]:
        """One batched `effective_tool_states()` call for `tools`.

        Read via the same `getattr(..., None)` + `callable()` +
        try/except fail-soft pattern as every other T4 seam here. A service
        without the Phase 4 permission methods yet (older
        fakes, a still-initializing service) resolves to an empty dict
        rather than raising.

        T8: shared by `_sync_tools_mode()` (State column) and
        `_sync_permissions_mode()` (matrix rows).

        T10: a full `_sync_children()` pass now calls this itself exactly
        ONCE and threads the one result into both `_sync_tools_mode()` and
        `_sync_permissions_mode()` (see `_sync_children()`'s docstring) --
        this method itself is unchanged, and the three STANDALONE callers
        of `_sync_permissions_mode()` (kill-switch toggle, Space-press
        state cycle, re-allow) still call THIS directly with no
        preceding `_sync_tools_mode()` pass in the same call, and must
        always resolve fresh so a just-applied permission change is
        reflected immediately rather than waiting for the next full
        `_sync_children()` pass.
        """
        service = self._service()
        loader = getattr(service, "effective_tool_states", None)
        states: dict[tuple[str, str], EffectiveToolState] = {}
        if callable(loader):
            try:
                states = dict(loader(tools))
            except Exception as exc:
                logger.warning(
                    "{}",
                    _safe_diagnostic_message(
                        "MCP effective tool state resolution failed", exc
                    ),
                )

        for tool in tools:
            if not _is_raw_shell_tool(tool.server_key, tool.name):
                continue
            key = (tool.server_key, tool.name)
            stored = states.get(key) or EffectiveToolState(
                state="ask", origin="global_default"
            )
            states[key] = EffectiveToolState(
                state=_resolve_raw_shell_state()(stored),
                origin=stored.origin,
                # Raw shell never honors persistent Allow, so its generic
                # definition-hash and inherited-risk-floor markers would
                # only advertise a misleading Re-allow path.
                config_changed=False,
                risk_floored=False,
            )
        return states

    def _builtin_permission_rows(self, payload: dict[str, Any]) -> list:
        """This run's built-in tool rows, resolved by the BUILT-IN resolver.

        Deliberately NOT merged into `_resolve_effective_states()`: that
        method calls `effective_tool_states()`, which applies MCP semantics
        (ask-floor + hash check) and calls `store.mark_config_changed()` --
        a rug-pull marker `resolve_builtin_state` ignores. Routing built-ins
        through it would resolve them wrongly AND store an inert flag. See
        the design doc's spike findings for the failure this avoids.

        `payload` is the SAME dict `_sync_permissions_mode()` already loaded
        once via `store.load()` for the MCP path (review finding, Task 3
        fix round) -- this method does NOT read the store itself. A second,
        independent read here would cost an extra file access every
        `_sync_children()` pass/Space press, and would open a coherence
        window: this section's `state_label` would come from a DIFFERENT
        snapshot than its `cycle_current`/server-default label (derived
        from the caller's own `servers_payload`, itself sliced from the
        SAME first read) if a store write raced between the two reads.
        `payload` may legitimately be `{}` (no `permission_store` seam, or
        the caller's own read failed) -- `builtin_permission_rows({})` is
        documented as valid and resolves everything to the built-in ALLOW
        floor, so the caller's own fail-soft default is sufficient; no
        second guard is needed here for "no payload available".

        Fail-soft like every other service seam here: any failure in
        `builtin_permission_rows()` itself yields an empty list rather than
        raising into a render pass.
        """
        try:
            return builtin_permission_rows(payload)
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message(
                    "builtin permission row enumeration failed", exc
                ),
            )
            return []

    def _builtin_permission_matrix_rows(
        self, payload: dict[str, Any], servers_payload: Mapping[str, Any]
    ) -> list[PermRow]:
        """Render this pass's built-in tool rows as matrix `PermRow`s.

        Task 3 (built-in permissions UI, TASK-627): a SIBLING section to
        the MCP matrix `_build_permission_rows()` builds -- appended after
        it in `_sync_permissions_mode()`, never merged into it, and never
        threaded through that method's `tools`/`effective` arguments (this
        section's tools are never part of `_last_hub_tools` -- see
        Constraint 1/5). Namespaced entirely under `BUILTIN_TOOL_SERVER_KEY`
        ("agent:builtin") and labeled `_BUILTIN_SECTION_LABEL`, distinct
        from the built-in MCP *server* ("tldw_chatbook",
        `builtin:tldw_chatbook`) per Constraint 3 -- the two must never
        share a row key or a label a user could mistake for the same
        thing. This is also this method's Constraint 4 (fail closed):
        because every row built here hard-codes `BUILTIN_TOOL_SERVER_KEY`
        rather than accepting or branching on an externally supplied
        `server_key`, there is no code path here that could resolve an
        unrecognized key by inheriting MCP's (or any other) branch.

        `payload`/`servers_payload` are the caller's OWN already-loaded
        values (see `_builtin_permission_rows()`'s docstring) -- passed
        straight through, never re-read.

        Every tool row's `state_label` is `format_tool_state_label()` of
        the `EffectiveToolState` `_builtin_permission_rows()` already
        resolved via `resolve_builtin_state` (never the MCP resolver) --
        same marker precedence (`⚠`/`⚑`/`•`) as every MCP row, so the two
        sections read consistently. An orphaned stored entry (a decision
        for a tool a later release removed) is marked via its Tags cell
        ("orphaned") rather than its Tool cell -- `tool_name` stays the
        raw stored name so a future cycle/clear action still addresses the
        right store entry instead of a decorated string.

        The pinned "Server default" row is ALWAYS returned, even when
        `_builtin_permission_rows()` yields zero tool rows (review finding:
        enumeration failing must not also hide a user's stored built-in
        server default, making it invisible and impossible to clear) -- only
        the per-tool rows beneath it are conditional on that list being
        non-empty.

        Fix 1 (PR #906 review): also refreshes `self._last_builtin_
        effective` -- REBUILT fresh here, not mutated in place, mirroring
        `_last_cascade`'s own "computed once this pass, reused" precedent
        -- so `on_mcp_permissions_mode_row_selected()` can route a built-in
        `"tool"` row selection to the inspector's permission view using the
        SAME resolution this method already did for the matrix cell, rather
        than re-deriving the built-in catalog (and re-reading `payload`) a
        second time.
        """
        rows_in = self._builtin_permission_rows(payload)
        self._last_builtin_effective = {
            (BUILTIN_TOOL_SERVER_KEY, row.name): row.effective for row in rows_in
        }

        server_entry = servers_payload.get(BUILTIN_TOOL_SERVER_KEY)
        raw_default = (
            server_entry.get("default") if isinstance(server_entry, Mapping) else None
        )
        if raw_default in STORE_STATES:
            server_state_label = f"{EffectiveToolState(state=raw_default, origin='server_default').ui_label} •"
            server_cycle_current: str | None = raw_default
        else:
            # Inherit: nothing explicit at the server level -- shown as the
            # BUILT-IN allow floor (`BUILTIN_DEFAULT_STATE`), never the MCP
            # global default (Constraint 1: built-ins never inherit MCP's
            # posture).
            server_state_label = EffectiveToolState(
                state=BUILTIN_DEFAULT_STATE, origin="builtin_default"
            ).ui_label
            server_cycle_current = None

        matrix_rows: list[PermRow] = [
            PermRow(
                kind="server",
                server_key=BUILTIN_TOOL_SERVER_KEY,
                server_label=_BUILTIN_SECTION_LABEL,
                tool_name=None,
                state_label=server_state_label,
                tags_label="—",
                cycle_current=server_cycle_current,
            )
        ]
        for row in rows_in:
            matrix_rows.append(
                PermRow(
                    kind="tool",
                    server_key=BUILTIN_TOOL_SERVER_KEY,
                    server_label=_BUILTIN_SECTION_LABEL,
                    tool_name=row.name,
                    state_label=format_tool_state_label(row.effective),
                    tags_label="orphaned" if row.orphaned else "—",
                    cycle_current=self._raw_tool_state(
                        servers_payload, BUILTIN_TOOL_SERVER_KEY, row.name
                    ),
                )
            )
        return matrix_rows

    async def _sync_permissions_mode(
        self,
        effective: dict[tuple[str, str], EffectiveToolState] | None = None,
        *,
        refresh_governance: bool = False,
        echo: str | None = None,
    ) -> None:
        """Push the current permission matrix into `MCPPermissionsMode`.

        Mirrors `_sync_tools_mode()`: runs on every `_sync_children()` pass
        (the ContentSwitcher never unmounts the inactive canvas, only hides
        it) so switching INTO Permissions mode never shows a stale matrix
        from before the last background resync. Computed from
        `self._last_hub_tools` (already derived this same pass by
        `_sync_children()`, immediately before it calls this) plus one
        `permission_store.load()` read -- no extra service I/O beyond that
        (T8's server-source governance fetch, below, is the one exception,
        and T11 now caches it -- see `_server_governance_profiles()`).
        TASK-627 Task 3: the built-in section (`_builtin_permission_matrix_
        rows()`, appended below) is rendered from this SAME one read
        (`payload`/`servers_payload`, passed straight through) -- it must
        never trigger a second `store.load()` of its own; see that
        method's docstring for why a second read would also open a
        state/cycle_current coherence window, not just cost extra I/O.

        Fix 2 (PR #906 review): `_builtin_permission_matrix_rows()` is now
        called BEFORE `_build_permission_rows()` (it only ever needed
        `payload`/`servers_payload`, both already loaded above -- nothing
        `_build_permission_rows()` itself derives), so its rows can be
        threaded into that call's `extra_override_rows` and counted by the
        preview's override suffix too. `update_matrix()`'s own docstring
        says the preview "ALWAYS summarizes the full, UNFILTERED matrix" --
        that used to be false for a built-in override (the table cell
        changed, the summary line's count didn't), because the preview was
        built from `_build_permission_rows()`'s MCP-only `rows` before the
        built-in section was even appended. The built-in rows are still
        never merged into `tools`/`effective` (Constraint 1/5 -- see
        `_builtin_permission_matrix_rows()`'s own docstring) -- only fed to
        the preview's override COUNT, a separate concern from tool/state
        resolution.

        T10: `effective` is the SAME batch `EffectiveToolState` resolution
        `_sync_children()` already computed once this pass (via
        `_resolve_effective_states()`) for `_sync_tools_mode()`'s State
        column -- passed in rather than resolved again here, so one full
        `_sync_children()` pass no longer means two back-to-back
        `effective_tool_states()` calls (each a full store load, plus any
        mark/audit side effects that method performs) for identical input.
        `None` (the default) means resolve fresh instead: every STANDALONE
        caller (kill-switch toggle, Space-press state cycle, re-allow) just
        mutated the permission store itself and must see that change
        reflected immediately, not the previous pass's now-stale snapshot.

        T11: `refresh_governance` gates whether `_server_governance_
        profiles()` may actually fetch this pass -- only `_sync_children()`'s
        own full pass sets it True; every standalone caller leaves it False
        (the default) so a Space-press/kill-switch/re-allow resync reuses
        whatever governance listing was last cached rather than re-awaiting
        `load_section("governance")` for data that interaction can't have
        changed.

        Every T4 seam is read via `getattr(..., None)` + `callable()` --
        so a service that hasn't been upgraded
        with the Phase 4 permission methods (older fakes, a
        still-initializing service) renders an all-"Ask", switch-off matrix
        instead of raising out of every `_sync_children()` call.

        Task 3 (MCP Hub Phase 6): `echo`, when given, is threaded straight
        through to `MCPPermissionsMode.update_matrix()` -- the transient
        mutation-confirmation copy a STANDALONE caller computes for its own
        just-applied change (Space-cycle/kill-switch/re-allow). `None` (the
        default -- every full `_sync_children()` pass) clears whatever a
        previous standalone resync showed; see `update_matrix()`'s own
        docstring for the render contract.
        """
        service = self._service()
        tools = self._last_hub_tools

        kill_switch = False
        get_kill_switch = getattr(service, "get_kill_switch", None)
        if callable(get_kill_switch):
            try:
                kill_switch = bool(get_kill_switch())
            except Exception as exc:
                # task-545/T6: this switch is global -- it also gates every
                # built-in tool via `BuiltinToolGate._kill_switch()` -- so
                # the log line no longer says "MCP" (matches that method's
                # own "kill switch read failed" wording).
                logger.warning(
                    "{}", _safe_diagnostic_message("kill switch read failed", exc)
                )

        standalone_resync = effective is None
        if effective is None:
            effective = self._resolve_effective_states(tools)
        # T7: cache this batch resolution for `_effective_for_display()` --
        # both Tools-mode's tool-detail permission block and Permissions-
        # mode's own matrix-row selection reuse it instead of a second,
        # redundant per-tool resolution.
        self._last_effective_states = effective

        if standalone_resync:
            # Defect 1 fix (MCP Hub Phase 4 live QA, 2026-07-16): a
            # STANDALONE caller (Space-cycle, kill-switch toggle, Re-allow)
            # just resolved this batch fresh for ITS OWN matrix resync,
            # above -- hand that same dict to `MCPToolsMode` too, so its
            # State column reflects the mutation immediately instead of
            # waiting for the next full `_sync_children()` pass. No extra
            # `effective_tool_states()` call, no governance fetch, no
            # `_sync_tools_mode()`/tool-list rebuild -- just this one
            # widget's own narrow row re-render (`update_states()`).
            self.query_one(MCPToolsMode).update_states(effective)

        payload: dict[str, Any] = {}
        store = getattr(service, "permission_store", None)
        if store is not None:
            try:
                payload = store.load()
            except Exception as exc:
                logger.warning(
                    "{}",
                    _safe_diagnostic_message("MCP permission store read failed", exc),
                )
                payload = {}

        profile = (payload.get("profiles") or {}).get("default") or {}
        global_state = profile.get("global_default")
        if global_state not in STORE_STATES:
            global_state = DEFAULT_GLOBAL
        servers_payload = profile.get("servers") or {}
        if not isinstance(servers_payload, Mapping):
            servers_payload = {}

        # TASK-627 Task 3: the agent-runtime built-in section, appended
        # AFTER the MCP sections and never merged into `_build_permission_
        # rows()`'s own grouping -- it renders even when `tools` is empty
        # (no MCP servers configured), since it derives from the live
        # built-in tool registry, not the MCP catalog `tools` came from.
        # Fix 2: computed FIRST now, so it can also feed the preview's
        # override count below (see this method's own docstring).
        builtin_rows = self._builtin_permission_matrix_rows(payload, servers_payload)
        rows, preview, cascade_map = self._build_permission_rows(
            tools,
            effective=effective,
            servers_payload=servers_payload,
            global_state=global_state,
            extra_override_rows=builtin_rows,
        )
        # Task 3: cache this pass's per-tool cascade map for
        # `_cascade_for_tool()` -- same "computed once, reused" precedent as
        # `_last_effective_states` immediately above.
        self._last_cascade = cascade_map
        rows = rows + builtin_rows
        # task-3240 PRIMARY breadcrumb: computed fresh every pass (cheap --
        # the same settings-time-enumeration cost `_builtin_permission_
        # matrix_rows()` above already pays every pass) so it can never
        # drift from the gates' actual current state.
        await self.query_one(MCPPermissionsMode).update_matrix(
            rows,
            kill_switch=kill_switch,
            preview=preview,
            echo=echo,
            gate_breadcrumb=tool_gate_breadcrumb(),
        )
        await self.query_one(MCPPermissionsMode).update_server_profiles(
            await self._server_governance_profiles(service, refresh=refresh_governance)
        )

    def _cascade_for_tool(
        self, tool: HubTool
    ) -> tuple[str | None, str | None, str] | None:
        """One tool's raw cascade tuple (Task 3, MCP Hub Phase 6), from the
        last `_sync_permissions_mode()` pass's own `_build_permission_rows()`
        derivation -- `None` when the tool isn't in that map (e.g. nothing
        has synced Permissions mode yet), which `show_permission(...,
        cascade=None)` renders as the pre-Task-3 single origin sentence
        rather than crashing or showing an empty cascade block.
        """
        return self._last_cascade.get((tool.server_key, tool.name))

    async def _server_governance_profiles(
        self, service: Any, *, refresh: bool
    ) -> list[dict[str, Any]] | None:
        """T11: this pass's server-source governance listing, fetched at
        most once per `(source, target)` identity.

        `_load_server_governance_profiles()` reads STATIC server-side
        profile data -- nothing a permission-matrix interaction (Space-press
        cycle, kill-switch toggle, re-allow) can change -- so this is the
        single point deciding whether the current pass actually needs to hit
        `load_section("governance")` again:

        - `refresh=True` (only `_sync_children()`'s own full pass): fetch
          when `(self._source, self._active_service_target_id())` differs
          from the key of the last fetch, or nothing has ever been fetched
          yet (`_governance_profiles_cache_key` still the module's `_UNSET`
          sentinel); otherwise reuse the cached value with no I/O at all.
        - `refresh=False` (every standalone resync): NEVER fetches --
          returns whatever is cached for the CURRENT key. A key mismatch
          here (returns `None`, section renders absent rather than a
          stale different identity's profiles) is defensive only: every
          source/target switch (`_switch_source()`, `_select_server_key()`)
          runs its own full `_sync_children()` pass -- which refreshes the
          cache for the new key -- before any standalone resync can run
          against it, so this branch should not normally trigger.

        No separate cache-invalidation call is needed at the rail's
        source/scope-switch handlers: the key is recomputed from live state
        on every call, so any actual identity change is caught by the
        comparison above rather than requiring every switch site to
        remember to clear a cache by hand.
        """
        key = (self._source, self._active_service_target_id())
        if key != self._governance_profiles_cache_key:
            if not refresh:
                return None
            self._governance_profiles_cache = (
                await self._load_server_governance_profiles(service)
            )
            self._governance_profiles_cache_key = key
        return self._governance_profiles_cache

    async def _load_server_governance_profiles(
        self, service: Any
    ) -> list[dict[str, Any]] | None:
        """T8: the server-source read-only governance listing's data.

        Only ever fetched under the server source -- local/builtin never
        call `load_section("governance")` at all, so the section stays
        entirely absent there (not merely empty) without needing its own
        source check downstream. Guarded the same fail-soft way as every
        other seam in `_sync_permissions_mode()`: any exception (no active
        target, a backend error, a service too old to expose the section)
        -> `None` -> `MCPPermissionsMode.update_server_profiles()` renders
        no section at all, same as local/builtin.

        A malformed-but-present response (not a Mapping, or a
        `permission_profiles` key that isn't a list) still counts as a
        successful fetch -- returns `[]` so the section renders with its
        pointer text and zero rows, rather than disappearing outright the
        way an actual fetch failure does.
        """
        if self._source != "server" or service is None:
            return None
        loader = getattr(service, "load_section", None)
        if not callable(loader):
            return None
        try:
            governance_payload = await loader("governance")
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message("MCP governance section fetch failed", exc),
            )
            return None
        if not isinstance(governance_payload, Mapping):
            # Malformed-but-present, per this method's own docstring above --
            # the fetch itself succeeded, it just didn't hand back the shape
            # expected. `[]` (not `None`), same as a bad `permission_profiles`
            # value below.
            return []
        raw_profiles = governance_payload.get("permission_profiles")
        return raw_profiles if isinstance(raw_profiles, list) else []

    @staticmethod
    def _tool_state_label(effective: EffectiveToolState) -> str:
        """T8: thin delegation to the shared, module-level rendering
        helper (`mcp_permissions_mode.format_tool_state_label()`) -- kept
        as a staticmethod here too since `_build_permission_rows()` below
        already calls `self._tool_state_label(...)`, and
        `test_tool_state_label_marker_precedence` pins this exact call
        shape (`MCPWorkbench._tool_state_label`).
        """
        return format_tool_state_label(effective)

    @staticmethod
    def _raw_tool_state(
        servers_payload: Mapping[str, Any], server_key: str, tool_name: str
    ) -> str | None:
        """The raw STORE value for one tool entry (`cycle_ui_state()`'s
        input), or `None` when nothing is set at the tool level -- distinct
        from the tool's *resolved* effective state, which may inherit from
        the server or global default."""
        server_entry = servers_payload.get(server_key)
        if not isinstance(server_entry, Mapping):
            return None
        tools_entry = server_entry.get("tools")
        if not isinstance(tools_entry, Mapping):
            return None
        tool_entry = tools_entry.get(tool_name)
        if not isinstance(tool_entry, Mapping):
            return None
        state = tool_entry.get("state")
        return state if state in STORE_STATES else None

    def _build_permission_rows(
        self,
        tools: list[HubTool],
        *,
        effective: dict[tuple[str, str], EffectiveToolState],
        servers_payload: Mapping[str, Any],
        global_state: str,
        extra_override_rows: Sequence[PermRow] = (),
    ) -> tuple[
        list[PermRow], str, dict[tuple[str, str], tuple[str | None, str | None, str]]
    ]:
        """Derive the pinned global -> server-default -> tool `PermRow`
        list (grouped by server, both servers and their tools sorted by
        label/name), the rail-scoped policy preview sentence, and (Task 3,
        MCP Hub Phase 6) a per-tool cascade map -- `(tool_entry_state,
        server_default, global_state)`, the SAME raw STORE values this
        method already reads to build `PermRow.cycle_current`/
        `server_cycle_current`/the global row, just packaged one tuple per
        tool for `_cascade_for_tool()`/`show_permission(..., cascade=...)`
        rather than split across rows.

        Fix 2 (PR #906 review): `extra_override_rows` is passed straight
        through to `_build_permission_preview()`'s own `extra_override_rows`
        -- it is counted toward the preview's override suffix but never
        merged into `tools`/`effective`/`rows`/`cascade_map` here, so it has
        no effect on tool/state resolution or the returned matrix rows.
        The caller (`_sync_permissions_mode()`) hands in the built-in
        section's own rows (`_builtin_permission_matrix_rows()`) so a
        persistent built-in override is reflected in the preview's count
        too, without folding built-ins into this method's MCP-only
        catalog walk (Constraint 1/5 -- see that method's docstring).
        """
        global_label = EffectiveToolState(
            state=global_state, origin="global_default"
        ).ui_label
        rows: list[PermRow] = [
            PermRow(
                kind="global",
                server_key="",
                server_label="",
                tool_name=None,
                state_label=global_label,
                tags_label="—",
                cycle_current=global_state,
            )
        ]
        cascade_map: dict[tuple[str, str], tuple[str | None, str | None, str]] = {}

        tools_by_server: dict[str, list[HubTool]] = {}
        labels_by_key: dict[str, str] = {}
        for tool in tools:
            tools_by_server.setdefault(tool.server_key, []).append(tool)
            labels_by_key.setdefault(tool.server_key, tool.server_label)

        for server_key in sorted(
            tools_by_server, key=lambda key: (labels_by_key[key], key)
        ):
            server_label = labels_by_key[server_key]
            server_entry = servers_payload.get(server_key)
            raw_default = (
                server_entry.get("default")
                if isinstance(server_entry, Mapping)
                else None
            )
            if raw_default in STORE_STATES:
                server_state_label = f"{EffectiveToolState(state=raw_default, origin='server_default').ui_label} •"
                server_cycle_current: str | None = raw_default
            else:
                # Inherit: nothing explicit at the server level -- shown as
                # the resolved (global) value, plain (no override marker).
                server_state_label = global_label
                server_cycle_current = None
            rows.append(
                PermRow(
                    kind="server",
                    server_key=server_key,
                    server_label=server_label,
                    tool_name=None,
                    state_label=server_state_label,
                    tags_label="—",
                    cycle_current=server_cycle_current,
                )
            )
            for tool in sorted(tools_by_server[server_key], key=lambda t: t.name):
                tool_effective = effective.get(
                    (tool.server_key, tool.name)
                ) or EffectiveToolState(state="ask", origin="global_default")
                tool_cycle_current = self._raw_tool_state(
                    servers_payload, tool.server_key, tool.name
                )
                rows.append(
                    PermRow(
                        kind="tool",
                        server_key=tool.server_key,
                        server_label=server_label,
                        tool_name=tool.name,
                        state_label=self._tool_state_label(tool_effective),
                        tags_label=", ".join(tool.tags) if tool.tags else "—",
                        cycle_current=tool_cycle_current,
                    )
                )
                if _is_raw_shell_tool(tool.server_key, tool.name):
                    # The generic store may contain Allow at any rung, but
                    # raw shell projects every such value to Ask. Keep the
                    # inspector's provenance cascade truthful too: showing
                    # "Permission: Ask" above "Tool override: Allow" would
                    # imply the forbidden silent-authority path still wins.
                    cascade_map[(tool.server_key, tool.name)] = (
                        _project_raw_shell_store_state(tool_cycle_current),
                        _project_raw_shell_store_state(server_cycle_current),
                        _project_raw_shell_store_state(global_state) or "ask",
                    )
                else:
                    cascade_map[(tool.server_key, tool.name)] = (
                        tool_cycle_current,
                        server_cycle_current,
                        global_state,
                    )

        preview = self._build_permission_preview(
            rows,
            tools_by_server,
            labels_by_key,
            effective,
            global_label,
            extra_override_rows=extra_override_rows,
        )
        return rows, preview, cascade_map

    def _build_permission_preview(
        self,
        rows: list[PermRow],
        tools_by_server: dict[str, list[HubTool]],
        labels_by_key: dict[str, str],
        effective: dict[tuple[str, str], EffectiveToolState],
        global_label: str,
        *,
        extra_override_rows: Sequence[PermRow] = (),
    ) -> str:
        """One plain-language sentence -- UX batch item 9: Library counts-
        line vocabulary (lowercase state words, no noun, " · " separators,
        no trailing period -- `library_ingest_state._queue_counts_line()`'s
        own precedent), scoped to the rail's currently selected server when
        that server has any discovered tools:
        `"<label>: N allow · M ask · K off — global default: <word>"`.

        With no selection (or the selected server has no discovered
        tools): `"global default: <word>"`, plus a
        `" · N overrides across M servers"` suffix when at least one
        explicit server- or tool-level override exists anywhere in `rows`
        OR `extra_override_rows` (omitted entirely when there are none,
        rather than a "0 overrides" segment nobody needs).

        Fix 2 (PR #906 review): `extra_override_rows` -- the built-in
        section's own `PermRow`s, when the caller has them
        (`_build_permission_rows()`'s own `extra_override_rows`) -- is
        counted into this suffix alongside `rows` so a persistent built-in
        override is reflected here too, matching `update_matrix()`'s own
        documented contract that this sentence "ALWAYS summarizes the
        full, UNFILTERED matrix". It plays no part in the rail-scoped
        branch immediately below: that branch only ever triggers for a
        SELECTED MCP server (`self._selected_server_key`), which the
        built-in section's pseudo server key can never be (the rail never
        lists it -- see `_builtin_permission_matrix_rows()`'s docstring).
        """
        global_word = global_label.lower()
        server_key = self._selected_server_key
        tools = tools_by_server.get(server_key) if server_key else None
        if tools:
            counts = {"allow": 0, "ask": 0, "deny": 0}
            for tool in tools:
                tool_effective = effective.get((tool.server_key, tool.name))
                state = tool_effective.state if tool_effective is not None else "ask"
                counts[state] = counts.get(state, 0) + 1
            label = labels_by_key.get(server_key, server_key)
            return (
                f"{label}: {counts['allow']} allow · {counts['ask']} ask · "
                f"{counts['deny']} off — global default: {global_word}"
            )
        override_rows = [
            row
            for row in rows
            if row.kind in ("server", "tool") and row.cycle_current is not None
        ] + [
            row
            for row in extra_override_rows
            if row.kind in ("server", "tool") and row.cycle_current is not None
        ]
        if not override_rows:
            return f"global default: {global_word}"
        override_servers = {row.server_key for row in override_rows}
        override_word = "override" if len(override_rows) == 1 else "overrides"
        server_word = "server" if len(override_servers) == 1 else "servers"
        return (
            f"global default: {global_word} · {len(override_rows)} {override_word} "
            f"across {len(override_servers)} {server_word}"
        )

    async def on_mcp_permissions_mode_state_cycle_requested(
        self, event: MCPPermissionsMode.StateCycleRequested
    ) -> None:
        """Apply one Space-cycled permission change (T6) via the T4 typed
        methods, then resync the matrix -- the single-writer contract:
        `MCPPermissionsMode` never touches the store itself, only posts
        the row it means and the state T2's cycle helpers already resolved.

        `event.new_state` is validated against `STORE_STATES` (trust
        boundary: this event crosses from the render-only child widget,
        and `MCPPermissionStore.set_*` would otherwise raise on a bad
        value AFTER partially reasoning about the write) before any setter
        runs -- `None` is legal (Inherit, for a `"server"`/`"tool"` row
        only) but anything else that isn't `"allow"`/`"ask"`/`"deny"` is
        rejected outright, no setter call at all.

        Minor 5: a `"tool"` row whose tool has dropped out of the catalog
        (`_tool_for()` returns `None`) cycled to `"allow"` is caught HERE,
        before the setter -- `set_tool_state(..., "allow", tool=None)`
        raises a `ValueError` whose message is an internal implementation
        detail ("tool is required to set state 'allow' ..."), which the
        generic `except` below would otherwise toast verbatim. Every other
        state (`"ask"`/`"deny"`/`None`) works fine with `tool=None` -- only
        `"allow"` needs the live tool to fingerprint.

        Task 4: `agent:builtin` rows have no `HubTool` at all -- they never
        appear in `_last_hub_tools` (that list is the MCP catalog), so
        `_tool_for()` always returns `None` for them. Without a branch here,
        EVERY built-in row's first cycle (inherit -> allow) would hit the
        "no longer in the catalog" guard above and never write. Skip the
        `HubTool` lookup for `BUILTIN_TOOL_SERVER_KEY` entirely and call
        `set_tool_state()` with no `tool=` -- safe only because Task 1 put
        `agent:builtin` in `HASH_FREE_SERVER_KEYS`, so the service doesn't
        require a tool to fingerprint for the rug-pull hash.
        """
        event.stop()
        service = self._service()
        if service is None:
            return
        if event.new_state is not None and event.new_state not in STORE_STATES:
            logger.warning(
                f"MCP permission cycle rejected invalid state: {event.new_state!r}"
            )
            self.app.notify(
                _toast(f"Ignored invalid permission state {event.new_state!r}."),
                severity="warning",
            )
            return
        cycled_tool: HubTool | None = None
        raw_cycled_state: str | None = None
        try:
            if event.row_kind == "global":
                if event.new_state is not None:
                    service.set_global_default(event.new_state)
            elif event.row_kind == "server":
                service.set_server_default(event.server_key, event.new_state)
            elif event.row_kind == "tool":
                if _is_raw_shell_tool(event.server_key, event.tool_name):
                    cycled_tool = self._tool_for(
                        event.server_key, event.tool_name or ""
                    )
                    if cycled_tool is None:
                        self.app.notify(
                            _toast(
                                "Raw shell policy is no longer in the catalog — "
                                "refresh and try again."
                            ),
                            severity="warning",
                        )
                        return
                    current = _resolve_raw_shell_state()(
                        self._effective_for_display(cycled_tool)
                    )
                    # This exact row is a two-state control. Re-derive from
                    # the rendered effective state because the generic child
                    # table cycles four raw-store rungs (including Allow and
                    # Inherit), neither of which is valid raw-shell policy.
                    next_state = "ask" if current == "deny" else "deny"
                    raw_cycled_state = next_state
                    service.set_tool_state(
                        event.server_key,
                        event.tool_name or "",
                        next_state,
                        tool=cycled_tool,
                    )
                elif event.server_key == BUILTIN_TOOL_SERVER_KEY:
                    # Task 4: built-in tools have no `HubTool` -- skip the
                    # catalog lookup and its "no longer in the catalog"
                    # guard, which would otherwise reject every built-in
                    # row's first press. `tool=` is omitted deliberately:
                    # `agent:builtin` is in `HASH_FREE_SERVER_KEYS`
                    # (Task 1), so `set_tool_state()` doesn't need a
                    # `HubTool` to fingerprint an "allow".
                    service.set_tool_state(
                        event.server_key, event.tool_name or "", event.new_state
                    )
                else:
                    cycled_tool = self._tool_for(
                        event.server_key, event.tool_name or ""
                    )
                    if cycled_tool is None and event.new_state == "allow":
                        self.app.notify(
                            _toast(
                                "Tool is no longer in the catalog — refresh and try again."
                            ),
                            severity="warning",
                        )
                        return
                    service.set_tool_state(
                        event.server_key,
                        event.tool_name or "",
                        event.new_state,
                        tool=cycled_tool,
                    )
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message("MCP permission cycle failed", exc),
            )
            self.app.notify(
                _toast(f"Permission update failed: {exc}"), severity="error"
            )
            return
        # Task 3 (MCP Hub Phase 6): the transient mutation echo -- pinned
        # copy shape `"{tool_name} → {ui_label} · "`, TOOL-row cycles only
        # (the pinned shape names a tool; a "server"/"global" row cycle has
        # no equivalent pinned copy, so those stay unechoed -- deliberate,
        # narrower scope, not an oversight).
        echo: str | None = None
        if event.row_kind == "tool":
            echoed_state = event.new_state
            if raw_cycled_state is not None:
                echoed_state = raw_cycled_state
            echo = f"{event.tool_name} → {_cycled_ui_label(echoed_state)} · "
        async with self._sync_children_lock:
            await self._sync_permissions_mode(echo=echo)

        # Minor 3: `_sync_permissions_mode()` above rebuilds the matrix's
        # OWN rows, but an already-open `#mcp-inspector-permission` block
        # for this same tool is a separate render (`show_permission()`,
        # driven by `MCPPermissionsMode.RowSelected`, not the matrix
        # resync) that keeps showing the pre-cycle rule until something
        # re-renders it -- today only the re-allow handler does. Refresh it
        # here too when it's explaining the tool that was just cycled.
        if event.row_kind == "tool" and cycled_tool is not None:
            inspector = self.query_one(MCPInspector)
            current_tool = inspector.current_permission_tool
            if (
                current_tool is not None
                and current_tool.server_key == cycled_tool.server_key
                and current_tool.name == cycled_tool.name
            ):
                await inspector.show_permission(
                    cycled_tool,
                    self._effective_for_display(cycled_tool),
                    cascade=self._cascade_for_tool(cycled_tool),
                )

    async def on_mcp_permissions_mode_kill_switch_toggled(
        self, event: MCPPermissionsMode.KillSwitchToggled
    ) -> None:
        event.stop()
        service = self._service()
        set_kill_switch = getattr(service, "set_kill_switch", None)
        if not callable(set_kill_switch):
            return
        try:
            set_kill_switch(event.value)
        except Exception as exc:
            # task-545/T6: global switch (MCP + built-in tools) -- see the
            # matching read-path comment in `_sync_permissions_mode` above.
            logger.warning(
                "{}", _safe_diagnostic_message("kill switch save failed", exc)
            )
            self.app.notify(
                _toast(f"Failed to save kill switch: {exc}"), severity="error"
            )
            return
        # Task 3: pinned mutation-echo shape for the kill switch --
        # `"kill switch → on/off · "`.
        echo = f"kill switch → {'on' if event.value else 'off'} · "
        async with self._sync_children_lock:
            await self._sync_permissions_mode(echo=echo)

    async def _show_selected_detail(
        self, canvas: MCPServersMode, selected: ReadinessSnapshot | None
    ) -> None:
        """Route the selected snapshot to the read-only detail pane or,
        for an external-server record when mutations are available, to the
        `MCPServerMutationsPanel` edit-mode host (T9).

        Credential slots are fetched fresh on every selection (not cached)
        -- they can change from other clients/sessions, and this only runs
        on an actual selection change, not on every keystroke.
        """
        if (
            selected is not None
            and self._is_external_record_key(selected.server_key)
            and self._server_mutations_available
        ):
            record = dict((selected.detail or {}).get("raw") or {})
            record.setdefault("server_id", selected.server_key.rsplit("/", 1)[-1])
            slots = await self._fetch_credential_slots(record.get("server_id"))
            await canvas.show_server_mutations(record, slots)
            return
        await canvas.show_detail(
            selected, mutations_available=self._server_mutations_available
        )

    async def _fetch_credential_slots(self, server_id: Any) -> list[dict[str, Any]]:
        service = self._service()
        if service is None or not server_id:
            return []
        try:
            result = await service.run_action(
                "external_server.slots.list", {"server_id": server_id}
            )
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message("MCP credential slot listing failed", exc),
            )
            return []
        slots = result.get("credential_slots") if isinstance(result, Mapping) else None
        return (
            [dict(s) for s in slots if isinstance(s, Mapping)]
            if isinstance(slots, list)
            else []
        )

    # -- modes & view state ---------------------------------------------------

    def set_mode(self, mode: str) -> None:
        if mode not in MCP_HUB_MODES:
            mode = "servers"
        # task-2901: `ContentSwitcher.current` raises for an id with no
        # child. A mode request landing before the deferred canvases mount
        # (a fast chip click in the first paint-to-load window) is stashed
        # and replayed by `_mount_deferred_canvases` — the same stash idea
        # `_reloading` uses for restores.
        if not self.query(f"#mcp-mode-canvas-{mode}"):
            self._pending_deferred_mode = mode
            return
        mode_changed = mode != self._active_mode
        self._active_mode = mode
        self.query_one(ContentSwitcher).current = f"mcp-mode-canvas-{mode}"
        if mode_changed:
            # Single emission point for mode changes (see ModeChanged) --
            # covers restore and inspector hub-action paths, which bypass
            # the screen's _activate_mode chip sync.
            self.post_message(self.ModeChanged(mode))
            # T7 review fix: a mode change is an "other interaction" per the
            # arm-then-confirm contract, so it must disarm any pending delete
            # confirmation -- the ContentSwitcher hides the servers canvas
            # without unmounting it, so nothing else resets the arm state on
            # a Servers -> Tools -> Servers round-trip. Dispatched as a
            # worker (set_mode is sync); no-op when unarmed.
            self.run_worker(
                self._disarm_canvas_delete,
                group="mcp-detail-disarm",
                exclusive=True,
            )
            # T6: a mode change also invalidates whatever tool the inspector
            # was showing -- switching AWAY from Tools mode leaves a stale
            # Test Tool panel behind otherwise; switching INTO it starts
            # with nothing selected anyway, so this is a no-op there.
            self.run_worker(
                self._clear_tool_view,
                group="mcp-tool-clear",
                exclusive=True,
            )

    async def _clear_tool_view(self) -> None:
        await self.query_one(MCPInspector).show_tool(None)
        # T7 (MCP Hub Phase 5): a mode change also invalidates whatever
        # execution-log entry the inspector was showing -- same rationale
        # as the `show_tool(None)` call above, one line up.
        await self.query_one(MCPInspector).show_audit_entry(None)
        # T8 (MCP Hub Phase 5): a mode change also invalidates whatever
        # Findings-table selection the inspector was showing -- same
        # rationale as the `show_audit_entry(None)` call above.
        await self.query_one(MCPInspector).show_finding(None)

    async def _disarm_canvas_delete(self) -> None:
        # Under `_sync_children_lock`: `disarm_delete()` rebuilds the detail
        # toolbar (awaited remove+mount), which must not interleave with a
        # concurrently running `_sync_children()` doing the same via
        # `show_detail()` -- same DuplicateIds hazard the lock exists for.
        async with self._sync_children_lock:
            await self.query_one(MCPServersMode).disarm_delete()

    def get_view_state(self) -> dict[str, Any]:
        return {
            "mode": self.active_mode,
            "source": self._source,
            "selected_server_key": self._selected_server_key,
            "scope": self._scope,
            "scope_ref": self._scope_ref,
        }

    def set_initial_view_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if self.is_mounted:
            # Always stash the latest requested state so a reload already in
            # flight (see `reload()`) can pick it up when it finishes,
            # instead of racing it with a worker started here.
            self._pending_view_state = dict(state)
            if not self._reloading:
                self.run_worker(
                    self._consume_pending_view_state(),
                    group="mcp-workbench-restore",
                    exclusive=True,
                )
        else:
            self._pending_view_state = dict(state)

    async def _consume_pending_view_state(self) -> None:
        """Apply `_pending_view_state` exactly once, then clear it."""
        state = self._pending_view_state
        self._pending_view_state = None
        if state:
            await self._apply_view_state(state)

    async def _apply_view_state(self, state: dict[str, Any]) -> None:
        # Tolerant restore: unknown keys ignored; legacy panel shape accepted.
        source = state.get("source") or state.get("selected_source")
        if source in ("local", "server") and source != self._source:
            await self._switch_source(str(source))
        # I2: a restored non-"servers" mode must also move the screen's
        # chip highlight -- set_mode() itself posts ModeChanged on any
        # actual change (single emission point), so no extra post here.
        self.set_mode(str(state.get("mode") or "servers"))
        # Distinguish "key absent" (leave the current selection alone --
        # e.g. the F-054 lone-problem preselect) from "key present with
        # value None" (an explicit "All servers" clear from the previous
        # session, which must win over the preselect). Mirrors the
        # `scope_ref` handling below.
        if "selected_server_key" in state:
            server_key = state["selected_server_key"]
            if server_key is None:
                self._selected_server_key = None
            elif (
                isinstance(server_key, str)
                and self._snapshot_for(server_key) is not None
            ):
                self._selected_server_key = server_key
        scope = state.get("scope") or state.get("selected_scope")
        if isinstance(scope, str) and scope:
            self._scope = scope
        # T7 carry-over: distinguish "key absent" (keep the current
        # scope_ref untouched) from "key present with value None" (an
        # explicit clear). `dict.get(key, _UNSET)` is required here because
        # `state.get("scope_ref")` alone can't tell "absent" from
        # "present-but-None" apart -- both return None.
        if "scope_ref" in state:
            raw_scope_ref = state["scope_ref"]
        elif "selected_scope_ref" in state:
            raw_scope_ref = state["selected_scope_ref"]
        else:
            raw_scope_ref = _UNSET
        if raw_scope_ref is not _UNSET:
            self._scope_ref = None if raw_scope_ref is None else str(raw_scope_ref)
        await self._sync_children()

    # -- event wiring -----------------------------------------------------------

    async def _switch_source(self, source: str) -> None:
        service = self._service()
        if service is not None:
            try:
                await service.select_source(source)
            except Exception as exc:
                logger.warning(
                    "{}", _safe_diagnostic_message("MCP source switch failed", exc)
                )
        self._source = source
        self._selected_server_key = None
        # T6: switching source invalidates any Tools-mode selection the
        # inspector was showing (the tool belonged to the OTHER source's
        # catalog), and also clears the finding detail pane (same reasoning).
        inspector = self.query_one(MCPInspector)
        await inspector.show_tool(None)
        await inspector.show_finding(None)
        self._snapshots = await self._collect_snapshots()
        await self._sync_children()
        self._rebind_inspector_advanced_context(service)

    async def on_mcp_rail_source_changed(self, event: MCPRail.SourceChanged) -> None:
        event.stop()
        await self._switch_source(event.source)

    async def _select_server_key(self, server_key: str | None) -> None:
        """Shared selection path for both the rail and the overview table.

        T9: previously only the rail's handler informed the service which
        target is active (`select_server_target`) and re-collected
        snapshots; the table's row-click handler just resynced from the
        existing `_snapshots`. That gap didn't matter in Phase 1 (nothing
        was target-scoped), but now that `_collect_snapshots()` loads a
        selected target's external-server records off the service's *active*
        target, a table-driven selection had no way to make it active --
        both entry points now share this one path.

        I1 (MCP Hub Phase 6 finale, review -- the program's 6th occurrence
        of this same stale-panel class): selecting a different server also
        invalidates any Findings-detail pane the inspector was showing --
        the previous server's finding, complete with remediation buttons
        wired to its (now stale) `_current_finding_server_key`, must not
        survive into the new selection. Its own "Refresh" button pressed
        after the switch would otherwise silently refresh the WRONG
        target while still toasting a success message. Mirrors
        `_switch_source()`'s identical T6 clear.
        """
        self._selected_server_key = server_key
        # T6: selecting a different server invalidates any Tools-mode
        # selection the inspector was showing -- "switching modes or
        # servers clears the tool view" -- and (I1 above) the Findings
        # detail pane for the same reason.
        inspector = self.query_one(MCPInspector)
        await inspector.show_tool(None)
        await inspector.show_finding(None)
        service = self._service()
        if (
            service is not None
            and server_key is not None
            and server_key.startswith("server:")
            and "/" not in server_key
        ):
            try:
                await service.select_server_target(server_key.split(":", 1)[1])
            except Exception as exc:
                logger.warning(
                    "{}",
                    _safe_diagnostic_message("MCP server target selection failed", exc),
                )
        if self._source == "server":
            self._snapshots = await self._collect_snapshots()
        await self._sync_children()
        self._rebind_inspector_advanced_context(service)

    async def on_mcp_rail_server_selected(self, event: MCPRail.ServerSelected) -> None:
        event.stop()
        await self._select_server_key(event.server_key)

    async def on_mcp_rail_scope_changed(self, event: MCPRail.ScopeChanged) -> None:
        event.stop()
        # C1 defense in depth: a no-op ScopeChanged (already-tracked scope +
        # scope_ref) must not round-trip to the service or resync children.
        # The primary fix is the rail's own mount-echo guard (mcp_rail.py),
        # but this dedup means a stray duplicate here can't self-sustain a
        # recompose storm even if some future caller posts one.
        if (event.scope, event.scope_ref) == (self._scope, self._scope_ref):
            return
        service = self._service()
        if service is not None:
            try:
                await service.select_scope(event.scope, event.scope_ref)
            except Exception as exc:
                logger.warning(
                    "{}", _safe_diagnostic_message("MCP scope selection failed", exc)
                )
        self._scope = event.scope
        self._scope_ref = event.scope_ref
        # No `_sync_children()` here: nothing scope-dependent renders in
        # Phase 1 (the rail's scope selects reflect it purely from the last
        # explicit sync_state() call), and resyncing would recompose the
        # rail -> remount its Selects -> another mount-echo -> another
        # ScopeChanged, which is exactly the storm this handler used to feed.
        #
        # T9: mutation availability IS scope-dependent though -- recompute it
        # cheaply (no snapshot/rail/detail resync, just the Add-server
        # button's gating) so a scope change alone doesn't leave it stale.
        if self._source == "server":
            self._server_mutations_available = self._compute_server_mutations_available(
                service
            )
            self.query_one(MCPServersMode).set_mutations_available(
                self._server_mutations_available,
                mutation_target_label=self._active_target_label(),
            )

    async def on_mcp_servers_mode_server_row_selected(
        self, event: MCPServersMode.ServerRowSelected
    ) -> None:
        event.stop()
        await self._select_server_key(event.server_key)

    async def on_mcp_inspector_hub_action_requested(
        self, event: MCPInspector.HubActionRequested
    ) -> None:
        """Route one `HubActionRequested` -- from either the readiness pane's
        own action buttons or (Task 2, MCP Hub Phase 6) a Findings-detail
        remediation button (`MCPInspector.show_finding()`'s mapped
        `HubAction` buttons).

        Task 2 adds the `server:`-key branches below: `REFRESH_DISCOVERY`
        invalidates the (source, target)-cached governance/findings
        listings and runs a full resync (`_refresh_server_discovery()`,
        shared with the Tools-mode empty-state's own "refresh" action under
        server source -- and, New Minor 2/MCP Hub Phase 6 finale, now passed
        `event.server_key` so the resync lands on THAT target rather than
        whatever was already active/rail-selected); `OPEN_CREDENTIALS`
        selects the server and switches to Servers mode with an honest
        "managed on the server" notice (no credentials editor exists for
        server source). Every other action that reaches this handler with a
        `server:` key -- `CONNECT`/`VALIDATE`/`EDIT_CONFIG` (lifecycle-only,
        local-source seams) or anything else -- falls through to the final
        catch-all toast rather than being silently dropped.
        """
        event.stop()
        if event.action is HubAction.VIEW_DETAILS and event.server_key:
            # F1 (PR #722 Qodo bot review): route through `_select_server_
            # key()` rather than assigning `_selected_server_key` directly --
            # that shared path also tells the SERVICE which target is now
            # active (`select_server_target()`) and re-collects `_snapshots`
            # under it. Without it, a remediation button naming a target
            # OTHER than the one the service already considers active would
            # desync the two: `_collect_snapshots()`'s external-servers fetch
            # stays scoped to the OLD (service) target while this workbench
            # labels/caches whatever comes back under the NEW (UI-selected)
            # key -- wrong-target data under the right-looking key. Mirrors
            # the rail/table selection path, which already gets this right.
            await self._select_server_key(event.server_key)
            self.set_mode("servers")
        elif event.action is HubAction.OPEN_TOOL_CATALOG:
            self.set_mode("tools")
        elif event.action is HubAction.OPEN_AUDIT:
            self.set_mode("audit")
        elif (
            event.action in _HUB_ACTION_TO_LIFECYCLE_VERB
            and event.server_key
            and event.server_key.startswith("local:")
        ):
            profile_id = event.server_key.split(":", 1)[1]
            self._start_lifecycle(
                event.server_key,
                profile_id,
                _HUB_ACTION_TO_LIFECYCLE_VERB[event.action],
            )
        elif (
            event.action is HubAction.EDIT_CONFIG
            and event.server_key
            and event.server_key.startswith("local:")
        ):
            profile_id = event.server_key.split(":", 1)[1]
            record = self._catalog_records.get(profile_id)
            await self.query_one(MCPServersMode).show_form(record)
        elif (
            event.action is HubAction.REFRESH_DISCOVERY
            and event.server_key
            and event.server_key.startswith("server:")
            and self._source == "server"
        ):
            await self._refresh_server_discovery(event.server_key)
        elif (
            event.action is HubAction.OPEN_CREDENTIALS
            and event.server_key
            and event.server_key.startswith("server:")
            and self._source == "server"
        ):
            # F1 (PR #722 Qodo bot review): same fix as VIEW_DETAILS above --
            # route through `_select_server_key()` so a target switch
            # implied by this remediation button actually reaches the
            # service.
            await self._select_server_key(event.server_key)
            self.set_mode("servers")
            self.app.notify("Credentials are managed in the server's config.")
        elif event.server_key and event.server_key.startswith("server:"):
            # No more silent drops: CONNECT/VALIDATE/EDIT_CONFIG (local-only
            # lifecycle seams) and any other unrecognized action posted with
            # a server-source key land here instead of doing nothing.
            self.app.notify("Managed on the server.")

    async def _refresh_server_discovery(self, server_key: str | None = None) -> None:
        """Cache-invalidating full resync for a server-source "refresh
        discovery" request (Task 2, MCP Hub Phase 6).

        Shared by two entry points that both need it instead of a per-server
        lifecycle action that stays disabled for server-source snapshots in
        the inspector (`_wired_actions()` only wires CONNECT/VALIDATE/
        REFRESH_DISCOVERY for local source): the inspector's own
        REFRESH_DISCOVERY routing above (a `server:` key -- e.g. a Findings-
        detail remediation button), and the Tools-mode empty-state's
        "refresh" button under server source (`on_mcp_tools_mode_empty_
        action_requested()`, UX item 10's fix, which has no specific server
        in mind and always passes `None`).

        The findings (T8) and governance-profiles (T11) listings are each
        cached by `(source, target)` identity and otherwise only refetched
        on an actual identity change (`_server_findings()`/`_server_
        governance_profiles()`) -- resetting both cache keys back to the
        module's `_UNSET` sentinel forces the next full pass to treat the
        identity as "changed" and refetch, exactly like a genuine
        source/target switch would. `self._snapshots` is also re-collected
        (mirrors `_select_server_key()`/`_switch_source()`) so the
        readiness/tool catalog reflects a fresh discovery pass too, not
        just the two Advanced-derived caches.

        New Minor 2 (MCP Hub Phase 6 finale, review, linked to I1): a blanket
        cache-key reset alone is not enough -- the identity the NEXT fetch
        actually lands on is `(self._source, self._active_service_target_id())`,
        which previously ignored `server_key` entirely and always resolved
        to whatever was already active/rail-selected. A Findings-detail
        remediation button for a server OTHER than that one (its own
        `target_id` field, resolved by `_finding_owning_server_key()`) would
        then refresh the WRONG target's cache while leaving the finding's
        real owning target's data untouched. When `server_key` names a
        DIFFERENT target than the one already active, this switches the
        service's own active target (so the real fetch lands on it too, not
        just the client-side cache key) and this workbench's own selection
        (so `_active_service_target_id()` resolves to it for the rest of
        this pass) before invalidating and resyncing. `None` (the Tools-mode
        empty-state's call site) preserves the original "refresh whatever's
        active" behavior untouched.
        """
        service = self._service()
        target_id = _target_id_from_server_key(server_key)
        if target_id is not None and target_id != self._active_service_target_id():
            if service is not None:
                try:
                    await service.select_server_target(target_id)
                except Exception as exc:
                    logger.warning(
                        "{}",
                        _safe_diagnostic_message(
                            "MCP server target selection failed", exc
                        ),
                    )
            self._selected_server_key = server_key
            self._rebind_inspector_advanced_context(service)
        self._findings_cache = None
        self._findings_cache_key = _UNSET
        self._governance_profiles_cache = None
        self._governance_profiles_cache_key = _UNSET
        self._snapshots = await self._collect_snapshots()
        await self._sync_children()
        self.app.notify("Server discovery refreshed.")

    async def on_mcp_servers_mode_add_server_requested(
        self, event: MCPServersMode.AddServerRequested
    ) -> None:
        event.stop()
        await self._open_add_server(notify_if_gated=False)

    def on_mcp_tools_mode_local_tools_enabled_changed(
        self, event: MCPToolsMode.LocalToolsEnabledChanged
    ) -> None:
        """Persist the workspace, web, and Watchlists master switch."""
        event.stop()
        self.run_worker(
            self._save_tools_mode_local_enabled(event.enabled),
            group="mcp-tools-local-enabled",
            exclusive=True,
        )

    async def _save_tools_mode_local_enabled(self, enabled: bool) -> None:
        try:
            saved = await asyncio.to_thread(
                save_setting_to_cli_config,
                "console",
                "local_tools_enabled",
                enabled,
            )
        except Exception as exc:
            logger.warning(
                "MCP Tools-mode local master save failed (error_type={}).",
                type(exc).__name__,
            )
            canvas = self._refresh_local_tools_controls()
            if canvas is not None:
                canvas.set_local_config_status(
                    "Save failed. The persisted setting is shown.", error=True
                )
            self.app.notify(
                _toast(f"Failed to save local tool setting: {exc}"),
                severity="error",
            )
            return
        if not saved:
            canvas = self._refresh_local_tools_controls()
            if canvas is not None:
                canvas.set_local_config_status(
                    "Save failed. The persisted setting is shown.", error=True
                )
            self.app.notify("Failed to save local tool setting.", severity="error")
            return

        self._snapshots = await self._collect_snapshots()
        await self._sync_children()
        canvas = self.query_one(MCPToolsMode)
        state = "Enabled" if enabled else "Disabled"
        canvas.set_local_config_status(
            f"{state}. The next Console agent run will use this setting; "
            "calls still follow Ask, Allow, or Off permissions.",
            error=False,
        )

    def on_mcp_tools_mode_workspace_root_save_requested(
        self, event: MCPToolsMode.WorkspaceRootSaveRequested
    ) -> None:
        """Validate and persist Tools mode's workspace confinement root."""
        event.stop()
        self.run_worker(
            self._save_tools_mode_workspace_root(event.workspace_root),
            group="mcp-tools-workspace-root",
            exclusive=True,
        )

    async def _save_tools_mode_workspace_root(self, raw_root: str) -> None:
        requested = raw_root.strip()
        stored = ""
        display = "the app folder"
        if requested:
            try:
                candidate = Path(requested).expanduser()
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                resolved = validate_path(
                    candidate,
                    candidate.parent,
                    redact_paths=True,
                    allow_hidden=True,
                )
                if not resolved.is_dir():
                    raise ValueError("path is not an existing directory")
            except (OSError, RuntimeError, ValueError) as exc:
                canvas = self.query_one(MCPToolsMode)
                canvas.set_local_config_status(
                    f"Workspace root not saved: {exc}.", error=True
                )
                self.app.notify(
                    _toast(f"Workspace root not saved: {exc}."), severity="error"
                )
                return
            stored = str(resolved)
            display = stored

        try:
            saved = await asyncio.to_thread(
                save_setting_to_cli_config,
                "console",
                "workspace_root",
                stored,
            )
        except Exception as exc:
            logger.warning(
                "MCP Tools-mode workspace root save failed (error_type={}).",
                type(exc).__name__,
            )
            canvas = self._refresh_local_tools_controls()
            if canvas is not None:
                canvas.set_local_config_status(
                    "Save failed. The persisted workspace root is shown.",
                    error=True,
                )
            self.app.notify(
                _toast(f"Failed to save workspace root: {exc}"), severity="error"
            )
            return
        if not saved:
            canvas = self._refresh_local_tools_controls()
            if canvas is not None:
                canvas.set_local_config_status(
                    "Save failed. The persisted workspace root is shown.",
                    error=True,
                )
            self.app.notify("Failed to save workspace root.", severity="error")
            return

        self._snapshots = await self._collect_snapshots()
        await self._sync_children()
        self.query_one(MCPToolsMode).set_local_config_status(
            f"Saved. The next Console agent run is confined to {display}.",
            error=False,
        )

    async def on_mcp_tools_mode_empty_action_requested(
        self, event: MCPToolsMode.EmptyActionRequested
    ) -> None:
        """Route the Tools mode diagnostic empty state's primary Button.

        `"add_server"` reuses the existing add-form path (same as the
        overview's own Add-server button, notifying if gated -- reachable
        here with no disabled-button affordance to lean on, same rationale
        as `open_add_server_form()`'s `a`-keybinding entry point). `"connect"`
        (local source only, see `_empty_tools_diagnosis()`) just points the
        user at Servers mode, where the actual per-server lifecycle actions
        live (Task 5 scope; Tools mode itself gains no lifecycle buttons).

        `"refresh"` under SERVER source (UX item 10, Task 2 MCP Hub Phase 6)
        instead routes to the cache-invalidating full resync
        (`_refresh_server_discovery()`) -- the plain "go look at Servers
        mode" copy below would point at a disabled per-server action there.
        `"refresh"` under local source keeps the original behavior.
        """
        event.stop()
        if event.action_key == "add_server":
            await self._open_add_server(notify_if_gated=True)
        elif event.action_key == "refresh" and self._source == "server":
            await self._refresh_server_discovery()
        elif event.action_key in ("connect", "refresh"):
            self.set_mode("servers")
            self.app.notify("Select a server below to connect or refresh its tools.")

    def _tool_for_row_key(self, tool_id: str) -> HubTool | None:
        """Resolve a Tools-mode DataTable row key (`HubTool.tool_id`, a
        packed `"server_key::name"` display/dedup string -- see
        `mcp_tools_mode.py`) against `_last_hub_tools`.

        task-233: packed ids remain legal AS ROW KEYS (mcp_tools_mode.py is
        unchanged), but this is the only place in this module that still
        compares against one -- everything downstream of a selection (Test
        Tool execution) carries `(server_key, tool_name)` as separate
        fields instead. See `_tool_for()` for that field-based lookup.
        """
        for tool in self._last_hub_tools:
            if tool.tool_id == tool_id:
                return tool
        return None

    def _tool_for(self, server_key: str, tool_name: str) -> HubTool | None:
        """Resolve a `(server_key, tool_name)` pair against `_last_hub_tools`.

        task-233: the field-based counterpart to `_tool_for_row_key()` --
        compares `HubTool.server_key`/`HubTool.name` directly rather than
        parsing (or matching) a packed id string.
        """
        for tool in self._last_hub_tools:
            if tool.server_key == server_key and tool.name == tool_name:
                return tool
        return None

    async def on_mcp_tools_mode_tool_selected(
        self, event: MCPToolsMode.ToolSelected
    ) -> None:
        """T6: route a Tools-mode row selection to the inspector's tool
        detail view. `_tool_for_row_key()` resolves the row's packed
        `tool_id` against `_last_hub_tools` (populated by the same
        `_sync_tools_mode()` pass that fed the DataTable this selection came
        from) -- a stale selection whose tool has since dropped out of the
        catalog (disconnect, refresh) resolves to `None`, which
        `show_tool()` renders as "nothing selected" rather than crashing.

        T7: a resolved tool also gets its permission rule explained --
        `_effective_for_display()` appends the block below the tool detail
        via `show_tool()`'s `effective` keyword.
        """
        event.stop()
        tool = self._tool_for_row_key(event.tool_id)
        effective = self._effective_for_display(tool) if tool is not None else None
        await self.query_one(MCPInspector).show_tool(tool, effective=effective)

    def _effective_for_display(self, tool: HubTool) -> EffectiveToolState:
        """Resolve one tool's `EffectiveToolState` for the inspector's
        permission explanation -- Tools-mode's tool-detail-plus-permission
        block (`show_tool(tool, effective=...)`) and Permissions-mode's own
        matrix-row selection (`show_permission()`) both go through here.

        Prefers `self._last_effective_states` -- the batch resolution
        `_sync_permissions_mode()` already computed this same
        `_sync_children()` pass via `effective_tool_states()` -- over a
        second, redundant lookup. Falls back to one fresh single-tool
        `service.gate_tool_test()` call (T4) when the tool isn't in that
        cache (e.g. a service that exposes `gate_tool_test()` but not the
        batch `effective_tool_states()`); a raising gate fails CLOSED
        (deny). No seam at all -> the
        same `EffectiveToolState(state="ask", origin="global_default")`
        fallback `_build_permission_rows()` already uses for a tool missing
        from the batch dict.
        """
        cached = self._last_effective_states.get((tool.server_key, tool.name))
        if cached is not None:
            return cached
        service = self._service()
        gate_check = getattr(service, "gate_tool_test", None)
        if callable(gate_check):
            try:
                return gate_check(tool)
            except Exception as exc:
                logger.warning(
                    "{}",
                    _safe_diagnostic_message(
                        "MCP permission resolution failed; failing closed", exc
                    ),
                )
                return EffectiveToolState(state="deny", origin="gate_error")
        return EffectiveToolState(state="ask", origin="global_default")

    async def on_mcp_permissions_mode_row_selected(
        self, event: MCPPermissionsMode.RowSelected
    ) -> None:
        """T7: route a Permissions-mode matrix row selection to the
        inspector. A `"tool"` row resolves its `HubTool` (`_tool_for()`)
        and shows the permission explanation standalone
        (`MCPInspector.show_permission()`) -- NOT the full tool-detail
        block (`#mcp-inspector-tool`, Test Tool button and all), which is
        Tools-mode's own selection surface. A pinned `"global"`/`"server"`
        row -- or a `"tool"` row whose tool has since dropped out of the
        catalog -- just clears whatever tool/permission view the inspector
        was last showing (`show_tool(None)`, which also hides
        `#mcp-inspector-permission`; see that method).

        Task 3 (MCP Hub Phase 6): also threads this tool's cascade tuple
        (`_cascade_for_tool()`) so the block renders the three provenance
        rungs instead of the single origin sentence.

        Fix 1 (PR #906 review): an `agent:builtin` `"tool"` row has no
        `HubTool` at all -- `_tool_for()` only ever searches
        `_last_hub_tools` (the MCP catalog, Constraint 1/5), so it always
        returned `None` for one and this handler fell through to
        `show_tool(None)` -- the newest, most interactive section of this
        matrix was the only one that blanked the inspector on click instead
        of explaining the row. Routed here instead, from `_last_builtin_
        effective` (the state `_builtin_permission_matrix_rows()` already
        resolved this same pass -- no second read of the store or the
        built-in catalog) into a `HubTool` built locally for display only
        (`show_permission()`'s Static widgets and its Re-allow/goto-
        permission buttons read `tool.name`/`tool.server_key`/
        `tool.server_label`, never anything catalog-specific), via
        `show_permission()` -- the SAME entry point an MCP tool row uses.
        `cascade=None` is deliberate: built-ins have no MCP tool/server/
        global cascade to show three rungs for, so this falls back to the
        plain per-tool origin sentence (`_ORIGIN_SENTENCES` already carries
        a `"builtin_default"` entry for it). A built-in row with no cached
        state (dropped from the live registry between resyncs) clears the
        inspector the same as a dropped MCP tool would.
        """
        event.stop()
        inspector = self.query_one(MCPInspector)
        if event.row_kind == "tool" and event.server_key == BUILTIN_TOOL_SERVER_KEY:
            effective = self._last_builtin_effective.get(
                (event.server_key, event.tool_name or "")
            )
            if effective is None:
                await inspector.show_tool(None)
                return
            builtin_tool = HubTool(
                server_key=BUILTIN_TOOL_SERVER_KEY,
                server_label=_BUILTIN_SECTION_LABEL,
                source="builtin",
                name=event.tool_name or "",
                description="",
                input_schema=None,
                tags=(),
                stale=False,
                executable=False,
            )
            await inspector.show_permission(builtin_tool, effective, cascade=None)
            return
        tool = (
            self._tool_for(event.server_key, event.tool_name or "")
            if event.row_kind == "tool"
            else None
        )
        if tool is None:
            await inspector.show_tool(None)
            return
        await inspector.show_permission(
            tool,
            self._effective_for_display(tool),
            cascade=self._cascade_for_tool(tool),
        )

    # -- T7 (MCP Hub Phase 5): Audit mode ------------------------------------

    async def on_mcp_audit_mode_entry_selected(
        self, event: MCPAuditMode.EntrySelected
    ) -> None:
        """Route an Audit-mode row selection to the inspector's audit-entry
        detail view. `event.index` is looked up against `_last_audit_entries`
        (the SAME list `_sync_audit_mode()` handed `MCPAuditMode` this pass)
        -- an out-of-range index (a stale selection racing a background
        resync that shrank the window) resolves to `None`, which
        `show_audit_entry()` renders as "nothing selected" rather than
        crashing.
        """
        event.stop()
        entry = (
            self._last_audit_entries[event.index]
            if 0 <= event.index < len(self._last_audit_entries)
            else None
        )
        await self.query_one(MCPInspector).show_audit_entry(entry)

    async def on_mcp_audit_mode_finding_selected(
        self, event: MCPAuditMode.FindingSelected
    ) -> None:
        """Route an Audit-mode Findings-table row selection to the
        inspector's finding detail view (T8, MCP Hub Phase 5). Mirrors
        `on_mcp_audit_mode_entry_selected()` exactly -- `event.index` is
        looked up against `_last_audit_findings` (the SAME list `_sync_
        audit_mode()` handed `MCPAuditMode` this pass); an out-of-range
        index (a stale selection racing a background resync that shrank
        the list) resolves to `None`, which `show_finding()` renders as
        "nothing selected" rather than crashing.

        Task 2 (MCP Hub Phase 6): a resolved finding also gets its owning
        server key resolved (`_finding_owning_server_key()`) and threaded
        into `show_finding()`'s `server_key` keyword, so the detail view's
        new remediation-action buttons know which server a routed
        `HubActionRequested` belongs to.
        """
        event.stop()
        finding = (
            self._last_audit_findings[event.index]
            if 0 <= event.index < len(self._last_audit_findings)
            else None
        )
        server_key = (
            self._finding_owning_server_key(finding) if finding is not None else None
        )
        await self.query_one(MCPInspector).show_finding(finding, server_key=server_key)

    def _finding_owning_server_key(self, finding: Mapping[str, Any]) -> str | None:
        """The finding's owning server key (Task 2, MCP Hub Phase 6).

        Findings carry no fixed wire schema -- when one happens to carry a
        target-level identity field (a handful of defensive key aliases,
        mirroring `mcp_inspector._finding_text()`'s own multi-alias style),
        that wins: `f"server:{value}"`. Otherwise falls back to the
        currently selected rail server (`self._selected_server_key`) --
        Findings only ever render under server source (`MCPAuditMode.
        update_findings()`'s own `source` gate), so whatever is selected
        there, if anything, is already a `"server:..."` key. `None` when
        neither resolves (nothing rail-selected either).
        """
        for key in ("target_id", "server_target_id", "server_id"):
            value = finding.get(key)
            if value not in (None, ""):
                return f"server:{value}"
        return self._selected_server_key

    async def on_mcp_audit_mode_sub_view_changed(
        self, event: MCPAuditMode.SubViewChanged
    ) -> None:
        """Clear the now-inactive Audit sub-view pane's inspector detail
        (Critical fix, MCP Hub Phase 5 T8 review). `MCPAuditMode` itself
        only flips which of its two panes is visible on a toggle press --
        it never touches `MCPInspector` at all, so a prior Executions-row
        selection's `#mcp-inspector-audit` (or a prior Findings-row
        selection's `#mcp-inspector-finding`) stayed mounted and visible
        after switching sub-view, and selecting a row in the newly-visible
        pane then left BOTH detail panels stacked on screen at once.
        `event.sub_view` is the pane now visible, so this clears the OTHER
        one.

        Sequential awaits directly in this handler, not a worker dispatch
        (T7 lesson, see `_open_audit_tool()`/`_open_audit_permission()`
        above): those two methods must share the "mcp-tool-clear" exclusive
        worker group with `set_mode()`'s own clear, so they re-do the clear
        themselves rather than trust a same-group predecessor to have run
        it. There is no such predecessor here -- this handler's own awaits
        are the only write to the inspector this pass, so a plain
        sequential await is enough for the clear to actually execute.
        """
        event.stop()
        inspector = self.query_one(MCPInspector)
        if event.sub_view == "findings":
            await inspector.show_audit_entry(None)
        else:
            await inspector.show_finding(None)

    async def on_mcp_inspector_audit_open_tool_requested(
        self, event: MCPInspector.AuditOpenToolRequested
    ) -> None:
        """Route the audit-entry detail's "Open tool" button: resolve the
        entry's `(server_key, tool_name)` against `_last_hub_tools`
        (`_tool_for()`, same lookup `on_mcp_tools_mode_tool_selected()`
        uses) and, when found, switch to Tools mode, select its row, and
        show its full tool detail in the inspector -- a tool that has since
        dropped out of the catalog is a warning toast, never a crash.

        The populate work is dispatched into the SAME exclusive worker
        group (`"mcp-tool-clear"`) `set_mode()` just used for its own
        `_clear_tool_view()` call, added HERE synchronously (no `await`
        between the two `run_worker()` calls) -- Textual cancels the
        PREVIOUSLY-QUEUED worker in an exclusive group at `add_worker()`
        time, before it ever runs, not at completion time. That means
        the queued `_clear_tool_view` callable is never invoked here --
        `_open_audit_tool()`
        below does NOT rely on that cancelled worker to clear the audit
        panel; it clears `#mcp-inspector-audit` itself, explicitly, before
        populating the Tools-mode detail (Critical fix: the stale
        audit-entry detail -- including its own live "Open tool"/"Adjust
        permission" buttons -- used to stay mounted underneath the new
        detail otherwise).
        """
        event.stop()
        tool = self._tool_for(event.server_key, event.tool_name)
        if tool is None:
            self.app.notify(
                _toast(
                    f"{event.server_key}::{event.tool_name}: tool no longer available."
                ),
                severity="warning",
            )
            return
        self.set_mode("tools")
        self.run_worker(
            partial(self._open_audit_tool, tool),
            group="mcp-tool-clear",
            exclusive=True,
        )

    async def _open_audit_tool(self, tool: HubTool) -> None:
        inspector = self.query_one(MCPInspector)
        # Explicit clear -- see on_mcp_inspector_audit_open_tool_requested()'s
        # docstring: set_mode()'s _clear_tool_view() worker (which would
        # otherwise hide #mcp-inspector-audit via show_audit_entry(None))
        # is cancelled before it runs by this method's own dispatch into
        # the same exclusive "mcp-tool-clear" group, so it must not be
        # relied upon here.
        await inspector.show_audit_entry(None)
        await self.query_one(MCPToolsMode).select_tool_row(tool.tool_id)
        await inspector.show_tool(tool, effective=self._effective_for_display(tool))

    async def on_mcp_inspector_audit_adjust_permission_requested(
        self, event: MCPInspector.AuditAdjustPermissionRequested
    ) -> None:
        """Route the audit-entry detail's "Adjust permission" button through
        the shared jump helper (`_goto_permission_row()`, Task 3, MCP Hub
        Phase 6) -- one of three callers; see that method's own docstring.
        """
        event.stop()
        await self._goto_permission_row(event.server_key, event.tool_name)

    async def on_mcp_inspector_change_in_permissions_requested(
        self, event: MCPInspector.ChangeInPermissionsRequested
    ) -> None:
        """Route either "Change in Permissions" button (Task 3, MCP Hub
        Phase 6: the Tools-mode permission block's own button, and the Test
        Tool panel's blocked/ask button) through the SAME shared jump
        helper the audit drill uses -- see `_goto_permission_row()`.
        """
        event.stop()
        await self._goto_permission_row(event.server_key, event.tool_name)

    async def _goto_permission_row(self, server_key: str, tool_name: str) -> None:
        """Shared routing for every "jump to this tool's Permissions-mode
        row" entry point (Task 3, MCP Hub Phase 6): the audit drill's
        "Adjust permission" button, the Tools-mode permission block's
        "Change in Permissions" button, and the Test Tool panel's own
        blocked/ask button -- one implementation, three callers, no
        duplicated mode-switch-plus-matrix-row-selection logic (extracted
        from what was `on_mcp_inspector_audit_adjust_permission_requested()`'s
        own body pre-Task-3).

        Mirrors `on_mcp_inspector_audit_open_tool_requested()`'s own
        exclusive-group dispatch rationale and explicit-clear fix: an
        unresolvable tool (dropped out of the catalog) is a warning toast,
        never a mode switch; a resolved one switches to Permissions mode
        synchronously (so `active_mode` reads correctly the instant this
        returns) and dispatches the actual row-select-plus-render work
        (`_open_audit_permission()`) into the SAME exclusive
        `"mcp-tool-clear"` worker group `set_mode()` just used for its own
        `_clear_tool_view()` -- added HERE synchronously (no `await`
        between the two `run_worker()` calls) so Textual cancels the
        previously-queued clear before it ever runs, and
        `_open_audit_permission()` does its own explicit
        `show_audit_entry(None)` rather than relying on that cancelled
        worker (see its own comment).
        """
        tool = self._tool_for(server_key, tool_name)
        if tool is None:
            self.app.notify(
                _toast(f"{server_key}::{tool_name}: tool no longer available."),
                severity="warning",
            )
            return
        self.set_mode("permissions")
        self.run_worker(
            partial(self._open_audit_permission, tool),
            group="mcp-tool-clear",
            exclusive=True,
        )

    async def _open_audit_permission(self, tool: HubTool) -> None:
        inspector = self.query_one(MCPInspector)
        # Explicit clear -- same stale-audit-panel hazard as
        # _open_audit_tool() above; see its comment for the mechanism.
        # Harmless (a no-op re-hide) for the two non-audit
        # `_goto_permission_row()` callers, where `#mcp-inspector-audit` is
        # already hidden.
        await inspector.show_audit_entry(None)
        # Critical review fix: the other two `_goto_permission_row()`
        # callers -- the Tools-mode permission block's own "Change in
        # Permissions" button, and the Test Tool panel's blocked/ask
        # button -- fire from Tools mode, where `#mcp-inspector-tool` and
        # its open Test Tool panel are populated. `set_mode()`'s own
        # `_clear_tool_view()` worker
        # -- which would otherwise hide it via `show_tool(None)` -- is
        # cancelled by this method's SAME exclusive `"mcp-tool-clear"`
        # dispatch before it ever runs (the exact mechanism the comment
        # above already documents for the audit panel), so this must clear
        # `#mcp-inspector-tool` itself too, or the stale tool detail stays
        # stacked underneath the new
        # Permissions-mode block. Harmless no-op for the audit-drill
        # caller, where `#mcp-inspector-tool` is already hidden.
        await inspector.show_tool(None)
        self.query_one(MCPPermissionsMode).select_tool_row(tool.server_key, tool.name)
        await inspector.show_permission(
            tool,
            self._effective_for_display(tool),
            cascade=self._cascade_for_tool(tool),
        )

    async def on_mcp_inspector_reallow_requested(
        self, event: MCPInspector.ReallowRequested
    ) -> None:
        """T7: re-allow a rug-pull-downgraded tool override -- store the
        tool's CURRENT definition hash and set its state back to "allow"
        via T4's `set_tool_state()`, then resync the matrix so its ⚠
        marker clears. Also refreshes the inspector's own (already-open)
        permission block, since the matrix resync above doesn't touch it.

        Guard: an unresolvable tool (dropped out of the catalog since the
        permission block was rendered) is a warning toast, never a store
        call -- `set_tool_state(..., "allow", ...)` requires a live
        `HubTool` to fingerprint (see that method's own `tool` docstring).
        """
        event.stop()
        tool = self._tool_for(event.server_key, event.tool_name)
        if tool is None:
            self.app.notify(
                _toast(
                    f"{event.server_key}::{event.tool_name}: tool no longer available."
                ),
                severity="warning",
            )
            return
        service = self._service()
        set_tool_state = getattr(service, "set_tool_state", None)
        if not callable(set_tool_state):
            return
        try:
            set_tool_state(event.server_key, event.tool_name, "allow", tool=tool)
        except Exception as exc:
            logger.warning("{}", _safe_diagnostic_message("MCP re-allow failed", exc))
            self.app.notify(_toast(f"Re-allow failed: {exc}"), severity="error")
            return
        # Task 3: re-allow always sets "allow" -- reuses the tool-cycle
        # mutation-echo shape (`_cycled_ui_label("allow")` == "Allow").
        echo = f"{event.tool_name} → {_cycled_ui_label('allow')} · "
        async with self._sync_children_lock:
            await self._sync_permissions_mode(echo=echo)
        await self.query_one(MCPInspector).show_permission(
            tool,
            self._effective_for_display(tool),
            cascade=self._cascade_for_tool(tool),
        )

    async def open_test_for_selected_tool(self) -> None:
        """T8: entry point for the `t` keybinding (mcp_screen.py's
        `action_mcp_test_tool`) -- open the Test Tool panel for whatever
        tool the inspector currently has selected.

        Mirrors `open_add_server_form()`'s T13 rationale for a keybinding
        that can reach a state a disabled/absent button would otherwise
        gate: with nothing selected, or a selected-but-non-executable
        (server-source) tool -- neither has a `Test Tool` button to press --
        this notifies instead of silently no-opping. The two cases get
        distinct copy (`MCPInspector.open_test_panel()`'s three-way status
        tells them apart): "Select a tool in Tools mode first." for no
        selection, and the same "Server-source tools are display-only."
        copy the inline detail view already shows
        (`mcp_inspector.py`'s `#mcp-inspector-tool-phase-note` `Static`)
        when a tool IS selected but isn't executable -- "select a tool"
        would be actively wrong there.

        F-055: the panel is opened FIRST and the mode switch only happens
        on success -- the old switch-first order force-landed the user in
        Tools mode with a "Select a tool first." toast on top when nothing
        was selected (a mode hijack for a key the footer advertised in
        every mode). With no tool selected the active mode now stays put
        and the hint says where the working key lives. (`set_mode("tools")`
        is a no-op once already there -- no mode change means
        `_clear_tool_view()` never fires, see its own docstring -- and a
        non-None `_current_tool` only exists in Tools mode anyway, since
        every mode change clears the tool view.)
        """
        inspector = self.query_one(MCPInspector)
        status = await inspector.open_test_panel()
        if status == "no_tool":
            self.app.notify("Select a tool in Tools mode first.", severity="warning")
            return
        if status == "not_executable":
            tool = inspector.current_tool
            message = "Server-source tools are display-only."
            if tool is not None and _is_raw_shell_tool(tool.server_key, tool.name):
                message = (
                    "Raw shell is policy-only here; run commands from Console "
                    "under its separate approval flow."
                )
            elif tool is not None and tool.source != "server":
                message = "Tool testing is unavailable from this policy view."
            self.app.notify(
                message,
                severity="information",
            )
            return
        self.set_mode("tools")

    def on_mcp_inspector_tool_test_preview_requested(
        self, event: MCPInspector.ToolTestPreviewRequested
    ) -> None:
        """Prepare a service-owned preview off the UI loop."""
        event.stop()
        tool = self._tool_for(event.server_key, event.tool_name)
        inspector = self.query_one(MCPInspector)
        inspector.show_test_preparing()
        self._tool_test_generation += 1
        generation = self._tool_test_generation
        if tool is None:
            inspector.show_test_unavailable("The selected tool is no longer available.")
            return
        self.run_worker(
            self._prepare_tool_test_preview(tool, generation),
            name="mcp-tool-test-preview",
            group="mcp-tool-test-preview",
            exclusive=True,
        )

    async def _prepare_tool_test_preview(self, tool: HubTool, generation: int) -> None:
        service = self._service()
        required = (
            "prepare_hub_test",
            "execute_prepared_hub_test",
            "revoke_hub_test_preview",
            "hub_test_active",
        )
        if service is None or any(
            not callable(getattr(service, name, None)) for name in required
        ):
            self._render_test_unavailable_if_current(
                tool,
                generation,
                "Prepared tool testing is not supported by this service.",
            )
            return
        try:
            was_active = False
            while service.hub_test_active(tool.server_key, tool.name):
                if not self._test_panel_is_current(tool, generation):
                    return
                self.query_one(MCPInspector).show_test_active(True)
                was_active = True
                await asyncio.sleep(_TOOL_TEST_ACTIVE_POLL_SECONDS)
            if not self._test_panel_is_current(tool, generation):
                return
            if was_active:
                self.query_one(MCPInspector).show_test_preparing()
            preview = await self._mint_test_preview(service, tool)
            if not isinstance(preview, ToolTestAdmissionPreview):
                raise TypeError("The service returned an invalid test preview.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._render_test_unavailable_if_current(
                tool, generation, _safe_exception_text(exc)
            )
            return
        if not self._test_panel_is_current(tool, generation):
            await self._revoke_test_nonce(preview.nonce)
            return
        self._tool_test_preview_nonce = preview.nonce
        self.query_one(MCPInspector).show_test_preview(preview)

    async def _mint_test_preview(
        self, service: Any, tool: HubTool
    ) -> ToolTestAdmissionPreview:
        """Mint off-loop and reclaim a nonce even if its owner is cancelled."""
        mint_task = asyncio.create_task(
            asyncio.to_thread(service.prepare_hub_test, tool),
            name=f"mcp-tool-test-preview-mint:{tool.tool_id}",
        )
        try:
            return await asyncio.shield(mint_task)
        except asyncio.CancelledError:
            mint_task.add_done_callback(
                lambda task: self._reclaim_abandoned_test_preview(task, service=service)
            )
            raise

    def _reclaim_abandoned_test_preview(
        self, mint_task: asyncio.Task[Any], *, service: Any
    ) -> None:
        """Transfer a cancelled worker's mint to cancellation-proof cleanup."""
        try:
            preview = mint_task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug(
                "Cancelled MCP tool-test preview mint ended with {}",
                type(exc).__name__,
            )
            return
        if not isinstance(preview, ToolTestAdmissionPreview):
            return
        cleanup = asyncio.create_task(
            self._revoke_test_nonce(preview.nonce, service=service),
            name=f"mcp-tool-test-preview-reclaim:{preview.nonce}",
        )
        self._tool_test_reclaim_tasks.add(cleanup)

        def _observe_cleanup(task: asyncio.Task[None]) -> None:
            self._tool_test_reclaim_tasks.discard(task)
            try:
                task.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(
                    "Cancelled MCP tool-test preview cleanup ended with {}",
                    type(exc).__name__,
                )

        cleanup.add_done_callback(_observe_cleanup)

    def _test_panel_is_current(self, tool: HubTool, generation: int) -> bool:
        if generation != self._tool_test_generation or not self.is_attached:
            return False
        try:
            inspector = self.query_one(MCPInspector)
        except Exception:
            return False
        current = inspector.current_tool
        return (
            current is not None
            and current.server_key == tool.server_key
            and current.name == tool.name
            and bool(inspector.query("#mcp-inspector-test-panel"))
        )

    def _render_test_unavailable_if_current(
        self, tool: HubTool, generation: int, reason: str
    ) -> None:
        if self._test_panel_is_current(tool, generation):
            self.query_one(MCPInspector).show_test_unavailable(reason)

    def on_mcp_inspector_tool_test_preview_revocation_requested(
        self, event: MCPInspector.ToolTestPreviewRevocationRequested
    ) -> None:
        """Revoke a nonce leaving the visible panel, best effort."""
        event.stop()
        self._tool_test_generation += 1
        if event.preview_nonce == self._tool_test_preview_nonce:
            self._tool_test_preview_nonce = None
        self.run_worker(
            self._revoke_test_nonce(event.preview_nonce),
            name="mcp-tool-test-preview-revoke",
            group="mcp-tool-test-preview-revoke",
            exclusive=False,
        )

    async def _revoke_test_nonce(
        self, nonce: str | None, *, service: Any | None = None
    ) -> None:
        if not nonce:
            return
        service = service or self._service()
        revoke = getattr(service, "revoke_hub_test_preview", None)
        if not callable(revoke):
            return
        try:
            await asyncio.to_thread(revoke, nonce)
        except Exception as exc:
            logger.debug("MCP tool-test preview revoke failed: {}", type(exc).__name__)

    def on_mcp_inspector_tool_test_requested(
        self, event: MCPInspector.ToolTestRequested
    ) -> None:
        """Dispatch one immutable preview intent through the service only.

        Duplicate admission is deliberately owned by the service registry; this
        client may deliver concurrent clicks but never authorizes or falls back.
        """
        event.stop()
        tool = self._tool_for(event.server_key, event.tool_name)
        if tool is None:
            self.query_one(MCPInspector).show_test_unavailable(
                "The selected tool is no longer available."
            )
            return
        generation = self._tool_test_generation
        self.run_worker(
            self._run_prepared_tool_test(
                tool,
                event.preview_nonce,
                event.intent,
                dict(event.arguments),
                generation,
            ),
            name="mcp-tool-test-execute",
            group="mcp-tool-test-execute",
            exclusive=False,
        )
        return

    async def _run_prepared_tool_test(
        self,
        tool: HubTool,
        nonce: str,
        intent: str,
        arguments: dict[str, Any],
        generation: int,
    ) -> None:
        """Execute one preview-bound click and render only to its live panel."""
        service = self._service()
        execute = getattr(service, "execute_prepared_hub_test", None)
        if not callable(execute):
            self._render_test_unavailable_if_current(
                tool,
                generation,
                "Prepared tool testing is not supported by this service.",
            )
            return
        if self._test_panel_is_current(tool, generation):
            inspector = self.query_one(MCPInspector)
            inspector.clear_test_preview()
            inspector.show_test_active(True)
        if nonce == self._tool_test_preview_nonce:
            self._tool_test_preview_nonce = None
        try:
            outcome = await execute(nonce, intent, arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Canonical-argument validation happens before the service consumes
            # the preview. Revocation is idempotent when a later failure did
            # consume it, and prevents an early validation error from leaving
            # a live bearer behind while this panel mints its replacement.
            await self._revoke_test_nonce(nonce, service=service)
            if self._test_panel_is_current(tool, generation):
                self._show_tool_test_result(
                    server_key=tool.server_key,
                    tool_name=tool.name,
                    ok=False,
                    text=_safe_exception_text(exc),
                    duration_ms=0,
                    blocked=_is_permission_refusal(exc),
                    show_permission_jump=False,
                )
                inspector = self.query_one(MCPInspector)
                inspector.show_test_preparing()
                await self._prepare_tool_test_preview(tool, generation)
            await self._refresh_test_audit()
            return
        if not self._test_panel_is_current(tool, generation):
            refreshed = getattr(outcome, "refreshed_preview", None)
            if isinstance(refreshed, ToolTestAdmissionPreview):
                await self._revoke_test_nonce(refreshed.nonce)
            await self._refresh_test_audit()
            return

        inspector = self.query_one(MCPInspector)
        if isinstance(outcome, ToolTestAdmissionStale):
            reason = self._prepared_test_reason(outcome.reason)
            inspector.show_tool_result(
                server_key=tool.server_key,
                tool_name=tool.name,
                ok=False,
                text=reason,
                duration_ms=0,
                admission_changed=True,
            )
            if outcome.refreshed_preview is not None:
                self._tool_test_preview_nonce = outcome.refreshed_preview.nonce
                inspector.show_test_preview(outcome.refreshed_preview)
            else:
                inspector.show_test_preparing()
                await self._prepare_tool_test_preview(tool, generation)
            await self._refresh_test_audit()
            return

        if isinstance(outcome, ToolTestAdmissionBlocked):
            reason = self._prepared_test_reason(outcome.reason)
            inspector.show_tool_result(
                server_key=tool.server_key,
                tool_name=tool.name,
                ok=False,
                text=reason,
                duration_ms=0,
                blocked=True,
            )
            if outcome.refreshed_preview is not None:
                self._tool_test_preview_nonce = outcome.refreshed_preview.nonce
                inspector.show_test_preview(outcome.refreshed_preview)
            else:
                inspector.show_test_preparing()
                await self._prepare_tool_test_preview(tool, generation)
            await self._refresh_test_audit()
            return

        if isinstance(outcome, LocalHubExecutionOutcome):
            result = outcome.result
            text = result.content if result.ok else result.error
            self._show_tool_test_result(
                server_key=tool.server_key,
                tool_name=tool.name,
                ok=result.ok,
                text=text,
                duration_ms=outcome.duration_ms,
                blocked=outcome.status == "blocked",
                decision_note=(
                    "Approved for this invocation only; permission was not changed."
                    if outcome.approval_consumed
                    else "Ran from the prepared Allow preview."
                ),
            )
        elif isinstance(outcome, Mapping):
            try:
                redacted = redact_mapping(outcome)
                raw = json.dumps(redacted, indent=2, default=str)
            except Exception as exc:
                self._show_tool_test_result(
                    server_key=tool.server_key,
                    tool_name=tool.name,
                    ok=False,
                    text=_safe_exception_text(exc),
                    duration_ms=0,
                )
            else:
                self._show_tool_test_result(
                    server_key=tool.server_key,
                    tool_name=tool.name,
                    ok=True,
                    result=redacted.get("result"),
                    source=redacted.get("source"),
                    raw=raw,
                    duration_ms=0,
                )
        else:
            self._show_tool_test_result(
                server_key=tool.server_key,
                tool_name=tool.name,
                ok=False,
                text="The service returned an unsupported tool-test result.",
                duration_ms=0,
            )
        if self._test_panel_is_current(tool, generation):
            inspector.show_test_preparing()
            await self._prepare_tool_test_preview(tool, generation)
        await self._refresh_test_audit()

    async def _refresh_test_audit(self) -> None:
        """Refresh the persistent audit canvas without touching stale inspectors."""
        try:
            await self._sync_audit_log_entries()
        except Exception as exc:
            message = _safe_diagnostic_message(
                "MCP audit entries resync after tool test failed", exc
            )
            logger.warning("{}", message)

    @staticmethod
    def _prepared_test_reason(reason: str) -> str:
        return {
            "permission_denied": "Blocked by Permissions. Change this tool from Off to retry.",
            "permission_unresolved": _TOOL_TEST_BLOCKED_UNKNOWN_TEXT,
            "intent_mismatch": "Permission changed before the run. Review the refreshed preview.",
            "gate_changed": "Permission changed before the run. Review the refreshed preview.",
            "preview_unavailable": "The preview expired or was already used. A fresh preview is required.",
            "identity_changed": "The tool or workspace changed. Reopen the panel and retry.",
            "definition_changed": "The tool definition changed. Review the refreshed preview.",
            "already_active": "A test for this tool is already active. Wait for it to finish.",
        }.get(
            str(reason),
            "The prepared run was not admitted. Review the refreshed preview.",
        )

    def _show_tool_test_result(
        self,
        *,
        server_key: str,
        tool_name: str,
        ok: bool,
        duration_ms: int,
        text: str | None = None,
        result: object = None,
        source: str | None = None,
        raw: str | None = None,
        decision_note: str | None = None,
        blocked: bool = False,
        show_permission_jump: bool = True,
    ) -> None:
        try:
            self.query_one(MCPInspector).show_tool_result(
                server_key=server_key,
                tool_name=tool_name,
                ok=ok,
                duration_ms=duration_ms,
                text=text,
                result=result,
                source=source,
                raw=raw,
                decision_note=decision_note,
                blocked=blocked,
                show_permission_jump=show_permission_jump,
            )
        except Exception as exc:
            # Task 3 (PR-T3): the run genuinely completed -- only the
            # RENDER failed (a malformed envelope, a missing widget, ...) --
            # so a log line alone left the user with literally nothing on
            # screen and no reason to suspect the run even happened. A
            # toast closes that gap; the log line stays for diagnosis.
            safe_error = _safe_exception_text(exc)
            logger.warning(
                "{}",
                _safe_diagnostic_message("MCP tool test result render failed", exc),
            )
            self.app.notify(
                _toast(
                    _safe_tool_test_text(
                        f"{tool_name} finished running, but its result couldn't be "
                        f"shown: {safe_error}"
                    )
                ),
                severity="error",
            )

    async def open_add_server_form(self) -> None:
        """Open the Add-server form/panel from outside the overview button.

        T13: entry point for the `a` keybinding, which -- unlike a
        `Button.Pressed` on the overview's own (already gate-disabled)
        Add-server button -- can fire while server-source mutations are
        gated off. Reachable-while-gated means silently no-opping (the
        button-press behavior) would leave the user with no explanation, so
        this notifies with the button's own gate copy instead.
        """
        await self._open_add_server(notify_if_gated=True)

    async def _open_add_server(self, *, notify_if_gated: bool) -> None:
        canvas = self.query_one(MCPServersMode)
        if self._source == "server":
            # T9: mirrors `MCPServersMode._update_add_server_button()`'s gate
            # precedence -- scope gate first, then the no-active-target gate.
            # A real Button.Pressed can't reach here while either fails (the
            # button is disabled), but a defensive check costs nothing.
            if not self._server_mutations_available:
                if notify_if_gated:
                    self._notify_add_server_gated(canvas)
                return
            if self._active_service_target_id() is None:
                if notify_if_gated:
                    self._notify_add_server_gated(canvas)
                return
            await canvas.show_server_mutations(None, [])
        else:
            await canvas.show_form(None)

    def _notify_add_server_gated(self, canvas: MCPServersMode) -> None:
        """Surface the overview Add-server button's own gate tooltip as a notification.

        Reuses whatever `MCPServersMode._update_add_server_button()` already
        computed rather than duplicating the gate copy, so the notification
        and the button's own explanation can never drift apart.
        """
        try:
            button = canvas.query_one("#mcp-add-server")
        except Exception:
            button = None
        message = (
            str(button.tooltip)
            if button is not None and button.tooltip
            else ("Adding a server is unavailable right now.")
        )
        self.app.notify(message, severity="warning")

    async def on_mcp_servers_mode_import_servers_requested(
        self, event: MCPServersMode.ImportServersRequested
    ) -> None:
        event.stop()
        # T8: existing catalog ids drive the panel's overwrite warnings --
        # `_catalog_records` is kept in sync with `_snapshots` by
        # `_collect_snapshots()` (Task 6).
        await self.query_one(MCPServersMode).show_import(set(self._catalog_records))

    async def on_mcp_servers_mode_disconnect_requested(
        self, event: MCPServersMode.DisconnectRequested
    ) -> None:
        """Route the detail toolbar's Disconnect button through the same
        `_start_lifecycle()` dispatch T5 wired for connect/test/refresh --
        disconnect is a detail-view-only action, so it never comes through
        `HubActionRequested`/`_HUB_ACTION_TO_LIFECYCLE_VERB` like those three.
        """
        event.stop()
        if event.server_key and event.server_key.startswith("local:"):
            profile_id = event.server_key.split(":", 1)[1]
            self._start_lifecycle(event.server_key, profile_id, "disconnect")

    def on_mcp_servers_mode_builtin_flag_changed(
        self, event: MCPServersMode.BuiltinFlagChanged
    ) -> None:
        """Dispatch a built-in server enable/expose toggle in the background.

        Synchronous (not `async def`), mirroring
        `on_mcp_servers_mode_delete_confirmed()`/`on_mcp_profile_form_
        submit_requested()`: the handler returns immediately so the message
        pump stays responsive while the config write + catalog reload run.
        No in-flight guard (unlike those two): each Checkbox already
        displays its own last-known value between toggles, so a rapid
        second toggle -- of the same or a different flag -- simply cancels
        the still-running worker via `exclusive=True` (safe: `Checkbox.
        Changed` is idempotent config state, not an append-only mutation)
        and starts fresh from the latest event.
        """
        event.stop()
        self.run_worker(
            self._save_builtin_flag(event.key, event.value),
            group="mcp-builtin-flag",
            exclusive=True,
        )

    async def _save_builtin_flag(self, key: str, value: bool) -> None:
        """Persist one `[mcp]` enable/expose flag, then reload the catalog.

        The write itself is the blocking part (TOML read-modify-write to
        disk, `save_setting_to_cli_config()` in config.py) -- offloaded via
        `asyncio.to_thread` rather than Textual's `@work(thread=True)`
        decorator (the fire-and-forget precedent at
        library_screen.py:5534's `_save_library_rail_preferences()`)
        because this call, unlike that one, MUST follow the write with
        async work that touches live widgets (`_collect_snapshots()` +
        `_sync_children()`, so the built-in row's readiness badge and this
        detail pane's own checkboxes reflect the change). Keeping both
        steps in one coroutine dispatched via `run_worker(coroutine, ...)`
        mirrors this file's own `_load_import_file()` (`asyncio.to_thread`
        for a blocking `Path.read_text` followed by an in-coroutine UI
        update) instead of adding a `call_from_thread` marshaling hop back
        onto the event loop that a sync `@work(thread=True)` method would
        need for the same follow-up.
        """
        try:
            saved = await asyncio.to_thread(
                save_setting_to_cli_config, "mcp", key, value
            )
        except Exception as exc:
            logger.warning(
                "{}",
                _safe_diagnostic_message("MCP built-in flag save failed", exc),
            )
            self.app.notify(_toast(f"Failed to save {key}: {exc}"), severity="error")
            return
        if not saved:
            self.app.notify(f"Failed to save {key}.", severity="error")
            return
        self._snapshots = await self._collect_snapshots()
        await self._sync_children()

    def on_mcp_servers_mode_tool_gate_changed(
        self, event: MCPServersMode.ToolGateChanged
    ) -> None:
        """Dispatch a `[tools]`/`[console]` gate toggle in the background.

        task-3240 sibling of `on_mcp_servers_mode_builtin_flag_changed()` --
        identical shape (sync handler, `exclusive=True` group so a rapid
        second toggle simply cancels and restarts from the latest event),
        just routed to `_save_tool_gate()` instead of `_save_builtin_flag()`
        so the write targets the event's own `section`, not a hardcoded
        `"mcp"`.
        """
        event.stop()
        self.run_worker(
            self._save_tool_gate(event.section, event.key, event.value),
            group="mcp-tool-gate",
            exclusive=True,
        )

    async def _save_tool_gate(self, section: str, key: str, value: bool) -> None:
        """Persist one `[tools]`/`[console]` registration gate, then reload.

        Mirrors `_save_builtin_flag()` exactly (same offload-then-resync
        shape) except `section` is a parameter rather than hardcoded --
        task-3240's gates span both `[tools]` (the `_GATEABLE_BUILTINS` rows
        plus `web_deep_search`) and `[console]` (the local group's master
        switch, `local_tools_enabled`). The resync's `_show_selected_detail()`
        call rebuilds the gate checkboxes fresh from `all_tool_gates()`
        (via `MCPServersMode._rebuild_tool_gate_checkboxes()`), so a failed
        write shows the truth rather than an optimistic local flip.
        """
        try:
            saved = await asyncio.to_thread(
                save_setting_to_cli_config, section, key, value
            )
        except Exception as exc:
            logger.warning(
                "{}", _safe_diagnostic_message("MCP tool gate save failed", exc)
            )
            self.app.notify(_toast(f"Failed to save {key}: {exc}"), severity="error")
            return
        if not saved:
            self.app.notify(f"Failed to save {key}.", severity="error")
            return
        self._snapshots = await self._collect_snapshots()
        await self._sync_children()

    def on_mcp_servers_mode_delete_confirmed(
        self, event: MCPServersMode.DeleteConfirmed
    ) -> None:
        """Dispatch a profile delete in the background.

        Synchronous (not `async def`), mirroring
        `on_mcp_profile_form_submit_requested()`: the handler itself must
        return immediately so Textual's message pump stays responsive while
        the delete runs -- the actual `await
        service.delete_local_profile(...)` happens inside the worker
        coroutine below. `_profile_delete_in_flight` (set here,
        synchronously, before dispatch; cleared in the worker's `finally`)
        is what makes a double confirm safe.
        """
        event.stop()
        if not event.server_key or not event.server_key.startswith("local:"):
            return
        profile_id = event.server_key.split(":", 1)[1]
        if self._profile_delete_in_flight:
            self.app.notify(
                _toast(f"{profile_id}: delete already running."), severity="warning"
            )
            return
        self._profile_delete_in_flight = True
        self.run_worker(
            self._delete_local_profile(event.server_key, profile_id),
            group="mcp-profile-delete",
            exclusive=True,
        )

    async def _delete_local_profile(self, server_key: str, profile_id: str) -> None:
        try:
            service = self._service()
            if service is None:
                return
            try:
                await service.delete_local_profile(profile_id)
            except Exception as exc:
                logger.warning(
                    "{}", _safe_diagnostic_message("MCP profile delete failed", exc)
                )
                self.app.notify(_toast(f"Delete failed: {exc}"), severity="error")
                return
            self.app.notify(_toast(f"Deleted {profile_id}."))
            if self._selected_server_key == server_key:
                self._selected_server_key = None
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()
        finally:
            self._profile_delete_in_flight = False

    def on_mcp_profile_form_submit_requested(
        self, event: MCPProfileForm.SubmitRequested
    ) -> None:
        """Dispatch a profile save in the background.

        Synchronous (not `async def`), mirroring `_start_lifecycle()`: the
        handler itself must return immediately so Textual's message pump
        stays responsive while the save runs -- the actual `await
        service.save_local_profile(...)` happens inside the worker coroutine
        below. The `_profile_save_in_flight` guard (set here, synchronously,
        before dispatch; cleared in the worker's `finally`) is what makes a
        double Save safe: without it, `exclusive=True` alone let a second
        submit CANCEL the in-flight save mid-write and start over.
        """
        event.stop()
        if self._profile_save_in_flight:
            self.app.notify("Save already running.", severity="warning")
            return
        self._profile_save_in_flight = True
        self.run_worker(
            self._save_local_profile(dict(event.payload), warning=event.warning),
            group="mcp-profile-save",
            exclusive=True,
        )

    def _form_or_none(self) -> MCPProfileForm | None:
        try:
            return self.query_one(MCPProfileForm)
        except Exception:
            return None

    async def _save_local_profile(
        self, payload: dict[str, Any], warning: str | None = None
    ) -> None:
        """Run one profile save; on success, also re-surface the form's args
        secret-lint `warning` as a toast (I4 follow-up). The in-form
        `#mcp-form-args-warning` Static only survives FAILED saves -- the
        success path's `hide_form()` below unmounts the whole form
        sub-second after the warning rendered, so without this toast the
        user would never see it on exactly the path where the secret
        actually got persisted into a profile's args.
        """
        try:
            service = self._service()
            if service is None:
                return
            try:
                await service.save_local_profile(payload)
            except ValueError as exc:
                # Store-validation copy is user-ready. If the form is gone
                # (e.g. cancelled while the save was in flight), the failure
                # must still surface -- never vanish silently.
                form = self._form_or_none()
                if form is not None:
                    form.show_error(str(exc))
                else:
                    self.app.notify(_toast(str(exc)), severity="error")
                return
            except Exception as exc:
                logger.warning(
                    "{}", _safe_diagnostic_message("MCP profile save failed", exc)
                )
                # Route through show_error when possible: it also re-enables
                # the form's Save button (disabled at submit) for a retry.
                form = self._form_or_none()
                if form is not None:
                    form.show_error(f"Save failed: {exc}")
                else:
                    self.app.notify(_toast(f"Save failed: {exc}"), severity="error")
                return
            canvas = self.query_one(MCPServersMode)
            await canvas.hide_form()
            self.app.notify(_toast(f"Saved {payload.get('profile_id')}."))
            if warning:
                self.app.notify(warning, severity="warning")
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()
        finally:
            self._profile_save_in_flight = False

    async def on_mcp_profile_form_cancelled(
        self, event: MCPProfileForm.Cancelled
    ) -> None:
        event.stop()
        await self.query_one(MCPServersMode).hide_form()

    # -- T9: server-source external-server + credential-slot mutations --------

    def _mutations_panel_or_none(self) -> MCPServerMutationsPanel | None:
        try:
            return self.query_one(MCPServerMutationsPanel)
        except Exception:
            return None

    def on_mcp_server_mutations_submit_requested(
        self, event: MCPServerMutationsPanel.SubmitRequested
    ) -> None:
        """Dispatch one `run_action(action, payload)` call in the background.

        Synchronous (not `async def`), mirroring
        `on_mcp_profile_form_submit_requested()`: `_server_mutation_in_flight`
        is set here, before dispatch, so a second Save/Add-slot/Set-secret
        press arriving in the same pump window is reliably swallowed with a
        warning toast instead of racing the in-flight call.
        """
        event.stop()
        if self._server_mutation_in_flight:
            self.app.notify("Save already running.", severity="warning")
            return
        self._server_mutation_in_flight = True
        self.run_worker(
            self._run_server_mutation(event.action, dict(event.payload)),
            group="mcp-server-mutation",
            exclusive=True,
        )

    async def _run_server_mutation(self, action: str, payload: dict[str, Any]) -> None:
        try:
            service = self._service()
            if service is None:
                return
            try:
                await service.run_action(action, payload)
            except Exception as exc:
                logger.warning(
                    "{}",
                    _safe_diagnostic_message(
                        f"MCP server mutation failed ({action})", exc
                    ),
                )
                panel = self._mutations_panel_or_none()
                if panel is not None:
                    panel.show_error(str(exc))
                else:
                    self.app.notify(_toast(f"{action} failed: {exc}"), severity="error")
                return
            self.app.notify(
                _SERVER_MUTATION_MESSAGES.get(
                    action,
                    f"{action.rsplit('.', 1)[-1].replace('_', ' ').title()} saved.",
                )
            )
            if action == "external_server.create":
                # Drill straight into the record just created -- credential
                # setup is the natural next step, and `_sync_children()`
                # below will fetch its slots and show the mutation panel in
                # edit mode (T9's `_show_selected_detail()`). Review fix:
                # derived from the SERVICE's active target, because create
                # only ever runs from the overview where the local UI
                # selection is None -- `_selected_target_id()` alone made
                # this branch dead.
                target_id = self._active_service_target_id()
                server_id = payload.get("server_id")
                if target_id and server_id:
                    self._selected_server_key = f"server:{target_id}/{server_id}"
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()
        finally:
            self._server_mutation_in_flight = False

    async def on_mcp_server_mutations_cancelled(
        self, event: MCPServerMutationsPanel.Cancelled
    ) -> None:
        """Close the mutations panel AND clear the selection that opened it.

        I2 fix: `show_server_mutations()` never updates `_detail_snapshot`
        (it hosts an edit-mode panel for an external-server record instead
        of routing through `show_detail()`), so `hide_form()`'s own
        `_detail_snapshot is None` check can't be trusted to land back on
        the overview -- and, worse, `_selected_server_key` was left pointing
        at the external record, so the very next `_sync_children()` (a
        background lifecycle completion, the `r` keybinding, a
        runtime-backend refresh) would call `_show_selected_detail()` again
        and re-open this same panel out of nowhere.
        Routes through the exact same path `ServerRowSelected(None)` uses
        (`_select_server_key(None)`): clears `_selected_server_key`, then a
        full resync lands on the overview with the table cursor restored.
        `hide_form()` runs first purely to close the panel/unmount its
        widgets; the resync right after is what settles the correct
        final container visibility regardless of `hide_form()`'s own
        (possibly stale) guess.
        """
        event.stop()
        await self.query_one(MCPServersMode).hide_form()
        await self._select_server_key(None)

    # -- T8: mcpServers import (paste or file) ---------------------------------

    def _import_panel_or_none(self) -> MCPImportPanel | None:
        try:
            return self.query_one(MCPImportPanel)
        except Exception:
            return None

    async def on_mcp_import_panel_file_requested(
        self, event: MCPImportPanel.FileRequested
    ) -> None:
        event.stop()
        from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen, Filters

        def on_file_selected(file_path: Any) -> None:
            if file_path:
                self.run_worker(
                    self._load_import_file(str(file_path)),
                    group="mcp-import-file",
                    exclusive=True,
                )

        await self.app.push_screen(
            EnhancedFileOpen(
                location=".",
                title="Select MCP config JSON",
                filters=Filters(("JSON", lambda p: p.suffix.lower() == ".json")),
                context="mcp_import",
            ),
            callback=on_file_selected,
        )

    async def _load_import_file(self, file_path: str) -> None:
        """Read a user-picked mcpServers config file into the import panel.

        F1 fix: `EnhancedFileOpen` lets the user browse anywhere on disk, so
        the picked path is validated the same way the Console attachments
        flow validates a picked attachment (`attachment_core.load_processed_
        file()`'s `is_safe_path(file_path, home_dir)` + size-cap pattern)
        before this ever touches the filesystem.
        """
        if not is_safe_path(file_path, _mcp_import_home()):
            self.app.notify("Import file path failed validation.", severity="error")
            return
        try:
            file_size = await asyncio.to_thread(os.path.getsize, file_path)
        except OSError as exc:
            self.app.notify(
                _toast(f"Could not read {file_path}: {exc}"), severity="error"
            )
            return
        if file_size > MAX_MCP_IMPORT_FILE_BYTES:
            self.app.notify(
                _toast(
                    f"Import file too large: {file_size / 1024 / 1024:.1f}MB "
                    f"(max {MAX_MCP_IMPORT_FILE_BYTES / 1024 / 1024:.0f}MB)."
                ),
                severity="error",
            )
            return
        try:
            text = await asyncio.to_thread(Path(file_path).read_text, encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # UnicodeDecodeError (a ValueError subclass, not an OSError) is
            # raised by `read_text` for any non-UTF-8 file -- e.g. a
            # Claude-Desktop config saved with a BOM/legacy encoding. Left
            # uncaught, it escapes this worker and, with Textual's default
            # `exit_on_error=True`, takes down the whole app (C1).
            self.app.notify(
                _toast(f"Could not read {file_path}: {exc}"), severity="error"
            )
            return
        panel = self._import_panel_or_none()
        if panel is not None:
            panel.set_file_text(text)

    async def on_mcp_import_panel_cancelled(
        self, event: MCPImportPanel.Cancelled
    ) -> None:
        event.stop()
        await self.query_one(MCPServersMode).hide_form()

    def on_mcp_import_panel_import_requested(
        self, event: MCPImportPanel.ImportRequested
    ) -> None:
        """Dispatch a batch of candidate saves in the background.

        Synchronous (not `async def`), mirroring
        `on_mcp_profile_form_submit_requested()`: `_profile_import_in_flight`
        is set here, before dispatch, so a second Import press arriving in
        the same pump window is reliably swallowed with a warning toast
        instead of racing the in-flight batch.
        """
        event.stop()
        if self._profile_import_in_flight:
            self.app.notify("Import already running.", severity="warning")
            return
        self._profile_import_in_flight = True
        self.run_worker(
            self._apply_import(list(event.candidates)),
            group="mcp-profile-import",
            exclusive=True,
        )

    async def _apply_import(self, candidates: list[ImportCandidate]) -> None:
        try:
            service = self._service()
            if service is None:
                return
            succeeded: list[str] = []
            failed: list[tuple[str, str]] = []
            for candidate in candidates:
                try:
                    await service.save_local_profile(candidate.to_payload())
                except Exception as exc:
                    logger.warning(
                        "{}", _safe_diagnostic_message("MCP import failed", exc)
                    )
                    failed.append((candidate.profile_id, _safe_exception_text(exc)))
                else:
                    succeeded.append(candidate.profile_id)
            self.app.notify(
                _toast(_import_summary(succeeded, failed)),
                severity=_import_severity(succeeded, failed),
            )
            canvas = self.query_one(MCPServersMode)
            await canvas.hide_form()
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()
        finally:
            self._profile_import_in_flight = False

    def on_mcp_inspector_cancel_requested(
        self, event: MCPInspector.CancelRequested
    ) -> None:
        """Cancel an in-flight lifecycle worker.

        Synchronous (not `async def`): `Worker.cancel()` is itself
        synchronous, and the caller (Textual's message pump, or a test
        calling this directly) doesn't need to await anything here -- the
        display resync is fired off as its own worker below instead of being
        awaited inline.
        """
        event.stop()
        worker = self._in_flight.pop(event.server_key, None)
        self._in_flight_action.pop(event.server_key, None)
        if worker is None:
            # Stale cancel: the operation already finished and popped itself
            # (its own completion toast + resync have run). Toasting
            # "Cancelled." here would falsely claim a completed operation
            # was stopped -- silent no-op instead.
            return
        worker.cancel()
        self.app.notify("Cancelled.")
        self.run_worker(
            self._sync_children(), group="mcp-lifecycle-sync", exclusive=True
        )

    # -- lifecycle actions (T5: connect/test/refresh/disconnect) --------------

    def _start_lifecycle(self, server_key: str, profile_id: str, action: str) -> None:
        """Dispatch a local-profile lifecycle action in the background.

        Synchronous (not `async def`): must register `self._in_flight`
        synchronously, before returning to the caller, so a `CancelRequested`
        arriving right after this call (or a second click of the same
        action) reliably observes the worker that was just started -- if
        this were `async def` and awaited only later, the bookkeeping below
        wouldn't run until the event loop actually scheduled this coroutine,
        leaving a window where the guard/cancel logic would see stale state.
        """
        if server_key in self._in_flight:
            self.app.notify(
                _toast(f"{profile_id}: {action} already running."), severity="warning"
            )
            return
        service = self._service()
        method_name = _LIFECYCLE_METHOD_NAMES.get(action)
        method = (
            getattr(service, method_name, None)
            if service is not None and method_name
            else None
        )
        if not callable(method):
            logger.warning(
                f"MCP workbench: no lifecycle method for action={action!r} "
                f"(server_key={server_key!r})"
            )
            return
        coro = method(profile_id)
        worker = self.run_worker(
            self._lifecycle_wrapper(server_key, profile_id, action, coro),
            group="mcp-lifecycle",
            exclusive=False,
        )
        self._in_flight[server_key] = worker
        self._in_flight_action[server_key] = action
        # Render the CHECKING badge + inspector Cancel button immediately --
        # decoupled from the lifecycle worker above, which may be sitting on
        # a slow (or, in tests, gated) network/subprocess call and must not
        # block this optimistic UI update.
        self.run_worker(
            self._sync_children(), group="mcp-lifecycle-sync", exclusive=True
        )

    async def _lifecycle_wrapper(
        self, server_key: str, profile_id: str, action: str, coro: Any
    ) -> None:
        """Run one lifecycle coroutine, then always clean up and resync.

        The T2 typed methods (`connect_local_profile` etc.) already record
        their own attempt state and raise a user-ready message on failure --
        this must not duplicate that recording, only surface the outcome and
        drop the in-flight marker. `except Exception` deliberately does not
        catch `asyncio.CancelledError` (a `BaseException` since Python 3.8):
        a cancelled worker skips straight to `finally`, which is exactly the
        cleanup `on_mcp_inspector_cancel_requested()` needs and which the
        cancel handler's own notify()/resync above already covers, so no
        redundant "cancelled" notification is sent from here.
        """
        try:
            result = await coro
        except Exception as exc:
            self.app.notify(
                _toast(f"{profile_id}: {action} failed — {exc}"), severity="error"
            )
        else:
            verb = _LIFECYCLE_PAST_TENSE.get(action, action)
            tool_count = self._lifecycle_tool_count(result)
            if tool_count is None:
                self.app.notify(_toast(f"{profile_id}: {verb}."))
            else:
                noun = "tool" if tool_count == 1 else "tools"
                self.app.notify(_toast(f"{profile_id}: {verb} — {tool_count} {noun}."))
        finally:
            self._in_flight.pop(server_key, None)
            self._in_flight_action.pop(server_key, None)
            self._snapshots = await self._collect_snapshots()
            await self._sync_children()

    @staticmethod
    def _lifecycle_tool_count(result: Any) -> int | None:
        """Best-effort tool count from a lifecycle result for the success notice.

        `connect_local_profile`/`refresh_local_profile` return a dict with a
        `tools` list; `test_local_profile` returns a dict with a `tools`
        *count* (int); `disconnect_local_profile` returns a bare bool. Any
        other shape just omits the tool count from the notification.
        """
        if not isinstance(result, Mapping):
            return None
        tools = result.get("tools")
        if isinstance(tools, list):
            return len(tools)
        if isinstance(tools, int) and not isinstance(tools, bool):
            return tools
        return None
