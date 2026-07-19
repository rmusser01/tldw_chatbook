"""MCP Hub inspector: readiness explanation, actions, Advanced escape hatch."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Collapsible, Label, Select, Static, TextArea

from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.MCP.readiness import (
    REASON_LABELS,
    STATE_CSS_CLASSES,
    HubAction,
    ReadinessSnapshot,
    ReadinessState,
)
from tldw_chatbook.MCP.redaction import redact_mapping
from tldw_chatbook.UI.MCP_Modules.mcp_schema_form import MCPSchemaForm, parse_schema
from tldw_chatbook.UI.MCP_Modules.unified_mcp_sections import render_unified_mcp_section

# Actions that have first-class UI in every source. Everything else renders
# disabled and points at the Advanced runner below (capability preserved).
_BASE_WIRED_ACTIONS = {HubAction.VIEW_DETAILS, HubAction.OPEN_TOOL_CATALOG, HubAction.OPEN_AUDIT}

# Local-profile lifecycle actions (Task 5): wired only for local-source
# snapshots, where MCPWorkbench._start_lifecycle() can actually run them
# against the typed T2 control-plane methods. Server-source servers are
# mutated on the server side (Advanced), not from this pane.
_LIFECYCLE_ACTIONS = {HubAction.CONNECT, HubAction.VALIDATE, HubAction.REFRESH_DISCOVERY}

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
}

# Disabled-button tooltip for a lifecycle action on a server-source snapshot
# (managed server-side, not from this local-lifecycle pane).
_SERVER_MANAGED_TOOLTIP = "Managed on the server — use Advanced."
# Disabled-button tooltip for every other still-unwired action.
_LATER_PHASE_TOOLTIP = "Available in a later phase — use Advanced below."

# Task 5: the Test Tool Run button's tooltip in its normal (unarmed) state,
# and once `require_confirm()` has armed it into a one-shot "Confirm run"
# control -- `MCPWorkbench` resolves deny/ask/allow via `gate_tool_test()`
# and arms this pane for "ask", but the button copy/mechanics live here.
_TEST_RUN_TOOLTIP = "Send these arguments to the tool and show the result."
_TEST_RUN_CONFIRM_TOOLTIP = "Ask is set for this tool — press again to run once."
# UX batch item 6: always shown while the Run button is armed (mounted
# above Run/Close, `#mcp-inspector-test-armed-hint`) -- distinct from the
# SPECIFIC `notice` `require_confirm()` also accepts (config_changed /
# unverifiable, `#mcp-inspector-test-arm-notice`, mounted below the
# buttons, unchanged position/contract): this one explains the Ask
# mechanic itself, present on every arm regardless of whether a specific
# reason is also shown.
_TEST_RUN_ARMED_HINT = "This tool is set to Ask — press again to run; anything else cancels."

# Task 7: permission-explanation copy (spec-verbatim, binding) -- rendered
# into `#mcp-inspector-permission` by `_render_permission_container()`,
# shared by `show_tool()`'s `effective` keyword (Tools-mode selections
# append the block below the tool detail) and the standalone
# `show_permission()` (Permissions-mode matrix tool-row selections).
_ORIGIN_SENTENCES: dict[str, str] = {
    "tool_override": "From this tool's override.",
    "server_default": "Inherited from the server default.",
    "global_default": "Inherited from the global default.",
}
# Minor 6: fallback for an origin this dict doesn't recognize (e.g.
# "gate_error", `_resolve_test_gate()`'s synthetic fail-closed origin) --
# `.get(effective.origin, "")` used to render a blank line here instead of
# ANY explanation, which reads as a broken UI rather than "we don't know
# why, but don't trust it".
_UNKNOWN_ORIGIN_SENTENCE = "Permission state could not be resolved."
_CONFIG_CHANGED_NOTICE = "Definition changed since you allowed it."
_RISK_FLOORED_NOTICE = "High-risk tool — asks even though the inherited default is Allow."
_REALLOW_TOOLTIP = "Store the new definition hash and allow again."


def _format_duration_ms(duration_ms: int) -> str:
    """Format a Test Tool run's duration for the result status line.

    Mirrors `library_ingest_state._format_elapsed()`'s granularity at the
    minute tier (integer minutes/seconds, "{m}m {s}s") but adds a finer,
    millisecond-aware tier below it -- a Test Tool run is typically
    sub-second, where a bare "0s" would say nothing useful:
      - under 1000ms: "{n}ms"
      - under 60s: "{s:.1f}s" (one decimal)
      - 60s or more: "{m}m {s}s" (integer minutes/seconds)
    """
    if duration_ms < 1000:
        return f"{duration_ms}ms"
    total_seconds = duration_ms / 1000
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    minutes, seconds = divmod(int(round(total_seconds)), 60)
    return f"{minutes}m {seconds}s"


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


class MCPInspector(Vertical):
    """Right-pane inspector: what is selected, why, what can I do."""

    DEFAULT_CSS = """
    MCPInspector {
        width: 3fr;
        min-width: 28;
        height: 100%;
        min-height: 0;
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
    /* T12: the Advanced block moved from a direct-child VerticalScroll to a
    Collapsible's body. Give the Collapsible itself the 1fr the scroll used
    to claim directly (so it still fills the remaining pane height when
    expanded) and drop back to auto when collapsed (Contents is display:
    none then -- reserving 1fr of empty space below the title bar would
    waste most of the pane). #mcp-adv-scroll keeps height:1fr for when it
    IS visible; nested inside Collapsible's own auto-height Contents this
    mostly falls back to intrinsic sizing, but VerticalScroll still scrolls
    on overflow regardless, so nothing breaks -- exact geometry polish is
    T13's job. */
    #mcp-adv-collapsible {
        height: 1fr;
        min-height: 0;
    }
    #mcp-adv-collapsible.-collapsed {
        height: auto;
    }
    #mcp-adv-scroll {
        height: 1fr;
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
        documented on Button.mcp-rail-row in MCPRail.DEFAULT_CSS and
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
        """Posted when the user presses Run in the Test Tool panel with a
        validly-collected argument dict (`MCPSchemaForm.collect_arguments()`
        raised nothing). `MCPWorkbench` owns the actual `test_hub_tool()`
        call and reports the outcome back via `show_tool_result()`.

        Carries `server_key`/`tool_name` as separate fields (not a packed
        `"server_key::tool_name"` id) -- task-233: nothing downstream of
        this message parses a `"::"`-joined string anymore."""

        def __init__(self, server_key: str, tool_name: str, arguments: dict[str, Any]) -> None:
            super().__init__()
            self.server_key = server_key
            self.tool_name = tool_name
            self.arguments = arguments

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
        # T12 review fix: the collapsed state this widget last knew about --
        # set to the constructed value in compose(), updated only by the
        # Toggled handler. Used to drop the spurious mount-time Toggled that
        # `Collapsible(collapsed=False)` posts (see the handler); default
        # True matches Collapsible's own reactive default.
        self._advanced_last_collapsed: bool = True
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
        # Task 5: True once `require_confirm()` has armed the Test Tool Run
        # button into a one-shot "Confirm run" control (the tool's gate
        # resolved to "ask" -- `MCPWorkbench` decides that, this pane only
        # renders it). Mirrors `MCPServersMode._delete_armed`: reset to
        # False by every "other interaction" per the arm-then-confirm
        # contract -- a new/cleared tool selection (`show_tool()`) and the
        # test panel's own Close button (`_close_test_tool_panel()`).
        self._test_run_armed: bool = False

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
        yield Static("Select an item to inspect.", id="mcp-inspector-state",
                     classes="ds-status-badge", markup=False)
        yield Static("", id="mcp-inspector-message", classes="ds-field-row", markup=False)
        yield Vertical(id="mcp-inspector-actions")
        # T6: tool-detail container, populated by show_tool() -- hidden
        # (display: none, see DEFAULT_CSS) until a Tools-mode row is
        # selected.
        yield Vertical(id="mcp-inspector-tool")
        # T7: permission-explanation container, populated by
        # `_render_permission_container()` (via `show_tool()`'s `effective`
        # keyword or the standalone `show_permission()`) -- hidden (display:
        # none, see DEFAULT_CSS) until a permission context is supplied.
        yield Vertical(id="mcp-inspector-permission")
        # T12: default collapsed unless the user has previously opened it --
        # per-user GLOBAL preference (Console rail section-preference
        # precedent), NOT per-server. `get_cli_setting` reads the real user
        # config in a bare test App; tests monkeypatch this module's
        # `get_cli_setting` name for determinism (see test_mcp_inspector.py).
        persisted_open = bool(get_cli_setting("mcp.hub_state", "advanced_open", False))
        self._advanced_last_collapsed = not persisted_open
        with Collapsible(
            title="Advanced (legacy control plane)",
            collapsed=not persisted_open,
            id="mcp-adv-collapsible",
        ):
            yield Static(self._advanced_object_label(), id="mcp-adv-object", markup=False)
            with VerticalScroll(id="mcp-adv-scroll"):
                yield Label("Section", classes="form-label")
                yield Select(self._sections, id="mcp-adv-section-select", allow_blank=False,
                             value="overview")
                yield Static("", id="mcp-adv-content", classes="ds-field-row", markup=False)
                yield Label("Action", classes="form-label")
                yield Select([("No actions available", Select.BLANK)], id="mcp-adv-action-select",
                             value=Select.BLANK)
                # Task 4: guidance shown only while the section above has zero
                # runnable action descriptors (see `_refresh_advanced_actions`),
                # so a user landing on e.g. Overview isn't left staring at a
                # disabled "No actions available" select with no next step.
                yield Static("", id="mcp-adv-empty-hint", classes="ds-field-row", markup=False)
                yield Label("Payload (JSON)", classes="form-label")
                yield TextArea("{}", id="mcp-adv-payload")
                yield Button("Run Action", id="mcp-adv-run", classes="console-action-primary",
                             compact=True,
                             tooltip="Run the selected legacy control-plane action with this JSON payload.")
                yield Static("", id="mcp-adv-result", classes="ds-field-row", markup=False)

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
            logger.warning(f"MCP advanced-open preference save failed: {exc}")

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
                state.update("Select an item to inspect.")
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
                    if action in (_LIFECYCLE_ACTIONS | _CONFIG_ACTIONS) and snapshot.source != "local":
                        button.tooltip = _SERVER_MANAGED_TOOLTIP
                    else:
                        button.tooltip = _LATER_PHASE_TOOLTIP
                else:
                    button.tooltip = _WIRED_ACTION_TOOLTIPS.get(action, _ACTION_LABELS[action])
                buttons.append(button)
            if buttons:
                await actions.mount_all(buttons)

    # -- T6: tool detail view + Test Tool runner ------------------------------

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
            # Task 5: a tool-selection change (a different tool, or clearing
            # the selection entirely -- e.g. a mode switch, see
            # MCPWorkbench._clear_tool_view()) is an "other interaction" per
            # the arm-then-confirm contract, so it disarms a pending Test
            # Tool confirm. The panel `remove_children()` below discards the
            # armed Run button regardless; this just keeps the flag from
            # lying about a button that no longer exists.
            self._test_run_armed = False
            container = self.query_one("#mcp-inspector-tool", Vertical)
            await container.remove_children()
            if tool is None:
                container.display = False
                await self._render_permission_container(None, None)
                return
            container.display = True
            widgets: list[Any] = [
                Static(
                    f"{tool.name} — {tool.server_label}",
                    id="mcp-inspector-tool-name", classes="ds-field-row", markup=False,
                ),
                Static(
                    tool.description, id="mcp-inspector-tool-description",
                    classes="ds-field-row", markup=False,
                ),
                Static(
                    f"Tags: {', '.join(tool.tags) if tool.tags else '—'}",
                    id="mcp-inspector-tool-tags", classes="ds-field-row", markup=False,
                ),
                Static(
                    "Parameters: form" if parse_schema(tool.input_schema) is not None
                    else "Parameters: raw JSON",
                    id="mcp-inspector-tool-schema", classes="ds-field-row", markup=False,
                ),
            ]
            if tool.stale:
                widgets.append(
                    Static(
                        "Stale — not currently connected.",
                        id="mcp-inspector-tool-stale", classes="ds-field-row", markup=False,
                    )
                )
            if tool.executable:
                widgets.append(
                    Button(
                        "Test Tool", id="mcp-inspector-test-tool",
                        classes="console-action-primary", compact=True,
                        tooltip="Run this tool with test arguments.",
                    )
                )
            else:
                widgets.append(
                    Static(
                        "Testing server-source tools isn't available yet.",
                        id="mcp-inspector-tool-phase-note",
                        classes="ds-field-row", markup=False,
                    )
                )
            await container.mount_all(widgets)
            await self._render_permission_container(tool, effective)

    async def _render_permission_container(
        self, tool: HubTool | None, effective: EffectiveToolState | None
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
        """
        container = self.query_one("#mcp-inspector-permission", Vertical)
        await container.remove_children()
        if tool is None or effective is None:
            container.display = False
            self._current_permission_tool = None
            return
        container.display = True
        self._current_permission_tool = tool
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
                id="mcp-inspector-permission-tool", classes="ds-field-row", markup=False,
            ),
            Static(
                f"Permission: {effective.ui_label}",
                id="mcp-inspector-permission-state", classes="ds-field-row", markup=False,
            ),
            Static(
                _ORIGIN_SENTENCES.get(effective.origin, _UNKNOWN_ORIGIN_SENTENCE),
                id="mcp-inspector-permission-origin", classes="ds-field-row", markup=False,
            ),
        ]
        if effective.config_changed:
            widgets.append(
                Static(
                    _CONFIG_CHANGED_NOTICE,
                    id="mcp-inspector-permission-notice", classes="ds-field-row", markup=False,
                )
            )
            widgets.append(
                Button(
                    "Re-allow", id="mcp-inspector-reallow",
                    classes="console-action-primary", compact=True,
                    tooltip=_REALLOW_TOOLTIP,
                )
            )
        elif effective.risk_floored:
            widgets.append(
                Static(
                    _RISK_FLOORED_NOTICE,
                    id="mcp-inspector-permission-notice", classes="ds-field-row", markup=False,
                )
            )
        await container.mount_all(widgets)

    async def show_permission(self, tool: HubTool, effective: EffectiveToolState) -> None:
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
        """
        async with self._refresh_lock:
            await self._render_permission_container(tool, effective)

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
            # UX batch item 6: blank until `require_confirm()` fills it in
            # (every arm, unconditionally) -- mounted ABOVE Run/Close so the
            # armed-explainer reads before the button whose behavior it's
            # explaining, unlike the specific `#mcp-inspector-test-arm-
            # notice` below, which keeps its pre-existing position.
            Static("", id="mcp-inspector-test-armed-hint", classes="ds-field-row", markup=False),
            Button(
                "Run", id="mcp-inspector-test-run",
                classes="console-action-primary", compact=True,
                tooltip=_TEST_RUN_TOOLTIP,
            ),
            Button(
                "Close", id="mcp-inspector-test-close",
                classes="console-action-secondary", compact=True,
                tooltip="Close this test form without running the tool.",
            ),
            # Task 5: blank until `require_confirm()` fills it in for a
            # config_changed/unverifiable downgrade -- see that method's
            # docstring.
            Static("", id="mcp-inspector-test-arm-notice", classes="ds-field-row", markup=False),
            Static("", id="mcp-inspector-test-result", classes="ds-field-row", markup=False),
            id="mcp-inspector-test-panel",
        )
        await container.mount(panel)

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
        # Task 5: Close is an "other interaction" per the arm-then-confirm
        # contract -- disarm before tearing the panel down (the panel itself
        # discards the armed Run button regardless, this just keeps the
        # flag from lying about a button that's about to be gone).
        self._test_run_armed = False
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
    def test_run_armed(self) -> bool:
        """Whether the Test Tool Run button is currently armed into its
        one-shot "Confirm run" state (see `require_confirm()`)."""
        return self._test_run_armed

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

    def require_confirm(self, notice: str | None) -> None:
        """Arm the Test Tool Run button into a one-shot "Confirm run" control.

        Called by `MCPWorkbench` when `gate_tool_test()` resolves a tool to
        "ask": the press that triggered this did NOT run the tool -- the
        SAME Run button (relabeled/re-tooltipped in place, not replaced)
        becomes the confirm control instead. `notice`, when given (Task 5:
        a `config_changed` downgrade, or UX batch item 15's unverifiable-
        by-key variant), is rendered as an extra, SPECIFIC line explaining
        why a confirm is required this time (`#mcp-inspector-test-arm-
        notice`); `None` clears it. UX batch item 6: independent of
        `notice`, `#mcp-inspector-test-armed-hint` always gets the generic
        armed-explainer text on every arm -- the two can render together
        (specific reason below, generic mechanic above it, see
        `_mount_test_tool_panel()`'s widget order).

        No-op (beyond the label/variant/tooltip writes) if the panel isn't
        actually mounted -- tolerant of a race where the panel closed
        between the Run press and this call.
        """
        self._test_run_armed = True
        try:
            run_button = self.query_one("#mcp-inspector-test-run", Button)
        except NoMatches:
            pass
        else:
            run_button.label = "Confirm run"
            run_button.variant = "primary"
            run_button.tooltip = _TEST_RUN_CONFIRM_TOOLTIP
            run_button.disabled = False
        try:
            hint_widget = self.query_one("#mcp-inspector-test-armed-hint", Static)
        except NoMatches:
            pass
        else:
            hint_widget.update(_TEST_RUN_ARMED_HINT)
        try:
            notice_widget = self.query_one("#mcp-inspector-test-arm-notice", Static)
        except NoMatches:
            pass
        else:
            notice_widget.update(notice or "")

    def disarm_test_run(self) -> None:
        """Revert the Run button to its normal, unarmed state (no-op if
        already unarmed).

        The arm-then-confirm contract is "any other interaction disarms" --
        `show_tool()` and `_close_test_tool_panel()` cover tool switch/mode
        switch/Close (mirrors `MCPServersMode.disarm_delete()`); `MCPWorkbench`
        also calls this directly when it consumes a confirming press (the
        run is about to dispatch, so the button should read "Run" again by
        the time it re-enables).
        """
        if not self._test_run_armed:
            return
        self._test_run_armed = False
        try:
            run_button = self.query_one("#mcp-inspector-test-run", Button)
        except NoMatches:
            pass
        else:
            run_button.label = "Run"
            run_button.variant = "default"
            run_button.tooltip = _TEST_RUN_TOOLTIP
        try:
            hint_widget = self.query_one("#mcp-inspector-test-armed-hint", Static)
        except NoMatches:
            pass
        else:
            hint_widget.update("")
        try:
            notice_widget = self.query_one("#mcp-inspector-test-arm-notice", Static)
        except NoMatches:
            pass
        else:
            notice_widget.update("")

    def _handle_test_run(self) -> None:
        tool = self._current_tool
        if tool is None:
            return
        try:
            form = self.query_one("#mcp-inspector-test-form", MCPSchemaForm)
            result_widget = self.query_one("#mcp-inspector-test-result", Static)
            run_button = self.query_one("#mcp-inspector-test-run", Button)
        except NoMatches:
            return
        try:
            arguments = form.collect_arguments()
        except ValueError as exc:
            result_widget.update(str(exc))
            return
        run_button.disabled = True
        self.post_message(self.ToolTestRequested(tool.server_key, tool.name, arguments))

    def show_tool_result(
        self, *, server_key: str, tool_name: str, ok: bool, text: str, duration_ms: int,
        blocked: bool = False,
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
        (debug-logged only); it belongs to a panel that is no longer
        showing.

        `blocked` (Task 5, UX batch item 5): True for the permissions
        deny-gate's synthetic result -- the call never reached the tool at
        all, so the status line reads "Blocked · not run" instead of
        routing through the ok/duration_ms failure template ("Failed ·
        0ms"), which would misleadingly imply an attempted, timed run.
        `ok`/`duration_ms` are still accepted (the deny-gate call site
        passes its usual `ok=False, duration_ms=0`) but ignored for the
        status line when `blocked` is True.
        """
        current = self._current_tool
        if current is None or current.server_key != server_key or current.name != tool_name:
            logger.debug(
                f"MCPInspector: dropping stale tool result for "
                f"server_key={server_key!r} tool_name={tool_name!r} "
                f"(current tool is "
                f"{(current.server_key, current.name) if current else None!r})"
            )
            return
        try:
            result_widget = self.query_one("#mcp-inspector-test-result", Static)
        except NoMatches:
            return
        if blocked:
            status_line = "Blocked · not run"
        else:
            status = "OK" if ok else "Failed"
            status_line = f"{status} · {_format_duration_ms(duration_ms)}"
        result_widget.update(f"{status_line}\n{text}")
        try:
            self.query_one("#mcp-inspector-test-run", Button).disabled = False
        except NoMatches:
            pass

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
        """
        self._service = service
        self._sections = sections or [("Overview", "overview")]
        self._advanced_source = source
        self._advanced_target_label = target_label
        self.query_one("#mcp-adv-object", Static).update(self._advanced_object_label())
        self.query_one("#mcp-adv-content", Static).update("")
        section_select = self.query_one("#mcp-adv-section-select", Select)
        with section_select.prevent(Select.Changed):
            section_select.set_options(self._sections)
            section_select.value = self._sections[0][1]
        self._refresh_advanced_actions()
        self.run_worker(self._load_advanced_section(self._sections[0][1]),
                        group="mcp-adv-section", exclusive=True)

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
        with action_select.prevent(Select.Changed):
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
            selected = previous_value if previous_value in option_values else options[0][1]
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
                f"MCPInspector: policy gate raised for action_id={action_id!r}; "
                f"failing closed: {exc}"
            )
            return False
        return bool(getattr(decision, "allowed", True))

    async def _load_advanced_section(self, section: str) -> None:
        if self._service is None:
            return
        payload = await self._service.load_section(section)
        self.query_one("#mcp-adv-content", Static).update(
            render_unified_mcp_section(section, payload)
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
            self.run_worker(self._load_advanced_section(str(event.value)),
                            group="mcp-adv-section", exclusive=True)
        elif select_id == "mcp-adv-action-select":
            event.stop()
            if not _is_blank(event.value):
                self.query_one("#mcp-adv-payload", TextArea).text = (
                    self._action_templates.get(str(event.value), "{}")
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "mcp-adv-run":
            event.stop()
            self.run_worker(self._run_advanced_action(), group="mcp-adv-run", exclusive=True)
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
                self._mount_test_tool_panel(), group="mcp-inspector-test-panel", exclusive=True
            )
            return
        if button_id == "mcp-inspector-test-run":
            event.stop()
            self._handle_test_run()
            return
        if button_id == "mcp-inspector-test-close":
            event.stop()
            self.run_worker(
                self._close_test_tool_panel(), group="mcp-inspector-test-panel", exclusive=True
            )
            return
        if button_id == "mcp-inspector-reallow":
            event.stop()
            tool = self._current_permission_tool
            if tool is not None:
                self.post_message(self.ReallowRequested(tool.server_key, tool.name))
            return

    async def _run_advanced_action(self) -> None:
        result_widget = self.query_one("#mcp-adv-result", Static)
        action_select = self.query_one("#mcp-adv-action-select", Select)
        if self._service is None or _is_blank(action_select.value):
            return
        raw = self.query_one("#mcp-adv-payload", TextArea).text or "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            result_widget.update(f"Invalid JSON payload: {exc}")
            return
        try:
            result = await self._service.run_action(str(action_select.value), payload)
        except Exception as exc:  # surface, never crash the inspector
            result_widget.update(f"Action failed: {exc}")
            return
        if isinstance(result, dict):
            result = redact_mapping(result)
        result_widget.update(json.dumps(result, default=str, indent=1)[:2000])
