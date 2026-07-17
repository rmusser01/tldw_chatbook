"""MCP destination shell: mode strip + rail/canvas/inspector workbench."""

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import QueryError
from textual.widgets import Button, Static

from ...Widgets.AppFooterStatus import AppFooterStatus
from ...Widgets.destination_workbench import DestinationModeStrip
from ..MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench
from ..Navigation.base_app_screen import BaseAppScreen

_MODE_BY_BUTTON_ID = {spec["button_id"]: mode for mode, spec in MCP_HUB_MODES.items()}

_MODE_TOOLTIPS = {
    "servers": "Servers mode: view MCP servers and their readiness.",
    "tools": "Tools mode: browse and test scoped MCP tools.",
    "permissions": (
        "Permissions mode: set Allow / Ask / Off per tool. Space cycles the selected row."
    ),
    "audit": "Audit mode: review MCP action history (arrives in a later phase).",
}

# T13: Console precedent is `CONSOLE_WORKBENCH_SHORTCUTS` (chat_screen.py) --
# rendered through the shared `AppFooterStatus.set_workbench_shortcuts()`
# context model, source="mcp" so it cannot clobber another screen's context
# (`clear_shortcut_context(source=...)` is a no-op unless "mcp" still owns
# it).
# T9 (P4): "space cycle permission" is display-only -- the actual `space`
# binding lives on `MCPPermissionsMode` itself (its matrix `DataTable` must
# own focus for the keypress to reach `action_cycle_state()`), not on this
# screen, mirroring how `("1-4", "mode")` documents the mode-strip Buttons'
# own bindings rather than a screen-level one.
MCP_SHORTCUTS = (
    ("1-4", "mode"), ("a", "add server"), ("r", "refresh"), ("t", "test tool"),
    ("space", "cycle permission"),
)

# T13: shared reload-worker identity between the runtime-backend-change path
# (`handle_runtime_backend_changed`) and the manual `r` keybinding
# (`action_mcp_refresh`) -- same group so `exclusive=True` also serializes a
# manual refresh against an in-flight runtime-triggered one (and vice versa),
# not just repeats of the same trigger.
_RELOAD_WORKER_GROUP = "mcp-screen-runtime-refresh"


class MCPScreen(BaseAppScreen):
    """MCP servers, tools, permissions, and audit surface."""

    BINDINGS = [
        Binding("1", "mcp_mode('servers')", "Servers", show=False),
        Binding("2", "mcp_mode('tools')", "Tools", show=False),
        Binding("3", "mcp_mode('permissions')", "Permissions", show=False),
        Binding("4", "mcp_mode('audit')", "Audit", show=False),
        Binding("a", "mcp_add_server", "Add server", show=False),
        Binding("r", "mcp_refresh", "Refresh", show=False),
        Binding("t", "mcp_test_tool", "Test tool", show=False),
    ]

    DEFAULT_CSS = """
    Button.mcp-mode-chip {
        width: auto;
        /* min-width kept in lockstep with the higher-specificity app-bundle
        rule (#mcp-mode-strip Button.mcp-mode-chip, _agentic_terminal.tcss);
        divergence here would silently lose to the bundle. */
        min-width: 10;
        height: 1;
        min-height: 1;
        padding: 0 1;
        border: none;
    }
    .mcp-mode-chip.is-active {
        border: none;
        text-style: bold underline;
    }
    /* A1: focus must not impersonate the active-mode indicator above (bold
    underline). A keyboard-focused, non-active chip gets the standard
    reverse-video focus affordance instead -- no underline -- so the two
    states read as visually distinct. Kept in lockstep with the
    higher-specificity app-bundle copy (#mcp-mode-strip .mcp-mode-chip:focus,
    _agentic_terminal.tcss).
    NOTE: uses the raw `$surface`/`$text`/`$text-muted` tokens (not the
    project's `$ds-focus-bg`/`$ds-focus-fg`/`$ds-text-muted` aliases)
    deliberately -- those aliases are only defined once the app-wide tcss
    bundle is loaded, and several destination-shell tests mount MCPScreen
    under a harness App that never sets CSS_PATH. `$ds-focus-bg` and
    `$ds-focus-fg` currently alias to exactly `$surface` and `$text` (see
    css/core/_variables.tcss), so this is not a visual compromise. */
    .mcp-mode-chip:focus,
    .mcp-mode-chip:hover:focus {
        background: $surface;
        color: $text;
        text-style: bold;
        outline: none;
    }
    /* Active AND focused: still reads as "active" (bold underline, same
    background as non-focused .is-active) rather than picking up the
    reverse-video focus treatment above. */
    .mcp-mode-chip.is-active:focus,
    .mcp-mode-chip.is-active:hover:focus {
        background: $surface;
        color: $text-muted;
        text-style: bold underline;
        outline: none;
    }
    """

    def __init__(self, app_instance: Any, **kwargs: Any) -> None:
        super().__init__(app_instance, "mcp", **kwargs)
        self.workbench: MCPWorkbench | None = None

    def compose_content(self) -> ComposeResult:
        with Vertical(id="mcp-shell"):
            yield Static("MCP", id="mcp-title", classes="ds-destination-header")
            yield Static(
                "Manage MCP servers, scoped tools, permissions, and audit readiness.",
                id="mcp-purpose",
                classes="destination-purpose",
            )
            with DestinationModeStrip(id="mcp-mode-strip", classes="destination-mode-strip"):
                for mode, spec in MCP_HUB_MODES.items():
                    chip = Button(
                        spec["label"],
                        id=spec["button_id"],
                        classes="mcp-mode-chip console-action-subdued",
                        compact=True,
                        tooltip=_MODE_TOOLTIPS.get(mode, spec["label"]),
                    )
                    chip.set_class(mode == "servers", "is-active")
                    yield chip
            self.workbench = MCPWorkbench(self.app_instance, id="mcp-hub-workbench")
            self.workbench.set_initial_view_state(self._initial_view_state())
            yield self.workbench

    def _initial_view_state(self) -> dict[str, Any] | None:
        state = self.state_data.get("mcp_hub_view_state")
        if isinstance(state, dict):
            return state
        legacy = self.state_data.get("unified_mcp_view_state")
        return legacy if isinstance(legacy, dict) else None

    def _sync_mode_chips(self, active_mode: str) -> None:
        for candidate, spec in MCP_HUB_MODES.items():
            chips = list(self.query(f"#{spec['button_id']}"))
            if chips:
                chips[0].set_class(candidate == active_mode, "is-active")

    def _activate_mode(self, mode: str) -> None:
        if self.workbench is None:
            return
        self.workbench.set_mode(mode)
        self._sync_mode_chips(self.workbench.active_mode)

    def action_mcp_mode(self, mode: str) -> None:
        self._activate_mode(mode)

    def action_mcp_add_server(self) -> None:
        """`a` keybinding: switch to Servers mode and open the Add-server form.

        Drives the workbench's `open_add_server_form()`, which follows the
        same path the overview Add-server button does (including the T9
        server-source mutation gate -- a notification with the button's own
        tooltip copy instead of opening when gated). Dispatched via a worker
        because opening the form is async (mounts `MCPProfileForm`/
        `MCPServerMutationsPanel`).
        """
        if self.workbench is None:
            return
        self._activate_mode("servers")
        self.run_worker(
            self.workbench.open_add_server_form(),
            name="mcp-screen-add-server",
            group="mcp-screen-add-server",
            exclusive=True,
        )

    def action_mcp_refresh(self) -> None:
        """`r` keybinding: reload the workbench via the existing exclusive worker.

        Shares `_RELOAD_WORKER_GROUP` with `handle_runtime_backend_changed()`
        so a manual refresh and a runtime-triggered one cannot run
        concurrently.
        """
        if self.workbench is None:
            return
        self.run_worker(
            self.workbench.reload(),
            name="mcp-screen-manual-refresh",
            group=_RELOAD_WORKER_GROUP,
            exclusive=True,
        )

    def action_mcp_test_tool(self) -> None:
        """`t` keybinding: switch to Tools mode and open the Test Tool panel
        for whatever tool the inspector currently has selected.

        Drives the workbench's `open_test_for_selected_tool()`, which both
        performs the mode switch and notifies "Select a tool first." when
        nothing is selected -- unlike `action_mcp_add_server`, there is no
        separate screen-level mode switch here because the mode switch
        happens inside that one method (see its own docstring). Dispatched
        via a worker because opening the panel is async (mounts
        `MCPSchemaForm` + Run/Close/result, mirrors `action_mcp_add_server`).
        """
        if self.workbench is None:
            return
        self.run_worker(
            self.workbench.open_test_for_selected_tool(),
            name="mcp-screen-test-tool",
            group="mcp-screen-test-tool",
            exclusive=True,
        )

    def _register_footer_shortcuts(self) -> None:
        """Register MCP Hub shortcuts with this screen's own footer if mounted."""
        try:
            footer = self.query_one(AppFooterStatus)
        except QueryError:
            return
        set_shortcuts = getattr(footer, "set_workbench_shortcuts", None)
        if callable(set_shortcuts):
            set_shortcuts(source="mcp", shortcuts=MCP_SHORTCUTS)

    def _clear_footer_shortcuts(self) -> None:
        """Clear MCP Hub shortcuts from this screen's own footer if mounted."""
        try:
            footer = self.query_one(AppFooterStatus)
        except QueryError:
            return
        clear_shortcuts = getattr(footer, "clear_shortcut_context", None)
        if callable(clear_shortcuts):
            clear_shortcuts(source="mcp")

    def on_mount(self) -> None:
        super().on_mount()
        self._register_footer_shortcuts()

    def on_screen_resume(self) -> None:
        """Called when returning to this screen (e.g. after a pushed overlay pops)."""
        self._register_footer_shortcuts()
        # Note: BaseAppScreen doesn't have on_screen_resume, so no super() call

    def on_screen_suspend(self) -> None:
        """Called when another screen is pushed on top of this one."""
        self._clear_footer_shortcuts()
        # Note: BaseAppScreen doesn't have on_screen_suspend, so no super() call

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mode = _MODE_BY_BUTTON_ID.get(event.button.id or "")
        if mode is None:
            return
        event.stop()
        self._activate_mode(mode)

    def on_mcp_workbench_mode_changed(self, event: MCPWorkbench.ModeChanged) -> None:
        event.stop()
        self._sync_mode_chips(event.mode)

    def save_state(self) -> dict[str, Any]:
        state = super().save_state()
        if self.workbench:
            state["mcp_hub_view_state"] = self.workbench.get_view_state()
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        super().restore_state(state)
        if self.workbench:
            self.workbench.set_initial_view_state(self._initial_view_state())

    async def handle_runtime_backend_changed(self, runtime_backend: str) -> None:
        """Schedule an MCP context refresh when runtime backend/source changes.

        Args:
            runtime_backend: Newly active runtime backend identifier.
        """
        _ = runtime_backend
        if self.workbench:
            self.run_worker(
                self.workbench.reload(),
                name="mcp-screen-runtime-refresh",
                group=_RELOAD_WORKER_GROUP,
                exclusive=True,
            )
