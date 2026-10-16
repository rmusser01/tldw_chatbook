"""Settings Tools & MCP category pane: mode strip + the MCP workbench.

Port of the retired MCP destination shell (`UI/Screens/mcp_screen.py`): the
same mode chips, per-mode shortcut sets, exclusive reload worker group, and
bounded initial view state -- hosted inside the Settings screen instead of a
standalone destination. DOM ids (#mcp-mode-strip, mcp-mode-* chips,
#mcp-hub-workbench) are kept so the app-bundle CSS and ported tests are
unchanged.
"""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench

_MODE_BY_BUTTON_ID = {spec["button_id"]: mode for mode, spec in MCP_HUB_MODES.items()}

_MODE_TOOLTIPS = {
    "servers": "Servers mode: view MCP servers and their readiness.",
    "tools": "Tools mode: browse and test scoped MCP tools.",
    "permissions": (
        "Permissions mode: set Allow / Ask / Off per tool. Space cycles the selected row."
    ),
    "audit": "Audit mode: review MCP tool execution history and drill into a call's detail.",
}

# F-055: the hint set is PER-MODE -- a key is only advertised where it
# actually works. Mirrors the retired screen's MCP_MODE_SHORTCUTS.
_COMMON_SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("1-4", "mode"),
    ("a", "add server"),
    ("r", "refresh"),
)
TOOLS_HUB_MODE_SHORTCUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "servers": _COMMON_SHORTCUTS,
    "tools": _COMMON_SHORTCUTS + (("t", "test tool"),),
    "permissions": _COMMON_SHORTCUTS + (("space", "cycle permission"),),
    "audit": _COMMON_SHORTCUTS,
}

# Shared reload-worker identity so a manual refresh, a runtime-triggered
# reload, and repeats of either serialize against each other.
_RELOAD_WORKER_GROUP = "settings-tools-hub-reload"


class ToolsHubPanel(Vertical):
    """Hosts the MCP workbench for the Settings Tools & MCP category."""

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
    /* A1: focus must not impersonate the active-mode indicator (bold
    underline). Uses the raw $surface/$text tokens deliberately -- the
    $ds-focus-* aliases are only defined once the app-wide tcss bundle is
    loaded, and harness-mounted tests set no CSS_PATH (same reasoning as the
    retired mcp_screen.py copy). */
    .mcp-mode-chip:focus,
    .mcp-mode-chip:hover:focus {
        background: $surface;
        color: $text;
        text-style: bold;
        outline: none;
    }
    .mcp-mode-chip.is-active:focus,
    .mcp-mode-chip.is-active:hover:focus {
        background: $surface;
        color: $text-muted;
        text-style: bold underline;
        outline: none;
    }
    """

    class ModeChanged(Message):
        """The workbench switched mode; Settings re-renders footer hints."""

        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

    def __init__(
        self,
        app_instance: Any,
        initial_view_state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._app_instance = app_instance
        self._initial_view_state = initial_view_state
        self.workbench: MCPWorkbench | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "MCP (Model Context Protocol) lets chatbook use external "
            "tools — most people never need to change anything here.",
            id="tools-hub-purpose",
            classes="destination-purpose",
        )
        with DestinationModeStrip(
            id="mcp-mode-strip", classes="destination-mode-strip"
        ):
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
        self.workbench = MCPWorkbench(self._app_instance, id="mcp-hub-workbench")
        self.workbench.set_initial_view_state(self._initial_view_state)
        yield self.workbench

    @property
    def active_mode(self) -> str:
        return self.workbench.active_mode if self.workbench else "servers"

    def get_view_state(self) -> dict[str, Any] | None:
        return self.workbench.get_view_state() if self.workbench else None

    def set_initial_view_state(self, state: dict[str, Any] | None) -> None:
        self._initial_view_state = state
        if self.workbench is not None:
            self.workbench.set_initial_view_state(state)

    def _sync_mode_chips(self, active_mode: str) -> None:
        for candidate, spec in MCP_HUB_MODES.items():
            chips = list(self.query(f"#{spec['button_id']}"))
            if chips:
                chips[0].set_class(candidate == active_mode, "is-active")

    def activate_mode(self, mode: str) -> None:
        if self.workbench is None:
            return
        self.workbench.set_mode(mode)
        self._sync_mode_chips(self.workbench.active_mode)

    def add_server(self) -> None:
        """Switch to Servers mode and open the Add-server form (worker-mounted)."""
        if self.workbench is None:
            return
        self.activate_mode("servers")
        self.run_worker(
            self.workbench.open_add_server_form(),
            name="settings-tools-hub-add-server",
            group="settings-tools-hub-add-server",
            exclusive=True,
        )

    def refresh_now(self) -> None:
        """Force-reload the workbench (bypasses the revisit cache, TASK-34965)."""
        if self.workbench is None:
            return
        self.run_worker(
            self.workbench.reload(force=True),
            name="settings-tools-hub-manual-refresh",
            group=_RELOAD_WORKER_GROUP,
            exclusive=True,
        )

    def test_selected_tool(self) -> None:
        """Open the Test Tool panel for the inspector's selected tool."""
        if self.workbench is None:
            return
        self.run_worker(
            self.workbench.open_test_for_selected_tool(),
            name="settings-tools-hub-test-tool",
            group="settings-tools-hub-test-tool",
            exclusive=True,
        )

    def handle_runtime_backend_change(self, runtime_backend: str) -> None:
        """A backend change must fully reload (force, TASK-34965)."""
        _ = runtime_backend
        if self.workbench:
            self.run_worker(
                self.workbench.reload(force=True),
                name="settings-tools-hub-runtime-refresh",
                group=_RELOAD_WORKER_GROUP,
                exclusive=True,
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mode = _MODE_BY_BUTTON_ID.get(event.button.id or "")
        if mode is None:
            return
        event.stop()
        self.activate_mode(mode)

    def on_mcp_workbench_mode_changed(self, event: MCPWorkbench.ModeChanged) -> None:
        event.stop()
        self._sync_mode_chips(event.mode)
        # F-055: Settings re-registers per-mode footer hints off this message.
        self.post_message(self.ModeChanged(event.mode))
