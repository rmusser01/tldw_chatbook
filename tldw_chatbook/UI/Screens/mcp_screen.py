"""MCP destination shell: mode strip + rail/canvas/inspector workbench."""

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench
from ..Navigation.base_app_screen import BaseAppScreen

_MODE_BY_BUTTON_ID = {spec["button_id"]: mode for mode, spec in MCP_HUB_MODES.items()}

_MODE_TOOLTIPS = {
    "servers": "Servers mode: view MCP servers and their readiness.",
    "tools": "Tools mode: browse scoped MCP tools (arrives in a later phase).",
    "permissions": "Permissions mode: manage MCP tool permissions (arrives in a later phase).",
    "audit": "Audit mode: review MCP action history (arrives in a later phase).",
}


class MCPScreen(BaseAppScreen):
    """MCP servers, tools, permissions, and audit surface."""

    BINDINGS = [
        Binding("1", "mcp_mode('servers')", "Servers", show=False),
        Binding("2", "mcp_mode('tools')", "Tools", show=False),
        Binding("3", "mcp_mode('permissions')", "Permissions", show=False),
        Binding("4", "mcp_mode('audit')", "Audit", show=False),
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

    def _activate_mode(self, mode: str) -> None:
        if self.workbench is None:
            return
        self.workbench.set_mode(mode)
        for candidate, spec in MCP_HUB_MODES.items():
            chips = list(self.query(f"#{spec['button_id']}"))
            if chips:
                chips[0].set_class(candidate == self.workbench.active_mode, "is-active")

    def action_mcp_mode(self, mode: str) -> None:
        self._activate_mode(mode)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mode = _MODE_BY_BUTTON_ID.get(event.button.id or "")
        if mode is None:
            return
        event.stop()
        self._activate_mode(mode)

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
                group="mcp-screen-runtime-refresh",
                exclusive=True,
            )
