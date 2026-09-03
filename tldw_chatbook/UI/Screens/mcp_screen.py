"""MCP destination shell: mode strip + rail/canvas/inspector workbench."""

from collections.abc import Mapping
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
    "tools": "Tools mode: browse and test scoped MCP tools.",
    "permissions": (
        "Permissions mode: set Allow / Ask / Off per tool. Space cycles the selected row."
    ),
    "audit": "Audit mode: review MCP tool execution history and drill into a call's detail.",
}

# T13: Console precedent is `CONSOLE_WORKBENCH_SHORTCUTS` (chat_screen.py) --
# rendered through the shared `AppFooterStatus.set_workbench_shortcuts()`
# context model, source="mcp" so it cannot clobber another screen's context
# (`clear_shortcut_context(source=...)` is a no-op unless "mcp" still owns
# it).
# F-055: the hint set is PER-MODE -- a key is only advertised where it
# actually works. `1-4`/`a`/`r` work everywhere (screen-level bindings);
# `t` only works in Tools mode (it needs a selected tool); `space` only
# works in Permissions mode with the matrix focused (the binding lives on
# `MCPPermissionsMode` itself, display-only here -- see T9 (P4) below).
_COMMON_SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("1-4", "mode"),
    ("a", "add server"),
    ("r", "refresh"),
)
MCP_MODE_SHORTCUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "servers": _COMMON_SHORTCUTS,
    "tools": _COMMON_SHORTCUTS + (("t", "test tool"),),
    "permissions": _COMMON_SHORTCUTS + (("space", "cycle permission"),),
    "audit": _COMMON_SHORTCUTS,
}

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

    BUNDLED_CSS = """
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
            # F-052/F-060 + task-2241: the plain-language onramp IS the
            # purpose line -- it leads directly under the title (nothing
            # jargon in front of it) and expands the acronym exactly once.
            # The old jargon purpose line ("Manage MCP servers, scoped
            # tools, permissions, and audit readiness.") is deleted rather
            # than demoted: the mode chips below already enumerate what the
            # screen manages, so the line was a tautology.
            yield Static(
                "MCP (Model Context Protocol) lets chatbook use external "
                "tools — most people never need to change anything here.",
                id="mcp-purpose",
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
            self.workbench = MCPWorkbench(self.app_instance, id="mcp-hub-workbench")
            self.workbench.set_initial_view_state(self._initial_view_state())
            yield self.workbench

    def _initial_view_state(self) -> dict[str, Any] | None:
        state = self.state_data.get("mcp_hub_view_state")
        if isinstance(state, dict):
            return state
        legacy = self.state_data.get("unified_mcp_view_state")
        return legacy if isinstance(legacy, dict) else None

    def apply_navigation_context(self, context: Mapping[str, object]) -> None:
        """Accept one bounded Settings deep-link into an exact profile."""
        allowed = {
            "mode",
            "tool_policy_profile_id",
            "profile_revision",
            "profile_policy_digest",
        }
        if set(context) != allowed or context.get("mode") != "permissions":
            return
        profile_id = context.get("tool_policy_profile_id")
        revision = context.get("profile_revision")
        digest = context.get("profile_policy_digest")
        if (
            type(profile_id) is not str
            or not profile_id
            or len(profile_id) > 128
            or (revision is not None and (type(revision) is not int or revision < 1))
            or (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            )
        ):
            return
        view_state = {
            "mode": "permissions",
            "tool_policy_profile_id": profile_id,
            "profile_revision": revision,
            "profile_policy_digest": digest,
        }
        self.state_data["mcp_hub_view_state"] = view_state
        if self.workbench is not None:
            self.workbench.set_initial_view_state(view_state)

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
        """`t` keybinding: open the Test Tool panel for whatever tool the
        inspector currently has selected.

        Drives the workbench's `open_test_for_selected_tool()`, which opens
        the panel first and only switches to Tools mode on success (F-055)
        -- with nothing selected it no-ops with a "Select a tool in Tools
        mode first." hint in the CURRENT mode instead of hijacking the mode
        and toasting there. Dispatched via a worker because opening the
        panel is async (mounts `MCPSchemaForm` + Run/Close/result, mirrors
        `action_mcp_add_server`).
        """
        if self.workbench is None:
            return
        self.run_worker(
            self.workbench.open_test_for_selected_tool(),
            name="mcp-screen-test-tool",
            group="mcp-screen-test-tool",
            exclusive=True,
        )

    def _register_footer_shortcuts(self, mode: str | None = None) -> None:
        """Register MCP Hub shortcuts via BaseAppScreen's persisting API.

        F-055: per-mode -- the active mode's set is the one registered, so
        the footer never advertises a key that is dead (or worse,
        hijacking) in the current context. Re-registered on every mode
        change (`on_mcp_workbench_mode_changed`) and on mount/resume.
        """
        if mode is None:
            mode = self.workbench.active_mode if self.workbench else "servers"
        shortcuts = MCP_MODE_SHORTCUTS.get(mode, _COMMON_SHORTCUTS)
        self.register_footer_shortcuts(source="mcp", shortcuts=shortcuts)

    def _clear_footer_shortcuts(self) -> None:
        """Clear MCP Hub shortcuts from this screen's own footer."""
        self.clear_footer_shortcuts(source="mcp")

    def on_mount(self) -> None:
        # No super().on_mount(): the dispatcher already invokes
        # BaseAppScreen.on_mount separately for this Mount event.
        self._register_footer_shortcuts()

    def on_screen_resume(self) -> None:
        """Called when returning to this screen (e.g. after a pushed overlay pops)."""
        self._register_footer_shortcuts()
        # Textual's MRO dispatch also invokes BaseAppScreen's shared reconciliation;
        # this handler extends that resume event with MCP-owned footer state.

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
        # F-055: keep the footer hint honest about which keys work in the
        # mode just entered.
        self._register_footer_shortcuts(event.mode)

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
