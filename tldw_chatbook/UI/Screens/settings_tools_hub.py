"""Settings Tools & MCP category pane: mode strip + the MCP workbench.

Port of the retired MCP destination shell (ADR-240): the
same mode chips, per-mode shortcut sets, exclusive reload worker group, and
bounded initial view state -- hosted inside the Settings screen instead of a
standalone destination. DOM ids (#mcp-mode-strip, mcp-mode-* chips,
#mcp-hub-workbench) are kept so the app-bundle CSS and ported tests are
unchanged.

The module also hosts the Settings-side tools-domain support code the ratchet
offset (spec amendment A2) moved out of ``settings_screen.py``: the deep-link
context validation, the category summary/ownership records, the MCP-service
accessors, and the Tool Pack failure copy. Tools-domain only (ADR-240) --
nothing unrelated to agent tools lives here.
"""

import logging
import os
from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Button, Static

from ...Widgets.destination_workbench import DestinationModeStrip
from ..MCP_Modules.mcp_workbench import MCP_HUB_MODES, MCPWorkbench
from .settings_config_models import (
    LISTED_SAVE_MODELS,
    SettingsCategorySummary,
    SettingsCategoryId,
    SettingsOwnershipRecord,
)

if TYPE_CHECKING:
    from ...Tool_Packs.contracts import ToolPackError

logger = logging.getLogger(__name__)

_MODE_BY_BUTTON_ID = {spec["button_id"]: mode for mode, spec in MCP_HUB_MODES.items()}

# TASK-35155: plain-language pass under the TASK-35018/35019 copy rules.
# Each tooltip answers the question that mode exists for; the chip names
# stay the surface's own concrete nouns (rows ARE servers, the list IS
# tools) except "audit", an accounting word for a log, now "History".
_MODE_TOOLTIPS = {
    "servers": "Servers mode: view MCP servers and their readiness.",
    "tools": "Tools mode: browse and test what your agents can use.",
    "permissions": (
        "Permissions mode: choose what agents may use freely and what "
        "asks you first (Allow / Ask / Off). Space cycles the selected row."
    ),
    "audit": (
        "History mode: review what your agents ran — tool execution "
        "history and a call's detail."
    ),
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

TOOLS_HUB_CATEGORY_SUMMARY = SettingsCategorySummary(
    SettingsCategoryId.TOOLS,
    "Tools & MCP",
    "What your agents can use, what asks first, and what they did.",
    LISTED_SAVE_MODELS[SettingsCategoryId.TOOLS],
)

TOOLS_HUB_OWNERSHIP_RECORD = SettingsOwnershipRecord(
    category=SettingsCategoryId.TOOLS,
    owns_config_sections=(
        "MCP permission-store Tool policy",
        "[mcp] enable/expose controls",
    ),
    reads_runtime_state_from=(
        "Unified MCP panel",
        "MCP configured server target store",
    ),
    writes_allowed=True,
    runtime_owner="MCP workbench (immediate reviewed actions)",
    boundary_copy=(
        "Tools & MCP owns servers, tools, permissions, and history "
        "in place; Tool Profiles next door owns profile lifecycle."
    ),
    recovery_copy=(
        "Re-open the category or press r to force a workbench reload."
    ),
)

# The sibling tools-boundary record: Tool Profiles' every row is about the
# MCP permission store and portable tool packs (ADR-240 keeps the category
# itself next door; only the record's home moved, ratchet offset A2).
TOOL_PROFILES_OWNERSHIP_RECORD = SettingsOwnershipRecord(
    category=SettingsCategoryId.TOOL_PROFILES,
    owns_config_sections=(
        "MCP permission-store Tool profiles",
        "portable Tool Pack import receipts",
    ),
    reads_runtime_state_from=(
        "ToolPackService lifecycle",
        "workspace profile references",
        "MCP permission inventory",
    ),
    writes_allowed=True,
    runtime_owner="ToolPackService (immediate reviewed actions)",
    boundary_copy=(
        "Tool Profiles manages profile lifecycle; Tools & MCP owns policy "
        "editing. Tool Packs never install tools."
    ),
    recovery_copy=(
        "Re-inspect stale imports and remove only profiles with no active or "
        "archived workspace references."
    ),
)


def tools_footer_hints(pane: "ToolsHubPanel | None") -> tuple[tuple[str, str], ...]:
    """The mounted pane's active-mode hint set (servers default off-pane)."""
    mode = pane.active_mode if pane is not None else "servers"
    return TOOLS_HUB_MODE_SHORTCUTS.get(mode, ())


def parse_tools_navigation_context(
    context: dict[str, object],
) -> dict[str, object] | None:
    """Validate one Tools & MCP navigation context (ADR-240 deep links).

    Same bounds the retired MCP destination's apply_navigation_context
    enforced: known keys only, a mode from ``MCP_HUB_MODES``, and a profile
    deep-link triple of a non-empty id (<= 128 chars), a revision that is
    None or >= 1, and a digest of exactly 64 hex chars.

    Args:
        context: The navigation-context payload addressed to the TOOLS
            category.

    Returns:
        The hub pane's initial view state; ``{}`` when the context carries
        nothing to apply; ``None`` when the context is unsafe, invalid, or
        malformed and must be ignored outright.
    """
    allowed_keys = {
        "category",
        "mode",
        "tool_policy_profile_id",
        "profile_revision",
        "profile_policy_digest",
    }
    if not set(context).issubset(allowed_keys):
        logger.debug("Ignoring unsafe Tools Settings navigation context")
        return None
    view_state: dict[str, object] | None = None
    mode = context.get("mode")
    if mode is not None:
        if not isinstance(mode, str) or mode not in MCP_HUB_MODES:
            logger.debug("Ignoring invalid Tools Settings mode: %r", mode)
            return None
        view_state = {"mode": mode}
    profile_id = context.get("tool_policy_profile_id")
    if profile_id is not None:
        revision = context.get("profile_revision")
        digest = context.get("profile_policy_digest")
        if (
            type(profile_id) is not str
            or not profile_id
            or len(profile_id) > 128
            or (
                revision is not None
                and (type(revision) is not int or revision < 1)
            )
            or (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            )
        ):
            logger.debug("Ignoring malformed Tools profile deep-link context")
            return None
        view_state = {
            "mode": "permissions",
            "tool_policy_profile_id": profile_id,
            "profile_revision": revision,
            "profile_policy_digest": digest,
        }
    return view_state if view_state is not None else {}


def mcp_permission_store(app_instance: Any) -> Any:
    """The unified MCP service's permission store, or None (guarded)."""
    service = getattr(app_instance, "unified_mcp_service", None)
    return getattr(service, "permission_store", None)


def builtin_hub_tool_names(app_instance: Any) -> list[str]:
    """Built-in hub tool names for the posture preview (guarded).

    Reads the unified service's local inventory — the same builtin
    catalog source the MCP workbench's hub list uses. Any missing seam
    yields an empty list, which the preview renders as its single
    "Tool catalog unavailable" degrade line.
    """
    service = getattr(app_instance, "unified_mcp_service", None)
    local_service = getattr(service, "local_service", None)
    get_inventory = getattr(local_service, "get_inventory", None)
    if not callable(get_inventory):
        return []
    try:
        inventory = get_inventory()
    except Exception:  # noqa: BLE001 -- preview degrades, never blocks
        return []
    from ...MCP.hub_tool_catalog import builtin_tools_from_inventory

    return [tool.name for tool in builtin_tools_from_inventory(inventory or {})]


def tool_pack_failure_copy(operation: str, error: "ToolPackError") -> str:
    """Return bounded recovery copy for one stable Tool Pack error."""
    if operation == "export" and error.category == "cancelled":
        return "Export cancelled before publication."
    if operation == "export" and error.category == "destination_changed":
        return "Export destination changed. Choose a new filename."
    if operation == "import" and error.category == "activation_uncertain":
        return (
            "Import outcome uncertain. Check the current profile state "
            "before retrying."
        )
    if operation == "export" and error.category == "durability_uncertain":
        return (
            "Export outcome uncertain. Check the chosen destination "
            "before retrying."
        )
    if operation == "remove":
        return {
            "outcome_uncertain": (
                "Removal outcome uncertain. Check the current profile state "
                "before retrying."
            ),
            "stale": "Profile changed. Review its current details before removing it.",
            "in_use": (
                "Profile is in use. Finish its active run, then reopen Tool Profiles."
            ),
            "referenced": (
                "Profile is referenced by a workspace. Update its active and "
                "archived workspace defaults first."
            ),
            "non_removable": "Profile cannot be removed. Review its current details.",
        }.get(
            error.category, "Removal unavailable. Check the current profile state."
        )
    if operation == "export" and error.category == "publication_unsupported":
        if os.name == "nt":
            return (
                "Export unavailable · publication_unsupported. Native Windows "
                "Tool Pack publication is not supported yet."
            )
        return (
            "Export unavailable · publication_unsupported. Required safe "
            "publication primitives are unavailable on this platform."
        )
    return f"{operation.title()} failed · {error.category}"

# Shared reload-worker identity so a manual refresh, a runtime-triggered
# reload, and repeats of either serialize against each other.
_RELOAD_WORKER_GROUP = "settings-tools-hub-reload"


class ToolsHubPanel(Vertical):
    """Hosts the MCP workbench for the Settings Tools & MCP category."""

    # Ported verbatim from the retired MCP destination's BINDINGS (ADR-240).
    # show=False: the surfaced contract is the per-mode footer hint set
    # (TOOLS_HUB_MODE_SHORTCUTS, F-055), not the binding list. Pane-held
    # bindings fire when focus is inside the pane, so the keys are scoped
    # to the Tools & MCP surface itself.
    BINDINGS = [
        Binding("1", "tools_mode('servers')", "Servers", show=False),
        Binding("2", "tools_mode('tools')", "Tools", show=False),
        Binding("3", "tools_mode('permissions')", "Permissions", show=False),
        Binding("4", "tools_mode('audit')", "History", show=False),
        Binding("a", "tools_add_server", "Add server", show=False),
        Binding("r", "tools_refresh", "Refresh", show=False),
        Binding("t", "tools_test_tool", "Test tool", show=False),
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
    /* A1: focus must not impersonate the active-mode indicator (bold
    underline). Uses the raw $surface/$text tokens deliberately -- the
    $ds-focus-* aliases are only defined once the app-wide tcss bundle is
    loaded, and harness-mounted tests set no CSS_PATH (same reasoning as the
    retired shell's copy). */
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
        # TASK-35155: the purpose line answers the three questions users
        # actually bring here (what can my agents use, what asks first,
        # what did they do) instead of restating the protocol's name.
        yield Static(
            "What your agents can use, what asks first, and what they "
            "did — most people never need to change anything here.",
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

    def suspend_recovery_review(self) -> None:
        """Invalidate an accepted recovery review's UI receipt (ADR-240 port).

        The workbench method is the single owner of that semantics; this
        proxy exists so Settings' suspend path needs no workbench knowledge.
        """
        if self.workbench is not None:
            self.workbench.suspend_recovery_review()

    def register_footer_hints(self) -> None:
        """(Re-)assert this pane's per-mode hint set on the footer slot.

        The screen's own "settings" footer source shares one slot with this
        "settings-tools" source (last writer wins), so every re-registration
        of the former must be followed by one of these -- focus changes,
        resume, and category swaps all funnel back here.
        """
        register = getattr(self.screen, "register_footer_shortcuts", None)
        if callable(register):
            register(source="settings-tools", shortcuts=tools_footer_hints(self))

    def action_tools_mode(self, mode: str) -> None:
        """1-4 -- switch the hub's mode."""
        self.activate_mode(mode)

    def action_tools_add_server(self) -> None:
        """a -- open the Add-server form."""
        self.add_server()

    def action_tools_refresh(self) -> None:
        """r -- force-reload the workbench."""
        self.refresh_now()

    def action_tools_test_tool(self) -> None:
        """t -- open the Test Tool panel for the selected tool."""
        self.test_selected_tool()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        mode = _MODE_BY_BUTTON_ID.get(event.button.id or "")
        if mode is None:
            return
        event.stop()
        self.activate_mode(mode)

    def on_mcp_workbench_mode_changed(self, event: MCPWorkbench.ModeChanged) -> None:
        event.stop()
        self._sync_mode_chips(event.mode)
        # F-055: re-assert the per-mode footer hints for the new mode.
        self.register_footer_hints()
