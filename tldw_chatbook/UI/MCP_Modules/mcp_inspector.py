"""MCP Hub inspector: readiness explanation, actions, Advanced escape hatch."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Label, Select, Static, TextArea

from tldw_chatbook.MCP.readiness import HubAction, ReadinessSnapshot
from tldw_chatbook.MCP.redaction import redact_mapping
from tldw_chatbook.UI.MCP_Modules.unified_mcp_sections import render_unified_mcp_section

# Actions that have first-class UI in Phase 1. Everything else renders
# disabled and points at the Advanced runner below (capability preserved).
_WIRED_ACTIONS = {HubAction.VIEW_DETAILS, HubAction.OPEN_TOOL_CATALOG, HubAction.OPEN_AUDIT}

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

# Tooltips for the actions that have first-class UI in Phase 1 (_WIRED_ACTIONS).
# Every rendered action button must explain its outcome -- disabled buttons get
# a tooltip below; these cover the wired, enabled ones.
_WIRED_ACTION_TOOLTIPS: dict[HubAction, str] = {
    HubAction.VIEW_DETAILS: "Show this server's detail view in Servers mode.",
    HubAction.OPEN_TOOL_CATALOG: "Switch to Tools mode.",
    HubAction.OPEN_AUDIT: "Switch to Audit mode.",
}


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
    }
    """

    class HubActionRequested(Message, namespace="mcp_inspector"):
        def __init__(self, action: HubAction, server_key: str | None) -> None:
            super().__init__()
            self.action = action
            self.server_key = server_key

    def __init__(self, **kwargs: Any) -> None:
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"ds-inspector {classes}".strip(), **kwargs)
        self._snapshot: ReadinessSnapshot | None = None
        self._service: Any = None
        self._sections: list[tuple[str, str]] = [("Overview", "overview")]
        self._action_templates: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Static("Inspector", classes="destination-section")
        yield Static("Select a server to see its readiness.", id="mcp-inspector-state",
                     classes="ds-status-badge", markup=False)
        yield Static("", id="mcp-inspector-message", classes="ds-field-row", markup=False)
        yield Vertical(id="mcp-inspector-actions")
        yield Static("Advanced (legacy control plane)", classes="destination-section")
        with VerticalScroll(id="mcp-adv-scroll"):
            yield Label("Section", classes="form-label")
            yield Select(self._sections, id="mcp-adv-section-select", allow_blank=False,
                         value="overview")
            yield Static("", id="mcp-adv-content", classes="ds-field-row", markup=False)
            yield Label("Action", classes="form-label")
            yield Select([("No actions available", Select.BLANK)], id="mcp-adv-action-select",
                         value=Select.BLANK)
            yield Label("Payload (JSON)", classes="form-label")
            yield TextArea("{}", id="mcp-adv-payload")
            yield Button("Run Action", id="mcp-adv-run", classes="console-action-primary",
                         compact=True,
                         tooltip="Run the selected legacy control-plane action with this JSON payload.")
            yield Static("", id="mcp-adv-result", classes="ds-field-row", markup=False)

    # -- readiness block -----------------------------------------------------

    def update_readiness(self, snapshot: ReadinessSnapshot | None) -> None:
        self._snapshot = snapshot
        state = self.query_one("#mcp-inspector-state", Static)
        message = self.query_one("#mcp-inspector-message", Static)
        actions = self.query_one("#mcp-inspector-actions", Vertical)
        actions.remove_children()
        if snapshot is None:
            state.update("Select a server to see its readiness.")
            message.update("")
            return
        state.update(f"{snapshot.badge_text()}  {snapshot.label}")
        reason = snapshot.primary_reason
        reason_suffix = f" [{reason.value}]" if reason else ""
        message.update(f"{snapshot.message}{reason_suffix}")
        for action in snapshot.allowed_actions:
            button = Button(
                _ACTION_LABELS[action],
                id=f"mcp-inspector-action-{action.value}",
                classes="mcp-inspector-action console-action-secondary",
                compact=True,
            )
            if action not in _WIRED_ACTIONS:
                button.disabled = True
                button.tooltip = "Available in a later phase — use Advanced below."
            else:
                button.tooltip = _WIRED_ACTION_TOOLTIPS.get(action, _ACTION_LABELS[action])
            actions.mount(button)

    # -- advanced escape hatch -----------------------------------------------

    def set_service_context(self, service: Any, sections: list[tuple[str, str]]) -> None:
        self._service = service
        self._sections = sections or [("Overview", "overview")]
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
        descriptors = []
        if self._service is not None:
            loader = getattr(self._service, "available_actions", None)
            if callable(loader):
                descriptors = [d for d in (loader() or []) if self._action_allowed(d)]
        self._action_templates = {
            str(d["name"]): str(d.get("payload_template") or "{}") for d in descriptors
        }
        with action_select.prevent(Select.Changed):
            if not descriptors:
                action_select.set_options([("No actions available", Select.BLANK)])
                action_select.value = Select.BLANK
                action_select.disabled = True
                run_button.disabled = True
                return
            options = [(str(d["label"]), str(d["name"])) for d in descriptors]
            action_select.set_options(options)
            action_select.value = options[0][1]
            action_select.disabled = False
            run_button.disabled = False
            payload.text = self._action_templates.get(options[0][1], "{}")

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
        if button_id.startswith("mcp-inspector-action-"):
            event.stop()
            action = HubAction(button_id.removeprefix("mcp-inspector-action-"))
            server_key = self._snapshot.server_key if self._snapshot else None
            self.post_message(self.HubActionRequested(action, server_key))

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
