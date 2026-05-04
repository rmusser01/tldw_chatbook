from __future__ import annotations

import json
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Label, Select, Static, TextArea

from tldw_chatbook.MCP.unified_control_models import ServerAccessContext, UnifiedMCPContext
from tldw_chatbook.runtime_policy.engine import PolicyEngine
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY

from .unified_mcp_sections import render_unified_mcp_section


class UnifiedMCPPanel(Container):
    """Minimal Slice 1 Unified MCP host inside Tools & Settings."""

    DEFAULT_CSS = """
    UnifiedMCPPanel {
        height: 100%;
        width: 100%;
    }

    .unified-mcp-shell {
        height: 100%;
        width: 100%;
    }

    .unified-mcp-toolbar {
        layout: vertical;
        height: auto;
        margin-bottom: 1;
    }

    .unified-mcp-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }

    .unified-mcp-field {
        width: 1fr;
        margin-right: 1;
    }

    .unified-mcp-field Label {
        margin-bottom: 0;
    }

    .unified-mcp-status {
        margin-bottom: 1;
        color: $text-muted;
    }

    .unified-mcp-content {
        height: 1fr;
        overflow: auto;
        border: round $background;
        padding: 1;
        background: $boost;
    }

    .unified-mcp-actions {
        height: auto;
        margin-top: 1;
        border-top: solid $panel;
        padding-top: 1;
    }

    .unified-mcp-actions TextArea {
        height: 8;
        margin-bottom: 1;
    }

    .unified-mcp-action-row {
        layout: horizontal;
        height: auto;
        margin-bottom: 1;
    }

    .unified-mcp-action-row Button {
        width: auto;
    }
    """

    def __init__(self, app_instance: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._app_instance = app_instance
        self.context = UnifiedMCPContext(selected_section="overview")
        self._refreshing_controls = False
        self._pending_view_state: dict[str, Any] | None = None
        self._has_loaded_context = False
        self._action_templates: dict[str, str] = {}

    @property
    def app_instance(self) -> Any:
        if self._app_instance is not None:
            return self._app_instance
        try:
            return self.app
        except Exception:
            return None

    def compose(self) -> ComposeResult:
        with Vertical(classes="unified-mcp-shell"):
            with Container(classes="settings-group"):
                yield Static("Unified MCP", classes="settings-group-title")
                yield Static(
                    "Browse the local and server Unified MCP control plane without leaving Tools & Settings.",
                    classes="help-text",
                )
                with Vertical(classes="unified-mcp-toolbar"):
                    with Horizontal(classes="unified-mcp-row"):
                        with Container(classes="unified-mcp-field"):
                            yield Label("Source", classes="form-label")
                            yield Select(
                                [("Local", "local"), ("Server", "server")],
                                id="unified-mcp-source",
                                allow_blank=False,
                                value="local",
                            )
                        with Container(classes="unified-mcp-field"):
                            yield Label("Server", classes="form-label")
                            yield Select(
                                [("No configured servers", Select.BLANK)],
                                id="unified-mcp-server-target",
                                value=Select.BLANK,
                            )
                    with Horizontal(classes="unified-mcp-row"):
                        with Container(classes="unified-mcp-field"):
                            yield Label("Scope", classes="form-label")
                            yield Select(
                                [("Personal", "personal")],
                                id="unified-mcp-scope",
                                allow_blank=False,
                                value="personal",
                            )
                        with Container(classes="unified-mcp-field"):
                            yield Label("Scope Entity", classes="form-label")
                            yield Select(
                                [("No scope entities", Select.BLANK)],
                                id="unified-mcp-scope-ref",
                                value=Select.BLANK,
                            )
                        with Container(classes="unified-mcp-field"):
                            yield Label("Section", classes="form-label")
                            yield Select(
                                [("Overview", "overview"), ("Inventory", "inventory"), ("External Servers", "external_servers")],
                                id="unified-mcp-section",
                                allow_blank=False,
                                value="overview",
                            )
                yield Static("", id="unified-mcp-status", classes="unified-mcp-status")
                yield Static("Unified MCP is loading...", id="unified-mcp-content", classes="unified-mcp-content")
                with Vertical(classes="unified-mcp-actions"):
                    yield Label("Action", classes="form-label")
                    yield Select(
                        [("No actions available", Select.BLANK)],
                        id="unified-mcp-action",
                        value=Select.BLANK,
                    )
                    yield Label("Payload (JSON)", classes="form-label")
                    yield TextArea("{}", id="unified-mcp-action-payload")
                    with Horizontal(classes="unified-mcp-action-row"):
                        yield Button(
                            "Run Action",
                            id="unified-mcp-action-run",
                            variant="primary",
                            tooltip="Run the selected Unified MCP action with the JSON payload above.",
                        )
                    yield Static("", id="unified-mcp-action-result", classes="help-text")

    async def on_mount(self) -> None:
        await self.load_context()
        if self._pending_view_state:
            await self._apply_pending_view_state()

    async def load_context(self) -> UnifiedMCPContext:
        service = self._service()
        if service is None:
            self.context = UnifiedMCPContext(selected_section="overview")
            self._has_loaded_context = True
            await self._sync_controls()
            self._update_status("Unified MCP service is unavailable in this app session.")
            self._update_content("Unified MCP is unavailable.")
            return self.context

        self.context = await service.load_context()
        if self.context.selected_section is None:
            self.context = self.context.__class__(**{**self.context.to_dict(), "selected_section": "overview"})
        self._has_loaded_context = True
        await self._sync_controls()
        await self._load_active_section()
        return self.context

    async def select_source(self, source: str) -> UnifiedMCPContext:
        service = self._service()
        if service is None:
            return self.context
        self.context = await service.select_source(source)
        await self._sync_controls()
        await self._load_active_section()
        return self.context

    async def select_server_target(self, server_id: str) -> UnifiedMCPContext:
        service = self._service()
        if service is None:
            return self.context
        self.context = await service.select_server_target(server_id)
        await self._sync_controls()
        await self._load_active_section()
        return self.context

    async def select_scope(self, scope: str, scope_ref: str | None = None) -> UnifiedMCPContext:
        service = self._service()
        if service is None:
            return self.context
        if scope in {"team", "org"} and scope_ref is None:
            options = self._available_scope_ref_options()
            scope_ref = options[0][1] if options else None
        self.context = await service.select_scope(scope, scope_ref)
        await self._sync_controls()
        await self._load_active_section()
        return self.context

    async def select_scope_ref(self, scope_ref: str | None) -> UnifiedMCPContext:
        current_scope = self.context.selected_scope or "personal"
        return await self.select_scope(current_scope, scope_ref)

    async def select_section(self, section: str) -> UnifiedMCPContext:
        service = self._service()
        if service is None:
            return self.context
        self.context = await service.select_section(section)
        await self._sync_controls()
        await self._load_active_section()
        return self.context

    def get_view_state(self) -> dict[str, Any]:
        return {
            "selected_source": self.context.selected_source,
            "selected_active_server_id": self.context.selected_active_server_id,
            "selected_scope": self.context.selected_scope,
            "selected_scope_ref": self.context.selected_scope_ref,
            "selected_section": self.context.selected_section,
        }

    def set_initial_view_state(self, state: dict[str, Any] | None) -> None:
        self._pending_view_state = dict(state or {})
        if self.is_mounted and self._has_loaded_context:
            self.run_worker(
                self._apply_pending_view_state(),
                group="unified-mcp-panel-restore",
                exclusive=True,
            )

    async def restore_view_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        selected_source = str(state.get("selected_source") or self.context.selected_source or "local")
        selected_server_id = state.get("selected_active_server_id")
        selected_scope = state.get("selected_scope")
        selected_scope_ref = state.get("selected_scope_ref")
        selected_section = state.get("selected_section") or "overview"

        await self.select_source(selected_source)
        if selected_source == "server" and selected_server_id not in (None, ""):
            await self.select_server_target(str(selected_server_id))
        if selected_source == "server" and selected_scope not in (None, ""):
            await self.select_scope(str(selected_scope), str(selected_scope_ref) if selected_scope_ref not in (None, "") else None)
        await self.select_section(str(selected_section))

    async def _apply_pending_view_state(self) -> None:
        if not self._pending_view_state:
            return
        pending = dict(self._pending_view_state)
        self._pending_view_state = None
        await self.restore_view_state(pending)

    async def _load_active_section(self) -> None:
        service = self._service()
        if service is None:
            self._update_content("Unified MCP is unavailable.")
            return
        section = self.context.selected_section or "overview"
        payload = await service.load_section(section)
        self._update_status(self._status_text())
        self._update_content(render_unified_mcp_section(section, payload))

    async def _sync_controls(self) -> None:
        self._refreshing_controls = True
        try:
            source_select = self.query_one("#unified-mcp-source", Select)
            server_select = self.query_one("#unified-mcp-server-target", Select)
            scope_select = self.query_one("#unified-mcp-scope", Select)
            scope_ref_select = self.query_one("#unified-mcp-scope-ref", Select)
            section_select = self.query_one("#unified-mcp-section", Select)

            with (
                source_select.prevent(Select.Changed),
                server_select.prevent(Select.Changed),
                scope_select.prevent(Select.Changed),
                scope_ref_select.prevent(Select.Changed),
                section_select.prevent(Select.Changed),
            ):
                source_select.value = self.context.selected_source or "local"

                server_targets = self._server_target_options()
                server_select.set_options(server_targets or [("No configured servers", Select.BLANK)])
                server_select.disabled = self.context.selected_source != "server" or not bool(server_targets)
                server_values = {value for _label, value in server_targets}
                if self.context.selected_active_server_id in server_values:
                    server_select.value = self.context.selected_active_server_id
                elif server_targets:
                    server_select.value = server_targets[0][1]
                else:
                    server_select.value = Select.BLANK

                scope_options = self._available_scope_options()
                scope_select.set_options(scope_options)
                scope_select.disabled = self.context.selected_source != "server" or self.context.selected_active_server_id is None
                scope_select.value = self.context.selected_scope or "personal"

                scope_ref_options = self._available_scope_ref_options()
                scope_ref_select.set_options(scope_ref_options or [("No scope entities", Select.BLANK)])
                scope_ref_select.disabled = not bool(scope_ref_options) or self.context.selected_source != "server"
                scope_ref_values = {value for _label, value in scope_ref_options}
                if self.context.selected_scope_ref in scope_ref_values:
                    scope_ref_select.value = self.context.selected_scope_ref
                elif scope_ref_options:
                    scope_ref_select.value = scope_ref_options[0][1]
                else:
                    scope_ref_select.value = Select.BLANK

                section_options = self._available_section_options()
                section_select.set_options(section_options)
                valid_section_values = {value for _label, value in section_options}
                selected_section = self.context.selected_section or "overview"
                if selected_section not in valid_section_values:
                    selected_section = next(iter(valid_section_values), "overview")
                section_select.value = selected_section
                self._sync_action_controls()
        finally:
            self._refreshing_controls = False

    def _server_target_options(self) -> list[tuple[str, str]]:
        service = self._service()
        target_store = getattr(service, "target_store", None)
        if target_store is None:
            return []
        targets = list(target_store.list_targets())
        return [(target.label, target.server_id) for target in targets]

    def _available_scope_options(self) -> list[tuple[str, str]]:
        access_context = self._active_server_access_context()
        options: list[tuple[str, str]] = [("Personal", "personal")]
        if access_context is None:
            return options
        if access_context.manageable_team_ids:
            options.append(("Team", "team"))
        if access_context.manageable_org_ids:
            options.append(("Org", "org"))
        if access_context.can_use_system_admin_scope:
            options.append(("System Admin", "system_admin"))
        return options

    def _available_scope_ref_options(self) -> list[tuple[str, str]]:
        access_context = self._active_server_access_context()
        if access_context is None:
            return []
        if self.context.selected_scope == "team":
            return [(f"Team {team_id}", str(team_id)) for team_id in access_context.manageable_team_ids]
        if self.context.selected_scope == "org":
            return [(f"Org {org_id}", str(org_id)) for org_id in access_context.manageable_org_ids]
        return []

    def _available_section_options(self) -> list[tuple[str, str]]:
        if self.context.selected_source != "server":
            return [
                ("Overview", "overview"),
                ("Inventory", "inventory"),
                ("External Servers", "external_servers"),
                ("Governance", "governance"),
                ("Advanced", "advanced"),
            ]

        access_context = self._active_server_access_context()
        if access_context is None:
            return [("Overview", "overview")]

        capabilities = access_context.section_capabilities.to_dict()
        options: list[tuple[str, str]] = []
        for label, value in (
            ("Overview", "overview"),
            ("Inventory", "inventory"),
            ("Catalogs", "catalogs"),
            ("External Servers", "external_servers"),
            ("Governance", "governance"),
            ("Advanced", "advanced"),
        ):
            if capabilities.get(value, False):
                options.append((label, value))
        return options or [("Overview", "overview")]

    def _active_server_access_context(self) -> ServerAccessContext | None:
        server_id = self.context.selected_active_server_id
        if server_id in (None, ""):
            return None
        return self.context.per_server_state.get(server_id)

    def _sync_action_controls(self) -> None:
        action_select = self.query_one("#unified-mcp-action", Select)
        payload_area = self.query_one("#unified-mcp-action-payload", TextArea)
        run_button = self.query_one("#unified-mcp-action-run", Button)

        descriptors = self._available_actions()
        self._action_templates = {
            str(descriptor["name"]): str(descriptor.get("payload_template") or "{}")
            for descriptor in descriptors
        }
        with action_select.prevent(Select.Changed):
            if not descriptors:
                action_select.set_options([("No actions available", Select.BLANK)])
                action_select.value = Select.BLANK
                action_select.disabled = True
                payload_area.disabled = True
                run_button.disabled = True
                payload_area.text = "{}"
                return

            action_options = [(str(descriptor["label"]), str(descriptor["name"])) for descriptor in descriptors]
            action_select.set_options(action_options)
            valid_action_names = {value for _label, value in action_options}
            if action_select.value not in valid_action_names:
                selected_action = action_options[0][1]
                action_select.value = selected_action
                payload_area.text = self._action_templates.get(selected_action, "{}")
            action_select.disabled = False
            payload_area.disabled = False
            run_button.disabled = False

    def _available_actions(self) -> list[dict[str, Any]]:
        service = self._service()
        if service is None:
            return []
        action_loader = getattr(service, "available_actions", None)
        if not callable(action_loader):
            return []
        descriptors = list(action_loader() or [])
        return [
            descriptor
            for descriptor in descriptors
            if self._ui_action_allowed(str(descriptor.get("action_id") or ""))
        ]

    def _ui_action_allowed(self, action_id: str) -> bool:
        if not action_id:
            return False
        service = self._service()
        if service is None:
            return False
        runtime_state_loader = getattr(service, "runtime_state_override", None)
        runtime_state = runtime_state_loader() if callable(runtime_state_loader) else None
        if runtime_state is None:
            return False
        app_instance = self.app_instance
        require_allowed = getattr(app_instance, "require_ui_action_allowed", None)
        if callable(require_allowed):
            decision = require_allowed(action_id=action_id, runtime_state_override=runtime_state)
        else:
            engine = getattr(app_instance, "ui_policy_engine", None)
            if engine is None:
                engine = PolicyEngine(CAPABILITY_REGISTRY)
                if app_instance is not None:
                    setattr(app_instance, "ui_policy_engine", engine)
            decision = engine.evaluate(action_id=action_id, state=runtime_state)
        return bool(decision.allowed)

    async def execute_selected_action(self) -> Any:
        service = self._service()
        if service is None:
            return None

        action_select = self.query_one("#unified-mcp-action", Select)
        payload_area = self.query_one("#unified-mcp-action-payload", TextArea)
        result_area = self.query_one("#unified-mcp-action-result", Static)
        action_name = None if action_select.value == Select.BLANK else str(action_select.value)
        if not action_name:
            result_area.update("No Unified MCP action is available for this section.")
            return None

        try:
            payload = json.loads(payload_area.text or "{}")
        except json.JSONDecodeError as exc:
            result_area.update(f"Invalid JSON payload: {exc}")
            return None
        if not isinstance(payload, dict):
            result_area.update("Unified MCP action payload must be a JSON object.")
            return None

        runner = getattr(service, "run_action", None)
        if not callable(runner):
            result_area.update("Unified MCP action runner is unavailable.")
            return None

        try:
            result = await runner(action_name, payload)
        except Exception as exc:
            result_area.update(f"Action failed: {exc}")
            return None

        result_area.update(json.dumps(result, indent=2, sort_keys=True, default=str))
        await self._load_active_section()
        await self._sync_controls()
        return result

    def _update_status(self, text: str) -> None:
        status = self.query_one("#unified-mcp-status", Static)
        status.update(text)

    def _update_content(self, text: str) -> None:
        content = self.query_one("#unified-mcp-content", Static)
        content.update(text)

    def _status_text(self) -> str:
        source = self.context.selected_source or "local"
        server_id = self.context.selected_active_server_id or "-"
        scope = self.context.selected_scope or "personal"
        scope_ref = self.context.selected_scope_ref or "-"
        section = self.context.selected_section or "overview"
        return f"Source: {source} | Server: {server_id} | Scope: {scope} | Scope Ref: {scope_ref} | Section: {section}"

    def _service(self) -> Any:
        return getattr(self.app_instance, "unified_mcp_service", None)

    @staticmethod
    def _is_empty_select_value(value: object) -> bool:
        return value in (None, Select.BLANK, Select.NULL)

    @on(Select.Changed, "#unified-mcp-source")
    async def _on_source_changed(self, event: Select.Changed) -> None:
        if self._refreshing_controls or self._is_empty_select_value(event.value):
            return
        await self.select_source(str(event.value))

    @on(Select.Changed, "#unified-mcp-server-target")
    async def _on_server_target_changed(self, event: Select.Changed) -> None:
        if (
            self._refreshing_controls
            or self._is_empty_select_value(event.value)
            or self.context.selected_source != "server"
        ):
            return
        await self.select_server_target(str(event.value))

    @on(Select.Changed, "#unified-mcp-scope")
    async def _on_scope_changed(self, event: Select.Changed) -> None:
        if self._refreshing_controls or self._is_empty_select_value(event.value):
            return
        await self.select_scope(str(event.value))

    @on(Select.Changed, "#unified-mcp-scope-ref")
    async def _on_scope_ref_changed(self, event: Select.Changed) -> None:
        if self._refreshing_controls:
            return
        scope_ref = None if self._is_empty_select_value(event.value) else str(event.value)
        await self.select_scope_ref(scope_ref)

    @on(Select.Changed, "#unified-mcp-section")
    async def _on_section_changed(self, event: Select.Changed) -> None:
        if self._refreshing_controls or self._is_empty_select_value(event.value):
            return
        await self.select_section(str(event.value))

    @on(Select.Changed, "#unified-mcp-action")
    async def _on_action_changed(self, event: Select.Changed) -> None:
        if self._refreshing_controls or self._is_empty_select_value(event.value):
            return
        payload_area = self.query_one("#unified-mcp-action-payload", TextArea)
        payload_area.text = self._action_templates.get(str(event.value), "{}")

    @on(Button.Pressed, "#unified-mcp-action-run")
    async def _on_action_run_pressed(self, _event: Button.Pressed) -> None:
        await self.execute_selected_action()
