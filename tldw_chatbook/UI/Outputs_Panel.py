"""Lightweight server outputs/templates/artifacts control panel."""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any, Mapping

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.validation import Number
from textual.widgets import Button, Checkbox, Input, Label, Select, Static, TextArea

if TYPE_CHECKING:
    from ..app import TldwCli


class OutputsPanel(ScrollableContainer):
    """First-slice source-aware panel for server output templates and artifacts."""

    DEFAULT_CSS = """
    OutputsPanel {
        layout: vertical;
        padding: 1;
        height: 100%;
        background: $panel;
    }

    OutputsPanel #outputs-disabled {
        padding: 2;
        color: $text-muted;
        text-style: italic;
    }

    OutputsPanel .outputs-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $secondary;
        background: $boost;
    }

    OutputsPanel .outputs-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }

    OutputsPanel .outputs-actions Button {
        margin-right: 1;
    }

    OutputsPanel TextArea {
        height: 6;
        margin-bottom: 1;
        background: $surface;
    }

    OutputsPanel #outputs-status {
        min-height: 9;
        padding: 1;
        border: solid $secondary;
        background: $surface;
    }
    """

    runtime_backend: reactive[str] = reactive("local")

    def __init__(self, app_instance: "TldwCli", **kwargs: Any):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.scope_service = getattr(app_instance, "server_outputs_scope_service", None)

    def compose(self) -> ComposeResult:
        yield Static("Server Outputs require server mode.", id="outputs-disabled")
        with Container(id="outputs-main"):
            with Container(classes="outputs-section"):
                yield Label("Output Templates")
                with Horizontal():
                    yield Input(placeholder="Query", id="outputs-template-query")
                    yield Input("50", id="outputs-template-limit", validators=[Number()])
                    yield Input("0", id="outputs-template-offset", validators=[Number()])
                yield Input(placeholder="Template ID", id="outputs-template-id", validators=[Number()])
                yield Input(placeholder="Template name", id="outputs-template-name")
                template_type = Select(
                    [
                        ("Briefing Markdown", "briefing_markdown"),
                        ("Newsletter Markdown", "newsletter_markdown"),
                        ("MECE Markdown", "mece_markdown"),
                        ("Newsletter HTML", "newsletter_html"),
                        ("TTS Audio", "tts_audio"),
                    ],
                    id="outputs-template-type",
                )
                template_type.value = "briefing_markdown"
                yield template_type
                template_format = Select(
                    [("Markdown", "md"), ("HTML", "html"), ("MP3", "mp3")],
                    id="outputs-template-format",
                )
                template_format.value = "md"
                yield template_format
                yield Input(placeholder="Description", id="outputs-template-description")
                yield Checkbox("Default template", id="outputs-template-default")
                yield TextArea("", id="outputs-template-body")
                with Horizontal(classes="outputs-actions"):
                    yield Button("List Templates", id="outputs-list-templates-btn")
                    yield Button("Create Template", variant="primary", id="outputs-create-template-btn")
                    yield Button("Get Template", id="outputs-get-template-btn")
                    yield Button("Update Template", id="outputs-update-template-btn")
                    yield Button("Delete Template", id="outputs-delete-template-btn")

            with Container(classes="outputs-section"):
                yield Label("Template Preview")
                yield Input(placeholder="Preview template ID", id="outputs-preview-template-id", validators=[Number()])
                yield Input(placeholder="Preview item IDs (comma separated)", id="outputs-preview-item-ids")
                yield Input("50", id="outputs-preview-limit", validators=[Number()])
                with Horizontal(classes="outputs-actions"):
                    yield Button("Preview Template", id="outputs-preview-template-btn")

            with Container(classes="outputs-section"):
                yield Label("Output Artifacts")
                with Horizontal():
                    yield Input("1", id="outputs-artifact-page", validators=[Number()])
                    yield Input("50", id="outputs-artifact-size", validators=[Number()])
                yield Input(placeholder="Run ID", id="outputs-artifact-run-id", validators=[Number()])
                yield Input(placeholder="Type filter", id="outputs-artifact-type")
                yield Input(placeholder="Workspace tag", id="outputs-artifact-workspace-tag")
                yield Input(placeholder="Output ID", id="outputs-artifact-id", validators=[Number()])
                with Horizontal(classes="outputs-actions"):
                    yield Button("List Outputs", id="outputs-list-artifacts-btn")
                    yield Button("List Deleted", id="outputs-list-deleted-btn")
                    yield Button("Get Output", id="outputs-get-output-btn")

            with Container(classes="outputs-section"):
                yield Label("Render Output")
                yield Input(placeholder="Create template ID", id="outputs-create-template-id", validators=[Number()])
                yield Input(placeholder="Create item IDs (comma separated)", id="outputs-create-item-ids")
                yield Input(placeholder="Title", id="outputs-create-title")
                yield Input(placeholder="Workspace tag", id="outputs-create-workspace-tag")
                yield Checkbox("Ingest to media DB", id="outputs-create-ingest")
                with Horizontal(classes="outputs-actions"):
                    yield Button("Create Output", variant="primary", id="outputs-create-output-btn")

            with Container(classes="outputs-section"):
                yield Label("Mutate Output")
                yield Input(placeholder="Update output ID", id="outputs-update-output-id", validators=[Number()])
                yield Input(placeholder="Update title", id="outputs-update-title")
                update_format = Select(
                    [("Keep Format", Select.BLANK), ("Markdown", "md"), ("HTML", "html")],
                    id="outputs-update-format",
                )
                update_format.value = Select.BLANK
                yield update_format
                yield Input(placeholder="Delete output ID", id="outputs-delete-output-id", validators=[Number()])
                yield Checkbox("Hard delete", id="outputs-delete-hard")
                yield Checkbox("Delete file", id="outputs-delete-file")
                with Horizontal(classes="outputs-actions"):
                    yield Button("Update Output", id="outputs-update-output-btn")
                    yield Button("Delete Output", id="outputs-delete-output-btn")

            yield Static("No Outputs operation run yet.", id="outputs-status")

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _current_runtime_backend(self) -> str:
        resolver = getattr(self.app_instance, "get_authoritative_runtime_source", None)
        if callable(resolver):
            runtime_backend = resolver()
        else:
            runtime_backend = (
                getattr(self.app_instance, "current_runtime_backend", None)
                or getattr(self.app_instance, "runtime_backend", None)
                or "local"
            )
        normalized_backend = str(runtime_backend or "local").strip().lower()
        return normalized_backend if normalized_backend in {"local", "server"} else "local"

    def _show_server_ui(self, enabled: bool) -> None:
        self.query_one("#outputs-disabled", Static).display = not enabled
        self.query_one("#outputs-main", Container).display = enabled

    def _set_controls_disabled(self, disabled: bool) -> None:
        for selector, widget_type in (
            ("#outputs-template-query", Input),
            ("#outputs-template-limit", Input),
            ("#outputs-template-offset", Input),
            ("#outputs-template-id", Input),
            ("#outputs-template-name", Input),
            ("#outputs-template-type", Select),
            ("#outputs-template-format", Select),
            ("#outputs-template-description", Input),
            ("#outputs-template-default", Checkbox),
            ("#outputs-template-body", TextArea),
            ("#outputs-list-templates-btn", Button),
            ("#outputs-create-template-btn", Button),
            ("#outputs-get-template-btn", Button),
            ("#outputs-update-template-btn", Button),
            ("#outputs-delete-template-btn", Button),
            ("#outputs-preview-template-id", Input),
            ("#outputs-preview-item-ids", Input),
            ("#outputs-preview-limit", Input),
            ("#outputs-preview-template-btn", Button),
            ("#outputs-artifact-page", Input),
            ("#outputs-artifact-size", Input),
            ("#outputs-artifact-run-id", Input),
            ("#outputs-artifact-type", Input),
            ("#outputs-artifact-workspace-tag", Input),
            ("#outputs-artifact-id", Input),
            ("#outputs-list-artifacts-btn", Button),
            ("#outputs-list-deleted-btn", Button),
            ("#outputs-get-output-btn", Button),
            ("#outputs-create-template-id", Input),
            ("#outputs-create-item-ids", Input),
            ("#outputs-create-title", Input),
            ("#outputs-create-workspace-tag", Input),
            ("#outputs-create-ingest", Checkbox),
            ("#outputs-create-output-btn", Button),
            ("#outputs-update-output-id", Input),
            ("#outputs-update-title", Input),
            ("#outputs-update-format", Select),
            ("#outputs-delete-output-id", Input),
            ("#outputs-delete-hard", Checkbox),
            ("#outputs-delete-file", Checkbox),
            ("#outputs-update-output-btn", Button),
            ("#outputs-delete-output-btn", Button),
        ):
            self.query_one(selector, widget_type).disabled = disabled

    async def refresh_for_mode(self) -> None:
        self.runtime_backend = self._current_runtime_backend()
        enabled = self.runtime_backend == "server" and self.scope_service is not None
        self._show_server_ui(enabled)
        self._set_controls_disabled(not enabled)
        if self.runtime_backend != "server":
            self.query_one("#outputs-status", Static).update("Server Outputs require server mode.")
        elif self.scope_service is None:
            self.query_one("#outputs-status", Static).update("Server Outputs service is unavailable.")

    def on_mount(self) -> None:
        self.run_worker(self.refresh_for_mode(), exclusive=True)

    @staticmethod
    def _clean_string(value: Any) -> str:
        return str(value or "").strip()

    def _input_value(self, selector: str) -> str:
        return self._clean_string(self.query_one(selector, Input).value)

    def _text_area_value(self, selector: str) -> str:
        return self._clean_string(self.query_one(selector, TextArea).text)

    def _select_value(self, selector: str) -> str | None:
        value = self.query_one(selector, Select).value
        if value in (None, Select.BLANK):
            return None
        return self._clean_string(value)

    def _int_input(self, selector: str, *, required: bool = True, default: int | None = None) -> int | None:
        raw_value = self._input_value(selector)
        if not raw_value:
            if required:
                raise ValueError(f"{selector} is required.")
            return default
        return int(raw_value)

    def _csv_ints(self, selector: str) -> list[int]:
        raw_value = self._input_value(selector)
        if not raw_value:
            return []
        values: list[int] = []
        for piece in raw_value.split(","):
            cleaned = piece.strip()
            if cleaned:
                values.append(int(cleaned))
        return values

    def _render_payload(self, title: str, payload: Mapping[str, Any] | list[Any]) -> None:
        formatted_payload = json.dumps(payload, indent=2, sort_keys=True, default=str)
        self.query_one("#outputs-status", Static).update(f"{title}\n{formatted_payload}")

    async def _run_operation(self, status_title: str, operation_name: str, **kwargs: Any) -> None:
        self.runtime_backend = self._current_runtime_backend()
        if self.runtime_backend != "server" or self.scope_service is None:
            self.notify("Server Outputs require server mode.", severity="warning")
            return
        try:
            operation = getattr(self.scope_service, operation_name)
            result = await self._maybe_await(operation(mode="server", **kwargs))
            self._render_payload(status_title, result if result is not None else {})
        except Exception as exc:
            logger.opt(exception=True).error(f"Server Outputs operation failed: {operation_name}: {exc}")
            self.query_one("#outputs-status", Static).update(f"Error: {exc}")
            self.notify(f"Server Outputs operation failed: {exc}", severity="error")

    def notify(self, message: str, *, severity: str = "information") -> None:
        notifier = getattr(self.app_instance, "notify", None)
        if callable(notifier):
            notifier(message, severity=severity)

    async def list_output_templates(self) -> None:
        await self._run_operation(
            "Output Templates",
            "list_output_templates",
            q=self._input_value("#outputs-template-query") or None,
            limit=self._int_input("#outputs-template-limit", default=50, required=False) or 50,
            offset=self._int_input("#outputs-template-offset", default=0, required=False) or 0,
        )

    async def create_output_template(self) -> None:
        await self._run_operation(
            "Output Template Created",
            "create_output_template",
            name=self._input_value("#outputs-template-name"),
            type=self._select_value("#outputs-template-type"),
            format=self._select_value("#outputs-template-format"),
            body=self._text_area_value("#outputs-template-body"),
            description=self._input_value("#outputs-template-description") or None,
            is_default=self.query_one("#outputs-template-default", Checkbox).value,
        )

    async def get_output_template(self) -> None:
        await self._run_operation(
            "Output Template Detail",
            "get_output_template",
            template_id=self._int_input("#outputs-template-id"),
        )

    async def update_output_template(self) -> None:
        await self._run_operation(
            "Output Template Updated",
            "update_output_template",
            template_id=self._int_input("#outputs-template-id"),
            name=self._input_value("#outputs-template-name") or None,
            type=self._select_value("#outputs-template-type"),
            format=self._select_value("#outputs-template-format"),
            body=self._text_area_value("#outputs-template-body") or None,
            description=self._input_value("#outputs-template-description") or None,
            is_default=self.query_one("#outputs-template-default", Checkbox).value,
        )

    async def delete_output_template(self) -> None:
        await self._run_operation(
            "Output Template Deleted",
            "delete_output_template",
            template_id=self._int_input("#outputs-template-id"),
        )

    async def preview_output_template(self) -> None:
        await self._run_operation(
            "Output Template Preview",
            "preview_output_template",
            template_id=self._int_input("#outputs-preview-template-id"),
            item_ids=self._csv_ints("#outputs-preview-item-ids"),
            limit=self._int_input("#outputs-preview-limit", default=50, required=False) or 50,
        )

    async def list_outputs(self) -> None:
        kwargs: dict[str, Any] = {
            "page": self._int_input("#outputs-artifact-page", default=1, required=False) or 1,
            "size": self._int_input("#outputs-artifact-size", default=50, required=False) or 50,
        }
        run_id = self._int_input("#outputs-artifact-run-id", required=False)
        if run_id is not None:
            kwargs["run_id"] = run_id
        artifact_type = self._input_value("#outputs-artifact-type")
        if artifact_type:
            kwargs["type"] = artifact_type
        workspace_tag = self._input_value("#outputs-artifact-workspace-tag")
        if workspace_tag:
            kwargs["workspace_tag"] = workspace_tag
        await self._run_operation("Output Artifacts", "list_outputs", **kwargs)

    async def list_deleted_outputs(self) -> None:
        await self._run_operation(
            "Deleted Output Artifacts",
            "list_deleted_outputs",
            page=self._int_input("#outputs-artifact-page", default=1, required=False) or 1,
            size=self._int_input("#outputs-artifact-size", default=50, required=False) or 50,
        )

    async def get_output(self) -> None:
        await self._run_operation(
            "Output Detail",
            "get_output",
            output_id=self._int_input("#outputs-artifact-id"),
        )

    async def create_output(self) -> None:
        await self._run_operation(
            "Output Created",
            "create_output",
            template_id=self._int_input("#outputs-create-template-id"),
            item_ids=self._csv_ints("#outputs-create-item-ids"),
            title=self._input_value("#outputs-create-title") or None,
            workspace_tag=self._input_value("#outputs-create-workspace-tag") or None,
            ingest_to_media_db=self.query_one("#outputs-create-ingest", Checkbox).value,
        )

    async def update_output(self) -> None:
        kwargs: dict[str, Any] = {
            "output_id": self._int_input("#outputs-update-output-id"),
        }
        title = self._input_value("#outputs-update-title")
        if title:
            kwargs["title"] = title
        update_format = self._select_value("#outputs-update-format")
        if update_format:
            kwargs["format"] = update_format
        await self._run_operation("Output Updated", "update_output", **kwargs)

    async def delete_output(self) -> None:
        await self._run_operation(
            "Output Deleted",
            "delete_output",
            output_id=self._int_input("#outputs-delete-output-id"),
            hard=self.query_one("#outputs-delete-hard", Checkbox).value,
            delete_file=self.query_one("#outputs-delete-file", Checkbox).value,
        )

    @on(Button.Pressed, "#outputs-list-templates-btn")
    def handle_list_output_templates(self) -> None:
        self.run_worker(self.list_output_templates(), exclusive=True)

    @on(Button.Pressed, "#outputs-create-template-btn")
    def handle_create_output_template(self) -> None:
        self.run_worker(self.create_output_template(), exclusive=True)

    @on(Button.Pressed, "#outputs-get-template-btn")
    def handle_get_output_template(self) -> None:
        self.run_worker(self.get_output_template(), exclusive=True)

    @on(Button.Pressed, "#outputs-update-template-btn")
    def handle_update_output_template(self) -> None:
        self.run_worker(self.update_output_template(), exclusive=True)

    @on(Button.Pressed, "#outputs-delete-template-btn")
    def handle_delete_output_template(self) -> None:
        self.run_worker(self.delete_output_template(), exclusive=True)

    @on(Button.Pressed, "#outputs-preview-template-btn")
    def handle_preview_output_template(self) -> None:
        self.run_worker(self.preview_output_template(), exclusive=True)

    @on(Button.Pressed, "#outputs-list-artifacts-btn")
    def handle_list_outputs(self) -> None:
        self.run_worker(self.list_outputs(), exclusive=True)

    @on(Button.Pressed, "#outputs-list-deleted-btn")
    def handle_list_deleted_outputs(self) -> None:
        self.run_worker(self.list_deleted_outputs(), exclusive=True)

    @on(Button.Pressed, "#outputs-get-output-btn")
    def handle_get_output(self) -> None:
        self.run_worker(self.get_output(), exclusive=True)

    @on(Button.Pressed, "#outputs-create-output-btn")
    def handle_create_output(self) -> None:
        self.run_worker(self.create_output(), exclusive=True)

    @on(Button.Pressed, "#outputs-update-output-btn")
    def handle_update_output(self) -> None:
        self.run_worker(self.update_output(), exclusive=True)

    @on(Button.Pressed, "#outputs-delete-output-btn")
    def handle_delete_output(self) -> None:
        self.run_worker(self.delete_output(), exclusive=True)
