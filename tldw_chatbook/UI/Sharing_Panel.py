"""Lightweight remote-only server Sharing panel."""

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


class SharingPanel(ScrollableContainer):
    """Panel for server-owned Sharing links, permissions, and shared-workspace actions."""

    DEFAULT_CSS = """
    SharingPanel {
        layout: vertical;
        padding: 1;
        height: 100%;
        background: $panel;
    }

    SharingPanel #sharing-disabled {
        padding: 2;
        color: $text-muted;
        text-style: italic;
    }

    SharingPanel .sharing-section {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $secondary;
        background: $boost;
    }

    SharingPanel .sharing-actions {
        layout: horizontal;
        height: auto;
        margin-top: 1;
    }

    SharingPanel .sharing-actions Button {
        margin-right: 1;
    }

    SharingPanel TextArea {
        height: 5;
        margin-bottom: 1;
        background: $surface;
    }

    SharingPanel #sharing-status {
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
        self.scope_service = getattr(app_instance, "server_sharing_scope_service", None)

    def compose(self) -> ComposeResult:
        yield Static("Server Sharing requires server mode.", id="sharing-disabled")
        with Container(id="sharing-main"):
            with Container(classes="sharing-section"):
                yield Label("Workspace Shares")
                yield Input(placeholder="Workspace ID", id="sharing-workspace-id")
                scope_type = Select([("Team", "team"), ("Org", "org")], id="sharing-scope-type")
                scope_type.value = "team"
                yield scope_type
                yield Input(placeholder="Scope ID", id="sharing-scope-id", validators=[Number()])
                access_level = Select(
                    [("View Chat", "view_chat"), ("View + Chat Add", "view_chat_add"), ("Full Edit", "full_edit")],
                    id="sharing-access-level",
                )
                access_level.value = "view_chat"
                yield access_level
                yield Checkbox("Allow Clone", value=True, id="sharing-allow-clone")
                yield Checkbox("Include Revoked", id="sharing-include-revoked")
                yield Input(placeholder="Share ID", id="sharing-share-id", validators=[Number()])
                with Horizontal(classes="sharing-actions"):
                    yield Button("Create Share", variant="primary", id="sharing-create-workspace-share-btn")
                    yield Button("List Shares", id="sharing-list-workspace-shares-btn")
                    yield Button("Update Share", id="sharing-update-share-btn")
                    yield Button("Revoke Share", id="sharing-revoke-share-btn")

            with Container(classes="sharing-section"):
                yield Label("Shared With Me")
                yield Input(placeholder="Shared media ID", id="sharing-shared-media-id", validators=[Number()])
                yield Input(placeholder="Clone name", id="sharing-clone-name")
                yield TextArea("", id="sharing-chat-query")
                with Horizontal(classes="sharing-actions"):
                    yield Button("List Shared", id="sharing-list-shared-with-me-btn")
                    yield Button("Get Workspace", id="sharing-get-shared-workspace-btn")
                    yield Button("Clone", id="sharing-clone-btn")
                    yield Button("List Sources", id="sharing-list-sources-btn")
                    yield Button("Get Media", id="sharing-get-media-btn")
                    yield Button("Chat", id="sharing-chat-btn")

            with Container(classes="sharing-section"):
                yield Label("Share Tokens")
                resource_type = Select([("Workspace", "workspace"), ("Chatbook", "chatbook")], id="sharing-resource-type")
                resource_type.value = "workspace"
                yield resource_type
                yield Input(placeholder="Resource ID", id="sharing-resource-id")
                yield Input(placeholder="Password (optional)", password=True, id="sharing-token-password")
                yield Input(placeholder="Max uses", id="sharing-token-max-uses", validators=[Number()])
                yield Input(placeholder="Expires at ISO 8601", id="sharing-token-expires-at")
                yield Input(placeholder="Token ID", id="sharing-token-id", validators=[Number()])
                yield Input(placeholder="Raw/public token", id="sharing-public-token")
                yield Input(placeholder="Password verify value", password=True, id="sharing-public-password")
                with Horizontal(classes="sharing-actions"):
                    yield Button("Create Token", variant="primary", id="sharing-create-token-btn")
                    yield Button("List Tokens", id="sharing-list-tokens-btn")
                    yield Button("Revoke Token", id="sharing-revoke-token-btn")
                    yield Button("Preview Public", id="sharing-preview-public-btn")
                    yield Button("Verify Password", id="sharing-verify-password-btn")
                    yield Button("Import Public", id="sharing-import-public-btn")

            yield Static("No Sharing operation run yet.", id="sharing-status")

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
        if normalized_backend not in {"local", "server"}:
            return "local"
        return normalized_backend

    def _show_server_ui(self, enabled: bool) -> None:
        self.query_one("#sharing-disabled", Static).display = not enabled
        self.query_one("#sharing-main", Container).display = enabled

    def _set_controls_disabled(self, disabled: bool) -> None:
        for selector, widget_type in (
            ("#sharing-workspace-id", Input),
            ("#sharing-scope-type", Select),
            ("#sharing-scope-id", Input),
            ("#sharing-access-level", Select),
            ("#sharing-allow-clone", Checkbox),
            ("#sharing-include-revoked", Checkbox),
            ("#sharing-share-id", Input),
            ("#sharing-create-workspace-share-btn", Button),
            ("#sharing-list-workspace-shares-btn", Button),
            ("#sharing-update-share-btn", Button),
            ("#sharing-revoke-share-btn", Button),
            ("#sharing-shared-media-id", Input),
            ("#sharing-clone-name", Input),
            ("#sharing-chat-query", TextArea),
            ("#sharing-list-shared-with-me-btn", Button),
            ("#sharing-get-shared-workspace-btn", Button),
            ("#sharing-clone-btn", Button),
            ("#sharing-list-sources-btn", Button),
            ("#sharing-get-media-btn", Button),
            ("#sharing-chat-btn", Button),
            ("#sharing-resource-type", Select),
            ("#sharing-resource-id", Input),
            ("#sharing-token-password", Input),
            ("#sharing-token-max-uses", Input),
            ("#sharing-token-expires-at", Input),
            ("#sharing-token-id", Input),
            ("#sharing-public-token", Input),
            ("#sharing-public-password", Input),
            ("#sharing-create-token-btn", Button),
            ("#sharing-list-tokens-btn", Button),
            ("#sharing-revoke-token-btn", Button),
            ("#sharing-preview-public-btn", Button),
            ("#sharing-verify-password-btn", Button),
            ("#sharing-import-public-btn", Button),
        ):
            self.query_one(selector, widget_type).disabled = disabled

    async def refresh_for_mode(self) -> None:
        self.runtime_backend = self._current_runtime_backend()
        enabled = self.runtime_backend == "server" and self.scope_service is not None
        self._show_server_ui(enabled)
        self._set_controls_disabled(not enabled)
        if self.runtime_backend != "server":
            self.query_one("#sharing-status", Static).update("Server Sharing requires server mode.")
        elif self.scope_service is None:
            self.query_one("#sharing-status", Static).update("Server Sharing service is unavailable.")

    def on_mount(self) -> None:
        self.run_worker(self.refresh_for_mode(), exclusive=True)

    @staticmethod
    def _clean_string(value: Any) -> str:
        return str(value or "").strip()

    def _input_value(self, selector: str) -> str:
        return self._clean_string(self.query_one(selector, Input).value)

    def _int_input(self, selector: str, *, required: bool = True) -> int | None:
        raw_value = self._input_value(selector)
        if not raw_value:
            if required:
                raise ValueError(f"{selector} is required.")
            return None
        return int(raw_value)

    def _share_id(self) -> int:
        return int(self._input_value("#sharing-share-id") or "0")

    def _public_token(self) -> str:
        token = self._input_value("#sharing-public-token")
        if not token:
            raise ValueError("Public token is required.")
        return token

    def _render_payload(self, title: str, payload: Mapping[str, Any] | list[Any]) -> None:
        formatted_payload = json.dumps(payload, indent=2, sort_keys=True, default=str)
        self.query_one("#sharing-status", Static).update(f"{title}\n{formatted_payload}")

    async def _run_operation(self, title: str, operation_name: str, **kwargs: Any) -> None:
        self.runtime_backend = self._current_runtime_backend()
        if self.runtime_backend != "server" or self.scope_service is None:
            self.notify("Server Sharing requires server mode.", severity="warning")
            return
        try:
            operation = getattr(self.scope_service, operation_name)
            result = await self._maybe_await(operation(mode="server", **kwargs))
            self._render_payload(title, result if result is not None else {})
        except Exception as exc:
            logger.error(f"Server Sharing operation failed: {operation_name}: {exc}", exc_info=True)
            self.query_one("#sharing-status", Static).update(f"Error: {exc}")
            self.notify(f"Server Sharing operation failed: {exc}", severity="error")

    def notify(self, message: str, *, severity: str = "information") -> None:
        notifier = getattr(self.app_instance, "notify", None)
        if callable(notifier):
            notifier(message, severity=severity)

    @on(Button.Pressed, "#sharing-create-workspace-share-btn")
    def handle_create_workspace_share(self) -> None:
        self.run_worker(self.create_workspace_share(), exclusive=True)

    @on(Button.Pressed, "#sharing-list-workspace-shares-btn")
    def handle_list_workspace_shares(self) -> None:
        self.run_worker(self.list_workspace_shares(), exclusive=True)

    @on(Button.Pressed, "#sharing-update-share-btn")
    def handle_update_share(self) -> None:
        self.run_worker(self.update_share(), exclusive=True)

    @on(Button.Pressed, "#sharing-revoke-share-btn")
    def handle_revoke_share(self) -> None:
        self.run_worker(self.revoke_share(), exclusive=True)

    @on(Button.Pressed, "#sharing-list-shared-with-me-btn")
    def handle_list_shared_with_me(self) -> None:
        self.run_worker(self.list_shared_with_me(), exclusive=True)

    @on(Button.Pressed, "#sharing-get-shared-workspace-btn")
    def handle_get_shared_workspace(self) -> None:
        self.run_worker(self.get_shared_workspace(), exclusive=True)

    @on(Button.Pressed, "#sharing-clone-btn")
    def handle_clone(self) -> None:
        self.run_worker(self.clone_shared_workspace(), exclusive=True)

    @on(Button.Pressed, "#sharing-list-sources-btn")
    def handle_list_sources(self) -> None:
        self.run_worker(self.list_shared_workspace_sources(), exclusive=True)

    @on(Button.Pressed, "#sharing-get-media-btn")
    def handle_get_media(self) -> None:
        self.run_worker(self.get_shared_workspace_media(), exclusive=True)

    @on(Button.Pressed, "#sharing-chat-btn")
    def handle_chat(self) -> None:
        self.run_worker(self.chat_with_shared_workspace(), exclusive=True)

    @on(Button.Pressed, "#sharing-create-token-btn")
    def handle_create_token(self) -> None:
        self.run_worker(self.create_share_token(), exclusive=True)

    @on(Button.Pressed, "#sharing-list-tokens-btn")
    def handle_list_tokens(self) -> None:
        self.run_worker(self.list_share_tokens(), exclusive=True)

    @on(Button.Pressed, "#sharing-revoke-token-btn")
    def handle_revoke_token(self) -> None:
        self.run_worker(self.revoke_share_token(), exclusive=True)

    @on(Button.Pressed, "#sharing-preview-public-btn")
    def handle_preview_public(self) -> None:
        self.run_worker(self.preview_public_share(), exclusive=True)

    @on(Button.Pressed, "#sharing-verify-password-btn")
    def handle_verify_password(self) -> None:
        self.run_worker(self.verify_public_share_password(), exclusive=True)

    @on(Button.Pressed, "#sharing-import-public-btn")
    def handle_import_public(self) -> None:
        self.run_worker(self.import_public_share(), exclusive=True)

    async def create_workspace_share(self) -> None:
        await self._run_operation(
            "Workspace share created",
            "share_workspace",
            workspace_id=self._input_value("#sharing-workspace-id"),
            share_scope_type=str(self.query_one("#sharing-scope-type", Select).value or "team"),
            share_scope_id=int(self._int_input("#sharing-scope-id")),
            access_level=str(self.query_one("#sharing-access-level", Select).value or "view_chat"),
            allow_clone=bool(self.query_one("#sharing-allow-clone", Checkbox).value),
        )

    async def list_workspace_shares(self) -> None:
        await self._run_operation(
            "Workspace shares",
            "list_workspace_shares",
            workspace_id=self._input_value("#sharing-workspace-id"),
            include_revoked=bool(self.query_one("#sharing-include-revoked", Checkbox).value),
        )

    async def update_share(self) -> None:
        await self._run_operation(
            "Workspace share updated",
            "update_share",
            share_id=int(self._share_id()),
            access_level=str(self.query_one("#sharing-access-level", Select).value or "view_chat"),
            allow_clone=bool(self.query_one("#sharing-allow-clone", Checkbox).value),
        )

    async def revoke_share(self) -> None:
        await self._run_operation("Workspace share revoked", "revoke_share", share_id=int(self._share_id()))

    async def list_shared_with_me(self) -> None:
        await self._run_operation("Shared with me", "list_shared_with_me")

    async def get_shared_workspace(self) -> None:
        await self._run_operation("Shared workspace", "get_shared_workspace", share_id=int(self._share_id()))

    async def clone_shared_workspace(self) -> None:
        new_name = self._input_value("#sharing-clone-name") or None
        await self._run_operation("Shared workspace clone", "clone_shared_workspace", share_id=int(self._share_id()), new_name=new_name)

    async def list_shared_workspace_sources(self) -> None:
        await self._run_operation("Shared workspace sources", "list_shared_workspace_sources", share_id=int(self._share_id()))

    async def get_shared_workspace_media(self) -> None:
        await self._run_operation(
            "Shared workspace media",
            "get_shared_workspace_media",
            share_id=int(self._share_id()),
            media_id=int(self._int_input("#sharing-shared-media-id")),
        )

    async def chat_with_shared_workspace(self) -> None:
        query = self.query_one("#sharing-chat-query", TextArea).text.strip()
        if not query:
            raise ValueError("Chat query is required.")
        await self._run_operation("Shared workspace chat", "chat_with_shared_workspace", share_id=int(self._share_id()), query=query)

    async def create_share_token(self) -> None:
        max_uses = self._int_input("#sharing-token-max-uses", required=False)
        await self._run_operation(
            "Share token created",
            "create_share_token",
            resource_type=str(self.query_one("#sharing-resource-type", Select).value or "workspace"),
            resource_id=self._input_value("#sharing-resource-id"),
            access_level=str(self.query_one("#sharing-access-level", Select).value or "view_chat"),
            allow_clone=bool(self.query_one("#sharing-allow-clone", Checkbox).value),
            password=self._input_value("#sharing-token-password") or None,
            max_uses=max_uses,
            expires_at=self._input_value("#sharing-token-expires-at") or None,
        )

    async def list_share_tokens(self) -> None:
        await self._run_operation("Share tokens", "list_share_tokens")

    async def revoke_share_token(self) -> None:
        await self._run_operation("Share token revoked", "revoke_share_token", token_id=int(self._int_input("#sharing-token-id")))

    async def preview_public_share(self) -> None:
        await self._run_operation("Public share preview", "preview_public_share", token=self._public_token())

    async def verify_public_share_password(self) -> None:
        await self._run_operation(
            "Public share password verified",
            "verify_public_share_password",
            token=self._public_token(),
            password=self._input_value("#sharing-public-password"),
        )

    async def import_public_share(self) -> None:
        await self._run_operation("Public share imported", "import_public_share", token=self._public_token())
