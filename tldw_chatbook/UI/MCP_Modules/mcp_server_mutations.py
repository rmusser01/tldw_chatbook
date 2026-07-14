# tldw_chatbook/UI/MCP_Modules/mcp_server_mutations.py
"""Scope-gated add/edit form for server-source external-server records, plus
their credential-slot management -- both drive `run_action()` through the
same `SubmitRequested(action, payload)` seam the workbench already wires for
Task 6-8's local-profile forms.

Mirrors `MCPProfileForm`'s state-driven-button/single-error-Static structure:
Save disables itself on a valid submit (re-armed by `show_error()`), one
`#mcp-srv-error` Static surfaces validation/host failures, and every Button
carries a tooltip. Never includes owner_scope_type/owner_scope_id in any
posted payload -- the control-plane service injects those itself, and its
action schemas are `extra="forbid"`.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Checkbox, Input, Select, Static

_TRANSPORT_OPTIONS: list[tuple[str, str]] = [
    ("HTTP", "http"), ("SSE", "sse"), ("stdio", "stdio"),
]
_SECRET_KIND_OPTIONS: list[tuple[str, str]] = [
    ("Bearer token", "bearer_token"), ("API key", "api_key"), ("Client secret", "client_secret"),
]
_PRIVILEGE_OPTIONS: list[tuple[str, str]] = [("Read", "read"), ("Write", "write")]


def _blank(value: Any) -> bool:
    return value is Select.BLANK or value is Select.NULL


class MCPServerMutationsPanel(Vertical):
    """Add/edit form for one external-server record plus its credential slots.

    Add mode (`record is None`): id/name/transport/url/enabled fields ->
    `external_server.create`. Edit mode (`record` supplied): name/enabled
    only -> `external_server.update`, plus a credentials section rendering
    `slots` (fetched by the workbench via `external_server.slots.list`
    before mounting this panel) with per-slot Delete/secret Set/Clear
    controls and an add-slot subform.
    """

    DEFAULT_CSS = """
    MCPServerMutationsPanel { height: auto; min-height: 0; }
    MCPServerMutationsPanel .mcp-slot-row { height: auto; min-height: 0; }
    """

    class SubmitRequested(Message, namespace="mcp_server_mutations"):
        def __init__(self, action: str, payload: dict[str, Any]) -> None:
            super().__init__()
            self.action = action
            self.payload = payload

    class Cancelled(Message, namespace="mcp_server_mutations"):
        pass

    def __init__(
        self,
        *,
        record: dict[str, Any] | None = None,
        slots: list[dict[str, Any]] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._record = dict(record) if record else None
        self._slots = [dict(s) for s in (slots or ())]

    @property
    def is_edit(self) -> bool:
        return self._record is not None

    # -- compose ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        record = self._record or {}
        title = "Edit external server" if self.is_edit else "Add external server"
        yield Static(title, classes="destination-section", markup=False)

        yield Static("Server id", classes="form-label")
        id_input = Input(
            value=str(record.get("server_id") or ""), id="mcp-srv-id",
            placeholder="web-search", compact=True,
        )
        id_input.disabled = self.is_edit
        yield id_input

        yield Static("Name", classes="form-label")
        yield Input(value=str(record.get("name") or ""), id="mcp-srv-name",
                    placeholder="Web Search", compact=True)

        if not self.is_edit:
            yield Static("Transport", classes="form-label")
            yield Select(
                _TRANSPORT_OPTIONS, id="mcp-srv-transport", allow_blank=False,
                value="http", compact=True,
            )
            yield Static("URL", classes="form-label")
            yield Input(value="", id="mcp-srv-url", placeholder="https://mcp.example/api",
                        compact=True)

        yield Checkbox(
            "Enabled", value=bool(record.get("enabled", True)), id="mcp-srv-enabled",
            compact=True,
        )

        yield Static("", id="mcp-srv-error", classes="ds-field-row", markup=False)
        with Horizontal(classes="ds-toolbar"):
            yield Button(
                "Save", id="mcp-srv-save", classes="console-action-primary", compact=True,
                tooltip="Validate and save this external server.",
            )
            yield Button(
                "Cancel", id="mcp-srv-cancel", classes="console-action-secondary", compact=True,
                tooltip="Discard changes.",
            )

        if self.is_edit:
            yield from self._compose_credentials(record)

    def _compose_credentials(self, record: dict[str, Any]) -> ComposeResult:
        yield Static("Credentials", classes="destination-section", markup=False)
        server_id = str(record.get("server_id") or "")
        if not self._slots:
            yield Static("No credential slots yet.", classes="ds-field-row", markup=False)
        for index, slot in enumerate(self._slots):
            slot_name = str(slot.get("slot_name") or "")
            display_name = str(slot.get("display_name") or slot_name)
            # Slot names/display names are server-supplied -- escape before
            # they reach a tooltip (Rich-markup-interpreting, unlike a
            # `markup=False` Static) so a hostile slot record can't inject
            # styling/control sequences. Mirrors mcp_rail.py's row tooltips.
            safe_display_name = escape_markup(display_name)
            with Vertical(classes="mcp-slot-row"):
                yield Static(
                    f"{display_name} ({slot_name})", classes="ds-field-row", markup=False,
                )
                secret_input = Input(
                    value="", id=f"mcp-slot-secret-{index}", password=True,
                    placeholder="New secret value", compact=True,
                )
                yield secret_input
                with Horizontal(classes="ds-toolbar"):
                    yield Button(
                        "Set secret", id=f"mcp-slot-secret-set-{index}",
                        classes="console-action-secondary", compact=True,
                        tooltip=f"Set the secret for {safe_display_name}.",
                    )
                    yield Button(
                        "Clear secret", id=f"mcp-slot-secret-clear-{index}",
                        classes="console-action-secondary", compact=True,
                        tooltip=f"Clear the stored secret for {safe_display_name}.",
                    )
                    yield Button(
                        "Delete slot", id=f"mcp-slot-delete-{index}",
                        classes="console-action-secondary", compact=True,
                        tooltip=f"Delete the {safe_display_name} credential slot.",
                    )

        yield Static("Add credential slot", classes="destination-section", markup=False)
        yield Static("Slot name", classes="form-label")
        yield Input(value="", id="mcp-slot-name", placeholder="token_readonly", compact=True)
        yield Static("Display name", classes="form-label")
        yield Input(value="", id="mcp-slot-display", placeholder="Read-only token", compact=True)
        yield Static("Secret kind", classes="form-label")
        yield Select(
            _SECRET_KIND_OPTIONS, id="mcp-slot-kind", allow_blank=False, value="bearer_token",
            compact=True,
        )
        yield Static("Privilege class", classes="form-label")
        yield Select(
            _PRIVILEGE_OPTIONS, id="mcp-slot-privilege", allow_blank=False, value="read",
            compact=True,
        )
        yield Checkbox("Required", value=False, id="mcp-slot-required", compact=True)
        yield Button(
            "Add slot", id="mcp-slot-add", classes="console-action-secondary", compact=True,
            tooltip=f"Add a new credential slot for {escape_markup(server_id) or 'this server'}.",
        )

    # -- payload builders --------------------------------------------------

    def _server_id(self) -> str:
        if self.is_edit:
            return str((self._record or {}).get("server_id") or "")
        return self.query_one("#mcp-srv-id", Input).value.strip()

    def build_payload(self) -> dict[str, Any]:
        """Parse the record fields into the exact `run_action` payload.

        Returns:
            `external_server.create` payload (add mode) or
            `external_server.update` payload (edit mode).

        Raises:
            ValueError: A required field is missing.
        """
        server_id = self._server_id()
        if not server_id:
            raise ValueError("Server id is required.")
        name = self.query_one("#mcp-srv-name", Input).value.strip()
        if not name:
            raise ValueError("Name is required.")
        enabled = self.query_one("#mcp-srv-enabled", Checkbox).value
        if self.is_edit:
            return {"server_id": server_id, "name": name, "enabled": bool(enabled)}
        transport_select = self.query_one("#mcp-srv-transport", Select)
        transport = "" if _blank(transport_select.value) else str(transport_select.value)
        if not transport:
            raise ValueError("Transport is required.")
        url = self.query_one("#mcp-srv-url", Input).value.strip()
        config: dict[str, Any] = {"url": url} if url else {}
        return {
            "server_id": server_id,
            "name": name,
            "transport": transport,
            "config": config,
            "enabled": bool(enabled),
        }

    def _slot_payload(self) -> dict[str, Any]:
        slot_name = self.query_one("#mcp-slot-name", Input).value.strip()
        if not slot_name:
            raise ValueError("Slot name is required.")
        display_name = self.query_one("#mcp-slot-display", Input).value.strip()
        if not display_name:
            raise ValueError("Display name is required.")
        kind_select = self.query_one("#mcp-slot-kind", Select)
        secret_kind = "" if _blank(kind_select.value) else str(kind_select.value)
        privilege_select = self.query_one("#mcp-slot-privilege", Select)
        privilege_class = "" if _blank(privilege_select.value) else str(privilege_select.value)
        is_required = self.query_one("#mcp-slot-required", Checkbox).value
        return {
            "server_id": self._server_id(),
            "slot_name": slot_name,
            "display_name": display_name,
            "secret_kind": secret_kind,
            "privilege_class": privilege_class,
            "is_required": bool(is_required),
        }

    # -- host contract -------------------------------------------------------

    def show_error(self, text: str) -> None:
        """Surface an error and re-enable Save so the user can retry.

        Mirrors `MCPProfileForm.show_error()`: `on_button_pressed` disables
        Save on a valid submit; the host reporting a failure through this
        method is what re-arms it (success unmounts the whole panel instead).
        """
        self.query_one("#mcp-srv-error", Static).update(text)
        self.query_one("#mcp-srv-save", Button).disabled = False

    # -- events -----------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "mcp-srv-save":
            event.stop()
            try:
                payload = self.build_payload()
            except ValueError as exc:
                self.show_error(str(exc))
                return
            action = "external_server.update" if self.is_edit else "external_server.create"
            # Disable while the host's save is in flight -- same
            # state-driven-button discipline as MCPProfileForm.Save.
            event.button.disabled = True
            self.post_message(self.SubmitRequested(action, payload))
            return
        if button_id == "mcp-srv-cancel":
            event.stop()
            self.post_message(self.Cancelled())
            return
        if button_id == "mcp-slot-add":
            event.stop()
            try:
                payload = self._slot_payload()
            except ValueError as exc:
                self.show_error(str(exc))
                return
            self.post_message(self.SubmitRequested("external_server.slot.create", payload))
            return
        if button_id.startswith("mcp-slot-secret-set-"):
            event.stop()
            index = button_id.removeprefix("mcp-slot-secret-set-")
            self._post_secret_set(index)
            return
        if button_id.startswith("mcp-slot-secret-clear-"):
            event.stop()
            index = button_id.removeprefix("mcp-slot-secret-clear-")
            self._post_slot_action("external_server.slot.secret.clear", index)
            return
        if button_id.startswith("mcp-slot-delete-"):
            event.stop()
            index = button_id.removeprefix("mcp-slot-delete-")
            self._post_slot_action("external_server.slot.delete", index)
            return

    def _slot_name_for(self, index: str) -> str | None:
        try:
            slot = self._slots[int(index)]
        except (ValueError, IndexError):
            return None
        name = slot.get("slot_name")
        return str(name) if name else None

    def _post_secret_set(self, index: str) -> None:
        slot_name = self._slot_name_for(index)
        if slot_name is None:
            return
        secret_input = self.query_one(f"#mcp-slot-secret-{index}", Input)
        secret = secret_input.value
        if not secret:
            self.show_error("Enter a secret value before setting it.")
            return
        self.post_message(
            self.SubmitRequested(
                "external_server.slot.secret.set",
                {"server_id": self._server_id(), "slot_name": slot_name, "secret": secret},
            )
        )
        # Never let the plaintext secret linger in the field once posted.
        secret_input.value = ""

    def _post_slot_action(self, action: str, index: str) -> None:
        slot_name = self._slot_name_for(index)
        if slot_name is None:
            return
        self.post_message(
            self.SubmitRequested(action, {"server_id": self._server_id(), "slot_name": slot_name})
        )
