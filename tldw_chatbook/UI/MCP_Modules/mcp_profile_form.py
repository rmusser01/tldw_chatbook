# tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py
"""Inline add/edit form for local MCP server profiles (stdio-only)."""

from __future__ import annotations

import re
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Static, TextArea

_ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_PLACEHOLDER_VALUE_RE = re.compile(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$")


class MCPProfileForm(Vertical):
    """State-driven inline form; validation errors surface in one Static."""

    DEFAULT_CSS = """
    MCPProfileForm { height: auto; min-height: 0; }
    #mcp-form-args, #mcp-form-env { height: 4; min-height: 2; }
    """

    class SubmitRequested(Message, namespace="mcp_profile_form"):
        def __init__(self, payload: dict[str, Any]) -> None:
            super().__init__()
            self.payload = payload

    class Cancelled(Message, namespace="mcp_profile_form"):
        pass

    def __init__(self, *, profile: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._profile = dict(profile) if profile else None

    @property
    def is_edit(self) -> bool:
        return self._profile is not None

    def compose(self) -> ComposeResult:
        title = "Edit server" if self.is_edit else "Add local server (stdio)"
        yield Static(title, classes="destination-section", markup=False)
        profile = self._profile or {}
        yield Static("Profile id", classes="form-label")
        id_input = Input(value=str(profile.get("profile_id") or ""), id="mcp-form-id",
                         placeholder="docs-server")
        id_input.disabled = self.is_edit
        yield id_input
        yield Static("Command", classes="form-label")
        yield Input(value=str(profile.get("command") or ""), id="mcp-form-command",
                    placeholder="npx")
        yield Static("Args — one per line", classes="form-label")
        yield TextArea("\n".join(str(a) for a in profile.get("args") or []),
                       id="mcp-form-args")
        yield Static("Env — one KEY=value per line", classes="form-label")
        yield Static(
            "Secrets are never stored — reference them as KEY=$ENV_VAR and export "
            "the variable before connecting.",
            classes="ds-field-row", markup=False,
        )
        env_lines = [f"{k}={v}" for k, v in (profile.get("env_placeholders") or {}).items()]
        env_lines += [f"{k}={v}" for k, v in (profile.get("env_literals") or {}).items()]
        yield TextArea("\n".join(env_lines), id="mcp-form-env")
        yield Static("", id="mcp-form-error", classes="ds-field-row", markup=False)
        with Horizontal(classes="ds-toolbar"):
            yield Button("Save", id="mcp-form-save", classes="console-action-primary",
                         compact=True, tooltip="Validate and save this profile.")
            yield Button("Cancel", id="mcp-form-cancel", classes="console-action-secondary",
                         compact=True, tooltip="Discard changes.")

    def build_payload(self) -> dict[str, Any]:
        """Parse the form into the store's exact save-payload keys.

        Returns:
            Payload with profile_id/command/args/env_placeholders/env_literals.

        Raises:
            ValueError: A malformed env line (with its 1-based line number).
        """
        placeholders: dict[str, str] = {}
        literals: dict[str, str] = {}
        env_text = self.query_one("#mcp-form-env", TextArea).text
        for index, raw_line in enumerate(env_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = _ENV_LINE_RE.match(line)
            if not match:
                raise ValueError(f"Env line {index} must look like KEY=value.")
            key, value = match.group(1), match.group(2).strip()
            if _PLACEHOLDER_VALUE_RE.match(value):
                placeholders[key] = value
            else:
                literals[key] = value
        args = [line.strip() for line in
                self.query_one("#mcp-form-args", TextArea).text.splitlines() if line.strip()]
        return {
            "profile_id": self.query_one("#mcp-form-id", Input).value.strip(),
            "command": self.query_one("#mcp-form-command", Input).value.strip(),
            "args": args,
            "env_placeholders": placeholders,
            "env_literals": literals,
        }

    def show_error(self, text: str) -> None:
        self.query_one("#mcp-form-error", Static).update(text)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-form-save":
            event.stop()
            try:
                payload = self.build_payload()
            except ValueError as exc:
                self.show_error(str(exc))
                return
            self.post_message(self.SubmitRequested(payload))
        elif event.button.id == "mcp-form-cancel":
            event.stop()
            self.post_message(self.Cancelled())
