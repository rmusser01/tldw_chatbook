# tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py
"""Inline add/edit form for local MCP server profiles (stdio-only)."""

from __future__ import annotations

import re
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.MCP.mcp_import import ImportCandidate, parse_mcp_servers_json

_ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
# Balanced forms only: $VAR or ${VAR}. Unbalanced values ($VAR} / ${VAR)
# deliberately fall through to the literals path, where the store's own
# validation gives honest copy if they turn out to be secret-shaped.
_PLACEHOLDER_VALUE_RE = re.compile(
    r"^\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)$"
)


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
        """Surface an error and re-enable Save so the user can retry.

        State-driven buttons: `on_button_pressed` disables Save when a valid
        submit is posted; the host reporting failure through this method is
        what re-arms it (success unmounts the whole form instead).
        """
        self.query_one("#mcp-form-error", Static).update(text)
        self.query_one("#mcp-form-save", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-form-save":
            event.stop()
            try:
                payload = self.build_payload()
            except ValueError as exc:
                self.show_error(str(exc))
                return
            # Disable while the host's save is in flight -- Button.press()
            # gates on `disabled`, so a real second click can't even post
            # another Pressed. (A Pressed already queued before this line
            # still gets through; the workbench's in-flight guard is the
            # authoritative dedupe for that window.)
            event.button.disabled = True
            self.post_message(self.SubmitRequested(payload))
        elif event.button.id == "mcp-form-cancel":
            event.stop()
            self.post_message(self.Cancelled())


class MCPImportPanel(Vertical):
    """Paste-or-file import of Claude-Desktop-style `mcpServers` JSON.

    Mirrors `MCPProfileForm`'s state-driven-button/single-error-Static
    structure: Preview parses the pasted/loaded text into candidates (one
    plain-text Static per candidate under `#mcp-import-list`, `markup=False`
    so a hostile server name/command can't inject Rich markup -- same
    defense as the overview table in mcp_servers_mode.py), and only a
    successful preview arms Import.
    """

    DEFAULT_CSS = """
    MCPImportPanel { height: auto; min-height: 0; }
    #mcp-import-text { height: 8; min-height: 4; }
    """

    class FileRequested(Message, namespace="mcp_import_panel"):
        pass

    class ImportRequested(Message, namespace="mcp_import_panel"):
        def __init__(self, candidates: list[ImportCandidate]) -> None:
            super().__init__()
            self.candidates = candidates

    class Cancelled(Message, namespace="mcp_import_panel"):
        pass

    def __init__(self, *, existing_ids: set[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._existing_ids = set(existing_ids or ())
        self._candidates: list[ImportCandidate] = []

    def compose(self) -> ComposeResult:
        yield Static("Import from mcpServers JSON", classes="destination-section", markup=False)
        yield Static(
            "Paste a Claude-Desktop-style {\"mcpServers\": ...} config, or load one from a "
            "file. Secret-shaped values are never imported as literals -- they become "
            "placeholders you export before connecting.",
            classes="ds-field-row", markup=False,
        )
        yield TextArea("", id="mcp-import-text")
        with Horizontal(classes="ds-toolbar"):
            yield Button("From file…", id="mcp-import-file", classes="console-action-secondary",
                         compact=True, tooltip="Load mcpServers JSON from a file.")
            yield Button("Preview", id="mcp-import-preview", classes="console-action-secondary",
                         compact=True, tooltip="Parse the text and preview the servers it would import.")
        yield Static("", id="mcp-import-error", classes="ds-field-row", markup=False)
        yield Vertical(id="mcp-import-list")
        with Horizontal(classes="ds-toolbar"):
            apply_button = Button(
                "Import 0 servers", id="mcp-import-apply", classes="console-action-primary",
                compact=True, tooltip="Save every previewed candidate as a local profile.",
            )
            apply_button.disabled = True
            yield apply_button
            yield Button("Cancel", id="mcp-import-cancel", classes="console-action-secondary",
                         compact=True, tooltip="Discard and close.")

    def set_file_text(self, text: str) -> None:
        """Populate the paste area from a file the workbench just read."""
        self.query_one("#mcp-import-text", TextArea).text = text

    async def _preview(self) -> None:
        text = self.query_one("#mcp-import-text", TextArea).text
        error = self.query_one("#mcp-import-error", Static)
        apply_button = self.query_one("#mcp-import-apply", Button)
        try:
            candidates = parse_mcp_servers_json(text, existing_ids=self._existing_ids)
        except ValueError as exc:
            self._candidates = []
            error.update(str(exc))
            apply_button.disabled = True
            await self._render_candidates()
            return
        self._candidates = candidates
        error.update("")
        noun = "server" if len(candidates) == 1 else "servers"
        apply_button.label = f"Import {len(candidates)} {noun}"
        apply_button.disabled = False
        await self._render_candidates()

    async def _render_candidates(self) -> None:
        container = self.query_one("#mcp-import-list", Vertical)
        await container.remove_children()
        widgets: list[Static] = []
        for candidate in self._candidates:
            lines = [f"{candidate.profile_id} — {candidate.command}"]
            lines.extend(f"  {warning}" for warning in candidate.warnings)
            widgets.append(Static("\n".join(lines), classes="ds-field-row", markup=False))
        if widgets:
            await container.mount_all(widgets)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "mcp-import-file":
            event.stop()
            self.post_message(self.FileRequested())
        elif button_id == "mcp-import-preview":
            event.stop()
            await self._preview()
        elif button_id == "mcp-import-apply":
            event.stop()
            if not self._candidates:
                return
            # State-driven, same discipline as MCPProfileForm's Save: disable
            # so a real second click can't post another ImportRequested --
            # the workbench's own in-flight guard is the authoritative dedupe.
            event.button.disabled = True
            self.post_message(self.ImportRequested(list(self._candidates)))
        elif button_id == "mcp-import-cancel":
            event.stop()
            self.post_message(self.Cancelled())
