# tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py
"""Inline add/edit form for local MCP server profiles (stdio-only)."""

from __future__ import annotations

import re
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.MCP.local_store import _looks_like_raw_secret_value
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
        """`warning` (I4 follow-up): the args secret-lint text computed at
        submit time, or None when the args are clean. Carried on the message
        because the in-form `#mcp-form-args-warning` Static is only durable
        on FAILED saves -- a successful save unmounts the whole form
        (`hide_form()`) sub-second, so the host must re-surface the warning
        as a toast alongside its own success notify for the user to ever
        see it."""

        def __init__(self, payload: dict[str, Any], warning: str | None = None) -> None:
            super().__init__()
            self.payload = payload
            self.warning = warning

    class Cancelled(Message, namespace="mcp_profile_form"):
        pass

    def __init__(self, *, profile: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._profile = dict(profile) if profile else None
        # A4: Save is state-driven -- disabled while either required field is
        # empty/whitespace, or while a submit this form already posted is
        # still awaiting the host's outcome (show_error()/unmount).
        self._submit_in_flight = False

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
        # A2: mcp-status-warning colors the secret-lint warning text (the
        # color rule itself has existed since T13 -- nothing applied the
        # class to this Static until now).
        yield Static(
            "", id="mcp-form-args-warning", classes="ds-field-row mcp-status-warning",
            markup=False,
        )
        yield Static("Env — one KEY=value per line", classes="form-label")
        yield Static(
            "Secrets are never stored — reference them as KEY=$ENV_VAR and export "
            "the variable before connecting.",
            classes="ds-field-row", markup=False,
        )
        env_lines = [f"{k}={v}" for k, v in (profile.get("env_placeholders") or {}).items()]
        env_lines += [f"{k}={v}" for k, v in (profile.get("env_literals") or {}).items()]
        yield TextArea("\n".join(env_lines), id="mcp-form-env")
        # A2: mcp-status-error colors the validation/save-failure text (the
        # color rule itself has existed since T13 -- nothing applied the
        # class to this Static until now).
        yield Static("", id="mcp-form-error", classes="ds-field-row mcp-status-error", markup=False)
        with Horizontal(classes="ds-toolbar"):
            yield Button("Save", id="mcp-form-save", classes="console-action-primary",
                         compact=True, tooltip="Validate and save this profile.")
            yield Button("Cancel", id="mcp-form-cancel", classes="console-action-secondary",
                         compact=True, tooltip="Discard changes.")

    def on_mount(self) -> None:
        # A4: `self.watch(...)` (not the `on_input_changed` message handler)
        # deliberately -- Reactive watchers registered this way run
        # SYNCHRONOUSLY inside the Input's own `value` setter (see
        # `textual.reactive._check_watchers`/`_watch`), whereas a message
        # handler only runs once the async message pump gets a turn (i.e.
        # after a `pilot.pause()`). The workbench's double-submit regression
        # test sets `#mcp-form-id`/`#mcp-form-command` `.value` directly and
        # then calls `Button.press()` twice with NO intervening pause --
        # `Button.press()` no-ops when `disabled` is still True at that exact
        # call, so an async-only refresh would leave Save permanently
        # disabled there and silently break that test. The synchronous watch
        # guarantees Save's enabled state is already correct by the time
        # `press()` runs, matching real typing (which always has pump turns
        # between keystrokes and the eventual click) with no observable
        # difference.
        self.watch(self.query_one("#mcp-form-id", Input), "value",
                   self._on_required_field_changed, init=False)
        self.watch(self.query_one("#mcp-form-command", Input), "value",
                   self._on_required_field_changed, init=False)
        self._refresh_save_enabled()

    def _on_required_field_changed(self, old_value: str, new_value: str) -> None:
        self._refresh_save_enabled()

    def _refresh_save_enabled(self) -> None:
        """A4: Save is enabled only once both required fields are filled in
        (edit mode's disabled-but-prefilled id Input still counts -- this
        checks `.value`, not `.disabled`) and no submit posted by this form
        is still awaiting the host's outcome. Called after compose (here, via
        `on_mount`), on every required-field change, and from `show_error()`
        so a failed save only re-arms Save when the fields are still valid.

        Review fix: the two disable-reasons (missing fields vs. a submit
        already in flight) get DISTINCT tooltips -- collapsing them into one
        `enabled` boolean previously left the "Enter a profile id and
        command first." tooltip showing even right after a valid submit,
        while both fields were still filled in, which is simply false.
        """
        try:
            save_button = self.query_one("#mcp-form-save", Button)
            has_id = bool(self.query_one("#mcp-form-id", Input).value.strip())
            has_command = bool(self.query_one("#mcp-form-command", Input).value.strip())
        except Exception:
            return
        fields_valid = has_id and has_command
        enabled = fields_valid and not self._submit_in_flight
        save_button.disabled = not enabled
        if enabled:
            save_button.tooltip = "Validate and save this profile."
        elif not fields_valid:
            save_button.tooltip = "Enter a profile id and command first."
        else:
            save_button.tooltip = "Save already in progress."

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
        """Surface an error and clear the in-flight flag so Save can re-arm.

        State-driven buttons: `on_button_pressed` marks a submit in-flight
        when a valid submit is posted; the host reporting failure through
        this method is what clears that (success unmounts the whole form
        instead). Routed through `_refresh_save_enabled()` (A4) rather than
        unconditionally re-enabling -- Save only re-arms if the required
        fields are STILL both filled in.
        """
        self.query_one("#mcp-form-error", Static).update(text)
        self._submit_in_flight = False
        self._refresh_save_enabled()

    def _args_secret_warning(self) -> str:
        """Spec §7: "warn when a secret-looking value appears in args
        (visible in `ps`); suggest env". Non-blocking -- unlike the env
        literal path (`_sanitize_env_literals()` in local_store.py, which
        RAISES for a secret-shaped literal), a stdio command's args are
        never persisted through that guard at all, so nothing stops a
        pasted `--api-key sk-live-...` from landing here undetected. Reuses
        the store's own secret-value shape check (`_looks_like_raw_secret_value`,
        imported rather than re-implemented so the two never drift) against
        each non-blank args line and each whitespace/`=`-separated token
        within it, covering both a value on its own line and a `--flag=value`
        line.

        Returns:
            A one-line warning naming the first offending line (1-based, one
            index per physical line -- including blanks -- matching how the
            line-numbered env-parse errors above count), or "" when clean.
        """
        text = self.query_one("#mcp-form-args", TextArea).text
        for index, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            candidates = [line, *line.split()]
            if "=" in line:
                candidates.append(line.partition("=")[2].strip())
            if any(_looks_like_raw_secret_value(candidate) for candidate in candidates):
                return (
                    f"Warning: args line {index} looks like a secret — it will be "
                    "visible in process listings; pass it via env ($VAR) instead."
                )
        return ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-form-save":
            event.stop()
            try:
                payload = self.build_payload()
            except ValueError as exc:
                self.show_error(str(exc))
                return
            # Non-blocking (I4/spec §7): a secret-shaped arg does not stop
            # the submit below, it only surfaces a warning alongside it.
            # The in-form Static covers the failed-save case (form stays
            # up); the SubmitRequested `warning` carries the same text to
            # the host, whose success path unmounts this form sub-second --
            # the host re-surfaces it as a toast there instead.
            warning = self._args_secret_warning()
            self.query_one("#mcp-form-args-warning", Static).update(warning)
            # Disable while the host's save is in flight -- Button.press()
            # gates on `disabled`, so a real second click can't even post
            # another Pressed. (A Pressed already queued before this line
            # still gets through; the workbench's in-flight guard is the
            # authoritative dedupe for that window.) Routed through A4's
            # `_refresh_save_enabled()` rather than a bare `disabled = True`
            # so the in-flight flag (not just the button) reflects reality.
            self._submit_in_flight = True
            self._refresh_save_enabled()
            self.post_message(self.SubmitRequested(payload, warning or None))
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
    #mcp-import-list { height: auto; min-height: 0; }
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
        # A2: mcp-status-error colors the invalid-JSON preview error text
        # (the color rule itself has existed since T13 -- nothing applied
        # the class to this Static until now), mirroring #mcp-form-error.
        yield Static(
            "", id="mcp-import-error", classes="ds-field-row mcp-status-error", markup=False,
        )
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
