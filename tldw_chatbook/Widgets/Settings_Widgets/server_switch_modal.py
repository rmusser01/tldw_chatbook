"""Modal for switching the runtime source between local and a server.

The modal only collects the decision (which server, or local) and reports
reachability honestly; the Settings screen performs the actual activation
(config persistence, runtime-policy rebind, Sync v2 enrollment) so this
surface stays side-effect free until the user confirms.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
from urllib.parse import urlsplit

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static


class ServerSwitchModal(ModalScreen[Optional[dict]]):
    """Choose the runtime source: this device only, or a tldw server."""

    BINDINGS = [("escape", "dismiss", "Cancel")]

    DEFAULT_CSS = """
    ServerSwitchModal {
        align: center middle;
    }
    #server-switch-modal {
        width: 84;
        max-width: 95%;
        height: auto;
        max-height: 80%;
        border: round $primary;
        background: $surface;
        padding: 1 2;
    }
    #server-switch-modal .server-switch-row {
        height: 3;
        margin-bottom: 1;
    }
    #server-switch-modal .server-switch-label {
        width: 14;
        min-width: 14;
        padding-top: 1;
        color: $text-muted;
    }
    #server-switch-modal Input {
        width: 1fr;
    }
    #server-switch-current,
    #server-switch-status {
        height: auto;
        min-height: 1;
        margin-bottom: 1;
        color: $text-muted;
    }
    #server-switch-actions {
        height: 3;
        align: right middle;
    }
    #server-switch-actions Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        *,
        current_source: str = "local",
        current_server_label: str | None = None,
        current_base_url: str = "",
        current_auth_token: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._current_source = (current_source or "local").strip().lower()
        self._current_server_label = (current_server_label or "").strip()
        self._initial_base_url = current_base_url or ""
        self._initial_auth_token = current_auth_token or ""

    def compose(self) -> ComposeResult:
        if self._current_source == "server" and self._current_server_label:
            current_copy = f"Current source: server ({self._current_server_label})"
        elif self._current_server_label:
            current_copy = (
                f"Current source: local. Server {self._current_server_label} is "
                "configured but not active."
            )
        else:
            current_copy = "Current source: local. No server is configured."

        with Vertical(id="server-switch-modal"):
            yield Static("Switch Runtime Source", classes="destination-section")
            yield Static(current_copy, id="server-switch-current", markup=False)
            with Horizontal(classes="server-switch-row"):
                yield Static("Server URL", classes="server-switch-label")
                yield Input(
                    value=self._initial_base_url,
                    placeholder="http://127.0.0.1:8000",
                    id="server-switch-url",
                )
            with Horizontal(classes="server-switch-row"):
                yield Static("API token", classes="server-switch-label")
                yield Input(
                    value=self._initial_auth_token,
                    placeholder="X-API-KEY value",
                    password=True,
                    id="server-switch-token",
                )
            yield Static(
                "Test the connection before activating; activation also "
                "prepares the Sync v2 profile for this device.",
                id="server-switch-status",
                markup=False,
            )
            with Horizontal(id="server-switch-actions"):
                yield Button(
                    "Test Connection",
                    id="server-switch-test",
                    compact=True,
                    classes="destination-action-button console-action-secondary",
                    tooltip="Check reachability and whether the token is accepted.",
                )
                yield Button(
                    "Use Local",
                    id="server-switch-local",
                    compact=True,
                    classes="destination-action-button console-action-secondary",
                    tooltip="Keep all data on this device only.",
                )
                yield Button(
                    "Activate Server",
                    id="server-switch-activate",
                    compact=True,
                    classes="destination-action-button console-action-primary",
                    tooltip="Bind this server as the runtime source and prepare Sync v2.",
                )
                yield Button(
                    "Cancel",
                    id="server-switch-cancel",
                    compact=True,
                    classes="destination-action-button console-action-secondary",
                    tooltip="Close without changing the runtime source.",
                )

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#server-switch-status", Static).update(text)
        except QueryError:
            pass

    def _entered_url(self) -> str:
        try:
            return self.query_one("#server-switch-url", Input).value.strip().rstrip("/")
        except QueryError:
            return ""

    def _validated_root_url(self) -> str | None:
        """Return the entered URL if it is a usable server ROOT, else None.

        The runtime API client appends /api/v1/... itself, so the URL must be
        scheme + host[:port] with no path/query/fragment; anything else is
        rejected here instead of producing confusing joined endpoints later.
        """
        from tldw_chatbook.Utils.input_validation import validate_url

        url = self._entered_url()
        if not url:
            self._set_status("Enter a server URL first.")
            return None
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            self._set_status("Use a full URL with scheme, e.g. http://127.0.0.1:8000")
            return None
        if parts.path.strip("/") or parts.query or parts.fragment:
            self._set_status(
                "Enter the server root only (no path) — e.g. http://host:8000; "
                "the client adds /api/v1/... itself."
            )
            return None
        if not validate_url(url):
            self._set_status("That URL did not pass validation; check it and retry.")
            return None
        return url

    def _entered_token(self) -> str:
        try:
            return self.query_one("#server-switch-token", Input).value.strip()
        except QueryError:
            return ""

    @on(Button.Pressed, "#server-switch-test")
    def handle_test_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        url = self._validated_root_url()
        if not url:
            return
        self._set_status(f"Testing {url} ...")
        self._run_connection_test(url, self._entered_token())

    @work(exclusive=True, group="server-switch-test")
    async def _run_connection_test(self, url: str, token: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                reach = await client.get(f"{url}/docs")
                auth_state = "not checked (no token entered)"
                if token:
                    probe = await client.post(
                        f"{url}/api/v1/sync/send",
                        headers={"X-API-KEY": token},
                        json={},
                    )
                    if probe.status_code in {401, 403}:
                        auth_state = f"token REJECTED (HTTP {probe.status_code})"
                    elif probe.status_code == 404:
                        auth_state = "could not verify (sync endpoint missing, HTTP 404)"
                    elif probe.status_code >= 500:
                        auth_state = f"could not verify (server error HTTP {probe.status_code})"
                    else:
                        # 2xx, or 422 schema rejection — the token itself passed.
                        auth_state = f"token accepted (HTTP {probe.status_code})"
        except httpx.HTTPError as exc:
            self._set_status(f"Unreachable: {exc.__class__.__name__}: {exc}")
            return
        except Exception as exc:  # never leave the status stuck on "Testing..."
            self._set_status(f"Test failed: {exc.__class__.__name__}: {exc}")
            return
        self._set_status(
            f"Reachable (HTTP {reach.status_code}). Auth: {auth_state}."
        )

    @on(Button.Pressed, "#server-switch-activate")
    def handle_activate_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        url = self._validated_root_url()
        if not url:
            return
        self.dismiss(
            {
                "action": "activate",
                "base_url": url,
                "auth_token": self._entered_token(),
            }
        )

    @on(Button.Pressed, "#server-switch-local")
    def handle_local_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss({"action": "local"})

    @on(Button.Pressed, "#server-switch-cancel")
    def handle_cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)
