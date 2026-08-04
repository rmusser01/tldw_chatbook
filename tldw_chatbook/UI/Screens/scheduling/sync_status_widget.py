"""Sync status bar widget for the Schedules workbench."""

from __future__ import annotations

from textual.widgets import Button, Static
from textual.containers import Horizontal


class SyncStatusWidget(Horizontal):
    """Bar showing current owner, last sync timestamps, and latest error."""

    DEFAULT_CSS = """
    SyncStatusWidget {
        height: auto;
        padding: 1;
    }
    #scheduling-owner-local, #scheduling-owner-server {
        width: auto;
    }
    #scheduling-last-pull, #scheduling-last-push {
        width: auto;
    }
    #scheduling-last-pull {
        margin: 0 2 0 1;
    }
    #scheduling-last-push {
        margin-right: 1;
    }
    #scheduling-sync-error {
        width: 1fr;
        height: 1;
        overflow: hidden;
        color: $error;
    }
    #scheduling-clear-error {
        width: auto;
    }
    """

    def __init__(
        self,
        current_owner: str = "local",
        active_server_id: str | None = None,
        server_available: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.current_owner = current_owner
        self.active_server_id = active_server_id
        self.server_available = server_available

    def compose(self):
        local_variant = "primary" if self.current_owner == "local" else "default"
        server_variant = "primary" if self.current_owner.startswith("server:") else "default"
        local_btn = Button("Local", id="scheduling-owner-local", variant=local_variant)
        local_btn.tooltip = "Show schedules stored on this machine."
        server_btn = Button(
            "Server" if self.server_available else "Server (no connection)",
            id="scheduling-owner-server",
            variant=server_variant,
            disabled=not self.server_available,
        )
        server_btn.tooltip = self._server_tooltip()
        yield local_btn
        yield server_btn
        yield Static("Last pull: —", id="scheduling-last-pull")
        yield Static("Last push: —", id="scheduling-last-push")
        yield Static("", id="scheduling-sync-error")
        clear_btn = Button("Clear errors", id="scheduling-clear-error")
        clear_btn.tooltip = "Dismiss the current sync error messages."
        yield clear_btn

    def _server_tooltip(self) -> str:
        """Explain what the Server owner button points at."""
        if not self.server_available:
            return "No server connection available."
        return f"Show schedules synced with the server ({self.active_server_id})."

    def set_owner_state(
        self,
        current_owner: str,
        active_server_id: str | None,
        server_available: bool,
    ) -> None:
        """Update owner button labels, variants, and disabled state."""
        self.current_owner = current_owner
        self.active_server_id = active_server_id
        self.server_available = server_available

        local_btn = self.query_one("#scheduling-owner-local", Button)
        server_btn = self.query_one("#scheduling-owner-server", Button)

        local_btn.variant = "primary" if current_owner == "local" else "default"
        server_btn.variant = "primary" if current_owner.startswith("server:") else "default"
        server_btn.label = "Server" if server_available else "Server (no connection)"
        server_btn.disabled = not server_available
        server_btn.tooltip = self._server_tooltip()

    def update_status(
        self,
        last_pull_at: str | None,
        last_push_at: str | None,
        sync_errors: list[dict],
    ) -> None:
        self.query_one("#scheduling-last-pull", Static).update(
            f"Last pull: {last_pull_at or '—'}"
        )
        self.query_one("#scheduling-last-push", Static).update(
            f"Last push: {last_push_at or '—'}"
        )
        error_widget = self.query_one("#scheduling-sync-error", Static)
        if sync_errors:
            message = str(sync_errors[-1].get("message", ""))
            error_widget.update(message)
            # One line on screen; the full message stays available on hover.
            error_widget.tooltip = message
        else:
            error_widget.update("")
            error_widget.tooltip = None
        clear_button = self.query_one("#scheduling-clear-error", Button)
        clear_button.disabled = not sync_errors
