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
    #scheduling-sync-error {
        width: 1fr;
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
        server_label = f"Server ({self.active_server_id or 'unavailable'})"
        yield Button("Local", id="scheduling-owner-local", variant=local_variant)
        yield Button(server_label, id="scheduling-owner-server", variant=server_variant, disabled=not self.server_available)
        yield Static("Last pull: —", id="scheduling-last-pull")
        yield Static("Last push: —", id="scheduling-last-push")
        yield Static("", id="scheduling-sync-error")
        yield Button("Clear", id="scheduling-clear-error")

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
        server_btn.label = f"Server ({active_server_id or 'unavailable'})"
        server_btn.disabled = not server_available

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
            error_widget.update(str(sync_errors[-1].get("message", "")))
        else:
            error_widget.update("")
        clear_button = self.query_one("#scheduling-clear-error", Button)
        clear_button.disabled = not sync_errors
