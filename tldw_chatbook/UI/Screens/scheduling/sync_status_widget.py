"""Sync status bar widget for the Schedules workbench."""

from __future__ import annotations

from textual.widgets import Button, Static
from textual.containers import Horizontal


class SyncStatusWidget(Horizontal):
    """Bar showing current owner, last sync timestamps, and latest error.

    task-23105: when the owner is Local and no server connection exists,
    the server plumbing (owner buttons, pull/push timestamps) collapses
    to a single honest line, and Clear only appears once an error exists
    (a hidden control carries its state better than a color-only
    disabled one).
    """

    BUNDLED_CSS = """
    SyncStatusWidget {
        height: auto;
        padding: 1;
    }
    #scheduling-owner-local, #scheduling-owner-server {
        width: auto;
    }
    #scheduling-sync-local-note {
        width: auto;
        color: $text-muted;
    }
    /* task-2723: without margins these Statics render flush against each
       other — "Last pull: —Last push: —<error text>" read as one run. */
    #scheduling-last-pull, #scheduling-last-push {
        width: auto;
        margin-left: 2;
    }
    #scheduling-sync-error {
        width: 1fr;
        color: $error;
        margin-left: 2;
        text-wrap: nowrap;
        text-overflow: ellipsis;
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
        server_variant = (
            "primary" if self.current_owner.startswith("server:") else "default"
        )
        server_label = f"Server ({self.active_server_id or 'unavailable'})"
        server_tooltip = (
            "Use the connected server as the Schedules owner."
            if self.server_available
            else "Connect a scheduling server before switching Schedules ownership."
        )
        yield Button(
            "Local",
            id="scheduling-owner-local",
            variant=local_variant,
            tooltip="Use local storage as the Schedules owner.",
        )
        yield Button(
            server_label,
            id="scheduling-owner-server",
            variant=server_variant,
            disabled=not self.server_available,
            tooltip=server_tooltip,
        )
        yield Static("", id="scheduling-sync-local-note")
        yield Static("Last pull: —", id="scheduling-last-pull")
        yield Static("Last push: —", id="scheduling-last-push")
        yield Static("", id="scheduling-sync-error")
        yield Button(
            "Clear",
            id="scheduling-clear-error",
            tooltip="Clear the latest scheduling sync error.",
        )

    def on_mount(self) -> None:
        """Apply the local-owner collapse and hide Clear until needed."""
        self._apply_collapse()
        self.query_one("#scheduling-clear-error", Button).display = False

    def _apply_collapse(self) -> None:
        """Collapse server plumbing to one line for local-only setups.

        Deliberately NOT collapsed (task-23105 review F11): the error
        Static and the Clear button. Honesty beats compactness -- a
        persisted sync error from a since-removed server must stay
        visible and clearable on a now-local-only profile, so collapsed
        mode shows local note + error + Clear whenever an error exists.
        """
        collapsed = self.current_owner == "local" and not self.server_available
        for selector in (
            "#scheduling-owner-local",
            "#scheduling-owner-server",
            "#scheduling-last-pull",
            "#scheduling-last-push",
        ):
            self.query_one(selector).display = not collapsed
        note = self.query_one("#scheduling-sync-local-note", Static)
        note.display = collapsed
        note.update(
            "Local schedules — no scheduling server connected; sync is off."
            if collapsed
            else ""
        )

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
        server_btn.variant = (
            "primary" if current_owner.startswith("server:") else "default"
        )
        server_btn.label = f"Server ({active_server_id or 'unavailable'})"
        server_btn.disabled = not server_available
        server_btn.tooltip = (
            "Use the connected server as the Schedules owner."
            if server_available
            else "Connect a scheduling server before switching Schedules ownership."
        )
        self._apply_collapse()

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
        # Hidden until an error exists (task-23105): visibility carries the
        # state, instead of a color-only disabled button.
        clear_button = self.query_one("#scheduling-clear-error", Button)
        clear_button.display = bool(sync_errors)
        clear_button.disabled = not sync_errors
