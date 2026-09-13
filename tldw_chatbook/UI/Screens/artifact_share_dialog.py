# artifact_share_dialog.py
"""Modal dialog for starting an artifact share session."""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    RadioButton,
    RadioSet,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection

_CONFIRM_PHRASE = "share"
#: Length bounds enforced at submission (Qodo #2): generous for real use,
#: small enough to keep manifests, headers, and the page sane.
_SHARE_NAME_MAX = 200
_USERNAME_MAX = 64
_PASSWORD_MAX = 256


def _option_title(record: dict[str, Any]) -> str:
    name = str(record.get("name") or "Unnamed artifact")
    if record.get("file_path"):
        return name
    return f"{name}  (no exported bundle on disk)"


class ArtifactShareDialog(ModalScreen[dict | None]):
    """Choose artifacts and share options; dismisses with an options dict or None."""

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Cancel", show=False)]

    # Styling lives in DEFAULT_CSS, not BUNDLED_CSS, on purpose: BUNDLED_CSS
    # from every widget is merged into the generated boot bundle
    # (css/widget_defaults_self.tcss), and both the boot-parsed byte budget
    # and the bare-type rule census (Tests/Performance/
    # test_boot_css_byte_budget.py, test_textual_css_fastpath.py) sit at zero
    # headroom. DEFAULT_CSS parses when this dialog mounts — runtime, not
    # boot — so the share dialog adds zero stylesheet-budget footprint.
    DEFAULT_CSS = """
    ArtifactShareDialog { align: center middle; background: $background 70%; }
    ArtifactShareDialog > VerticalScroll {
        width: 76; max-width: 96%; height: auto; max-height: 90%;
        background: $surface; border: solid $primary; padding: 1 2;
    }
    ArtifactShareDialog Input, ArtifactShareDialog RadioSet, ArtifactShareDialog SelectionList {
        margin-bottom: 1;
    }
    ArtifactShareDialog #share-dialog-status { color: $warning; }
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        active_share_notice: str | None = None,
    ) -> None:
        super().__init__()
        self._records = list(records)
        self._active_share_notice = active_share_notice

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static("Share artifacts", markup=False)
            yield Static(
                "Recipients browse a temporary web page and download only the "
                "selected bundles. Traffic is plain HTTP: a password is an access "
                "gate, not encryption. Stop sharing to revoke access.",
                markup=False,
            )
            if self._active_share_notice:
                yield Static(self._active_share_notice, markup=False, id="share-active-note")
            yield Static("Share name (page title)", markup=False)
            yield Input(placeholder="Shared artifacts", id="share-name")
            yield Static("Artifacts", markup=False)
            yield SelectionList(
                *(
                    Selection(
                        _option_title(record),
                        str(record.get("id")),
                        disabled=not bool(record.get("file_path")),
                    )
                    for record in self._records
                ),
                id="share-artifact-list",
            )
            yield Checkbox("Require a password (single shared login)", False, id="share-auth-toggle")
            yield Input(placeholder="Username", id="share-username")
            yield Input(placeholder="Password", password=True, id="share-password")
            yield Static("Who can reach it", markup=False)
            with RadioSet(id="share-bind"):
                yield RadioButton("This computer only (localhost)", value=True, id="share-bind-loopback")
                yield RadioButton("Local network (all interfaces)", id="share-bind-lan")
            yield Static("Port (blank = pick automatically)", markup=False)
            yield Input(placeholder="auto", id="share-port")
            yield Static(
                "Sharing without a password on the local network exposes these "
                f"artifacts to everyone on that network. Type '{_CONFIRM_PHRASE}' "
                "to confirm.",
                markup=False,
                id="share-confirm-label",
            )
            yield Input(placeholder=_CONFIRM_PHRASE, id="share-confirm")
            yield Static("", markup=False, id="share-dialog-status")
            with Horizontal():
                yield Button("Start sharing", id="share-start", variant="primary")
                yield Button("Cancel", id="share-cancel")

    def on_mount(self) -> None:
        self._sync_auth_visibility(False)
        self._sync_confirm_visibility()

    def _selected_records(self) -> list[dict[str, Any]]:
        selected_ids = {
            str(value) for value in self.query_one("#share-artifact-list", SelectionList).selected
        }
        return [record for record in self._records if str(record.get("id")) in selected_ids]

    def _bind_choice(self) -> str:
        return (
            "0.0.0.0"
            if self.query_one("#share-bind-lan", RadioButton).value
            else "127.0.0.1"
        )

    def _port_value(self) -> int:
        raw = self.query_one("#share-port", Input).value.strip()
        if not raw:
            return 0
        try:
            port = int(raw)
        except ValueError:
            return -1
        return port if 1 <= port <= 65535 else -1

    def _auth_enabled(self) -> bool:
        return self.query_one("#share-auth-toggle", Checkbox).value

    def _sync_auth_visibility(self, enabled: bool) -> None:
        self.query_one("#share-username", Input).display = enabled
        self.query_one("#share-password", Input).display = enabled

    def _sync_confirm_visibility(self) -> None:
        needs_confirm = self._bind_choice() == "0.0.0.0" and not self._auth_enabled()
        self.query_one("#share-confirm-label", Static).display = needs_confirm
        self.query_one("#share-confirm", Input).display = needs_confirm

    def _status(self, message: str) -> None:
        self.query_one("#share-dialog-status", Static).update(message)

    def action_cancel(self) -> None:
        """Dismiss the dialog with None (no share started)."""
        self.dismiss(None)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "share-auth-toggle":
            self._sync_auth_visibility(event.value)
            self._sync_confirm_visibility()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._sync_confirm_visibility()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "share-cancel":
            self.dismiss(None)
            return
        if event.button.id != "share-start":
            return
        selected = self._selected_records()
        if not selected:
            self._status("Select at least one artifact to share.")
            return
        share_name = self.query_one("#share-name", Input).value.strip()
        username = self.query_one("#share-username", Input).value.strip()
        password = self.query_one("#share-password", Input).value
        if self._auth_enabled() and (not username or not password):
            self._status("Enter both a username and a password, or disable the password.")
            return
        # Qodo #14: Basic auth splits on the first ':', so a colon username
        # can never authenticate; refuse it here with actionable copy.
        if self._auth_enabled() and ":" in username:
            self._status("Username cannot contain ':' -- pick a different username.")
            return
        # Qodo #2: bound the free-text fields before they reach manifests,
        # headers, and the served page.
        if self._auth_enabled() and len(username) > _USERNAME_MAX:
            self._status(f"Username must be {_USERNAME_MAX} characters or fewer.")
            return
        if self._auth_enabled() and len(password) > _PASSWORD_MAX:
            self._status(f"Password must be {_PASSWORD_MAX} characters or fewer.")
            return
        if len(share_name) > _SHARE_NAME_MAX:
            self._status(f"Share name must be {_SHARE_NAME_MAX} characters or fewer.")
            return
        port = self._port_value()
        if port < 0:
            self._status("Port must be a number between 1 and 65535, or blank for automatic.")
            return
        if self._bind_choice() == "0.0.0.0" and not self._auth_enabled():
            if self.query_one("#share-confirm", Input).value.strip() != _CONFIRM_PHRASE:
                self._status(
                    f"Type '{_CONFIRM_PHRASE}' to share without a password on the local network."
                )
                return
        self.dismiss(
            {
                "selected_records": selected,
                "share_name": share_name or "Shared artifacts",
                "username": username if self._auth_enabled() else "",
                "password": password if self._auth_enabled() else "",
                "bind": self._bind_choice(),
                "port": port,
            }
        )
