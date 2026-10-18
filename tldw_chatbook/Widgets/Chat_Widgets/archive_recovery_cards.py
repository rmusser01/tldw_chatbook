"""Native controls for app-owned archive evidence and exact one-shot consent."""

from __future__ import annotations

import time
import unicodedata

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Key
from textual.message import Message
from textual.widgets import Button, Select, Static

from tldw_chatbook.Chat.archive_recovery.contracts import RecoveryConfirmation
from tldw_chatbook.Chat.archive_recovery.runtime import ArchiveRecoveryPresentation

SCOPE_LABELS = {
    "chat_only": "Restore chat only",
    "chat_and_workspace": "Restore chat and workspace",
    "workspace_only": "Restore workspace",
}
SCOPE_COPY = {
    "chat_only": "The chat will be restored. Its workspace will remain archived.\nOpening the chat will still require restoring that workspace.",
    "chat_and_workspace": "The workspace and its other active chats will become visible.\nChats archived individually will stay archived.",
    "workspace_only": "The workspace and its other active chats will become visible.\nChats archived individually will stay archived.",
}


MAX_CONFIRMATION_TITLE_CHARS = 80


def safe_text(value: object) -> str:
    """Keep labels literal and remove terminal/control formatting."""
    return "".join(
        " " if c in "\n\r\t" else c
        for c in str(value or "")
        if c in "\n\r\t" or not unicodedata.category(c).startswith("C")
    )


class RecoveryActionButton(Button):
    """Activate locally before transcript key handlers consume Enter/Space."""

    def on_key(self, event: Key) -> None:
        if event.key in {"enter", "space"}:
            self.action_press()
            event.stop()
            event.prevent_default()


class ArchiveRecoveryResultsCard(Vertical):
    """Render a runtime record; never create actions from transcript JSON."""

    class ActionRequested(Message):
        def __init__(
            self,
            session_id: str,
            result_set_id: str,
            result_ref: str | None,
            action: str,
        ) -> None:
            super().__init__()
            self.session_id = session_id
            self.result_set_id = result_set_id
            self.result_ref = result_ref
            self.action = action

    def __init__(self, record: ArchiveRecoveryPresentation) -> None:
        super().__init__(classes="archive-recovery-card archive-recovery-results")
        self.record = record
        self._actions = {}

    def compose(self) -> ComposeResult:
        request = self.record.request
        scope = (
            "All local workspaces"
            if request.workspace_id is None
            else f"Workspace: {request.workspace_id}"
        )
        lower = (
            request.archived_after.isoformat() if request.archived_after else "any time"
        )
        upper = (
            request.archived_before.isoformat()
            if request.archived_before
            else "no upper bound"
        )
        yield Static(
            f"Archived chats · {safe_text(scope)}\nArchive dates: {lower} (inclusive) to {upper} (exclusive)",
            markup=False,
        )
        if not self.record.model_visible:
            yield Static(
                "Manual results · agent selection requires a fresh authorized search",
                markup=False,
            )
        for index, row in enumerate(self.record.projection["items"]):
            workspace = row["workspace"]
            workspace_name = (
                workspace["name"]
                if workspace
                else (
                    "Default"
                    if row["workspace_id"] is None
                    else "Workspace unavailable"
                )
            )
            chat_state = (
                "Chat archived" if row["conversation_archived"] else "Chat active"
            )
            workspace_state = (
                "Workspace archived"
                if workspace and workspace["archived"]
                else "Workspace active"
            )
            dates = []
            if row["conversation_archived"]:
                dates.append(
                    f"chat: {row['conversation_archived_at'] or 'unknown time'}"
                )
            if workspace and workspace["archived"]:
                dates.append(f"workspace: {workspace['archived_at'] or 'unknown time'}")
            basis = ", ".join(row["archive_match_basis"])
            with Vertical(classes="archive-recovery-row"):
                yield Static(
                    f"{index + 1}. {safe_text(row['title'])} · {safe_text(workspace_name)}\n{chat_state} · {workspace_state}\nArchive events: {safe_text(', '.join(dates) or 'unknown time')}\nMatch basis: {safe_text(basis or 'no proven dated match')}\n{safe_text(row['excerpt'])}",
                    markup=False,
                )
                if row["inactive_branch"]:
                    yield Static("Match found in inactive branch.", markup=False)
                if row["chronology_incomplete"]:
                    yield Static("Archive chronology incomplete", markup=False)
                    if (
                        request.archived_after or request.archived_before
                    ) and not basis:
                        yield Static(
                            "Archive time unknown; may match this date range.",
                            markup=False,
                        )
                for action, label in [("preview", "Preview"), ("restore", "Restore")]:
                    if action == "restore" and not row["allowed_scopes"]:
                        continue
                    widget_id = f"recovery-row-{index}-{action}"
                    self._actions[widget_id] = (row["result_ref"], action)
                    yield RecoveryActionButton(
                        label, id=widget_id, classes=f"recovery-row-{action}"
                    )
        if not self.record.projection["items"]:
            yield Static("No archived chats matched this search.", markup=False)
        if self.record.projection.get("cursor"):
            yield RecoveryActionButton("More results", id="recovery-more")
        yield RecoveryActionButton("Refresh", id="recovery-refresh")
        if (
            self.record.projection.get("unknown_times_present")
            and not request.include_unknown_archive_times
        ):
            yield RecoveryActionButton("Include unknown times", id="recovery-unknown")
        yield RecoveryActionButton("Search in Library", id="recovery-library")

    @on(Button.Pressed)
    def action_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        widget_id = event.button.id
        ref, action = self._actions.get(
            widget_id,
            (
                None,
                {
                    "recovery-more": "more",
                    "recovery-refresh": "refresh",
                    "recovery-unknown": "unknown",
                    "recovery-library": "library",
                }.get(widget_id),
            ),
        )
        if action:
            self.post_message(
                self.ActionRequested(
                    self.record.session_id,
                    self.record.projection["result_set_id"],
                    ref,
                    action,
                )
            )


class RecoveryDecisionButton(RecoveryActionButton):
    """Stamp identity at emission, before queued dispatch can see a new card."""

    binding: tuple[str, str] | None = None

    class Pressed(Button.Pressed):
        def __init__(self, button: RecoveryDecisionButton) -> None:
            super().__init__(button)
            self.binding = button.binding

    def press(self) -> RecoveryDecisionButton:
        if not self.disabled and self.display:
            self._start_active_affect()
            self.post_message(self.Pressed(self))
        return self


class ArchiveRecoveryConfirmCard(Vertical):
    """Fixed mounted controls; payload updates never schedule consent mounting."""

    class RecoveryDecided(Message):
        def __init__(
            self, session_id: str, request_id: str, allow: bool, scope: str | None
        ) -> None:
            super().__init__()
            self.session_id, self.request_id = session_id, request_id
            self.allow, self.scope = allow, scope

    class RenameRequested(Message):
        def __init__(self, session_id: str, request_id: str) -> None:
            super().__init__()
            self.session_id, self.request_id = session_id, request_id

    def __init__(self, confirmation: RecoveryConfirmation | None = None) -> None:
        super().__init__(classes="archive-recovery-card archive-recovery-confirmation")
        self.confirmation = confirmation
        self._consumed = False
        self.display = confirmation is not None

    def compose(self) -> ComposeResult:
        yield Static("", id="recovery-target", markup=False)
        yield Static("", id="recovery-effects", markup=False)
        yield Select(
            [(label, scope) for scope, label in SCOPE_LABELS.items()],
            id="recovery-scope",
            prompt="Choose exact scope",
        )
        yield RecoveryDecisionButton(
            "Confirm restore", id="recovery-confirm", disabled=True
        )
        yield RecoveryDecisionButton("Cancel", id="recovery-cancel")
        yield RecoveryDecisionButton("Restore as…", id="recovery-rename")

    def on_mount(self) -> None:
        self.set_confirmation(self.confirmation)

    def set_confirmation(self, card: RecoveryConfirmation | None) -> None:
        """Synchronously update already mounted, exact-target controls."""
        previous = self.confirmation
        self.confirmation = card
        self.display = card is not None
        for button in self.query(RecoveryDecisionButton):
            button.binding = (card.session_id, card.request_id) if card else None
        if card is None:
            self.query_one("#recovery-confirm", Button).disabled = True
            return
        if previous is None or previous.request_id != card.request_id:
            self._consumed = False
        scope = self.query_one("#recovery-scope", Select)
        if (
            previous is None
            or previous.request_id != card.request_id
            or scope.value is Select.NULL
        ):
            scope.set_options([(SCOPE_LABELS[s], s) for s in card.allowed_scopes])
            scope.value = card.selected_scope or (
                card.allowed_scopes[0] if len(card.allowed_scopes) == 1 else Select.NULL
            )
        title = safe_text(card.target.title)
        if len(title) > MAX_CONFIRMATION_TITLE_CHARS:
            title = title[: MAX_CONFIRMATION_TITLE_CHARS - 1] + "…"
        self.query_one("#recovery-confirm", Button).label = Text(
            f"Confirm restore · {title}"
        )
        workspace = safe_text(
            card.target.workspace.name if card.target.workspace else "Default"
        )
        name = (
            f"\nFinal workspace name: {safe_text(card.replacement_name)}"
            if card.replacement_name
            else ""
        )
        self.query_one("#recovery-target", Static).update(
            f"Restore {title}\nWorkspace: {workspace}{name}"
        )
        self.query_one("#recovery-rename", Button).display = bool(
            card.target.workspace and card.target.workspace.archived
        )
        self._sync_scope()

    @on(Select.Changed, "#recovery-scope")
    def scope_changed(self) -> None:
        self._sync_scope()

    def _sync_scope(self) -> None:
        card = self.confirmation
        scope = self.query_one("#recovery-scope", Select).value
        text = SCOPE_COPY.get(scope, "Choose the exact restoration scope.")
        if (
            card
            and scope == "chat_only"
            and not (card.target.workspace and card.target.workspace.archived)
        ):
            text = "The chat will be restored. Opening it is a separate action."
        self.query_one("#recovery-effects", Static).update(text)
        self.query_one("#recovery-confirm", Button).disabled = bool(
            not card
            or self._consumed
            or card.expires_at <= time.monotonic()
            or scope not in card.allowed_scopes
        )

    @on(Button.Pressed)
    def decide_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        card = self.confirmation
        if (
            not card
            or self._consumed
            or getattr(event, "binding", None) != (card.session_id, card.request_id)
        ):
            return
        if event.button.id == "recovery-rename":
            self.post_message(self.RenameRequested(card.session_id, card.request_id))
            return
        if card.expires_at <= time.monotonic():
            self._sync_scope()
            return
        scope = self.query_one("#recovery-scope", Select).value
        allow = event.button.id == "recovery-confirm"
        if event.button.id not in {"recovery-confirm", "recovery-cancel"} or (
            allow and scope not in card.allowed_scopes
        ):
            return
        self._consumed = True
        self._sync_scope()
        self.post_message(
            self.RecoveryDecided(
                card.session_id,
                card.request_id,
                allow,
                None if scope is Select.NULL else scope,
            )
        )
