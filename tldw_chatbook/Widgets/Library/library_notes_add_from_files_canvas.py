"""Flat retained canvas for authority choice, reviewed effects, and recovery.

The status/authority header and nearest valid action stay outside the bounded
scroll body so a 60x20 terminal never hides what the user is about to do.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LibraryNotesLastingSyncSnapshot,
)


class LibraryNotesAddFromFilesCanvas(Vertical):
    """Render one message-only Add from files workflow."""

    class RelationshipRequested(Message):
        def __init__(self, relationship: str) -> None:
            super().__init__()
            self.relationship = relationship

    class SetupChanged(Message):
        def __init__(self, field: str, value: str) -> None:
            super().__init__()
            self.field = field
            self.value = value

    class FolderRequested(Message):
        pass

    class CheckRequested(Message):
        pass

    class ApplyRequested(Message):
        pass

    class AttentionChoiceRequested(Message):
        def __init__(self, item_id: str, choice: str) -> None:
            super().__init__()
            self.item_id = item_id
            self.choice = choice

    class ActivateRequested(Message):
        pass

    class PageRequested(Message):
        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class BackRequested(Message):
        pass

    def __init__(
        self, snapshot: LibraryNotesLastingSyncSnapshot, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.snapshot = snapshot
        self.add_class("library-notes-lasting-sync-canvas")

    def compose(self) -> ComposeResult:
        yield Static(
            "Library notes · Add from files · Choose authority before changes",
            id="notes-sync-authority",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            self.snapshot.status_line,
            id="notes-sync-status",
            classes="library-notes-lasting-status",
            markup=False,
        )
        with VerticalScroll(id="notes-sync-body"):
            yield from self._compose_phase()
        if self._expects_body_overflow():
            yield Static(
                "Additional setup content is scrollable."
                if self.snapshot.phase == "configure"
                else "Additional reviewed effects are scrollable.",
                id="notes-sync-fold-hint",
                classes="library-disabled-reason",
                markup=False,
            )
        with Horizontal(id="notes-sync-pinned-actions", classes="ds-toolbar"):
            yield from self._compose_pinned_actions()

    def _expects_body_overflow(self) -> bool:
        """Name scrollability only for phases whose bounded body can overflow."""

        if self.snapshot.phase == "configure":
            return True
        if self.snapshot.phase != "review":
            return False
        review = self.snapshot.review
        return (
            review.page_count > 1
            or len(review.rows) > 1
            or any(row.choices for row in review.rows)
        )

    def _compose_phase(self) -> ComposeResult:
        phase = self.snapshot.phase
        if phase == "choose":
            yield Static(
                "Choose the relationship before selecting a file or folder.",
                classes="destination-purpose",
                markup=False,
            )
            yield Static(
                "Import once — Copy files into Notes. Later changes to the originals are not tracked.",
                markup=False,
            )
            yield Static(
                "Keep a folder synced — Create a lasting connection. Changes continue between the folder and Notes.",
                markup=False,
            )
            lasting = self.snapshot.lasting_available
            keep = Button(
                "Keep a folder synced" if lasting else "○ Keep a folder synced",
                id="notes-add-keep-synced",
                classes="library-canvas-action",
                compact=True,
            )
            # The chooser remains physically actionable while inert so it can
            # explain the cutover gate instead of becoming unreadable.
            keep.tooltip = (
                None
                if lasting
                else "Unavailable until the reviewed lasting-sync cutover."
            )
            yield keep
            if not lasting:
                yield Static(
                    "Unavailable until the reviewed lasting-sync cutover. Nearest valid action: Import once.",
                    classes="library-disabled-reason",
                    markup=False,
                )
            return
        if phase == "configure":
            setup = self.snapshot.setup
            yield Static(
                "Keep a folder synced",
                classes="destination-section",
                markup=False,
            )
            yield Static(
                "Chatbook watches only while running. Startup and manual checks still reconcile; conflicts and deletions pause for review. Paths and recovery data stay on this device.",
                classes="destination-purpose",
                markup=False,
            )
            yield Static("Display name", markup=False)
            yield Input(
                value=setup.display_name,
                placeholder="Folder label in Notes",
                id="notes-sync-display-name",
            )
            yield Static("Folder", markup=False)
            yield Static(
                setup.folder or "No folder selected",
                id="notes-sync-folder-summary",
                classes="destination-purpose",
                markup=False,
            )
            yield Button(
                "Choose folder…",
                id="notes-sync-folder-choose",
                classes="library-canvas-action",
                compact=True,
            )
            yield Static("Notes destination", markup=False)
            with Horizontal(classes="ds-toolbar"):
                yield Button(
                    "Local Library notes (selected)",
                    id="notes-sync-destination-local",
                    classes="library-canvas-action",
                    compact=True,
                )
                yield Button(
                    "○ Server notes",
                    id="notes-sync-destination-server",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=True,
                    tooltip=setup.server_disabled_reason,
                )
            yield Static(
                setup.server_disabled_reason,
                id="notes-sync-server-disabled-reason",
                classes="library-disabled-reason",
                markup=False,
            )
            yield Static("Direction", markup=False)
            with Horizontal(id="notes-sync-directions", classes="ds-toolbar"):
                for value, label in (
                    ("bidirectional", "⇄ Both ways"),
                    ("folder_to_notes", "→ Folder to Notes"),
                    ("notes_to_folder", "← Notes to folder"),
                ):
                    button = Button(
                        f"✓ {label}" if setup.direction == value else label,
                        name=value,
                        id=f"notes-sync-direction-{value.replace('_', '-')}",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield button
            yield Static(
                setup.validation_message,
                id="notes-sync-validation",
                classes="library-disabled-reason",
                markup=False,
            )
            return
        if phase in {"checking", "activating"}:
            verb = "Checking" if phase == "checking" else "Activating"
            yield Static(
                f"◌ {verb}. No unreviewed action is being hidden.",
                classes="destination-purpose",
                markup=False,
            )
            return
        if phase == "review":
            review = self.snapshot.review
            yield Static(
                f"{review.safe_count} safe · {review.attention_count} need attention · "
                f"{review.skip_count} skipped · {review.managed_count} managed placements",
                id="notes-sync-review-summary",
                markup=False,
            )
            if review.stale:
                yield Static(
                    "⚠ This review is stale. Check again before applying.",
                    classes="library-disabled-reason",
                    markup=False,
                )
            for index, row in enumerate(review.rows):
                with Vertical(
                    id=f"notes-sync-review-row-{index}",
                    classes="library-notes-sync-review-row",
                ):
                    yield Static(
                        f"{row.category.title()} · {row.item_id}",
                        classes="destination-section",
                        markup=False,
                    )
                    yield Static(row.effect, markup=False)
                    if row.choices:
                        yield Static(
                            "Conflict and deletion choices are unavailable in this release.",
                            classes="library-disabled-reason",
                            markup=False,
                        )
                        with Vertical(classes="library-notes-sync-attention-actions"):
                            for choice_index, choice in enumerate(row.choices):
                                yield Button(
                                    choice,
                                    name=row.item_id,
                                    id=f"notes-sync-attention-{index}-{choice_index}",
                                    classes="library-canvas-action",
                                    compact=True,
                                    disabled=True,
                                    tooltip=(
                                        "Conflict and deletion choices are unavailable "
                                        "in this release."
                                    ),
                                )
            if review.page_count > 1:
                yield Static(
                    f"Page {review.page} of {review.page_count}",
                    id="notes-sync-page-status",
                    markup=False,
                )
                with Horizontal(classes="ds-toolbar"):
                    yield Button(
                        "Previous",
                        id="notes-sync-page-previous",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=review.page <= 1,
                    )
                    yield Button(
                        "Next",
                        id="notes-sync-page-next",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=review.page >= review.page_count,
                    )
            return
        if phase == "receipt":
            yield Static(
                self.snapshot.receipt_line or "No changes were applied.",
                id="notes-sync-receipt",
                classes="destination-purpose",
                markup=False,
            )

    def _compose_pinned_actions(self) -> ComposeResult:
        phase = self.snapshot.phase
        if phase == "choose":
            yield Button(
                "Import once",
                id="notes-add-import-once",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Back to Notes",
                id="notes-sync-back",
                classes="library-canvas-action",
                compact=True,
            )
        elif phase == "configure":
            setup = self.snapshot.setup
            yield Button(
                "Check changes" if setup.can_check else "○ Check changes",
                id="notes-sync-check",
                classes="library-canvas-action",
                compact=True,
                disabled=not setup.can_check,
                tooltip=setup.validation_message or None,
            )
            yield Button(
                "Back",
                id="notes-sync-back",
                classes="library-canvas-action",
                compact=True,
            )
        elif phase in {"checking", "activating"}:
            yield Static(
                "Wait for the current step to finish.",
                id="notes-sync-wait-status",
                classes="library-disabled-reason",
                markup=False,
            )
        elif phase == "review":
            if self.snapshot.review.stale:
                yield Button(
                    "Check again",
                    id="notes-sync-check-again",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
                if self.snapshot.review.activation:
                    yield Button(
                        "Activate reviewed root",
                        id="notes-sync-activate",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=self.snapshot.review.attention_count > 0,
                    )
                else:
                    yield Button(
                        "Apply reviewed",
                        id="notes-sync-apply",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=self.snapshot.review.attention_count > 0,
                        tooltip=(
                            "Resolve attention before applying safe actions."
                            if self.snapshot.review.attention_count
                            else None
                        ),
                    )
            yield Button(
                "Back",
                id="notes-sync-back",
                classes="library-canvas-action",
                compact=True,
            )
        elif phase == "receipt":
            yield Button(
                "Back to Notes",
                id="notes-sync-back",
                classes="library-canvas-action",
                compact=True,
            )

    def sync_state(self, snapshot: LibraryNotesLastingSyncSnapshot) -> None:
        """Apply a snapshot while retaining live fields in the same form mode."""

        previous_phase = self.snapshot.phase
        self.snapshot = snapshot
        if previous_phase == snapshot.phase == "configure":
            status = self.query("#notes-sync-status")
            if status:
                status.first(Static).update(snapshot.status_line)
            check = self.query("#notes-sync-check")
            if check:
                button = check.first(Button)
                button.disabled = not snapshot.setup.can_check
                button.label = (
                    "Check changes" if snapshot.setup.can_check else "○ Check changes"
                )
                button.tooltip = snapshot.setup.validation_message or None
            validation = self.query("#notes-sync-validation")
            if validation:
                validation.first(Static).update(snapshot.setup.validation_message)
            folder = self.query("#notes-sync-folder-summary")
            if folder:
                folder.first(Static).update(
                    snapshot.setup.folder or "No folder selected"
                )
            server = self.query("#notes-sync-server-disabled-reason")
            if server:
                server.first(Static).update(snapshot.setup.server_disabled_reason)
            for value, label in (
                ("bidirectional", "⇄ Both ways"),
                ("folder_to_notes", "→ Folder to Notes"),
                ("notes_to_folder", "← Notes to folder"),
            ):
                controls = self.query(
                    f"#notes-sync-direction-{value.replace('_', '-')}"
                )
                if controls:
                    controls.first(Button).label = (
                        f"✓ {label}" if snapshot.setup.direction == value else label
                    )
            return
        self.refresh(recompose=True)

    def focus_first_safe_control(self) -> None:
        """Focus the first non-destructive control for the current phase."""

        selectors = {
            "choose": "#notes-add-import-once",
            "configure": "#notes-sync-display-name",
            "review": "#notes-sync-check-again"
            if self.snapshot.review.stale
            else "#notes-sync-back",
            "receipt": "#notes-sync-back",
        }
        selector = selectors.get(self.snapshot.phase)
        if selector is None:
            return
        controls = self.query(selector)
        if controls:
            controls.first().focus()

    @on(Input.Changed)
    def _setup_changed(self, event: Input.Changed) -> None:
        fields = {
            "notes-sync-display-name": "display_name",
        }
        field = fields.get(event.input.id or "")
        if field:
            self.post_message(self.SetupChanged(field, event.value))

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "notes-add-import-once":
            self.post_message(self.RelationshipRequested("import_once"))
        elif button_id == "notes-add-keep-synced":
            self.post_message(self.RelationshipRequested("keep_synced"))
        elif button_id == "notes-sync-destination-local":
            self.post_message(self.SetupChanged("destination", "local"))
        elif button_id == "notes-sync-folder-choose":
            self.post_message(self.FolderRequested())
        elif button_id.startswith("notes-sync-direction-"):
            self.post_message(self.SetupChanged("direction", event.button.name or ""))
        elif button_id in {"notes-sync-check", "notes-sync-check-again"}:
            self.post_message(self.CheckRequested())
        elif button_id == "notes-sync-apply":
            self.post_message(self.ApplyRequested())
        elif button_id == "notes-sync-activate":
            self.post_message(self.ActivateRequested())
        elif button_id.startswith("notes-sync-attention-"):
            _, _, _, row_text, choice_text = button_id.rsplit("-", 4)
            row = self.snapshot.review.rows[int(row_text)]
            self.post_message(
                self.AttentionChoiceRequested(
                    row.item_id, row.choices[int(choice_text)]
                )
            )
        elif button_id == "notes-sync-page-previous":
            self.post_message(self.PageRequested(-1))
        elif button_id == "notes-sync-page-next":
            self.post_message(self.PageRequested(1))
        elif button_id == "notes-sync-back":
            self.post_message(self.BackRequested())


__all__ = ["LibraryNotesAddFromFilesCanvas"]
