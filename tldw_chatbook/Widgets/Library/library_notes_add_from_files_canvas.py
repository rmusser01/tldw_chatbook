"""Flat retained canvas for authority choice, reviewed effects, and recovery.

The status/authority header and nearest valid action stay outside the bounded
scroll body so a 60x20 terminal never hides what the user is about to do.
"""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncApplyBlocker,
    LastingSyncHistoryRow,
    LastingSyncReviewRow,
    LibraryNotesLastingSyncSnapshot,
)
from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictComparison,
    NotesSyncConflictChoice,
)


_CHOICE_SLUGS = {
    "Keep file": "keep-file",
    "Keep note": "keep-note",
    "Keep both": "keep-both",
    "Skip for now": "skip",
}
_CHOICE_EFFECTS = {
    "Keep file": "update the Library note",
    "Keep note": "replace the folder file",
    "Keep both": "preserve a note copy, then update the bound note",
    "Skip for now": "make no changes",
}
_CHOICE_LABELS = {
    NotesSyncConflictChoice.KEEP_FILE: "Keep file",
    NotesSyncConflictChoice.KEEP_NOTE: "Keep note",
    NotesSyncConflictChoice.KEEP_BOTH: "Keep both",
    NotesSyncConflictChoice.SKIP: "Skip for now",
}


class _ConflictChoiceButton(Button):
    """Use Button semantics while adding the expected Space activation."""

    BINDINGS = [*Button.BINDINGS, Binding("space", "press", show=False)]


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

    class ChoiceRequested(Message):
        def __init__(self, binding_id: str, choice: str) -> None:
            super().__init__()
            self.binding_id = binding_id
            self.choice = choice

    class ViewRequested(Message):
        def __init__(self, binding_id: str) -> None:
            super().__init__()
            self.binding_id = binding_id

    class ReturnRequested(Message):
        pass

    class UndoRequested(Message):
        def __init__(self, root_id: str, operation_id: str) -> None:
            super().__init__()
            self.root_id = root_id
            self.operation_id = operation_id

    class DismissRequested(Message):
        def __init__(self, root_id: str, operation_id: str) -> None:
            super().__init__()
            self.root_id = root_id
            self.operation_id = operation_id

    class HistoryRequested(Message):
        def __init__(self, root_id: str) -> None:
            super().__init__()
            self.root_id = root_id

    class HistoryPageRequested(Message):
        def __init__(self, root_id: str, page: int) -> None:
            super().__init__()
            self.root_id = root_id
            self.page = page

    class HistoryReturnRequested(Message):
        pass

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
            if review.page == 1:
                yield from self._compose_receipts()
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
                        f"{row.category.title()} item {index + 1}",
                        classes="destination-section",
                        markup=False,
                    )
                    yield Static(row.effect, markup=False)
                    yield from self._compose_review_row_body(index, row)
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
            yield from self._compose_receipts()
            return
        if phase == "history":
            yield from self._compose_history()

    def _compose_review_row_body(
        self, index: int, row: LastingSyncReviewRow
    ) -> ComposeResult:
        comparison = self.snapshot.comparison
        expanded = comparison is not None and comparison.binding_id == row.item_id
        if row.conflict_eligible:
            yield Button(
                "View comparison",
                name=row.item_id,
                id=f"notes-sync-conflict-view-{index}",
                classes="library-canvas-action notes-sync-conflict-view",
                compact=True,
            )
        choices_panel = Vertical(
            id=f"notes-sync-conflict-choices-{index}",
            classes="library-notes-sync-conflict-choices",
        )
        choices_panel.display = not expanded
        with choices_panel:
            if row.conflict_eligible:
                with Vertical(classes="library-notes-sync-attention-actions"):
                    for choice in row.choices:
                        selected = _CHOICE_LABELS.get(row.selected_choice) == choice
                        label = f"{choice} — {_CHOICE_EFFECTS[choice]}"
                        yield _ConflictChoiceButton(
                            f"✓ {label}" if selected else label,
                            name=row.item_id,
                            id=(f"notes-sync-conflict-{index}-{_CHOICE_SLUGS[choice]}"),
                            classes=(
                                "library-canvas-action notes-sync-conflict-choice"
                                + (" is-selected" if selected else "")
                            ),
                            compact=True,
                        )
                yield Static(
                    row.selected_label or "No choice selected.",
                    id=f"notes-sync-conflict-selected-{index}",
                    classes="notes-sync-conflict-selected",
                    markup=False,
                )
            elif row.choices:
                unavailable = (
                    "Resolution unavailable for this item. No changes can be staged."
                )
                yield Static(
                    unavailable,
                    classes="library-disabled-reason",
                    markup=False,
                )
                with Vertical(classes="library-notes-sync-attention-actions"):
                    for choice_index, choice in enumerate(row.choices):
                        yield Button(
                            f"○ {choice}",
                            name=row.item_id,
                            id=f"notes-sync-attention-{index}-{choice_index}",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=True,
                            tooltip=unavailable,
                        )

        comparison_panel = Vertical(
            id=f"notes-sync-comparison-{index}",
            classes="library-notes-sync-comparison",
        )
        comparison_panel.display = expanded
        with comparison_panel:
            yield Static(
                self._comparison_summary(comparison) if expanded else "Comparison",
                id=f"notes-sync-comparison-summary-{index}",
                classes="destination-purpose",
                markup=False,
            )
            yield TextArea(
                comparison.diff if expanded else "",
                language=None,
                soft_wrap=False,
                read_only=True,
                show_cursor=False,
                id=f"notes-sync-comparison-diff-{index}",
                classes="notes-sync-comparison-diff",
            )
            yield Button(
                "Return to choices",
                id=f"notes-sync-comparison-return-{index}",
                classes="library-canvas-action",
                compact=True,
            )

    @staticmethod
    def _comparison_summary(comparison: ConflictComparison | None) -> str:
        if comparison is None:
            return "Comparison"
        omitted = (
            " · Diff omitted because an input exceeds the display limit."
            if comparison.input_elided
            else ""
        )
        clipped = (
            " · Diff shortened to the display limit."
            if comparison.output_elided
            else ""
        )
        return (
            f"{comparison.note_title} · {comparison.relative_path}\n"
            f"Note v{comparison.note_version}, updated "
            f"{comparison.note_updated_label} · "
            f"{comparison.note_line_count} lines/{comparison.note_character_count} chars\n"
            f"File modified {comparison.file_modified_ns} ns · "
            f"{comparison.file_line_count} lines/{comparison.file_character_count} chars"
            f"{omitted}{clipped}"
        )

    def _root_id(self) -> str:
        return self.snapshot.review.root_id or self.snapshot.history.root_id

    def _compose_receipts(self) -> ComposeResult:
        if self.snapshot.receipts_unavailable:
            yield Static(
                "At-action receipts are unavailable. Open Resolution history.",
                classes="library-disabled-reason",
                markup=False,
            )
        for index, receipt in enumerate(self.snapshot.receipts):
            with Vertical(
                id=f"notes-sync-receipt-{index}",
                classes="library-notes-sync-receipt",
            ):
                yield Static(
                    f"Resolved · {receipt.item_label}",
                    classes="destination-section",
                    markup=False,
                )
                yield Static(
                    f"Choice: {_CHOICE_LABELS[receipt.choice]} · State: {receipt.state}",
                    markup=False,
                )
                with Horizontal(classes="ds-toolbar notes-sync-receipt-actions"):
                    if receipt.undo_available:
                        yield Button(
                            "Undo",
                            name=receipt.operation_id,
                            id=f"notes-sync-receipt-undo-{index}",
                            classes="library-canvas-action",
                            compact=True,
                        )
                    else:
                        yield Button(
                            self._undo_label(receipt.undo_reason, receipt.state),
                            name=receipt.operation_id,
                            id=f"notes-sync-receipt-undo-{index}",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=True,
                            tooltip=receipt.undo_reason or "Undo unavailable",
                        )
                    yield Button(
                        "Dismiss",
                        name=receipt.operation_id,
                        id=f"notes-sync-receipt-dismiss-{index}",
                        classes="library-canvas-action",
                        compact=True,
                    )

    @staticmethod
    def _undo_label(reason: str | None, state: str) -> str:
        normalized = f"{reason or ''} {state}".casefold().replace("_", " ")
        if "undone" in normalized:
            return "Undone"
        if "expired" in normalized:
            return "Undo expired"
        if "changed since" in normalized:
            return "Changed since resolution"
        return reason or "Undo unavailable"

    def _compose_history(self) -> ComposeResult:
        history = self.snapshot.history
        yield Static("Resolution history", classes="destination-section", markup=False)
        if history.unavailable:
            yield Static(
                "Resolution history is unavailable. Try again.",
                classes="library-disabled-reason",
                markup=False,
            )
        elif not history.rows:
            yield Static(
                "No conflict resolutions are recorded for this root.",
                classes="destination-purpose",
                markup=False,
            )
        for index, row in enumerate(history.rows):
            yield from self._compose_history_row(index, row)
        yield Static(f"Page {history.page}", id="notes-sync-history-page", markup=False)
        with Horizontal(classes="ds-toolbar notes-sync-history-actions"):
            yield Button(
                "Previous",
                name=history.root_id,
                id="notes-sync-history-previous",
                classes="library-canvas-action",
                compact=True,
                disabled=history.page <= 1,
            )
            yield Button(
                "Next",
                name=history.root_id,
                id="notes-sync-history-next",
                classes="library-canvas-action",
                compact=True,
                disabled=not history.has_next,
            )
            yield Button(
                "Return",
                id="notes-sync-history-return",
                classes="library-canvas-action",
                compact=True,
            )

    def _compose_history_row(
        self, index: int, row: LastingSyncHistoryRow
    ) -> ComposeResult:
        with Vertical(
            id=f"notes-sync-history-row-{index}",
            classes="library-notes-sync-history-row",
        ):
            yield Static(row.item_label, classes="destination-section", markup=False)
            yield Static(
                f"{_CHOICE_LABELS[row.choice]} · {row.completed_at or row.updated_at} · "
                f"{row.state}",
                markup=False,
            )
            if row.undo_available:
                yield Button(
                    "Undo",
                    name=row.operation_id,
                    id=f"notes-sync-history-undo-{index}",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
                yield Static(
                    self._undo_label(row.undo_reason, row.state),
                    classes="library-disabled-reason",
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
                        disabled=not self.snapshot.review.can_apply,
                        tooltip=self._apply_tooltip(),
                    )
            root_id = self._root_id()
            if root_id:
                yield Button(
                    "Resolution history",
                    name=root_id,
                    id="notes-sync-history-open",
                    classes="library-canvas-action",
                    compact=True,
                )
            yield Button(
                "Back",
                id="notes-sync-back",
                classes="library-canvas-action",
                compact=True,
            )
        elif phase == "receipt":
            root_id = self._root_id()
            if root_id:
                yield Button(
                    "Resolution history",
                    name=root_id,
                    id="notes-sync-history-open",
                    classes="library-canvas-action",
                    compact=True,
                )
            yield Button(
                "Back to Notes",
                id="notes-sync-back",
                classes="library-canvas-action",
                compact=True,
            )

    def _apply_tooltip(self) -> str | None:
        blocker = self.snapshot.review.apply_blocker
        return {
            LastingSyncApplyBlocker.NONE: None,
            LastingSyncApplyBlocker.NOTHING_SELECTED: (
                "Choose a mutating conflict resolution or review a safe action."
            ),
            LastingSyncApplyBlocker.STALE_REVIEW: "Check again before applying.",
            LastingSyncApplyBlocker.ACTIVATION_REVIEW: (
                "Use Activate reviewed root for this setup review."
            ),
            LastingSyncApplyBlocker.DELETION_REVIEW: (
                "Deletion review is unavailable in this release."
            ),
            LastingSyncApplyBlocker.MANAGED_PLACEMENT: (
                "Managed placement review is unavailable in this release."
            ),
            LastingSyncApplyBlocker.ROOT_OR_CAPABILITY: (
                "Resolve the root or capability blocker before applying."
            ),
            LastingSyncApplyBlocker.UNSUPPORTED_ATTENTION: (
                "This attention item cannot be resolved here."
            ),
        }[blocker]

    def sync_state(self, snapshot: LibraryNotesLastingSyncSnapshot) -> None:
        """Apply a snapshot while retaining live fields in the same form mode."""

        previous_phase = self.snapshot.phase
        previous_snapshot = self.snapshot
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
        if (
            previous_phase == snapshot.phase == "review"
            and previous_snapshot.review.root_id == snapshot.review.root_id
            and previous_snapshot.review.observation_token
            == snapshot.review.observation_token
            and previous_snapshot.review.page == snapshot.review.page
            and tuple(row.item_id for row in previous_snapshot.review.rows)
            == tuple(row.item_id for row in snapshot.review.rows)
            and previous_snapshot.receipts == snapshot.receipts
            and previous_snapshot.receipts_unavailable == snapshot.receipts_unavailable
        ):
            self._sync_review(snapshot)
            return
        self.refresh(recompose=True)

    def _sync_review(self, snapshot: LibraryNotesLastingSyncSnapshot) -> None:
        status = self.query("#notes-sync-status")
        if status:
            status.first(Static).update(snapshot.status_line)
        summary = self.query("#notes-sync-review-summary")
        if summary:
            review = snapshot.review
            summary.first(Static).update(
                f"{review.safe_count} safe · {review.attention_count} need attention · "
                f"{review.skip_count} skipped · {review.managed_count} managed placements"
            )
        apply = self.query("#notes-sync-apply")
        if apply:
            apply_button = apply.first(Button)
            apply_button.disabled = not snapshot.review.can_apply
            apply_button.tooltip = self._apply_tooltip()

        comparison = snapshot.comparison
        for index, row in enumerate(snapshot.review.rows):
            if not row.conflict_eligible:
                continue
            selected_label = _CHOICE_LABELS.get(row.selected_choice)
            for choice in row.choices:
                slug = _CHOICE_SLUGS[choice]
                button = self.query_one(f"#notes-sync-conflict-{index}-{slug}", Button)
                selected = choice == selected_label
                label = f"{choice} — {_CHOICE_EFFECTS[choice]}"
                button.label = f"✓ {label}" if selected else label
                button.set_class(selected, "is-selected")
            selected = self.query_one(f"#notes-sync-conflict-selected-{index}", Static)
            selected.update(row.selected_label or "No choice selected.")

            expanded = comparison is not None and comparison.binding_id == row.item_id
            view = self.query_one(f"#notes-sync-conflict-view-{index}", Button)
            move_focus = expanded and self.screen.focused is view
            choices = self.query_one(f"#notes-sync-conflict-choices-{index}")
            comparison_panel = self.query_one(f"#notes-sync-comparison-{index}")
            choices.display = not expanded
            comparison_panel.display = expanded
            if expanded:
                self.query_one(
                    f"#notes-sync-comparison-summary-{index}", Static
                ).update(self._comparison_summary(comparison))
                diff = self.query_one(f"#notes-sync-comparison-diff-{index}", TextArea)
                diff.load_text(comparison.diff)
                if move_focus:
                    self.call_after_refresh(diff.focus)

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
        elif button_id.startswith("notes-sync-conflict-view-"):
            self.post_message(self.ViewRequested(event.button.name or ""))
        elif button_id.startswith("notes-sync-conflict-"):
            parts = button_id.removeprefix("notes-sync-conflict-").split("-", 1)
            row = self.snapshot.review.rows[int(parts[0])]
            choice_slug = parts[1]
            choice = next(
                label for label, slug in _CHOICE_SLUGS.items() if slug == choice_slug
            )
            self.post_message(self.ChoiceRequested(row.item_id, choice))
        elif button_id.startswith("notes-sync-comparison-return-"):
            index = int(button_id.rsplit("-", 1)[1])
            choices = self.query_one(f"#notes-sync-conflict-choices-{index}")
            comparison = self.query_one(f"#notes-sync-comparison-{index}")
            choices.display = True
            comparison.display = False
            self.query_one(f"#notes-sync-conflict-view-{index}", Button).focus()
            self.post_message(self.ReturnRequested())
        elif button_id.startswith("notes-sync-receipt-undo-"):
            self.post_message(
                self.UndoRequested(self._root_id(), event.button.name or "")
            )
        elif button_id.startswith("notes-sync-receipt-dismiss-"):
            self.post_message(
                self.DismissRequested(self._root_id(), event.button.name or "")
            )
        elif button_id.startswith("notes-sync-history-undo-"):
            self.post_message(
                self.UndoRequested(self._root_id(), event.button.name or "")
            )
        elif button_id == "notes-sync-history-open":
            self.post_message(self.HistoryRequested(event.button.name or ""))
        elif button_id in {"notes-sync-history-previous", "notes-sync-history-next"}:
            delta = -1 if button_id.endswith("previous") else 1
            self.post_message(
                self.HistoryPageRequested(
                    event.button.name or "", self.snapshot.history.page + delta
                )
            )
        elif button_id == "notes-sync-history-return":
            self.post_message(self.HistoryReturnRequested())
        elif button_id == "notes-sync-page-previous":
            self.post_message(self.PageRequested(-1))
        elif button_id == "notes-sync-page-next":
            self.post_message(self.PageRequested(1))
        elif button_id == "notes-sync-back":
            self.post_message(self.BackRequested())


__all__ = ["LibraryNotesAddFromFilesCanvas"]
