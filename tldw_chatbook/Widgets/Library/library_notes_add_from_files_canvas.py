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
from textual.widget import Widget
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncApplyBlocker,
    LastingSyncHistoryRow,
    LastingSyncReviewRow,
    LastingSyncReviewSource,
    LibraryNotesLastingSyncSnapshot,
)
from tldw_chatbook.Notes.notes_sync_conflicts import (
    ConflictComparison,
    NotesSyncConflictChoice,
)
from tldw_chatbook.Notes.notes_sync_models import validate_notes_sync_opaque_id


_CHOICE_SLUGS = {
    "Keep file": "keep-file",
    "Keep note": "keep-note",
    "Keep both": "keep-both",
    "Skip for now": "skip",
}
_CHOICE_EFFECTS = {
    "Keep file": "update the Library note",
    "Keep note": "replace the folder file",
    "Keep both": "preserve an unbound note copy, then update the bound note.",
    "Skip for now": "make no changes",
}
_CHOICE_LABELS = {
    NotesSyncConflictChoice.KEEP_FILE: "Keep file",
    NotesSyncConflictChoice.KEEP_NOTE: "Keep note",
    NotesSyncConflictChoice.KEEP_BOTH: "Keep both",
    NotesSyncConflictChoice.SKIP: "Skip for now",
}


class _ReviewActionButton(Button):
    """Retain the exact review provenance rendered with an action."""

    def __init__(
        self,
        label: str,
        *,
        review_root_id: str,
        review_observation_token: str,
        review_source: LastingSyncReviewSource | None = None,
        rendered_page: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(label, **kwargs)
        self.review_root_id = review_root_id
        self.review_observation_token = review_observation_token
        self.review_source = review_source
        self.rendered_page = rendered_page


class _ConflictChoiceButton(_ReviewActionButton):
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
        def __init__(
            self,
            root_id: str = "",
            observation_token: str = "",
            source: LastingSyncReviewSource | None = None,
        ) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token
            self.source = source

    class ApplyRequested(Message):
        def __init__(self, root_id: str, observation_token: str) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token

    class ChoiceRequested(Message):
        def __init__(
            self,
            root_id: str,
            observation_token: str,
            binding_id: str,
            choice: str,
        ) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token
            self.binding_id = binding_id
            self.choice = choice

    class ViewRequested(Message):
        def __init__(
            self, root_id: str, observation_token: str, binding_id: str
        ) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token
            self.binding_id = binding_id

    class ReturnRequested(Message):
        def __init__(
            self, root_id: str, observation_token: str, binding_id: str
        ) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token
            self.binding_id = binding_id

    class UndoRequested(Message):
        def __init__(
            self,
            root_id: str,
            observation_token: str,
            operation_id: str,
            page: int | None,
        ) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token
            self.operation_id = operation_id
            self.page = page

    class DismissRequested(Message):
        def __init__(
            self, root_id: str, observation_token: str, operation_id: str
        ) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token
            self.operation_id = operation_id

    class HistoryRequested(Message):
        def __init__(self, root_id: str, observation_token: str) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token

    class HistoryPageRequested(Message):
        def __init__(
            self,
            root_id: str,
            observation_token: str,
            from_page: int,
            page: int,
        ) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token
            self.from_page = from_page
            self.page = page

    class HistoryReturnRequested(Message):
        pass

    class ActivateRequested(Message):
        def __init__(self, root_id: str, observation_token: str) -> None:
            super().__init__()
            self.root_id = root_id
            self.observation_token = observation_token

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
        self._handled_conflict_focus_request: tuple[str, str, str] | None = None
        self._scheduled_conflict_focus_request: tuple[str, str, str] | None = None
        self.add_class("library-notes-lasting-sync-canvas")

    def on_mount(self) -> None:
        self.call_after_refresh(self._schedule_conflict_focus_request)

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
        shown_comparison = (
            comparison
            if comparison is not None and comparison.binding_id == row.item_id
            else None
        )
        expanded = shown_comparison is not None
        if row.conflict_eligible:
            yield Static(
                row.conflict_title,
                id=f"notes-sync-conflict-title-{index}",
                classes="notes-sync-conflict-title",
                markup=False,
            )
            yield Static(
                row.conflict_relative_path,
                id=f"notes-sync-conflict-path-{index}",
                classes="notes-sync-conflict-path",
                markup=False,
            )
            yield _ReviewActionButton(
                "View comparison",
                review_root_id=self.snapshot.review.root_id,
                review_observation_token=self.snapshot.review.observation_token,
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
                yield Static(
                    row.selected_label or "No choice selected.",
                    id=f"notes-sync-conflict-selected-{index}",
                    classes="notes-sync-conflict-selected",
                    markup=False,
                )
                with Vertical(classes="library-notes-sync-attention-actions"):
                    ordered_choices = sorted(
                        row.choices, key=lambda choice: choice != "Keep both"
                    )
                    for choice in ordered_choices:
                        selected = _CHOICE_LABELS.get(row.selected_choice) == choice
                        effect = _CHOICE_EFFECTS[choice]
                        yield _ConflictChoiceButton(
                            f"✓ {choice}" if selected else choice,
                            review_root_id=self.snapshot.review.root_id,
                            review_observation_token=(
                                self.snapshot.review.observation_token
                            ),
                            name=row.item_id,
                            id=(f"notes-sync-conflict-{index}-{_CHOICE_SLUGS[choice]}"),
                            classes=(
                                "library-canvas-action notes-sync-conflict-choice"
                                + (" is-selected" if selected else "")
                            ),
                            compact=True,
                            tooltip=effect,
                        )
                        yield Static(
                            (
                                "preserve an unbound note copy\n"
                                "then update the bound note."
                                if choice == "Keep both"
                                else effect
                            ),
                            id=(
                                f"notes-sync-conflict-effect-{index}-"
                                f"{_CHOICE_SLUGS[choice]}"
                            ),
                            classes="notes-sync-conflict-effect",
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
                (
                    self._comparison_summary(shown_comparison)
                    if shown_comparison is not None
                    else "Comparison"
                ),
                id=f"notes-sync-comparison-summary-{index}",
                classes="destination-purpose",
                markup=False,
            )
            yield TextArea(
                shown_comparison.diff if shown_comparison is not None else "",
                language=None,
                soft_wrap=False,
                read_only=True,
                show_cursor=False,
                id=f"notes-sync-comparison-diff-{index}",
                classes="notes-sync-comparison-diff",
            )
            yield _ReviewActionButton(
                "Return to choices",
                review_root_id=self.snapshot.review.root_id,
                review_observation_token=self.snapshot.review.observation_token,
                name=row.item_id,
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
                        yield _ReviewActionButton(
                            "Undo",
                            review_root_id=self.snapshot.review.root_id,
                            review_observation_token=self.snapshot.review.observation_token,
                            name=receipt.operation_id,
                            id=f"notes-sync-receipt-undo-{index}",
                            classes="library-canvas-action",
                            compact=True,
                        )
                    else:
                        yield _ReviewActionButton(
                            self._undo_label(receipt.undo_reason, receipt.state),
                            review_root_id=self.snapshot.review.root_id,
                            review_observation_token=self.snapshot.review.observation_token,
                            name=receipt.operation_id,
                            id=f"notes-sync-receipt-undo-{index}",
                            classes="library-canvas-action",
                            compact=True,
                            disabled=True,
                            tooltip=receipt.undo_reason or "Undo unavailable",
                        )
                    yield _ReviewActionButton(
                        "Dismiss",
                        review_root_id=self.snapshot.review.root_id,
                        review_observation_token=self.snapshot.review.observation_token,
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
        yield Static(
            "Scroll history for more entries; paging controls stay below.",
            id="notes-sync-history-scroll-cue",
            classes="library-disabled-reason",
            markup=False,
        )
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

    def _compose_history_actions(self) -> ComposeResult:
        history = self.snapshot.history
        yield Static(
            f"Page {history.page}",
            id="notes-sync-history-page",
            classes="notes-sync-history-page",
            markup=False,
        )
        yield _ReviewActionButton(
            "Previous",
            review_root_id=history.root_id,
            review_observation_token=self.snapshot.review.observation_token,
            rendered_page=history.page,
            name=history.root_id,
            id="notes-sync-history-previous",
            classes="library-canvas-action",
            compact=True,
            disabled=history.page <= 1,
        )
        yield _ReviewActionButton(
            "Next",
            review_root_id=history.root_id,
            review_observation_token=self.snapshot.review.observation_token,
            rendered_page=history.page,
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
        history = self.snapshot.history
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
                yield _ReviewActionButton(
                    "Undo",
                    review_root_id=history.root_id,
                    review_observation_token=self.snapshot.review.observation_token,
                    rendered_page=history.page,
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
                yield _ReviewActionButton(
                    "Check again",
                    review_root_id=self.snapshot.review.root_id,
                    review_observation_token=self.snapshot.review.observation_token,
                    review_source=self.snapshot.review.source,
                    id="notes-sync-check-again",
                    classes="library-canvas-action",
                    compact=True,
                )
            else:
                if self.snapshot.review.activation:
                    yield _ReviewActionButton(
                        "Activate reviewed root",
                        review_root_id=self.snapshot.review.root_id,
                        review_observation_token=self.snapshot.review.observation_token,
                        id="notes-sync-activate",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=self.snapshot.review.attention_count > 0,
                    )
                else:
                    yield _ReviewActionButton(
                        "Apply reviewed",
                        review_root_id=self.snapshot.review.root_id,
                        review_observation_token=(
                            self.snapshot.review.observation_token
                        ),
                        id="notes-sync-apply",
                        classes="library-canvas-action",
                        compact=True,
                        disabled=not self.snapshot.review.can_apply,
                        tooltip=self._apply_tooltip(),
                    )
            root_id = self._root_id()
            if root_id:
                history_available = self.snapshot.history_available
                yield _ReviewActionButton(
                    (
                        "Resolution history"
                        if history_available
                        else "○ Resolution history"
                    ),
                    review_root_id=root_id,
                    review_observation_token=self.snapshot.review.observation_token,
                    name=root_id,
                    id="notes-sync-history-open",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=not history_available,
                    tooltip=(
                        None
                        if history_available
                        else "No durable conflict resolutions are available for this root."
                    ),
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
                history_available = self.snapshot.history_available
                yield _ReviewActionButton(
                    (
                        "Resolution history"
                        if history_available
                        else "○ Resolution history"
                    ),
                    review_root_id=root_id,
                    review_observation_token=self.snapshot.review.observation_token,
                    name=root_id,
                    id="notes-sync-history-open",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=not history_available,
                    tooltip=(
                        None
                        if history_available
                        else "No durable conflict resolutions are available for this root."
                    ),
                )
            yield Button(
                "Back to Notes",
                id="notes-sync-back",
                classes="library-canvas-action",
                compact=True,
            )
        elif phase == "history":
            yield from self._compose_history_actions()

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
            and previous_snapshot.review.stale == snapshot.review.stale
            and previous_snapshot.review.page == snapshot.review.page
            and tuple(row.item_id for row in previous_snapshot.review.rows)
            == tuple(row.item_id for row in snapshot.review.rows)
            and previous_snapshot.receipts == snapshot.receipts
            and previous_snapshot.receipts_unavailable == snapshot.receipts_unavailable
        ):
            self._sync_review(snapshot, previous_snapshot)
            return
        self.refresh(recompose=True)
        self._schedule_conflict_focus_request()

    def _sync_review(
        self,
        snapshot: LibraryNotesLastingSyncSnapshot,
        previous_snapshot: LibraryNotesLastingSyncSnapshot,
    ) -> None:
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
        history = self.query("#notes-sync-history-open")
        if history:
            history_button = history.first(Button)
            history_button.disabled = not snapshot.history_available
            history_button.label = (
                "Resolution history"
                if snapshot.history_available
                else "○ Resolution history"
            )
            history_button.tooltip = (
                None
                if snapshot.history_available
                else "No durable conflict resolutions are available for this root."
            )

        comparison = snapshot.comparison
        for index, row in enumerate(snapshot.review.rows):
            if not row.conflict_eligible:
                continue
            selected_label = _CHOICE_LABELS.get(row.selected_choice)
            for choice in row.choices:
                slug = _CHOICE_SLUGS[choice]
                button = self.query_one(f"#notes-sync-conflict-{index}-{slug}", Button)
                selected = choice == selected_label
                button.label = f"✓ {choice}" if selected else choice
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
            if comparison is not None and expanded:
                self.query_one(
                    f"#notes-sync-comparison-summary-{index}", Static
                ).update(self._comparison_summary(comparison))
                diff = self.query_one(f"#notes-sync-comparison-diff-{index}", TextArea)
                diff.load_text(comparison.diff)
                if move_focus:
                    self.call_after_refresh(
                        self._focus_published_comparison,
                        row.item_id,
                        view,
                        diff,
                    )
        self._schedule_conflict_focus_request()

    def _focus_published_comparison(
        self,
        binding_id: str,
        view: Button,
        diff: TextArea,
    ) -> None:
        """Focus a published diff only while its exact View still owns focus."""

        if not self.is_mounted or self.screen.focused is not view:
            return
        comparison = self.snapshot.comparison
        if comparison is None or comparison.binding_id != binding_id:
            return
        views = tuple(self.query(".notes-sync-conflict-view"))
        diffs = tuple(self.query(".notes-sync-comparison-diff"))
        if (
            view not in views
            or view.name != binding_id
            or diff not in diffs
            or not diff.display
        ):
            return
        self.screen.set_focus(diff)

    def _current_conflict_focus_request(self) -> tuple[str, str, str] | None:
        review = self.snapshot.review
        binding_id = self.snapshot.conflict_focus_binding_id
        if (
            self.snapshot.phase != "review"
            or not review.root_id
            or not review.observation_token
            or binding_id is None
        ):
            return None
        return review.root_id, review.observation_token, binding_id

    def _requested_conflict_view(self, request: tuple[str, str, str]) -> Button | None:
        if not self.is_mounted or self._current_conflict_focus_request() != request:
            return None
        binding_id = request[2]
        if not any(
            row.item_id == binding_id and row.conflict_eligible
            for row in self.snapshot.review.rows
        ):
            return None
        for view in self.query(".notes-sync-conflict-view"):
            if (
                isinstance(view, Button)
                and view.is_mounted
                and not view.disabled
                and view.name == binding_id
            ):
                return view
        return None

    def _schedule_conflict_focus_request(self) -> None:
        request = self._current_conflict_focus_request()
        if request is None:
            self._scheduled_conflict_focus_request = None
            return
        if request in {
            self._handled_conflict_focus_request,
            self._scheduled_conflict_focus_request,
        }:
            return
        self._scheduled_conflict_focus_request = request
        focused = self.screen.focused
        focused_in_canvas = focused is self or (
            focused is not None and self in focused.ancestors
        )
        self.call_after_refresh(
            self._defer_requested_conflict_focus,
            request,
            focused,
            focused_in_canvas,
        )

    def _defer_requested_conflict_focus(
        self,
        request: tuple[str, str, str],
        focused: Widget | None,
        focused_in_canvas: bool,
    ) -> None:
        self.call_after_refresh(
            self._focus_requested_conflict,
            request,
            focused,
            focused_in_canvas,
        )

    def _focus_requested_conflict(
        self,
        request: tuple[str, str, str],
        focused: Widget | None,
        focused_in_canvas: bool,
    ) -> None:
        """Honor one fresh focus request without stealing newer user focus."""

        view = self._requested_conflict_view(request)
        current_focus = self.screen.focused
        same_canvas_origin = (
            focused_in_canvas
            and focused is not None
            and focused.is_attached
            and current_focus is focused
        )
        no_initial_origin = focused is None and current_focus is None
        old_canvas_origin_lost_focus = (
            focused_in_canvas
            and focused is not None
            and not focused.is_attached
            and current_focus is None
        )
        if view is None or not (
            same_canvas_origin or no_initial_origin or old_canvas_origin_lost_focus
        ):
            return
        self.screen.set_focus(view)
        self._handled_conflict_focus_request = request

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
        elif button_id == "notes-sync-check":
            self.post_message(self.CheckRequested())
        elif button_id == "notes-sync-check-again":
            if not isinstance(event.button, _ReviewActionButton):
                return
            self.post_message(
                self.CheckRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                    event.button.review_source,
                )
            )
        elif button_id == "notes-sync-apply":
            if not isinstance(event.button, _ReviewActionButton):
                return
            self.post_message(
                self.ApplyRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                )
            )
        elif button_id == "notes-sync-activate":
            if not isinstance(event.button, _ReviewActionButton):
                return
            self.post_message(
                self.ActivateRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                )
            )
        elif button_id.startswith("notes-sync-conflict-view-"):
            if not isinstance(event.button, _ReviewActionButton):
                return
            self.post_message(
                self.ViewRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                    event.button.name or "",
                )
            )
        elif button_id.startswith("notes-sync-conflict-"):
            parts = button_id.removeprefix("notes-sync-conflict-").split("-", 1)
            if len(parts) != 2:
                return
            choice_slug = parts[1]
            choice = next(
                (label for label, slug in _CHOICE_SLUGS.items() if slug == choice_slug),
                None,
            )
            binding_id = event.button.name
            if (
                choice is None
                or not binding_id
                or not isinstance(event.button, _ReviewActionButton)
            ):
                return
            try:
                validate_notes_sync_opaque_id(binding_id, field_name="binding_id")
            except (TypeError, ValueError):
                return
            self.post_message(
                self.ChoiceRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                    binding_id,
                    choice,
                )
            )
        elif button_id.startswith("notes-sync-comparison-return-"):
            if not isinstance(event.button, _ReviewActionButton):
                return
            binding_id = event.button.name or ""
            current_review = self.snapshot.review
            is_current = (
                event.button.review_root_id == current_review.root_id
                and event.button.review_observation_token
                == current_review.observation_token
                and self.snapshot.comparison is not None
                and self.snapshot.comparison.binding_id == binding_id
            )
            self.post_message(
                self.ReturnRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                    binding_id,
                )
            )
            if not is_current:
                return
            index = int(button_id.rsplit("-", 1)[1])
            choices = self.query_one(f"#notes-sync-conflict-choices-{index}")
            comparison = self.query_one(f"#notes-sync-comparison-{index}")
            choices.display = True
            comparison.display = False
            self.screen.set_focus(
                self.query_one(f"#notes-sync-conflict-view-{index}", Button)
            )
        elif button_id.startswith("notes-sync-receipt-undo-"):
            if not isinstance(event.button, _ReviewActionButton):
                return
            self.post_message(
                self.UndoRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                    event.button.name or "",
                    None,
                )
            )
        elif button_id.startswith("notes-sync-receipt-dismiss-"):
            if not isinstance(event.button, _ReviewActionButton):
                return
            self.post_message(
                self.DismissRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                    event.button.name or "",
                )
            )
        elif button_id.startswith("notes-sync-history-undo-"):
            if not isinstance(event.button, _ReviewActionButton):
                return
            self.post_message(
                self.UndoRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                    event.button.name or "",
                    event.button.rendered_page,
                )
            )
        elif button_id == "notes-sync-history-open":
            if not isinstance(event.button, _ReviewActionButton):
                return
            self.post_message(
                self.HistoryRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                )
            )
        elif button_id in {"notes-sync-history-previous", "notes-sync-history-next"}:
            if not isinstance(event.button, _ReviewActionButton):
                return
            delta = -1 if button_id.endswith("previous") else 1
            if event.button.rendered_page is None:
                return
            self.post_message(
                self.HistoryPageRequested(
                    event.button.review_root_id,
                    event.button.review_observation_token,
                    event.button.rendered_page,
                    event.button.rendered_page + delta,
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
