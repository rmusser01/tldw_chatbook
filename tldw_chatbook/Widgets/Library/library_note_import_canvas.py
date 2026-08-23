"""Render-only canvas for the reviewed one-time Notes import workflow."""

from __future__ import annotations

from itertools import groupby
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_note_import_state import (
    LibraryNoteImportItemSnapshot,
    LibraryNoteImportSnapshot,
)
from tldw_chatbook.Widgets.Library.library_canvas_sync import (
    PostRecomposeCallback,
)


_CLASSIFICATION_LABELS = {
    "new": "New",
    "unchanged_repeat": "Unchanged repeat",
    "changed_repeat": "Changed repeat",
    "uncertain_match": "Uncertain match",
    "unsupported": "Unsupported",
    "failed": "Failed",
}


def _choice_label(*, selected: bool, text: str) -> str:
    """Return a monochrome-readable selected/unselected action label."""
    return f"{'✓' if selected else '○'} {text}"


def _disabled_action_label(text: str, *, disabled: bool) -> str:
    """Keep an unavailable action's reason discoverable without colour."""
    return f"{text} unavailable" if disabled else text


def _bounded_source_name(name: str) -> str:
    """Keep one selected filename useful without dominating compact layouts."""
    return name if len(name) <= 48 else f"{name[:47]}…"


class _ImportBody(VerticalScroll):
    """Keyboard-focusable scroll owner for the changing import detail."""

    can_focus = True


class LibraryNoteImportCanvas(PostRecomposeCallback, Vertical):
    """Render one immutable import snapshot and post typed physical intents."""

    BUNDLED_CSS = """
    $ds-status-error-readable: #ff8fa3;

    LibraryNoteImportCanvas {
        width: 1fr;
        min-width: 40;
        height: 1fr;
        overflow: hidden;
    }

    LibraryNoteImportCanvas #note-import-heading,
    LibraryNoteImportCanvas #note-import-status,
    LibraryNoteImportCanvas #note-import-overflow-hint,
    LibraryNoteImportCanvas .note-import-primary {
        height: auto;
    }

    LibraryNoteImportCanvas #note-import-body {
        height: 1fr;
        overflow-y: auto;
    }

    LibraryNoteImportCanvas Static,
    LibraryNoteImportCanvas Input,
    LibraryNoteImportCanvas Button {
        width: 1fr;
    }

    LibraryNoteImportCanvas .note-import-group-heading {
        text-style: bold;
        margin-top: 1;
    }

    LibraryNoteImportCanvas .note-import-item-name {
        text-style: bold;
    }

    LibraryNoteImportCanvas .note-import-quiet {
        color: $text-muted;
    }

    LibraryNoteImportCanvas .note-import-error {
        color: $ds-status-error-readable;
        text-style: bold;
    }
    """

    class AddSourceRequested(Message):
        """Request one more physical file selection."""

    class DestinationChanged(Message):
        """Report the proposed, not-yet-created Notes destination."""

        def __init__(self, destination: str) -> None:
            super().__init__()
            self.destination = destination

    class CheckRequested(Message):
        """Request read-only discovery and planning."""

    class CollisionChoiceRequested(Message):
        """Report one explicit imported-root collision choice."""

        def __init__(self, choice: str) -> None:
            super().__init__()
            self.choice = choice

    class CollisionNameChanged(Message):
        """Report the proposed replacement root label."""

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    class ItemActionRequested(Message):
        """Report one item-scoped Skip/Create/Update selection."""

        def __init__(self, item_id: str, action: str) -> None:
            super().__init__()
            self.item_id = item_id
            self.action = action

    class ItemChoiceRequested(Message):
        """Report an independent content or membership decision."""

        def __init__(self, item_id: str, choice: str, enabled: bool) -> None:
            super().__init__()
            self.item_id = item_id
            self.choice = choice
            self.enabled = enabled

    class UncertainMatchConfirmed(Message):
        """Request confirmation of one uncertain existing-note match."""

        def __init__(self, item_id: str) -> None:
            super().__init__()
            self.item_id = item_id

    class ImportRequested(Message):
        """Request approval and execution of the exact reviewed plan."""

    class CancelRequested(Message):
        """Request cooperative cancellation of checking or execution."""

    class RetryRequested(Message):
        """Request retry of only receipt-reported retryable failures."""

    class PageRequested(Message):
        """Request a bounded preview-page change."""

        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    def __init__(self, snapshot: LibraryNoteImportSnapshot, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.snapshot = snapshot
        self._destination_value = snapshot.destination
        self._collision_name = (
            snapshot.collision_rename_input or snapshot.collision_name
        )

    def sync_state(self, snapshot: LibraryNoteImportSnapshot) -> None:
        """Apply one snapshot without replacing an actively edited input."""
        previous = self.snapshot
        self.snapshot = snapshot
        self._destination_value = snapshot.destination
        self._collision_name = (
            snapshot.collision_rename_input or snapshot.collision_name
        )
        if not self.is_attached or previous.phase != snapshot.phase:
            self.refresh(recompose=True)
            return
        if snapshot.phase in {"select", "destination"}:
            self._sync_destination_controls(snapshot)
            return
        collision_only = (
            snapshot.phase == "review"
            and previous.preview_items == snapshot.preview_items
            and previous.page == snapshot.page
        )
        if collision_only:
            self._sync_collision_controls(snapshot)
            return
        self.refresh(recompose=True)

    def _sync_destination_controls(self, snapshot: LibraryNoteImportSnapshot) -> None:
        try:
            destination = self.query_one("#note-import-destination", Input)
            if destination.value != snapshot.destination:
                destination.value = snapshot.destination
            self.query_one("#note-import-destination-error", Static).update(
                snapshot.destination_error
            )
            check = self.query_one("#note-import-check", Button)
            check.disabled = not snapshot.can_check
            check.label = _disabled_action_label(
                "Check selection", disabled=not snapshot.can_check
            )
            check.tooltip = snapshot.check_disabled_reason or (
                "Check the selected sources without changing Notes."
            )
        except Exception:
            self.refresh(recompose=True)

    def _sync_collision_controls(self, snapshot: LibraryNoteImportSnapshot) -> None:
        try:
            self.query_one("#note-import-collision-heading", Static).update(
                f"Folder collision: {snapshot.collision_name}"
            )
            self.query_one("#note-import-collision-reason", Static).update(
                snapshot.collision_reason
            )
            rename_input = self.query_one("#note-import-collision-name", Input)
            visible_name = snapshot.collision_rename_input or snapshot.collision_name
            if rename_input.value != visible_name:
                rename_input.value = visible_name
            self.query_one("#note-import-collision-rename-error", Static).update(
                snapshot.collision_rename_error
            )
            rename = self.query_one("#note-import-collision-rename", Button)
            rename.disabled = not snapshot.collision_rename_available
            for choice, button_id, label in (
                ("use_existing", "use-existing", "Use existing folder"),
                ("unique_sibling", "unique", "Create a unique sibling"),
                ("renamed_root", "rename", "Use another name"),
            ):
                button = self.query_one(f"#note-import-collision-{button_id}", Button)
                button.label = _choice_label(
                    selected=snapshot.collision_choice == choice,
                    text=label,
                )
            submit = self.query_one("#note-import-import", Button)
            submit.disabled = not snapshot.can_import
            submit.label = _disabled_action_label(
                "Import selected items", disabled=not snapshot.can_import
            )
            submit.tooltip = snapshot.import_disabled_reason or (
                "Import the exact choices shown in this review."
            )
        except Exception:
            self.refresh(recompose=True)

    def on_mount(self) -> None:
        self.call_after_refresh(self._update_overflow_hint)

    def _after_recompose(self) -> None:
        self.call_after_refresh(self._update_overflow_hint)

    def _update_overflow_hint(self) -> None:
        try:
            body = self.query_one("#note-import-body", VerticalScroll)
            hint = self.query_one("#note-import-overflow-hint", Static)
            hint.display = body.virtual_size.height > body.container_size.height
        except Exception:
            return

    def compose(self) -> ComposeResult:
        state = self.snapshot
        yield Static(
            "Import once",
            id="note-import-heading",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            state.status_line,
            id="note-import-status",
            markup=False,
        )
        with _ImportBody(id="note-import-body"):
            if state.phase in {"select", "destination"}:
                yield from self._compose_selection(state)
            elif state.phase == "review":
                yield from self._compose_review(state)
            elif state.phase == "importing":
                yield from self._compose_importing(state)
            elif state.phase == "receipt":
                yield from self._compose_receipt(state)
        hint = Static(
            "More below — focus this panel and use Up/Down to scroll.",
            id="note-import-overflow-hint",
            classes="note-import-quiet",
            markup=False,
        )
        hint.display = False
        yield hint
        yield from self._compose_primary_action(state)

    def _compose_primary_action(
        self, state: LibraryNoteImportSnapshot
    ) -> ComposeResult:
        if state.phase in {"select", "destination"}:
            check = Button(
                _disabled_action_label("Check selection", disabled=not state.can_check),
                id="note-import-check",
                classes="library-canvas-action note-import-primary",
                compact=True,
                disabled=not state.can_check,
            )
            check.tooltip = state.check_disabled_reason or (
                "Check the selected sources without changing Notes."
            )
            yield check
        elif state.phase in {"checking", "importing"}:
            yield Button(
                (
                    "Cancel check"
                    if state.phase == "checking" and state.can_cancel
                    else "Cancel import"
                    if state.can_cancel
                    else "Stopping…"
                ),
                id="note-import-cancel",
                classes="library-canvas-action note-import-primary",
                compact=True,
                disabled=not state.can_cancel,
            )
        elif state.phase == "review":
            submit = Button(
                _disabled_action_label(
                    "Import selected items", disabled=not state.can_import
                ),
                id="note-import-import",
                classes="library-canvas-action note-import-primary",
                compact=True,
                disabled=not state.can_import,
            )
            submit.tooltip = state.import_disabled_reason or (
                "Import the exact choices shown in this review."
            )
            yield submit
        elif state.phase == "receipt" and (
            state.retry_available or state.retryable_failures
        ):
            noun = "failure" if state.retryable_failures == 1 else "failures"
            yield Button(
                state.retry_label or f"Retry {state.retryable_failures} {noun}",
                id="note-import-retry",
                classes="library-canvas-action note-import-primary",
                compact=True,
            )

    def _compose_selection(self, state: LibraryNoteImportSnapshot) -> ComposeResult:
        count = len(state.selected_names)
        if not count:
            source_copy = "No source selected."
        elif state.selection_kind == "folder":
            source_copy = (
                f"1 folder selected: {_bounded_source_name(state.selected_names[0])}"
            )
        else:
            noun = "file" if count == 1 else "files"
            visible_names = tuple(
                _bounded_source_name(name) for name in state.selected_names[:3]
            )
            remainder = count - len(visible_names)
            more = f"; and {remainder} more" if remainder else ""
            source_copy = f"{count} {noun} selected: {', '.join(visible_names)}{more}"
        yield Static(
            source_copy,
            id="note-import-source-summary",
            markup=False,
        )

        if state.selection_kind != "folder":
            yield Button(
                "Add another file" if count else "Choose a file or folder",
                id="note-import-add-source",
                classes="library-canvas-action",
                compact=True,
            )
        if state.selection_kind == "files":
            yield Static(
                "Notes destination",
                id="note-import-destination-label",
                markup=False,
            )
            yield Input(
                value=state.destination,
                placeholder="Existing or new folder path",
                id="note-import-destination",
            )
            yield Static(
                state.destination_error,
                id="note-import-destination-error",
                classes="note-import-error",
                markup=False,
            )

    def _compose_review(self, state: LibraryNoteImportSnapshot) -> ComposeResult:
        if state.collision_kind:
            yield Static(
                f"Folder collision: {state.collision_name}",
                id="note-import-collision-heading",
                classes="note-import-item-name",
                markup=False,
            )
            yield Static(
                state.collision_reason,
                id="note-import-collision-reason",
                markup=False,
            )
            for choice, button_id, label in (
                ("use_existing", "use-existing", "Use existing folder"),
                ("unique_sibling", "unique", "Create a unique sibling"),
                ("renamed_root", "rename", "Use another name"),
            ):
                yield Button(
                    _choice_label(
                        selected=state.collision_choice == choice, text=label
                    ),
                    id=f"note-import-collision-{button_id}",
                    name=choice,
                    classes="library-canvas-action note-import-collision-choice",
                    compact=True,
                    disabled=(
                        choice == "renamed_root"
                        and not state.collision_rename_available
                    ),
                )
            yield Input(
                value=state.collision_rename_input or state.collision_name,
                placeholder="New top-level folder name",
                id="note-import-collision-name",
            )
            yield Static(
                state.collision_rename_error,
                id="note-import-collision-rename-error",
                classes="note-import-error",
                markup=False,
            )

        order = tuple(_CLASSIFICATION_LABELS)
        sorted_items = sorted(
            state.preview_items,
            key=lambda item: order.index(item.classification),
        )
        dom_tokens = {
            item.item_id: f"item-{index}"
            for index, item in enumerate(sorted_items, start=1)
        }
        for classification, grouped in groupby(
            sorted_items,
            key=lambda item: item.classification,
        ):
            items = tuple(grouped)
            yield Static(
                f"{_CLASSIFICATION_LABELS[classification]} ({len(items)})",
                classes="note-import-group-heading",
                markup=False,
            )
            for item in items:
                yield from self._compose_review_item(item, dom_tokens[item.item_id])

        if state.page_count > 1:
            previous = Button(
                "Previous page",
                id="note-import-page-previous",
                classes="library-canvas-action",
                compact=True,
                disabled=state.page <= 1,
            )
            yield previous
            yield Static(
                f"Page {state.page} of {state.page_count}",
                id="note-import-page",
                markup=False,
            )
            next_button = Button(
                "Next page",
                id="note-import-page-next",
                classes="library-canvas-action",
                compact=True,
                disabled=state.page >= state.page_count,
            )
            yield next_button

    def _compose_review_item(
        self,
        item: LibraryNoteImportItemSnapshot,
        dom_token: str,
    ) -> ComposeResult:
        yield Static(
            item.name,
            classes="note-import-item-name",
            markup=False,
        )
        if item.reason:
            yield Static(
                item.reason,
                classes="note-import-quiet",
                markup=False,
            )
        for detail in (
            item.target_label,
            item.effect_summary,
            item.membership_summary,
            item.content_diff,
        ):
            if detail:
                yield Static(
                    detail,
                    classes="note-import-quiet",
                    markup=False,
                )

        yield Button(
            _choice_label(selected=item.action == "skip", text="Skip"),
            id=f"note-import-action-{dom_token}-skip",
            name=f"{item.item_id}:skip",
            classes="library-canvas-action note-import-item-action",
            compact=True,
        )
        if item.classification not in {"unsupported", "failed"}:
            yield Button(
                _choice_label(selected=item.action == "create_new", text="Create new"),
                id=f"note-import-action-{dom_token}-create",
                name=f"{item.item_id}:create_new",
                classes="library-canvas-action note-import-item-action",
                compact=True,
            )
        if item.can_update or item.uncertain:
            update = Button(
                _choice_label(
                    selected=item.action == "update_existing",
                    text="Update existing",
                ),
                id=f"note-import-action-{dom_token}-update",
                name=f"{item.item_id}:update_existing",
                classes="library-canvas-action note-import-item-action",
                compact=True,
                disabled=not item.can_update,
            )
            update.tooltip = (
                "Update the confirmed existing note."
                if item.can_update
                else "Confirm the match before updating."
            )
            yield update

        if item.uncertain and not item.confirmed:
            yield Button(
                "Confirm this match",
                id=f"note-import-confirm-{dom_token}",
                name=item.item_id,
                classes="library-canvas-action note-import-confirm-match",
                compact=True,
            )
        if item.action == "update_existing":
            yield Button(
                _choice_label(
                    selected=item.replace_content,
                    text="Replace note content",
                ),
                id=f"note-import-replace-{dom_token}",
                name=f"{item.item_id}:replace_content",
                classes="library-canvas-action note-import-item-choice",
                compact=True,
            )
            yield Button(
                _choice_label(
                    selected=item.add_membership,
                    text="Add folder placement",
                ),
                id=f"note-import-membership-{dom_token}",
                name=f"{item.item_id}:add_membership",
                classes="library-canvas-action note-import-item-choice",
                compact=True,
            )

    def _compose_importing(self, state: LibraryNoteImportSnapshot) -> ComposeResult:
        detail = f" · {state.progress_detail}" if state.progress_detail else ""
        yield Static(
            f"{state.progress_completed} of {state.progress_total} complete{detail}",
            id="note-import-progress",
            markup=False,
        )

    def _compose_receipt(self, state: LibraryNoteImportSnapshot) -> ComposeResult:
        yield Static(
            state.receipt_line,
            id="note-import-receipt",
            markup=False,
        )
        yield Static(
            state.receipt_detail,
            id="note-import-receipt-detail",
            classes="note-import-quiet",
            markup=False,
        )

    @on(Input.Changed, "#note-import-destination")
    def _destination_changed(self, event: Input.Changed) -> None:
        if event.value == self._destination_value:
            return
        self._destination_value = event.value
        self.post_message(self.DestinationChanged(event.value))

    @on(Input.Changed, "#note-import-collision-name")
    def _collision_name_changed(self, event: Input.Changed) -> None:
        if event.value == self._collision_name:
            return
        self._collision_name = event.value
        self.post_message(self.CollisionNameChanged(event.value))

    @on(Button.Pressed, "#note-import-add-source")
    def _add_source(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.AddSourceRequested())

    @on(Button.Pressed, "#note-import-check")
    def _check(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.CheckRequested())

    @on(Button.Pressed, ".note-import-collision-choice")
    def _choose_collision(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.name:
            self.post_message(self.CollisionChoiceRequested(event.button.name))

    @on(Button.Pressed, ".note-import-item-action")
    def _choose_item_action(self, event: Button.Pressed) -> None:
        event.stop()
        item_id, separator, action = (event.button.name or "").rpartition(":")
        if separator and item_id and action:
            self.post_message(self.ItemActionRequested(item_id, action))

    @on(Button.Pressed, ".note-import-item-choice")
    def _choose_item_effect(self, event: Button.Pressed) -> None:
        event.stop()
        item_id, separator, choice = (event.button.name or "").rpartition(":")
        if not separator or not item_id or not choice:
            return
        current = next(
            (item for item in self.snapshot.preview_items if item.item_id == item_id),
            None,
        )
        if current is None:
            return
        enabled = (
            not current.replace_content
            if choice == "replace_content"
            else not current.add_membership
        )
        self.post_message(self.ItemChoiceRequested(item_id, choice, enabled))

    @on(Button.Pressed, ".note-import-confirm-match")
    def _confirm_match(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.name:
            self.post_message(self.UncertainMatchConfirmed(event.button.name))

    @on(Button.Pressed, "#note-import-import")
    def _import(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.ImportRequested())

    @on(Button.Pressed, "#note-import-cancel")
    def _cancel(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.CancelRequested())

    @on(Button.Pressed, "#note-import-retry")
    def _retry(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.RetryRequested())

    @on(Button.Pressed, "#note-import-page-previous")
    def _previous_page(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.PageRequested(-1))

    @on(Button.Pressed, "#note-import-page-next")
    def _next_page(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.PageRequested(1))
