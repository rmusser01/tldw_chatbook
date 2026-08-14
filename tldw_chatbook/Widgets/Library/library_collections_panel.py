"""Textual widget for Library Collections management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Collapsible, Input, Static
from textual.widgets._input import Selection

from ...Library.library_collections_state import LibraryCollectionsPanelState
from ...Library.library_shell_state import library_disabled_action_label
from .library_canvas_sync import PostRecomposeCallback


LIBRARY_COLLECTIONS_STATUS_LINE = (
    "Collections hold saved items for review — adding items is coming; "
    "you can create and name collections now."
)


def _compact_receipt_name(value: str, limit: int = 42) -> str:
    """Keep a Collection name literal, single-line, and bounded."""
    normalized = " ".join(value.splitlines()).strip() or "Untitled"
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class _CollectionsInputCapture:
    """Portable state for one focused Input replaced by panel recompose."""

    widget_id: str
    value: str
    selection: Selection
    select_on_focus: bool
    outgoing_focus: Input


class LibraryCollectionsPanel(PostRecomposeCallback, Vertical):
    """Render-only Library Collections list, detail, and form controls."""

    def __init__(
        self,
        state: LibraryCollectionsPanelState,
        *,
        name_value: str = "",
        description_value: str = "",
        delete_pending: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.state = state
        self.name_value = name_value
        self.description_value = description_value
        self.delete_pending = delete_pending
        self._pending_input_capture: _CollectionsInputCapture | None = None

    def sync_state(
        self,
        state: LibraryCollectionsPanelState,
        *,
        name_value: str,
        description_value: str,
        delete_pending: bool,
        deferred_guard: Callable[[], bool] | None = None,
    ) -> None:
        """Synchronize every compose input on the retained Collections owner."""
        captured_now = self._capture_focused_input()
        if captured_now is not None:
            self._pending_input_capture = captured_now
        capture = self._pending_input_capture
        if capture is not None:
            if capture.widget_id == "library-collection-name-input":
                name_value = capture.value
            elif capture.widget_id == "library-collection-description-input":
                description_value = capture.value
        self.state = state
        self.name_value = name_value
        self.description_value = description_value
        self.delete_pending = delete_pending
        self.queue_after_recompose(
            None
            if capture is None
            else lambda: self._restore_pending_focused_input(
                capture, deferred_guard
            )
        )
        self.refresh(recompose=True)

    def _restore_pending_focused_input(
        self,
        capture: _CollectionsInputCapture,
        deferred_guard: Callable[[], bool] | None,
    ) -> None:
        """Consume the latest focus capture after coalesced panel syncs."""
        if self._pending_input_capture is not capture:
            return
        self._pending_input_capture = None
        self._restore_focused_input(capture, deferred_guard)

    def _capture_focused_input(self) -> _CollectionsInputCapture | None:
        """Detach and capture the focused form Input before replacing it."""
        if not self.is_attached:
            return None
        focused = self.screen.focused
        if (
            not isinstance(focused, Input)
            or self not in focused.ancestors_with_self
            or focused.disabled
            or not focused.id
        ):
            return None
        capture = _CollectionsInputCapture(
            widget_id=str(focused.id),
            value=focused.value,
            selection=focused.selection,
            select_on_focus=focused.select_on_focus,
            outgoing_focus=focused,
        )
        self.screen.set_focus(None)
        return capture

    def _restore_focused_input(
        self,
        capture: _CollectionsInputCapture,
        deferred_guard: Callable[[], bool] | None,
    ) -> None:
        """Restore one current, enabled Input without stealing later focus."""
        if deferred_guard is not None and not deferred_guard():
            return
        if not self.is_attached or getattr(self, "_pruning", False):
            return
        try:
            panels = list(self.screen.query("#library-collections-panel"))
            target = self.query_one(f"#{capture.widget_id}", Input)
        except (NoMatches, QueryError):
            return
        if len(panels) != 1 or panels[0] is not self or target.disabled:
            return
        if self.screen.focused not in (None, capture.outgoing_focus, target):
            return
        target.value = capture.value
        target.select_on_focus = False
        self.screen.set_focus(target)
        target.selection = capture.selection
        target.call_later(
            self._restore_input_selection,
            target,
            capture,
            deferred_guard,
        )

    def _restore_input_selection(
        self,
        target: Input,
        capture: _CollectionsInputCapture,
        deferred_guard: Callable[[], bool] | None,
    ) -> None:
        """Restore selection after Input's own focus handler has settled."""
        target.select_on_focus = capture.select_on_focus
        if deferred_guard is not None and not deferred_guard():
            return
        if not target.is_attached or self.screen.focused is not target:
            return
        target.selection = capture.selection

    def _compose_collection_form(self) -> ComposeResult:
        with Vertical(id="library-collection-form"):
            yield Static("Create / Rename", classes="destination-section")
            if not self.state.create_action.enabled:
                # A single sentence replaces the three that used to repeat
                # the same "enter a name" rule; it disappears entirely once
                # a valid, non-duplicate name makes Create available (kept
                # in sync in place by
                # `_refresh_collections_panel_action_state_widgets`).
                yield Static(
                    self.state.create_action.disabled_reason
                    or "Enter a Collection name to enable Create.",
                    id="library-collection-form-guidance",
                )
            yield Input(
                value=self.name_value,
                placeholder="Collection name",
                id="library-collection-name-input",
            )
            yield Input(
                value=self.description_value,
                placeholder="Optional description",
                id="library-collection-description-input",
            )
            with Horizontal(id="library-collection-actions"):
                # task-4023 AC#1 (RC-07): these three measured 2.30:1 while
                # disabled, with colour the only state carrier. The "○"
                # marker is the non-colour half; the state's F-018 tooltip
                # (its ``disabled_reason``) and the guidance line above are
                # the reason at the control. The screen's in-place patcher
                # (`_refresh_collections_panel_action_state_widgets`)
                # rebuilds the same marker label when it flips ``disabled``.
                for action in (
                    self.state.create_action,
                    self.state.rename_action,
                    self.state.delete_action,
                ):
                    yield Button(
                        library_disabled_action_label(
                            action.label, not action.enabled
                        ),
                        id=action.widget_id,
                        disabled=not action.enabled,
                        tooltip=action.tooltip,
                        classes=(
                            "library-source-action library-collection-form-action"
                        ),
                    )
                if self.delete_pending:
                    yield Button(
                        "Confirm delete",
                        id="library-confirm-delete-collection",
                        tooltip=(
                            "Delete the selected local Collection. Its items "
                            "stay in the Library. Undo will be available in "
                            "this Collections panel."
                        ),
                        disabled=self.state.mutation_in_flight,
                        classes="library-source-action library-collection-form-action",
                    )

    def compose(self) -> ComposeResult:
        # task-2859 item 7: match the sibling "Name (n)" pattern
        # (Media/Notes/Prompts/Skills) and drop the "Library " prefix --
        # the canvas already lives inside the Library destination, so
        # "Library Collections" restated the destination twice.
        yield Static(
            f"Collections ({len(self.state.collections)})",
            id="library-collections-title",
            classes="destination-section",
            markup=False,
        )
        receipt = self.state.delete_receipt
        if receipt is not None:
            receipt_row = Horizontal(
                id="library-collections-delete-receipt", classes="ds-toolbar"
            )
            receipt_row.styles.height = "auto"
            with receipt_row:
                yield Static(
                    "✓ deleted · Collection · "
                    f"{_compact_receipt_name(receipt.name)}",
                    id="library-collections-delete-receipt-copy",
                    classes="library-toolbar-count",
                    markup=False,
                )
                yield Button(
                    "Undo",
                    id="library-collections-delete-undo",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.state.mutation_in_flight,
                )
                yield Button(
                    "Dismiss",
                    id="library-collections-delete-receipt-dismiss",
                    classes="library-canvas-action",
                    compact=True,
                    disabled=self.state.mutation_in_flight,
                )
        if self.state.status == "error":
            yield Static(
                self.state.recovery_copy or self.state.error_message,
                id="library-collections-error",
            )
            return

        if self.state.status == "empty":
            # task-4023 AC#7: the empty state stacked FOUR "nothing here"
            # sentences (headline, next-action, purpose dump, and a
            # meaningless "No Collection selected." with zero collections
            # in existence). Two lines now: the fact, then one sentence
            # combining purpose + next action (``state.empty_copy``).
            yield Static(
                "No Collections yet.",
                id="library-collections-empty-title",
                classes="destination-section",
            )
            yield Static(self.state.empty_copy, id="library-collections-empty")
            yield from self._compose_collection_form()
            return

        if self.state.sync_profile_status is not None:
            with Vertical(
                id="library-sync-profile-status-banner",
                classes=f"sync-profile-status {self.state.sync_profile_status.severity}",
            ):
                yield Static(
                    self.state.sync_profile_status.label,
                    id="library-sync-profile-status",
                    markup=False,
                )
                yield Static(
                    self.state.sync_profile_status.detail,
                    id="library-sync-profile-detail",
                    markup=False,
                )
                yield Static(
                    self.state.sync_profile_status.read_only_notice,
                    id="library-sync-profile-read-only",
                    markup=False,
                )

        with Horizontal(id="library-collections-workbench"):
            with Vertical(id="library-collections-list"):
                yield Static("Collections", classes="destination-section")
                for index, collection in enumerate(self.state.collections):
                    # task-4023 AC#5: the selected collection was marked by
                    # colour alone (`is-active`); every other Library list
                    # (rail, media, conversations) leads its selected row
                    # with "▸ ". One marker vocabulary.
                    marker = "▸" if collection.selected else " "
                    label = (
                        f"{marker} {collection.name} - {collection.item_count_label}"
                    )
                    button = Button(
                        label,
                        id=f"library-collection-select-{index}",
                        classes="library-collection-row",
                        tooltip=collection.sync_status_label,
                    )
                    button.collection_id = collection.collection_id
                    if collection.selected:
                        button.add_class("is-active")
                    yield button

            with Vertical(id="library-collection-detail"):
                yield Static("Stored collection content", classes="destination-section")
                selected = self.state.selected_collection
                if selected is None:
                    yield Static(
                        "No Collection selected.",
                        id="library-collection-selected-empty",
                    )
                else:
                    yield Static(
                        f"Selected: {selected.name}",
                        id="library-collection-selected-context",
                    )
                    yield Static(selected.name, id="library-collection-name")
                    yield Static(
                        selected.description or "No description.",
                        id="library-collection-description",
                    )
                    yield Static(
                        LIBRARY_COLLECTIONS_STATUS_LINE,
                        id="library-collection-status-line",
                    )
                    yield Static("Action status", classes="destination-section")
                    yield Static(
                        "Available now: create, rename, delete records",
                        id="library-collection-local-actions",
                    )
                    with Collapsible(
                        title="Details",
                        collapsed=True,
                        id="library-collection-details",
                    ):
                        yield Static(
                            selected.item_count_label,
                            id="library-collection-item-count",
                        )
                        yield Static(
                            selected.sync_status_label,
                            id="library-collection-sync-status",
                        )
                        if (
                            selected.sync_status != "local-only"
                            or selected.sync_status_label != "Sync: local-only"
                        ):
                            yield Static(
                                selected.sync_status_detail,
                                id="library-collection-sync-detail",
                            )
                        yield Static(
                            selected.updated_at_label,
                            id="library-collection-updated-at",
                        )

        if self.state.status != "empty":
            yield from self._compose_collection_form()
