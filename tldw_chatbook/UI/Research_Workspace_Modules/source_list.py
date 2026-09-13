"""Stable paged source rows for the Research Sources pane."""

from __future__ import annotations

from collections.abc import Mapping

from textual import on
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Static

from ...Research_Workspace import (
    ResearchCapability,
    ResearchSourcePage,
    ResearchSourceSummary,
    SourceReadiness,
    WorkspaceDataSource,
)


MAX_VISIBLE_SOURCE_ROWS = 25


def _desired_owner_id(source: ResearchSourceSummary) -> str:
    return (
        source.catalog_item_id
        if source.ref.data_source is WorkspaceDataSource.LOCAL
        else source.source_id
    )


class _ResearchSourceRowSlot(Vertical):
    """One recyclable source row.

    The child widgets are built at construction and kept as direct
    references, so ``sync_source`` stays synchronous (and cheap) even while
    the slot's subtree is still being mounted by the message loop
    (TASK-23024): slots are mounted on demand as rows arrive, and the first
    ``sync_source`` for a fresh slot runs before its children hit the DOM.
    """

    def __init__(self, index: int) -> None:
        title = Static("", id=f"research-source-row-title-{index}")
        select = Button(
            "Select",
            id=f"research-source-row-select-{index}",
            compact=True,
        )
        badges = Static("", id=f"research-source-row-badges-{index}")
        selection = Static("", id=f"research-source-row-selection-{index}")
        readiness = Static("", id=f"research-source-row-readiness-{index}")
        details = Button(
            "Details", id=f"research-source-row-details-{index}", compact=True
        )
        folders = Button(
            "Folders", id=f"research-source-row-folders-{index}", compact=True
        )
        preview = Button(
            "Preview / annotate",
            id=f"research-source-row-preview-{index}",
            compact=True,
        )
        move_copy = Button(
            "Move / Copy",
            id=f"research-source-row-copy-{index}",
            compact=True,
            disabled=True,
        )
        up = Button(
            "^",
            id=f"research-source-row-up-{index}",
            name="Move source up",
            tooltip="Move source up",
            compact=True,
            disabled=True,
        )
        down = Button(
            "v",
            id=f"research-source-row-down-{index}",
            name="Move source down",
            tooltip="Move source down",
            compact=True,
            disabled=True,
        )
        remove = Button(
            "Remove", id=f"research-source-row-remove-{index}", compact=True
        )
        super().__init__(
            Horizontal(title, select, classes="research-source-row-heading"),
            badges,
            selection,
            readiness,
            Horizontal(
                details,
                folders,
                preview,
                move_copy,
                up,
                down,
                remove,
                classes="research-source-row-actions",
            ),
            id=f"research-source-row-{index}",
        )
        self.index = index
        self.source: ResearchSourceSummary | None = None
        self.desired_owner_id = ""
        self.desired_selected = False
        self._title = title
        self._select = select
        self._badges = badges
        self._selection = selection
        self._readiness = readiness
        self._preview = preview
        self._move_copy = move_copy
        self._up = up
        self._down = down
        self._remove = remove

    def sync_source(
        self,
        source: ResearchSourceSummary | None,
        *,
        desired_ids: frozenset[str],
        readiness: SourceReadiness | None,
        folder_selected: bool,
        selection_available: bool,
        selection_reason: str,
        preview_available: bool,
        preview_reason: str,
        remove_available: bool,
        remove_reason: str,
        reorder_available: bool,
        reorder_reason: str,
        move_copy_available: bool,
        row_count: int,
    ) -> None:
        self.source = source
        self.display = source is not None
        if source is None:
            return
        self.desired_owner_id = _desired_owner_id(source)
        self.desired_selected = self.desired_owner_id in desired_ids
        direct = self.desired_selected
        badge = (
            "Direct + folder"
            if direct and folder_selected
            else (
                "From folder"
                if folder_selected
                else ("Direct" if direct else "Not selected")
            )
        )
        readiness_label = (
            readiness.state.value.replace("_", " ").title()
            if readiness is not None
            else "Unavailable"
        )
        self._title.update(source.title)
        self._badges.update(f"{source.source_type.title()} · {badge}")
        self._selection.update(f"Selected intent: {'Yes' if direct else 'No'}")
        self._readiness.update(f"Readiness: {readiness_label}")
        select = self._select
        select.label = "Deselect" if direct else "Select"
        select.disabled = not selection_available
        select.tooltip = (
            "Change selected intent" if selection_available else selection_reason
        )
        preview = self._preview
        preview.disabled = not preview_available
        preview.tooltip = (
            "Preview and annotate" if preview_available else preview_reason
        )
        remove = self._remove
        remove.disabled = not remove_available
        remove.tooltip = (
            "Remove workspace association" if remove_available else remove_reason
        )
        move_copy = self._move_copy
        move_copy.disabled = not move_copy_available
        move_copy.tooltip = (
            "Move or copy this source"
            if move_copy_available
            else "The selected owner exposes no canonical Move / Copy action."
        )
        up = self._up
        down = self._down
        up.disabled = not (reorder_available and self.index > 0)
        down.disabled = not (reorder_available and self.index + 1 < row_count)
        up.tooltip = "Move source up" if reorder_available else reorder_reason
        down.tooltip = "Move source down" if reorder_available else reorder_reason


class ResearchSourceList(Vertical):
    """Demand-grown slot pool; page refreshes never rebuild the region.

    The pool starts empty (an unused Research profile pays for zero row
    widgets, TASK-23024) and grows monotonically with the largest page seen,
    capped at ``MAX_VISIBLE_SOURCE_ROWS``. Slots are never unmounted:
    shrinking a page recycles surplus slots via ``display = False`` exactly
    as the fully pre-mounted pool did, so paging at the maximum row count
    still mounts and unmounts nothing.
    """

    class SelectionToggled(Message):
        def __init__(
            self, source_id: str, desired_owner_id: str, selected: bool
        ) -> None:
            super().__init__()
            self.source_id = source_id
            self.desired_owner_id = desired_owner_id
            self.selected = selected

    class ActionRequested(Message):
        def __init__(self, action: str, source_id: str) -> None:
            super().__init__()
            self.action = action
            self.source_id = source_id

    class ReorderRequested(Message):
        def __init__(self, source_id: str, delta: int) -> None:
            super().__init__()
            self.source_id = source_id
            self.delta = delta

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._slots: list[_ResearchSourceRowSlot] = []

    def _ensure_slot_pool(self, needed: int) -> None:
        """Grow the mounted slot pool to ``needed`` rows (never shrink)."""

        needed = min(needed, MAX_VISIBLE_SOURCE_ROWS)
        if needed <= len(self._slots):
            return
        new_slots = [
            _ResearchSourceRowSlot(index)
            for index in range(len(self._slots), needed)
        ]
        self._slots.extend(new_slots)
        if self.is_attached:
            self.mount(*new_slots)

    def on_mount(self) -> None:
        detached = [slot for slot in self._slots if slot.parent is None]
        if detached:
            self.mount(*detached)
        self.sync_page(None)

    def sync_page(
        self,
        page: ResearchSourcePage | None,
        *,
        readiness: tuple[SourceReadiness, ...] = (),
        folder_source_ids: frozenset[str] = frozenset(),
        capabilities: Mapping[str, ResearchCapability] | None = None,
        temporary_sort: bool = False,
    ) -> None:
        rows = page.items if page is not None else ()
        self._ensure_slot_pool(len(rows))
        desired_ids = frozenset(page.desired_source_ids if page is not None else ())
        readiness_by_id = {item.source_id: item for item in readiness}
        capabilities = capabilities or {}
        selection = capabilities.get("set_selected_scope")
        preview = capabilities.get("preview_source")
        remove = capabilities.get("remove_source")
        reorder = capabilities.get("reorder_sources")
        reorder_available = bool(reorder and reorder.available and not temporary_sort)
        # Neither canonical owner currently exposes a source move/copy mutation.
        # Keep the visible parity control honest even if an unknown capability key
        # is accidentally projected into this view.
        move_copy_available = False
        for index, slot in enumerate(self._slots):
            source = rows[index] if index < len(rows) else None
            slot.sync_source(
                source,
                desired_ids=desired_ids,
                readiness=(
                    readiness_by_id.get(source.source_id)
                    if source is not None
                    else None
                ),
                folder_selected=(
                    source is not None and source.source_id in folder_source_ids
                ),
                selection_available=bool(selection and selection.available),
                selection_reason=(
                    selection.user_message
                    if selection is not None
                    else "Selection capability is unavailable."
                ),
                preview_available=bool(preview and preview.available),
                preview_reason=(
                    preview.user_message
                    if preview is not None
                    else "Preview capability is unavailable."
                ),
                remove_available=bool(remove and remove.available),
                remove_reason=(
                    remove.user_message
                    if remove is not None
                    else "Remove capability is unavailable."
                ),
                reorder_available=reorder_available,
                reorder_reason=(
                    "Manual reorder is disabled while a temporary sort is active."
                    if temporary_sort
                    else (
                        reorder.user_message
                        if reorder is not None
                        else "Reorder capability is unavailable."
                    )
                ),
                move_copy_available=move_copy_available,
                row_count=len(rows),
            )

    def _slot_for_button(self, button: Button) -> _ResearchSourceRowSlot | None:
        current = button.parent
        while current is not None and not isinstance(current, _ResearchSourceRowSlot):
            current = current.parent
        return current if isinstance(current, _ResearchSourceRowSlot) else None

    @on(Button.Pressed, ".research-source-row-heading Button")
    def toggle_selection(self, event: Button.Pressed) -> None:
        slot = self._slot_for_button(event.button)
        if slot is not None and slot.source is not None:
            self.post_message(
                self.SelectionToggled(
                    slot.source.source_id,
                    slot.desired_owner_id,
                    not slot.desired_selected,
                )
            )

    @on(Button.Pressed, ".research-source-row-actions Button")
    def request_action(self, event: Button.Pressed) -> None:
        slot = self._slot_for_button(event.button)
        if slot is None or slot.source is None:
            return
        suffix = str(event.button.id or "").rsplit("-", 2)[-2]
        if suffix in {"up", "down"}:
            self.post_message(
                self.ReorderRequested(
                    slot.source.source_id, -1 if suffix == "up" else 1
                )
            )
            return
        self.post_message(self.ActionRequested(suffix, slot.source.source_id))
