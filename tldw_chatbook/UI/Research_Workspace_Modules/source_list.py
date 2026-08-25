"""Stable paged source rows for the Research Sources pane."""

from __future__ import annotations

from collections.abc import Mapping

from textual import on
from textual.app import ComposeResult
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
    def __init__(self, index: int) -> None:
        super().__init__(id=f"research-source-row-{index}")
        self.index = index
        self.source: ResearchSourceSummary | None = None
        self.desired_owner_id = ""
        self.desired_selected = False

    def compose(self) -> ComposeResult:
        with Horizontal(classes="research-source-row-heading"):
            yield Static("", id=f"research-source-row-title-{self.index}")
            yield Button(
                "Select",
                id=f"research-source-row-select-{self.index}",
                compact=True,
            )
        yield Static("", id=f"research-source-row-badges-{self.index}")
        yield Static("", id=f"research-source-row-selection-{self.index}")
        yield Static("", id=f"research-source-row-readiness-{self.index}")
        with Horizontal(classes="research-source-row-actions"):
            yield Button(
                "Details", id=f"research-source-row-details-{self.index}", compact=True
            )
            yield Button(
                "Folders", id=f"research-source-row-folders-{self.index}", compact=True
            )
            yield Button(
                "Preview / annotate",
                id=f"research-source-row-preview-{self.index}",
                compact=True,
            )
            yield Button(
                "Move / Copy",
                id=f"research-source-row-copy-{self.index}",
                compact=True,
                disabled=True,
            )
            yield Button(
                "^",
                id=f"research-source-row-up-{self.index}",
                name="Move source up",
                tooltip="Move source up",
                compact=True,
                disabled=True,
            )
            yield Button(
                "v",
                id=f"research-source-row-down-{self.index}",
                name="Move source down",
                tooltip="Move source down",
                compact=True,
                disabled=True,
            )
            yield Button(
                "Remove", id=f"research-source-row-remove-{self.index}", compact=True
            )

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
        self.query_one(f"#research-source-row-title-{self.index}", Static).update(
            source.title
        )
        self.query_one(f"#research-source-row-badges-{self.index}", Static).update(
            f"{source.source_type.title()} · {badge}"
        )
        self.query_one(f"#research-source-row-selection-{self.index}", Static).update(
            f"Selected intent: {'Yes' if direct else 'No'}"
        )
        self.query_one(f"#research-source-row-readiness-{self.index}", Static).update(
            f"Readiness: {readiness_label}"
        )
        select = self.query_one(f"#research-source-row-select-{self.index}", Button)
        select.label = "Deselect" if direct else "Select"
        select.disabled = not selection_available
        select.tooltip = (
            "Change selected intent" if selection_available else selection_reason
        )
        preview = self.query_one(f"#research-source-row-preview-{self.index}", Button)
        preview.disabled = not preview_available
        preview.tooltip = (
            "Preview and annotate" if preview_available else preview_reason
        )
        remove = self.query_one(f"#research-source-row-remove-{self.index}", Button)
        remove.disabled = not remove_available
        remove.tooltip = (
            "Remove workspace association" if remove_available else remove_reason
        )
        move_copy = self.query_one(f"#research-source-row-copy-{self.index}", Button)
        move_copy.disabled = not move_copy_available
        move_copy.tooltip = (
            "Move or copy this source"
            if move_copy_available
            else "The selected owner exposes no canonical Move / Copy action."
        )
        up = self.query_one(f"#research-source-row-up-{self.index}", Button)
        down = self.query_one(f"#research-source-row-down-{self.index}", Button)
        up.disabled = not (reorder_available and self.index > 0)
        down.disabled = not (reorder_available and self.index + 1 < row_count)
        up.tooltip = "Move source up" if reorder_available else reorder_reason
        down.tooltip = "Move source down" if reorder_available else reorder_reason


class ResearchSourceList(Vertical):
    """Twenty-five pre-mounted slots; page refreshes never rebuild the region."""

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

    def compose(self) -> ComposeResult:
        for index in range(MAX_VISIBLE_SOURCE_ROWS):
            yield _ResearchSourceRowSlot(index)

    def on_mount(self) -> None:
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
        for index, slot in enumerate(self.query(_ResearchSourceRowSlot)):
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
