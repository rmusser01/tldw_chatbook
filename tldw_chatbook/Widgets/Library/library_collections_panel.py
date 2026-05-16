"""Textual widget for Library Collections management."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from ...Library.library_collections_state import LibraryCollectionsPanelState


class LibraryCollectionsPanel(Vertical):
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

    def compose(self) -> ComposeResult:
        yield Static("Library Collections", id="library-collections-title", classes="destination-section")
        if self.state.status == "error":
            yield Static(
                self.state.recovery_copy or self.state.error_message,
                id="library-collections-error",
            )
            return

        if self.state.status == "empty":
            yield Static(self.state.empty_copy, id="library-collections-empty")

        with Horizontal(id="library-collections-workbench"):
            with Vertical(id="library-collections-list"):
                yield Static("Collections", classes="destination-section")
                for index, collection in enumerate(self.state.collections):
                    label = f"{collection.name} - {collection.item_count_label}"
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
                yield Static("Selected Collection", classes="destination-section")
                selected = self.state.selected_collection
                if selected is None:
                    yield Static("No Collection selected.", id="library-collection-selected-empty")
                else:
                    yield Static(selected.name, id="library-collection-name")
                    yield Static(
                        selected.description or "No description.",
                        id="library-collection-description",
                    )
                    yield Static(selected.sync_status_label, id="library-collection-sync-status")
                    if selected.sync_status != "local-only":
                        yield Static(
                            selected.sync_status_detail,
                            id="library-collection-sync-detail",
                        )
                    yield Static(selected.item_count_label, id="library-collection-item-count")
                    yield Static(selected.updated_at_label, id="library-collection-updated-at")

        with Vertical(id="library-collection-form"):
            yield Static("Create / Rename", classes="destination-section")
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
                yield Button(
                    self.state.create_action.label,
                    id=self.state.create_action.widget_id,
                    disabled=not self.state.create_action.enabled,
                    tooltip=self.state.create_action.tooltip,
                )
                yield Button(
                    self.state.rename_action.label,
                    id=self.state.rename_action.widget_id,
                    disabled=not self.state.rename_action.enabled,
                    tooltip=self.state.rename_action.tooltip,
                )
                yield Button(
                    self.state.delete_action.label,
                    id=self.state.delete_action.widget_id,
                    disabled=not self.state.delete_action.enabled,
                    tooltip=self.state.delete_action.tooltip,
                )
                if self.delete_pending:
                    yield Button(
                        "Confirm delete",
                        id="library-confirm-delete-collection",
                        tooltip="Delete the selected local Collection.",
                    )
