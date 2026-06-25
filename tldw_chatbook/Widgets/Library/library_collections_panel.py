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

    def _compose_collection_form(self) -> ComposeResult:
        with Vertical(id="library-collection-form"):
            yield Static("Create / Rename", classes="destination-section")
            yield Static(
                "Type a Collection name to enable Create.",
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
            yield Static(
                "Form actions: enter a name to enable Create.",
                id="library-collection-form-action-state",
            )
            yield Static(
                "Create, Rename, and Delete stay inactive until their requirements are met.",
                id="library-collection-form-action-boundary",
            )
            with Horizontal(id="library-collection-actions"):
                yield Button(
                    self.state.create_action.label,
                    id=self.state.create_action.widget_id,
                    disabled=not self.state.create_action.enabled,
                    tooltip=self.state.create_action.tooltip,
                    classes="library-source-action library-collection-form-action",
                )
                yield Button(
                    self.state.rename_action.label,
                    id=self.state.rename_action.widget_id,
                    disabled=not self.state.rename_action.enabled,
                    tooltip=self.state.rename_action.tooltip,
                    classes="library-source-action library-collection-form-action",
                )
                yield Button(
                    self.state.delete_action.label,
                    id=self.state.delete_action.widget_id,
                    disabled=not self.state.delete_action.enabled,
                    tooltip=self.state.delete_action.tooltip,
                    classes="library-source-action library-collection-form-action",
                )
                if self.delete_pending:
                    yield Button(
                        "Confirm delete",
                        id="library-confirm-delete-collection",
                        tooltip="Delete the selected local Collection.",
                        classes="library-source-action library-collection-form-action",
                    )

    def compose(self) -> ComposeResult:
        yield Static("Library Collections", id="library-collections-title", classes="destination-section")
        if self.state.status == "error":
            yield Static(
                self.state.recovery_copy or self.state.error_message,
                id="library-collections-error",
            )
            return

        if self.state.status == "empty":
            yield Static(
                "No Collections yet.",
                id="library-collections-empty-title",
                classes="destination-section",
            )
            yield Static(
                "Create a local Collection record to start reviewing saved content.",
                id="library-collections-empty-next-action",
            )
            yield Static(self.state.empty_copy, id="library-collections-empty")
            yield Static(
                "Stored content preview",
                id="library-collection-empty-reader-title",
                classes="destination-section",
            )
            yield Static(
                "No stored collection items are available locally yet.",
                id="library-collection-empty-reader",
            )
            yield Static("No Collection selected.", id="library-collection-selected-empty")
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
                yield Static("Stored collection content", classes="destination-section")
                selected = self.state.selected_collection
                if selected is None:
                    yield Static("No Collection selected.", id="library-collection-selected-empty")
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
                        "Item reader readiness",
                        id="library-collection-membership-heading",
                        classes="destination-section",
                    )
                    yield Static(
                        f"Stored item count: {selected.item_count_label}",
                        id="library-collection-membership-count",
                    )
                    yield Static(
                        f"Authority: {selected.source_authority}",
                        id="library-collection-source-authority",
                    )
                    yield Static(
                        "Content use boundary",
                        id="library-collection-workspace-heading",
                        classes="destination-section",
                    )
                    yield Static(
                        "Browse/review remains global; active workspace controls staging and manipulation.",
                        id="library-collection-workspace-rule",
                    )
                    yield Static("Action status", classes="destination-section")
                    yield Static(
                        "Available now: create, rename, delete records",
                        id="library-collection-local-actions",
                    )
                    yield Static(
                        "Blocked later: item reader, Search/RAG, Study, Console handoff, server sync",
                        id="library-collection-deferred-actions",
                    )
                    yield Static(
                        "Next: collection item adapters are required before item-level actions unlock.",
                        id="library-collection-reader-later",
                    )
                    yield Static(
                        "Write Sync Safety",
                        id="library-collection-sync-safety-heading",
                        classes="destination-section",
                    )
                    yield Static(
                        "Review these labels before any future server write promotion.",
                        id="library-collection-sync-safety-help",
                    )
                    yield Static(selected.sync_status_label, id="library-collection-sync-status")
                    if selected.sync_status != "local-only" or selected.sync_status_label != "Sync: local-only":
                        yield Static(
                            selected.sync_status_detail,
                            id="library-collection-sync-detail",
                        )
                    yield Static(selected.item_count_label, id="library-collection-item-count")
                    yield Static(selected.updated_at_label, id="library-collection-updated-at")

        if self.state.status != "empty":
            yield from self._compose_collection_form()
