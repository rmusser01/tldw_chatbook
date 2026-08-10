"""Textual widget for Library Collections management."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Collapsible, Input, Static

from ...Library.library_collections_state import LibraryCollectionsPanelState
from ...Library.library_shell_state import library_disabled_action_label


LIBRARY_COLLECTIONS_STATUS_LINE = (
    "Collections hold saved items for review — adding items is coming; "
    "you can create and name collections now."
)


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
                        tooltip="Delete the selected local Collection.",
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
                "No Collection selected.", id="library-collection-selected-empty"
            )
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
