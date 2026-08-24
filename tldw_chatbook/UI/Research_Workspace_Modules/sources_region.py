"""Compose-once Sources workbench for Research Workspace."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Select, Static

from ...Research_Workspace import (
    ResearchCapability,
    ResearchSourcePage,
    SourceReadiness,
)
from ...Research_Workspace.overlay_store import ResearchSourceFolder
from ...Research_Workspace.source_operations import ResearchSourceOperation
from .source_receipt import ResearchSourceReceiptList
from .source_list import ResearchSourceList


_SOURCE_SORT_OPTIONS = (
    ("Manual order", "manual"),
    ("Title A-Z", "title_asc"),
    ("Title Z-A", "title_desc"),
    ("Newest", "updated_desc"),
    ("Oldest", "updated_asc"),
)


class ResearchSourcesRegion(VerticalScroll):
    """Stable Sources controls; async owners patch their content in place."""

    can_focus = True

    class AddRequested(Message):
        """Open authority-qualified intake."""

    class RefreshRequested(Message):
        """Refresh attached source and readiness projections."""

    class QuickUrlRequested(Message):
        def __init__(self, url: str) -> None:
            super().__init__()
            self.url = url

    class SelectionScopeRequested(Message):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

    class PageRequested(Message):
        def __init__(self, delta: int) -> None:
            super().__init__()
            self.delta = delta

    class FolderRequested(Message):
        def __init__(self, action: str, folder_id: str, name: str) -> None:
            super().__init__()
            self.action = action
            self.folder_id = folder_id
            self.name = name

    class BatchRequested(Message):
        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._page: ResearchSourcePage | None = None
        self._readiness: tuple[SourceReadiness, ...] = ()
        self._capabilities: Mapping[str, ResearchCapability] = {}
        self._folders: tuple[ResearchSourceFolder, ...] = ()
        self._focused_folder_id = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="research-source-header"):
            yield Static(
                "Sources", id="research-sources-heading", classes="research-pane-title"
            )
            yield Button(
                "Add Sources", id="research-source-add", variant="primary", compact=True
            )
            yield Button("Refresh", id="research-source-refresh", compact=True)
        yield Static(
            "No workspace selected · choose Local or Server above",
            id="research-source-recovery",
            classes="research-recovery-callout",
        )
        with Horizontal(id="research-source-quick-row"):
            yield Input(placeholder="Quick add URL", id="research-source-quick-url")
            yield Button("Add", id="research-source-quick-submit", compact=True)
        yield Input(placeholder="Search attached sources", id="research-source-search")
        with Horizontal(id="research-source-view-row"):
            yield Button("Advanced", id="research-source-advanced", compact=True)
            yield Select(
                _SOURCE_SORT_OPTIONS,
                value="manual",
                allow_blank=False,
                id="research-source-sort",
            )
        with Vertical(id="research-source-advanced-controls"):
            with Horizontal(classes="research-source-filter-row"):
                yield Select(
                    (
                        ("All status", ""),
                        ("Ready", "ready"),
                        ("Not ready", "not_ready"),
                    ),
                    value="",
                    allow_blank=False,
                    id="research-source-filter-status",
                )
                yield Select(
                    (
                        ("All types", ""),
                        ("PDF", "pdf"),
                        ("Audio", "audio"),
                        ("Video", "video"),
                        ("Website", "website"),
                        ("Text", "text"),
                    ),
                    value="",
                    allow_blank=False,
                    id="research-source-filter-type",
                )
            with Horizontal(classes="research-source-filter-row"):
                yield Select(
                    (
                        ("Any date", ""),
                        ("Today", "today"),
                        ("This week", "week"),
                        ("This month", "month"),
                    ),
                    value="",
                    allow_blank=False,
                    id="research-source-filter-date",
                )
                yield Select(
                    (
                        ("Any selection", ""),
                        ("Direct", "direct"),
                        ("Not selected", "unselected"),
                    ),
                    value="",
                    allow_blank=False,
                    id="research-source-filter-selection",
                )
            yield Static(
                "Owner does not report URL, file-size, duration, or page-count fields for every source; unavailable filters are not guessed.",
                id="research-source-filter-owner-limits",
            )
            yield Button(
                "Clear filters", id="research-source-filter-clear", compact=True
            )
        yield Static(
            "Status/type filters: none · Temporary sort off",
            id="research-source-filter-summary",
        )
        with Horizontal(id="research-source-selection-row"):
            yield Button("Select all", id="research-source-select-all", compact=True)
            yield Button(
                "Select visible", id="research-source-select-visible", compact=True
            )
        with Horizontal(id="research-source-selection-summary"):
            yield Button("Clear", id="research-source-selection-clear", compact=True)
            yield Static("0 selected", id="research-source-selected-count")
        yield Static(
            "Selection unavailable until a workspace owner loads.",
            id="research-source-selection-reason",
        )
        with Horizontal(id="research-source-batch-actions"):
            yield Button("Move / Copy", id="research-source-move-copy", disabled=True)
            yield Button(
                "Preview selected",
                id="research-source-preview-selected",
                disabled=True,
            )
            yield Button("Remove", id="research-source-remove-selected", disabled=True)
        yield Static(
            "Move / Copy — canonical owner unavailable",
            id="research-source-move-copy-reason",
        )
        with Vertical(id="research-source-folders"):
            yield Static(
                "Folders · Device-only organization",
                id="research-source-folders-label",
            )
            with Horizontal(classes="research-source-folder-actions"):
                yield Input(
                    placeholder="Device-only folder name",
                    id="research-source-folder-name",
                )
                yield Button("New", id="research-source-folder-new", compact=True)
                yield Button(
                    "Rename",
                    id="research-source-folder-rename",
                    compact=True,
                    disabled=True,
                )
                yield Button(
                    "Focus",
                    id="research-source-folder-focus",
                    compact=True,
                    disabled=True,
                )
                yield Button(
                    "Select folder sources",
                    id="research-source-select-folder",
                    compact=True,
                    disabled=True,
                )
            yield Select(
                (("No device-only folders yet", ""),),
                value="",
                allow_blank=False,
                id="research-source-folder-tree",
            )
        yield ResearchSourceList(id="research-source-list")
        with Horizontal(id="research-source-page-actions"):
            yield Button("Previous", id="research-source-page-prev", disabled=True)
            yield Static("Page 1 · 0 sources", id="research-source-page-summary")
            yield Button("Next", id="research-source-page-next", disabled=True)
        yield ResearchSourceReceiptList(id="research-source-receipts")
        yield Static(
            "Remove means workspace association only; canonical Library/Media survives.",
            id="research-source-remove-scope",
        )

    @on(Button.Pressed, "#research-source-add")
    def request_add(self) -> None:
        self.post_message(self.AddRequested())

    @on(Button.Pressed, "#research-source-refresh")
    def request_refresh(self) -> None:
        self.post_message(self.RefreshRequested())

    @on(Button.Pressed, "#research-source-quick-submit")
    def request_quick_url(self) -> None:
        raw = self.query_one("#research-source-quick-url", Input).value.strip()
        if raw:
            self.post_message(self.QuickUrlRequested(raw))

    @on(Button.Pressed, "#research-source-advanced")
    def toggle_advanced(self) -> None:
        controls = self.query_one("#research-source-advanced-controls")
        controls.display = not controls.display

    @on(Button.Pressed, "#research-source-filter-clear")
    def clear_filters(self) -> None:
        self.query_one("#research-source-filter-status", Select).value = ""
        self.query_one("#research-source-filter-type", Select).value = ""
        self.query_one("#research-source-filter-date", Select).value = ""
        self.query_one("#research-source-filter-selection", Select).value = ""
        self.query_one("#research-source-search", Input).value = ""
        self._render_page()

    @on(Input.Changed, "#research-source-search")
    @on(Select.Changed, "#research-source-filter-status")
    @on(Select.Changed, "#research-source-filter-type")
    @on(Select.Changed, "#research-source-filter-selection")
    @on(Select.Changed, "#research-source-filter-date")
    @on(Select.Changed, "#research-source-sort")
    def apply_view(self) -> None:
        self._render_page()

    @on(Button.Pressed, "#research-source-select-all")
    def select_all(self) -> None:
        self.post_message(self.SelectionScopeRequested("all"))

    @on(Button.Pressed, "#research-source-select-visible")
    def select_visible(self) -> None:
        self.post_message(self.SelectionScopeRequested("visible"))

    @on(Button.Pressed, "#research-source-selection-clear")
    def clear_selection(self) -> None:
        self.post_message(self.SelectionScopeRequested("clear"))

    @on(Button.Pressed, "#research-source-page-prev")
    def previous_page(self) -> None:
        self.post_message(self.PageRequested(-1))

    @on(Button.Pressed, "#research-source-page-next")
    def next_page(self) -> None:
        self.post_message(self.PageRequested(1))

    @on(Button.Pressed, "#research-source-folder-new")
    @on(Button.Pressed, "#research-source-folder-rename")
    @on(Button.Pressed, "#research-source-folder-focus")
    @on(Button.Pressed, "#research-source-select-folder")
    def folder_action(self, event: Button.Pressed) -> None:
        action = str(event.button.id or "").removeprefix("research-source-folder-")
        folder_id = str(
            self.query_one("#research-source-folder-tree", Select).value or ""
        )
        name = self.query_one("#research-source-folder-name", Input).value.strip()
        self.post_message(self.FolderRequested(action, folder_id, name))

    @on(Button.Pressed, "#research-source-preview-selected")
    @on(Button.Pressed, "#research-source-remove-selected")
    @on(Button.Pressed, "#research-source-move-copy")
    def batch_action(self, event: Button.Pressed) -> None:
        action = str(event.button.id or "").removeprefix("research-source-")
        self.post_message(self.BatchRequested(action))

    def on_mount(self) -> None:
        self.query_one("#research-source-advanced-controls").display = False
        self.set_workspace_available(False, authority="Local")

    def set_workspace_available(self, available: bool, *, authority: str) -> None:
        """Patch empty/recovery state without recomposing stable controls."""

        recovery = self.query_one("#research-source-recovery", Static)
        recovery.update(
            f"Workspace data: {authority} · Refresh sources to recover"
            if available
            else f"No workspace selected · {authority} authority"
        )
        self.query_one("#research-source-add", Button).disabled = not available
        self.query_one("#research-source-refresh", Button).disabled = not available
        self.query_one("#research-source-quick-submit", Button).disabled = not available
        for selector in (
            "#research-source-select-all",
            "#research-source-select-visible",
            "#research-source-selection-clear",
            "#research-source-folder-new",
        ):
            self.query_one(selector, Button).disabled = not available

    def clear_workspace(self, *, authority: str, reason: str = "") -> None:
        """Clear prior owner content immediately on authority/ref switches."""

        self._page = None
        self._readiness = ()
        self._capabilities = {}
        self._folders = ()
        self._focused_folder_id = ""
        self.query_one("#research-source-list", ResearchSourceList).sync_page(None)
        self.query_one(
            "#research-source-receipts", ResearchSourceReceiptList
        ).sync_operations((), incomplete=False)
        self.query_one("#research-source-recovery", Static).update(
            reason or f"No {authority} workspace selected"
        )
        self.query_one("#research-source-selected-count", Static).update("0 selected")
        self.query_one("#research-source-page-summary", Static).update(
            "Page 1 · 0 sources"
        )
        self.query_one("#research-source-page-prev", Button).disabled = True
        self.query_one("#research-source-page-next", Button).disabled = True
        self._sync_folders()
        self._sync_capabilities()
        self.set_workspace_available(False, authority=authority)

    def sync_workspace(
        self,
        page: ResearchSourcePage,
        *,
        readiness: tuple[SourceReadiness, ...],
        capabilities: Mapping[str, ResearchCapability],
        folders: tuple[ResearchSourceFolder, ...],
        operations: tuple[ResearchSourceOperation, ...],
        focused_folder_id: str = "",
        receipts_incomplete: bool = False,
    ) -> None:
        """Patch one qualified owner projection without recomposing controls."""

        self._page = page
        self._readiness = readiness
        self._capabilities = dict(capabilities)
        self._folders = folders
        self._focused_folder_id = focused_folder_id
        authority = (
            page.items[0].ref.data_source.value.title() if page.items else "selected"
        )
        self.set_workspace_available(True, authority=authority)
        self.query_one("#research-source-recovery", Static).update(
            "Sources are current for the selected workspace owner."
        )
        self.query_one("#research-source-selected-count", Static).update(
            f"{len(page.desired_source_ids)} selected"
        )
        current_page = page.offset // page.limit + 1
        self.query_one("#research-source-page-summary", Static).update(
            f"Page {current_page} · {page.total} sources"
        )
        self.query_one("#research-source-page-prev", Button).disabled = page.offset == 0
        self.query_one(
            "#research-source-page-next", Button
        ).disabled = not page.has_more
        self.query_one(
            "#research-source-receipts", ResearchSourceReceiptList
        ).sync_operations(operations, incomplete=receipts_incomplete)
        self._sync_folders()
        self._sync_capabilities()
        self._render_page()

    def visible_owner_ids(self) -> tuple[str, ...]:
        """Return the currently filtered rows' exact desired-selection IDs."""

        source_list = self.query_one("#research-source-list", ResearchSourceList)
        return tuple(
            slot.desired_owner_id
            for slot in source_list.query("_ResearchSourceRowSlot")
            if slot.display and slot.source is not None
        )

    def selected_source_ids(self) -> tuple[str, ...]:
        """Return selected associations present on the current owner page."""

        page = self._page
        if page is None:
            return ()
        desired = frozenset(page.desired_source_ids)
        return tuple(
            source.source_id
            for source in page.items
            if (
                source.catalog_item_id
                if source.ref.data_source.value == "local"
                else source.source_id
            )
            in desired
        )

    def _sync_capabilities(self) -> None:
        page = self._page
        desired_count = len(page.desired_source_ids) if page is not None else 0
        preview = self._capabilities.get("preview_source")
        remove = self._capabilities.get("remove_source")
        selection = self._capabilities.get("set_selected_scope")
        visible_selected_count = len(self.selected_source_ids())
        selection_available = bool(selection and selection.available)
        for selector in (
            "#research-source-select-all",
            "#research-source-select-visible",
            "#research-source-selection-clear",
        ):
            self.query_one(selector, Button).disabled = not selection_available
        self.query_one("#research-source-selection-reason", Static).update(
            (
                f"Selected intent belongs to the {selection.owner.title()} owner."
                if selection_available
                else (
                    f"Selection unavailable · {selection.user_message}"
                    if selection is not None
                    else "Selection unavailable until a workspace owner loads."
                )
            )
        )
        self.query_one("#research-source-preview-selected", Button).disabled = not (
            desired_count == 1
            and visible_selected_count == 1
            and preview is not None
            and preview.available
        )
        self.query_one("#research-source-remove-selected", Button).disabled = not (
            visible_selected_count > 0 and remove is not None and remove.available
        )
        self.query_one("#research-source-move-copy", Button).disabled = True
        self.query_one("#research-source-move-copy-reason", Static).update(
            "Move / Copy — canonical owner action is not exposed here"
        )

    def _sync_folders(self) -> None:
        select = self.query_one("#research-source-folder-tree", Select)
        options = [(folder.name, folder.folder_id) for folder in self._folders]
        select.set_options(options or [("No device-only folders yet", "")])
        if not self._folders:
            select.value = ""
        enabled = bool(self._folders)
        for selector in (
            "#research-source-folder-rename",
            "#research-source-folder-focus",
            "#research-source-select-folder",
        ):
            self.query_one(selector, Button).disabled = not enabled

    def _render_page(self) -> None:
        page = self._page
        if page is None or not self.is_mounted:
            return
        query = (
            self.query_one("#research-source-search", Input).value.strip().casefold()
        )
        source_type = str(
            self.query_one("#research-source-filter-type", Select).value or ""
        )
        selection = str(
            self.query_one("#research-source-filter-selection", Select).value or ""
        )
        status = str(
            self.query_one("#research-source-filter-status", Select).value or ""
        )
        date_filter = str(
            self.query_one("#research-source-filter-date", Select).value or ""
        )
        readiness_by_id = {item.source_id: item for item in self._readiness}
        desired = frozenset(page.desired_source_ids)
        focused_folder = next(
            (
                folder
                for folder in self._folders
                if folder.folder_id == self._focused_folder_id
            ),
            None,
        )
        now = datetime.now(timezone.utc)

        def within_date(source) -> bool:
            if not date_filter:
                return True
            try:
                updated = datetime.fromisoformat(
                    source.updated_at.replace("Z", "+00:00")
                )
            except (AttributeError, ValueError):
                return False
            window = {
                "today": timedelta(days=1),
                "week": timedelta(days=7),
                "month": timedelta(days=31),
            }[date_filter]
            return now - updated.astimezone(timezone.utc) <= window

        def is_ready(source) -> bool:
            readiness = readiness_by_id.get(source.source_id)
            return bool(readiness and (readiness.fts_ready or readiness.vector_ready))

        rows = [
            source
            for source in page.items
            if (not query or query in source.title.casefold())
            and (
                focused_folder is None or source.source_id in focused_folder.source_ids
            )
            and (not source_type or source.source_type.casefold() == source_type)
            and (
                not selection
                or (selection == "direct")
                == (
                    (
                        source.catalog_item_id
                        if source.ref.data_source.value == "local"
                        else source.source_id
                    )
                    in desired
                )
            )
            and (not status or (status == "ready") == is_ready(source))
            and within_date(source)
        ]
        sort_value = str(
            self.query_one("#research-source-sort", Select).value or "manual"
        )
        if sort_value.startswith("title_"):
            rows.sort(
                key=lambda item: item.title.casefold(),
                reverse=sort_value.endswith("desc"),
            )
        elif sort_value.startswith("updated_"):
            rows.sort(
                key=lambda item: item.updated_at,
                reverse=sort_value.endswith("desc"),
            )
        filtered = ResearchSourcePage(
            items=tuple(rows),
            limit=page.limit,
            offset=page.offset,
            total=page.total,
            has_more=page.has_more,
            desired_source_ids=page.desired_source_ids,
        )
        folder_source_ids = frozenset(
            source_id for folder in self._folders for source_id in folder.source_ids
        )
        self.query_one("#research-source-list", ResearchSourceList).sync_page(
            filtered,
            readiness=self._readiness,
            folder_source_ids=folder_source_ids,
            capabilities=self._capabilities,
            temporary_sort=sort_value != "manual",
        )
        self.query_one("#research-source-filter-summary", Static).update(
            f"Showing {len(rows)} of {len(page.items)} on page · "
            f"Temporary sort {'on; reorder disabled' if sort_value != 'manual' else 'off'}"
            + (
                f" · Focused folder: {focused_folder.name} (retrieval unchanged)"
                if focused_folder is not None
                else ""
            )
        )
