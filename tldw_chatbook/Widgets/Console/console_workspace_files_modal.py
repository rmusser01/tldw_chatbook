"""Read-only, non-activating Workspace Files inspector modal.

The modal receives already-captured ``BindingScope`` addresses and a narrow
filesystem service.  It deliberately has no workspace-registry, persistence,
Git, or logging dependency: rendering a label never creates filesystem
authority, and all blocking service calls run off the Textual event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.geometry import Size
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, Input, Static

from tldw_chatbook.Workspaces.file_inspector import (
    FILTER_DEBOUNCE_MS,
    BindingScope,
    DirectoryContinuation,
    DirectoryPage,
    DirectoryStatus,
    FileReadKind,
    FileReadResult,
    FileRef,
    FilterResult,
)
from tldw_chatbook.Widgets.modal_dismissal import SafeModalDismissMixin


class WorkspaceFilesService(Protocol):
    """The only filesystem boundary available to the modal."""

    def list_directory(
        self,
        scope: BindingScope,
        directory_parts: tuple[str, ...] = (),
        *,
        continuation: DirectoryContinuation | None = None,
    ) -> DirectoryPage: ...

    def filter_paths(
        self,
        scope: BindingScope,
        query: str,
        *,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> FilterResult: ...

    def read_file(
        self,
        scope: BindingScope,
        raw_parts: tuple[str, ...],
        *,
        page_offset: int | None = None,
        expected_revision: object | None = None,
    ) -> FileReadResult: ...


@dataclass(frozen=True)
class WorkspaceFilesBinding:
    """One presentation-safe binding captured by the Console controller."""

    binding_id: str
    label: str
    scope: BindingScope | None
    access_label: str = "Read-only"
    available: bool = True
    availability_copy: str = "Available"


@dataclass(frozen=True)
class WorkspaceFilesAttention:
    """Privacy-minimized attention state injected by the Console owner."""

    status_copy: str = "No pending Console attention."
    pending_approval_count: int = 0
    has_blocked_activity: bool = False
    has_failed_activity: bool = False
    has_new_activity: bool = False


@dataclass(frozen=True)
class _DirectoryTreePage:
    """One bounded page addressed by its raw root-relative directory parts."""

    directory_parts: tuple[str, ...]
    page: DirectoryPage | None


@dataclass(frozen=True)
class WorkspaceFilesViewState:
    """Immutable presentation state; raw identities stay separate from labels."""

    selected_binding_id: str | None = None
    directory_parts: tuple[str, ...] = ()
    directory_page: DirectoryPage | None = None
    directory_pages: tuple[_DirectoryTreePage, ...] = ()
    expanded_directory_parts: tuple[tuple[str, ...], ...] = ()
    selected_tree_parts: tuple[str, ...] | None = None
    filter_query: str = ""
    filter_result: FilterResult | None = None
    selected_file: FileRef | None = None
    file_result: FileReadResult | None = None
    status_copy: str = "Choose a folder binding."
    compact: bool = False
    short: bool = False
    compact_stage: str = "tree"


@dataclass(frozen=True)
class _LaneRequest:
    generation: int
    operation: Callable[[], Any]
    publish: Callable[[Any], Awaitable[None]]
    can_publish: Callable[[], bool] | None = None


class _OperationLane:
    """One active request plus one latest coalesced request for a modal lane."""

    def __init__(self, owner: "ConsoleWorkspaceFilesModal", name: str) -> None:
        self._owner = owner
        self.name = name
        self._active: asyncio.Task[None] | None = None
        self._latest: _LaneRequest | None = None
        self._closed = False

    @property
    def active(self) -> bool:
        return self._active is not None and not self._active.done()

    @property
    def has_latest(self) -> bool:
        return self._latest is not None

    def submit(self, request: _LaneRequest) -> None:
        if self._closed:
            return
        if self.active:
            self._latest = request
            return
        self._start(request)

    def _start(self, request: _LaneRequest) -> None:
        self._active = asyncio.create_task(self._run(request))

    async def _run(self, request: _LaneRequest) -> None:
        try:
            try:
                outcome = await asyncio.to_thread(request.operation)
            except asyncio.CancelledError:
                raise
            except Exception:
                outcome = None
            can_publish = request.can_publish or (
                lambda: self._owner._can_publish(request.generation)
            )
            if can_publish() and outcome is not None:
                await request.publish(outcome)
        finally:
            next_request = self._latest
            self._latest = None
            self._active = None
            if not self._closed and next_request is not None:
                self._start(next_request)

    async def close(self) -> None:
        """Discard queued work and join the one already-started operation.

        ``asyncio.to_thread`` cancellation only cancels its awaiting wrapper,
        not the filesystem call in the worker thread.  Joining the active lane
        keeps that call modal-owned until its bounded service contract returns;
        the filter service also sees the owner's cooperative cancellation flag.
        """
        self._closed = True
        self._latest = None
        if self._active is not None and not self._active.done():
            try:
                await self._active
            except asyncio.CancelledError:
                pass
        self._active = None


class ConsoleWorkspaceFilesModal(SafeModalDismissMixin, ModalScreen[None]):
    """Safely inspect one workspace's already-captured local-folder bindings."""

    SAFE_MODAL_CONTENT = "#console-workspace-files-modal"
    BINDINGS = [
        Binding("escape", "request_safe_cancel", "Back"),
        Binding("backspace", "back_to_console", "Back"),
        Binding("f", "focus_filter", "Filter"),
        Binding("left", "collapse_or_parent", "Collapse"),
        Binding("right", "expand_selected", "Expand"),
    ]
    AUTO_FOCUS = None

    def __init__(
        self,
        *,
        inspector: WorkspaceFilesService,
        inspected_workspace_id: str,
        inspected_workspace_name: str,
        active_workspace_id: str | None,
        active_workspace_name: str,
        bindings: Sequence[WorkspaceFilesBinding],
        attention: WorkspaceFilesAttention | None = None,
        on_back_to_console: Callable[[], None] | None = None,
        on_visit_closed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._inspector = inspector
        self.inspected_workspace_id = inspected_workspace_id
        self.inspected_workspace_name = inspected_workspace_name
        self.active_workspace_id = active_workspace_id
        self.active_workspace_name = active_workspace_name
        self._workspace_bindings = tuple(bindings)
        self._attention = attention or WorkspaceFilesAttention()
        self._on_back_to_console = on_back_to_console
        self._on_visit_closed = on_visit_closed
        self._visit_closed_notified = False
        self._attention_generation = 0
        first_available = next((item for item in self._workspace_bindings if item.available), None)
        self._state = WorkspaceFilesViewState(
            selected_binding_id=first_available.binding_id if first_available else None,
            status_copy="Loading folder…" if first_available else "No available local folder binding.",
        )
        self._generation = 0
        self._workspace_files_closing = False
        self._filter_timer: Timer | None = None
        self._pre_filter_tree_state: WorkspaceFilesViewState | None = None
        self._tree_entries: dict[str, Any] = {}
        self._tree_more: dict[str, tuple[tuple[str, ...], DirectoryContinuation]] = {}
        self._directory_request_tokens: dict[tuple[str, ...], int] = {}
        self._list_lane = _OperationLane(self, "list")
        self._read_lane = _OperationLane(self, "read")
        self._filter_lane = _OperationLane(self, "filter")

    @property
    def state(self) -> WorkspaceFilesViewState:
        """Expose immutable presentation state for focused UI tests."""
        return self._state

    @property
    def owned_lane_count(self) -> int:
        """Return the number of active or queued modal operation lanes."""
        return sum(
            lane.active or lane.has_latest
            for lane in (self._list_lane, self._read_lane, self._filter_lane)
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="console-workspace-files-modal"):
            yield Static("Workspace Files", classes="console-modal-header", markup=False)
            yield Static(
                f"Inspector only · Console remains {self.active_workspace_name}",
                id="console-workspace-files-pinned",
                markup=False,
            )
            yield Static(
                f"Viewing {self.inspected_workspace_name} · Read-only access",
                id="console-workspace-files-contract",
                markup=False,
            )
            yield Static(self._attention.status_copy, id="console-workspace-files-attention", markup=False)
            with Horizontal(id="console-workspace-files-binding-row"):
                for index, binding in enumerate(self._workspace_bindings):
                    yield Button(
                        Text(f"{binding.label} · {binding.access_label} · {binding.availability_copy}"),
                        id=f"console-workspace-files-binding-{index}",
                        classes="console-workspace-files-binding",
                        compact=True,
                    )
            with Horizontal(id="console-workspace-files-filter-row"):
                yield Input(placeholder="Filter paths…", id="console-workspace-files-filter")
                yield Button("Clear", id="console-workspace-files-filter-clear", compact=True)
                yield Button("Cancel", id="console-workspace-files-filter-cancel", compact=True)
            yield Static("", id="console-workspace-files-status", markup=False)
            with Horizontal(id="console-workspace-files-body"):
                with VerticalScroll(id="console-workspace-files-tree"):
                    yield Static("Loading folder…", markup=False)
                with VerticalScroll(id="console-workspace-files-viewer"):
                    yield Static("Select a file to view its safe preview.", markup=False)
            with Horizontal(id="console-workspace-files-actions"):
                yield Button("Back to Console", id="console-workspace-files-back", compact=True)
                yield Button(
                    "Details",
                    id="console-workspace-files-details",
                    compact=True,
                    tooltip=(
                        f"Inspector only. Active Console workspace: {self.active_workspace_name}. "
                        f"Viewing: {self.inspected_workspace_name}."
                    ),
                )
                yield Button("Previous", id="console-workspace-files-previous", compact=True, disabled=True)
                yield Button("Next", id="console-workspace-files-next", compact=True, disabled=True)
                yield Button("Refresh", id="console-workspace-files-refresh", compact=True)

    def update_attention(
        self, attention: WorkspaceFilesAttention, generation: int
    ) -> bool:
        """Publish a newer generic attention snapshot while this visit lives."""
        if generation <= self._attention_generation or self._workspace_files_closing:
            return False
        self._attention_generation = generation
        self._attention = attention
        if self.is_mounted:
            self.query_one("#console-workspace-files-attention", Static).update(
                attention.status_copy
            )
        return True

    async def on_mount(self) -> None:  # type: ignore[override]
        super().on_mount()
        self._sync_layout()
        self._sync_binding_buttons()
        self._sync_status()
        if self._state.selected_binding_id is not None:
            if len([binding for binding in self._workspace_bindings if binding.available]) > 1:
                self.query_one("#console-workspace-files-binding-0", Button).focus()
            else:
                self.query_one("#console-workspace-files-tree", VerticalScroll).focus()
            self._request_directory()
        else:
            self.query_one("#console-workspace-files-back", Button).focus()

    def on_resize(self, event: Resize) -> None:
        self._sync_layout(event.size)

    def _sync_layout(self, size: Size | None = None) -> None:
        size = size or self.size
        compact = size.width <= 100
        short = size.height < 30
        self._state = replace(self._state, compact=compact, short=short)
        self.set_class(compact, "-compact")
        self.set_class(short, "-short")
        self.set_class(compact and self._state.compact_stage == "viewer", "-viewer-stage")

    def _binding_for_id(self, binding_id: str | None) -> WorkspaceFilesBinding | None:
        return next((item for item in self._workspace_bindings if item.binding_id == binding_id), None)

    def _selected_binding(self) -> WorkspaceFilesBinding | None:
        return self._binding_for_id(self._state.selected_binding_id)

    def _can_publish(self, generation: int) -> bool:
        return (
            not self._workspace_files_closing
            and self.is_mounted
            and generation == self._generation
        )

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def _page_for(self, directory_parts: tuple[str, ...]) -> DirectoryPage | None:
        if directory_parts == ():
            return self._state.directory_page
        return next(
            (
                item.page
                for item in self._state.directory_pages
                if item.directory_parts == directory_parts
            ),
            None,
        )

    def _replace_directory_page(
        self, directory_parts: tuple[str, ...], page: DirectoryPage | None
    ) -> tuple[_DirectoryTreePage, ...]:
        pages = [
            item
            for item in self._state.directory_pages
            if item.directory_parts != directory_parts
        ]
        if directory_parts:
            pages.append(_DirectoryTreePage(directory_parts, page))
        return tuple(pages)

    def _directory_request_token(self, directory_parts: tuple[str, ...]) -> int:
        token = self._directory_request_tokens.get(directory_parts, 0) + 1
        self._directory_request_tokens[directory_parts] = token
        return token

    def _invalidate_directory_subtree(self, directory_parts: tuple[str, ...]) -> None:
        for parts in tuple(self._directory_request_tokens):
            if parts[: len(directory_parts)] == directory_parts:
                del self._directory_request_tokens[parts]

    def _can_publish_directory(
        self,
        binding_id: str,
        directory_parts: tuple[str, ...],
        token: int,
        continuation: DirectoryContinuation | None,
    ) -> bool:
        if self._workspace_files_closing or not self.is_mounted:
            return False
        if self._state.selected_binding_id != binding_id:
            return False
        if self._directory_request_tokens.get(directory_parts) != token:
            return False
        if continuation is not None:
            page = self._page_for(directory_parts)
            if page is None or page.continuation != continuation:
                return False
        return directory_parts == () or directory_parts in self._state.expanded_directory_parts

    @staticmethod
    def _merge_directory_page(
        previous: DirectoryPage, incoming: DirectoryPage
    ) -> DirectoryPage:
        """Append a continuation page by raw identity under the service bound."""
        seen: set[tuple[str, ...]] = set()
        entries = []
        for entry in (*previous.entries, *incoming.entries):
            if entry.raw_parts not in seen:
                seen.add(entry.raw_parts)
                entries.append(entry)
        if len(entries) > 10_000:
            return replace(
                incoming,
                status=DirectoryStatus.TRUNCATED,
                entries=tuple(entries[:10_000]),
                continuation=None,
            )
        return replace(incoming, entries=tuple(entries))

    def _request_directory(
        self,
        continuation: DirectoryContinuation | None = None,
        *,
        directory_parts: tuple[str, ...] | None = None,
    ) -> None:
        binding = self._selected_binding()
        if binding is None or binding.scope is None or not binding.available:
            self._state = replace(self._state, status_copy="Selected binding is unavailable.")
            self._sync_status()
            return
        generation = self._next_generation()
        directory_parts = (
            self._state.directory_parts
            if directory_parts is None
            else directory_parts
        )
        self._state = replace(
            self._state,
            directory_parts=directory_parts,
            status_copy="Loading folder…",
            directory_page=(
                self._state.directory_page
                if continuation is not None
                else None
                if directory_parts == ()
                else self._state.directory_page
            ),
            directory_pages=(
                self._state.directory_pages
                if continuation is not None
                else self._replace_directory_page(directory_parts, None)
            ),
            filter_result=None,
        )
        self._sync_status()
        token = self._directory_request_token(directory_parts)
        self._list_lane.submit(
            _LaneRequest(
                generation,
                lambda: self._inspector.list_directory(
                    binding.scope, directory_parts, continuation=continuation
                ),
                lambda page: self._publish_directory(
                    page,
                    directory_parts=directory_parts,
                    continuation=continuation,
                ),
                lambda: self._can_publish_directory(
                    binding.binding_id, directory_parts, token, continuation
                ),
            )
        )

    async def _publish_directory(
        self,
        page: DirectoryPage,
        *,
        directory_parts: tuple[str, ...] | None = None,
        continuation: DirectoryContinuation | None = None,
    ) -> None:
        directory_parts = (
            self._state.directory_parts
            if directory_parts is None
            else directory_parts
        )
        if continuation is not None:
            previous = self._page_for(directory_parts)
            if previous is None or previous.continuation != continuation:
                return
            page = self._merge_directory_page(previous, page)
        copy = {
            "empty": "Folder is empty.", "partial": "More folder entries available.",
            "truncated": "Folder listing reached its safety limit.", "failed": "Folder is unavailable.",
        }.get(page.status.value, "Folder loaded.")
        self._state = replace(
            self._state,
            directory_page=page if directory_parts == () else self._state.directory_page,
            directory_pages=self._replace_directory_page(directory_parts, page),
            filter_result=None,
            status_copy=copy,
        )
        await self._render_tree()
        self._sync_status()

    def _request_filter(self, query: str) -> None:
        binding = self._selected_binding()
        if binding is None or binding.scope is None or not binding.available:
            return
        if not self._state.filter_query:
            self._pre_filter_tree_state = replace(
                self._state, filter_query="", filter_result=None
            )
        generation = self._next_generation()
        self._state = replace(self._state, filter_query=query, filter_result=None, status_copy="Searching paths…")
        self._sync_status()
        self._filter_lane.submit(
            _LaneRequest(
                generation,
                lambda: self._inspector.filter_paths(
                    binding.scope,
                    query,
                    is_cancelled=lambda: self._workspace_files_closing
                    or generation != self._generation,
                ),
                self._publish_filter,
            )
        )

    async def _publish_filter(self, result: FilterResult) -> None:
        self._state = replace(self._state, filter_result=result, status_copy=result.status_copy or f"Filter {result.status.value}.")
        await self._render_tree()
        self._sync_status()

    def _request_file(self, file_ref: FileRef, *, offset: int | None = None) -> None:
        binding = self._selected_binding()
        if binding is None or binding.scope is None or not binding.available:
            return
        expected = self._state.file_result.revision if offset is not None and self._state.file_result else None
        generation = self._next_generation()
        self._state = replace(self._state, selected_file=file_ref, status_copy="Loading safe preview…")
        self._sync_status()
        self._read_lane.submit(
            _LaneRequest(
                generation,
                lambda: self._inspector.read_file(binding.scope, file_ref.raw_parts, page_offset=offset, expected_revision=expected),
                self._publish_file,
            )
        )

    async def _publish_file(self, result: FileReadResult) -> None:
        copy = "Preview loaded."
        if result.kind is FileReadKind.METADATA_ONLY:
            copy = "File is metadata-only above the safe preview limit."
        elif result.kind is FileReadKind.REVISION_CHANGED:
            copy = "File changed. Refresh to view the new revision."
        elif result.kind is FileReadKind.FAILED:
            copy = "File is unavailable."
        self._state = replace(self._state, file_result=result, status_copy=copy, compact_stage="viewer")
        await self._render_viewer()
        self._sync_status()

    async def _render_tree(self) -> None:
        tree = self.query_one("#console-workspace-files-tree", VerticalScroll)
        await tree.remove_children()
        result = self._state.filter_result
        if result is not None:
            if not result.matches:
                await tree.mount(Static(result.status_copy or "No matching paths.", markup=False))
                return
            await tree.mount_all(
                Button(Text(item.display_path), id=f"console-workspace-files-filter-result-{index}", classes="console-workspace-files-file", compact=True)
                for index, item in enumerate(result.matches)
            )
            return
        page = self._page_for(())
        if page is None:
            await tree.mount(Static("Loading folder…", markup=False))
            return
        if not page.entries:
            await tree.mount(Static(self._state.status_copy, markup=False))
            return
        rows: list[Button] = []
        self._tree_entries = {}
        self._tree_more = {}

        def append_page(
            directory_parts: tuple[str, ...], current: DirectoryPage, depth: int
        ) -> None:
            for entry in current.entries:
                row_id = f"console-workspace-files-entry-{len(self._tree_entries)}"
                self._tree_entries[row_id] = entry
                expanded = entry.raw_parts in self._state.expanded_directory_parts
                prefix = "▾ " if entry.is_directory and expanded else "▸ " if entry.is_directory else "  "
                row = Button(
                    Text("  " * depth + prefix + entry.display_name),
                    id=row_id,
                    classes="console-workspace-files-entry",
                    compact=True,
                )
                row.set_class(entry.raw_parts == self._state.selected_tree_parts, "-selected")
                rows.append(row)
                child_page = self._page_for(entry.raw_parts)
                if entry.is_directory and expanded and child_page is not None:
                    append_page(entry.raw_parts, child_page, depth + 1)
            if current.continuation is not None:
                row_id = (
                    "console-workspace-files-more"
                    if directory_parts == ()
                    else f"console-workspace-files-more-{len(self._tree_more)}"
                )
                self._tree_more[row_id] = (directory_parts, current.continuation)
                rows.append(
                    Button(
                        "Load more",
                        id=row_id,
                        classes="console-workspace-files-more",
                        compact=True,
                    )
                )

        append_page((), page, 0)
        await tree.mount_all(rows)

    async def _render_viewer(self) -> None:
        viewer = self.query_one("#console-workspace-files-viewer", VerticalScroll)
        await viewer.remove_children()
        result = self._state.file_result
        selected = self._state.selected_file
        if result is None or selected is None:
            await viewer.mount(Static("Select a file to view its safe preview.", markup=False))
            return
        await viewer.mount(Static(selected.display_path, classes="console-workspace-files-path", markup=False))
        if result.character_range is not None:
            start, end = result.character_range
            await viewer.mount(Static(f"Characters {start + 1}–{end} of {result.total_characters}", markup=False))
        await viewer.mount(Static(result.text or self._state.status_copy, id="console-workspace-files-preview", markup=False))
        self.query_one("#console-workspace-files-previous", Button).disabled = result.previous_page_offset is None
        self.query_one("#console-workspace-files-next", Button).disabled = result.next_page_offset is None

    def _sync_status(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#console-workspace-files-status", Static).update(self._state.status_copy)

    def _sync_binding_buttons(self) -> None:
        if not self.is_mounted:
            return
        for index, binding in enumerate(self._workspace_bindings):
            button = self.query_one(f"#console-workspace-files-binding-{index}", Button)
            button.set_class(binding.binding_id == self._state.selected_binding_id, "-selected")

    def _expand_directory(self, entry: Any) -> None:
        """Expand one displayed directory using its retained raw identity."""
        expanded = (*self._state.expanded_directory_parts, entry.raw_parts)
        self._state = replace(
            self._state,
            directory_parts=entry.raw_parts,
            expanded_directory_parts=expanded,
            selected_tree_parts=entry.raw_parts,
            status_copy="Expanding folder…",
        )
        self._sync_status()
        self._request_directory(directory_parts=entry.raw_parts)

    def _collapse_directory(self, directory_parts: tuple[str, ...]) -> None:
        """Remove one expanded directory and all raw-identity descendants."""
        self._invalidate_directory_subtree(directory_parts)
        expanded = tuple(
            item
            for item in self._state.expanded_directory_parts
            if item[: len(directory_parts)] != directory_parts
        )
        pages = tuple(
            item
            for item in self._state.directory_pages
            if item.directory_parts[: len(directory_parts)] != directory_parts
        )
        parent = directory_parts[:-1] or None
        self._state = replace(
            self._state,
            directory_parts=parent or (),
            directory_pages=pages,
            expanded_directory_parts=expanded,
            selected_tree_parts=parent,
            status_copy="Folder collapsed.",
        )
        self._sync_status()

    async def action_collapse_or_parent(self) -> None:
        """Collapse the selected directory, or move to its raw parent."""
        selected = self._state.selected_tree_parts
        if selected is None:
            return
        if selected in self._state.expanded_directory_parts:
            self._collapse_directory(selected)
        else:
            parent = selected[:-1] or None
            self._state = replace(
                self._state,
                directory_parts=parent or (),
                selected_tree_parts=parent,
                status_copy="Moved to parent folder.",
            )
            self._sync_status()
        await self._render_tree()

    def action_expand_selected(self) -> None:
        """Expand the selected directory; files retain their current preview."""
        selected = self._state.selected_tree_parts
        if selected is None:
            return
        entry = next(
            (
                item
                for item in self._tree_entries.values()
                if item.raw_parts == selected and item.is_directory
            ),
            None,
        )
        if entry is not None and selected not in self._state.expanded_directory_parts:
            self._expand_directory(entry)

    def _cancel_filter_timer(self) -> None:
        if self._filter_timer is not None:
            self._filter_timer.stop()
            self._filter_timer = None

    @on(Button.Pressed, ".console-workspace-files-binding")
    async def _select_binding(self, event: Button.Pressed) -> None:
        event.stop()
        try:
            index = int((event.button.id or "").rsplit("-", 1)[-1])
            binding = self._workspace_bindings[index]
        except (IndexError, ValueError):
            return
        self._cancel_filter_timer()
        self._next_generation()
        self._directory_request_tokens.clear()
        self._pre_filter_tree_state = None
        self.query_one("#console-workspace-files-filter", Input).value = ""
        self._state = WorkspaceFilesViewState(
            selected_binding_id=binding.binding_id,
            status_copy=(
                "Loading folder…" if binding.available else "Selected binding is unavailable."
            ),
            compact=self._state.compact,
            short=self._state.short,
        )
        self._sync_binding_buttons()
        if binding.available:
            self._request_directory()
        else:
            self._sync_status()
            await self._render_tree()

    @on(Input.Changed, "#console-workspace-files-filter")
    async def _filter_changed(self, event: Input.Changed) -> None:
        self._cancel_filter_timer()
        query = event.value
        if not query:
            if self._state.filter_query or self._state.filter_result is not None:
                await self._clear_filter()
            return
        self._filter_timer = self.set_timer(FILTER_DEBOUNCE_MS / 1000, lambda: self._request_filter(query))

    @on(Input.Submitted, "#console-workspace-files-filter")
    def _filter_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._cancel_filter_timer()
        self._request_filter(event.value)

    async def _clear_filter(self) -> None:
        self._cancel_filter_timer()
        self._next_generation()
        previous = self._pre_filter_tree_state
        self._pre_filter_tree_state = None
        if previous is None:
            self._state = replace(
                self._state,
                filter_query="",
                filter_result=None,
                status_copy="Folder restored.",
            )
            self._request_directory()
            return
        self._state = replace(
            previous,
            filter_query="",
            filter_result=None,
            status_copy="Folder restored.",
        )
        await self._render_tree()
        self._sync_status()

    @on(Button.Pressed, "#console-workspace-files-filter-clear")
    async def _clear_filter_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.query_one("#console-workspace-files-filter", Input).value = ""
        await self._clear_filter()

    @on(Button.Pressed, "#console-workspace-files-filter-cancel")
    def _cancel_filter(self, event: Button.Pressed) -> None:
        event.stop()
        self._cancel_filter_timer()
        self._next_generation()
        self._state = replace(self._state, status_copy="Filter cancelled.")
        self._sync_status()

    @on(Button.Pressed, ".console-workspace-files-entry")
    async def _entry_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        entry = self._tree_entries.get(event.button.id or "")
        if entry is None:
            return
        if entry.is_directory:
            if entry.raw_parts in self._state.expanded_directory_parts:
                self._collapse_directory(entry.raw_parts)
                await self._render_tree()
            else:
                self._expand_directory(entry)
                await self._render_tree()
            return
        self._state = replace(self._state, selected_tree_parts=entry.raw_parts)
        self._request_file(FileRef(entry.raw_parts, entry.display_name))

    @on(Button.Pressed, ".console-workspace-files-file")
    def _filtered_file_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        result = self._state.filter_result
        if result is None:
            return
        try:
            index = int((event.button.id or "").rsplit("-", 1)[-1])
            self._request_file(result.matches[index])
        except (IndexError, ValueError):
            return

    @on(Button.Pressed, ".console-workspace-files-more")
    def _more_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        request = self._tree_more.get(event.button.id or "")
        if request is not None:
            directory_parts, continuation = request
            self._request_directory(continuation, directory_parts=directory_parts)

    @on(Button.Pressed, "#console-workspace-files-previous")
    def _previous_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._state.selected_file is not None and self._state.file_result is not None:
            offset = self._state.file_result.previous_page_offset
            if offset is not None:
                self._request_file(self._state.selected_file, offset=offset)

    @on(Button.Pressed, "#console-workspace-files-next")
    def _next_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._state.selected_file is not None and self._state.file_result is not None:
            offset = self._state.file_result.next_page_offset
            if offset is not None:
                self._request_file(self._state.selected_file, offset=offset)

    @on(Button.Pressed, "#console-workspace-files-refresh")
    def _refresh_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._state.selected_file is not None:
            self._request_file(self._state.selected_file)
        else:
            self._request_directory()

    @on(Button.Pressed, "#console-workspace-files-back")
    async def _back_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        await self.action_back_to_console()

    @on(Button.Pressed, "#console-workspace-files-details")
    def _details_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._state = replace(
            self._state,
            status_copy=(
                f"Inspector only. Console remains {self.active_workspace_name}. "
                f"Viewing {self.inspected_workspace_name} read-only."
            ),
        )
        self._sync_status()

    def action_focus_filter(self) -> None:
        self.query_one("#console-workspace-files-filter", Input).focus()

    def action_show_tree(self) -> None:
        self._state = replace(self._state, compact_stage="tree")
        self._sync_layout()
        self.query_one("#console-workspace-files-tree", VerticalScroll).focus()

    def action_show_viewer(self) -> None:
        self._state = replace(self._state, compact_stage="viewer")
        self._sync_layout()
        self.query_one("#console-workspace-files-viewer", VerticalScroll).focus()

    async def action_back_to_console(self) -> None:
        await self.request_safe_cancel(source="back")

    async def _teardown(self) -> None:
        if self._workspace_files_closing:
            return
        self._workspace_files_closing = True
        self._next_generation()
        self._cancel_filter_timer()
        await self._list_lane.close()
        await self._read_lane.close()
        await self._filter_lane.close()

    async def _perform_safe_cancel(self, *, source: str) -> None:
        del source
        await self.run_cancel_effect_once(self._teardown)
        if self.dismiss_safe_once(None) and self._on_back_to_console is not None:
            self._on_back_to_console()

    async def on_unmount(self) -> None:
        """Join owned work and close the controller visit on every pop path."""
        await self.run_cancel_effect_once(self._teardown)
        if not self._visit_closed_notified:
            self._visit_closed_notified = True
            if self._on_visit_closed is not None:
                self._on_visit_closed()
        super().on_unmount()


__all__ = [
    "ConsoleWorkspaceFilesModal",
    "WorkspaceFilesAttention",
    "WorkspaceFilesBinding",
    "WorkspaceFilesService",
    "WorkspaceFilesViewState",
]
