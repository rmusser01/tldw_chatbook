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


@dataclass(frozen=True)
class WorkspaceFilesViewState:
    """Immutable presentation state; raw identities stay separate from labels."""

    selected_binding_id: str | None = None
    directory_parts: tuple[str, ...] = ()
    directory_page: DirectoryPage | None = None
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
            if self._owner._can_publish(request.generation) and outcome is not None:
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
        Binding("left", "show_tree", "Tree"),
        Binding("right", "show_viewer", "Viewer"),
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
        first_available = next((item for item in self._workspace_bindings if item.available), None)
        self._state = WorkspaceFilesViewState(
            selected_binding_id=first_available.binding_id if first_available else None,
            status_copy="Loading folder…" if first_available else "No available local folder binding.",
        )
        self._generation = 0
        self._workspace_files_closing = False
        self._filter_timer: Timer | None = None
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

    def _request_directory(self, continuation: DirectoryContinuation | None = None) -> None:
        binding = self._selected_binding()
        if binding is None or binding.scope is None or not binding.available:
            self._state = replace(self._state, status_copy="Selected binding is unavailable.")
            self._sync_status()
            return
        generation = self._next_generation()
        directory_parts = self._state.directory_parts
        self._state = replace(self._state, status_copy="Loading folder…", directory_page=None, filter_result=None)
        self._sync_status()
        self._list_lane.submit(
            _LaneRequest(
                generation,
                lambda: self._inspector.list_directory(binding.scope, directory_parts, continuation=continuation),
                self._publish_directory,
            )
        )

    async def _publish_directory(self, page: DirectoryPage) -> None:
        copy = {
            "empty": "Folder is empty.", "partial": "More folder entries available.",
            "truncated": "Folder listing reached its safety limit.", "failed": "Folder is unavailable.",
        }.get(page.status.value, "Folder loaded.")
        self._state = replace(self._state, directory_page=page, filter_result=None, status_copy=copy)
        await self._render_tree()
        self._sync_status()

    def _request_filter(self, query: str) -> None:
        binding = self._selected_binding()
        if binding is None or binding.scope is None or not binding.available:
            return
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
        page = self._state.directory_page
        if page is None:
            await tree.mount(Static("Loading folder…", markup=False))
            return
        if not page.entries:
            await tree.mount(Static(self._state.status_copy, markup=False))
            return
        rows: list[Button] = []
        for index, entry in enumerate(page.entries):
            prefix = "▸ " if entry.is_directory else "  "
            rows.append(Button(Text(prefix + entry.display_name), id=f"console-workspace-files-entry-{index}", classes="console-workspace-files-entry", compact=True))
        if page.continuation is not None:
            rows.append(Button("Load more", id="console-workspace-files-more", compact=True))
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
        self._state = WorkspaceFilesViewState(selected_binding_id=binding.binding_id, status_copy=("Loading folder…" if binding.available else "Selected binding is unavailable."), compact=self._state.compact, short=self._state.short)
        self._sync_binding_buttons()
        if binding.available:
            self._request_directory()
        else:
            self._sync_status()
            await self._render_tree()

    @on(Input.Changed, "#console-workspace-files-filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self._cancel_filter_timer()
        query = event.value
        if not query:
            self._clear_filter()
            return
        self._filter_timer = self.set_timer(FILTER_DEBOUNCE_MS / 1000, lambda: self._request_filter(query))

    @on(Input.Submitted, "#console-workspace-files-filter")
    def _filter_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._cancel_filter_timer()
        self._request_filter(event.value)

    def _clear_filter(self) -> None:
        self._cancel_filter_timer()
        self._next_generation()
        self._state = replace(self._state, filter_query="", filter_result=None, status_copy="Folder restored.")
        self._request_directory()

    @on(Button.Pressed, "#console-workspace-files-filter-clear")
    def _clear_filter_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.query_one("#console-workspace-files-filter", Input).value = ""
        self._clear_filter()

    @on(Button.Pressed, "#console-workspace-files-filter-cancel")
    def _cancel_filter(self, event: Button.Pressed) -> None:
        event.stop()
        self._cancel_filter_timer()
        self._next_generation()
        self._state = replace(self._state, status_copy="Filter cancelled.")
        self._sync_status()

    @on(Button.Pressed, ".console-workspace-files-entry")
    def _entry_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        page = self._state.directory_page
        if page is None:
            return
        try:
            index = int((event.button.id or "").rsplit("-", 1)[-1])
            entry = page.entries[index]
        except (IndexError, ValueError):
            return
        if entry.is_directory:
            self._state = replace(self._state, directory_parts=entry.raw_parts)
            self._request_directory()
            return
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

    @on(Button.Pressed, "#console-workspace-files-more")
    def _more_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._state.directory_page is not None and self._state.directory_page.continuation is not None:
            self._request_directory(self._state.directory_page.continuation)

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

    def on_unmount(self) -> None:
        super().on_unmount()
        self._workspace_files_closing = True
        self._cancel_filter_timer()


__all__ = [
    "ConsoleWorkspaceFilesModal",
    "WorkspaceFilesAttention",
    "WorkspaceFilesBinding",
    "WorkspaceFilesService",
    "WorkspaceFilesViewState",
]
