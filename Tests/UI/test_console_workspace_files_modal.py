"""Behavior contracts for the read-only Console Workspace Files modal."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.events import Resize
from textual.geometry import Size
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
    ConsoleWorkspaceFilesModal,
    WorkspaceFilesBinding,
    _LaneRequest,
    _OperationLane,
)
from tldw_chatbook.Workspaces.file_inspector import (
    BindingScope,
    DirectoryEntry,
    DirectoryContinuation,
    DirectoryPage,
    DirectoryRevision,
    DirectoryStatus,
    FileRef,
    FileRevision,
    FileReadKind,
    FileReadResult,
    FilterResult,
    FilterStatus,
)


@dataclass
class _Inspector:
    """A real modal boundary fake; it records only public service operations."""

    calls: list[tuple[str, object]]

    def list_directory(self, scope: BindingScope, directory_parts=(), *, continuation=None):
        self.calls.append(("list", (scope, directory_parts, continuation)))
        return DirectoryPage(
            DirectoryStatus.COMPLETE,
            (DirectoryEntry(("unsafe[bold]\\n",), "unsafe[bold]\\n", False),),
        )

    def filter_paths(self, scope: BindingScope, query: str, *, is_cancelled=None):
        self.calls.append(("filter", (scope, query)))
        return FilterResult(FilterStatus.EMPTY, status_copy="No matching paths.")

    def read_file(self, scope: BindingScope, raw_parts, *, page_offset=None, expected_revision=None):
        self.calls.append(("read", (scope, raw_parts, page_offset, expected_revision)))
        return FileReadResult(FileReadKind.TEXT, text="safe\\npreview")


class _Host(App[None]):
    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Button("Open files", id="files-opener")


def _scope() -> BindingScope:
    return BindingScope("ws-a", "binding-a", "fingerprint", "/not-read", 1, 1)


@pytest.mark.asyncio
async def test_modal_shows_pinned_read_only_identity_and_loads_its_only_binding() -> None:
    """Catch a modal that hides the inspected scope or does not begin a safe root listing."""
    inspector = _Inspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector,
        inspected_workspace_id="ws-a",
        inspected_workspace_name="Workspace A",
        active_workspace_id="ws-b",
        active_workspace_name="Workspace B",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()

    async with app.run_test(size=(120, 40)) as pilot:
        opener = app.query_one("#files-opener", Button)
        opener.focus()
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        assert str(modal.query(".console-modal-header").first(Static).renderable) == "Workspace Files"
        assert str(modal.query_one("#console-workspace-files-pinned", Static).renderable) == "Inspector only · Console remains Workspace B"
        assert str(modal.query_one("#console-workspace-files-contract", Static).renderable) == "Viewing Workspace A · Read-only access"
        assert inspector.calls and inspector.calls[0][0] == "list"


@pytest.mark.asyncio
async def test_unavailable_binding_is_selected_without_falling_back_to_another_scope() -> None:
    """Catch a stale binding click that silently keeps a previously valid folder."""
    inspector = _Inspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector,
        inspected_workspace_id="ws-a",
        inspected_workspace_name="Workspace A",
        active_workspace_id="ws-a",
        active_workspace_name="Workspace A",
        bindings=(
            WorkspaceFilesBinding("binding-a", "Available", _scope()),
            WorkspaceFilesBinding("binding-b", "Changed", None, available=False, availability_copy="Unavailable"),
        ),
    )
    app = _Host()

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-workspace-files-binding-1")
        await pilot.pause()

        assert modal.state.selected_binding_id == "binding-b"
        assert modal.state.status_copy == "Selected binding is unavailable."
        assert [name for name, _call in inspector.calls] == ["list"]


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["back", "escape", "backdrop"])
async def test_safe_dismissal_sources_are_one_shot_and_restore_the_opener(
    source: str,
) -> None:
    """All visible/keyboard/backdrop exits share the same terminal callback."""
    callbacks: list[str] = []
    results: list[None] = []
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Unavailable", None, available=False),),
        on_back_to_console=lambda: callbacks.append("back"),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        opener = app.query_one("#files-opener", Button)
        opener.focus()
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        if source == "back":
            await pilot.click("#console-workspace-files-back")
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()
        await pilot.pause()
        assert app.screen is not modal
        assert app.focused is opener

        await modal.action_request_safe_cancel()
        assert callbacks == ["back"]
        assert results == [None]


@pytest.mark.asyncio
async def test_inside_click_does_not_dismiss_and_resize_keeps_modal_state() -> None:
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Unavailable", None, available=False),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        original_state = modal.state
        await pilot.click("#console-workspace-files-pinned")
        await pilot.pause()
        assert app.screen is modal

        modal.on_resize(Resize(Size(80, 24), Size(120, 40)))
        assert app.screen is modal
        assert modal.state.selected_binding_id == original_state.selected_binding_id
        assert modal.has_class("-compact") and modal.has_class("-short")


def test_modal_declares_the_shared_safe_dismissal_boundary() -> None:
    """Catch a future modal change that bypasses the Console safe-modal contract."""
    assert ConsoleWorkspaceFilesModal.SAFE_MODAL_CONTENT == "#console-workspace-files-modal"
    assert ConsoleWorkspaceFilesModal.BINDINGS[0].action == "request_safe_cancel"


@pytest.mark.asyncio
async def test_teardown_preserves_textual_message_pump_and_binding_state() -> None:
    """The modal must not reuse Textual's private lifecycle or binding fields."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]),
        inspected_workspace_id="ws-a",
        inspected_workspace_name="A",
        active_workspace_id="ws-a",
        active_workspace_name="A",
        bindings=(
            WorkspaceFilesBinding(
                "binding-a", "Unavailable", None, available=False
            ),
        ),
    )
    app = _Host()

    async with app.run_test(size=(100, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert not modal._closed
        assert callable(modal._bindings.copy)
        await modal._teardown()
        assert not modal._closed
        assert callable(modal._bindings.copy)

        await modal.action_request_safe_cancel()
        await pilot.pause()
        assert app.screen is not modal


@pytest.mark.asyncio
async def test_lane_teardown_waits_for_the_bounded_active_operation() -> None:
    """Catch teardown that cancels only its wrapper and leaves a thread-owned read alive."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    entered = __import__("threading").Event()
    release = __import__("threading").Event()
    lane = _OperationLane(modal, "test")
    lane.submit(_LaneRequest(0, lambda: (entered.set(), release.wait(), "done")[2], lambda _value: None))
    while not entered.is_set():
        await __import__("asyncio").sleep(0)
    closing = __import__("asyncio").create_task(lane.close())
    await __import__("asyncio").sleep(0)
    assert not closing.done()
    release.set()
    await closing
    assert not lane.active and not lane.has_latest


@pytest.mark.asyncio
async def test_lane_coalesces_to_one_latest_request_and_discards_it_on_close() -> None:
    """A rapid burst must retain only the newest request behind the active one."""
    entered = Event()
    release = Event()
    published: list[str] = []

    class _Owner:
        @staticmethod
        def _can_publish(_generation: int) -> bool:
            return True

    async def publish(value: str) -> None:
        published.append(value)

    lane = _OperationLane(_Owner(), "test")  # type: ignore[arg-type]
    lane.submit(
        _LaneRequest(
            1,
            lambda: (entered.set(), release.wait(), "first")[2],
            publish,
        )
    )
    while not entered.is_set():
        await asyncio.sleep(0)
    lane.submit(_LaneRequest(2, lambda: "second", publish))
    lane.submit(_LaneRequest(3, lambda: "latest", publish))
    assert lane.active and lane.has_latest

    release.set()
    while lane.active:
        await asyncio.sleep(0)
    assert published == ["first", "latest"]

    entered.clear()
    release.clear()
    lane.submit(
        _LaneRequest(
            4,
            lambda: (entered.set(), release.wait(), "active")[2],
            publish,
        )
    )
    while not entered.is_set():
        await asyncio.sleep(0)
    lane.submit(_LaneRequest(5, lambda: "discarded", publish))
    closing = asyncio.create_task(lane.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    assert published == ["first", "latest", "active"]
    assert not lane.active and not lane.has_latest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "copy"),
    [
        (DirectoryStatus.EMPTY, "Folder is empty."),
        (DirectoryStatus.PARTIAL, "More folder entries available."),
        (DirectoryStatus.TRUNCATED, "Folder listing reached its safety limit."),
        (DirectoryStatus.FAILED, "Folder is unavailable."),
    ],
)
async def test_directory_states_are_explicitly_announced(
    status: DirectoryStatus, copy: str
) -> None:
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Unavailable", None, available=False),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await modal._publish_directory(DirectoryPage(status))
        assert modal.state.status_copy == copy
        assert str(modal.query_one("#console-workspace-files-status", Static).renderable) == copy


@pytest.mark.asyncio
async def test_directory_page_and_viewer_paging_keep_raw_identity_and_revision() -> None:
    """Load-more and Next send the captured raw ref, never a rendered label."""
    inspector = _Inspector([])
    revision = FileRevision(1, 2, 3, 4)
    continuation = DirectoryContinuation(
        "fingerprint", (), DirectoryRevision(1, 2, 4), 200, "opaque"
    )
    raw = ("unsafe[bold]", "file.txt")
    file_ref = FileRef(raw, "unsafe[bold]/file.txt")
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector, inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._next_generation()
        await modal._publish_directory(
            DirectoryPage(
                DirectoryStatus.PARTIAL,
                (DirectoryEntry(("first.txt",), "first.txt", False),),
                continuation,
            )
        )
        await pilot.pause()
        await pilot.click("#console-workspace-files-more")
        await pilot.pause()
        assert inspector.calls[-1] == ("list", (_scope(), (), continuation))

        modal._state = modal.state.__class__(
            selected_binding_id="binding-a", selected_file=file_ref,
            file_result=FileReadResult(
                FileReadKind.PAGED, revision=revision, text="page one",
                character_range=(0, 8), total_characters=16, next_page_offset=8,
            ),
            status_copy="Preview loaded.",
        )
        await modal._render_viewer()
        await pilot.click("#console-workspace-files-next")
        await pilot.pause()
        assert inspector.calls[-1] == ("read", (_scope(), raw, 8, revision))


@pytest.mark.asyncio
async def test_filter_enter_cancel_and_clear_restore_the_directory_view() -> None:
    inspector = _Inspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector, inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-workspace-files-filter")
        await pilot.press("n", "e", "e", "d", "l", "e", "enter")
        await pilot.pause()
        assert any(name == "filter" and call[1] == "needle" for name, call in inspector.calls)

        await pilot.click("#console-workspace-files-filter-cancel")
        assert modal.state.status_copy == "Filter cancelled."
        await pilot.click("#console-workspace-files-filter-clear")
        await pilot.pause()
        assert modal.state.filter_query == ""
        assert inspector.calls[-1][0] == "list"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "compact", "short"),
    [((80, 24), True, True), ((100, 30), True, False), ((120, 40), False, False), ((160, 50), False, False)],
)
async def test_production_bundle_geometry_keeps_pinned_controls_focusable(
    size: tuple[int, int], compact: bool, short: bool
) -> None:
    """The four supported terminal sizes retain a live back path and layout mode."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="workspace-with-a-long-id",
        inspected_workspace_name="A long inspected workspace name [literal]",
        active_workspace_id="active-workspace-with-a-long-id",
        active_workspace_name="A long active workspace name [literal]",
        bindings=(WorkspaceFilesBinding("binding-a", "A very long binding label [literal]", None, available=False),),
    )
    app = _Host()
    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        root = modal.query_one("#console-workspace-files-modal", Vertical)
        back = modal.query_one("#console-workspace-files-back", Button)
        details = modal.query_one("#console-workspace-files-details", Button)
        assert modal.has_class("-compact") is compact
        assert modal.has_class("-short") is short
        assert root.region.width > 0 and root.region.height > 0
        assert back.region.width > 0 and back.region.height > 0
        assert details.can_focus and details.focusable and details.tooltip
