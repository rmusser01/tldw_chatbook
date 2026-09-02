"""Behavior contracts for the read-only Console Workspace Files modal."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import inspect
from pathlib import Path
from threading import Event

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.geometry import Size
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
    ConsoleWorkspaceFilesModal,
    WorkspaceFilesAttention,
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
    FilterProgress,
    FilterResult,
    FilterStatus,
)


pytestmark = pytest.mark.ui


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

    def filter_paths(self, scope: BindingScope, query: str, *, is_cancelled=None, on_progress=None):
        self.calls.append(("filter", (scope, query)))
        return FilterResult(FilterStatus.EMPTY, status_copy="No matching paths.")

    def read_file(self, scope: BindingScope, raw_parts, *, page_offset=None, expected_revision=None):
        self.calls.append(("read", (scope, raw_parts, page_offset, expected_revision)))
        return FileReadResult(FileReadKind.TEXT, text="safe\\npreview")


class _Host(App[None]):
    _CSS_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    CSS_PATH = [
        str(_CSS_ROOT / "tldw_cli_modular.tcss"),
        str(_CSS_ROOT / "screen_agentic_console.tcss"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Button("Open files", id="files-opener")
            yield Input(id="console-chat-input")

    def _focus_console_composer_if_needed(self, *, force: bool = False) -> None:
        if force:
            self.query_one("#console-chat-input", Input).focus()


def _scope() -> BindingScope:
    return BindingScope("ws-a", "binding-a", "fingerprint", "/not-read", 1, 1)


async def _wait_for_thread_event(
    event: Event, pilot=None, *, what: str, attempts: int = 100
) -> None:
    """Poll one worker-thread signal with a hard test-local bound."""
    for _ in range(attempts):
        if event.is_set():
            return
        if pilot is None:
            await asyncio.sleep(0.02)
        else:
            await pilot.pause(0.02)
    pytest.fail(f"timed out waiting for {what}")


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
async def test_initial_focus_uses_the_selected_available_binding() -> None:
    """An unavailable leading binding cannot steal initial keyboard focus."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]),
        inspected_workspace_id="ws-a",
        inspected_workspace_name="Workspace A",
        active_workspace_id="ws-a",
        active_workspace_name="Workspace A",
        bindings=(
            WorkspaceFilesBinding("missing", "Missing", None, available=False),
            WorkspaceFilesBinding("binding-a", "First available", _scope()),
            WorkspaceFilesBinding(
                "binding-c",
                "Second available",
                BindingScope("ws-a", "binding-c", "fingerprint-c", "/not-read", 1, 2),
            ),
        ),
    )
    app = _Host()

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        assert modal.state.selected_binding_id == "binding-a"
        assert app.focused is modal.query_one(
            "#console-workspace-files-binding-1", Button
        )


@pytest.mark.asyncio
async def test_all_unavailable_bindings_show_access_changed_guidance_without_hiding_identity() -> None:
    """A stale nonempty binding set must not be presented as no folders at all."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]),
        inspected_workspace_id="ws-a",
        inspected_workspace_name="Workspace A",
        active_workspace_id="ws-a",
        active_workspace_name="Workspace A",
        bindings=(
            WorkspaceFilesBinding(
                "binding-a",
                "Project folder",
                None,
                available=False,
                availability_copy="Unavailable",
            ),
        ),
    )
    app = _Host()

    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        guidance = "This local folder is unavailable or its access changed. Update it in Settings."
        assert modal.state.status_copy == guidance
        assert "Project folder · Read-only · Unavailable" in str(
            modal.query_one("#console-workspace-files-binding-0", Button).label
        )
        assert str(modal.query_one("#console-workspace-files-status", Static).renderable) == guidance
        tree = modal.query_one("#console-workspace-files-tree", VerticalScroll)
        assert str(tree.query(Static).first(Static).renderable) == guidance


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
async def test_safe_dismissal_falls_back_to_composer_when_opener_recomposes_away() -> None:
    """Back remains safe when the row that launched the inspector no longer exists."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Unavailable", None, available=False),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        opener = app.query_one("#files-opener", Button)
        opener.focus()
        await app.push_screen(modal)
        await pilot.pause()
        await opener.remove()
        await pilot.click("#console-workspace-files-back")
        await pilot.pause()
        assert app.focused is app.query_one("#console-chat-input", Input)


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
    lane.submit(
        _LaneRequest(
            0,
            lambda: (entered.set(), release.wait(timeout=2), "done")[2],
            lambda _value: None,
        )
    )
    await _wait_for_thread_event(entered, what="lane operation start")
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
            lambda: (entered.set(), release.wait(timeout=2), "first")[2],
            publish,
        )
    )
    await _wait_for_thread_event(entered, what="coalesced lane operation start")
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
            lambda: (entered.set(), release.wait(timeout=2), "active")[2],
            publish,
        )
    )
    await _wait_for_thread_event(entered, what="closing lane operation start")
    lane.submit(_LaneRequest(5, lambda: "discarded", publish))
    closing = asyncio.create_task(lane.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    assert published == ["first", "latest", "active"]
    assert not lane.active and not lane.has_latest


@pytest.mark.asyncio
async def test_external_pop_tears_down_once_and_reports_a_closed_visit() -> None:
    """An app-driven pop must not strand a read lane or the visit ledger."""
    closed: list[str] = []
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Unavailable", None, available=False),),
        on_visit_closed=lambda: closed.append("closed"),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await app.pop_screen()
        await pilot.pause()
        assert app.screen is not modal
        assert closed == ["closed"]
        assert modal.owned_lane_count == 0


@pytest.mark.asyncio
async def test_external_pop_waits_for_an_active_owned_lane_before_closing() -> None:
    """A real app pop must join the lane rather than orphan its thread work."""
    entered = Event()
    release = Event()
    closed: list[str] = []

    async def _published(_value: object) -> None:
        return None

    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Unavailable", None, available=False),),
        on_visit_closed=lambda: closed.append("closed"),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._read_lane.submit(
            _LaneRequest(
                0,
                lambda: (entered.set(), release.wait(timeout=1), "done")[2],
                _published,
            )
        )
        await _wait_for_thread_event(entered, pilot, what="modal read operation start")
        # Textual returns an AwaitComplete rather than a bare coroutine here.
        popping = asyncio.ensure_future(app.pop_screen())
        await asyncio.sleep(0)
        assert not popping.done()
        release.set()
        await asyncio.wait_for(popping, timeout=2)
        await pilot.pause()
        assert closed == ["closed"]
        assert modal.owned_lane_count == 0


@pytest.mark.asyncio
async def test_attention_updates_are_generation_checked_and_private() -> None:
    """A stale attention snapshot cannot replace newer generic copy."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Unavailable", None, available=False),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        assert modal.update_attention(
            WorkspaceFilesAttention("Console needs attention · 2 approvals waiting"), 2
        ) is True
        assert modal.update_attention(WorkspaceFilesAttention("secret /path args"), 1) is False
        copy = str(modal.query_one("#console-workspace-files-attention", Static).renderable)
        assert copy == "Console needs attention · 2 approvals waiting"
        assert "secret" not in copy and "/path" not in copy and "args" not in copy


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
async def test_load_more_is_deduplicated_while_its_continuation_is_pending() -> None:
    """Rapid activation cannot submit the same one-shot continuation twice."""
    entered = Event()
    release = Event()

    class _BlockingInspector(_Inspector):
        def list_directory(self, scope, directory_parts=(), *, continuation=None):
            self.calls.append(("list", (scope, directory_parts, continuation)))
            if continuation is not None:
                entered.set()
                release.wait(timeout=2)
            return DirectoryPage(DirectoryStatus.COMPLETE)

    inspector = _BlockingInspector([])
    continuation = DirectoryContinuation(
        "fingerprint", (), DirectoryRevision(1, 2, 4), 200, "one-shot"
    )
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector,
        inspected_workspace_id="ws-a",
        inspected_workspace_name="A",
        active_workspace_id="ws-a",
        active_workspace_name="A",
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
                continuation=continuation,
            )
        )
        button = modal.query_one("#console-workspace-files-more", Button)
        button.press()
        button.press()
        await _wait_for_thread_event(entered, pilot, what="continuation request")
        release.set()
        for _ in range(100):
            if modal.owned_lane_count == 0:
                break
            await pilot.pause(0.02)

        continuation_calls = [
            call
            for name, call in inspector.calls
            if name == "list" and call[2] is continuation
        ]
        assert len(continuation_calls) == 1


@pytest.mark.asyncio
async def test_compact_viewer_has_a_focusable_back_to_files_route() -> None:
    """A compact preview must return to the tree without dismissing the visit."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()
    async with app.run_test(size=(100, 30)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._state = replace(
            modal.state,
            compact=True,
            compact_stage="viewer",
            selected_file=FileRef(("preview.txt",), "preview.txt"),
            file_result=FileReadResult(FileReadKind.TEXT, text="safe preview"),
        )
        modal._sync_layout()
        await modal._render_viewer()
        back_to_files = modal.query_one("#console-workspace-files-back-to-files", Button)
        assert back_to_files.can_focus and back_to_files.region.width > 0
        await pilot.click(back_to_files)
        await pilot.pause()
        assert modal.state.compact_stage == "tree"
        assert modal.query_one("#console-workspace-files-tree").display is True


@pytest.mark.asyncio
async def test_publishing_a_compact_preview_immediately_syncs_viewer_layout() -> None:
    """The state transition to viewer must update responsive classes immediately."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]),
        inspected_workspace_id="ws-a",
        inspected_workspace_name="A",
        active_workspace_id="ws-a",
        active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()
    async with app.run_test(size=(100, 30)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._state = replace(
            modal.state,
            selected_file=FileRef(("preview.txt",), "preview.txt"),
        )

        await modal._publish_file(FileReadResult(FileReadKind.TEXT, text="preview"))

        assert modal.state.compact_stage == "viewer"
        assert modal.has_class("-viewer-stage")


@pytest.mark.asyncio
async def test_invalid_utf8_preview_has_truthful_status_copy() -> None:
    """A rejected binary preview cannot announce that a preview loaded."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]),
        inspected_workspace_id="ws-a",
        inspected_workspace_name="A",
        active_workspace_id="ws-a",
        active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._state = replace(
            modal.state,
            selected_file=FileRef(("binary.bin",), "binary.bin"),
        )

        await modal._publish_file(
            FileReadResult(FileReadKind.INVALID_UTF8, error_code="invalid_utf8")
        )

        assert "UTF-8" in modal.state.status_copy
        assert "loaded" not in modal.state.status_copy.casefold()
        assert app.screen is modal


@pytest.mark.asyncio
@pytest.mark.parametrize("width", (80, 99, 100, 111))
async def test_compact_viewer_actions_fit_labels_and_remain_focusable(width: int) -> None:
    """Pinned compact actions reserve their labels and Textual button chrome."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()
    async with app.run_test(size=(width, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._state = replace(
            modal.state,
            compact_stage="viewer",
            selected_file=FileRef(("preview.txt",), "preview.txt"),
            file_result=FileReadResult(FileReadKind.TEXT, text="safe preview"),
        )
        modal._sync_layout()
        await modal._render_viewer()
        action_ids = (
            "console-workspace-files-back",
            "console-workspace-files-back-to-files",
            "console-workspace-files-details",
            "console-workspace-files-previous",
            "console-workspace-files-next",
            "console-workspace-files-refresh",
        )
        actions = [modal.query_one(f"#{action_id}", Button) for action_id in action_ids]
        assert all(button.can_focus and button.region.height > 0 for button in actions)
        widths = [
            (
                str(button.label),
                button.region.width,
                len(str(button.label)) + 2,
                str(button.styles.width),
            )
            for button in actions
        ]
        assert all(actual >= minimum for _label, actual, minimum, _style in widths), widths
        for left, right in zip(actions, actions[1:]):
            assert left.region.x + left.region.width <= right.region.x


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
        list_calls_before_clear = sum(name == "list" for name, _call in inspector.calls)
        await pilot.click("#console-workspace-files-filter-clear")
        await pilot.pause()
        assert modal.state.filter_query == ""
        assert sum(name == "list" for name, _call in inspector.calls) == list_calls_before_clear


@pytest.mark.asyncio
async def test_submitting_an_empty_filter_does_not_start_a_recursive_search() -> None:
    """Enter on a blank filter behaves like Clear and never calls the service."""
    inspector = _Inspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector,
        inspected_workspace_id="ws-a",
        inspected_workspace_name="A",
        active_workspace_id="ws-a",
        active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.click("#console-workspace-files-filter")
        await pilot.press("enter")
        await pilot.pause()

        assert all(name != "filter" for name, _call in inspector.calls)


@pytest.mark.asyncio
async def test_filter_progress_is_rendered_and_stale_worker_updates_cannot_overwrite_new_binding() -> None:
    """Catch the omitted progress bridge or a stale worker repainting a new binding."""
    entered = Event()
    release = Event()

    class _ProgressInspector(_Inspector):
        def filter_paths(self, scope, query, *, is_cancelled=None, on_progress=None):
            self.calls.append(("filter", (scope, query)))
            if on_progress is None:
                return FilterResult(FilterStatus.EMPTY, status_copy="No matching paths.")
            on_progress(FilterProgress(3, 1))
            entered.set()
            release.wait(timeout=1)
            on_progress(FilterProgress(99, 99))
            return FilterResult(FilterStatus.EMPTY, status_copy="No matching paths.")

    modal = ConsoleWorkspaceFilesModal(
        inspector=_ProgressInspector([]),
        inspected_workspace_id="ws-a",
        inspected_workspace_name="A",
        active_workspace_id="ws-a",
        active_workspace_name="A",
        bindings=(
            WorkspaceFilesBinding("binding-a", "Old", _scope()),
            WorkspaceFilesBinding("binding-b", "New", _scope()),
        ),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._request_filter("old")
        for _ in range(50):
            if entered.is_set() and "3 visited" in modal.state.status_copy:
                break
            await pilot.pause(0.02)
        assert entered.is_set()
        assert "3 visited" in modal.state.status_copy
        assert "1 result" in modal.state.status_copy

        await pilot.click("#console-workspace-files-binding-1")
        release.set()
        for _ in range(50):
            if modal.owned_lane_count == 0:
                break
            await pilot.pause(0.02)
        assert modal.state.selected_binding_id == "binding-b"
        assert "99 visited" not in modal.state.status_copy
        assert "99 result" not in modal.state.status_copy


@pytest.mark.asyncio
async def test_cancel_keeps_partial_filter_results_but_clear_restores_tree_state() -> None:
    """Canonical design §384: Cancel retains; Clear restores expansion/selection."""
    inspector = _Inspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector, inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    root_entry = DirectoryEntry(("folder",), "folder", True)
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._next_generation()
        await modal._publish_directory(
            DirectoryPage(DirectoryStatus.COMPLETE, (root_entry,))
        )
        modal._expand_directory(root_entry)
        await pilot.pause()
        await modal._publish_directory(
            DirectoryPage(
                DirectoryStatus.COMPLETE,
                (DirectoryEntry(("folder", "child.txt"), "child.txt", False),),
            ),
            directory_parts=("folder",),
        )
        modal._pre_filter_tree_state = replace(
            modal.state, filter_query="", filter_result=None
        )
        modal._state = replace(
            modal.state,
            filter_query="child",
            expanded_directory_parts=(),
            selected_tree_parts=None,
            filter_result=FilterResult(
                FilterStatus.PARTIAL,
                (FileRef(("folder", "child.txt"), "folder/child.txt"),),
                status_copy="Partial: 1 result.",
            ),
        )
        await modal._render_tree()
        await pilot.click("#console-workspace-files-filter-cancel")
        assert modal.state.filter_result is not None
        assert modal.state.filter_result.status is FilterStatus.PARTIAL
        assert "cancelled" in modal.state.status_copy.lower()

        await pilot.click("#console-workspace-files-filter-clear")
        await pilot.pause()
        assert modal.state.filter_query == ""
        assert modal.state.filter_result is None
        assert modal.state.expanded_directory_parts == (("folder",),)
        assert modal.state.selected_tree_parts == ("folder",)


@pytest.mark.asyncio
async def test_tree_expands_nested_directories_and_left_collapses_raw_subtree() -> None:
    inspector = _Inspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector, inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    root = DirectoryEntry(("folder",), "folder", True)
    nested = DirectoryEntry(("folder", "nested"), "nested", True)
    continuation = DirectoryContinuation(
        "fingerprint", ("folder",), DirectoryRevision(1, 2, 3), 200, "nested-page"
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._next_generation()
        await modal._publish_directory(DirectoryPage(DirectoryStatus.COMPLETE, (root,)))
        modal._expand_directory(root)
        await pilot.pause()
        await modal._publish_directory(
            DirectoryPage(DirectoryStatus.PARTIAL, (nested,), continuation), directory_parts=("folder",)
        )
        await pilot.pause()
        await pilot.click("#console-workspace-files-more-0")
        await pilot.pause()
        assert inspector.calls[-1] == ("list", (_scope(), ("folder",), continuation))
        modal._next_generation()
        await modal._publish_directory(
            DirectoryPage(DirectoryStatus.COMPLETE, (nested,)), directory_parts=("folder",)
        )
        modal._expand_directory(nested)
        await pilot.pause()
        assert modal.state.expanded_directory_parts == (("folder",), ("folder", "nested"))
        assert inspector.calls[-1] == ("list", (_scope(), ("folder", "nested"), None))

        await modal.action_collapse_or_parent()
        assert modal.state.expanded_directory_parts == (("folder",),)
        assert modal.state.selected_tree_parts == ("folder",)
        assert "collapsed" in modal.state.status_copy.lower()

        await modal.action_collapse_or_parent()
        assert modal.state.expanded_directory_parts == ()
        assert modal.state.selected_tree_parts is None


@pytest.mark.asyncio
async def test_clicking_an_expanded_directory_removes_descendant_rows_immediately() -> None:
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    root = DirectoryEntry(("folder",), "folder", True)
    child = DirectoryEntry(("folder", "child.txt"), "child.txt", False)
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._next_generation()
        await modal._publish_directory(DirectoryPage(DirectoryStatus.COMPLETE, (root,)))
        modal._expand_directory(root)
        await modal._publish_directory(
            DirectoryPage(DirectoryStatus.COMPLETE, (child,)), directory_parts=("folder",)
        )
        await pilot.pause()
        assert len(modal.query(".console-workspace-files-entry")) == 2

        await pilot.click("#console-workspace-files-entry-0")
        await pilot.pause()
        assert len(modal.query(".console-workspace-files-entry")) == 1
        assert modal.state.expanded_directory_parts == ()


@pytest.mark.asyncio
async def test_late_collapsed_child_list_result_cannot_reinsert_its_subtree() -> None:
    """A released worker result must prove its raw directory is still expanded."""
    entered = Event()
    release = Event()

    class _BarrierInspector(_Inspector):
        def list_directory(self, scope, directory_parts=(), *, continuation=None):
            self.calls.append(("list", (scope, directory_parts, continuation)))
            if directory_parts == ("folder",):
                entered.set()
                release.wait(timeout=2)
                return DirectoryPage(
                    DirectoryStatus.COMPLETE,
                    (DirectoryEntry(("folder", "late.txt"), "late.txt", False),),
                )
            return DirectoryPage(
                DirectoryStatus.COMPLETE,
                (DirectoryEntry(("folder",), "folder", True),),
            )

    inspector = _BarrierInspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector, inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()
        root = modal.state.directory_page.entries[0]
        modal._expand_directory(root)
        await _wait_for_thread_event(entered, pilot, what="child directory listing")

        await pilot.click("#console-workspace-files-entry-0")
        assert modal.state.expanded_directory_parts == ()
        release.set()
        await pilot.pause()
        await pilot.pause()
        assert modal._page_for(("folder",)) is None
        assert "late.txt" not in str(modal.query_one("#console-workspace-files-tree").render())
        assert modal.state.selected_tree_parts is None


@pytest.mark.asyncio
async def test_binding_change_discards_filter_snapshot_and_old_tree_before_clear() -> None:
    inspector = _Inspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector, inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(
            WorkspaceFilesBinding("binding-a", "Old", _scope()),
            WorkspaceFilesBinding("binding-b", "New", _scope()),
        ),
    )
    old_folder = DirectoryEntry(("old-folder",), "old-folder", True)
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._next_generation()
        await modal._publish_directory(DirectoryPage(DirectoryStatus.COMPLETE, (old_folder,)))
        modal._state = replace(
            modal.state,
            expanded_directory_parts=(("old-folder",),),
            selected_tree_parts=("old-folder",),
            filter_query="old",
            filter_result=FilterResult(FilterStatus.PARTIAL, status_copy="Partial old."),
        )
        modal._pre_filter_tree_state = modal.state
        modal.query_one("#console-workspace-files-filter", Input).value = "old"

        await pilot.click("#console-workspace-files-binding-1")
        await pilot.pause()
        await pilot.click("#console-workspace-files-filter-clear")
        await pilot.pause()
        assert modal.state.selected_binding_id == "binding-b"
        assert modal.state.filter_query == "" and modal.state.filter_result is None
        assert modal.state.expanded_directory_parts == ()
        assert modal.state.selected_tree_parts is None
        assert modal._pre_filter_tree_state is None
        assert modal.query_one("#console-workspace-files-filter", Input).value == ""


@pytest.mark.asyncio
async def test_old_directory_result_cannot_publish_after_binding_round_trip() -> None:
    """A→B→A selection still fences the first A request by visit generation."""
    first_entered = Event()
    first_release = Event()
    second_entered = Event()
    second_release = Event()
    a_calls = 0

    class _RoundTripInspector(_Inspector):
        def list_directory(self, scope, directory_parts=(), *, continuation=None):
            nonlocal a_calls
            self.calls.append(("list", (scope, directory_parts, continuation)))
            if scope.binding_id == "binding-a":
                a_calls += 1
                if a_calls == 1:
                    first_entered.set()
                    first_release.wait(timeout=2)
                    return DirectoryPage(
                        DirectoryStatus.COMPLETE,
                        (DirectoryEntry(("stale.txt",), "stale.txt", False),),
                    )
                second_entered.set()
                second_release.wait(timeout=2)
                return DirectoryPage(
                    DirectoryStatus.COMPLETE,
                    (DirectoryEntry(("current.txt",), "current.txt", False),),
                )
            return DirectoryPage(DirectoryStatus.COMPLETE)

    inspector = _RoundTripInspector([])
    scope_a = _scope()
    scope_b = BindingScope(
        "ws-a", "binding-b", "fingerprint-b", "/not-read", 1, 2
    )
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector,
        inspected_workspace_id="ws-a",
        inspected_workspace_name="A",
        active_workspace_id="ws-a",
        active_workspace_name="A",
        bindings=(
            WorkspaceFilesBinding("binding-a", "A", scope_a),
            WorkspaceFilesBinding("binding-b", "B", scope_b),
        ),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await _wait_for_thread_event(first_entered, pilot, what="first A listing")
        await pilot.click("#console-workspace-files-binding-1")
        await pilot.click("#console-workspace-files-binding-0")
        first_release.set()
        await _wait_for_thread_event(second_entered, pilot, what="second A listing")

        assert modal.state.directory_page is None or all(
            entry.raw_parts != ("stale.txt",)
            for entry in modal.state.directory_page.entries
        )

        second_release.set()
        for _ in range(100):
            if modal.owned_lane_count == 0:
                break
            await pilot.pause(0.02)
        assert modal.state.directory_page is not None
        assert [entry.raw_parts for entry in modal.state.directory_page.entries] == [
            ("current.txt",)
        ]


def test_workspace_files_protocol_documents_google_style_contracts() -> None:
    """The modal's public service seam documents each operation contract."""
    from tldw_chatbook.Widgets.Console.console_workspace_files_modal import (
        WorkspaceFilesService,
    )

    for method_name in ("list_directory", "filter_paths", "read_file"):
        docstring = inspect.getdoc(getattr(WorkspaceFilesService, method_name)) or ""
        assert "Args:" in docstring
        assert "Returns:" in docstring


@pytest.mark.asyncio
async def test_load_more_merges_unique_entries_without_losing_page_one() -> None:
    """Continuation results append by raw identity and carry their new token."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    continuation_one = DirectoryContinuation(
        "fingerprint", (), DirectoryRevision(1, 2, 3), 200, "page-one"
    )
    continuation_two = DirectoryContinuation(
        "fingerprint", (), DirectoryRevision(1, 2, 3), 400, "page-two"
    )
    first = DirectoryEntry(("first.txt",), "first.txt", False)
    duplicate = DirectoryEntry(("first.txt",), "first duplicate", False)
    second = DirectoryEntry(("second.txt",), "second.txt", False)
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal._next_generation()
        await modal._publish_directory(
            DirectoryPage(DirectoryStatus.PARTIAL, (first,), continuation_one)
        )
        await modal._publish_directory(
            DirectoryPage(DirectoryStatus.PARTIAL, (duplicate, second), continuation_two),
            continuation=continuation_one,
        )
        await pilot.pause()
        page = modal.state.directory_page
        assert page is not None
        assert [entry.raw_parts for entry in page.entries] == [
            ("first.txt",), ("second.txt",)
        ]
        assert page.continuation == continuation_two
        assert len(modal.query(".console-workspace-files-entry")) == 2


def test_load_more_caps_merged_entries_and_marks_truncation_honestly() -> None:
    existing = DirectoryPage(
        DirectoryStatus.PARTIAL,
        tuple(
            DirectoryEntry((f"{index}.txt",), f"{index}.txt", False)
            for index in range(10_000)
        ),
    )
    incoming = DirectoryPage(
        DirectoryStatus.PARTIAL,
        (DirectoryEntry(("overflow.txt",), "overflow.txt", False),),
    )

    merged = ConsoleWorkspaceFilesModal._merge_directory_page(existing, incoming)

    assert merged.status is DirectoryStatus.TRUNCATED
    assert len(merged.entries) == 10_000
    assert merged.continuation is None


@pytest.mark.asyncio
async def test_directory_scoped_a_then_b_list_results_publish_in_order() -> None:
    """B becoming latest must not stale an already-expanded A request."""
    entered_a = Event()
    release_a = Event()

    class _TwoDirectoryInspector(_Inspector):
        def list_directory(self, scope, directory_parts=(), *, continuation=None):
            self.calls.append(("list", (scope, directory_parts, continuation)))
            if directory_parts == ("a",):
                entered_a.set()
                release_a.wait(timeout=2)
                return DirectoryPage(DirectoryStatus.COMPLETE, (DirectoryEntry(("a", "a.txt"), "a.txt", False),))
            if directory_parts == ("b",):
                return DirectoryPage(DirectoryStatus.COMPLETE, (DirectoryEntry(("b", "b.txt"), "b.txt", False),))
            return DirectoryPage(DirectoryStatus.COMPLETE, (
                DirectoryEntry(("a",), "a", True), DirectoryEntry(("b",), "b", True),
            ))

    inspector = _TwoDirectoryInspector([])
    modal = ConsoleWorkspaceFilesModal(
        inspector=inspector, inspected_workspace_id="ws-a", inspected_workspace_name="A",
        active_workspace_id="ws-a", active_workspace_name="A",
        bindings=(WorkspaceFilesBinding("binding-a", "Project", _scope()),),
    )
    app = _Host()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()
        root = modal.state.directory_page.entries
        modal._expand_directory(root[0])
        await _wait_for_thread_event(
            entered_a, pilot, what="first sibling directory listing"
        )
        modal._expand_directory(root[1])
        release_a.set()
        await pilot.pause()
        await pilot.pause()
        assert modal._page_for(("a",)) is not None
        assert modal._page_for(("b",)) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "compact", "fullscreen", "short"),
    [
        ((80, 24), True, True, True),
        ((99, 30), True, True, False),
        ((100, 30), True, False, False),
        ((101, 30), True, False, False),
        ((111, 30), True, False, False),
        ((112, 30), False, False, False),
        ((120, 40), False, False, False),
        ((160, 50), False, False, False),
    ],
)
async def test_production_bundle_geometry_keeps_pinned_controls_focusable(
    size: tuple[int, int], compact: bool, fullscreen: bool, short: bool
) -> None:
    """The four supported terminal sizes retain a live back path and layout mode."""
    modal = ConsoleWorkspaceFilesModal(
        inspector=_Inspector([]), inspected_workspace_id="workspace-with-a-long-id",
        inspected_workspace_name="A long inspected workspace name [literal]",
        active_workspace_id="active-workspace-with-a-long-id",
        active_workspace_name="A long active workspace name [literal]",
        bindings=(WorkspaceFilesBinding("binding-a", "A very long binding label [literal]", None, available=False),),
        attention=WorkspaceFilesAttention(
            status_copy="Console needs attention · approval waiting",
            pending_approval_count=1,
        ),
    )
    app = _Host()
    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        root = modal.query_one("#console-workspace-files-modal", Vertical)
        back = modal.query_one("#console-workspace-files-back", Button)
        details = modal.query_one("#console-workspace-files-details", Button)
        assert modal.has_class("-compact") is compact
        assert modal.has_class("-fullscreen") is fullscreen
        assert modal.has_class("-short") is short
        assert root.region.width > 0 and root.region.height > 0
        assert back.region.width > 0 and back.region.height > 0
        assert details.can_focus and details.focusable and details.tooltip
        if short:
            attention = modal.query_one("#console-workspace-files-attention", Static)
            fold = modal.query_one("#console-workspace-files-fold", Static)
            actions = modal.query_one("#console-workspace-files-actions", Horizontal)
            assert attention.region.width > 0 and attention.region.height > 0
            assert fold.region.width > 0 and fold.region.height > 0
            assert actions.region.width > 0 and actions.region.height > 0
