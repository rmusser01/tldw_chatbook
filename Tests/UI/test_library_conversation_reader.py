"""Mounted Conversations reader journeys through the production Library screen."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest
from textual.containers import VerticalScroll
from textual.events import DescendantFocus
from textual.widgets import Button, Input, Static
from textual.worker import WorkerCancelled

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens.library_screen import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_NOTES,
    LibraryScreen,
)
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationFindMatch,
    ConversationMessageView,
    ConversationReaderState,
)
from tldw_chatbook.Widgets.Library import (
    LibraryAdaptiveReaderShell,
    LibraryConversationReader,
    LibraryConversationsCanvas,
)


def _conversation_records() -> list[dict[str, object]]:
    return [
        {
            "id": "chat-a",
            "title": "Alpha planning",
            "version": 4,
            "message_count": 1,
            "last_modified": "2026-08-23T12:00:00Z",
            "keywords": ["alpha", "planning"],
        },
        {
            "id": "chat-b",
            "title": "Beta review",
            "version": 7,
            "message_count": 1,
            "last_modified": "2026-08-24T12:00:00Z",
        },
    ]


def _active_conversations_screen(app) -> LibraryScreen:
    screen = LibraryScreen(app)
    screen.restore_state({"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS})
    return screen


def _loaded_reader_state() -> ConversationReaderState:
    return ConversationReaderState(
        selected_id="chat-a",
        selected_version=4,
        loaded_id="chat-a",
        loaded_version=4,
        loaded_generation=2,
        generation=2,
        messages=(
            ConversationMessageView(
                "message-a", "user", "2026-08-23T12:01:00Z", "revision-a", 5, "hello"
            ),
        ),
        message_total=1,
        complete=True,
    )


@pytest.mark.asyncio
async def test_pending_error_and_unavailable_selection_keep_loaded_metadata_truth(
    widget_pilot,
) -> None:
    loaded = _loaded_reader_state()
    pending = replace(
        loaded,
        selected_id="chat-b",
        selected_version=7,
        generation=3,
        mode="info",
        loading=True,
        complete=True,
    )
    loaded_metadata = {
        "title": "Alpha planning",
        "keywords": ["alpha"],
        "last_modified": "2026-08-23T12:00:00Z",
    }
    selected_metadata = {"title": "Beta review"}
    async with await widget_pilot(
        LibraryConversationReader,
        state=pending,
        loaded_metadata=loaded_metadata,
        selected_metadata=selected_metadata,
        id="library-conversation-reader",
    ) as pilot:
        reader = pilot.app.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        await pilot.pause()
        info = str(
            reader.query_one(
                "#library-conversation-reader-info-body", Static
            ).renderable
        )
        status = str(
            reader.query_one("#library-conversation-reader-status", Static).renderable
        )
        assert "Title: Alpha planning" in info
        assert "Conversation ID: chat-a" in info
        assert "Version: 4" in info
        assert "Keywords: alpha" in info
        assert "Loading Beta review (chat-b); showing Alpha planning (chat-a)" in status

        for changed in (
            replace(pending, loading=False, error="detail failed"),
            replace(
                pending,
                loading=False,
                error="Conversation unavailable.",
                unavailable=True,
            ),
        ):
            reader.sync_state(
                changed,
                loaded_metadata=loaded_metadata,
                selected_metadata=selected_metadata,
            )
            await pilot.pause()
            assert "Title: Alpha planning" in str(
                reader.query_one(
                    "#library-conversation-reader-info-body", Static
                ).renderable
            )
            assert "Beta review (chat-b)" in str(
                reader.query_one(
                    "#library-conversation-reader-status", Static
                ).renderable
            )


@pytest.mark.asyncio
async def test_open_console_requires_final_complete_error_free_match(
    widget_pilot,
) -> None:
    loaded = _loaded_reader_state()
    first_page = replace(loaded, complete=False, loading=True, message_total=2)
    async with await widget_pilot(
        LibraryConversationReader,
        state=first_page,
        loaded_metadata={"title": "Alpha planning"},
        selected_metadata={"title": "Alpha planning"},
        id="library-conversation-reader",
    ) as pilot:
        reader = pilot.app.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        action = reader.query_one("#library-conversation-open-console", Button)
        assert action.disabled
        continuation = replace(
            loaded,
            messages=(replace(loaded.messages[0], total_chars=12),),
            complete=False,
            loading=True,
        )
        reader.sync_state(continuation)
        await pilot.pause()
        assert action.disabled
        reader.sync_state(replace(first_page, loading=False, error="later page failed"))
        await pilot.pause()
        assert action.disabled
        reader.sync_state(loaded)
        await pilot.pause()
        assert not action.disabled
        assert action.tooltip is None


@pytest.mark.asyncio
async def test_conversations_mount_three_retained_roles_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records())
    compose_calls: list[LibraryScreen] = []
    original_compose = LibraryScreen.compose_content

    def recorded_compose(screen: LibraryScreen):
        compose_calls.append(screen)
        return original_compose(screen)

    monkeypatch.setattr(LibraryScreen, "compose_content", recorded_compose)
    screen = _active_conversations_screen(app)
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-conversation-reader")

        shell = screen.query_one(
            "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
        )
        rail = shell.query_one("#library-rail")
        items = shell.query_one(
            "#library-conversations-canvas", LibraryConversationsCanvas
        )
        work = shell.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        identities = (id(shell), id(rail), id(items), id(work))

        shell.library_grip.press()
        await pilot.pause()
        shell.library_grip.press()
        await pilot.pause()
        shell.items_grip.press()
        await pilot.pause()
        shell.items_grip.press()
        await pilot.resize_terminal(120, 35)
        await pilot.pause()

        assert (id(shell), id(rail), id(items), id(work)) == identities
        assert shell.work is work and work.display and work.is_mounted
        assert len(shell.query(".library-adaptive-reader-pane-grip")) == 2
        assert compose_calls.count(screen) == 1


@pytest.mark.asyncio
async def test_reader_info_is_explicit_and_truthful() -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records())
    screen = _active_conversations_screen(app)
    screen._library_conversation_reader_state = ConversationReaderState(
        selected_id="chat-a",
        selected_version=4,
        loaded_id="chat-a",
        loaded_version=4,
        loaded_generation=2,
        generation=2,
        messages=(
            ConversationMessageView(
                "message-1",
                "user",
                "2026-08-23T12:01:00Z",
                "revision-1",
                5,
                "hello",
            ),
        ),
        message_total=1,
        complete=True,
    )
    screen._library_conversation_reader_loaded_metadata = _conversation_records()[0]
    screen._library_conversation_reader_selected_metadata = _conversation_records()[0]
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        await _wait_for_library_shell(active, pilot)
        reader = active.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        assert reader.state.mode == "read"
        assert reader.query_one("#library-conversation-reader-messages").display

        reader.query_one("#library-conversation-reader-info", Button).press()
        await pilot.pause()

        body = reader.query_one("#library-conversation-reader-info-body", Static)
        copy = str(body.renderable)
        assert reader.state.mode == "info"
        assert body.display
        assert "Title: Alpha planning" in copy
        assert "Conversation ID: chat-a" in copy
        assert "Version: 4" in copy
        assert "Messages: 1" in copy
        assert "Workspace: unassigned" in copy
        assert "Keywords: alpha, planning" in copy
        assert "Authority: local saved conversation" in copy


class _ProgressiveConversationService:
    """Synchronous detail seam whose second bounded page is test-gated."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.thread_ids: list[int] = []
        self.second_started = threading.Event()
        self.release_second = threading.Event()

    def get_library_conversation_messages(self, conversation_id: str, **kwargs):
        self.calls.append({"conversation_id": conversation_id, **kwargs})
        self.thread_ids.append(threading.get_ident())
        offset = int(kwargs.get("message_offset", 0))
        limit = int(kwargs.get("message_limit", 20))
        if offset == 20:
            self.second_started.set()
            assert self.release_second.wait(timeout=10)
        total = 21
        stop = min(offset + limit, total)
        messages = []
        for index in range(offset, stop):
            text = "later needle" if index == 20 else f"message {index}"
            messages.append(
                {
                    "id": f"message-{index}",
                    "sender": "user" if index % 2 == 0 else "assistant",
                    "timestamp": f"2026-08-24T12:{index:02d}:00Z",
                    "revision": f"revision-{index}",
                    "total_chars": len(text),
                    "char_start": 0,
                    "returned_chars": len(text),
                    "has_more": False,
                    "text": text,
                }
            )
        next_offset = stop if stop < total else None
        return {
            "id": conversation_id,
            "title": "Alpha planning",
            "version": 4,
            "message_total": total,
            "message_offset": offset,
            "returned_message_count": len(messages),
            "has_more": next_offset is not None,
            "next_message_offset": next_offset,
            "include_rag_context": False,
            "messages": messages,
        }


class _OutOfOrderConversationService:
    """Hold the first selection until a later selection has settled."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def get_library_conversation_messages(self, conversation_id: str, **kwargs):
        call = {"conversation_id": conversation_id, **kwargs}
        self.calls.append(call)
        if len(self.calls) == 1:
            self.first_started.set()
            assert self.release_first.wait(timeout=10)
        version = 4 if conversation_id == "chat-a" else 7
        text = f"transcript for {conversation_id}"
        return {
            "id": conversation_id,
            "title": conversation_id,
            "version": version,
            "message_total": 1,
            "message_offset": int(kwargs.get("message_offset", 0)),
            "returned_message_count": 1,
            "has_more": False,
            "next_message_offset": None,
            "include_rag_context": False,
            "messages": [
                {
                    "id": f"message-{conversation_id}",
                    "sender": "user",
                    "timestamp": "2026-08-24T12:00:00Z",
                    "revision": f"revision-{conversation_id}",
                    "total_chars": len(text),
                    "char_start": 0,
                    "returned_chars": len(text),
                    "has_more": False,
                    "text": text,
                }
            ],
        }


class _ContinuationConversationService:
    """Gate one long body window after its first bounded prefix paints."""

    def __init__(self) -> None:
        self.continuation_started = threading.Event()
        self.release_continuation = threading.Event()

    def get_library_conversation_messages(self, conversation_id: str, **kwargs):
        full = "prefix hidden needle"
        if kwargs.get("message_id"):
            self.continuation_started.set()
            assert self.release_continuation.wait(timeout=10)
            start = int(kwargs["char_start"])
            text = full[start:]
            return {
                "id": conversation_id,
                "version": 4,
                "message_total": 1,
                "messages": [
                    {
                        "id": "message-long",
                        "sender": "user",
                        "timestamp": "2026-08-24T12:00:00Z",
                        "revision": "revision-long",
                        "total_chars": len(full),
                        "char_start": start,
                        "returned_chars": len(text),
                        "has_more": False,
                        "text": text,
                    }
                ],
            }
        prefix = "prefix "
        return {
            "id": conversation_id,
            "title": "Alpha planning",
            "version": 4,
            "message_total": 1,
            "message_offset": 0,
            "returned_message_count": 1,
            "has_more": False,
            "next_message_offset": None,
            "include_rag_context": False,
            "messages": [
                {
                    "id": "message-long",
                    "sender": "user",
                    "timestamp": "2026-08-24T12:00:00Z",
                    "revision": "revision-long",
                    "total_chars": len(full),
                    "char_start": 0,
                    "returned_chars": len(prefix),
                    "has_more": True,
                    "text": prefix,
                }
            ],
        }


class _GatedVersionConversationService:
    def __init__(self, version: int) -> None:
        self.version = version
        self.started = threading.Event()
        self.release = threading.Event()

    def get_library_conversation_messages(self, conversation_id: str, **kwargs):
        self.started.set()
        assert self.release.wait(timeout=10)
        text = f"version {self.version}"
        return {
            "id": conversation_id,
            "title": f"Alpha v{self.version}",
            "version": self.version,
            "message_total": 1,
            "message_offset": 0,
            "returned_message_count": 1,
            "has_more": False,
            "next_message_offset": None,
            "include_rag_context": False,
            "messages": [
                {
                    "id": f"message-v{self.version}",
                    "sender": "user",
                    "timestamp": "2026-08-24T12:00:00Z",
                    "revision": f"revision-v{self.version}",
                    "total_chars": len(text),
                    "char_start": 0,
                    "returned_chars": len(text),
                    "has_more": False,
                    "text": text,
                }
            ],
        }


class _GatedFailureConversationService:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.started = threading.Event()
        self.release = threading.Event()

    def get_library_conversation_messages(self, conversation_id: str, **_kwargs):
        self.started.set()
        assert self.release.wait(timeout=10)
        if self.outcome == "unavailable":
            return None
        return {
            "id": conversation_id,
            "title": "Beta response must not relabel Alpha",
            "version": 7,
            "message_total": 1,
            "message_offset": 0,
            "returned_message_count": 1,
            "has_more": False,
            "next_message_offset": None,
            "messages": "invalid",
        }


class _GatedBootstrapConversationService:
    """Gate a missing-version bootstrap and return one configured outcome."""

    def __init__(self, outcome: str = "success") -> None:
        self.outcome = outcome
        self.started = threading.Event()
        self.release = threading.Event()

    def get_library_conversation_messages(self, conversation_id: str, **_kwargs):
        self.started.set()
        assert self.release.wait(timeout=10)
        if self.outcome == "exception":
            raise RuntimeError("bootstrap failed")
        if self.outcome == "unavailable":
            return None
        text = "bootstrap transcript"
        return {
            "id": conversation_id,
            "title": "Bootstrap title",
            "version": 4,
            "message_total": 1,
            "message_offset": 0,
            "returned_message_count": 1,
            "has_more": False,
            "next_message_offset": None,
            "messages": [
                {
                    "id": "message-bootstrap",
                    "sender": "user",
                    "timestamp": "2026-08-24T12:00:00Z",
                    "revision": "revision-bootstrap",
                    "total_chars": len(text),
                    "char_start": 0,
                    "returned_chars": len(text),
                    "has_more": False,
                    "text": text,
                }
            ],
        }


class _RejectedLaterPageConversationService:
    """Accept page one, then return a duplicate-ID page with hostile metadata."""

    def get_library_conversation_messages(self, conversation_id: str, **kwargs):
        offset = int(kwargs.get("message_offset", 0))
        if offset:
            text = "duplicate"
            return {
                "id": conversation_id,
                "title": "REJECTED TITLE",
                "version": 4,
                "message_total": 21,
                "message_offset": 20,
                "returned_message_count": 1,
                "has_more": False,
                "next_message_offset": None,
                "messages": [
                    {
                        "id": "message-0",
                        "sender": "user",
                        "timestamp": "2026-08-24T12:20:00Z",
                        "revision": "revision-0",
                        "total_chars": len(text),
                        "char_start": 0,
                        "returned_chars": len(text),
                        "has_more": False,
                        "text": text,
                    }
                ],
            }
        messages = []
        for index in range(20):
            text = f"message {index}"
            messages.append(
                {
                    "id": f"message-{index}",
                    "sender": "user",
                    "timestamp": f"2026-08-24T12:{index:02d}:00Z",
                    "revision": f"revision-{index}",
                    "total_chars": len(text),
                    "char_start": 0,
                    "returned_chars": len(text),
                    "has_more": False,
                    "text": text,
                }
            )
        return {
            "id": conversation_id,
            "title": "Accepted title",
            "version": 4,
            "message_total": 21,
            "message_offset": 0,
            "returned_message_count": 20,
            "has_more": True,
            "next_message_offset": 20,
            "messages": messages,
        }


class _GatedFindRetryConversationService:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def get_library_conversation_messages(self, conversation_id: str, **_kwargs):
        self.started.set()
        assert self.release.wait(timeout=10)
        text = "retry reveals needle"
        return {
            "id": conversation_id,
            "title": "Alpha planning",
            "version": 4,
            "message_total": 1,
            "message_offset": 0,
            "returned_message_count": 1,
            "has_more": False,
            "next_message_offset": None,
            "messages": [
                {
                    "id": "message-retry",
                    "sender": "user",
                    "timestamp": "2026-08-24T12:00:00Z",
                    "revision": "revision-retry",
                    "total_chars": len(text),
                    "char_start": 0,
                    "returned_chars": len(text),
                    "has_more": False,
                    "text": text,
                }
            ],
        }


def _missing_version_record() -> dict[str, object]:
    return {
        key: value
        for key, value in _conversation_records()[0].items()
        if key != "version"
    }


@pytest.mark.parametrize(
    ("outcome", "unavailable", "copy"),
    (
        ("exception", False, "try again"),
        ("unavailable", True, "unavailable"),
    ),
)
@pytest.mark.asyncio
async def test_missing_version_bootstrap_distinguishes_error_from_unavailable(
    outcome: str,
    unavailable: bool,
    copy: str,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, [_missing_version_record()])
    service = _GatedBootstrapConversationService(outcome)
    app.local_chat_conversation_service = service
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await asyncio.to_thread(service.started.wait, 10)
        try:
            state = screen._library_conversation_reader_state
            assert state.selected_version is None and state.loading
        finally:
            service.release.set()
        try:
            await screen.workers.wait_for_complete()
        except WorkerCancelled:
            pass
        state = screen._library_conversation_reader_state
        assert state.unavailable is unavailable
        assert state.error and copy in state.error.casefold()
        assert state.selected_version is None and not state.loading


@pytest.mark.asyncio
async def test_missing_version_bootstrap_loses_authority_on_leave_return() -> None:
    app = _build_test_app()
    _seed_conversations(app, [_missing_version_record()])
    service = _GatedBootstrapConversationService()
    app.local_chat_conversation_service = service
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await asyncio.to_thread(service.started.wait, 10)
        bootstrap_generation = screen._library_conversation_reader_state.generation
        try:
            await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
            await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
            assert screen._library_conversation_reader_state.generation > (
                bootstrap_generation
            )
        finally:
            service.release.set()
        try:
            await screen.workers.wait_for_complete()
        except WorkerCancelled:
            pass
        state = screen._library_conversation_reader_state
        assert state.loaded_generation != bootstrap_generation


@pytest.mark.asyncio
async def test_missing_version_bootstrap_loses_authority_on_real_unmount() -> None:
    app = _build_test_app()
    _seed_conversations(app, [_missing_version_record()])
    service = _GatedBootstrapConversationService()
    app.local_chat_conversation_service = service
    screen = _active_conversations_screen(app)
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        await _wait_for_library_shell(active, pilot)
        await asyncio.to_thread(service.started.wait, 10)
        bootstrap_generation = active._library_conversation_reader_state.generation
        try:
            pop_task = asyncio.ensure_future(host.pop_screen())
            while active._library_conversation_reader_mounted_authority:
                await asyncio.sleep(0)
        finally:
            service.release.set()
        await pop_task
        await active.workers.wait_for_complete()
        state = active._library_conversation_reader_state
        assert state.generation > bootstrap_generation
        assert state.selected_version is None and state.loaded_id is None


@pytest.mark.asyncio
async def test_rejected_later_page_cannot_promote_hostile_metadata() -> None:
    app = _build_test_app()
    record = _conversation_records()[0]
    _seed_conversations(app, [record])
    app.local_chat_conversation_service = _RejectedLaterPageConversationService()
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen.workers.wait_for_complete()
        state = screen._library_conversation_reader_state
        assert state.error and "progress" in state.error.casefold()
        assert not state.complete and len(state.messages) == 20
        assert screen._library_conversation_reader_loaded_metadata["title"] == (
            "Accepted title"
        )
        assert screen._library_conversation_reader_loaded_metadata["version"] == 4
        assert screen._library_conversation_reader_loaded_metadata["keywords"] == [
            "alpha",
            "planning",
        ]
        assert screen._library_conversation_reader_selected_metadata == record


@pytest.mark.parametrize("steal_focus", (False, True))
@pytest.mark.asyncio
async def test_find_retry_reveals_match_only_for_current_focus_intent(
    steal_focus: bool,
) -> None:
    app = _build_test_app()
    record = _conversation_records()[0]
    _seed_conversations(app, [record])
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen.workers.wait_for_complete()
        service = _GatedFindRetryConversationService()
        app.local_chat_conversation_service = service
        screen._library_conversation_reader_state = replace(
            _loaded_reader_state(),
            complete=False,
            error="later page failed",
            loading=False,
        )
        screen._sync_library_conversation_reader()
        reader = screen.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        find = reader.query_one("#library-conversation-reader-find", Input)
        find.value = "needle"
        find.focus()
        await pilot.press("enter")
        await asyncio.to_thread(service.started.wait, 10)
        replacement_focus = reader.query_one(
            "#library-conversation-reader-info", Button
        )
        try:
            if steal_focus:
                replacement_focus.focus()
                await pilot.pause()
        finally:
            service.release.set()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        state = screen._library_conversation_reader_state
        match = next(
            row
            for row in reader.query(".library-conversation-reader-message")
            if getattr(row, "message_id", None) == "message-retry"
        )
        assert state.complete and state.find_matches
        if not steal_focus:
            for _ in range(10):
                if match.has_focus:
                    break
                await pilot.pause()
        assert match.has_focus is (not steal_focus)
        if steal_focus:
            assert replacement_focus.has_focus


@pytest.mark.parametrize("steal_focus", (False, True))
@pytest.mark.asyncio
async def test_messages_synced_revalidates_find_focus_before_deferred_reveal(
    monkeypatch: pytest.MonkeyPatch,
    steal_focus: bool,
) -> None:
    app = _build_test_app()
    record = _conversation_records()[0]
    _seed_conversations(app, [record])
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen.workers.wait_for_complete()
        reader = screen.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        messages = reader.query_one(
            "#library-conversation-reader-messages", VerticalScroll
        )
        match_message = ConversationMessageView(
            "message-deferred",
            "assistant",
            "2026-08-24T12:30:00Z",
            "revision-deferred",
            15,
            "deferred needle",
        )
        state = replace(
            _loaded_reader_state(),
            messages=(match_message,),
            find_query="needle",
            find_matches=(ConversationFindMatch("message-deferred", 0, 9, 9, 6),),
            find_complete=True,
        )
        screen._library_conversation_reader_state = state
        screen._library_conversation_find_focus_intent = (
            state.generation,
            screen._library_notes_focus_intent_generation,
            state.find_query,
        )

        mount_started = asyncio.Event()
        mount_finished = asyncio.Event()
        release_mount = asyncio.Event()
        original_mount = VerticalScroll.mount

        async def gated_mount(container, *widgets, **kwargs):
            if container is messages:
                mount_started.set()
                await release_mount.wait()
            await original_mount(container, *widgets, **kwargs)
            if container is messages:
                mount_finished.set()

        monkeypatch.setattr(VerticalScroll, "mount", gated_mount)
        focus_calls: list[str] = []
        original_focus = reader.focus_find_match

        def tracked_focus(message_id: str) -> bool:
            focus_calls.append(message_id)
            return original_focus(message_id)

        monkeypatch.setattr(reader, "focus_find_match", tracked_focus)
        replacement_focus = reader.query_one("#library-conversation-reader-find", Input)
        try:
            screen._sync_library_conversation_reader()
            await asyncio.wait_for(mount_started.wait(), timeout=10)
            for _ in range(100):
                if focus_calls:
                    break
                await asyncio.sleep(0)
            assert focus_calls == ["message-deferred"]
            if steal_focus:
                screen._library_notes_programmatic_focus_target = None
                screen._library_notes_restoring_focus = False
                screen._library_notes_resize_settling = False
                prior_focus_generation = screen._library_notes_focus_intent_generation
                assert replacement_focus.focusable
                screen.set_focus(None)
                screen.set_focus(replacement_focus)
                screen.on_descendant_focus(DescendantFocus(replacement_focus))
                assert screen.focused is replacement_focus
                assert (
                    screen._library_notes_focus_intent_generation
                    > prior_focus_generation
                )
        finally:
            release_mount.set()

        await asyncio.wait_for(mount_finished.wait(), timeout=10)
        for _ in range(100):
            rows = [
                row
                for row in reader.query(".library-conversation-reader-message")
                if getattr(row, "message_id", None) == "message-deferred"
            ]
            if rows and (
                len(focus_calls) == 2
                or screen._library_conversation_find_focus_intent is None
            ):
                break
            await asyncio.sleep(0)
        assert rows
        row = rows[0]
        if steal_focus:
            for _ in range(20):
                await asyncio.sleep(0)
        else:
            for _ in range(100):
                if row.has_focus:
                    break
                await asyncio.sleep(0)
        assert screen._library_conversation_find_focus_intent is None
        if steal_focus:
            assert screen.focused is replacement_focus
            assert focus_calls == ["message-deferred"]
        else:
            assert row.has_focus
            assert focus_calls == ["message-deferred", "message-deferred"]


@pytest.mark.asyncio
async def test_progressive_reader_paints_first_page_then_completes_find_off_loop() -> (
    None
):
    app = _build_test_app()
    _seed_conversations(app, _conversation_records()[:1])
    service = _ProgressiveConversationService()
    app.local_chat_conversation_service = service
    screen = _active_conversations_screen(app)
    host = LibraryHarness(app, screen=screen)
    event_loop_thread = threading.get_ident()

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        await _wait_for_library_shell(active, pilot)
        reader = await _wait_for_selector(active, pilot, "#library-conversation-reader")
        shell = active.query_one(
            "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
        )
        items = shell.items.query_one(
            "#library-conversations-canvas", LibraryConversationsCanvas
        )
        identities = (id(shell), id(items), id(reader))

        await asyncio.to_thread(service.second_started.wait, 10)
        try:
            await pilot.pause()
            assert len(active._library_conversation_reader_state.messages) == 20
            assert active._library_conversation_reader_state.complete is False
            assert len(reader.query(".library-conversation-reader-message")) == 20

            find = reader.query_one("#library-conversation-reader-find", Input)
            find.value = "needle"
            find.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert active._library_conversation_reader_state.find_complete is False
            assert active._library_conversation_reader_state.find_matches == ()
        finally:
            service.release_second.set()
        await active.workers.wait_for_complete()
        await pilot.pause()
        state = active._library_conversation_reader_state
        assert state.complete and state.find_complete
        assert len(state.messages) == state.message_total == 21
        assert state.find_matches[0].message_id == "message-20"
        assert "Find: 1 exact match." in str(
            reader.query_one("#library-conversation-reader-status", Static).renderable
        )
        assert next(
            row
            for row in reader.query(".library-conversation-reader-message")
            if getattr(row, "message_id", None) == "message-20"
        ).has_focus
        assert (id(shell), id(items), id(reader)) == identities
        assert [call["message_offset"] for call in service.calls] == [0, 20]
        assert [call["message_limit"] for call in service.calls] == [20, 20]
        assert all(thread_id != event_loop_thread for thread_id in service.thread_ids)


@pytest.mark.asyncio
async def test_late_previous_selection_cannot_overwrite_current_reader() -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records())
    service = _OutOfOrderConversationService()
    app.local_chat_conversation_service = service
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await asyncio.to_thread(service.first_started.wait, 10)
        try:
            first_id = str(service.calls[0]["conversation_id"])
            target_id = "chat-b" if first_id == "chat-a" else "chat-a"
            target_row = next(
                row
                for row in screen.query(".library-conversation-row")
                if getattr(row, "conversation_id", None) == target_id
            )

            target_row.press()
            await _wait_for_selector(screen, pilot, "#library-conversation-reader")
            for _ in range(100):
                state = screen._library_conversation_reader_state
                if state.loaded_id == target_id and state.complete:
                    break
                await pilot.pause()
            else:
                pytest.fail("Current conversation never settled.")
        finally:
            service.release_first.set()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        state = screen._library_conversation_reader_state
        assert state.selected_id == state.loaded_id == target_id
        assert state.messages[0].text == f"transcript for {target_id}"


@pytest.mark.parametrize("steal_focus", (False, True))
@pytest.mark.asyncio
async def test_long_continuation_find_reveals_only_current_focus_intent(
    steal_focus: bool,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records()[:1])
    service = _ContinuationConversationService()
    app.local_chat_conversation_service = service
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await asyncio.to_thread(service.continuation_started.wait, 10)
        try:
            reader = screen.query_one(
                "#library-conversation-reader", LibraryConversationReader
            )
            find = reader.query_one("#library-conversation-reader-find", Input)
            find.value = "needle"
            find.focus()
            await pilot.press("enter")
            await pilot.pause()
            assert "Searching complete transcript" in str(
                reader.query_one(
                    "#library-conversation-reader-status", Static
                ).renderable
            )
            replacement_focus = reader.query_one(
                "#library-conversation-reader-info", Button
            )
            if steal_focus:
                replacement_focus.focus()
                await pilot.pause()
        finally:
            service.release_continuation.set()
        await screen.workers.wait_for_complete()
        await pilot.pause()
        match_row = next(
            row
            for row in reader.query(".library-conversation-reader-message")
            if getattr(row, "message_id", None) == "message-long"
        )
        assert screen._library_conversation_reader_state.find_complete
        assert match_row.has_focus is (not steal_focus)
        if steal_focus:
            assert replacement_focus.has_focus


@pytest.mark.parametrize("phase", ("later-page", "continuation"))
@pytest.mark.asyncio
async def test_reader_request_loses_page_and_continuation_authority_on_route_leave(
    phase: str,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records()[:1])
    service = (
        _ProgressiveConversationService()
        if phase == "later-page"
        else _ContinuationConversationService()
    )
    app.local_chat_conversation_service = service
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        started = (
            service.second_started
            if isinstance(service, _ProgressiveConversationService)
            else service.continuation_started
        )
        await asyncio.to_thread(started.wait, 10)
        try:
            partial = screen._library_conversation_reader_state
            assert partial.loading and not partial.complete
            await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
            await pilot.pause()
        finally:
            if isinstance(service, _ProgressiveConversationService):
                service.release_second.set()
            else:
                service.release_continuation.set()
        await screen.workers.wait_for_complete()
        settled = screen._library_conversation_reader_state
        assert settled.generation > partial.generation
        assert settled.messages == partial.messages
        assert not settled.complete


@pytest.mark.parametrize("phase", ("later-page", "continuation"))
@pytest.mark.asyncio
async def test_reader_request_loses_authority_when_screen_unmounts(phase: str) -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records()[:1])
    service = (
        _ProgressiveConversationService()
        if phase == "later-page"
        else _ContinuationConversationService()
    )
    app.local_chat_conversation_service = service
    screen = _active_conversations_screen(app)
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        started = (
            service.second_started
            if isinstance(service, _ProgressiveConversationService)
            else service.continuation_started
        )
        await asyncio.to_thread(started.wait, 10)
        try:
            partial = screen._library_conversation_reader_state
            pop_task = asyncio.ensure_future(host.pop_screen())
            while screen._library_conversation_reader_mounted_authority:
                await asyncio.sleep(0)
        finally:
            if isinstance(service, _ProgressiveConversationService):
                service.release_second.set()
            else:
                service.release_continuation.set()
        await pop_task
        await screen.workers.wait_for_complete()
        settled = screen._library_conversation_reader_state
        assert settled.generation > partial.generation
        assert settled.messages == partial.messages
        assert not settled.complete


@pytest.mark.parametrize("phase", ("later-page", "continuation"))
@pytest.mark.parametrize("leave_path", ("navigation-context", "export"))
@pytest.mark.asyncio
async def test_reader_request_cannot_resurrect_after_bypass_leave_and_return(
    phase: str,
    leave_path: str,
) -> None:
    """Every admitted leave advances authority even when Conversations returns."""
    app = _build_test_app()
    _seed_conversations(app, _conversation_records()[:1])
    service = (
        _ProgressiveConversationService()
        if phase == "later-page"
        else _ContinuationConversationService()
    )
    app.local_chat_conversation_service = service
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        started = (
            service.second_started
            if isinstance(service, _ProgressiveConversationService)
            else service.continuation_started
        )
        release = (
            service.release_second
            if isinstance(service, _ProgressiveConversationService)
            else service.release_continuation
        )
        await asyncio.to_thread(started.wait, 10)
        partial = screen._library_conversation_reader_state
        try:
            if leave_path == "navigation-context":
                screen.apply_navigation_context({"mode": "notes"})
                for _ in range(100):
                    if screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES:
                        break
                    await pilot.pause()
                screen.apply_navigation_context({"mode": "conversations"})
                for _ in range(100):
                    if (
                        screen._library_selected_row_id
                        == LIBRARY_ROW_BROWSE_CONVERSATIONS
                    ):
                        break
                    await pilot.pause()
            else:
                await screen._open_library_export_canvas(
                    ExportScope(kind="conversations")
                )
                await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
            assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
            assert screen._library_conversation_reader_state.generation > (
                partial.generation
            )
        finally:
            release.set()
        try:
            await screen.workers.wait_for_complete()
        except WorkerCancelled:
            # Export return cancels its unrelated counts worker; the reader
            # fence below remains the authority under test.
            pass
        await pilot.pause()
        settled = screen._library_conversation_reader_state
        assert settled.generation > partial.generation
        assert (
            settled.messages == partial.messages
            or settled.loaded_generation == settled.generation
        )


@pytest.mark.asyncio
async def test_same_identity_version_refresh_fences_old_loaded_revision() -> None:
    app = _build_test_app()
    records = _conversation_records()[:1]
    _seed_conversations(app, records)
    service = _GatedVersionConversationService(5)
    screen = _active_conversations_screen(app)
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        await _wait_for_library_shell(active, pilot)
        await active.workers.wait_for_complete()
        app.local_chat_conversation_service = service
        active._library_conversation_reader_state = _loaded_reader_state()
        active._library_conversation_reader_loaded_metadata = records[0]
        active._library_conversation_reader_selected_metadata = records[0]
        active._library_conversation_page_records = (
            {
                **records[0],
                "version": 5,
                "title": "Alpha v5",
                "keywords": ["refreshed"],
            },
        )
        active._ensure_library_conversation_reader_selection()
        await asyncio.to_thread(service.started.wait, 10)
        try:
            action = active.query_one("#library-conversation-open-console", Button)
            state = active._library_conversation_reader_state
            assert state.selected_version == 5
            assert state.loaded_version == 4
            assert not state.loaded_actions_eligible and action.disabled
        finally:
            service.release.set()
        await active.workers.wait_for_complete()
        await pilot.pause()
        state = active._library_conversation_reader_state
        assert state.loaded_version == 5 and state.loaded_actions_eligible
        assert not action.disabled
        active.query_one("#library-conversation-reader-info", Button).press()
        await pilot.pause()
        info = str(
            active.query_one(
                "#library-conversation-reader-info-body", Static
            ).renderable
        )
        assert "Version: 5" in info
        assert "Keywords: refreshed" in info
        assert "Keywords: alpha, planning" not in info


@pytest.mark.parametrize("outcome", ("invalid", "unavailable"))
@pytest.mark.asyncio
async def test_mounted_failed_b_selection_never_relabels_retained_a_metadata(
    outcome: str,
) -> None:
    app = _build_test_app()
    records = _conversation_records()
    _seed_conversations(app, records)
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        await _wait_for_library_shell(active, pilot)
        await active.workers.wait_for_complete()
        service = _GatedFailureConversationService(outcome)
        app.local_chat_conversation_service = service
        active._library_conversation_reader_state = _loaded_reader_state()
        active._library_conversation_reader_loaded_metadata = records[0]
        active._library_conversation_reader_selected_metadata = records[0]
        active._selected_conversation_id = "chat-a"

        active._start_library_conversation_reader_selection("chat-b")
        await asyncio.to_thread(service.started.wait, 10)
        try:
            pending = active._library_conversation_reader_state
            assert pending.selected_id == "chat-b" and pending.loaded_id == "chat-a"
            assert active._library_conversation_reader_loaded_metadata["title"] == (
                "Alpha planning"
            )
        finally:
            service.release.set()
        await active.workers.wait_for_complete()
        await pilot.pause()

        state = active._library_conversation_reader_state
        assert state.selected_id == "chat-b" and state.loaded_id == "chat-a"
        assert state.error and not state.loaded_actions_eligible
        assert active._library_conversation_reader_loaded_metadata["title"] == (
            "Alpha planning"
        )
        status = str(
            active.query_one("#library-conversation-reader-status", Static).renderable
        )
        assert "Beta review (chat-b)" in status
        assert "Alpha planning (chat-a)" in status


@pytest.mark.asyncio
async def test_authoritative_refresh_marks_selected_conversation_deleted_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    records = _conversation_records()
    _seed_conversations(app, records)
    screen = _active_conversations_screen(app)
    screen._library_conversation_reader_state = _loaded_reader_state()
    screen._library_conversation_reader_loaded_metadata = records[0]
    screen._library_conversation_reader_selected_metadata = records[0]
    screen._selected_conversation_id = "chat-a"
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        await _wait_for_library_shell(active, pilot)
        await active.workers.wait_for_complete()
        active._library_conversation_reader_state = _loaded_reader_state()
        active._library_conversation_reader_loaded_metadata = records[0]
        active._library_conversation_reader_selected_metadata = records[0]
        active._selected_conversation_id = "chat-a"

        def remaining_page(**_kwargs):
            return {
                "items": [records[1]],
                "pagination": {
                    "limit": 20,
                    "offset": 0,
                    "total": 1,
                    "has_more": False,
                },
            }

        locator_calls: list[str] = []

        async def missing_exact(conversation_id: str, **_kwargs):
            locator_calls.append(conversation_id)
            return None

        monkeypatch.setattr(
            app.chat_conversation_scope_service,
            "list_conversations",
            remaining_page,
        )
        monkeypatch.setattr(
            app.chat_conversation_scope_service,
            "locate_conversation_page",
            missing_exact,
        )
        _query, generation = active._prepare_library_conversation_page_request("")
        await active._load_library_conversation_page(1, "", generation)
        await pilot.pause()

        state = active._library_conversation_reader_state
        assert state.unavailable and state.error == "Conversation deleted."
        assert locator_calls == ["chat-a"]
        assert state.loaded_id is None and state.selected_id == "chat-a"
        assert active._selected_conversation_id == ""
        rows = list(active.query(".library-conversation-row"))
        assert rows and not any(getattr(row, "selected", False) for row in rows)
        assert "Conversation deleted" in str(
            active.query_one("#library-conversation-reader-status", Static).renderable
        )


@pytest.mark.asyncio
async def test_page_drift_confirms_exact_identity_before_declaring_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    records = [
        {
            "id": f"chat-{index:02d}",
            "title": f"Chat {index:02d}",
            "version": 4,
            "message_count": 1,
        }
        for index in range(21)
    ]
    records[0] = _conversation_records()[0]
    _seed_conversations(app, records)
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen.workers.wait_for_complete()
        screen._library_conversation_reader_state = _loaded_reader_state()
        screen._library_conversation_reader_loaded_metadata = records[0]
        screen._library_conversation_reader_selected_metadata = records[0]
        screen._selected_conversation_id = "chat-a"
        screen._library_conversation_page_records = tuple(records[:20])
        screen._library_conversation_page = 1
        screen._library_conversation_page_loaded = True
        screen._library_conversation_query = ""

        shifted_page = records[1:21]

        async def reordered_page(**_kwargs):
            return {
                "items": shifted_page,
                "pagination": {
                    "limit": 20,
                    "offset": 0,
                    "total": 21,
                    "has_more": True,
                },
            }

        async def moved_exact(conversation_id: str, **_kwargs):
            assert conversation_id == "chat-a"
            return {
                "items": [records[0]],
                "pagination": {
                    "limit": 20,
                    "offset": 20,
                    "page": 2,
                    "total": 21,
                    "target_index": 20,
                    "has_more": False,
                },
            }

        monkeypatch.setattr(
            app.chat_conversation_scope_service,
            "list_conversations",
            reordered_page,
        )
        monkeypatch.setattr(
            app.chat_conversation_scope_service,
            "locate_conversation_page",
            moved_exact,
        )
        _query, generation = screen._prepare_library_conversation_page_request("")
        await screen._load_library_conversation_page(1, "", generation)
        await pilot.pause()

        state = screen._library_conversation_reader_state
        assert not state.unavailable and state.error is None
        assert state.selected_id == state.loaded_id == "chat-a"
        assert state.messages == _loaded_reader_state().messages
        assert screen._selected_conversation_id == "chat-a"
        rows = list(screen.query(".library-conversation-row"))
        assert rows and not any(getattr(row, "selected", False) for row in rows)


@pytest.mark.parametrize("locator_outcome", ("invalid", "error"))
@pytest.mark.asyncio
async def test_unconfirmed_page_absence_is_stale_not_deleted_or_reselected(
    locator_outcome: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    records = _conversation_records()
    _seed_conversations(app, records)
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen.workers.wait_for_complete()
        screen._library_conversation_reader_state = _loaded_reader_state()
        screen._library_conversation_reader_loaded_metadata = records[0]
        screen._library_conversation_reader_selected_metadata = records[0]
        screen._selected_conversation_id = "chat-a"
        screen._library_conversation_page_records = tuple(records)
        screen._library_conversation_page = 1
        screen._library_conversation_page_loaded = True
        screen._library_conversation_query = ""

        async def remaining_page(**_kwargs):
            return {
                "items": [records[1]],
                "pagination": {
                    "limit": 20,
                    "offset": 0,
                    "total": 1,
                    "has_more": False,
                },
            }

        async def unconfirmed_exact(_conversation_id: str, **_kwargs):
            if locator_outcome == "error":
                raise RuntimeError("locator failed")
            return {"not": "a locator envelope"}

        monkeypatch.setattr(
            app.chat_conversation_scope_service,
            "list_conversations",
            remaining_page,
        )
        monkeypatch.setattr(
            app.chat_conversation_scope_service,
            "locate_conversation_page",
            unconfirmed_exact,
        )
        _query, generation = screen._prepare_library_conversation_page_request("")
        await screen._load_library_conversation_page(1, "", generation)
        await pilot.pause()

        state = screen._library_conversation_reader_state
        assert not state.unavailable and state.error is None
        assert state.selected_id == state.loaded_id == "chat-a"
        assert screen._selected_conversation_id == "chat-a"
        assert screen._library_conversation_freshness == "stale"
        assert "confirm" in screen._library_conversation_stale_copy.casefold()
        rows = list(screen.query(".library-conversation-row"))
        assert rows and not any(getattr(row, "selected", False) for row in rows)


@pytest.mark.asyncio
async def test_missing_version_pending_selection_can_be_confirmed_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    records = _conversation_records()
    records[0] = {key: value for key, value in records[0].items() if key != "version"}
    _seed_conversations(app, records)
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen.workers.wait_for_complete()
        pending = ConversationReaderState(
            selected_id="chat-a",
            selected_version=None,
            generation=9,
            loading=True,
        )
        screen._library_conversation_reader_state = pending
        screen._selected_conversation_id = "chat-a"
        screen._library_conversation_page_records = tuple(records)
        screen._library_conversation_page = 1
        screen._library_conversation_page_loaded = True
        screen._library_conversation_query = ""

        async def remaining_page(**_kwargs):
            return {
                "items": [records[1]],
                "pagination": {
                    "limit": 20,
                    "offset": 0,
                    "total": 1,
                    "has_more": False,
                },
            }

        async def missing_exact(_conversation_id: str, **_kwargs):
            return None

        monkeypatch.setattr(
            app.chat_conversation_scope_service,
            "list_conversations",
            remaining_page,
        )
        monkeypatch.setattr(
            app.chat_conversation_scope_service,
            "locate_conversation_page",
            missing_exact,
        )
        _query, generation = screen._prepare_library_conversation_page_request("")
        await screen._load_library_conversation_page(1, "", generation)
        await pilot.pause()

        state = screen._library_conversation_reader_state
        assert state.unavailable and state.error == "Conversation deleted."
        assert state.selected_id == "chat-a" and state.selected_version is None
        assert screen._selected_conversation_id == ""
        rows = list(screen.query(".library-conversation-row"))
        assert rows and not any(getattr(row, "selected", False) for row in rows)


@pytest.mark.asyncio
async def test_conversations_global_f6_cycles_visible_destination_roles() -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records())
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        shell = screen.query_one(
            "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
        )
        rail = screen.query_one("#library-search-input", Input)
        items = screen.query_one("#library-conversations-filter", Input)
        work = screen.query_one("#library-conversation-reader-find", Input)
        await screen.workers.wait_for_complete()
        await pilot.pause()
        rail.focus()
        await pilot.pause()
        for expected in (items, work, rail):
            screen.action_focus_next_workbench_pane()
            await pilot.pause()
            assert expected.has_focus

        shell.library_grip.press()
        await pilot.pause()
        items.focus()
        await pilot.pause()
        for expected in (work, items):
            screen.action_focus_next_workbench_pane()
            await pilot.pause()
            assert expected.has_focus

        shell.items_grip.press()
        await pilot.pause()
        work.focus()
        await pilot.pause()
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        assert work.has_focus
        shell.library_grip.press()
        await pilot.pause()
        screen.action_focus_next_workbench_pane()
        await pilot.pause()
        assert rail.has_focus
        assert not any(
            str(entry.key if hasattr(entry, "key") else entry[0]).casefold() == "f6"
            for entry in screen.BINDINGS
        )


@pytest.mark.asyncio
async def test_conversations_slash_targets_only_visible_items_filter() -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records())
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        shell = screen.query_one(
            "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
        )
        filter_input = screen.query_one("#library-conversations-filter", Input)
        read = screen.query_one("#library-conversation-reader-read", Button)
        read.focus()
        await pilot.press("/")
        await pilot.pause()
        assert filter_input.has_focus

        filter_input.value = "typed"
        await pilot.press("/")
        await pilot.pause()
        assert filter_input.has_focus and filter_input.value.endswith("/")

        shell.items_grip.press()
        await pilot.pause()
        read.focus()
        await pilot.press("/")
        await pilot.pause()
        assert read.has_focus


@pytest.mark.asyncio
async def test_conversations_escape_moves_to_nearest_visible_prior_role() -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records())
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=(160, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        shell = screen.query_one(
            "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
        )
        rail = screen.query_one("#library-search-input", Input)
        items = screen.query_one("#library-conversations-filter", Input)
        work = screen.query_one("#library-conversation-reader-read", Button)
        work.focus()
        await pilot.press("escape")
        await pilot.pause()
        assert items.has_focus
        assert screen.check_action("library_list_focus_rail", ()) is True
        await pilot.press("escape")
        await pilot.pause()
        assert rail.has_focus

        shell.library_grip.press()
        await pilot.pause()
        work.focus()
        await pilot.press("escape")
        await pilot.pause()
        assert items.has_focus
        assert screen.check_action("library_list_focus_rail", ()) is False

        work.focus()
        shell.items_grip.press()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert work.has_focus
        assert screen.check_action("library_list_focus_rail", ()) is False

        shell.library_grip.press()
        await pilot.pause()
        work.focus()
        shortcuts = screen._library_route_shortcuts_for_current_state()
        assert ("esc", "focus Library") in shortcuts
        await pilot.press("escape")
        await pilot.pause()
        assert rail.has_focus
        assert ("esc", "focus Library") not in (
            screen._library_route_shortcuts_for_current_state()
        )


@pytest.mark.parametrize("size", ((160, 50), (120, 35), (100, 30), (80, 24)))
@pytest.mark.asyncio
async def test_conversations_geometry_contains_protected_work_and_restore_grips(
    size: tuple[int, int],
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _conversation_records())
    host = LibraryHarness(app, screen=_active_conversations_screen(app))

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        shell = screen.query_one(
            "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
        )
        await pilot.pause()
        visible = screen._compositor.visible_widgets
        assert shell.work in visible and shell.region.contains_region(shell.work.region)
        for grip in (shell.library_grip, shell.items_grip):
            assert grip in visible
            assert grip.region.width == 5
            assert shell.region.contains_region(grip.region)
            assert grip.can_focus
        for pane, open_ in (
            (shell.library, shell.effective_layout.library_open),
            (shell.items, shell.effective_layout.items_open),
        ):
            if open_:
                assert pane in visible and shell.region.contains_region(pane.region)
