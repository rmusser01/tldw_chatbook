"""Focused policy tests for the Console review/selection owner."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
import threading
from typing import Any

import pytest

from tldw_chatbook.UI.Console_Modules import review_selection as review_module
from tldw_chatbook.UI.Console_Modules.review_selection import (
    ConsoleReviewSelectionController,
)
from tldw_chatbook.Widgets.Console.console_selection import SELECTION_QUOTE_CAP


class _EventLoopOnlyMap(dict[str, tuple[str, ...]]):
    """Reject mutation from any thread except the creating event-loop thread."""

    def __init__(self, owner_thread: int) -> None:
        super().__init__()
        self.owner_thread = owner_thread

    def __setitem__(self, key: str, value: tuple[str, ...]) -> None:
        assert threading.get_ident() == self.owner_thread
        super().__setitem__(key, value)


async def _no_comment(_action: str, _quote: str) -> str | None:
    return None


async def _no_dispatch(_text: str) -> None:
    return None


def _controller(
    *,
    store_accessor: Callable[[], Any] = lambda: None,
    agent_conversation_id_accessor: Callable[[], str | None] = lambda: None,
    change_review_provider_accessor: Callable[[str], Any | None] = lambda _id: None,
    run_active_accessor: Callable[[], bool] = lambda: False,
    run_active_for_root: Callable[[str], bool] = lambda _root: False,
    workspace_roots_accessor: Callable[[], tuple[str, ...] | None] = lambda: None,
    agent_runs_db_accessor: Callable[[], Any | None] = lambda: None,
    capture_policy_bindings_accessor: Callable[
        [str, str], Any | None
    ] = lambda _session, _conversation: None,
    native_messages_accessor: Callable[[], list[Any]] = lambda: [],
    run_worker: Callable[..., Any] = lambda *_args, **_kwargs: None,
    show_feedback_comment: Callable[[str, str], Awaitable[str | None]] = _no_comment,
    dispatch_prompt: Callable[[str], Awaitable[Any]] = _no_dispatch,
    notify: Callable[..., None] = lambda *_args, **_kwargs: None,
) -> ConsoleReviewSelectionController:
    return ConsoleReviewSelectionController(
        store_accessor=store_accessor,
        agent_conversation_id_accessor=agent_conversation_id_accessor,
        change_review_provider_accessor=change_review_provider_accessor,
        run_active_accessor=run_active_accessor,
        run_active_for_root=run_active_for_root,
        workspace_roots_accessor=workspace_roots_accessor,
        agent_runs_db_accessor=agent_runs_db_accessor,
        capture_policy_bindings_accessor=capture_policy_bindings_accessor,
        native_messages_accessor=native_messages_accessor,
        run_worker=run_worker,
        show_feedback_comment=show_feedback_comment,
        dispatch_prompt=dispatch_prompt,
        marshal_to_ui=lambda *_args: None,
        present_trajectory=lambda *_args: None,
        notify=notify,
    )


def test_change_review_provider_uses_live_run_probes() -> None:
    provider = SimpleNamespace()
    active = False
    roots: set[str] = set()
    controller = _controller(
        agent_conversation_id_accessor=lambda: "conversation-1",
        change_review_provider_accessor=lambda _conversation_id: provider,
        run_active_accessor=lambda: active,
        run_active_for_root=lambda root: root in roots,
    )

    assert controller._console_change_review_provider() is provider
    assert provider.run_active() is False
    active = True
    roots.add("/workspace")
    assert provider.run_active() is True
    assert provider.run_active_for_root("/workspace") is True


@pytest.mark.parametrize(
    ("action", "quote", "anchor_message_id"),
    [
        ("unsupported", "selected", "message-1"),
        ("comment", 42, "message-1"),
        ("comment", "x" * (SELECTION_QUOTE_CAP + 1), "message-1"),
        ("comment", "selected", "m" * 256),
    ],
)
def test_feedback_request_rejects_invalid_boundary_values(
    action: str, quote: object, anchor_message_id: str
) -> None:
    scheduled: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    notifications: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    controller = _controller(
        run_worker=lambda *args, **kwargs: scheduled.append((args, kwargs)),
        notify=lambda *args, **kwargs: notifications.append((args, kwargs)),
    )

    controller.request_selection_feedback(action, quote, anchor_message_id)  # type: ignore[arg-type]

    assert scheduled == []
    assert controller.selection_feedback_inflight is False
    assert notifications == [
        (("Selection feedback is invalid or too large.",), {"severity": "warning"})
    ]


@pytest.mark.asyncio
async def test_annotation_loader_discards_stale_conversation_result() -> None:
    database = SimpleNamespace(
        get_transcript_annotations=lambda _conversation_id: [
            {"message_id": "persisted-1", "comment": "stale"}
        ]
    )
    message = SimpleNamespace(id="native-1", persisted_message_id="persisted-1")
    controller = _controller(native_messages_accessor=lambda: [message])
    controller.annotation_loaded_conversation = "conversation-2"
    controller.annotation_previews = {"native-current": ("keep",)}

    await controller._load_console_annotation_previews(
        database, object(), "conversation-1"
    )

    assert controller.annotation_previews == {"native-current": ("keep",)}


@pytest.mark.asyncio
async def test_annotation_loader_rekeys_on_event_loop_after_worker_read() -> None:
    event_loop_thread = threading.get_ident()
    read_threads: list[int] = []

    def read(_conversation_id: str) -> list[dict[str, str]]:
        read_threads.append(threading.get_ident())
        return [{"message_id": "persisted-1", "comment": "note"}]

    message = SimpleNamespace(id="native-1", persisted_message_id="persisted-1")
    controller = _controller(native_messages_accessor=lambda: [message])
    controller.annotation_loaded_conversation = "conversation-1"

    await controller._load_console_annotation_previews(
        SimpleNamespace(get_transcript_annotations=read),
        object(),
        "conversation-1",
    )

    assert read_threads and read_threads[0] != event_loop_thread
    assert controller.annotation_previews == {"native-1": ("note",)}


@pytest.mark.asyncio
async def test_feedback_audit_finishes_before_dispatch() -> None:
    order: list[str] = []

    class _Store:
        active_session_id = "session-1"

        def record_feedback_event(self, *_args: Any, **_kwargs: Any) -> None:
            order.append("audit")

        def record_feedback_annotation(self, *_args: Any, **_kwargs: Any) -> str:
            return "annotation-1"

    async def show(_action: str, _quote: str) -> str:
        return "comment"

    async def dispatch(_text: str) -> None:
        order.append("dispatch")

    controller = _controller(
        store_accessor=_Store,
        show_feedback_comment=show,
        dispatch_prompt=dispatch,
    )
    controller.selection_feedback_inflight = True

    await controller._console_selection_feedback_flow(
        "comment", "selected", "message-1"
    )

    assert order == ["audit", "dispatch"]


@pytest.mark.asyncio
async def test_feedback_guard_releases_after_dispatch_error() -> None:
    async def show(_action: str, _quote: str) -> str:
        return ""

    async def dispatch(_text: str) -> None:
        raise RuntimeError("dispatch failed")

    controller = _controller(
        show_feedback_comment=show,
        dispatch_prompt=dispatch,
    )
    controller.selection_feedback_inflight = True

    with pytest.raises(RuntimeError, match="dispatch failed"):
        await controller._console_selection_feedback_flow("lgm", "selected")

    assert controller.selection_feedback_inflight is False


@pytest.mark.asyncio
async def test_feedback_preview_map_changes_only_on_event_loop() -> None:
    event_loop_thread = threading.get_ident()
    write_threads: list[int] = []

    class _Store:
        active_session_id = "session-1"

        def record_feedback_event(self, *_args: Any, **_kwargs: Any) -> None:
            write_threads.append(threading.get_ident())

        def record_feedback_annotation(self, *_args: Any, **_kwargs: Any) -> str:
            write_threads.append(threading.get_ident())
            return "annotation-1"

    async def show(_action: str, _quote: str) -> str:
        return "comment"

    controller = _controller(
        store_accessor=_Store,
        show_feedback_comment=show,
    )
    controller.annotation_previews = _EventLoopOnlyMap(event_loop_thread)

    await controller._console_selection_feedback_flow(
        "comment", "selected", "message-1"
    )

    assert write_threads and all(
        thread != event_loop_thread for thread in write_threads
    )
    assert controller.annotation_previews == {"message-1": ("comment",)}


@pytest.mark.asyncio
async def test_note_write_failure_never_logs_selected_text(monkeypatch) -> None:
    selected = "SECRET transcript selection"
    logged: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fail_write(_title: str, _content: str) -> None:
        raise RuntimeError("write failed")

    store = SimpleNamespace(
        active_session_id="session-1",
        _sessions={"session-1": SimpleNamespace(title="Console")},
        persistence=SimpleNamespace(db=SimpleNamespace(add_note=fail_write)),
    )
    monkeypatch.setattr(
        review_module.logger,
        "warning",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )
    controller = _controller(store_accessor=lambda: store)

    await controller._create_console_selection_note(selected)

    rendered = repr(logged)
    assert selected not in rendered
    assert "SECRET" not in rendered
