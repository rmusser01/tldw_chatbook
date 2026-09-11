"""Durable, content-free Console attention projection contracts."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage
from rich_pixels import Pixels
from textual.app import ComposeResult

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    GenerationVariantMeta,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Console_Modules.image import ConsoleImageController
from tldw_chatbook.UI.Console_Modules.video import ConsoleVideoController
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Widgets.Console.console_assistant_turn import (
    ConsoleAssistantTurnWidget,
)
from tldw_chatbook.Widgets.Console.console_generation_card import (
    ConsoleGenerationCard,
    ConsoleGenerationCardSpec,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_video_card import (
    ConsoleVideoCard,
    ConsoleVideoCardSpec,
)


RECEIPT_A = "11111111-1111-4111-8111-111111111111"
RECEIPT_B = "22222222-2222-4222-8222-222222222222"
RECEIPT_C = "33333333-3333-4333-8333-333333333333"


class _TranscriptHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")


def _mounted_media_messages():
    ordinary = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="ordinary",
        id="ordinary-result",
        metadata=MessageMetadata(terminal_receipt_id=RECEIPT_A),
    )
    generation_meta = GenerationVariantMeta(
        prompt="private image prompt",
        negative_prompt="",
        backend="test",
        model=None,
        seed=7,
        style=None,
        params={},
    )
    image = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[image] result",
        id="image-result",
        metadata=MessageMetadata(terminal_receipt_id=RECEIPT_B),
        generation_metadata=(generation_meta,),
    )
    video_meta = VideoGenerationMetadata(
        name="video-result",
        prompt="private video prompt",
        backend="test",
        terminal_receipt_id=RECEIPT_C,
    )
    video = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[video] result",
        id="video-result",
        video_metadata=video_meta,
    )
    pixels = Pixels.from_image(PILImage.new("RGB", (4, 4), "blue"))
    image_spec = ConsoleGenerationCardSpec(
        message_id=image.id,
        browsed_index=0,
        variant_count=1,
        meta=generation_meta,
        mode="pixels",
        pixels=pixels,
    )
    video_spec = ConsoleVideoCardSpec(
        message_id=video.id,
        meta=video_meta,
        status="expired",
    )
    return ordinary, image, video, image_spec, video_spec


class _Marks:
    def __init__(self, pairs: tuple[tuple[str, str], ...] = ()) -> None:
        self.pairs = list(pairs)
        self.acknowledged: list[tuple[str, str]] = []
        self.read_error: Exception | None = None

    def list_console_unseen_marks(self) -> tuple[tuple[str, str], ...]:
        if self.read_error is not None:
            raise self.read_error
        return tuple(self.pairs)

    def acknowledge_console_unseen(
        self, conversation_id: str, receipt_id: str
    ) -> bool:
        pair = (conversation_id, receipt_id)
        if pair not in self.pairs:
            return False
        self.pairs.remove(pair)
        self.acknowledged.append(pair)
        return True


@dataclass
class _DecisionController:
    hidden_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self._approval_state_lock = __import__("threading").RLock()
        self._announced_pending_decision_ids = set(self.hidden_ids)
        self.prompt_queue_coordinator = None
        self.fleet_wake = None
        self.app: Any = None


class _App:
    def __init__(self, marks: _Marks) -> None:
        self.conversation_local_marks_service = marks
        self.console_attention_updates: list[bool] = []
        self.notifications: list[tuple[str, str]] = []
        self.notify_failures = 0
        self.chachanotes_db: Any | None = None

    def set_console_attention_projection(self, value: bool) -> None:
        self.console_attention_updates.append(value)

    def notify(self, message: str, *, severity: str = "information") -> None:
        if self.notify_failures:
            self.notify_failures -= 1
            raise RuntimeError("notification renderer unavailable")
        self.notifications.append((message, severity))


def test_runtime_attention_is_receipt_or_hidden_decision_and_exact_ack() -> None:
    marks = _Marks((("conv-a", RECEIPT_A), ("conv-b", RECEIPT_B)))
    app = _App(marks)
    runtime = ConsoleRuntime(app)
    controller = _DecisionController(("opaque-decision",))
    runtime.set_chat_controller(controller)

    assert runtime.recompute_console_attention() is True
    assert runtime.console_needs_attention is True

    # Opening/attaching a Console projection is not receipt acknowledgement.
    view = SimpleNamespace(console_view_hooks=lambda: {})
    generation = runtime.attach_view(view)
    assert marks.pairs == [("conv-a", RECEIPT_A), ("conv-b", RECEIPT_B)]

    # One exact row render clears one receipt and leaves both other sources.
    assert runtime.acknowledge_rendered_terminal_receipts(
        (("conv-a", RECEIPT_A),),
        view=view,
        attachment_generation=generation,
    ) == (RECEIPT_A,)
    assert marks.pairs == [("conv-b", RECEIPT_B)]
    assert runtime.console_needs_attention is True

    # Resolving the decision does not clear the remaining durable receipt.
    with controller._approval_state_lock:
        controller._announced_pending_decision_ids.clear()
    assert runtime.recompute_console_attention() is True
    assert marks.pairs == [("conv-b", RECEIPT_B)]

    assert runtime.acknowledge_rendered_terminal_receipts(
        (("conv-b", RECEIPT_B),),
        view=view,
        attachment_generation=generation,
    ) == (RECEIPT_B,)
    assert runtime.console_needs_attention is False
    assert app.console_attention_updates[-1] is False


@pytest.mark.parametrize("producer", ("pending", "ack", "detach", "attach"))
@pytest.mark.parametrize("ui_action", ("ack", "detach"))
def test_worker_attention_handoff_does_not_block_ui_owner_fences(
    producer, ui_action
) -> None:
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    marks = _Marks((("conv-a", RECEIPT_A),))
    app = _App(marks)
    runtime = ConsoleRuntime(app)
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(store=store, provider_gateway=SimpleNamespace())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    ready = threading.Event()
    release = threading.Event()
    ui_done = threading.Event()
    callbacks = []
    errors = []
    ui_thread_ids = []
    projection_thread_ids = []

    def enqueue(callback):
        callbacks.append(callback)
        ready.set()
        return True

    def blocking_handoff(callback):
        enqueue(callback)
        if not release.wait(3):
            errors.append("synchronous UI handoff timed out")

    def publish(value):
        projection_thread_ids.append(threading.get_ident())
        app.console_attention_updates.append(value)

    app.call_later = enqueue
    app.call_from_thread = blocking_handoff
    app.set_console_attention_projection = publish

    def produce():
        try:
            if producer == "pending":
                controller.add_pending_round(session.id, "round-a")
            elif producer == "ack":
                runtime.acknowledge_rendered_terminal_receipts((("conv-a", RECEIPT_A),))
            elif producer == "detach":
                runtime.detach_view(None)
            else:

                def hooks():
                    runtime.recompute_console_attention(force_projection=True)
                    return {}

                view = SimpleNamespace(console_view_hooks=hooks)
                runtime.attach_view(view)
        except BaseException as exc:
            errors.append(exc)

    async def apply_ui_action():
        ui_thread_ids.append(threading.get_ident())
        if ui_action == "ack":
            runtime.acknowledge_rendered_terminal_receipts(
                (("conv-a", RECEIPT_A),),
                view=runtime.view,
                attachment_generation=runtime._attached_generation,
            )
        else:
            marks.pairs.clear()
            runtime.detach_view(None)
        runtime.recompute_console_attention(force_projection=True)
        # The old worker projection arrives after the newer owner operation.
        for callback in tuple(callbacks):
            callback()

    def run_ui():
        try:
            asyncio.run(apply_ui_action())
        except BaseException as exc:
            errors.append(exc)
        finally:
            ui_done.set()
            release.set()

    worker = threading.Thread(target=produce)
    ui = threading.Thread(target=run_ui)
    worker.start()
    completed_without_watchdog = False
    try:
        assert ready.wait(2)
        ui.start()
        completed_without_watchdog = ui_done.wait(1)
    finally:
        release.set()
        worker.join(3)
        if ui.ident is not None:
            ui.join(3)
    assert not worker.is_alive() and not ui.is_alive()
    assert completed_without_watchdog, (
        "UI owner operation waited on the worker's handoff lock"
    )
    assert errors == []
    assert runtime.console_needs_attention is False
    assert app.console_attention_updates and set(app.console_attention_updates) == {
        False
    }
    assert set(projection_thread_ids) == set(ui_thread_ids)


def test_attention_scheduler_unavailable_keeps_state_without_blocking_fallback():
    app = _App(_Marks((("conv-a", RECEIPT_A),)))
    app.call_later = lambda callback: False
    app.call_from_thread = MagicMock(side_effect=AssertionError("blocking fallback"))
    runtime = ConsoleRuntime(app)

    assert runtime.recompute_console_attention() is True
    assert runtime.console_needs_attention is True
    assert app.console_attention_updates == []
    app.call_from_thread.assert_not_called()


@pytest.mark.parametrize("desired", (False, True))
def test_same_value_recomputes_keep_one_queued_attention_publication(desired):
    marks = _Marks(() if desired else (("conv-a", RECEIPT_A),))
    app = _App(marks)
    callbacks = []
    app.call_later = lambda callback: callbacks.append(callback) or True
    runtime = ConsoleRuntime(app)
    assert runtime.recompute_console_attention() is not desired
    callbacks.pop()()
    app.console_attention_updates.clear()

    marks.pairs = [("conv-a", RECEIPT_A)] if desired else []
    assert runtime.recompute_console_attention() is desired
    assert runtime.recompute_console_attention() is desired
    assert runtime.recompute_console_attention() is desired
    assert len(callbacks) == 1
    callbacks.pop()()

    assert app.console_attention_updates == [desired]
    assert runtime.console_needs_attention is desired


@pytest.mark.parametrize("first", (False, True))
def test_queued_attention_cannot_overwrite_newer_opposite_value_or_ack(first):
    marks = _Marks((("conv-a", RECEIPT_A),) if first else ())
    app = _App(marks)
    callbacks = []
    app.call_later = lambda callback: callbacks.append(callback) or True
    runtime = ConsoleRuntime(app)
    assert runtime.recompute_console_attention() is first
    if first:
        assert runtime.acknowledge_rendered_terminal_receipts(
            (("conv-a", RECEIPT_A),)
        ) == (RECEIPT_A,)
    else:
        marks.pairs.append(("conv-a", RECEIPT_A))
        assert runtime.recompute_console_attention() is True
    assert runtime.recompute_console_attention() is not first
    assert len(callbacks) == 2
    for callback in callbacks:
        callback()

    assert app.console_attention_updates == [not first]
    assert runtime.console_needs_attention is not first


def test_terminal_notice_is_sanitized_severity_aware_and_retries_after_failure() -> None:
    marks = _Marks((("conv-a", RECEIPT_A), ("conv-b", RECEIPT_B)))
    app = _App(marks)
    app.chachanotes_db = SimpleNamespace(
        get_messages_for_conversation=lambda conversation_id, **_kwargs: [
            {
                "metadata_json": MessageMetadata(
                    terminal_receipt_id=(
                        RECEIPT_A if conversation_id == "conv-a" else RECEIPT_B
                    )
                ).to_json(),
                "assistant_generation_state": (
                    "failed" if conversation_id == "conv-a" else "complete"
                ),
            }
        ]
    )
    app.notify_failures = 1
    runtime = ConsoleRuntime(app)

    # A failed notification does not mark the exact receipt as notified.
    assert runtime.recompute_console_attention() is True
    assert len(app.notifications) == 1
    assert app.notifications[0][1] == "information"

    # The failed receipt is retried; the already-notified receipt is not.
    assert runtime.recompute_console_attention() is True
    assert sorted(severity for _message, severity in app.notifications) == [
        "error",
        "information",
    ]
    assert runtime.recompute_console_attention() is True
    assert len(app.notifications) == 2

    rendered = repr(app.notifications)
    for secret in (
        RECEIPT_A,
        RECEIPT_B,
        "conv-a",
        "conv-b",
        "provider secret",
        "/private/path",
    ):
        assert secret not in rendered
    assert "Return to Console" in rendered


def test_mark_query_failure_preserves_last_known_attention_and_notice_dedupe() -> None:
    marks = _Marks((("conv-a", RECEIPT_A),))
    app = _App(marks)
    app.chachanotes_db = SimpleNamespace(
        get_messages_for_conversation=lambda *_args, **_kwargs: [
            {
                "metadata_json": MessageMetadata(
                    terminal_receipt_id=RECEIPT_A
                ).to_json(),
                "assistant_generation_state": "complete",
            }
        ]
    )
    runtime = ConsoleRuntime(app)

    assert runtime.recompute_console_attention() is True
    assert len(app.notifications) == 1

    marks.read_error = RuntimeError("database unavailable")
    assert runtime.recompute_console_attention() is True
    assert runtime.console_needs_attention is True
    assert len(app.notifications) == 1

    marks.read_error = None
    assert runtime.recompute_console_attention() is True
    assert len(app.notifications) == 1


def test_unknown_terminal_outcome_defers_notice_and_dedupe_until_authoritative() -> None:
    marks = _Marks((("conv-a", RECEIPT_A),))
    rows: list[dict[str, Any]] = []
    app = _App(marks)

    def read_rows(_conversation_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        offset = kwargs["offset"]
        limit = kwargs["limit"]
        return rows[offset : offset + limit]

    app.chachanotes_db = SimpleNamespace(get_messages_for_conversation=read_rows)
    runtime = ConsoleRuntime(app)

    assert runtime.recompute_console_attention() is True
    assert app.notifications == []
    assert RECEIPT_A not in runtime._notified_terminal_receipts

    rows.extend(
        {
            "metadata_json": None,
            "assistant_generation_state": "complete",
        }
        for _index in range(1001)
    )
    rows.append(
        {
            "metadata_json": MessageMetadata(
                terminal_receipt_id=RECEIPT_A
            ).to_json(),
            "assistant_generation_state": "failed",
        }
    )
    assert runtime.recompute_console_attention() is True
    assert len(app.notifications) == 1
    assert app.notifications[0][1] == "error"


def test_stale_mark_read_cannot_publish_false_after_newer_recompute() -> None:
    class _BarrierMarks(_Marks):
        def __init__(self) -> None:
            super().__init__()
            self.first_read_captured = threading.Event()
            self.release_first_read = threading.Event()
            self.reads = 0

        def list_console_unseen_marks(self) -> tuple[tuple[str, str], ...]:
            self.reads += 1
            captured = tuple(self.pairs)
            if self.reads == 1:
                self.first_read_captured.set()
                assert self.release_first_read.wait(timeout=5)
            return captured

    marks = _BarrierMarks()
    app = _App(marks)
    app.chachanotes_db = SimpleNamespace(
        get_messages_for_conversation=lambda *_args, **_kwargs: [
            {
                "metadata_json": MessageMetadata(
                    terminal_receipt_id=RECEIPT_A
                ).to_json(),
                "assistant_generation_state": "complete",
            }
        ]
    )
    runtime = ConsoleRuntime(app)

    revisions_reserved = threading.Event()
    original_reserve = runtime._reserve_attention_revision
    reservations = 0

    def reserve() -> int:
        nonlocal reservations
        revision = original_reserve()
        reservations += 1
        if reservations == 2:
            revisions_reserved.set()
        return revision

    runtime._reserve_attention_revision = reserve  # type: ignore[method-assign]

    first = threading.Thread(target=runtime.recompute_console_attention)
    first.start()
    assert marks.first_read_captured.wait(timeout=5)
    marks.pairs.append(("conv-a", RECEIPT_A))
    second = threading.Thread(target=runtime.recompute_console_attention)
    second.start()
    assert revisions_reserved.wait(timeout=5)
    marks.release_first_read.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert runtime.console_needs_attention is True
    assert False not in app.console_attention_updates


def test_exact_ack_fences_in_flight_notice_and_dedupe_recording() -> None:
    marks = _Marks((("conv-a", RECEIPT_A),))
    app = _App(marks)
    app.chachanotes_db = SimpleNamespace(
        get_messages_for_conversation=lambda *_args, **_kwargs: [
            {
                "metadata_json": MessageMetadata(
                    terminal_receipt_id=RECEIPT_A
                ).to_json(),
                "assistant_generation_state": "complete",
            }
        ]
    )
    notice_started = threading.Event()
    release_notice = threading.Event()
    events: list[str] = []

    def notify(_message: str, *, severity: str = "information") -> None:
        del severity
        notice_started.set()
        assert release_notice.wait(timeout=5)
        events.append("notice")

    app.notify = notify
    runtime = ConsoleRuntime(app)
    recompute = threading.Thread(target=runtime.recompute_console_attention)
    recompute.start()
    assert notice_started.wait(timeout=5)

    acknowledged: list[tuple[str, ...]] = []

    def ack() -> None:
        acknowledged.append(
            runtime.acknowledge_rendered_terminal_receipts(
                (("conv-a", RECEIPT_A),)
            )
        )
        events.append("ack")

    acknowledgement = threading.Thread(target=ack)
    acknowledgement.start()
    release_notice.set()
    recompute.join(timeout=5)
    acknowledgement.join(timeout=5)

    assert acknowledged == [(RECEIPT_A,)]
    assert events == ["notice", "ack"]
    assert marks.pairs == []
    assert RECEIPT_A not in runtime._notified_terminal_receipts
    assert runtime.console_needs_attention is False


def test_overlapping_recomputes_record_one_successful_terminal_notice() -> None:
    marks = _Marks((("conv-a", RECEIPT_A),))
    app = _App(marks)
    app.chachanotes_db = SimpleNamespace(
        get_messages_for_conversation=lambda *_args, **_kwargs: [
            {
                "metadata_json": MessageMetadata(
                    terminal_receipt_id=RECEIPT_A
                ).to_json(),
                "assistant_generation_state": "complete",
            }
        ]
    )
    notice_started = threading.Event()
    release_notice = threading.Event()

    def notify(message: str, *, severity: str = "information") -> None:
        app.notifications.append((message, severity))
        notice_started.set()
        assert release_notice.wait(timeout=5)

    app.notify = notify
    runtime = ConsoleRuntime(app)
    second_reserved = threading.Event()
    real_reserve = runtime._reserve_attention_revision
    reservations = 0

    def reserve() -> int:
        nonlocal reservations
        revision = real_reserve()
        reservations += 1
        if reservations == 2:
            second_reserved.set()
        return revision

    runtime._reserve_attention_revision = reserve  # type: ignore[method-assign]
    first = threading.Thread(target=runtime.recompute_console_attention)
    second = threading.Thread(target=runtime.recompute_console_attention)
    first.start()
    assert notice_started.wait(timeout=5)
    second.start()
    assert second_reserved.wait(timeout=5)
    release_notice.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(app.notifications) == 1
    assert RECEIPT_A in runtime._notified_terminal_receipts


@pytest.mark.parametrize(
    ("message_kind", "expected_state", "expected_severity"),
    [
        ("ordinary_complete", "complete", "information"),
        ("ordinary_failed", "failed", "error"),
        ("image", "complete", "information"),
        ("video", "complete", "information"),
    ],
)
def test_real_terminal_receipt_reopens_with_authoritative_notice_severity(
    tmp_path,
    message_kind: str,
    expected_state: str,
    expected_severity: str,
) -> None:
    database_path = tmp_path / f"{message_kind}.sqlite"
    database = CharactersRAGDB(database_path, client_id="attention-first")
    store = ConsoleChatStore(persistence=ChatPersistenceService(database))
    session = store.create_session(title="Attention persistence")
    if message_kind.startswith("ordinary"):
        message = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        store.append_stream_chunk(message.id, "terminal result")
        message = (
            store.mark_message_failed(message.id)
            if message_kind.endswith("failed")
            else store.mark_message_complete(message.id)
        )
    elif message_kind == "image":
        meta = GenerationVariantMeta(
            prompt="private prompt",
            negative_prompt="",
            backend="test",
            model=None,
            seed=1,
            style=None,
            params={},
        )
        message = store.append_generation_message(
            session.id,
            content="[image] result",
            variants=[(b"png", "image/png", meta)],
            persist=True,
        )
    else:
        message = store.append_video_message(
            session.id,
            video_metadata=VideoGenerationMetadata(
                name="result",
                prompt="private prompt",
                backend="test",
            ),
            persist=True,
        )
    receipt_id = (
        message.video_metadata.terminal_receipt_id
        if message.video_metadata is not None
        else message.metadata.terminal_receipt_id
        if message.metadata is not None
        else ""
    )
    conversation_id = session.persisted_conversation_id
    assert receipt_id and conversation_id is not None
    database.close_connection()

    reopened = CharactersRAGDB(database_path, client_id="attention-second")
    durable = reopened.get_message_by_id(message.persisted_message_id)
    assert durable is not None
    assert durable["assistant_generation_state"] == expected_state
    app = _App(ConversationLocalMarksService(reopened))  # type: ignore[arg-type]
    app.chachanotes_db = reopened
    runtime = ConsoleRuntime(app)

    assert runtime.recompute_console_attention() is True
    assert len(app.notifications) == 1
    assert app.notifications[0][1] == expected_severity


@pytest.mark.parametrize("media", ["image", "video"])
def test_direct_media_commit_recomputes_attention_once_after_atomic_persistence(
    media: str,
) -> None:
    events: list[str] = []
    runtime = SimpleNamespace(
        recompute_console_attention=lambda: events.append("recompute")
    )
    app = SimpleNamespace(console_runtime=runtime)

    class _Store:
        def append_generation_message(self, *_args: Any, **_kwargs: Any) -> object:
            events.append("image-commit")
            return object()

        def append_video_message(self, *_args: Any, **_kwargs: Any) -> object:
            events.append("video-commit")
            return object()

    store = _Store()
    if media == "image":
        controller = ConsoleImageController.__new__(ConsoleImageController)
        controller.app_instance = app
        controller._append_durable_generation_message(
            store,
            "session-background",
            content="[image]",
            variants=(object(),),
        )
    else:
        controller = ConsoleVideoController.__new__(ConsoleVideoController)
        controller.app_instance = app
        controller._ensure_console_chat_store_fn = lambda: store
        controller._persist_generated_video_tuple(
            (SimpleNamespace(), SimpleNamespace()),
            session_id="session-background",
            message_id="video-message",
        )

    assert events == [f"{media}-commit", "recompute"]


@pytest.mark.parametrize("media", ["image", "video"])
def test_failed_direct_media_persistence_does_not_recompute_attention(media: str) -> None:
    runtime = MagicMock()
    app = SimpleNamespace(console_runtime=runtime)

    class _Store:
        def append_generation_message(self, *_args: Any, **_kwargs: Any) -> object:
            raise RuntimeError("atomic image persistence failed")

        def append_video_message(self, *_args: Any, **_kwargs: Any) -> object:
            raise RuntimeError("atomic video persistence failed")

    store = _Store()
    if media == "image":
        controller = ConsoleImageController.__new__(ConsoleImageController)
        controller.app_instance = app

        def invoke() -> object:
            return controller._append_durable_generation_message(
                store,
                "session-background",
                content="[image]",
                variants=(object(),),
            )

    else:
        controller = ConsoleVideoController.__new__(ConsoleVideoController)
        controller.app_instance = app
        controller._ensure_console_chat_store_fn = lambda: store

        def invoke() -> None:
            controller._persist_generated_video_tuple(
                (SimpleNamespace(), SimpleNamespace()),
                session_id="session-background",
                message_id="video-message",
            )

    with pytest.raises(RuntimeError, match="atomic .* persistence failed"):
        invoke()
    runtime.recompute_console_attention.assert_not_called()


def test_failed_or_stale_exact_ack_keeps_durable_attention() -> None:
    marks = _Marks((("conv-a", RECEIPT_A),))
    app = _App(marks)
    runtime = ConsoleRuntime(app)

    assert runtime.recompute_console_attention() is True
    assert runtime.acknowledge_rendered_terminal_receipts(
        (("wrong-conversation", RECEIPT_A),)
    ) == ()
    assert marks.pairs == [("conv-a", RECEIPT_A)]
    assert runtime.console_needs_attention is True


def test_ack_rechecks_view_claim_inside_serialized_deletion() -> None:
    marks = _Marks((("conv-a", RECEIPT_A),))
    runtime = ConsoleRuntime(_App(marks))
    outgoing = SimpleNamespace(console_view_hooks=lambda: {})
    outgoing_generation = runtime.attach_view(outgoing)
    reached_serialization = threading.Event()
    original_reserve = runtime._reserve_attention_revision

    def reserve_attention_revision() -> int:
        revision = original_reserve()
        reached_serialization.set()
        return revision

    runtime._reserve_attention_revision = reserve_attention_revision
    result: list[tuple[str, ...]] = []

    runtime._attention_operation_lock.acquire()
    try:
        worker = threading.Thread(
            target=lambda: result.append(
                runtime.acknowledge_rendered_terminal_receipts(
                    (("conv-a", RECEIPT_A),),
                    view=outgoing,
                    attachment_generation=outgoing_generation,
                )
            )
        )
        worker.start()
        assert reached_serialization.wait(timeout=1.0)
        successor = SimpleNamespace(console_view_hooks=lambda: {})
        assert runtime.attach_view(successor) is not None
    finally:
        runtime._attention_operation_lock.release()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert result == [()]
    assert marks.pairs == [("conv-a", RECEIPT_A)]


@pytest.mark.parametrize("bad_pairs", [(), (("", RECEIPT_A),), (("conv", ""),)])
def test_acknowledgement_input_is_bounded_and_content_free(
    bad_pairs: tuple[tuple[str, str], ...],
) -> None:
    runtime = ConsoleRuntime(_App(_Marks()))

    assert runtime.acknowledge_rendered_terminal_receipts(bad_pairs) == ()


def _sync_harness(*, refresh_raises: bool = False):
    events: list[str] = []
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="private answer",
        metadata=MessageMetadata(terminal_receipt_id=RECEIPT_A),
    )

    async def refresh_messages() -> None:
        events.append("refresh")
        if refresh_raises:
            raise RuntimeError("render failed")

    transcript = MagicMock()
    transcript._row_widgets = {f"assistant-turn:{message.id}": object()}
    transcript.mounted_message_content_ids.return_value = frozenset({message.id})
    transcript.refresh_messages.side_effect = refresh_messages
    transcript.apply_turn_activity.return_value = ""

    store = MagicMock()
    store.active_session_id = "session-a"
    store._sessions = {
        "session-a": SimpleNamespace(persisted_conversation_id="conv-a")
    }
    store.session_context_summary.return_value = (None, None)

    runtime = MagicMock()

    def acknowledge(pairs, **_kwargs):
        events.append("ack")
        return tuple(receipt for _conversation, receipt in pairs)

    runtime.acknowledge_rendered_terminal_receipts.side_effect = acknowledge
    screen = SimpleNamespace(
        query_one=lambda *_args, **_kwargs: transcript,
        _message=SimpleNamespace(
            _native_console_messages=lambda: [message],
            sync_selected_fork_eligibility=lambda _transcript: (None, None),
        ),
        _change_review_projection=SimpleNamespace(project=lambda messages: messages),
        _review_selection=SimpleNamespace(
            _console_change_review_provider=lambda: None,
            _sync_console_annotation_discovery=lambda _store: None,
        ),
        _show_model_thinking=lambda: False,
        _library_activity=SimpleNamespace(sync_transcript=lambda _transcript: {}),
        _console_transcript_region_or_none=lambda: None,
        _console_presentation_context=lambda: MagicMock(),
        _console_change_review_provider=lambda: None,
        _sync_console_citation_count_discovery=lambda _messages: None,
        _console_chat_controller=None,
        _console_original_attempt_previews={},
        _pending_console_swipe_selection=None,
        _ensure_console_chat_store=lambda: store,
        _agent=SimpleNamespace(
            console_turn_activity=lambda: "",
            console_turn_activity_abandon_action=lambda: "",
        ),
        _console_citation_counts={},
        _sync_console_annotation_discovery=lambda _store: None,
        _console_annotation_previews={},
        _current_console_run_status_value=lambda: "completed",
        _image=SimpleNamespace(
            _build_console_image_specs=lambda _messages: {},
            _build_generation_card_specs=lambda _messages: {},
            _pending_console_generation_card_images=lambda *_args: (),
        ),
        _video=SimpleNamespace(_build_video_card_specs=lambda _messages: {}),
        _ensure_console_image_view=lambda: (
            None,
            SimpleNamespace(pending_ids=lambda _messages: ()),
        ),
        _recent_console_image_messages=lambda _messages: (),
        _console_image_preparing=set(),
        run_worker=lambda *_args, **_kwargs: None,
        _native_console_transcript_fingerprint=lambda _messages: (),
        _console_speech_states={},
        _last_native_transcript_refresh_key=None,
        _sync_console_transcript_guidance=lambda: None,
        _console_runtime=lambda: runtime,
        _rendered_console_terminal_receipts=(
            ChatScreen._rendered_console_terminal_receipts
        ),
    )
    return screen, runtime, events, message


@pytest.mark.asyncio
async def test_successful_full_row_refresh_acknowledges_after_exact_mount() -> None:
    screen, runtime, events, message = _sync_harness()

    await ChatScreen._sync_native_console_transcript(screen)

    assert events == ["refresh", "ack"]
    runtime.acknowledge_rendered_terminal_receipts.assert_called_once_with(
        (("conv-a", RECEIPT_A),),
        view=screen,
        attachment_generation=None,
    )
    assert message.content not in repr(runtime.mock_calls)


@pytest.mark.asyncio
async def test_failed_full_row_refresh_acknowledges_nothing() -> None:
    screen, runtime, events, _message = _sync_harness(refresh_raises=True)

    with pytest.raises(RuntimeError, match="render failed"):
        await ChatScreen._sync_native_console_transcript(screen)

    assert events == ["refresh"]
    runtime.acknowledge_rendered_terminal_receipts.assert_not_called()


def test_terminal_receipt_change_invalidates_transcript_fingerprint() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="same visible result",
        status="complete",
    )
    screen = SimpleNamespace(
        _ensure_console_chat_store=lambda: SimpleNamespace(
            active_session_id="session-a"
        ),
        _console_presentation_context=lambda: SimpleNamespace(
            user_name="User",
            assistant_kind="Assistant",
            character_name=None,
            transcript_style=SimpleNamespace(value="default"),
            revision=0,
        ),
    )

    before = ChatScreen._native_console_transcript_fingerprint(screen, [message])
    message.metadata = MessageMetadata(terminal_receipt_id=RECEIPT_A)
    after = ChatScreen._native_console_transcript_fingerprint(screen, [message])

    assert after != before


@pytest.mark.asyncio
async def test_outgoing_refresh_cannot_ack_after_successor_claims_runtime() -> None:
    screen, _runtime_double, _events, _message = _sync_harness()
    entered_refresh = asyncio.Event()
    release_refresh = asyncio.Event()
    transcript = screen.query_one()

    async def refresh_messages() -> None:
        entered_refresh.set()
        await release_refresh.wait()

    transcript.refresh_messages.side_effect = refresh_messages
    marks = _Marks((("conv-a", RECEIPT_A),))
    runtime = ConsoleRuntime(_App(marks))
    screen.console_view_hooks = lambda: {}
    generation = runtime.attach_view(screen)
    screen._console_runtime_attachment_generation = generation
    screen._console_runtime = lambda: runtime

    refresh = asyncio.create_task(ChatScreen._sync_native_console_transcript(screen))
    await entered_refresh.wait()
    successor = SimpleNamespace(console_view_hooks=lambda: {})
    successor_generation = runtime.attach_view(successor)
    release_refresh.set()
    await refresh

    assert successor_generation is not None
    assert runtime.view is successor
    assert marks.pairs == [("conv-a", RECEIPT_A)]
    assert runtime.console_needs_attention is False


def test_only_receipts_whose_exact_rows_are_mounted_are_acknowledgeable() -> None:
    first = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
        metadata=MessageMetadata(terminal_receipt_id=RECEIPT_A),
    )
    second = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="second",
        metadata=MessageMetadata(terminal_receipt_id=RECEIPT_B),
    )
    transcript = SimpleNamespace(
        mounted_message_content_ids=lambda: frozenset({first.id}),
        # Contradictory implementation detail: ChatScreen must consume only
        # the transcript's successful-render evidence seam.
        _row_widgets={f"assistant-turn:{second.id}": object()},
    )

    assert ChatScreen._rendered_console_terminal_receipts(
        transcript,
        [first, second],
        conversation_id="conv-a",
    ) == (("conv-a", RECEIPT_A),)


def test_media_receipts_require_their_exact_hydrated_card_rows() -> None:
    ordinary = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="ordinary",
        metadata=MessageMetadata(terminal_receipt_id=RECEIPT_A),
    )
    image = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="image",
        metadata=MessageMetadata(terminal_receipt_id=RECEIPT_B),
        generation_metadata=(SimpleNamespace(),),
    )
    video = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="video",
        video_metadata=SimpleNamespace(terminal_receipt_id=RECEIPT_C),
    )
    transcript = SimpleNamespace(
        mounted_message_content_ids=lambda: frozenset(
            {ordinary.id, image.id}
        ),
        _row_widgets={
            f"assistant-turn:{ordinary.id}": object(),
            # These top-level details are intentionally wrong/incomplete;
            # the transcript-owned evidence above is authoritative.
            f"assistant-turn:{video.id}": object(),
        }
    )

    assert ChatScreen._rendered_console_terminal_receipts(
        transcript,
        [ordinary, image, video],
        conversation_id="conv-a",
    ) == (
        ("conv-a", RECEIPT_A),
        ("conv-a", RECEIPT_B),
    )


@pytest.mark.asyncio
async def test_transcript_reports_real_nested_ordinary_image_and_video_mounts() -> None:
    ordinary, image, video, image_spec, video_spec = _mounted_media_messages()
    app = _TranscriptHarness()

    async with app.run_test(size=(100, 36)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([ordinary, image, video])
        transcript.set_generation_card_specs({image.id: image_spec})
        transcript.set_video_card_specs({video.id: video_spec})
        await transcript.refresh_messages()
        await pilot.pause()

        assert transcript.mounted_message_content_ids() == frozenset(
            {ordinary.id, image.id, video.id}
        )
        image_turn = transcript.query_one(
            f"#console-assistant-turn-{image.id}", ConsoleAssistantTurnWidget
        )
        video_turn = transcript.query_one(
            f"#console-assistant-turn-{video.id}", ConsoleAssistantTurnWidget
        )
        assert image_turn.query_one(ConsoleGenerationCard).parent is image_turn.adjunct_stack
        assert video_turn.query_one(ConsoleVideoCard).parent is video_turn.adjunct_stack


@pytest.mark.asyncio
async def test_transcript_media_evidence_requires_successful_nested_card_mount() -> None:
    ordinary, image, video, image_spec, video_spec = _mounted_media_messages()
    app = _TranscriptHarness()

    async with app.run_test(size=(100, 36)) as pilot:
        transcript = app.query_one(ConsoleTranscript)
        transcript.set_messages([ordinary, image, video])
        # Missing media specs model failed hydration: the assistant shells may
        # mount, but neither terminal media result is acknowledgeable.
        await transcript.refresh_messages()
        await pilot.pause()
        assert transcript.mounted_message_content_ids() == frozenset({ordinary.id})

        transcript.set_generation_card_specs({image.id: image_spec})
        transcript.set_video_card_specs({video.id: video_spec})
        await transcript.refresh_messages()
        await pilot.pause()
        image_card = transcript.query_one(ConsoleGenerationCard)
        assert image.id in transcript.mounted_message_content_ids()

        # A card removed after reconciliation no longer counts as mounted
        # evidence even though its owning Assistant shell remains attached.
        await image_card.remove()
        assert image.id not in transcript.mounted_message_content_ids()
