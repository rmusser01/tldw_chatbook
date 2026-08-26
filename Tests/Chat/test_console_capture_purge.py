from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_chat_store import RecordingPersistence
from Tests.Chat.test_console_chat_controller_exchanges import StreamingGateway
from tldw_chatbook.Chat.console_chat_controller import (
    CapturePurgeStatus,
    ConsoleChatController,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_exchange_capture import CaptureDetail, ExchangeCapture


def _capture(run_tag: str, detail: CaptureDetail) -> ExchangeCapture:
    return ExchangeCapture(
        run_tag=run_tag,
        seq=0,
        created_at="t",
        provider="p",
        model="m",
        endpoint=None,
        request={"messages_payload": []},
        response={"content": run_tag},
        status="complete",
        usage_json=None,
        omitted_keys=(),
        capture_detail=detail,
    )


class PurgePersistence(RecordingPersistence):
    def __init__(self) -> None:
        super().__init__()
        self.deleted_conversations: list[str] = []
        self.raise_delete = False
        self.exchange_appends: list[list[dict]] = []
        self.full_keys_by_conversation: dict[
            str, set[tuple[str, str, int]]
        ] = {}
        self.delete_hook = None
        self.delete_threads: list[int] = []

    def list_full_exchange_keys_for_conversation(self, conversation_id):
        return frozenset(self.full_keys_by_conversation.get(conversation_id, ()))

    def delete_full_exchanges_for_conversation(
        self, conversation_id, *, expected_count=None
    ):
        if self.raise_delete:
            raise RuntimeError("injected rollback")
        keys = self.full_keys_by_conversation.get(conversation_id, set())
        if expected_count is not None and len(keys) != expected_count:
            raise RuntimeError("inventory changed")
        self.delete_threads.append(threading.get_ident())
        if self.delete_hook is not None:
            self.delete_hook()
        self.deleted_conversations.append(conversation_id)
        self.full_keys_by_conversation[conversation_id] = set()
        return len(keys)

    def append_message_exchanges(self, *, message_id, rows):
        self.exchange_appends.append([dict(row) for row in rows])
        return True


def _store_with_captures(*, ephemeral: bool = False):
    persistence = PurgePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(ephemeral=ephemeral)
    if not ephemeral:
        session.persisted_conversation_id = "conversation-1"
        persistence.full_keys_by_conversation["conversation-1"] = {
            ("message-1", "full", 0)
        }
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        persist=False,
    )
    message.persisted_message_id = None if ephemeral else "message-1"
    store._nodes_by_session[session.id][message.id].persisted_message_id = (
        message.persisted_message_id
    )
    store.attach_message_exchanges(
        message.id,
        [
            _capture("safe", CaptureDetail.SAFE),
            _capture("full", CaptureDetail.FULL),
        ],
    )
    store._abandoned_exchange_run_tags[message.id] = {"safe", "full"}
    store._exchange_blob_cache[message.id] = {
        ("safe", 0, "complete"): b"safe",
        ("full", 0, "complete"): b"full",
    }
    return store, session, message, persistence


def test_store_stages_every_fallible_replacement_before_durable_delete():
    store, session, message, persistence = _store_with_captures()
    original_exchanges = store.get_message(message.id).exchanges
    original_cache = store._exchange_blob_cache
    original_tags = store._abandoned_exchange_run_tags
    persistence.raise_delete = True

    stage = store.stage_full_capture_purge(session.id)
    with pytest.raises(RuntimeError, match="injected rollback"):
        store.commit_full_capture_purge(stage)

    assert store.get_message(message.id).exchanges == original_exchanges
    assert store._exchange_blob_cache is original_cache
    assert store._abandoned_exchange_run_tags is original_tags
    assert store.capture_revision(session.id) == 0


def test_store_commit_swaps_only_full_capture_state_and_advances_revision():
    store, session, message, persistence = _store_with_captures()

    removed = store.commit_full_capture_purge(
        store.stage_full_capture_purge(session.id)
    )

    assert removed == 1
    assert [capture.run_tag for capture in store.get_message(message.id).exchanges] == [
        "safe"
    ]
    assert store._exchange_blob_cache[message.id] == {
        ("safe", 0, "complete"): b"safe"
    }
    assert store._abandoned_exchange_run_tags[message.id] == {"safe"}
    assert store.capture_revision(session.id) == 1
    assert persistence.deleted_conversations == ["conversation-1"]

    persistence.exchange_appends.clear()
    store._persist_exchanges_only(store._nodes_by_session[session.id][message.id])
    assert [
        row["capture_detail"]
        for append in persistence.exchange_appends
        for row in append
    ] == ["safe"]


def test_store_purge_preserves_unrelated_session_owners_and_exact_safe_keys():
    store, session_a, message_a, persistence = _store_with_captures()
    session_b = store.create_session()
    session_b.persisted_conversation_id = "conversation-2"
    message_b = store.append_message(
        session_b.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer b",
        persist=False,
    )
    message_b.persisted_message_id = "message-2"
    store._nodes_by_session[session_b.id][message_b.id].persisted_message_id = (
        "message-2"
    )
    store.attach_message_exchanges(
        message_b.id,
        [_capture("safe-b", CaptureDetail.SAFE), _capture("full-b", CaptureDetail.FULL)],
    )
    store._exchange_blob_cache[message_b.id] = {
        ("safe-b", 0, "complete"): b"safe-b",
        ("full-b", 0, "complete"): b"full-b",
    }
    store._abandoned_exchange_run_tags[message_b.id] = {"safe-b", "full-b"}
    persistence.full_keys_by_conversation["conversation-2"] = {
        ("message-2", "full-b", 0)
    }
    store._exchange_blob_cache[message_a.id][("safe", 99, "complete")] = b"stale"

    store.commit_full_capture_purge(store.stage_full_capture_purge(session_a.id))

    assert store._exchange_blob_cache[message_a.id] == {
        ("safe", 0, "complete"): b"safe"
    }
    assert store._exchange_blob_cache[message_b.id] == {
        ("safe-b", 0, "complete"): b"safe-b",
        ("full-b", 0, "complete"): b"full-b",
    }
    assert store._abandoned_exchange_run_tags[message_b.id] == {"safe-b", "full-b"}
    assert {capture.run_tag for capture in store.get_message(message_b.id).exchanges} == {
        "safe-b",
        "full-b",
    }


def test_overlapping_session_stages_publish_only_monotonic_target_revision():
    store, session_a, message_a, persistence = _store_with_captures()
    session_b = store.create_session()
    session_b.persisted_conversation_id = "conversation-2"
    message_b = store.append_message(
        session_b.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="answer b",
        persist=False,
    )
    message_b.persisted_message_id = "message-2"
    store._nodes_by_session[session_b.id][message_b.id].persisted_message_id = (
        "message-2"
    )
    store.attach_message_exchanges(
        message_b.id, [_capture("full-b", CaptureDetail.FULL)]
    )
    store._exchange_blob_cache[message_b.id] = {
        ("full-b", 0, "complete"): b"full-b"
    }
    persistence.full_keys_by_conversation["conversation-2"] = {
        ("message-2", "full-b", 0)
    }
    stage_a = store.stage_full_capture_purge(session_a.id)
    stage_b = store.stage_full_capture_purge(session_b.id)

    store.commit_full_capture_purge(stage_a)
    store.commit_full_capture_purge(stage_b)

    assert store.capture_revision(session_a.id) == 1
    assert store.capture_revision(session_b.id) == 1
    assert [capture.run_tag for capture in store.get_message(message_a.id).exchanges] == [
        "safe"
    ]
    assert store._exchange_blob_cache[message_a.id] == {
        ("safe", 0, "complete"): b"safe"
    }


def test_staged_purge_nested_assignments_are_immutable():
    store, session, _message, _persistence = _store_with_captures()

    stage = store.stage_full_capture_purge(session.id)

    assert isinstance(stage.blob_cache, tuple)
    assert isinstance(stage.abandoned_tags, tuple)
    assert isinstance(stage.capture_revisions, tuple)
    with pytest.raises(TypeError):
        stage.blob_cache[0][1][("full", 9, "complete")] = b"mutation"
    with pytest.raises(AttributeError):
        stage.abandoned_tags[0][1].add("mutation")


def test_store_reports_union_of_durable_and_live_full_capture_identities():
    store, session, message, persistence = _store_with_captures()
    store.attach_message_exchanges(
        message.id, [_capture("live-only", CaptureDetail.FULL)]
    )
    persistence.full_keys_by_conversation["conversation-1"].add(
        ("durable-only-message", "durable-only", 0)
    )

    removed = store.commit_full_capture_purge(store.stage_full_capture_purge(session.id))

    assert removed == 3
    assert [capture.run_tag for capture in store.get_message(message.id).exchanges] == [
        "safe"
    ]


def test_ephemeral_store_uses_the_same_swaps_without_database_delete():
    store, session, message, persistence = _store_with_captures(ephemeral=True)

    removed = store.commit_full_capture_purge(
        store.stage_full_capture_purge(session.id)
    )

    assert removed == 1
    assert [capture.run_tag for capture in store.get_message(message.id).exchanges] == [
        "safe"
    ]
    assert store.capture_revision(session.id) == 1
    assert persistence.deleted_conversations == []


@pytest.mark.asyncio
async def test_controller_purge_blocks_an_active_primary_writer():
    store, session, _message, _persistence = _store_with_captures()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    blocker = asyncio.create_task(asyncio.sleep(10))
    controller._active_stream_tasks[session.id] = blocker
    try:
        availability = controller.capture_purge_availability(session.id)
        result = await controller.purge_full_captures(
            session.id, expected_capture_revision=0
        )
    finally:
        blocker.cancel()

    assert availability.can_purge is False
    assert availability.reason_code == "primary_writer_active"
    assert result.status is CapturePurgeStatus.BLOCKED
    assert result.removed_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writer", "reason_code"),
    [
        ("preparation", "preparation_active"),
        ("fleet", "fleet_writer_active"),
        ("retained", "retained_signals_active"),
        ("flush", "exchange_flush_active"),
    ],
)
async def test_controller_purge_blocks_every_non_primary_writer(
    writer: str, reason_code: str
):
    store, session, message, _persistence = _store_with_captures()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    if writer == "preparation":
        controller._active_submit_tasks[asyncio.current_task()] = session.id
    elif writer == "fleet":
        controller._agent_bridge = SimpleNamespace(
            has_unsettled_children=lambda _conversation_id: True
        )
    elif writer == "retained":
        controller._fleet_usage_reattach_sources[message.id] = object()
    else:
        controller._capture_exchange_flush_sessions.add(session.id)

    availability = controller.capture_purge_availability(session.id)
    result = await controller.purge_full_captures(session.id, 0)

    assert availability.can_purge is False
    assert availability.reason_code == reason_code
    assert result.status is CapturePurgeStatus.BLOCKED
    assert result.reason_code == reason_code
    assert result.removed_count == 0


@pytest.mark.asyncio
async def test_controller_purge_rejects_stale_revision_without_mutation():
    store, session, message, persistence = _store_with_captures()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    result = await controller.purge_full_captures(
        session.id, expected_capture_revision=9
    )

    assert result.status is CapturePurgeStatus.STALE
    assert {
        capture.run_tag for capture in store.get_message(message.id).exchanges
    } == {"safe", "full"}
    assert persistence.deleted_conversations == []


@pytest.mark.asyncio
async def test_controller_failed_delete_releases_lease_without_live_mutation():
    store, session, message, persistence = _store_with_captures()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    persistence.raise_delete = True

    result = await controller.purge_full_captures(session.id, 0)

    assert result.status is CapturePurgeStatus.FAILED
    assert result.removed_count == 0
    assert result.reason_code == "persistence_unavailable"
    assert store.capture_revision(session.id) == 0
    assert store.capture_quiescent(session.id) is False
    assert {
        capture.run_tag for capture in store.get_message(message.id).exchanges
    } == {"safe", "full"}


@pytest.mark.asyncio
async def test_controller_commit_is_one_owner_loop_section_while_lease_is_held():
    store, session, message, persistence = _store_with_captures()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    owner_thread = threading.get_ident()
    observations = []

    def observe_during_delete():
        observations.append(
            (
                threading.get_ident(),
                store.capture_quiescent(session.id),
                store.capture_revision(session.id),
                tuple(capture.run_tag for capture in store.get_message(message.id).exchanges),
            )
        )
        store.attach_message_exchanges(
            message.id, [_capture("late-full", CaptureDetail.FULL)]
        )

    persistence.delete_hook = observe_during_delete

    result = await controller.purge_full_captures(session.id, 0)

    assert result.status is CapturePurgeStatus.DELETED
    assert observations == [(owner_thread, True, 0, ("full", "safe"))]
    assert persistence.delete_threads == [owner_thread]
    assert [capture.run_tag for capture in store.get_message(message.id).exchanges] == [
        "safe"
    ]
    assert store.capture_revision(session.id) == 1
    assert store.capture_quiescent(session.id) is False


@pytest.mark.asyncio
async def test_controller_lease_rejects_new_submit_before_any_await():
    store, session, _message, _persistence = _store_with_captures()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    assert store.begin_capture_quiescence(session.id) is True
    try:
        submit = await controller.submit_draft(
            "must not be admitted", session_id=session.id
        )
    finally:
        store.end_capture_quiescence(session.id)

    assert submit.accepted is False
    assert submit.session_id == session.id
    assert "retry" in submit.visible_copy.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", [False, True])
async def test_missing_capture_target_returns_content_free_bounded_results(closed):
    store = ConsoleChatStore(persistence=PurgePersistence())
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    target = "missing-session"
    if closed:
        session = store.create_session()
        target = session.id
        store.close_session(target)

    availability = controller.capture_purge_availability(target)
    result = await controller.purge_full_captures(target, 0)

    assert availability.can_purge is False
    assert availability.reason_code == "target_missing"
    assert result.status is CapturePurgeStatus.BLOCKED
    assert result.removed_count == 0
    assert result.capture_revision == -1
    assert result.reason_code == "target_missing"
