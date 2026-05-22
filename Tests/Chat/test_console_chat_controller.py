import asyncio

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunStatus,
    ConsoleStagedSource,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class BlockedGateway:
    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": False,
                "visible_copy": "Provider blocked: select a model",
            },
        )()


class StreamingGateway:
    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": True,
                "provider": "llama_cpp",
                "model": "test-model",
                "base_url": "http://127.0.0.1:9099",
                "visible_copy": "",
            },
        )()

    async def stream_chat(self, resolution, messages):
        for chunk in ("hel", "lo"):
            yield chunk


class RecordingStreamingGateway(StreamingGateway):
    def __init__(self):
        self.messages_seen = None

    async def stream_chat(self, resolution, messages):
        self.messages_seen = messages
        yield "ok"


class FailingStreamingGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages):
        yield "partial"
        raise RuntimeError("llama.cpp stream failed")


class FailingBeforeChunkGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages):
        if getattr(resolution, "never_yield", False):
            yield ""
        raise RuntimeError("retry failed before streaming")


class EmptyStreamingGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages):
        if getattr(resolution, "never_yield", False):
            yield ""


class FakePersistence:
    def __init__(self):
        self.created_conversations = []
        self.created_messages = []
        self.updated_messages = []

    def create_conversation(self, **kwargs):
        self.created_conversations.append(kwargs)
        return "conv-1"

    def create_message(
        self,
        *,
        conversation_id,
        sender,
        content,
        image_data,
        image_mime_type,
        message_id=None,
        parent_message_id=None,
        feedback=None,
    ):
        kwargs = {
            "conversation_id": conversation_id,
            "sender": sender,
            "content": content,
            "image_data": image_data,
            "image_mime_type": image_mime_type,
            "message_id": message_id,
            "parent_message_id": parent_message_id,
            "feedback": feedback,
        }
        self.created_messages.append(kwargs)
        return f"msg-{len(self.created_messages)}"

    def update_message_content(
        self,
        *,
        message_id,
        content,
        image_data,
        image_mime_type,
        parent_message_id=None,
        feedback=None,
        update_parent=False,
        update_feedback=False,
    ):
        self.updated_messages.append(
            {
                "message_id": message_id,
                "content": content,
                "image_data": image_data,
                "image_mime_type": image_mime_type,
                "parent_message_id": parent_message_id,
                "feedback": feedback,
                "update_parent": update_parent,
                "update_feedback": update_feedback,
            }
        )
        return True


@pytest.mark.asyncio
async def test_blocked_send_preserves_draft_and_adds_recovery_message():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=BlockedGateway())

    result = await controller.submit_draft("hello")

    assert result.accepted is False
    assert result.should_clear_draft is False
    assert controller.run_state.status is ConsoleRunStatus.BLOCKED
    assert "Provider blocked" in controller.run_state.visible_copy
    assert store.messages_for_session(store.active_session_id)[-1].role.value == "system"


@pytest.mark.asyncio
async def test_blocked_workspace_source_preserves_draft_and_skips_provider_call():
    class RecordingGateway(BlockedGateway):
        calls = 0

        async def resolve_for_send(self, selection):
            self.calls += 1
            return await super().resolve_for_send(selection)

    context = ConsoleWorkspaceContext(
        active_workspace_id="workspace-a",
        staged_sources=(
            ConsoleStagedSource(
                source_id="note-1",
                label="Workspace B note",
                source_type="note",
                workspace_id="workspace-b",
            ),
        ),
    )
    gateway = RecordingGateway()
    store = ConsoleChatStore(workspace_context=context)
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    result = await controller.submit_draft("hello")

    assert result.accepted is False
    assert result.should_clear_draft is False
    assert gateway.calls == 0
    assert controller.run_state.status is ConsoleRunStatus.BLOCKED
    assert "Workspace B note" in controller.run_state.visible_copy


@pytest.mark.asyncio
async def test_submit_draft_streams_assistant_message_to_completion():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    result = await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert result.should_clear_draft is True
    assert messages[-2].content == "hello"
    assert messages[-1].content == "hello"
    assert messages[-1].status == "complete"
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_provider_messages_exclude_visible_recovery_system_messages():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=BlockedGateway())
    await controller.submit_draft("blocked")

    recording_gateway = RecordingStreamingGateway()
    controller.provider_gateway = recording_gateway
    await controller.submit_draft("hello")

    assert recording_gateway.messages_seen == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_stop_active_run_marks_assistant_message_stopped():
    class WaitingGateway(StreamingGateway):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_chat(self, resolution, messages):
            self.started.set()
            yield "partial"
            await self.release.wait()
            yield "ignored"

    gateway = WaitingGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    task = asyncio.create_task(controller.submit_draft("hello"))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert controller.stop_active_run() is True

    gateway.release.set()
    result = await task
    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert messages[-1].content == "partial"
    assert messages[-1].status == "stopped"
    assert controller.run_state.status is ConsoleRunStatus.STOPPED


@pytest.mark.asyncio
async def test_submit_draft_rejects_concurrent_send_while_streaming():
    class WaitingGateway(StreamingGateway):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_chat(self, resolution, messages):
            self.started.set()
            yield "partial"
            await self.release.wait()
            yield "done"

    gateway = WaitingGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    task = asyncio.create_task(controller.submit_draft("first"))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)

    blocked = await asyncio.wait_for(controller.submit_draft("second"), timeout=0.5)

    assert blocked.accepted is False
    assert blocked.should_clear_draft is False
    assert "already running" in blocked.visible_copy
    assert [
        message.content for message in store.messages_for_session(store.active_session_id)
        if message.role.value == "user"
    ] == ["first"]

    gateway.release.set()
    await task


@pytest.mark.asyncio
async def test_stop_active_run_returns_without_waiting_for_next_provider_chunk():
    class StalledGateway(StreamingGateway):
        def __init__(self):
            self.started = asyncio.Event()
            self.never_release = asyncio.Event()

        async def stream_chat(self, resolution, messages):
            self.started.set()
            yield "partial"
            await self.never_release.wait()
            yield "ignored"

    gateway = StalledGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    task = asyncio.create_task(controller.submit_draft("hello"))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert controller.stop_active_run() is True
    result = await asyncio.wait_for(task, timeout=0.5)

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert messages[-1].content == "partial"
    assert messages[-1].status == "stopped"
    assert controller.run_state.status is ConsoleRunStatus.STOPPED


@pytest.mark.asyncio
async def test_submit_draft_marks_assistant_failed_when_stream_errors():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(store=store, provider_gateway=FailingStreamingGateway())

    result = await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert result.should_clear_draft is True
    assert messages[-1].content == "partial"
    assert messages[-1].status == "failed"
    assert controller.run_state.status is ConsoleRunStatus.FAILED
    assert "stream failed" in controller.run_state.visible_copy
    assert persistence.updated_messages[-1]["message_id"] == messages[-1].persisted_message_id
    assert persistence.updated_messages[-1]["content"] == "partial"


@pytest.mark.asyncio
async def test_retry_failed_message_streams_replacement_from_original_turn():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    failing = FailingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=failing)
    await controller.submit_draft("hello")
    failed_id = store.messages_for_session(store.active_session_id)[-1].id

    controller.provider_gateway = StreamingGateway()
    result = await controller.retry_message(failed_id)

    assert result.accepted is True
    assert store.get_message(failed_id).status == "complete"
    assert store.get_message(failed_id).content == "hello"
    assert persistence.updated_messages[-1]["message_id"] == store.get_message(failed_id).persisted_message_id
    assert persistence.updated_messages[-1]["content"] == "hello"


@pytest.mark.asyncio
async def test_retry_rejects_failed_message_from_inactive_session():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=FailingStreamingGateway())
    await controller.submit_draft("hello")
    first_session_id = store.active_session_id
    failed_id = store.messages_for_session(first_session_id)[-1].id
    store.create_session(title="Chat 2")

    controller.provider_gateway = StreamingGateway()
    result = await controller.retry_message(failed_id)

    assert result.accepted is False
    assert result.should_clear_draft is False
    assert "original session" in result.visible_copy
    assert store.get_message(failed_id).status == "failed"
    assert store.active_session_id != first_session_id


@pytest.mark.asyncio
async def test_retry_failed_message_records_retrying_then_streaming_transition():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=FailingStreamingGateway())
    await controller.submit_draft("hello")
    failed_id = store.messages_for_session(store.active_session_id)[-1].id

    observed = []

    class ObservingGateway(StreamingGateway):
        async def stream_chat(self, resolution, messages):
            observed.append(controller.run_state.status)
            yield "recovered"

    controller.provider_gateway = ObservingGateway()
    result = await controller.retry_message(failed_id)

    assert result.accepted is True
    assert ConsoleRunStatus.RETRYING in controller.run_state_history
    assert observed == [ConsoleRunStatus.STREAMING]
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_retry_keeps_failed_content_if_replacement_fails_before_first_chunk():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=FailingStreamingGateway())
    await controller.submit_draft("hello")
    failed = store.messages_for_session(store.active_session_id)[-1]

    controller.provider_gateway = FailingBeforeChunkGateway()
    result = await controller.retry_message(failed.id)

    retried = store.get_message(failed.id)
    assert result.accepted is True
    assert retried.status == "failed"
    assert retried.content == "partial"
    assert controller.run_state.status is ConsoleRunStatus.FAILED


@pytest.mark.asyncio
async def test_initial_empty_stream_marks_assistant_failed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=EmptyStreamingGateway())

    result = await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert result.should_clear_draft is True
    assert messages[-1].status == "failed"
    assert messages[-1].content == ""
    assert controller.run_state.status is ConsoleRunStatus.FAILED
    assert "without content" in controller.run_state.visible_copy


@pytest.mark.asyncio
async def test_retry_keeps_failed_content_if_replacement_stream_is_empty():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=FailingStreamingGateway())
    await controller.submit_draft("hello")
    failed = store.messages_for_session(store.active_session_id)[-1]

    controller.provider_gateway = EmptyStreamingGateway()
    result = await controller.retry_message(failed.id)

    retried = store.get_message(failed.id)
    assert result.accepted is True
    assert retried.status == "failed"
    assert retried.content == "partial"
    assert controller.run_state.status is ConsoleRunStatus.FAILED
