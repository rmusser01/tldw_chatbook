import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleStagedSource,
    ConsoleWorkspaceContext,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
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


class CapturingGateway(StreamingGateway):
    def __init__(self):
        self.selection = None

    async def resolve_for_send(self, selection):
        self.selection = selection
        return await super().resolve_for_send(selection)


class WipBlockedGateway:
    async def resolve_for_send(self, selection):
        return type(
            "Resolution",
            (),
            {
                "ready": False,
                "visible_copy": "WIP: Console native provider 'openai' is not wired yet.",
            },
        )()


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


class EmptyHeartbeatStreamingGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages):
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


def test_controller_creates_and_switches_sessions():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    first = store.ensure_session(title="Chat 1")
    second = controller.new_session(title="Chat 2")

    assert store.active_session_id == second.id

    controller.switch_session(first.id)

    assert store.active_session_id == first.id


def test_controller_session_changes_clear_terminal_run_copy() -> None:
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    first = store.ensure_session(title="Chat 1")

    controller.run_state = ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete.")
    controller.new_session(title="Chat 2")

    assert controller.run_state.status is ConsoleRunStatus.IDLE
    assert controller.run_state.visible_copy == ""

    controller.run_state = ConsoleRunState(ConsoleRunStatus.BLOCKED, "Provider blocked.")
    controller.switch_session(first.id)

    assert controller.run_state.status is ConsoleRunStatus.IDLE
    assert controller.run_state.visible_copy == ""


def test_controller_session_changes_preserve_active_run_copy() -> None:
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    first = store.ensure_session(title="Chat 1")

    controller.run_state = ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
    controller.new_session(title="Chat 2")

    assert controller.run_state.status is ConsoleRunStatus.STREAMING
    assert controller.run_state.visible_copy == "Streaming response."

    controller.run_state = ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider.")
    controller.switch_session(first.id)

    assert controller.run_state.status is ConsoleRunStatus.VALIDATING
    assert controller.run_state.visible_copy == "Validating provider."


def test_controller_new_session_accepts_settings_snapshot() -> None:
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    settings = ConsoleSessionSettings(provider="llama_cpp", model="configured-model")

    session = controller.new_session(title="Configured", settings=settings)

    assert store.active_session_id == session.id
    assert store.session_settings(session.id) == settings


def test_update_provider_selection_updates_all_selection_fields() -> None:
    controller = ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=StreamingGateway(),
    )
    selection = ConsoleProviderSelection(
        provider="local_llamacpp",
        base_url="http://127.0.0.1:9099",
        explicit_model="runtime-model",
        configured_model="configured-model",
        temperature=0.2,
        top_p=0.6,
        min_p=0.04,
        top_k=35,
        max_tokens=256,
        seed=99,
        presence_penalty=0.1,
        frequency_penalty=0.2,
        reasoning_effort="high",
        reasoning_summary="auto",
        verbosity="medium",
        thinking_effort="low",
        thinking_budget_tokens=2048,
        streaming=False,
    )

    controller.update_provider_selection(selection)

    assert controller.provider == "local_llamacpp"
    assert controller.model == "runtime-model"
    assert controller.configured_model == "configured-model"
    assert controller.base_url == "http://127.0.0.1:9099"
    assert controller.temperature == 0.2
    assert controller.top_p == 0.6
    assert controller.min_p == 0.04
    assert controller.top_k == 35
    assert controller.max_tokens == 256
    assert controller.seed == 99
    assert controller.presence_penalty == 0.1
    assert controller.frequency_penalty == 0.2
    assert controller.reasoning_effort == "high"
    assert controller.reasoning_summary == "auto"
    assert controller.verbosity == "medium"
    assert controller.thinking_effort == "low"
    assert controller.thinking_budget_tokens == 2048
    assert controller.streaming is False
    assert controller._provider_selection().seed == 99
    assert controller._provider_selection().reasoning_effort == "high"
    assert controller._provider_selection().thinking_budget_tokens == 2048


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
async def test_submit_draft_sanitizes_user_text_before_storage_and_provider_send():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    result = await controller.submit_draft("hel\x00lo")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert messages[-2].content == "hello"
    assert gateway.messages_seen == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_controller_provider_selection_includes_sampling_settings() -> None:
    gateway = CapturingGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="m",
        temperature=0.4,
        top_p=0.7,
        min_p=0.03,
        top_k=20,
        max_tokens=300,
        streaming=False,
    )

    await controller.submit_draft("hello")

    assert gateway.selection.temperature == 0.4
    assert gateway.selection.top_p == 0.7
    assert gateway.selection.min_p == 0.03
    assert gateway.selection.top_k == 20
    assert gateway.selection.max_tokens == 300
    assert gateway.selection.streaming is False


@pytest.mark.asyncio
async def test_submit_draft_blocks_unsafe_markup_before_storage_or_provider_send():
    class CountingGateway(StreamingGateway):
        def __init__(self):
            self.resolve_calls = 0

        async def resolve_for_send(self, selection):
            self.resolve_calls += 1
            return await super().resolve_for_send(selection)

    gateway = CountingGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    result = await controller.submit_draft("<script>alert('xss')</script>")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is False
    assert result.should_clear_draft is False
    assert gateway.resolve_calls == 0
    assert [message.role for message in messages] == [ConsoleMessageRole.SYSTEM]
    assert "unsafe" in messages[0].content


@pytest.mark.asyncio
async def test_blocked_provider_wip_copy_is_normalized_once_in_controller():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=WipBlockedGateway())

    result = await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is False
    assert result.visible_copy == "Provider blocked: WIP: Console native provider 'openai' is not wired yet."
    assert [message.content for message in messages] == [result.visible_copy]
    assert controller.run_state.visible_copy == result.visible_copy
    assert controller.run_state_history[-1] is ConsoleRunStatus.BLOCKED


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
    messages = store.messages_for_session(store.active_session_id)
    assert messages[-1].content == "partial"
    assert messages[-1].status == "stopped"
    assert controller.run_state.status is ConsoleRunStatus.STOPPED

    gateway.release.set()
    result = await task
    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert messages[-1].content == "partial"
    assert messages[-1].status == "stopped"
    assert controller.run_state.status is ConsoleRunStatus.STOPPED


def test_stop_active_run_falls_back_to_visible_streaming_assistant_message():
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    assistant = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.append_stream_chunk(assistant.id, "partial")
    controller.run_state = ConsoleRunState(
        ConsoleRunStatus.STREAMING,
        "Streaming response.",
    )
    controller._active_assistant_message_id = None

    assert controller.stop_active_run() is True

    messages = store.messages_for_session(session.id)
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
async def test_submit_draft_rejects_concurrent_send_during_provider_validation():
    class SlowResolveGateway(StreamingGateway):
        def __init__(self):
            self.resolve_started = asyncio.Event()
            self.release = asyncio.Event()

        async def resolve_for_send(self, selection):
            self.resolve_started.set()
            await self.release.wait()
            return await super().resolve_for_send(selection)

        async def stream_chat(self, resolution, messages):
            yield "done"

    gateway = SlowResolveGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    task = asyncio.create_task(controller.submit_draft("first"))
    await asyncio.wait_for(gateway.resolve_started.wait(), timeout=1)

    blocked = await asyncio.wait_for(controller.submit_draft("second"), timeout=0.5)

    assert blocked.accepted is False
    assert blocked.should_clear_draft is False
    assert "already running" in blocked.visible_copy
    assert controller.run_state.status is ConsoleRunStatus.VALIDATING

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
async def test_shutdown_stops_and_awaits_active_stream_task():
    """Verify controller shutdown stops and drains an active stream task."""
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

    await asyncio.wait_for(controller.shutdown(), timeout=0.5)
    result = await asyncio.wait_for(task, timeout=0.1)

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert messages[-1].content == "partial"
    assert messages[-1].status == "stopped"
    assert controller.run_state.status is ConsoleRunStatus.STOPPED
    assert controller._active_stream_task is None


@pytest.mark.asyncio
async def test_shutdown_ignores_failed_active_stream_task():
    async def fail_before_shutdown():
        raise RuntimeError("stream task failed before shutdown")

    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    task = asyncio.create_task(fail_before_shutdown())
    await asyncio.sleep(0)
    assert task.done()

    controller._active_stream_task = task
    controller._stop_requested = True

    await controller.shutdown()

    assert controller._active_stream_task is None
    assert controller._stop_requested is False


@pytest.mark.asyncio
async def test_close_streaming_session_stops_run_without_key_error():
    class WaitingGateway(StreamingGateway):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_chat(self, resolution, messages):
            yield "partial"
            self.started.set()
            await self.release.wait()
            yield "ignored"

    gateway = WaitingGateway()
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    task = asyncio.create_task(controller.submit_draft("hello"))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    session_id = store.active_session_id

    assert session_id is not None
    assert controller.run_state.status is ConsoleRunStatus.STREAMING

    controller.close_session(session_id)
    gateway.release.set()
    result = await asyncio.wait_for(task, timeout=0.5)

    assert result.accepted is True
    assert result.visible_copy == "Session closed."
    assert store.sessions() == []
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
    assert messages[-1].content.startswith("partial")
    assert "Provider stream failed: llama.cpp stream failed" in messages[-1].content
    assert messages[-1].status == "failed"
    assert controller.run_state.status is ConsoleRunStatus.FAILED
    assert "stream failed" in controller.run_state.visible_copy
    assert persistence.updated_messages[-1]["message_id"] == messages[-1].persisted_message_id
    assert "Provider stream failed: llama.cpp stream failed" in persistence.updated_messages[-1]["content"]


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
async def test_retry_failed_continuation_message_ends_provider_payload_with_user_instruction():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Prompt",
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Seed",
    )
    failed = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.append_stream_chunk(failed.id, "Partial continuation")
    store.mark_message_failed(failed.id)

    result = await controller.retry_message(failed.id)

    assert result.accepted is True
    assert gateway.messages_seen == [
        {"role": "user", "content": "Prompt"},
        {"role": "assistant", "content": "Seed"},
        {"role": "user", "content": "Continue and extend the selected message."},
    ]


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
    assert retried.content == failed.content
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
    assert retried.content == failed.content
    assert controller.run_state.status is ConsoleRunStatus.FAILED


@pytest.mark.asyncio
async def test_retry_ignores_empty_heartbeat_before_empty_replacement_stream_ends():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=FailingStreamingGateway())
    await controller.submit_draft("hello")
    failed = store.messages_for_session(store.active_session_id)[-1]

    controller.provider_gateway = EmptyHeartbeatStreamingGateway()
    result = await controller.retry_message(failed.id)

    retried = store.get_message(failed.id)
    assert result.accepted is True
    assert retried.status == "failed"
    assert retried.content == failed.content
    assert controller.run_state.status is ConsoleRunStatus.FAILED


@pytest.mark.asyncio
async def test_continue_from_message_streams_new_assistant_turn_after_selected_message():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    source = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="seed",
    )

    result = await controller.continue_from_message(source.id)

    messages = store.messages_for_session(session.id)
    assert result.accepted is True
    assert messages[-1].role is ConsoleMessageRole.ASSISTANT
    assert messages[-1].content == "hello"
    assert messages[-1].id != source.id


@pytest.mark.asyncio
async def test_continue_from_assistant_message_ends_provider_payload_with_user_instruction():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Prompt",
    )
    source = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Seed",
    )

    result = await controller.continue_from_message(source.id)

    assert result.accepted is True
    assert gateway.messages_seen == [
        {"role": "user", "content": "Prompt"},
        {"role": "assistant", "content": "Seed"},
        {"role": "user", "content": "Continue and extend the selected message."},
    ]


@pytest.mark.asyncio
async def test_continue_from_user_message_preserves_user_final_payload():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.ensure_session()
    source = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Tell me more",
    )

    result = await controller.continue_from_message(source.id)

    assert result.accepted is True
    assert gateway.messages_seen == [{"role": "user", "content": "Tell me more"}]


@pytest.mark.asyncio
async def test_regenerate_message_streams_new_selected_variant():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    source = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="seed",
    )

    result = await controller.regenerate_message(source.id)

    updated = store.get_message(source.id)
    assert result.accepted is True
    assert updated.variants.current.content == "hello"
    assert updated.variants.can_go_previous is True


@pytest.mark.asyncio
async def test_regenerate_continuation_message_ends_provider_payload_with_user_instruction():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Prompt",
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Seed",
    )
    continuation = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Continuation",
    )

    result = await controller.regenerate_message(continuation.id)

    assert result.accepted is True
    assert gateway.messages_seen == [
        {"role": "user", "content": "Prompt"},
        {"role": "assistant", "content": "Seed"},
        {"role": "user", "content": "Continue and extend the selected message."},
    ]


class _AutoTitleReadyGateway:
    async def resolve_for_send(self, selection):
        return SimpleNamespace(ready=True, visible_copy="")

    async def stream_chat(self, resolution, messages):
        yield "ok"


def _auto_title_controller() -> ConsoleChatController:
    return ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=_AutoTitleReadyGateway(),
    )


@pytest.mark.asyncio
async def test_submit_draft_auto_titles_default_session_from_first_message():
    controller = _auto_title_controller()
    session = controller.new_session()
    assert session.title == "Chat 1"

    await controller.submit_draft("fix the login bug in the auth flow")

    assert controller.store.sessions()[0].title == "fix the login bug in the au..."


@pytest.mark.asyncio
async def test_submit_draft_preserves_user_renamed_session_title():
    controller = _auto_title_controller()
    session = controller.new_session()
    controller.store.rename_session(session.id, "My research thread")

    await controller.submit_draft("hello there")

    assert controller.store.sessions()[0].title == "My research thread"


@pytest.mark.asyncio
async def test_submit_draft_does_not_retitle_after_first_send():
    controller = _auto_title_controller()
    controller.new_session()

    await controller.submit_draft("first message decides the title")
    first_title = controller.store.sessions()[0].title
    await controller.submit_draft("second message must not retitle")

    assert controller.store.sessions()[0].title == first_title
