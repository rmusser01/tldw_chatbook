import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.agent_models import (
    RUN_CANCELLED,
    RUN_DONE,
    RUN_ERROR,
    RunOutcome,
    ToolCall,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat import console_history_budget
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    build_mcp_review_hook,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleStagedSource,
    ConsoleWorkspaceContext,
    MessageAttachment,
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


class RaisingProbeGateway:
    async def resolve_for_send(self, selection):
        raise RuntimeError("probe boom")


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


def _last_failed_assistant(store, session_id=None):
    """Return the newest failed assistant message (skips failure system rows)."""
    messages = store.messages_for_session(session_id or store.active_session_id)
    return next(
        message
        for message in reversed(messages)
        if message.role is ConsoleMessageRole.ASSISTANT and message.status == "failed"
    )


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

    controller.run_state = ConsoleRunState(
        ConsoleRunStatus.COMPLETED, "Response complete."
    )
    controller.new_session(title="Chat 2")

    assert controller.run_state.status is ConsoleRunStatus.IDLE
    assert controller.run_state.visible_copy == ""

    controller.run_state = ConsoleRunState(
        ConsoleRunStatus.BLOCKED, "Provider blocked."
    )
    controller.switch_session(first.id)

    assert controller.run_state.status is ConsoleRunStatus.IDLE
    assert controller.run_state.visible_copy == ""


def test_controller_session_changes_preserve_active_run_copy() -> None:
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    first = store.ensure_session(title="Chat 1")

    controller.run_state = ConsoleRunState(
        ConsoleRunStatus.STREAMING, "Streaming response."
    )
    controller.new_session(title="Chat 2")

    assert controller.run_state.status is ConsoleRunStatus.STREAMING
    assert controller.run_state.visible_copy == "Streaming response."

    controller.run_state = ConsoleRunState(
        ConsoleRunStatus.VALIDATING, "Validating provider."
    )
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
        system_prompt="Session system prompt.",
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
    assert controller.system_prompt == "Session system prompt."
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
    assert (
        store.messages_for_session(store.active_session_id)[-1].role.value == "system"
    )


@pytest.mark.asyncio
async def test_not_ready_provider_still_echoes_the_user_message():
    """TASK-457(a): a not-ready provider must still echo the user's message
    (appended before the readiness probe) with the honest block-row after it,
    instead of silently dropping what the user sent."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=BlockedGateway())

    result = await controller.submit_draft("hello there")

    assert result.accepted is False
    messages = store.messages_for_session(store.active_session_id)
    assert [message.role.value for message in messages] == ["user", "system"]
    assert messages[0].content == "hello there"
    # The echoed row is failed so it never enters the next send's provider
    # context, and the draft is preserved for a re-attempt.
    assert messages[0].status == "failed"
    assert result.should_clear_draft is False


@pytest.mark.asyncio
async def test_probe_exception_after_optimistic_echo_marks_row_blocked():
    """TASK-457(a) (Qodo #777 review): if the readiness probe raises (or is
    cancelled) after the optimistic USER echo, the echoed row must still be
    failed so a never-sent message cannot leak into the next send's provider
    context (skip_failed only drops failed rows). The error still propagates."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=RaisingProbeGateway()
    )

    with pytest.raises(RuntimeError):
        await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert [message.role.value for message in messages] == ["user"]
    assert messages[0].content == "hello"
    assert messages[0].status == "failed"


@pytest.mark.asyncio
async def test_blocked_send_persists_no_durable_record():
    """TASK-485: a send blocked before it reaches the provider must leave NO
    durable record (no conversation, no message), so it cannot re-enter the next
    send's context after a resume/restart and leaves no orphan row. The in-memory
    echo is still shown (feedback) and failed (in-session context exclusion)."""
    from Tests.Chat.test_console_chat_store import FakePersistence

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(store=store, provider_gateway=BlockedGateway())

    result = await controller.submit_draft("hello")

    assert result.accepted is False
    messages = store.messages_for_session(store.active_session_id)
    assert messages[0].role.value == "user"
    assert messages[0].status == "failed"
    assert persistence.created_conversations == []
    assert persistence.created_messages == []


@pytest.mark.asyncio
async def test_accepted_send_persists_the_deferred_user_echo():
    """TASK-485: once a send is accepted the deferred USER echo is flushed to the
    durable conversation, so a reload shows the user's prompt (not just the
    assistant reply) — the successful path must not regress to a missing echo."""
    from Tests.Chat.test_console_chat_store import FakePersistence

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    await controller.submit_draft("hello")

    senders = [m["sender"] for m in persistence.created_messages]
    assert "user" in senders
    assert len(persistence.created_conversations) == 1


@pytest.mark.asyncio
async def test_skill_refuse_after_optimistic_echo_marks_row_blocked():
    """TASK-457(a) (code-review finding 1): a skill-substitution refusal after
    the optimistic echo is a block outcome like the not-ready / probe-raise
    paths — the echoed USER row must be failed so the refused command cannot
    leak into the next send's provider context (skip_failed only drops failed)."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    async def _refuse(messages):
        return messages, "Refused: untrusted skill."

    controller._apply_skill_substitution = _refuse

    result = await controller.submit_draft("run /evil")

    assert result.accepted is False
    messages = store.messages_for_session(store.active_session_id)
    assert messages[0].role.value == "user"
    assert messages[0].status == "failed"


@pytest.mark.asyncio
async def test_dictionary_apply_raise_after_optimistic_echo_marks_row_blocked():
    """TASK-457(a) (code-review finding 1): a raise from chat-dictionary / world-
    info application (or prefill) after the optimistic echo must also fail the
    echoed row so a never-sent message cannot leak into the next send."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    async def _boom(messages, session_id):
        raise RuntimeError("dict boom")

    controller._apply_chat_dictionaries = _boom

    with pytest.raises(RuntimeError):
        await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert messages[0].role.value == "user"
    assert messages[0].status == "failed"


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
async def test_submit_draft_prepends_system_prompt_message():
    """Native Console submit prepends a session's system prompt when set."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        system_prompt="Answer only in French.",
    )

    result = await controller.submit_draft("hello")

    assert result.accepted is True
    assert gateway.messages_seen == [
        {"role": "system", "content": "Answer only in French."},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.asyncio
async def test_submit_draft_omits_system_message_when_prompt_is_blank():
    """A whitespace-only system prompt is treated as no system prompt."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        system_prompt="   ",
    )

    await controller.submit_draft("hello")

    assert gateway.messages_seen == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_submit_draft_preserves_system_prompt_formatting_verbatim():
    """`strip()` is used only to decide "is this blank" -- the system
    message content sent to the provider must be the prompt exactly as
    set, leading/trailing whitespace and internal blank lines included."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    formatted_prompt = "  line1\n\n  line2  "
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        system_prompt=formatted_prompt,
    )

    result = await controller.submit_draft("hello")

    assert result.accepted is True
    assert gateway.messages_seen == [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": "hello"},
    ]


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
        system_prompt="Session system prompt.",
    )

    await controller.submit_draft("hello")

    assert gateway.selection.temperature == 0.4
    assert gateway.selection.top_p == 0.7
    assert gateway.selection.min_p == 0.03
    assert gateway.selection.top_k == 20
    assert gateway.selection.max_tokens == 300
    assert gateway.selection.streaming is False
    assert gateway.selection.system_prompt == "Session system prompt."


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
    controller = ConsoleChatController(
        store=store, provider_gateway=WipBlockedGateway()
    )

    result = await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is False
    assert (
        result.visible_copy
        == "Provider blocked: WIP: Console native provider 'openai' is not wired yet."
    )
    # TASK-457(a): the send now echoes the USER row before the block-row instead
    # of silently dropping it.
    assert [message.content for message in messages] == ["hello", result.visible_copy]
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
    # TASK-337: the durable stopped-by-user record follows the partial.
    assert messages[-1].content == "Response stopped by user."
    assert messages[-2].content == "partial"
    assert messages[-2].status == "stopped"
    assert controller.run_state.status is ConsoleRunStatus.STOPPED

    gateway.release.set()
    result = await task
    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    # TASK-337: the durable stopped-by-user record follows the partial.
    assert messages[-1].content == "Response stopped by user."
    assert messages[-2].content == "partial"
    assert messages[-2].status == "stopped"
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
    # TASK-337: the durable stopped-by-user record follows the partial.
    assert messages[-1].content == "Response stopped by user."
    assert messages[-2].content == "partial"
    assert messages[-2].status == "stopped"
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
        message.content
        for message in store.messages_for_session(store.active_session_id)
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
    # TASK-337: the durable stopped-by-user record follows the partial.
    assert messages[-1].content == "Response stopped by user."
    assert messages[-2].content == "partial"
    assert messages[-2].status == "stopped"
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
    # TASK-337: shutdown is not a user stop — no stopped-by-user row.
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
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )

    result = await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert result.accepted is True
    assert result.should_clear_draft is True
    assistant = messages[1]
    assert assistant.role is ConsoleMessageRole.ASSISTANT
    # The provider error must never be written into assistant content (it is
    # persisted and replayed to the model as conversation context).
    assert assistant.content == "partial"
    assert "Provider stream failed" not in assistant.content
    assert assistant.status == "failed"
    # The failure instead renders as a transcript-only system row.
    system_row = messages[-1]
    assert system_row.role is ConsoleMessageRole.SYSTEM
    assert system_row.content.startswith("Provider stream failed:")
    assert "llama.cpp stream failed" in system_row.content
    assert controller.run_state.status is ConsoleRunStatus.FAILED
    assert "stream failed" in controller.run_state.visible_copy
    assert result.visible_copy == system_row.content
    assert (
        persistence.updated_messages[-1]["message_id"] == assistant.persisted_message_id
    )
    assert persistence.updated_messages[-1]["content"] == "partial"
    persisted_contents = [
        str(entry.get("content", ""))
        for entry in [*persistence.created_messages, *persistence.updated_messages]
    ]
    assert not any(
        "Provider stream failed" in content for content in persisted_contents
    )


@pytest.mark.asyncio
async def test_retry_failed_message_streams_replacement_from_original_turn():
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    failing = FailingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=failing)
    await controller.submit_draft("hello")
    failed_id = _last_failed_assistant(store).id

    controller.provider_gateway = StreamingGateway()
    result = await controller.retry_message(failed_id)

    assert result.accepted is True
    assert store.get_message(failed_id).status == "complete"
    assert store.get_message(failed_id).content == "hello"
    assert (
        persistence.updated_messages[-1]["message_id"]
        == store.get_message(failed_id).persisted_message_id
    )
    assert persistence.updated_messages[-1]["content"] == "hello"


@pytest.mark.asyncio
async def test_retry_rejects_failed_message_from_inactive_session():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )
    await controller.submit_draft("hello")
    first_session_id = store.active_session_id
    failed_id = _last_failed_assistant(store, first_session_id).id
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
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )
    await controller.submit_draft("hello")
    failed_id = _last_failed_assistant(store).id

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
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        system_prompt="Answer only in French.",
    )
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
        {"role": "system", "content": "Answer only in French."},
        {"role": "user", "content": "Prompt"},
        {"role": "assistant", "content": "Seed"},
        {"role": "user", "content": "Continue and extend the selected message."},
    ]


@pytest.mark.asyncio
async def test_retry_keeps_failed_content_if_replacement_fails_before_first_chunk():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )
    await controller.submit_draft("hello")
    failed = _last_failed_assistant(store)

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
    controller = ConsoleChatController(
        store=store, provider_gateway=EmptyStreamingGateway()
    )

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
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )
    await controller.submit_draft("hello")
    failed = _last_failed_assistant(store)

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
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )
    await controller.submit_draft("hello")
    failed = _last_failed_assistant(store)

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
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Hi",
    )
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
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        system_prompt="Answer only in French.",
    )
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
        {"role": "system", "content": "Answer only in French."},
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
async def test_regenerate_message_streams_into_new_sibling_node():
    """TASK-6: regenerate forks a persisted sibling node under the anchor's
    own parent and streams into that NEW node -- the anchor is untouched and
    drops off the active path, reachable via ``set_active_leaf`` (see
    ``Tests/Chat/test_console_regenerate_branching.py`` for the full
    controller-level branching contract)."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Hi",
    )
    source = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="seed",
    )

    result = await controller.regenerate_message(source.id)

    assert result.accepted is True
    unchanged_source = store.get_message(source.id)
    assert unchanged_source.content == "seed"
    assert unchanged_source.variants is None
    assert source.id not in store.active_path_message_ids(session.id)

    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id != source.id
    new_sibling = store.get_message(new_leaf_id)
    assert new_sibling.content == "hello"
    assert new_sibling.variants is None


@pytest.mark.asyncio
async def test_regenerate_continuation_message_ends_provider_payload_with_user_instruction():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        system_prompt="Answer only in French.",
    )
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
        {"role": "system", "content": "Answer only in French."},
        {"role": "user", "content": "Prompt"},
        {"role": "assistant", "content": "Seed"},
        {"role": "user", "content": "Continue and extend the selected message."},
    ]


@pytest.mark.asyncio
async def test_leading_greeting_excluded_from_provider_payload():
    """A seeded character greeting (persisted ASSISTANT message before any
    user turn) must never reach the provider payload -- strict providers
    (Anthropic, Gemini) reject an assistant-first message array (task-427)."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.create_session(title="Chat with Elara")
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Greetings, traveler.",
        persist=False,
    )

    result = await controller.submit_draft("Hi")

    assert result.accepted is True
    sent = gateway.messages_seen
    roles = [m["role"] for m in sent]
    # No leading assistant: the first role sent is the user's turn.
    assert roles[0] == "user"
    # The greeting text is not in the outbound payload at all.
    assert all("Greetings, traveler." not in (m.get("content") or "") for m in sent)


@pytest.mark.asyncio
async def test_regenerate_on_leading_greeting_is_blocked():
    """Regenerating the seeded greeting before any user turn exists must be
    blocked rather than sending a payload with no user message."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.create_session(title="Chat with Elara")
    greeting = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Greetings.",
        persist=False,
    )

    result = await controller.regenerate_message(greeting.id)

    assert result.accepted is False
    assert gateway.messages_seen is None


@pytest.mark.asyncio
async def test_continue_from_leading_greeting_is_blocked():
    """Continuing from the seeded greeting before any user turn exists must
    be blocked rather than sending a payload with no user message."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = store.create_session(title="Chat with Elara")
    greeting = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Greetings.",
        persist=False,
    )

    result = await controller.continue_from_message(greeting.id)

    assert result.accepted is False
    assert gateway.messages_seen is None


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


def test_describe_stream_failure_classifies_common_errors():
    from tldw_chatbook.Chat.console_chat_controller import describe_stream_failure

    assert "timed out" in describe_stream_failure(asyncio.TimeoutError())
    assert "timed out" in describe_stream_failure(TimeoutError())
    assert "connection refused" in describe_stream_failure(ConnectionRefusedError())
    assert "could not connect" in describe_stream_failure(ConnectionError("boom"))

    class FakeHTTPStatusError(Exception):
        def __init__(self):
            super().__init__("")
            self.response = SimpleNamespace(status_code=502)

    assert "HTTP 502" in describe_stream_failure(FakeHTTPStatusError())
    # str(exc) alone was empty in the live failure ("[failed]"); the class
    # name must always be present so the copy is never blank.
    empty_detail = describe_stream_failure(RuntimeError())
    assert empty_detail == "RuntimeError error"
    with_detail = describe_stream_failure(RuntimeError("llama.cpp stream failed"))
    assert with_detail == "RuntimeError error (llama.cpp stream failed)"


@pytest.mark.asyncio
async def test_submit_draft_invokes_accepted_hook_after_acceptance_only():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    accepted_calls = []
    controller.on_submission_accepted = lambda: accepted_calls.append(True)

    result = await controller.submit_draft("hello")

    assert result.accepted is True
    assert accepted_calls == [True]


@pytest.mark.asyncio
async def test_submit_draft_does_not_invoke_accepted_hook_when_blocked():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=BlockedGateway())
    accepted_calls = []
    controller.on_submission_accepted = lambda: accepted_calls.append(True)

    result = await controller.submit_draft("hello")

    assert result.accepted is False
    assert accepted_calls == []


@pytest.mark.asyncio
async def test_submit_draft_accepted_hook_failure_does_not_break_run():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    def broken_hook():
        raise RuntimeError("composer vanished")

    controller.on_submission_accepted = broken_hook

    result = await controller.submit_draft("hello")

    assert result.accepted is True
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_regenerate_failure_adds_system_row_without_touching_variants():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    await controller.submit_draft("hello")
    messages = store.messages_for_session(store.active_session_id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)

    controller.provider_gateway = FailingStreamingGateway()

    class FailingBeforeAnyChunkGateway(StreamingGateway):
        async def stream_chat(self, resolution, messages):
            if getattr(resolution, "never_yield", False):
                yield ""
            raise RuntimeError("regen exploded")

    controller.provider_gateway = FailingBeforeAnyChunkGateway()
    result = await controller.regenerate_message(assistant.id)

    assert result.accepted is True
    assert "Provider stream failed:" in result.visible_copy
    assert "regen exploded" in result.visible_copy
    refreshed = store.get_message(assistant.id)
    assert refreshed.content == "hello"
    assert "Provider stream failed" not in refreshed.content
    system_row = store.messages_for_session(store.active_session_id)[-1]
    assert system_row.role is ConsoleMessageRole.SYSTEM
    assert "regen exploded" in system_row.content
    assert controller.run_state.status is ConsoleRunStatus.FAILED


def _pending_image(name="photo.png", data=b"\x89PNG-bytes"):
    return PendingAttachment(
        file_path=f"/tmp/{name}",
        display_name=name,
        file_type="image",
        insert_mode="attachment",
        data=data,
        mime_type="image/png",
        original_size=len(data),
        processed_size=len(data),
    )


def test_submit_draft_sends_image_parts_when_vision_capable(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, model="vision-model"
    )
    session = store.ensure_session()
    store.set_pending_attachment(session.id, _pending_image())

    result = asyncio.run(controller.submit_draft("what is this?"))

    assert result.accepted
    user_payload = gateway.messages_seen[-1]
    assert user_payload["role"] == "user"
    assert isinstance(user_payload["content"], list)
    assert user_payload["content"][0] == {"type": "text", "text": "what is this?"}
    assert user_payload["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert store.pending_attachment(session.id) is None  # consumed on send


def test_submit_draft_blocks_pending_image_on_non_vision_model(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: False)
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=RecordingStreamingGateway(), model="text-model"
    )
    session = store.ensure_session()
    store.set_pending_attachment(session.id, _pending_image())

    result = asyncio.run(controller.submit_draft("look at this"))

    assert not result.accepted
    assert "can't accept images" in result.visible_copy
    assert store.pending_attachment(session.id) is not None  # kept for model switch


def test_image_only_draft_is_sendable(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, model="vision-model"
    )
    session = store.ensure_session()
    store.set_pending_attachment(session.id, _pending_image())

    result = asyncio.run(controller.submit_draft(""))

    assert result.accepted
    user_payload = gateway.messages_seen[-1]
    assert [part["type"] for part in user_payload["content"]] == ["image_url"]


def test_history_images_capped_to_most_recent(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    monkeypatch.setattr(controller_module, "max_history_images", lambda p, m: 1)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, model="vision-model"
    )
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="first",
        image_data=b"img-1",
        image_mime_type="image/png",
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="second",
        image_data=b"img-2",
        image_mime_type="image/png",
    )

    asyncio.run(controller.submit_draft("and now?"))

    contents = [m["content"] for m in gateway.messages_seen if m["role"] == "user"]
    assert contents[0] == "first"  # over budget → text only
    assert isinstance(contents[1], list)  # most recent image kept
    assert contents[2] == "and now?"


def test_non_vision_history_stays_plain_strings(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: False)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, model="text-model"
    )
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="had an image",
        image_data=b"img-1",
        image_mime_type="image/png",
    )

    asyncio.run(controller.submit_draft("plain follow-up"))

    for message in gateway.messages_seen:
        assert isinstance(message["content"], str)


def test_submit_stages_all_pendings_and_clears(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, model="vision-model"
    )
    session = store.ensure_session()
    store.add_pending_attachment(session.id, _pending_image("a.png"))
    store.add_pending_attachment(session.id, _pending_image("b.png"))

    result = asyncio.run(controller.submit_draft("two pics"))

    assert result.accepted
    user_payload = gateway.messages_seen[-1]
    image_parts = [p for p in user_payload["content"] if p["type"] == "image_url"]
    assert len(image_parts) == 2
    assert store.pending_attachments(session.id) == []
    messages = store.messages_for_session(session.id)
    user_message = [m for m in messages if m.role is ConsoleMessageRole.USER][-1]
    assert len(user_message.attachments) == 2
    assert user_message.image_data is not None  # mirror holds


def test_image_budget_excludes_failed_send_blocked_echo(monkeypatch):
    """TASK-457(a) (code-review finding 2): a send-blocked USER echo persists as
    a `failed` row that KEEPS its attachment data but is dropped from the emitted
    payload by skip_failed. The image-budget RESERVATION loop must skip it too —
    otherwise the reserved-but-never-emitted slots starve a real older image
    message (silent wrong payload)."""
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    monkeypatch.setattr(controller_module, "max_history_images", lambda p, m: 1)
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=StreamingGateway(), model="vision-model"
    )
    session = store.ensure_session()
    from tldw_chatbook.Chat.console_chat_models import MessageAttachment

    def _att(tag):
        return (
            MessageAttachment(
                data=tag.encode(),
                mime_type="image/png",
                display_name=f"{tag}.png",
                position=0,
            ),
        )

    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="real",
        attachments=_att("real"),
    )
    blocked = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="blocked",
        attachments=_att("blocked"),
    )
    # Newer than the real message, failed, but still carrying its image bytes.
    store.mark_message_send_blocked(blocked.id)

    messages = store.messages_for_session(session.id)
    payloads = controller._provider_message_payloads(messages, skip_failed=True)

    user_payloads = [m for m in payloads if m["role"] == "user"]
    assert len(user_payloads) == 1
    images = (
        [p for p in user_payloads[0]["content"] if p["type"] == "image_url"]
        if isinstance(user_payloads[0]["content"], list)
        else []
    )
    assert len(images) == 1
    import base64

    decoded = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
    assert decoded == b"real"


def test_image_budget_counts_images_newest_first(monkeypatch):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    monkeypatch.setattr(controller_module, "max_history_images", lambda p, m: 3)
    # This test's subject is the image-count budget in
    # `_provider_message_payloads`, not the token-window trim added in
    # task 3. The default (unmocked) token window for an unrecognized
    # model/provider pair is small enough that 4 images at 1024 tokens
    # each would trip the trim and drop the "older" turn entirely --
    # stub a large window so the trim stays a no-op here.
    monkeypatch.setattr(
        console_history_budget, "get_model_token_limit", lambda model, provider: 100000
    )
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, model="vision-model"
    )
    session = store.ensure_session()
    from tldw_chatbook.Chat.console_chat_models import MessageAttachment

    def _atts(n, tag):
        return tuple(
            MessageAttachment(
                data=f"{tag}-{i}".encode(),
                mime_type="image/png",
                display_name=f"{tag}{i}.png",
                position=i,
            )
            for i in range(n)
        )

    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="older",
        attachments=_atts(2, "old"),
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="newer",
        attachments=_atts(2, "new"),
    )

    asyncio.run(controller.submit_draft("go"))

    user_payloads = [m for m in gateway.messages_seen if m["role"] == "user"]
    # newest ("newer") gets both images; "older" gets 1 (budget 3), oldest first-dropped.
    newer = user_payloads[1]
    older = user_payloads[0]
    newer_images = (
        [p for p in newer["content"] if p["type"] == "image_url"]
        if isinstance(newer["content"], list)
        else []
    )
    older_images = (
        [p for p in older["content"] if p["type"] == "image_url"]
        if isinstance(older["content"], list)
        else []
    )
    assert len(newer_images) == 2
    assert len(older_images) == 1
    # Budget-rule resolution: reservation walks messages newest-first, but a
    # partially-budgeted message emits its images in POSITION order up to the
    # reserved count -- "older" keeps its position-0 image ("old-0"), not its
    # newest-added one.
    import base64

    decoded = base64.b64decode(older_images[0]["image_url"]["url"].split(",", 1)[1])
    assert decoded == b"old-0"


def test_history_image_with_empty_mime_type_falls_back_to_default_mime(monkeypatch):
    """A resumed message can carry an attachment with ``mime_type=""`` (e.g.
    ``_console_messages_from_conversation_tree`` falls back to ``""`` when
    the persisted ``image_mime_type`` column is NULL). The provider payload
    builder must never emit a bare ``data:;base64,...`` URL for it -- that
    is an invalid data URI most providers reject outright. It must fall
    back to the same default mime the send-time staging path already uses
    (``pending.mime_type or "image/png"`` in this module, and
    ``image_mime_type or "image/png"`` in ``ConsoleChatStore.append_message``)."""
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, model="vision-model"
    )
    session = store.ensure_session()
    from tldw_chatbook.Chat.console_chat_models import MessageAttachment

    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="resumed image",
        attachments=(
            MessageAttachment(
                data=b"img-bytes", mime_type="", display_name="a.png", position=0
            ),
        ),
    )

    asyncio.run(controller.submit_draft("what is this?"))

    user_payloads = [m for m in gateway.messages_seen if m["role"] == "user"]
    resumed_payload = user_payloads[0]
    image_parts = [p for p in resumed_payload["content"] if p["type"] == "image_url"]
    assert len(image_parts) == 1
    url = image_parts[0]["image_url"]["url"]
    assert not url.startswith("data:;base64,")
    assert url.startswith("data:image/")


# ---------------------------------------------------------------------------
# build_mcp_review_hook (F1: per-turn stamp clearing, same-name sharing)
# ---------------------------------------------------------------------------


class _FakeReviewProvider:
    """Stands in for `MCPToolProvider` in `build_mcp_review_hook` unit tests."""

    def __init__(self, gated_names: set[str]) -> None:
        self._gated_names = gated_names
        self.apply_batch_decisions_calls: list[dict[str, str]] = []
        self._stamped: dict[str, str] = {}

    def pending_gate_for(self, name: str, args: dict) -> MCPPendingCall | None:
        if name not in self._gated_names:
            return None
        return MCPPendingCall(
            llm_name=name,
            server_key="local:srv",
            tool_name=name,
            server_label="Srv",
            arguments=dict(args or {}),
            reason="ask",
        )

    def apply_batch_decisions(self, decisions: dict[str, str]) -> None:
        self.apply_batch_decisions_calls.append(dict(decisions))
        # Mirrors MCPToolProvider.apply_batch_decisions' REPLACE semantics
        # (not merge) -- see that method's own docstring (Finding F1).
        self._stamped = dict(decisions or {})

    def stamped_decision(self, name: str) -> str | None:
        return self._stamped.get(name)


def test_build_mcp_review_hook_clears_stamps_even_when_nothing_needs_gating():
    """F1 (Qodo): a turn whose calls are all non-MCP (or already resolved
    without asking) must still clear any stamp an earlier turn set --
    pre-fix, this hook returned `{}` early WITHOUT ever calling
    `apply_batch_decisions`, leaving a stale stamp from a prior turn free
    to be misread by `invoke()`'s next same-name call as though it were
    stamped THIS turn (the "turn with no MCP calls between two MCP turns"
    leak)."""
    provider = _FakeReviewProvider(gated_names=set())
    hook = build_mcp_review_hook(provider, lambda pending: {})

    calls = [ToolCall(name="local_only_tool", args={}, call_id="1")]
    verdicts = hook(calls)

    assert verdicts == {}
    assert provider.apply_batch_decisions_calls == [{}]


def test_build_mcp_review_hook_stamps_decisions_when_gating_needed():
    provider = _FakeReviewProvider(gated_names={"mcp__srv__run"})
    seen_pending: list[list[MCPPendingCall]] = []

    def _approve(pending: list[MCPPendingCall]) -> dict[str, str]:
        seen_pending.append(pending)
        return {"mcp__srv__run": "approve_once"}

    hook = build_mcp_review_hook(provider, _approve)
    calls = [ToolCall(name="mcp__srv__run", args={"x": 1}, call_id="1")]

    verdicts = hook(calls)

    assert verdicts == {"mcp__srv__run": "proceed"}
    # I3: the hook clears at ENTRY (unconditionally, before the round trip)
    # and then stamps the real decisions -- two calls, not one, matching
    # `provider.apply_batch_decisions`'s own REPLACE semantics either way.
    assert provider.apply_batch_decisions_calls == [
        {},
        {"mcp__srv__run": "approve_once"},
    ]
    assert len(seen_pending) == 1


def test_build_mcp_review_hook_shares_one_verdict_for_same_name_calls_this_turn():
    """Two calls to the same llm_name in one turn are BOTH represented in
    `pending` (one `pending_gate_for` resolution each) but collapse to a
    single `request_mcp_approvals` round trip (T3/F1: same-name calls
    share one verdict) and a single verdict entry in the returned map."""
    provider = _FakeReviewProvider(gated_names={"mcp__srv__run"})
    round_trips: list[list[MCPPendingCall]] = []

    def _approve(pending: list[MCPPendingCall]) -> dict[str, str]:
        round_trips.append(pending)
        return {"mcp__srv__run": "approve_once"}

    hook = build_mcp_review_hook(provider, _approve)
    calls = [
        ToolCall(name="mcp__srv__run", args={"x": 1}, call_id="1"),
        ToolCall(name="mcp__srv__run", args={"x": 2}, call_id="2"),
    ]

    verdicts = hook(calls)

    assert verdicts == {"mcp__srv__run": "proceed"}
    assert len(round_trips) == 1  # ONE request_mcp_approvals round trip
    assert len(round_trips[0]) == 2  # ...covering both same-name calls
    # I3: the hook clears at ENTRY (unconditionally, before the round trip)
    # and then stamps the real decisions.
    assert provider.apply_batch_decisions_calls == [
        {},
        {"mcp__srv__run": "approve_once"},
    ]


def test_build_mcp_review_hook_clears_stamp_at_entry_before_a_raising_round_trip():
    """I3 (probe-verified): a raising `request_mcp_approvals` (e.g. the
    unguarded `_marshal_pending_approval` call during shutdown) must not
    leave the PREVIOUS turn's stamp live for `invoke()` to peek.
    `run_agent_loop`'s own hook-exception handling fails the WHOLE batch
    open (treats every call as "proceed"), so the clear must happen at hook
    ENTRY -- before the round trip can raise -- not only after one
    succeeds. Pre-fix, the clear only happened after a successful
    `apply_batch_decisions(decisions)` call, so a raise left turn 1's
    "approve_once" stamp live for the fail-open runtime to hand straight to
    invoke()."""
    provider = _FakeReviewProvider(gated_names={"mcp__srv__run"})

    # Turn 1: a normal round trip that approves.
    hook = build_mcp_review_hook(
        provider, lambda pending: {"mcp__srv__run": "approve_once"}
    )
    hook([ToolCall(name="mcp__srv__run", args={}, call_id="1")])
    assert provider.stamped_decision("mcp__srv__run") == "approve_once"

    # Turn 2: same tool, but request_mcp_approvals now raises mid-round-trip.
    def _raise(pending):
        raise RuntimeError("shutdown mid round-trip")

    hook2 = build_mcp_review_hook(provider, _raise)
    with pytest.raises(RuntimeError):
        hook2([ToolCall(name="mcp__srv__run", args={}, call_id="2")])

    # No stale stamp from turn 1 must survive the raise for invoke() to peek.
    assert provider.stamped_decision("mcp__srv__run") is None


# -----------------------------------------------------------------------------
# _finalize_agent_reply hardening (task-2)
# -----------------------------------------------------------------------------


def test_finalize_agent_reply_empty_final_text_uses_fallback():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="")
    result = controller._finalize_agent_reply(
        placeholder.id, session.id, outcome, variant_mode=False
    )

    messages = store.messages_for_session(session.id)
    assistant = messages[-1]
    assert assistant.content == "No response was generated."
    assert assistant.status == "complete"
    assert result.accepted is True
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


def test_finalize_agent_reply_missing_placeholder_appends_message():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    fake_id = "nonexistent-msg-id"

    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="hello back")
    result = controller._finalize_agent_reply(
        fake_id, session.id, outcome, variant_mode=False
    )

    messages = store.messages_for_session(session.id)
    assistant = messages[-1]
    assert assistant.role is ConsoleMessageRole.ASSISTANT
    assert assistant.content == "hello back"
    assert assistant.status == "complete"
    assert result.accepted is True
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


def test_finalize_agent_reply_error_marks_failed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(placeholder.id, "partial")

    outcome = RunOutcome(status=RUN_ERROR, steps=[], final_text="")
    result = controller._finalize_agent_reply(
        placeholder.id, session.id, outcome, variant_mode=False
    )

    messages = store.messages_for_session(session.id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.status == "failed"
    assert assistant.content == "partial"
    assert "Agent run failed" in controller.run_state.visible_copy
    assert result.accepted is True


def test_finalize_agent_reply_cancelled_marks_failed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    outcome = RunOutcome(status=RUN_CANCELLED, steps=[], final_text="")
    result = controller._finalize_agent_reply(
        placeholder.id, session.id, outcome, variant_mode=False
    )

    messages = store.messages_for_session(session.id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.status == "failed"
    assert controller.run_state.status is ConsoleRunStatus.FAILED
    assert result.accepted is True


def test_finalize_agent_reply_unknown_status_marks_failed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    outcome = RunOutcome(status="weird", steps=[], final_text="")
    result = controller._finalize_agent_reply(
        placeholder.id, session.id, outcome, variant_mode=False
    )

    messages = store.messages_for_session(session.id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.status == "failed"
    assert controller.run_state.status is ConsoleRunStatus.FAILED
    assert result.accepted is True


@pytest.mark.asyncio
async def test_build_context_snapshot_returns_current_and_next_send():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="Hi there")

    snapshot = await controller.build_context_snapshot(draft="Explain tools")

    assert len(snapshot.current_messages) == 2
    assert snapshot.current_messages[0].role == ConsoleMessageRole.USER
    assert snapshot.next_send_payload["messages"][-1]["content"].startswith("Explain tools")


@pytest.mark.asyncio
async def test_build_context_snapshot_does_not_execute_skills():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello")

    snapshot = await controller.build_context_snapshot(draft="/search tools")
    final_content = snapshot.next_send_payload["messages"][-1]["content"]
    assert "/search tools" in final_content
    assert "Skill command not resolved in preview" in final_content


@pytest.mark.asyncio
async def test_build_context_snapshot_empty_draft_does_not_annotate_historical_skill_command():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="/search tools")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="Here are some tools.")

    snapshot = await controller.build_context_snapshot(draft="")
    historical_user_content = snapshot.next_send_payload["messages"][0]["content"]

    assert historical_user_content == "/search tools"
    assert "Skill command not resolved in preview" not in historical_user_content


@pytest.mark.asyncio
async def test_build_context_snapshot_redacts_secrets():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="run")
    controller.system_prompt = "Use api_key=secret123"

    snapshot = await controller.build_context_snapshot(draft="ok")
    payload_text = str(snapshot.next_send_payload)
    assert "secret123" not in payload_text
    assert "[redacted]" in payload_text


@pytest.mark.asyncio
async def test_build_context_snapshot_redacts_quoted_secrets_without_mangling_json():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content='run with {"api_key": "secret123"}',
    )

    snapshot = await controller.build_context_snapshot(draft="ok")
    payload_text = str(snapshot.next_send_payload)
    assert "secret123" not in payload_text
    assert '"api_key": "[redacted]"' in payload_text


def test_redact_secrets_matches_hyphenated_and_camelcase_keys():
    payload = {
        "headers": {
            "x-api-key": "secret123",
            "apiKey": "secret456",
            "my_api_key": "secret789",
        }
    }

    redacted = ConsoleChatController._redact_secrets(payload)

    assert redacted["headers"]["x-api-key"] == "[redacted]"
    assert redacted["headers"]["apiKey"] == "[redacted]"
    assert redacted["headers"]["my_api_key"] == "[redacted]"


def test_redact_secrets_recursively_redacts_non_string_secret_values():
    payload = {"api_key": {"value": "secret"}}

    redacted = ConsoleChatController._redact_secrets(payload)

    assert "secret" not in str(redacted)
    assert redacted["api_key"] == {"value": "[redacted]"}


@pytest.mark.asyncio
async def test_build_context_snapshot_messages_are_independent_of_store():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    msg = store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello")

    snapshot = await controller.build_context_snapshot(draft="Follow up")
    original_content = snapshot.current_messages[0].content
    snapshot.current_messages[0].content = "mutated"

    reloaded = store.get_message(msg.id)
    assert reloaded.content == original_content


@pytest.mark.asyncio
async def test_build_context_snapshot_attachment_only_preview():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        provider="openai",
        model="gpt-4o",
    )
    store.ensure_session(title="Chat 1")

    attachment = MessageAttachment(
        data=b"fake-image-data",
        mime_type="image/png",
        display_name="image.png",
        position=0,
    )

    snapshot = await controller.build_context_snapshot(draft="", attachments=[attachment])

    messages = snapshot.next_send_payload["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert any(
        part.get("type") == "image_url"
        and part.get("image_url", {}).get("url") == "[image: data redacted for preview]"
        for part in content
    )


@pytest.mark.asyncio
async def test_build_context_snapshot_redacts_historical_image_data():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        provider="openai",
        model="gpt-4o",
    )
    session = store.ensure_session(title="Chat 1")

    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Previous image",
        attachments=(
            MessageAttachment(
                data=b"historical-image-data",
                mime_type="image/png",
                display_name="previous.png",
                position=0,
            ),
        ),
    )

    snapshot = await controller.build_context_snapshot(draft="Describe it")

    payload_text = str(snapshot.next_send_payload)
    assert "data:image/png;base64," not in payload_text
    assert "[image: data redacted for preview]" in payload_text


def test_replace_image_data_preserves_detail_and_handles_string_url():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc", "detail": "auto"},
                },
                {"type": "image_url", "image_url": "data:image/png;base64,def"},
                {"type": "image_url", "image_url": "http://example.com/img.png"},
            ],
        }
    ]

    redacted = ConsoleChatController._replace_image_data_with_placeholders(messages)

    dict_url = redacted[0]["content"][0]["image_url"]
    assert dict_url["url"] == "[image: data redacted for preview]"
    assert dict_url["detail"] == "auto"
    data_string_url = redacted[0]["content"][1]["image_url"]
    assert data_string_url == "[image: data redacted for preview]"
    plain_string_url = redacted[0]["content"][2]["image_url"]
    assert plain_string_url == "http://example.com/img.png"


def test_replace_image_data_redacts_anthropic_and_string_image_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
                    },
                },
                {"type": "image", "image": "data:image/png;base64,def"},
            ],
        }
    ]

    redacted = ConsoleChatController._replace_image_data_with_placeholders(messages)

    anthropic_part = redacted[0]["content"][0]
    assert anthropic_part["type"] == "image"
    assert anthropic_part["source"]["type"] == "base64"
    assert anthropic_part["source"]["media_type"] == "image/png"
    assert anthropic_part["source"]["data"] == "[image: data redacted for preview]"
    string_part = redacted[0]["content"][1]
    assert string_part["type"] == "image"
    assert string_part["image"] == "[image: data redacted for preview]"


def test_replace_image_data_redacts_string_content_with_data_urls():
    messages = [
        {
            "role": "user",
            "content": "Look at this image: data:image/png;base64,abc and this URL: http://example.com/img.png",
        },
        {
            "role": "assistant",
            "content": "data:image/jpeg;base64,xyz",
        },
    ]

    redacted = ConsoleChatController._replace_image_data_with_placeholders(messages)

    assert "data:image/png;base64,abc" not in redacted[0]["content"]
    assert "data:image/jpeg;base64,xyz" not in redacted[1]["content"]
    assert "http://example.com/img.png" in redacted[0]["content"]
    assert redacted[0]["content"].count("[image: data redacted for preview]") == 1
    assert redacted[1]["content"] == "[image: data redacted for preview]"


@pytest.mark.asyncio
async def test_build_context_snapshot_next_send_payload_independent_of_store():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello")

    snapshot = await controller.build_context_snapshot(draft="Follow up")
    original = str(snapshot.next_send_payload)

    # Mutate the returned payload in place; frozen only prevents reassignment
    # of the top-level field, not mutation of the nested dict/list structures.
    snapshot.next_send_payload["messages"].append(
        {"role": "user", "content": "injected"}
    )

    snapshot2 = await controller.build_context_snapshot(draft="Follow up")
    assert str(snapshot2.next_send_payload) == original


@pytest.mark.asyncio
async def test_build_context_snapshot_no_active_session_returns_empty():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    snapshot = await controller.build_context_snapshot(draft="hello")

    assert snapshot.current_messages == []
    assert snapshot.next_send_payload == {}


@pytest.mark.asyncio
async def test_build_context_snapshot_includes_staged_sources():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    store.ensure_session(title="Chat 1")

    sources = [
        ConsoleStagedSource(
            source_id="note-1",
            label="Note one",
            source_type="note",
            workspace_id="workspace-a",
        ),
        ConsoleStagedSource(
            source_id="file-2",
            label="File two",
            source_type="file",
        ),
    ]

    snapshot = await controller.build_context_snapshot(draft="Summarize", staged_sources=sources)

    staged = snapshot.next_send_payload["staged_sources"]
    assert len(staged) == 2
    assert staged[0] == {"source_id": "note-1", "label": "Note one", "type": "note"}
    assert staged[1] == {"source_id": "file-2", "label": "File two", "type": "file"}


@pytest.mark.asyncio
async def test_build_context_snapshot_isolates_assembly_errors():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello")

    async def _failing_apply(messages, session_id):
        raise RuntimeError("dictionary applier exploded")

    controller._apply_chat_dictionaries = _failing_apply

    snapshot = await controller.build_context_snapshot(draft="Follow up")

    assert len(snapshot.current_messages) == 1
    assert snapshot.current_messages[0].content == "Hello"
    payload = snapshot.next_send_payload
    assert "error" in payload
    assert "Failed to build context snapshot" in payload["error"]
    # The degraded payload must still include the transcript-derived messages
    # that were assembled before the failure, not an empty placeholder.
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["content"] == "Hello"
    assert payload["messages"][1]["content"].startswith("Follow up")
    assert payload["system"] == []


def test_annotate_skill_commands_multimodal_text_part():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "/search tools"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    ]

    annotated = ConsoleChatController._annotate_skill_commands(messages)

    text_part = annotated[0]["content"][0]
    assert text_part["type"] == "text"
    assert text_part["text"].startswith("/search tools")
    assert "Skill command not resolved in preview" in text_part["text"]
    assert annotated[0]["content"][1] == messages[0]["content"][1]


def test_annotate_skill_commands_ignores_leading_whitespace():
    messages = [{"role": "user", "content": "  /search tools"}]

    annotated = ConsoleChatController._annotate_skill_commands(messages)

    assert annotated[0]["content"].startswith("  /search tools")
    assert "Skill command not resolved in preview" in annotated[0]["content"]


def test_annotate_skill_commands_synthetic_turn_added_false_returns_unchanged():
    messages = [{"role": "user", "content": "/search tools"}]

    annotated = ConsoleChatController._annotate_skill_commands(
        messages, synthetic_turn_added=False
    )

    assert annotated == messages
    assert "Skill command not resolved in preview" not in annotated[0]["content"]


def test_build_tools_info_for_snapshot_no_bridge():
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=StreamingGateway()
    )

    info = controller._build_tools_info_for_snapshot()

    assert info["native_schemas"] == []
    assert info["mcp_note"] is None
    assert info["preview_note"] == "No native tools are configured for preview."


def test_build_tools_info_for_snapshot_with_native_schemas():
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=StreamingGateway()
    )
    controller._agent_bridge = SimpleNamespace(
        native_tool_schemas=lambda: [
            {"name": "calculator", "description": "Compute arithmetic.", "parameters": {}},
        ]
    )

    info = controller._build_tools_info_for_snapshot()

    assert info["native_schemas"] == [
        {"name": "calculator", "description": "Compute arithmetic.", "parameters": {}},
    ]
    assert info["mcp_note"] is None
    assert info["preview_note"] is not None
    assert "live run" in info["preview_note"]


def test_build_tools_info_for_snapshot_mcp_provider_present():
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=StreamingGateway()
    )
    controller._agent_bridge = SimpleNamespace(native_tool_schemas=lambda: [])
    controller._mcp_provider = object()

    info = controller._build_tools_info_for_snapshot()

    assert info["native_schemas"] == []
    assert info["mcp_note"] is not None
    assert "MCP tools are configured" in info["mcp_note"]
    assert info["preview_note"] == "No native tools are configured for preview."


def test_build_tools_info_for_snapshot_mcp_provider_absent():
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=StreamingGateway()
    )
    controller._agent_bridge = SimpleNamespace(native_tool_schemas=lambda: [])
    controller._mcp_provider = None

    info = controller._build_tools_info_for_snapshot()

    assert info["native_schemas"] == []
    assert info["mcp_note"] is None
    assert info["preview_note"] == "No native tools are configured for preview."


# -----------------------------------------------------------------------------
# Response prefill (SDD Task 5) — resolve, bypass, payload, seed, consume
# -----------------------------------------------------------------------------


def _arm_session(store):
    """Create+activate a session with settings; return it."""
    session = store.ensure_session(
        workspace_id=store.workspace_context.active_workspace_id
    )
    if session.settings is None:
        session.settings = ConsoleSessionSettings(provider="llama_cpp")
    return session


@pytest.mark.asyncio
async def test_submit_with_one_shot_prefill_appends_trailing_assistant_and_seeds():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "Sure thing:")

    result = await controller.submit_draft("hello")
    assert result.accepted
    assert gateway.messages_seen[-1] == {
        "role": "assistant",
        "content": "Sure thing:",
    }
    assert gateway.messages_seen[-2]["role"] == "user"
    messages = store.messages_for_session(session.id)
    assert messages[-1].content == "Sure thing:ok"  # seed + RecordingStreamingGateway's "ok"
    assert messages[-1].status == "complete"
    # one-shot consumed on complete
    assert store.session_one_shot_prefill(session.id) is None


@pytest.mark.asyncio
async def test_submit_with_pinned_prefill_applies_and_survives():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    store.set_session_pinned_prefill(session.id, "Voice:")

    await controller.submit_draft("hello")
    assert gateway.messages_seen[-1] == {"role": "assistant", "content": "Voice:"}
    # pinned survives the send
    assert store.session_settings(session.id).pinned_prefill == "Voice:"


@pytest.mark.asyncio
async def test_one_shot_wins_over_pinned_then_pinned_resumes():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    store.set_session_pinned_prefill(session.id, "PINNED")
    store.set_session_one_shot_prefill(session.id, "ONESHOT")

    await controller.submit_draft("first")
    assert gateway.messages_seen[-1]["content"] == "ONESHOT"
    await controller.submit_draft("second")
    assert gateway.messages_seen[-1]["content"] == "PINNED"


@pytest.mark.asyncio
async def test_blocked_send_retains_one_shot():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=BlockedGateway())
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "KEEP")
    await controller.submit_draft("hello")
    assert store.session_one_shot_prefill(session.id) == "KEEP"


@pytest.mark.asyncio
async def test_failed_send_retains_one_shot_and_shows_prefill():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingBeforeChunkGateway()
    )
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "KEEP")
    await controller.submit_draft("hello")
    assert store.session_one_shot_prefill(session.id) == "KEEP"
    # FailingBeforeChunkGateway raises, so a failure system row is appended
    # after the assistant message; _last_failed_assistant skips it (the
    # file's own convention for this exact shape, see line ~118).
    failed = _last_failed_assistant(store, session.id)
    assert failed.status == "failed"
    assert failed.content == "KEEP"  # seed materialized, no provider tokens


@pytest.mark.asyncio
async def test_zero_token_stream_fails_with_prefill_only_content():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=EmptyStreamingGateway()
    )
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "PRE")
    await controller.submit_draft("hello")
    messages = store.messages_for_session(session.id)
    assert messages[-1].status == "failed"
    assert messages[-1].content == "PRE"
    assert store.session_one_shot_prefill(session.id) == "PRE"


@pytest.mark.asyncio
async def test_stop_mid_stream_consumes_one_shot():
    store = ConsoleChatStore()

    class StopAfterFirstChunkGateway(StreamingGateway):
        def __init__(self):
            self.controller = None

        async def stream_chat(self, resolution, messages):
            yield "partial"
            self.controller._stop_requested = True
            yield "never-shown"

    gateway = StopAfterFirstChunkGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    gateway.controller = controller
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "PRE")
    await controller.submit_draft("hello")
    messages = store.messages_for_session(session.id)
    assert messages[-1].status == "stopped"
    assert messages[-1].content.startswith("PRE")
    assert store.session_one_shot_prefill(session.id) is None


@pytest.mark.asyncio
async def test_re_armed_one_shot_survives_in_flight_send_completion():
    """A ``/prefill`` issued mid-stream (re-arming the one-shot to a new
    value) must survive the in-flight send's completion: the send should
    only compare-and-clear the one-shot text it actually used, not
    whatever happens to be armed by the time it finishes."""
    store = ConsoleChatStore()

    class ReArmMidStreamGateway(StreamingGateway):
        def __init__(self):
            self.store = None
            self.session_id = None

        async def stream_chat(self, resolution, messages):
            yield "chunk-one"
            # Simulate a `/prefill SECOND` issued while this send is
            # still streaming.
            self.store.set_session_one_shot_prefill(self.session_id, "SECOND")
            yield "chunk-two"

    gateway = ReArmMidStreamGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    gateway.store = store
    gateway.session_id = session.id
    store.set_session_one_shot_prefill(session.id, "FIRST")

    result = await controller.submit_draft("hello")
    assert result.accepted
    messages = store.messages_for_session(session.id)
    assert messages[-1].status == "complete"
    assert messages[-1].content.startswith("FIRST")
    # SECOND survived — the send only consumed the FIRST it actually used.
    assert store.session_one_shot_prefill(session.id) == "SECOND"


@pytest.mark.asyncio
async def test_retry_zero_tokens_leaves_failed_content_untouched():
    """A pinned-prefill retry that yields no tokens must not seed: the lazy
    prepare_message_retry never runs, so the original failed content (the
    seed from the first attempt) stays exactly as it was."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingBeforeChunkGateway()
    )
    session = _arm_session(store)
    store.set_session_pinned_prefill(session.id, "PINNED")
    await controller.submit_draft("hello")
    # FailingBeforeChunkGateway raises, so a failure system row follows the
    # assistant message; _last_failed_assistant skips it.
    failed = _last_failed_assistant(store, session.id)
    assert failed.status == "failed"
    assert failed.content == "PINNED"  # seed from the failed first attempt

    controller.provider_gateway = EmptyStreamingGateway()
    await controller.retry_message(failed.id)
    after = store.get_message(failed.id)
    assert after.status == "failed"
    assert after.content == "PINNED"  # untouched — no double-seed, no wipe


@pytest.mark.asyncio
async def test_retry_applies_pinned_but_not_one_shot():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingBeforeChunkGateway()
    )
    session = _arm_session(store)
    store.set_session_pinned_prefill(session.id, "PINNED")
    await controller.submit_draft("hello")
    # FailingBeforeChunkGateway raises, so a failure system row follows the
    # assistant message; _last_failed_assistant skips it.
    failed = _last_failed_assistant(store, session.id)
    assert failed.status == "failed"

    gateway = RecordingStreamingGateway()
    controller.provider_gateway = gateway
    result = await controller.retry_message(failed.id)
    assert result.accepted
    assert gateway.messages_seen[-1] == {"role": "assistant", "content": "PINNED"}
    retried = store.get_message(failed.id)
    assert retried.status == "complete"
    assert retried.content == "PINNEDok"


@pytest.mark.asyncio
async def test_regenerate_applies_pinned_into_new_sibling():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    await controller.submit_draft("hello")
    original = store.messages_for_session(session.id)[-1]
    store.set_session_pinned_prefill(session.id, "PINNED")

    await controller.regenerate_message(original.id)
    assert gateway.messages_seen[-1] == {"role": "assistant", "content": "PINNED"}
    # The anchor is untouched; the pinned prefill lands in the NEW sibling.
    unchanged_original = store.get_message(original.id)
    assert unchanged_original.content == "ok"
    new_leaf_id = store.active_leaf(session.id)
    assert new_leaf_id != original.id
    regenerated = store.get_message(new_leaf_id)
    assert regenerated.content == "PINNEDok"


@pytest.mark.asyncio
async def test_continue_never_gets_prefill():
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    session = _arm_session(store)
    await controller.submit_draft("hello")
    assistant = store.messages_for_session(session.id)[-1]
    store.set_session_pinned_prefill(session.id, "PINNED")
    store.set_session_one_shot_prefill(session.id, "ONESHOT")

    await controller.continue_from_message(assistant.id)
    # continue keeps its synthetic USER instruction; nothing assistant-trailing
    assert gateway.messages_seen[-1]["role"] == "user"
    # one-shot untouched (continue is not a normal send)
    assert store.session_one_shot_prefill(session.id) == "ONESHOT"


@pytest.mark.asyncio
async def test_prefilled_send_bypasses_agent_loop():
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome

    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True
    )
    bridge_calls = []

    def run_reply(**kwargs):
        bridge_calls.append(kwargs)
        return RunOutcome(status=RUN_DONE, steps=[], final_text="agent says")

    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)
    session = _arm_session(store)

    # Control: without prefill the agent path handles the send.
    await controller.submit_draft("no prefill")
    assert len(bridge_calls) == 1
    assert gateway.messages_seen is None

    # With prefill armed the direct provider path handles it.
    store.set_session_one_shot_prefill(session.id, "PRE")
    await controller.submit_draft("with prefill")
    assert len(bridge_calls) == 1  # unchanged
    assert gateway.messages_seen[-1] == {"role": "assistant", "content": "PRE"}


class _SpyAgentBridge:
    """Records calls and refuses to be used -- for asserting the agent
    bridge is never invoked on a character session's send (task-427)."""

    def __init__(self):
        self.calls = 0

    def run_reply(self, **kwargs):
        self.calls += 1
        raise AssertionError("agent bridge should not be called for a character session")


@pytest.mark.asyncio
async def test_character_session_forces_plain_provider():
    """task-427: a session with character_id set always takes the plain
    provider branch, even with the global agent runtime enabled and a
    bridge present."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True
    )
    bridge = _SpyAgentBridge()
    controller._agent_bridge = bridge
    session = _arm_session(store)
    session.character_id = 7

    result = await controller.submit_draft("Hi")

    assert bridge.calls == 0
    assert result.accepted
    assert gateway.messages_seen is not None  # plain provider path ran


@pytest.mark.asyncio
async def test_normal_session_still_uses_agent_when_enabled():
    """Control for test_character_session_forces_plain_provider: a session
    with no character_id keeps using the agent bridge as before."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True
    )
    bridge_calls = []

    def run_reply(**kwargs):
        bridge_calls.append(kwargs)
        return RunOutcome(status=RUN_DONE, steps=[], final_text="agent says")

    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)
    session = _arm_session(store)
    assert session.character_id is None

    await controller.submit_draft("Hi")

    assert len(bridge_calls) == 1
    assert gateway.messages_seen is None  # agent path handled it, not the gateway


@pytest.mark.asyncio
async def test_stream_assistant_response_owner_lookup_survives_closed_session():
    """task-427 review fix: the force_plain owner-lookup added at the top of
    ``_stream_assistant_response`` calls ``store.session_id_for_message``,
    which raises ``KeyError`` for an unknown message id. ``retry_message`` /
    ``continue_from_message`` / ``regenerate_message`` resolve the message id
    and then ``await`` several times (resolve_for_send / skill substitution /
    chat dictionaries / world info) before reaching this method -- a
    ``close_session`` racing one of those awaits purges
    ``_message_session_index`` for that message, so the id is unknown by the
    time the gate runs. This must be treated exactly like every other
    "session vanished mid-flight" race in this method: swallowed and turned
    into the session-closed result, not an uncaught KeyError."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = _arm_session(store)
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    # Simulate the session closing while a caller (e.g. retry_message) was
    # still awaiting earlier stages of the pipeline: this purges
    # `_message_session_index` for `assistant.id` before the gate runs.
    controller.close_session(session.id)

    resolution = type(
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

    result = await controller._stream_assistant_response(
        resolution=resolution,
        provider_messages=[],
        assistant_message_id=assistant.id,
    )

    assert result.accepted is True
    assert result.visible_copy == "Session closed."


@pytest.mark.asyncio
async def test_build_context_snapshot_includes_armed_one_shot_prefill():
    """task-401: an armed prefill must appear in the preview exactly as the
    send would apply it -- trailing assistant turn + explicit indicator --
    and the snapshot read must not consume the one-shot."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = _arm_session(store)
    store.set_session_one_shot_prefill(session.id, "Sure thing:")

    snapshot = await controller.build_context_snapshot(draft="Explain tools")

    assert snapshot.next_send_payload["messages"][-1] == {
        "role": "assistant",
        "content": "Sure thing:",
    }
    assert snapshot.next_send_payload["response_prefill"] == {
        "source": "one-shot",
        "text": "Sure thing:",
        "agent_loop_bypassed": True,
    }
    # Read-only: the snapshot must not consume the armed one-shot.
    assert store.session_one_shot_prefill(session.id) == "Sure thing:"


@pytest.mark.asyncio
async def test_build_context_snapshot_includes_pinned_prefill():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = _arm_session(store)
    store.set_session_pinned_prefill(session.id, "Voice:")

    snapshot = await controller.build_context_snapshot(draft="Explain tools")

    assert snapshot.next_send_payload["messages"][-1] == {
        "role": "assistant",
        "content": "Voice:",
    }
    assert snapshot.next_send_payload["response_prefill"]["source"] == "pinned"


@pytest.mark.asyncio
async def test_build_context_snapshot_unchanged_when_no_prefill_armed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    _arm_session(store)

    snapshot = await controller.build_context_snapshot(draft="Explain tools")

    assert "response_prefill" not in snapshot.next_send_payload
    assert snapshot.next_send_payload["messages"][-1]["role"] == "user"


@pytest.mark.asyncio
async def test_send_trims_history_and_appends_note(monkeypatch):
    # Force a tiny window so a short history trims.
    monkeypatch.setattr(
        console_history_budget, "get_model_token_limit", lambda model, provider: 520
    )
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.update_provider_selection(
        ConsoleProviderSelection(
            provider="llama_cpp",
            explicit_model="test-model",
            configured_model="test-model",
            max_tokens=0,
        )
    )
    session = controller.new_session(title="Chat 1")  # creates + activates
    # Seed an over-budget history before the current turn.
    for i in range(6):
        store.append_message(session.id, role=ConsoleMessageRole.USER, content=f"old user {i} aa bb cc dd")
        store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content=f"old asst {i} aa bb cc dd")

    await controller.submit_draft("current question here")

    # The gateway saw a trimmed list (fewer than the full seeded history + turn).
    assert gateway.messages_seen is not None
    assert len(gateway.messages_seen) < 13
    # The latest user turn survived.
    assert any(
        m.get("role") == "user" and "current question here" in str(m.get("content", ""))
        for m in gateway.messages_seen
    )
    # A display-only SYSTEM trim note was appended to the transcript.
    rows = store.messages_for_session(session.id)
    assert any(
        r.role == ConsoleMessageRole.SYSTEM and "trimmed" in r.content.lower()
        for r in rows
    )


@pytest.mark.asyncio
async def test_send_that_fits_does_not_trim_or_note(monkeypatch):
    monkeypatch.setattr(
        console_history_budget, "get_model_token_limit", lambda model, provider: 100000
    )
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller.update_provider_selection(
        ConsoleProviderSelection(
            provider="llama_cpp", explicit_model="test-model", configured_model="test-model"
        )
    )
    session = controller.new_session(title="Chat 1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="one small turn")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="ok")

    await controller.submit_draft("next question")

    assert gateway.messages_seen is not None
    rows = store.messages_for_session(session.id)
    assert not any(
        r.role == ConsoleMessageRole.SYSTEM and "trimmed" in r.content.lower()
        for r in rows
    )


@pytest.mark.asyncio
async def test_trim_budgets_against_resolution_model_not_controller_state(monkeypatch):
    # Selection Race (Qodo review): the trim must budget against the model
    # captured in `resolution` -- the one the dispatch below actually sends --
    # not the controller's mutable self.model, which a provider/model switch
    # racing the pre-dispatch awaits could have changed. Give `resolution` a
    # tiny-window model and the controller a huge-window model; the trim must
    # fire on the small window, proving it reads resolution.*, not self.*.
    def _limit(model, provider):
        return 520 if model == "small-window" else 1_000_000

    monkeypatch.setattr(console_history_budget, "get_model_token_limit", _limit)
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    # Controller's mutable state points at a huge-window model...
    controller.update_provider_selection(
        ConsoleProviderSelection(
            provider="openai",
            explicit_model="huge-window",
            configured_model="huge-window",
            max_tokens=0,
        )
    )
    session = _arm_session(store)
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    provider_messages = []
    for i in range(6):
        provider_messages.append({"role": "user", "content": f"old user {i} aa bb cc dd"})
        provider_messages.append({"role": "assistant", "content": f"old asst {i} aa bb cc dd"})
    provider_messages.append({"role": "user", "content": "current question here"})

    # ...but the captured resolution (what actually dispatches) is the tiny model.
    resolution = SimpleNamespace(
        ready=True,
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        model="small-window",
        max_tokens=0,
        visible_copy="",
    )

    await controller._stream_assistant_response(
        resolution=resolution,
        provider_messages=provider_messages,
        assistant_message_id=assistant.id,
    )

    # Budgeted against the 520-token resolution window (not the 1M self.model),
    # so the 13-message history collapsed to just the current turn.
    assert gateway.messages_seen is not None
    assert len(gateway.messages_seen) < 13
    assert gateway.messages_seen[-1]["content"] == "current question here"
