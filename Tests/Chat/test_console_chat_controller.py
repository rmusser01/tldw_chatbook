import asyncio
import json
import threading
from dataclasses import replace
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
from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat import console_history_budget
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    build_mcp_review_hook,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.provider_continuation import parse_provider_continuation_json
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleNextSendHistoryProjection,
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleStagedSource,
    ConsoleWorkspaceContext,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore as _ConsoleChatStore
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpoint,
    ConsoleDispatchCheckpointState,
    ConsoleDispatchResultStatus,
    ConsoleDispatchWriteResult,
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


class ConsoleChatStore(_ConsoleChatStore):
    """Test store whose intentionally db-less sessions are explicitly ephemeral."""

    def create_session(self, **kwargs):
        kwargs.setdefault("ephemeral", self.persistence is None)
        return super().create_session(**kwargs)


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
                "resolved_destination": ConsoleResolvedDestination(
                    provider="llama_cpp",
                    model="test-model",
                    endpoint_identity="http://127.0.0.1:9099",
                    egress_class=ConsoleEgressClass.ON_DEVICE,
                ),
            },
        )()

    async def stream_chat(self, resolution, messages, **kwargs):
        for chunk in ("hel", "lo"):
            yield chunk


def _library_authority(attempt_id: str) -> ConsoleTurnLibraryAuthority:
    return ConsoleTurnLibraryAuthority(
        policy=ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
            policy_revision=1,
            source="durable",
        ),
        direct_library_tools=True,
        source_types=AUTOMATIC_LIBRARY_SOURCE_TYPES,
        scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), True),
        provider_intent=ConsoleProviderIntent("openai", "model-a", None),
        attempt_id=attempt_id,
    )


def _begin_controller_disclosure(
    store: ConsoleChatStore,
    session_id: str,
    *,
    content: str = "",
) -> tuple[object, ConsoleTurnExecutionContext]:
    local = ConsoleResolvedDestination(
        provider="llama_cpp",
        model="model-a",
        endpoint_identity="http://127.0.0.1:9099",
        egress_class=ConsoleEgressClass.ON_DEVICE,
    )
    external = ConsoleResolvedDestination(
        provider="openai",
        model="model-a",
        endpoint_identity="https://api.openai.com",
        egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
    )
    baseline = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    store.begin_session_library_destination_attempt(
        session_id,
        _library_authority("attempt-baseline"),
        local,
        baseline.id,
    )
    store.append_stream_chunk(baseline.id, "baseline")
    store.mark_message_complete(baseline.id)
    placeholder = store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
    )
    active_authority = _library_authority("attempt-active")
    store.begin_session_library_destination_attempt(
        session_id,
        active_authority,
        external,
        placeholder.id,
    )
    context = ConsoleTurnExecutionContext(
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=ConsoleProviderSelection(
                provider="openai",
                explicit_model="model-a",
            ),
            tool_configuration={"agent_runtime_enabled": True},
        ),
        library_authority=active_authority,
        resolved_destination=external,
    )
    return placeholder, context


class RecordingStreamingGateway(StreamingGateway):
    def __init__(self):
        self.messages_seen = None

    async def stream_chat(self, resolution, messages, **kwargs):
        self.messages_seen = messages
        yield "ok"


class CharacterEmoteStreamingGateway(StreamingGateway):
    def __init__(self, *chunks: str):
        self.chunks = chunks
        self.messages_seen = None

    async def stream_chat(self, resolution, messages, **kwargs):
        self.messages_seen = messages
        for chunk in self.chunks:
            yield chunk


def _activate_character_emote_pack(
    db: CharactersRAGDB,
    character_id: int,
) -> dict:
    assets = []
    for index, (expression_key, label) in enumerate(
        (("happy", "Never expose this label"), ("custom:smug", "Nor this one")),
        start=1,
    ):
        assets.append(
            {
                "expression_key": expression_key,
                "original_expression_key": expression_key,
                "display_label": label,
                "source_filename": f"asset-{index}.webp",
                "storage_relpath": f"fixture/asset-{index}.webp",
                "content_type": "image/webp",
                "bytes": index,
                "sha256": f"{index:064x}",
                "width": 8,
                "height": 8,
                "source_context": {"fixture": True},
                "is_animated": False,
                "frame_count": 1,
            }
        )
    return VisualIdentityRepository(db).activate_pack(
        pack={
            "title": "Controller emote fixture",
            "default_expression_key": "happy",
            "source_kind": "manual",
            "source_context": {"source_id": "controller.emote.fixture"},
        },
        manifest={"schema_id": "fixture/v1"},
        assets=assets,
        actor_kind="character",
        actor_id=character_id,
    )


class CapturingGateway(StreamingGateway):
    def __init__(self):
        self.selection = None

    async def resolve_for_send(self, selection):
        self.selection = selection
        return await super().resolve_for_send(selection)


class ContinuationHistoryGateway(ConsoleProviderGateway):
    """Real preparation with an in-process dispatch sink."""

    def __init__(self):
        super().__init__(environ={})
        self.prepared = None
        self.prepare_kwargs = None

    async def resolve_for_send(self, selection):
        return ConsoleProviderResolution(
            provider="deepseek",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-v4-flash",
            ready=True,
            readiness_key="deepseek",
            execution_key="deepseek",
            max_tokens=10,
            continuation_protocol="responses",
            resolved_destination=ConsoleResolvedDestination(
                provider="deepseek",
                model="deepseek-v4-flash",
                endpoint_identity="https://api.deepseek.com/v1",
                egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
            ),
        )

    def prepare_chat_request(self, resolution, messages, **kwargs):
        self.prepare_kwargs = kwargs
        return super().prepare_chat_request(
            resolution,
            messages,
            context_window_override_tokens=600,
            **kwargs,
        )

    async def stream_chat(self, resolution, messages, **kwargs):
        self.prepared = messages
        yield "ok"


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
    async def stream_chat(self, resolution, messages, **kwargs):
        yield "partial"
        raise RuntimeError("llama.cpp stream failed")


class FailingBeforeChunkGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages, **kwargs):
        if getattr(resolution, "never_yield", False):
            yield ""
        raise RuntimeError("retry failed before streaming")


class EmptyStreamingGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages, **kwargs):
        if getattr(resolution, "never_yield", False):
            yield ""


class EmptyHeartbeatStreamingGateway(StreamingGateway):
    async def stream_chat(self, resolution, messages, **kwargs):
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
        self.console_library_policy_repository = SimpleNamespace(read=self._read_policy)
        self.console_dispatch_repository = self
        self._policy_snapshot = None
        self._checkpoint = None

    def _read_policy(self, conversation_id):
        del conversation_id
        return SimpleNamespace(durable_policy=object(), snapshot=self._policy_snapshot)

    def _cas_state(self, transition):
        checkpoint = self._checkpoint
        if checkpoint is None:
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.NOT_FOUND, None, None, None
            )
        checkpoint = replace(
            checkpoint,
            state=transition.new_state,
            checkpoint_revision=checkpoint.checkpoint_revision + 1,
            assistant_message_version=checkpoint.assistant_message_version + 1,
            attempt_id=transition.new_attempt_id,
        )
        self._checkpoint = checkpoint
        return ConsoleDispatchWriteResult(
            ConsoleDispatchResultStatus.COMMITTED,
            checkpoint,
            checkpoint.assistant_message_version,
            "fake-payload-hash",
        )

    cas_state = _cas_state

    def settle_with_assistant(self, settlement):
        checkpoint = self._checkpoint
        if checkpoint is None:
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.NOT_FOUND, None, None, None
            )
        self.updated_messages.append(
            {
                "message_id": settlement.assistant_message_id,
                "content": settlement.content,
                "image_data": None,
                "image_mime_type": None,
                "parent_message_id": None,
                "feedback": None,
                "update_parent": False,
                "update_feedback": False,
            }
        )
        self._checkpoint = None
        return ConsoleDispatchWriteResult(
            ConsoleDispatchResultStatus.COMMITTED,
            None,
            checkpoint.assistant_message_version + 1,
            "fake-terminal-hash",
        )

    def create_conversation(self, **kwargs):
        self.created_conversations.append(kwargs)
        return "conv-1"

    def commit_durable_turn(self, *, acceptance, policy_candidate, conversation_kwargs):
        """Model the atomic adapter contract for durable controller tests."""
        self._policy_snapshot = ConsoleLibraryPolicySnapshot(
            auto_retrieve=policy_candidate.auto_retrieve,
            assistant_access=policy_candidate.assistant_access,
            policy_revision=1,
            source="durable",
        )
        self.created_conversations.append(dict(conversation_kwargs))
        self.created_messages.extend(
            (
                {
                    "conversation_id": acceptance.conversation_id,
                    "sender": "user",
                    "content": acceptance.user_content,
                    "message_id": acceptance.user_message_id,
                },
                {
                    "conversation_id": acceptance.conversation_id,
                    "sender": "assistant",
                    "content": "",
                    "message_id": acceptance.assistant_message_id,
                },
            )
        )
        checkpoint = ConsoleDispatchCheckpoint(
            assistant_message_id=acceptance.assistant_message_id,
            user_message_id=acceptance.user_message_id,
            conversation_id=acceptance.conversation_id,
            preparation_id=acceptance.preparation_id,
            attempt_id=acceptance.attempt_id,
            state=ConsoleDispatchCheckpointState.ACCEPTED,
            checkpoint_revision=1,
            user_message_version=1,
            assistant_message_version=1,
            origin=acceptance.origin,
            queue_entry_id=acceptance.queue_entry_id,
            frozen_authority=acceptance.frozen_authority,
            resolved_destination=acceptance.resolved_destination,
            reconstructability=acceptance.reconstructability,
        )
        self._checkpoint = checkpoint
        return checkpoint

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


def _roleplay_controller_fixture() -> tuple[
    ConsoleChatController, ConsoleChatStore, object
]:
    """Build a character chat whose trusted projections still say User."""
    store = ConsoleChatStore()
    settings = ConsoleSessionSettings(
        provider="llama_cpp", system_prompt="Speak with User."
    )
    session = store.create_session(
        settings=settings,
        assistant_kind="character",
        character_name="Alraune",
    )
    session.character_system_template = "Speak with {{user}}."
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Hello User.",
        metadata=MessageMetadata(
            template_kind="character_greeting",
            template_source="Hello {{user}}.",
        ),
    )
    session.user_display_name_override = "Captain Rowan"
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        system_prompt="Speak with User.",
        global_user_display_name=lambda: "User",
    )
    return controller, store, session


def test_roleplay_provider_messages_use_live_system_and_greeting_projection():
    controller, store, session = _roleplay_controller_fixture()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Say {{user}} literally",
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Generated {{user}} literally",
    )

    payload = controller._provider_messages_for_session(session.id)

    assert payload[0]["role"] == "system"
    assert payload[0]["content"].startswith("Speak with Captain Rowan.\n\n")
    assert payload[0]["content"].endswith("Hello Captain Rowan.")
    assert payload[1]["content"] == "Say {{user}} literally"
    assert payload[2]["content"] == "Generated {{user}} literally"
    assert all("Hello User" not in row["content"] for row in payload)


@pytest.mark.asyncio
async def test_roleplay_context_snapshot_matches_live_send_projection():
    controller, store, session = _roleplay_controller_fixture()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Continue",
    )

    expected = controller._provider_messages_for_session(session.id)
    snapshot = await controller.build_context_snapshot(draft="")

    assert snapshot.next_send_payload["messages"] == expected
    assert snapshot.next_send_payload["system"] == [expected[0]]
    assert snapshot.current_messages[0].content == "Hello Captain Rowan."


def test_controller_creates_and_switches_sessions():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    first = store.ensure_session(title="Chat 1")
    second = controller.new_session(title="Chat 2")

    assert store.active_session_id == second.id

    controller.switch_session(first.id)

    assert store.active_session_id == first.id


def test_controller_session_changes_clear_terminal_run_copy() -> None:
    """A session's own TERMINAL run copy is cleared only when it is the
    session being LEFT for another one -- never the session being arrived
    at, and never a session nothing has switched away from yet (spec §2:
    "clear the session you are leaving if terminal", implemented explicitly
    in `switch_session` -- see its own comment for why the id must be
    resolved before the store's active-session swap).
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    first = store.ensure_session(title="Chat 1")

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete.")
    )
    second = controller.new_session(title="Chat 2")

    # `new_session()`'s clear call targets the just-created session (always
    # a no-op, since it starts idle) -- `first`'s own COMPLETED state is
    # untouched by creating a sibling.
    assert controller.run_state_for(first.id).status is ConsoleRunStatus.COMPLETED
    assert controller.run_state_for(first.id).visible_copy == "Response complete."
    assert controller.run_state_for(second.id).status is ConsoleRunStatus.IDLE

    # Leaving `second` (idle, non-terminal) for `first`: nothing to clear,
    # so the facade now shows `first`'s still-untouched COMPLETED state.
    controller.switch_session(first.id)
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED

    # Put the CURRENTLY ACTIVE session (`first`) into a terminal state, then
    # leave it for `second`: this is the case `switch_session` actually
    # clears -- the session being LEFT, because it was terminal.
    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.BLOCKED, "Provider blocked.")
    )
    controller.switch_session(second.id)

    assert controller.run_state_for(first.id).status is ConsoleRunStatus.IDLE
    assert controller.run_state_for(first.id).visible_copy == ""
    # `second` (the session arrived at) was never targeted by this switch.
    assert controller.run_state_for(second.id).status is ConsoleRunStatus.IDLE


def test_controller_session_changes_preserve_active_run_copy() -> None:
    """A non-terminal run state is never reset by session-change cleanup --
    `_clear_terminal_run_state`'s guard only fires for TERMINAL statuses.

    Parallel-agents spec §2: run state is per-session now, so this checks
    `first`'s OWN state survives a sibling session being created (rather
    than asserting the facade -- which after `new_session()` is viewing the
    brand-new sibling, not `first` -- shows the same value).
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    first = store.ensure_session(title="Chat 1")

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
    )
    second = controller.new_session(title="Chat 2")

    assert controller.run_state_for(first.id).status is ConsoleRunStatus.STREAMING
    assert controller.run_state_for(first.id).visible_copy == "Streaming response."

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.VALIDATING, "Validating provider."),
        session_id=second.id,
    )
    controller.switch_session(first.id)

    # Leaving `second` (VALIDATING -- non-terminal) for `first`: nothing to
    # clear either way, so `first`'s own untouched STREAMING state is what
    # the facade shows back.
    assert controller.run_state.status is ConsoleRunStatus.STREAMING
    assert controller.run_state.visible_copy == "Streaming response."
    # `second`'s own non-terminal state also survives being left.
    assert controller.run_state_for(second.id).status is ConsoleRunStatus.VALIDATING


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
    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    await controller.submit_draft("hello")

    senders = [m["sender"] for m in persistence.created_messages]
    assert "user" in senders
    assert len(persistence.created_conversations) == 1


@pytest.mark.asyncio
async def test_skill_refuse_after_preparation_removes_transient_echo():
    """A preaccept refusal removes only the preparation's transient USER echo."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    async def _refuse(messages):
        return messages, "Refused: untrusted skill.", (), (), ""

    controller._apply_skill_substitution = _refuse

    result = await controller.submit_draft("run /evil")

    assert result.accepted is False
    messages = store.messages_for_session(store.active_session_id)
    assert all(message.role.value != "user" for message in messages)
    assert store.preparation_for_session(store.active_session_id) is None


@pytest.mark.asyncio
async def test_dictionary_apply_raise_after_preparation_removes_transient_echo():
    """A composition error removes its volatile preparation and transient echo."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    async def _boom(messages, session_id):
        raise RuntimeError("dict boom")

    controller._apply_chat_dictionaries = _boom

    with pytest.raises(RuntimeError):
        await controller.submit_draft("hello")

    messages = store.messages_for_session(store.active_session_id)
    assert all(message.role.value != "user" for message in messages)
    assert store.preparation_for_session(store.active_session_id) is None


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
async def test_character_dispatch_shares_active_pack_prompt_and_capture_snapshot(
    tmp_path,
):
    db = CharactersRAGDB(tmp_path / "controller-emote.db", "controller-emote")
    try:
        character_id = int(db.add_character_card({"name": "Emote actor"}))
        graph = _activate_character_emote_pack(db, character_id)
        store = ConsoleChatStore(persistence=ChatPersistenceService(db))
        session = store.create_session(
            settings=ConsoleSessionSettings(
                provider="llama_cpp",
                system_prompt="Stay in character.",
            ),
            assistant_kind="character",
            assistant_id=str(character_id),
            character_id=character_id,
        )
        gateway = CharacterEmoteStreamingGateway(
            "Emote: sm",
            "ug\nVisible answer",
        )
        controller = ConsoleChatController(
            store=store,
            provider_gateway=gateway,
            system_prompt="Stay in character.",
        )

        result = await controller.submit_draft("hello", session_id=session.id)

        assert result.accepted is True
        assert gateway.messages_seen[0]["role"] == "system"
        prompt = gateway.messages_seen[0]["content"]
        assert prompt.startswith("Stay in character.\n\n")
        assert "Prefer these available states: smug, happy." in prompt
        assert "Never expose this label" not in prompt
        assert "Nor this one" not in prompt
        assert session.settings.system_prompt == "Stay in character."
        completed = store.messages_for_session(session.id)[-1]
        assert completed.content == "Visible answer"
        assert completed.metadata.character_emote.pack_id == graph["pack"]["id"]
        assert (
            completed.metadata.character_emote.pack_version_id
            == graph["version"]["id"]
        )
        assert completed.metadata.character_emote.expression_key == "custom:smug"
        smug_asset = next(
            asset for asset in graph["assets"] if asset["expression_key"] == "custom:smug"
        )
        assert completed.metadata.character_emote.asset_id == smug_asset["id"]
    finally:
        db.close_connection()


def test_emote_snapshot_projection_normalizes_each_asset_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TASK-22227: the per-send snapshot build is O(assets), not O(assets^2).

    The retired implementation re-projected a singleton tuple per state
    against every raw asset (~1,700 regex-bearing normalize calls for a
    40-asset pack); the lookup now normalizes each asset exactly once.
    """

    import tldw_chatbook.Character_Chat.emote_directives as emote_directives_module

    calls = {"count": 0}
    real_normalize_state = emote_directives_module.normalize_character_emote_state
    real_normalize_key = emote_directives_module.normalize_expression_key

    def counting_state(value):
        calls["count"] += 1
        return real_normalize_state(value)

    def counting_key(value):
        calls["count"] += 1
        return real_normalize_key(value)

    monkeypatch.setattr(
        emote_directives_module, "normalize_character_emote_state", counting_state
    )
    monkeypatch.setattr(
        emote_directives_module, "normalize_expression_key", counting_key
    )

    asset_count = 40
    graph = {
        "pack": {"id": 11},
        "version": {"id": 13},
        "assets": [
            {"expression_key": f"custom:state_{index:02d}", "id": index + 1}
            for index in range(asset_count)
        ],
    }
    authority = controller_module._CharacterEmoteAuthority(
        identity_revision=1,
        runtime_backend="direct",
        assistant_id="7",
        assistant_authority_id="7",
        local_character_id=7,
    )

    snapshot = ConsoleChatController._build_character_emote_snapshot(
        authority, graph, fallback_reason="no_active_pack"
    )

    assert snapshot.states == tuple(
        f"state_{index:02d}" for index in range(asset_count)
    )
    assert [asset.asset_id for asset in snapshot.assets] == list(
        range(1, asset_count + 1)
    )
    assert calls["count"] <= 2 * asset_count + 8


@pytest.mark.asyncio
async def test_server_character_without_local_pack_still_sanitizes_controls():
    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp", system_prompt=""),
        runtime_backend="server",
        assistant_kind="character",
        assistant_id="server-character-id",
        assistant_authority_id="server-profile",
    )
    gateway = CharacterEmoteStreamingGateway("Emote: happy\nHello")
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    await controller.submit_draft("hello", session_id=session.id)

    assert gateway.messages_seen[0]["role"] == "system"
    assert gateway.messages_seen[0]["content"].startswith(
        "When the character expression should change"
    )
    completed = store.messages_for_session(session.id)[-1]
    assert completed.content == "Hello"
    assert completed.metadata.character_emote.mood_label == "happy"
    assert completed.metadata.character_emote.fallback_reason == "no_active_pack"


@pytest.mark.asyncio
async def test_generic_dispatch_does_not_arm_character_emote_protocol():
    store = ConsoleChatStore()
    gateway = CharacterEmoteStreamingGateway("Emote: happy\nHello")
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    await controller.submit_draft("hello")

    assert gateway.messages_seen == [{"role": "user", "content": "hello"}]
    completed = store.messages_for_session(store.active_session_id)[-1]
    assert completed.content == "Emote: happy\nHello"
    assert completed.metadata is None


@pytest.mark.asyncio
async def test_character_pack_read_failure_is_content_free_and_fail_soft():
    class RaisingRepository:
        def get_active_actor_pack(self, actor_kind, actor_id):
            raise RuntimeError("secret repository detail")

    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp"),
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
    )
    gateway = CharacterEmoteStreamingGateway("Emote: happy\nHello")
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller._visual_identity_repository = RaisingRepository()

    result = await controller.submit_draft("hello", session_id=session.id)

    assert result.accepted is True
    completed = store.messages_for_session(session.id)[-1]
    assert completed.content == "Hello"
    assert completed.metadata.character_emote.actor_id == 7
    assert completed.metadata.character_emote.fallback_reason == "resolver_error"


@pytest.mark.asyncio
async def test_character_retry_without_chunks_preserves_prior_emote_metadata():
    class FailingCharacterEmoteGateway(StreamingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            yield "Emote: sad\nPartial"
            raise RuntimeError("provider failed after one chunk")

    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp"),
        runtime_backend="server",
        assistant_kind="character",
        assistant_id="server-character-id",
        assistant_authority_id="server-profile",
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=FailingCharacterEmoteGateway(),
    )
    await controller.submit_draft("hello", session_id=session.id)
    failed = _last_failed_assistant(store, session.id)
    prior_metadata = failed.metadata

    controller.provider_gateway = EmptyStreamingGateway()
    await controller.retry_message(failed.id)

    after = store.get_message(failed.id)
    assert after.content == "Partial"
    assert after.metadata == prior_metadata


@pytest.mark.asyncio
async def test_character_snapshot_retries_when_actor_changes_during_pack_read():
    started = threading.Event()
    release = threading.Event()

    class BlockingRepository:
        def get_active_actor_pack(self, actor_kind, actor_id):
            assert actor_kind == "character"
            if actor_id == 7:
                started.set()
                assert release.wait(2)
                state = "old_state"
                identity = 70
            else:
                state = "new_state"
                identity = 80
            return {
                "pack": {"id": identity},
                "version": {"id": identity + 1},
                "assets": [
                    {
                        "id": identity + 2,
                        "expression_key": f"custom:{state}",
                    }
                ],
            }

    store = ConsoleChatStore()
    session = store.create_session(
        settings=ConsoleSessionSettings(provider="llama_cpp"),
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
    )
    gateway = CharacterEmoteStreamingGateway("Emote: new_state\nHello")
    controller = ConsoleChatController(store=store, provider_gateway=gateway)
    controller._visual_identity_repository = BlockingRepository()

    task = asyncio.create_task(controller.submit_draft("hello", session_id=session.id))
    for _attempt in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0)
    assert started.is_set()
    session.assistant_id = "8"
    session.character_id = 8
    session.identity_revision += 1
    release.set()

    result = await task

    assert result.accepted is True
    prompt = gateway.messages_seen[0]["content"]
    assert "new_state" in prompt
    assert "old_state" not in prompt
    completed = store.messages_for_session(session.id)[-1]
    assert completed.content == "Hello"
    assert completed.metadata.character_emote.actor_id == 8
    assert completed.metadata.character_emote.pack_id == 80


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

        async def stream_chat(self, resolution, messages, **kwargs):
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
    controller._set_run_state(
        ConsoleRunState(
            ConsoleRunStatus.STREAMING,
            "Streaming response.",
        )
    )
    # No entry registered in `_active_assistant_message_ids` for this
    # session -- `stop_active_run` must fall back to the visible streaming
    # assistant message in the store.
    assert session.id not in controller._active_assistant_message_ids

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

        async def stream_chat(self, resolution, messages, **kwargs):
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

        async def stream_chat(self, resolution, messages, **kwargs):
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

        async def stream_chat(self, resolution, messages, **kwargs):
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

        async def stream_chat(self, resolution, messages, **kwargs):
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
    assert controller._active_stream_tasks.get(store.active_session_id) is None


@pytest.mark.asyncio
async def test_shutdown_ignores_failed_active_stream_task():
    async def fail_before_shutdown():
        raise RuntimeError("stream task failed before shutdown")

    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    task = asyncio.create_task(fail_before_shutdown())
    await asyncio.sleep(0)
    assert task.done()

    controller._active_stream_tasks[session.id] = task
    controller._stop_requested = True

    await controller.shutdown()

    assert controller._active_stream_tasks == {}
    assert controller._stop_requested is False


@pytest.mark.asyncio
async def test_close_streaming_session_stops_run_without_key_error():
    class WaitingGateway(StreamingGateway):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
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
    # No session is active anymore (the only session was just closed), so
    # the `run_state` FACADE (keyed by the now-None active_session_id) is no
    # longer meaningful here -- check the closed session's own recorded
    # state instead, proving the stop was actually processed rather than
    # silently swallowed by a KeyError.
    assert store.active_session_id is None
    assert controller.run_state_for(session_id).status is ConsoleRunStatus.STOPPED


@pytest.mark.asyncio
async def test_close_streaming_session_result_does_not_set_dispatch_gap_toast_flag():
    """Task 4 fix-round-2 (I2): mid-run `_session_closed_result` sites
    (~19 of ~20, reached when the user closes a session they are actively
    viewing/streaming -- this scenario mirrors
    ``test_close_streaming_session_stops_run_without_key_error`` above
    exactly) must NOT set ``session_closed`` -- that session's run state
    already went STOPPED and the close was a deliberate, already-
    acknowledged user action, so the screen's dispatch-gap toast firing here
    too would be a redundant, confusing second signal. Only ``submit_draft``'s
    own dispatch-gap call site (the DISPATCHED session closing before the
    worker got a chance to run at all -- no other signal exists there) sets
    it."""

    class WaitingGateway(StreamingGateway):
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
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

    controller.close_session(session_id)
    gateway.release.set()
    result = await asyncio.wait_for(task, timeout=0.5)

    assert result.accepted is True
    assert result.visible_copy == "Session closed."
    assert result.session_closed is False


@pytest.mark.asyncio
async def test_submit_draft_dispatch_gap_session_closed_sets_toast_flag_with_informative_copy():
    """Task 4 fix-round-2 (I2/M2): the ONE call site that should toast --
    ``submit_draft``'s own dispatch-gap branch, where the session captured
    at DISPATCH time was closed before this coroutine got a chance to run
    (the exact scenario ``test_submit_draft_closed_session_id_fails_closed_
    without_touching_active`` in test_console_run_state_per_session.py
    already pins for ``accepted``/``visible_copy`` byte-identically) -- must
    set ``session_closed`` AND use the INFORMATIVE copy, not the generic
    "Session closed." every other call site uses."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())

    session_a = store.ensure_session(title="A")
    closed_session_id = session_a.id
    controller.new_session(title="B")
    controller.close_session(closed_session_id)

    result = await controller.submit_draft("hello", session_id=closed_session_id)

    assert result.accepted is True
    assert result.session_closed is True
    assert (
        result.visible_copy == "Console session closed before your message could send."
    )


@pytest.mark.asyncio
async def test_retry_message_active_run_rejection_does_not_append_system_row():
    """Task 4 fix-round-2 (I1): ``_active_run_rejection``'s SYSTEM-row
    append is scoped to ``submit_draft`` alone (``append_row=True``) --
    ``retry_message`` (like ``continue_from_message``/``regenerate_message``/
    ``summarize_up_to``/``edit_and_resend_message``) already toasts this
    exact copy via its own screen-level wrapper (TASK-232's mid-run gate,
    see Tests/UI/test_console_run_gate.py), so the controller must stay
    silent here or the user would see the identical rejection reported
    twice."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    pending = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    failed = store.mark_message_failed(pending.id)

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.STREAMING, "already streaming")
    )

    result = await controller.retry_message(failed.id)

    assert result.accepted is False
    assert "already running in this tab" in result.visible_copy
    messages = store.messages_for_session(session.id)
    system_messages = [m for m in messages if m.role is ConsoleMessageRole.SYSTEM]
    assert system_messages == []


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
        async def stream_chat(self, resolution, messages, **kwargs):
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
async def test_leading_greeting_folds_into_system_row_not_message_array():
    """A seeded character greeting (persisted ASSISTANT message before any
    user turn) must reach the provider inside the SYSTEM row, never as an
    assistant-first message -- strict providers (Anthropic, Gemini) reject an
    assistant-first message array (task-427), but dropping the greeting
    entirely made the model contradict the transcript (task-1531)."""
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
    # The greeting arrives via a system row even without a session prompt.
    assert sent[0]["role"] == "system"
    assert "Greetings, traveler." in sent[0]["content"]
    # The message array itself stays user-first with no assistant greeting.
    rest = sent[1:]
    assert rest[0]["role"] == "user"
    assert all("Greetings, traveler." not in (m.get("content") or "") for m in rest)


@pytest.mark.asyncio
async def test_leading_greeting_appends_to_existing_system_prompt():
    """The greeting fold appends to a configured system prompt; the prompt
    itself stays verbatim at the start of the system row."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        system_prompt="Stay in character.",
    )
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
    assert sent[0]["role"] == "system"
    assert sent[0]["content"].startswith("Stay in character.")
    assert "Greetings, traveler." in sent[0]["content"]
    assert [m["role"] for m in sent[1:]] == ["user"]


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

    async def stream_chat(self, resolution, messages, **kwargs):
        yield "ok"


def _auto_title_controller() -> ConsoleChatController:
    return ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=_AutoTitleReadyGateway(),
    )


@pytest.mark.asyncio
async def test_submit_draft_auto_titles_default_session_from_first_message():
    controller = _auto_title_controller()
    session = controller.new_session(ephemeral=True)
    assert session.title == "Chat 1"

    await controller.submit_draft("fix the login bug in the auth flow")

    assert controller.store.sessions()[0].title == "fix the login bug in the au..."


@pytest.mark.asyncio
async def test_submit_draft_preserves_user_renamed_session_title():
    controller = _auto_title_controller()
    session = controller.new_session(ephemeral=True)
    controller.store.rename_session(session.id, "My research thread")

    await controller.submit_draft("hello there")

    assert controller.store.sessions()[0].title == "My research thread"


@pytest.mark.asyncio
async def test_submit_draft_does_not_retitle_after_first_send():
    controller = _auto_title_controller()
    controller.new_session(ephemeral=True)

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
    # str(exc) alone was empty in the live failure ("[failed]"); the copy must
    # never be blank. FB-06 (task-2154.16): the generic fallback is a plain
    # category -- the exception class name must NOT reach user copy.
    empty_detail = describe_stream_failure(RuntimeError())
    assert empty_detail == "unexpected provider error"
    with_detail = describe_stream_failure(RuntimeError("llama.cpp stream failed"))
    assert with_detail == "unexpected provider error (llama.cpp stream failed)"


def test_describe_stream_failure_never_leaks_exception_class_names():
    """FB-06 (task-2154.16): generic Exception subclasses map to a plain
    category; useful detail (connection refused, URL) is preserved."""
    from tldw_chatbook.Chat.console_chat_controller import describe_stream_failure

    class LlamaCppSDKError(Exception):
        """Stand-in for a provider SDK's own error type."""

    for exc in (
        RuntimeError(
            "Connection refused: llama.cpp server not reachable at http://127.0.0.1:9099"
        ),
        ValueError("bad chunk encoding"),
        LlamaCppSDKError("weird sdk state"),
    ):
        copy = describe_stream_failure(exc)
        assert type(exc).__name__ not in copy
        assert copy.startswith("unexpected provider error")
        # The actionable detail survives the sanitization.
        assert str(exc) in copy

    # Empty-detail generic exceptions still produce non-empty copy.
    for exc in (RuntimeError(), ValueError(), LlamaCppSDKError()):
        assert describe_stream_failure(exc) == "unexpected provider error"


@pytest.mark.asyncio
async def test_active_session_stream_failure_fires_failure_toast_once():
    """FB-05 (task-2154.16): the VIEWED session's stream failure raises an
    ambient toast carrying the same copy as the transcript system row."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )
    toasts: list[str] = []
    controller.notify_run_failure = toasts.append

    result = await controller.submit_draft("hello")

    assert result.accepted is True
    system_row = store.messages_for_session(store.active_session_id)[-1]
    assert system_row.role is ConsoleMessageRole.SYSTEM
    assert toasts == [system_row.content]
    assert toasts[0].startswith("Provider stream failed:")


@pytest.mark.asyncio
async def test_active_session_failure_toast_not_refired_on_terminal_restamp():
    """FB-05 once-guard: re-stamping an already-terminal FAILED status must
    not re-toast (mirrors notify_run_outcome's transition guard)."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=FailingStreamingGateway()
    )
    toasts: list[str] = []
    controller.notify_run_failure = toasts.append
    await controller.submit_draft("hello")
    assert len(toasts) == 1

    controller._set_run_state(
        ConsoleRunState(ConsoleRunStatus.FAILED, "Provider stream failed: restamp"),
        session_id=store.active_session_id,
    )
    assert len(toasts) == 1


@pytest.mark.asyncio
async def test_active_session_success_stays_silent():
    """FB-05 scope: only failures toast on the viewed session (FB-07's
    positive-feedback gap is task-2154.17, not this one)."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    toasts: list[str] = []
    controller.notify_run_failure = toasts.append

    result = await controller.submit_draft("hello")

    assert result.accepted is True
    assert toasts == []


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
        async def stream_chat(self, resolution, messages, **kwargs):
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


def test_is_empty_transcript_row_tolerates_a_metadata_object_without_the_attribute():
    """Qodo Q5 (task-2391 review): the docstring promised duck-typed safety
    via `getattr` on `.metadata`, but the INNER `.transcript_status` read
    was a plain attribute access -- a metadata object that duck-types some
    fields but not that one raised `AttributeError` there. This helper runs
    on every row of three model-facing send paths (`_provider_message_
    payloads`, `summarize_up_to`, `impersonate_user_reply`), so that crash
    reached the main send path, not just a narrow test double."""
    from tldw_chatbook.Chat.console_chat_controller import _is_empty_transcript_row

    message = SimpleNamespace(
        role=ConsoleMessageRole.USER,
        content="hello",
        status="complete",
        metadata=SimpleNamespace(engine="realtime"),  # no transcript_status
    )

    assert _is_empty_transcript_row(message) is False


def test_provider_payloads_exclude_an_empty_transcript_placeholder():
    """task-2391 fix-now: a committed voice turn whose transcript came back
    empty persists real placeholder CONTENT ("(no speech detected)") so the
    row can survive a restart -- but that content is UI chrome written so
    the row could exist at all, not something the user said, and must
    never reach a provider as if it were a real turn. Before this fix,
    `_provider_message_payloads` had no `transcript_status` awareness, so
    the placeholder rode straight through `_emit` into `{"role": "user",
    "content": "(no speech detected)"}` on every ordinary send/retry/edit/
    fork built off this session (`_provider_messages_for_session` backs all
    of them) -- a fabricated user turn, permanently, for the life of the
    conversation. An ordinary user row in the same session must still ride
    through untouched."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
    )

    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=StreamingGateway(), model="test-model"
    )
    session = store.ensure_session()
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content=CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
        metadata=MessageMetadata(engine="realtime", transcript_status="empty"),
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="a real question",
    )

    messages = store.messages_for_session(session.id)
    payloads = controller._provider_message_payloads(messages, skip_failed=True)

    contents = [payload["content"] for payload in payloads]
    assert CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER not in contents, (
        "the empty-transcript placeholder must never be narrated to the "
        "model as if the user said it"
    )
    assert "a real question" in contents


@pytest.mark.asyncio
async def test_impersonate_excludes_an_empty_transcript_placeholder():
    """task-2391 fix-now (audit follow-up): `impersonate_user_reply` hand-
    rolls its own transcript builder rather than reusing
    `_provider_message_payloads` (its own comment says "mirror ... rules
    exactly"), so the payload fix alone did not cover it. This prompt
    explicitly asks the model to draft the user's NEXT message "in their
    voice" from this exact transcript -- a fabricated empty-transcript
    placeholder here is arguably worse than in the ordinary send path."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
    )

    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, model="test-model"
    )
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hello")
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content=CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
        metadata=MessageMetadata(engine="realtime", transcript_status="empty"),
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="hi there"
    )

    await controller.impersonate_user_reply(session.id)

    assert gateway.messages_seen is not None, "the completion must still run"
    blob = " ".join(str(m["content"]) for m in gateway.messages_seen)
    assert CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER not in blob
    assert "hello" in blob
    assert "hi there" in blob


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


def test_provider_messages_for_next_send_estimate_uses_lightweight_projection_without_media_serialization(
    monkeypatch,
):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    monkeypatch.setattr(controller_module, "max_history_images", lambda p, m: 1)
    monkeypatch.setattr(
        controller_module,
        "image_url_part",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("estimate serialized media")
        ),
    )
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=StreamingGateway(),
        model="vision-model",
        system_prompt="system",
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.SYSTEM, content="transcript system"
    )
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="hello")
    failed = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="failed"
    )
    store.mark_message_send_blocked(failed.id)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="(no speech detected)",
        metadata=MessageMetadata(engine="realtime", transcript_status="empty"),
    )
    disallowed = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="not admitted"
    )
    store._message_or_raise(disallowed.id).assistant_generation_state = "failed"
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="older",
        attachments=(MessageAttachment(b"old", "image/png", "old.png", 0),),
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="answer"
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="newer",
        attachments=(MessageAttachment(b"new", "image/png", "new.png", 0),),
    )
    live = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(live.id, "buffered answer")
    live_content = store._message_or_raise(live.id).content
    revisions = (
        dict(store._stream_materialized_counts),
        dict(store._payload_revisions),
        dict(store._message_speech_revisions),
    )

    result = controller.provider_messages_for_next_send_estimate(session.id)

    assert result == ConsoleNextSendHistoryProjection(
        rows=(
            (
                "system",
                "system\n\nYou already opened this conversation with the following "
                "message, which the user has seen:\nhello",
            ),
            ("user", "older"),
            ("assistant", "answer"),
            ("user", "newer"),
            ("assistant", "buffered answer"),
        ),
        historical_media_count=1,
    )
    assert store._message_or_raise(live.id).content == live_content == ""
    assert revisions == (
        dict(store._stream_materialized_counts),
        dict(store._payload_revisions),
        dict(store._message_speech_revisions),
    )


def test_provider_message_payloads_serializes_only_after_lightweight_projection(
    monkeypatch,
):
    monkeypatch.setattr(controller_module, "is_vision_capable", lambda p, m: True)
    monkeypatch.setattr(controller_module, "max_history_images", lambda p, m: 1)
    calls = []

    def _serialize(data, mime_type):
        calls.append((data, mime_type))
        return {"type": "image_url", "image_url": {"url": "serialized"}}

    monkeypatch.setattr(controller_module, "image_url_part", _serialize)
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(
        store=store, provider_gateway=StreamingGateway(), model="vision-model"
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="look",
        attachments=(MessageAttachment(b"image", "", "image.png", 0),),
    )

    payloads = controller._provider_message_payloads(
        store.messages_for_session(session.id), skip_failed=True
    )

    assert calls == [(b"image", "image/png")]
    assert payloads == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "serialized"}},
            ],
        }
    ]


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


#: PR2a Task 5: the review hooks take the id of the run whose batch they
#: are reviewing, and every gate mutation they make is scoped to it. These
#: hook-level tests each drive ONE run, so they name it once here -- their
#: assertions are unchanged (what a run stamps is what that run reads);
#: cross-run isolation is pinned by `Tests/Agents/test_gate_run_scoping.py`.
RUN = "run-1"


class _FakeReviewProvider:
    """Stands in for `MCPToolProvider` in `build_mcp_review_hook` unit tests."""

    def __init__(self, gated_names: set[str]) -> None:
        self._gated_names = gated_names
        self.apply_batch_decisions_calls: list[dict[str, str]] = []
        self._stamped: dict[tuple[str, str], str] = {}

    def pending_gate_for(
        self, name: str, args: dict, call_id: str = ""
    ) -> MCPPendingCall | None:
        if name not in self._gated_names:
            return None
        return MCPPendingCall(
            llm_name=name,
            server_key="local:srv",
            tool_name=name,
            server_label="Srv",
            arguments=dict(args or {}),
            # TASK-1861: the double must mirror the real provider, which
            # carries the per-call key so the card can offer one decision per
            # TARGET instead of one per tool name.
            call_id=call_id,
            reason="ask",
        )

    def apply_batch_decisions(self, run_id: str, decisions: dict[str, str]) -> None:
        self.apply_batch_decisions_calls.append(dict(decisions))
        # Mirrors MCPToolProvider.apply_batch_decisions' REPLACE semantics
        # (not merge) -- see that method's own docstring (Finding F1) --
        # and, since PR2a Task 5, that the replace is scoped to ONE run:
        # other runs' slices survive.
        self._stamped = {
            key: value for key, value in self._stamped.items() if key[0] != run_id
        }
        for name, verdict in (decisions or {}).items():
            self._stamped[(run_id, name)] = verdict

    def stamped_decision(self, run_id: str, name: str) -> str | None:
        return self._stamped.get((run_id, name))


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
    verdicts = hook(calls, RUN)

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

    verdicts = hook(calls, RUN)

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

    verdicts = hook(calls, RUN)

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
    hook([ToolCall(name="mcp__srv__run", args={}, call_id="1")], RUN)
    assert provider.stamped_decision(RUN, "mcp__srv__run") == "approve_once"

    # Turn 2: same tool, but request_mcp_approvals now raises mid-round-trip.
    def _raise(pending):
        raise RuntimeError("shutdown mid round-trip")

    hook2 = build_mcp_review_hook(provider, _raise)
    with pytest.raises(RuntimeError):
        hook2([ToolCall(name="mcp__srv__run", args={}, call_id="2")], RUN)

    # No stale stamp from turn 1 must survive the raise for invoke() to peek.
    assert provider.stamped_decision(RUN, "mcp__srv__run") is None


# ---------------------------------------------------------------------------
# build_tool_review_hook (task-545/T6: run-level hook, gates built-ins even
# with no MCP provider composed for the run)
# ---------------------------------------------------------------------------


class _FakeBuiltinGate:
    """Minimal stand-in for `BuiltinToolGate` in `build_tool_review_hook` tests."""

    def __init__(self, state: str = "ask", risk_floored: bool = True) -> None:
        self._state = state
        self._floored = risk_floored
        self.turns = 0
        self.stamped: list[tuple[str, str]] = []

    def begin_turn(self, run_id: str) -> None:
        self.turns += 1

    def resolve(self, tool) -> EffectiveToolState:
        return EffectiveToolState(
            state=self._state,
            origin="builtin_default",
            risk_floored=self._floored,
        )

    def stamp(self, run_id: str, name: str, decision: str) -> None:
        self.stamped.append((name, decision))

    def is_session_approved(self, name: str) -> bool:
        # Every existing test drives a single turn with a fresh gate and
        # never expects a session approval to already be live -- real
        # session tracking is covered separately by
        # `test_approve_for_session_is_not_re_prompted_next_turn`, which
        # uses the REAL `BuiltinToolGate` instead of this fake.
        return False


class _FakeBuiltinProvider:
    """Minimal stand-in for `BuiltinToolProvider` -- only `.tool_for` is used."""

    def __init__(self, tool) -> None:
        self._tool = tool

    def tool_for(self, name: str):
        return self._tool if name == self._tool.name else None


class _FakeMutatingTool:
    """A `Tool`-shaped double; `BuiltinToolGate.resolve` never inspects it
    beyond identity in these tests (the fake gate ignores its argument), so
    only `.name` needs to be real."""

    name = "write_thing"


def _builtin_call(name: str) -> ToolCall:
    # ToolCall is (name, args, call_id) -- there is NO llm_name on it (that
    # belongs to MCPPendingCall, the approval-row type). The verdict map
    # the runtime consumes is keyed by the LLM-facing name, which equals
    # ToolCall.name.
    return ToolCall(name=name, args={})


def test_review_hook_gates_builtins_with_no_mcp_provider():
    """The whole point of T6: a user with no MCP servers must still be gated."""
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    gate = _FakeBuiltinGate()
    asked: dict[str, list[MCPPendingCall]] = {}

    def request_approvals(pending: list[MCPPendingCall]) -> dict[str, str]:
        asked["pending"] = pending
        return {p.llm_name: "approve_once" for p in pending}

    hook = build_tool_review_hook(
        gate, _FakeBuiltinProvider(_FakeMutatingTool()), None, request_approvals
    )
    verdicts = hook([_builtin_call("write_thing")], RUN)

    assert gate.turns == 1  # begin_turn ran first
    assert gate.stamped == [("write_thing", "approve_once")]
    # Rows are MCPPendingCall dataclasses (what request_mcp_approvals takes),
    # NOT dicts -- the dict conversion happens inside it.
    row = asked["pending"][0]
    assert row.server_key == "agent:builtin"
    assert row.server_label == "Built-in"
    assert row.reason == "risk_floored"
    # Exclude ONLY always_allow -- deny is a turn-scoped refusal, not a
    # persistent write, so it must stay offered (spec correction 0e6e8a56d).
    assert row.options == ("approve_once", "approve_session", "deny")
    assert verdicts == {"write_thing": "proceed"}


def _file_tool(name: str):
    """A `Tool`-shaped double carrying a REAL file-tool name (read_file/
    list_directory/write_file), so `path_precheck_failed` (looked up by
    exact tool name) recognizes it. Unlike `_FakeMutatingTool`/`write_thing`
    above, this name must be one of the three file tools for the
    precheck to ever fire.
    """
    return type("_FakeFileTool", (), {"name": name})()


def test_review_hook_flags_read_file_path_outside_roots(monkeypatch, tmp_path):
    """TASK-1231/F3 AC2: a read_file row whose path the roots check will
    reject must carry `path_precheck_failed=True` -- a WARNING only. The
    row must still be offered every normal decision (never auto-denied);
    the user can still approve it.
    """
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
    from tldw_chatbook.Tools import file_operation_tools as fot
    from tldw_chatbook.Tools import workspace_file_roots as wfr

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())

    def _raise():
        raise RuntimeError("no workspace registry in this test")

    monkeypatch.setattr(wfr, "_registry_factory", _raise)

    outside = tmp_path / "outside.txt"
    outside.write_text("x")

    gate = _FakeBuiltinGate()
    asked: dict[str, list[MCPPendingCall]] = {}

    def request_approvals(pending: list[MCPPendingCall]) -> dict[str, str]:
        asked["pending"] = pending
        return {p.llm_name: "approve_once" for p in pending}

    hook = build_tool_review_hook(
        gate, _FakeBuiltinProvider(_file_tool("read_file")), None, request_approvals
    )
    verdicts = hook([ToolCall(name="read_file", args={"file_path": str(outside)})], RUN)

    row = asked["pending"][0]
    assert row.path_precheck_failed is True
    # Never auto-denied: still offered every normal decision, and still
    # proceeds if the user approves anyway.
    assert row.options == ("approve_once", "approve_session", "deny")
    assert verdicts == {"read_file": "proceed"}


def test_review_hook_does_not_flag_read_file_path_inside_roots(monkeypatch, tmp_path):
    """Counterpart to the above: a path the roots check WOULD accept must
    not carry the warning."""
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
    from tldw_chatbook.Tools import file_operation_tools as fot
    from tldw_chatbook.Tools import workspace_file_roots as wfr

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())

    def _raise():
        raise RuntimeError("no workspace registry in this test")

    monkeypatch.setattr(wfr, "_registry_factory", _raise)

    inside = sandbox / "notes.txt"
    inside.write_text("x")

    gate = _FakeBuiltinGate()
    asked: dict[str, list[MCPPendingCall]] = {}

    def request_approvals(pending: list[MCPPendingCall]) -> dict[str, str]:
        asked["pending"] = pending
        return {p.llm_name: "approve_once" for p in pending}

    hook = build_tool_review_hook(
        gate, _FakeBuiltinProvider(_file_tool("read_file")), None, request_approvals
    )
    hook([ToolCall(name="read_file", args={"file_path": str(inside)})], RUN)

    row = asked["pending"][0]
    assert row.path_precheck_failed is False


def test_review_hook_leaves_non_file_builtins_unflagged():
    """Scope guard (AC2): only read_file/list_directory/write_file are ever
    pre-flighted -- every other builtin tool's row stays False regardless
    of its arguments."""
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    gate = _FakeBuiltinGate()
    asked: dict[str, list[MCPPendingCall]] = {}

    def request_approvals(pending: list[MCPPendingCall]) -> dict[str, str]:
        asked["pending"] = pending
        return {p.llm_name: "approve_once" for p in pending}

    hook = build_tool_review_hook(
        gate, _FakeBuiltinProvider(_FakeMutatingTool()), None, request_approvals
    )
    hook([_builtin_call("write_thing")], RUN)

    row = asked["pending"][0]
    assert row.path_precheck_failed is False


def _two_workspace_registry(tmp_path):
    """Build a REAL registry with two workspaces, each bound to a DIFFERENT
    folder, and ws-b set ACTIVE. Used by the round-1-review CRITICAL 1
    regression tests below: a fake registry that merely raises (the
    pattern the earlier precheck tests use) cannot exercise `get_active_
    workspace()` resolving the WRONG workspace, since it never reaches
    that far.
    """
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="review-hook-test")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="A")
    registry.create_workspace(workspace_id="ws-b", name="B")
    folder_a = tmp_path / "folder-a"
    folder_b = tmp_path / "folder-b"
    folder_a.mkdir()
    folder_b.mkdir()
    registry.add_folder_binding("ws-a", folder_a)
    registry.add_folder_binding("ws-b", folder_b)
    # The UI happens to be showing ws-b -- a DIFFERENT workspace than the
    # one the reviewed run is actually bound to in every test below.
    registry.set_active_workspace("ws-b")
    return registry, folder_a, folder_b


def test_review_hook_precheck_uses_the_runs_workspace_not_the_active_one(
    monkeypatch, tmp_path
):
    """Round 1 review CRITICAL 1: `path_precheck_failed` (threaded through
    `build_tool_review_hook`'s `workspace_id` param) must resolve THIS RUN's
    OWN workspace -- never whatever workspace the UI happens to have
    active, which can differ for a parked/background session's approval
    round. A path inside the RUN's workspace's (ws-a) bound folder must not
    warn, even though a DIFFERENT workspace (ws-b, with no binding covering
    this path) is the one currently active.
    """
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
    from tldw_chatbook.Tools import file_operation_tools as fot
    from tldw_chatbook.Tools import workspace_file_roots as wfr

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    registry, folder_a, _folder_b = _two_workspace_registry(tmp_path)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)

    target_in_a = folder_a / "notes.txt"
    target_in_a.write_text("x")

    gate = _FakeBuiltinGate()
    asked: dict[str, list[MCPPendingCall]] = {}

    def request_approvals(pending: list[MCPPendingCall]) -> dict[str, str]:
        asked["pending"] = pending
        return {p.llm_name: "approve_once" for p in pending}

    # workspace_id="ws-a" simulates a session BOUND to ws-a while ws-b is
    # the workspace actually active in the UI.
    hook = build_tool_review_hook(
        gate,
        _FakeBuiltinProvider(_file_tool("read_file")),
        None,
        request_approvals,
        workspace_id="ws-a",
    )
    hook([ToolCall(name="read_file", args={"file_path": str(target_in_a)})], RUN)

    row = asked["pending"][0]
    assert row.path_precheck_failed is False


def test_review_hook_precheck_does_not_fall_back_to_the_active_workspace(
    monkeypatch, tmp_path
):
    """Inverse of the above: a path inside ws-b's (the ACTIVE workspace's)
    folder, while the reviewed run is bound to ws-a (which does NOT cover
    it) -- must WARN. Pre-fix, `path_precheck_failed` never bound a
    workspace at all, so `allowed_file_roots` fell back to `registry.
    get_active_workspace()` (ws-b) and this path would have resolved
    successfully -- a false negative (no warning) for a call that is, in
    fact, doomed against the RUN's real (ws-a) roots.
    """
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
    from tldw_chatbook.Tools import file_operation_tools as fot
    from tldw_chatbook.Tools import workspace_file_roots as wfr

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(fot, "_tool_sandbox_root", lambda: sandbox.resolve())
    registry, _folder_a, folder_b = _two_workspace_registry(tmp_path)
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)

    target_in_b = folder_b / "notes.txt"
    target_in_b.write_text("x")

    gate = _FakeBuiltinGate()
    asked: dict[str, list[MCPPendingCall]] = {}

    def request_approvals(pending: list[MCPPendingCall]) -> dict[str, str]:
        asked["pending"] = pending
        return {p.llm_name: "approve_once" for p in pending}

    hook = build_tool_review_hook(
        gate,
        _FakeBuiltinProvider(_file_tool("read_file")),
        None,
        request_approvals,
        workspace_id="ws-a",
    )
    hook([ToolCall(name="read_file", args={"file_path": str(target_in_b)})], RUN)

    row = asked["pending"][0]
    assert row.path_precheck_failed is True


def test_allow_resolved_builtin_never_prompts():
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    calls: list[list[MCPPendingCall]] = []
    hook = build_tool_review_hook(
        _FakeBuiltinGate(state="allow", risk_floored=False),
        _FakeBuiltinProvider(_FakeMutatingTool()),
        None,
        lambda pending: calls.append(pending) or {},
    )
    assert hook([_builtin_call("write_thing")], RUN) == {}
    assert calls == []  # no card shown


def test_deny_resolved_builtin_is_not_offered_to_the_user():
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    calls: list[list[MCPPendingCall]] = []
    hook = build_tool_review_hook(
        _FakeBuiltinGate(state="deny", risk_floored=False),
        _FakeBuiltinProvider(_FakeMutatingTool()),
        None,
        lambda pending: calls.append(pending) or {},
    )
    hook([_builtin_call("write_thing")], RUN)
    assert calls == []  # a tool that is Off gets no approval card


def test_begin_turn_runs_even_when_approvals_raise():
    """A raising approval path must not leave stale stamps for next turn."""
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    gate = _FakeBuiltinGate()

    def boom(pending):
        raise RuntimeError("ui gone")

    hook = build_tool_review_hook(
        gate, _FakeBuiltinProvider(_FakeMutatingTool()), None, boom
    )
    with pytest.raises(RuntimeError):
        hook([_builtin_call("write_thing")], RUN)
    assert gate.turns == 1


def test_unknown_names_are_returned_unreviewed():
    """Skill tools and native spawn are owned by neither gate."""
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    hook = build_tool_review_hook(
        _FakeBuiltinGate(),
        _FakeBuiltinProvider(_FakeMutatingTool()),
        None,
        lambda pending: {},
    )
    assert hook([_builtin_call("some_skill")], RUN) == {}


def test_mcp_and_builtin_share_one_round_trip():
    """One turn, one MCP call + one built-in call: exactly ONE
    `request_approvals` round trip carrying BOTH rows."""
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    mcp_provider = _FakeReviewProvider(gated_names={"mcp__srv__run"})
    gate = _FakeBuiltinGate()
    round_trips: list[list[MCPPendingCall]] = []

    def _approve(pending: list[MCPPendingCall]) -> dict[str, str]:
        round_trips.append(pending)
        return {row.llm_name: "approve_once" for row in pending}

    hook = build_tool_review_hook(
        gate, _FakeBuiltinProvider(_FakeMutatingTool()), mcp_provider, _approve
    )
    calls = [
        ToolCall(name="mcp__srv__run", args={"x": 1}, call_id="1"),
        _builtin_call("write_thing"),
    ]

    verdicts = hook(calls, RUN)

    assert len(round_trips) == 1
    names_asked = {row.llm_name for row in round_trips[0]}
    assert names_asked == {"mcp__srv__run", "write_thing"}
    assert verdicts == {"mcp__srv__run": "proceed", "write_thing": "proceed"}
    assert mcp_provider.apply_batch_decisions_calls[-1] == {
        "mcp__srv__run": "approve_once"
    }
    assert gate.stamped == [("write_thing", "approve_once")]


class _FakeSessionApprovalService:
    """A minimal `unified_mcp_service`-shaped double exercising the REAL
    `BuiltinToolGate`'s session-approval read/write seam (`approve_for_
    session`/`is_session_approved`) -- deliberately not a fake `Builtin
    ToolGate` itself, so this test proves the actual persistence path
    `BuiltinToolGate.stamp()`/`is_session_approved()` use, not a test
    double's own bookkeeping."""

    def __init__(self) -> None:
        self._approved: set[tuple[str, str]] = set()

    def get_kill_switch(self) -> bool:
        return False

    def approve_for_session(self, server_key: str, tool_name: str) -> None:
        self._approved.add((server_key, tool_name))

    def is_session_approved(self, server_key: str, tool_name: str) -> bool:
        return (server_key, tool_name) in self._approved


class _FakeMutatingRiskyTool:
    """A `Tool`-shaped double whose `risk_tags` actually intersect
    `HIGH_RISK_TAGS`, so the REAL `resolve_builtin_state` floors an
    inherited `allow` to `ask` from an empty (`{}`) permission payload --
    `_FakeMutatingTool` (used by the fake-gate tests above) has no
    `risk_tags`/`description`/`parameters` at all, which is fine for a
    fake gate that never calls `tool_ref()`, but the REAL `BuiltinToolGate.
    resolve()` does call it."""

    name = "write_thing"
    description = "writes a thing"
    parameters = {"type": "object", "properties": {}}
    risk_tags = ("mutates",)


def test_approve_for_session_is_not_re_prompted_next_turn():
    """Review finding 1 (T6 review, Important): `BuiltinToolGate.resolve()`
    reads the permission store ONLY -- never session approvals -- so
    without the hook's own `is_session_approved` skip, a user who picks
    "Approve for session" on turn 1 is silently re-prompted on turn 2 even
    though `invoke()`'s own `check()` already honors that same session
    approval. Drives the REAL `BuiltinToolGate` (not the fake used above)
    against a fake service that actually tracks session approvals, so this
    proves the real persistence path, not a test double's bookkeeping."""
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    service = _FakeSessionApprovalService()
    gate = BuiltinToolGate(service)
    tool = _FakeMutatingRiskyTool()
    provider = _FakeBuiltinProvider(tool)
    round_trips: list[list[MCPPendingCall]] = []

    def approve_session(pending: list[MCPPendingCall]) -> dict[str, str]:
        round_trips.append(pending)
        return {row.llm_name: "approve_session" for row in pending}

    hook = build_tool_review_hook(gate, provider, None, approve_session)

    # Turn 1: no session approval yet -- a card IS shown, and the user
    # approves for session.
    verdict1 = hook([_builtin_call("write_thing")], RUN)
    assert len(round_trips) == 1
    assert round_trips[0][0].llm_name == "write_thing"
    assert verdict1 == {"write_thing": "proceed"}

    # Turn 2: `begin_turn()` clears the turn-scoped `_stamps` dict, but the
    # SESSION approval lives on the fake service, not in `_stamps` -- no
    # second round trip.
    verdict2 = hook([_builtin_call("write_thing")], RUN)
    assert len(round_trips) == 1  # still just the one round trip
    # Nothing needed gating this turn, so the call is absent from the
    # returned map entirely -- purely documentary, exactly like an
    # already-session-approved MCP call today (`build_mcp_review_hook`'s
    # own docstring: `run_agent_loop` defaults any unmentioned name to
    # "proceed").
    assert verdict2 == {}
    # And the call genuinely still proceeds: this is the EXACT verdict
    # `BuiltinToolProvider.invoke()` consults on dispatch.
    assert gate.check(tool, RUN) is None


# ---------------------------------------------------------------------------
# _agent_failure_visible_copy (TASK-1231/F3 AC4, round 1 review Minor)
# ---------------------------------------------------------------------------


def test_agent_failure_visible_copy_avoids_double_lead_in_for_loop_guard():
    """Round 1 review (Minor): `agent_runtime`'s loop-guard summary already
    reads as a complete, user-facing sentence ("Agent stopped: ...") -- this
    must not become "Agent run stuck: Agent stopped: ...".
    """
    from tldw_chatbook.Agents.agent_models import RUN_STUCK, STEP_ERROR

    loop_guard_summary = (
        "Agent stopped: it kept calling calculator with the same "
        "arguments (3 times) without making progress."
    )
    outcome = SimpleNamespace(
        status=RUN_STUCK,
        steps=[SimpleNamespace(kind=STEP_ERROR, summary=loop_guard_summary)],
    )
    copy = ConsoleChatController._agent_failure_visible_copy(outcome)
    assert copy == loop_guard_summary
    assert not copy.startswith("Agent run stuck: Agent stopped")


def test_agent_failure_visible_copy_keeps_prefix_for_budget_reasons():
    """Every other RUN_STUCK reason (budget exhaustion) is not a complete
    sentence on its own -- the "Agent run stuck: " lead-in must stay."""
    from tldw_chatbook.Agents.agent_models import RUN_STUCK, STEP_ERROR

    outcome = SimpleNamespace(
        status=RUN_STUCK,
        steps=[SimpleNamespace(kind=STEP_ERROR, summary="step budget exhausted")],
    )
    copy = ConsoleChatController._agent_failure_visible_copy(outcome)
    assert copy == "Agent run stuck: step budget exhausted."


# -----------------------------------------------------------------------------
# _finalize_agent_reply hardening (task-2)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_agent_reply_empty_final_text_uses_fallback():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="")
    result = await controller._finalize_agent_reply(
        placeholder.id, session.id, outcome, variant_mode=False
    )

    messages = store.messages_for_session(session.id)
    assistant = messages[-1]
    assert assistant.content == "No response was generated."
    assert assistant.status == "complete"
    assert result.accepted is True
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_finalize_agent_reply_missing_placeholder_appends_message():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    fake_id = "nonexistent-msg-id"

    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="hello back")
    result = await controller._finalize_agent_reply(
        fake_id, session.id, outcome, variant_mode=False
    )

    messages = store.messages_for_session(session.id)
    assistant = messages[-1]
    assert assistant.role is ConsoleMessageRole.ASSISTANT
    assert assistant.content == "hello back"
    assert assistant.status == "complete"
    assert result.accepted is True
    assert controller.run_state.status is ConsoleRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_stream_wrapper_settles_missing_placeholder_append_fallback(monkeypatch):
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    placeholder, turn_context = _begin_controller_disclosure(store, session.id)
    assert session.library_destination_runtime.disclosure is not None

    async def missing_placeholder_inner(**_kwargs):
        return await controller._finalize_agent_reply(
            placeholder.id,
            session.id,
            RunOutcome(status=RUN_DONE, steps=[], final_text="completed fallback"),
            variant_mode=False,
        )

    monkeypatch.setattr(
        controller,
        "_stream_assistant_response_inner",
        missing_placeholder_inner,
    )
    monkeypatch.setattr(controller, "_ensure_assistant_placeholder", lambda *_: None)
    monkeypatch.setattr(controller, "_find_runtime_written_assistant", lambda *_: None)

    result = await controller._stream_assistant_response(
        resolution=SimpleNamespace(),
        provider_messages=[],
        assistant_message_id=placeholder.id,
        turn_context=turn_context,
    )

    assert result.accepted is True
    assert session.library_destination_runtime.disclosure is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal",
    ["success", "failure", "cancelled", "stopped", "variant_success"],
)
async def test_agent_terminal_paths_settle_the_bound_destination_attempt(
    terminal: str,
) -> None:
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    placeholder, _turn_context = _begin_controller_disclosure(
        store,
        session.id,
        content="original" if terminal == "variant_success" else "",
    )
    cancel_event = threading.Event()
    variant_mode = terminal == "variant_success"
    if variant_mode:
        store.begin_variant_stream(placeholder.id)
        store.append_stream_chunk(placeholder.id, "replacement")
    elif terminal in {"success", "failure"}:
        store.append_stream_chunk(placeholder.id, "reply")
    if terminal == "stopped":
        store.mark_message_stopped(placeholder.id)
        cancel_event.set()
    outcome = RunOutcome(
        status=(
            RUN_DONE
            if terminal in {"success", "variant_success", "stopped"}
            else RUN_CANCELLED
            if terminal == "cancelled"
            else RUN_ERROR
        ),
        steps=[],
        final_text=(
            "replacement"
            if terminal == "variant_success"
            else "reply"
            if terminal == "success"
            else ""
        ),
    )

    result = await controller._finalize_agent_reply(
        placeholder.id,
        session.id,
        outcome,
        variant_mode=variant_mode,
        cancel_event=cancel_event,
    )

    assert result.accepted is True
    assert session.library_destination_runtime.disclosure is None
    assert session.library_destination_runtime.owner_attempt_id is None
    assert session.library_destination_runtime.owner_message_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["refused", "cancelled"])
async def test_stream_wrapper_exactly_settles_predispatch_exit(
    monkeypatch,
    outcome: str,
) -> None:
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    placeholder, turn_context = _begin_controller_disclosure(store, session.id)

    async def predispatch_exit(**_kwargs):
        if outcome == "cancelled":
            raise asyncio.CancelledError
        return controller._block(session.id, "Provider request was not sent.")

    monkeypatch.setattr(
        controller,
        "_stream_assistant_response_inner",
        predispatch_exit,
    )

    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await controller._stream_assistant_response(
                resolution=SimpleNamespace(),
                provider_messages=[],
                assistant_message_id=placeholder.id,
                turn_context=turn_context,
            )
    else:
        result = await controller._stream_assistant_response(
            resolution=SimpleNamespace(),
            provider_messages=[],
            assistant_message_id=placeholder.id,
            turn_context=turn_context,
        )
        assert result.accepted is False

    assert session.library_destination_runtime.disclosure is None
    assert session.library_destination_runtime.owner_attempt_id is None


@pytest.mark.asyncio
async def test_finalize_agent_reply_error_marks_failed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    store.append_stream_chunk(placeholder.id, "partial")

    outcome = RunOutcome(status=RUN_ERROR, steps=[], final_text="")
    result = await controller._finalize_agent_reply(
        placeholder.id, session.id, outcome, variant_mode=False
    )

    messages = store.messages_for_session(session.id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.status == "failed"
    assert assistant.content == "partial"
    assert "Agent run failed" in controller.run_state.visible_copy
    assert result.accepted is True


@pytest.mark.asyncio
async def test_finalize_agent_reply_cancelled_marks_failed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    outcome = RunOutcome(status=RUN_CANCELLED, steps=[], final_text="")
    result = await controller._finalize_agent_reply(
        placeholder.id, session.id, outcome, variant_mode=False
    )

    messages = store.messages_for_session(session.id)
    assistant = next(m for m in messages if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.status == "failed"
    assert controller.run_state.status is ConsoleRunStatus.FAILED
    assert result.accepted is True


@pytest.mark.asyncio
async def test_finalize_agent_reply_unknown_status_marks_failed():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    outcome = RunOutcome(status="weird", steps=[], final_text="")
    result = await controller._finalize_agent_reply(
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
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Hi there"
    )

    snapshot = await controller.build_context_snapshot(draft="Explain tools")

    assert len(snapshot.current_messages) == 2
    assert snapshot.current_messages[0].role == ConsoleMessageRole.USER
    assert snapshot.next_send_payload["messages"][-1]["content"].startswith(
        "Explain tools"
    )


@pytest.mark.asyncio
async def test_build_context_snapshot_does_not_execute_skills():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(session.id, role=ConsoleMessageRole.USER, content="Hello")

    snapshot = await controller.build_context_snapshot(draft="$search tools")
    final_content = snapshot.next_send_payload["messages"][-1]["content"]
    assert "$search tools" in final_content
    assert "Skill command not resolved in preview" in final_content


@pytest.mark.asyncio
async def test_build_context_snapshot_empty_draft_does_not_annotate_historical_skill_command():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session(title="Chat 1")

    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="/search tools"
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="Here are some tools."
    )

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

    msg = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="Hello"
    )

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

    snapshot = await controller.build_context_snapshot(
        draft="", attachments=[attachment]
    )

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

    snapshot = await controller.build_context_snapshot(
        draft="Summarize", staged_sources=sources
    )

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
    # Qodo (PR #860): the failure here fires inside the annotate->strip
    # window, so the degraded payload must strip the private id-threading
    # key too -- it must never surface in the inspector snapshot.
    assert all(
        controller_module.NATIVE_MESSAGE_ID_KEY not in row
        for row in payload["messages"]
    )


def test_annotate_skill_commands_multimodal_text_part():
    """Fix 4 (Qodo PR #801 fix wave): a multimodal (list-content) draft must
    NEVER be annotated, even when its text part starts with a `$name`
    mention. `_apply_skill_substitution` early-returns on non-str content at
    send time (replacing list content would drop attachments), so
    annotating a list-content draft here promised a substitution the actual
    send never performed -- a dishonest preview. List content now passes
    through unchanged."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "$search tools"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
        }
    ]

    annotated = ConsoleChatController._annotate_skill_commands(messages)

    assert annotated[0]["content"] == messages[0]["content"]
    assert "Skill command not resolved in preview" not in str(annotated[0]["content"])


def test_annotate_skill_commands_ignores_leading_whitespace():
    messages = [{"role": "user", "content": "  $search tools"}]

    annotated = ConsoleChatController._annotate_skill_commands(messages)

    assert annotated[0]["content"].startswith("  $search tools")
    assert "Skill command not resolved in preview" in annotated[0]["content"]


def test_annotate_skill_commands_synthetic_turn_added_false_returns_unchanged():
    messages = [{"role": "user", "content": "$search tools"}]

    annotated = ConsoleChatController._annotate_skill_commands(
        messages, synthetic_turn_added=False
    )

    assert annotated == messages
    assert "Skill command not resolved in preview" not in annotated[0]["content"]


def test_annotate_skill_commands_slash_command_is_not_annotated():
    """A `/`-prefixed draft is a registered slash command post-migration, not a
    skill invocation (Task 5 of the `$`-mention migration) -- it must not be
    flagged as an unresolved skill command in the preview."""
    messages = [{"role": "user", "content": "/skills search"}]

    annotated = ConsoleChatController._annotate_skill_commands(messages)

    assert annotated[0]["content"] == "/skills search"
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
            {
                "name": "calculator",
                "description": "Compute arithmetic.",
                "parameters": {},
            },
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
    session.project_instruction_state = ProjectInstructionControlState.legacy_disabled()
    if session.settings is None:
        session.settings = ConsoleSessionSettings(provider="llama_cpp")
    return session


def _controller_history_checkpoint(canary: str):
    return parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "complete",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": [canary * 80],
                    "calls": [
                        {
                            "call_id": "joined-call",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "completed",
                            "result": "done",
                        }
                    ],
                }
            ],
        }
    )


def _controller_active_history_checkpoint(call_state: str):
    return parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 2,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": ["ACTIVE-SWITCH-PRIVATE-CANARY"],
                    "calls": [
                        {
                            "call_id": "active-switch-call",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": call_state,
                        }
                    ],
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_controller_real_gateway_budgets_active_continuation_owner_atomically():
    store = ConsoleChatStore()
    session = _arm_session(store)
    old_user = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="old"
    )
    owner = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="old answer"
    )
    checkpoint = _controller_history_checkpoint("CONTROLLER-PRIVATE-CANARY ")
    store._message_or_raise(owner.id).provider_continuation = checkpoint
    gateway = ContinuationHistoryGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
    )

    source_snapshot = store.messages_for_session(session.id)
    result = await controller.submit_draft("current")

    assert result.accepted
    assert gateway.prepared is not None
    assert [row["content"] for row in gateway.prepared.messages_payload] == ["current"]
    assert (
        gateway.prepare_kwargs["continuation_sidecar"][0].owner_message_id == owner.id
    )
    assert "CONTROLLER-PRIVATE-CANARY" not in repr(gateway.prepared)
    assert store.get_message(old_user.id).content == "old"
    assert store.get_message(owner.id).content == "old answer"
    assert source_snapshot[1].provider_continuation == checkpoint


@pytest.mark.asyncio
async def test_controller_bridge_agent_service_bound_private_history_on_real_send(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_EVICT_ENABLED", "true")
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_EVICT_MIN_RECENT_ROUNDS", "1")
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    monkeypatch.setattr(
        console_history_budget, "get_model_token_limit", lambda *a, **k: 650
    )
    store = ConsoleChatStore()
    session = _arm_session(store)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="old")
    owner = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="old answer"
    )
    store._message_or_raise(
        owner.id
    ).provider_continuation = _controller_history_checkpoint("JOINED-PRIVATE-CANARY ")
    gateway = ContinuationHistoryGateway()
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "runs.db", client_id="task6"),
        store=store,
        provider_gateway=gateway,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=True,
        agent_bridge=bridge,
    )

    result = await controller.submit_draft("current")

    assert result.accepted
    assert gateway.prepared is not None
    prepared = gateway.prepared.messages_payload
    assert not any(row.get("content") == "old answer" for row in prepared)
    assert any(row.get("content") == "current" for row in prepared)
    assert all("_native_message_id" not in row for row in prepared)
    assert "JOINED-PRIVATE-CANARY" not in repr(gateway.prepared)


@pytest.mark.asyncio
async def test_provider_switch_ignores_unrelated_completed_continuation_history():
    class OpenAIGateway(ContinuationHistoryGateway):
        async def resolve_for_send(self, selection):
            return ConsoleProviderResolution(
                provider="openai",
                base_url="https://api.openai.com/v1",
                model="gpt-4.1",
                ready=True,
                readiness_key="openai",
                execution_key="openai",
                max_tokens=10,
                resolved_destination=ConsoleResolvedDestination(
                    provider="openai",
                    model="gpt-4.1",
                    endpoint_identity="https://api.openai.com/v1",
                    egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
                ),
            )

    store = ConsoleChatStore()
    session = _arm_session(store)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="old")
    owner = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="old answer"
    )
    store._message_or_raise(
        owner.id
    ).provider_continuation = _controller_history_checkpoint(
        "PROVIDER-SWITCH-PRIVATE-CANARY "
    )
    gateway = OpenAIGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
    )

    result = await controller.submit_draft("current")

    assert result.accepted
    assert gateway.prepared is not None
    assert any(
        row.get("content") == "old answer" for row in gateway.prepared.messages_payload
    )
    assert gateway.prepare_kwargs["continuation_sidecar"] == ()
    assert "PROVIDER-SWITCH-PRIVATE-CANARY" not in repr(gateway.prepared)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_state", ["pending", "executing"])
async def test_provider_switch_race_blocks_active_continuation_before_dispatch(
    call_state: str,
):
    store = ConsoleChatStore()
    session = _arm_session(store)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="old")
    owner = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="old answer"
    )
    store._message_or_raise(
        owner.id
    ).provider_continuation = _controller_history_checkpoint(
        "COMPLETE-BEFORE-RESOLUTION "
    )

    class SwitchingGateway(ContinuationHistoryGateway):
        provider_calls = 0

        async def resolve_for_send(self, selection):
            store._message_or_raise(
                owner.id
            ).provider_continuation = _controller_active_history_checkpoint(call_state)
            return ConsoleProviderResolution(
                provider="openai",
                base_url="https://api.openai.com/v1",
                model="gpt-4.1",
                ready=True,
                readiness_key="openai",
                execution_key="openai",
                max_tokens=10,
                resolved_destination=ConsoleResolvedDestination(
                    provider="openai",
                    model="gpt-4.1",
                    endpoint_identity="https://api.openai.com/v1",
                    egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
                ),
            )

        async def stream_chat(self, resolution, messages, **kwargs):
            self.provider_calls += 1
            yield "must not dispatch"

    gateway = SwitchingGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=False,
    )

    result = await controller.submit_draft("current")

    assert result.visible_copy == (
        "Recover the interrupted tool run before sending a new message: "
        "Resume or Discard it first."
    )
    assert controller.run_state_for(session.id).status is ConsoleRunStatus.BLOCKED
    assert gateway.provider_calls == 0
    assert "ACTIVE-SWITCH-PRIVATE-CANARY" not in repr(result)


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
    assert (
        messages[-1].content == "Sure thing:ok"
    )  # seed + RecordingStreamingGateway's "ok"
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

        async def stream_chat(self, resolution, messages, **kwargs):
            yield "partial"
            # Fix round 1 (Critical 1): the direct/legacy stream loop now
            # reads only its OWN run's per-session cancel_event, never the
            # shared `_stop_requested` flag -- simulate Stop via the real
            # internal signalling path instead of the flag directly.
            self.controller._signal_stop(
                session_id=self.controller.store.active_session_id
            )
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

        async def stream_chat(self, resolution, messages, **kwargs):
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
        return "run-test", RunOutcome(
            status=RUN_DONE, steps=[], final_text="agent says"
        )

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
        raise AssertionError(
            "agent bridge should not be called for a character session"
        )


@pytest.mark.asyncio
async def test_server_character_session_without_local_projection_forces_plain_provider():
    """Trusted character kind, not a local numeric projection, owns routing."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True
    )
    bridge = _SpyAgentBridge()
    controller._agent_bridge = bridge
    session = _arm_session(store)
    session.runtime_backend = "server"
    session.assistant_kind = "character"
    session.assistant_id = "opaque-character"
    session.assistant_authority_id = None
    assert session.character_id is None

    result = await controller.submit_draft("Hi")

    assert bridge.calls == 0
    assert result.accepted
    assert gateway.messages_seen is not None  # plain provider path ran


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assistant_kind", "assistant_id", "character_id"),
    [
        ("generic", "console", None),
        ("persona", "persona-1", None),
        ("generic", "console", 7),
    ],
    ids=["generic", "persona", "stray-numeric-character-id"],
)
async def test_non_character_session_still_uses_agent_when_enabled(
    assistant_kind, assistant_id, character_id
):
    """Generic/Persona sessions do not become direct chat from a stray id."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True
    )
    bridge_calls = []

    def run_reply(**kwargs):
        bridge_calls.append(kwargs)
        return "run-test", RunOutcome(
            status=RUN_DONE, steps=[], final_text="agent says"
        )

    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)
    session = _arm_session(store)
    session.assistant_kind = assistant_kind
    session.assistant_id = assistant_id
    session.character_id = character_id

    await controller.submit_draft("Hi")

    assert len(bridge_calls) == 1
    assert gateway.messages_seen is None  # agent path handled it, not the gateway


@pytest.mark.asyncio
async def test_agent_path_applies_dictionary_before_bridge_sees_messages():
    """TASK-761: pins the ORDERING contract, not just that the dictionary
    is applied somewhere. `_apply_chat_dictionaries` and the agent
    bridge's `run_reply` both append to a single shared event log; the
    assertion requires the dictionary event to precede the bridge event
    AND the bridge's own captured payload to already carry the substituted
    text -- so a regression that calls the bridge first (even one that
    still gets the content right some other way) fails this test, unlike
    an assertion that only checks the final content in isolation."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    events: list[str] = []

    def applier(conversation_id, content):
        events.append("dictionary_applied")
        return content.replace("Warden", "grim jailer")

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=True,
        chat_dictionary_applier=applier,
    )
    session = _arm_session(store)
    session.persisted_conversation_id = "conv-1"

    captured: dict[str, list[dict[str, str]]] = {}

    def run_reply(*, agent_messages, **kwargs):
        events.append("bridge_called")
        captured["agent_messages"] = list(agent_messages)
        return "run-test", RunOutcome(status=RUN_DONE, steps=[], final_text="ok")

    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)

    await controller.submit_draft("The Warden nods.")

    # Ordering: the dictionary MUST run before the bridge is dispatched.
    assert events == ["dictionary_applied", "bridge_called"]
    # And the bridge must have RECEIVED the substituted content, not the
    # raw draft -- proving the substitution landed on the payload the
    # bridge actually sees, not merely that the applier was called.
    final_user = [m for m in captured["agent_messages"] if m.get("role") == "user"][-1]
    assert final_user["content"] == "The grim jailer nods."


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
async def test_agent_finalization_remains_inside_outer_active_stream_ownership():
    class OwnershipController(ConsoleChatController):
        active_during_finalization = None

        async def _finalize_agent_reply(
            self,
            assistant_message_id,
            session_id,
            outcome,
            **kwargs,
        ):
            self.active_during_finalization = (
                self._active_assistant_message_ids.get(session_id),
                self._active_stream_tasks.get(session_id),
                self._stop_requested,
            )
            return await super()._finalize_agent_reply(
                assistant_message_id,
                session_id,
                outcome,
                **kwargs,
            )

    class Bridge:
        def run_reply(self, **_kwargs):
            return "run-active-owner", RunOutcome(
                status=RUN_DONE,
                steps=[],
                final_text="agent reply",
            )

        def record_run_assistant_message(self, _run_id, _message_id):
            return None

    store = ConsoleChatStore()
    controller = OwnershipController(
        store=store,
        provider_gateway=StreamingGateway(),
        agent_bridge=Bridge(),
        agent_runtime_enabled=True,
    )
    _arm_session(store)
    current_task = asyncio.current_task()

    result = await controller.submit_draft("hello")

    assistant = next(
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
    )
    assert result.accepted is True
    assert controller.active_during_finalization == (
        assistant.id,
        current_task,
        False,
    )
    assert controller._active_assistant_message_ids.get(store.active_session_id) is None
    assert controller._active_stream_tasks.get(store.active_session_id) is None
    assert controller._stop_requested is False


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
    session = controller.new_session(
        title="Chat 1", ephemeral=True
    )  # creates + activates
    # Seed an over-budget history before the current turn.
    for i in range(6):
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=f"old user {i} aa bb cc dd",
        )
        store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content=f"old asst {i} aa bb cc dd",
        )

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
            provider="llama_cpp",
            explicit_model="test-model",
            configured_model="test-model",
        )
    )
    session = controller.new_session(title="Chat 1", ephemeral=True)
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="one small turn"
    )
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
        provider_messages.append(
            {"role": "user", "content": f"old user {i} aa bb cc dd"}
        )
        provider_messages.append(
            {"role": "assistant", "content": f"old asst {i} aa bb cc dd"}
        )
    provider_messages.append({"role": "user", "content": "current question here"})

    # ...but the captured resolution (what actually dispatches) is the tiny model.
    resolution = SimpleNamespace(
        ready=True,
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        model="small-window",
        max_tokens=0,
        visible_copy="",
        resolved_destination=ConsoleResolvedDestination(
            provider="llama_cpp",
            model="small-window",
            endpoint_identity="http://127.0.0.1:9099",
            egress_class=ConsoleEgressClass.ON_DEVICE,
        ),
    )
    configuration = controller.resolve_turn_configuration_snapshot(session.id)
    authority = await controller._capture_turn_library_authority(
        session.id, configuration
    )
    turn_context = controller._finalize_turn_execution_context(
        configuration, authority, resolution
    )

    await controller._stream_assistant_response(
        resolution=resolution,
        provider_messages=provider_messages,
        assistant_message_id=assistant.id,
        turn_context=turn_context,
    )

    # Budgeted against the 520-token resolution window (not the 1M self.model),
    # so the 13-message history collapsed to just the current turn.
    assert gateway.messages_seen is not None
    assert len(gateway.messages_seen) < 13
    assert gateway.messages_seen[-1]["content"] == "current question here"


# -- TASK-631: the kill switch must cover EVERY tool call the hook sees ----


@pytest.mark.unit
def test_kill_switch_refuses_unclaimed_tool_calls_at_the_review_hook():
    """The kill switch must refuse every call, including unclaimed names.

    Its label promises "block tool calls in chat" -- all of them. MCP
    composition is skipped and `BuiltinToolGate.check` refuses when the
    switch is on, but a name NEITHER provider claims (a skill,
    `spawn_subagent`, `find_tools`, `load_tools`) passed through the review
    hook unreviewed and dispatched normally: flipping the switch to stop all
    tool calls left four tool families running. A false sense of security in
    a security-relevant control.

    The hook is the one place every parsed call passes, and the runtime
    turns any non-"proceed" verdict into the call's result without
    dispatching it -- so enforcing here covers the unclaimed families with
    no new plumbing.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import (
        KILL_SWITCH_REFUSAL,
        build_tool_review_hook,
    )

    class _Gate:
        def begin_turn(self, run_id):
            pass

        def resolve(self, tool):
            return SimpleNamespace(state="ask", risk_floored=False)

        def stamp(self, run_id, name, decision):
            pass

        def is_session_approved(self, name):
            return False

        def options_for(self, tool):
            return ("approve_once", "approve_session", "deny")

    class _Provider:
        def tool_for(self, name):
            return None  # claims nothing

    prompted = []

    def request_approvals(pending):
        prompted.append(pending)
        return {}

    hook = build_tool_review_hook(
        _Gate(),
        _Provider(),
        None,
        request_approvals,
        workspace_id=None,
        kill_switch=lambda: True,
    )
    verdicts = hook(
        [
            ToolCall(name="spawn_subagent", args={"task": "x"}, call_id="c1"),
            ToolCall(name="skill__notes__summarize", args={}, call_id="c2"),
            ToolCall(name="find_tools", args={"query": "q"}),
        ],
        RUN,
    )

    assert not prompted, "the kill switch must refuse, not prompt"
    assert verdicts.get("c1") == KILL_SWITCH_REFUSAL
    assert verdicts.get("c2") == KILL_SWITCH_REFUSAL
    assert verdicts.get("find_tools") == KILL_SWITCH_REFUSAL, (
        "an id-less call must be refused by name, or the fence path "
        f"escapes the switch: {verdicts}"
    )


@pytest.mark.unit
def test_kill_switch_off_changes_nothing():
    """With the switch off (or absent) the hook behaves exactly as before."""
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    class _Gate:
        def begin_turn(self, run_id):
            pass

        def resolve(self, tool):
            return SimpleNamespace(state="ask", risk_floored=False)

        def stamp(self, run_id, name, decision):
            pass

        def is_session_approved(self, name):
            return False

        def options_for(self, tool):
            return ("approve_once", "approve_session", "deny")

    class _Provider:
        def tool_for(self, name):
            return SimpleNamespace(name=name)

    def request_approvals(pending):
        return {row.call_id: "approve_once" for row in pending}

    for switch in (None, lambda: False):
        hook = build_tool_review_hook(
            _Gate(),
            _Provider(),
            None,
            request_approvals,
            workspace_id=None,
            kill_switch=switch,
        )
        verdicts = hook(
            [ToolCall(name="read_file", args={"path": "a"}, call_id="c1")], RUN
        )
        assert verdicts.get("c1", "proceed") == "proceed", (switch, verdicts)


@pytest.mark.unit
def test_unclaimed_names_pass_through_the_hook_unreviewed_switch_off():
    """Pin the documented pass-through contract (TASK-294, P5 minor).

    `build_tool_review_hook`'s docstring states a name neither provider
    claims (a skill, `spawn_subagent`, `find_tools`, `load_tools`) passes
    through unreviewed. TASK-631's test proves those names are REFUSED with
    the kill switch on; nothing pinned the switch-OFF half -- no prompt, no
    verdict entry, so the runtime dispatches them normally. If routing ever
    started claiming these names by accident, gated prompts would appear
    for internal plumbing calls; if it started refusing them, agent spawn
    and tool discovery would silently break.
    """
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook

    class _Gate:
        def begin_turn(self, run_id):
            pass

        def resolve(self, tool):
            return SimpleNamespace(state="ask", risk_floored=False)

        def stamp(self, run_id, name, decision):
            pass

        def is_session_approved(self, name):
            return False

        def options_for(self, tool):
            return ("approve_once", "approve_session", "deny")

    class _Provider:
        def tool_for(self, name):
            return None  # claims nothing

    prompted: list = []

    def request_approvals(pending):
        prompted.append(pending)
        return {}

    hook = build_tool_review_hook(_Gate(), _Provider(), None, request_approvals)
    verdicts = hook(
        [
            ToolCall(name="spawn_subagent", args={"task": "x"}, call_id="c1"),
            ToolCall(name="find_tools", args={"query": "q"}, call_id="c2"),
            ToolCall(name="load_tools", args={"names": []}, call_id="c3"),
            ToolCall(name="skill__notes__summarize", args={}, call_id="c4"),
        ],
        RUN,
    )

    assert not prompted, "unclaimed names must not be offered a card"
    assert verdicts == {}, (
        f"unclaimed names must pass through unreviewed, got: {verdicts}"
    )


class UsageEmittingGateway(StreamingGateway):
    """Mirrors the real gateway's usage seam.

    One ``stream_chat`` invocation is one provider CALL: payloads recorded
    during the stream key-merge into the in-flight slot, and the call is
    closed out in a ``finally`` -- exactly what
    ``ConsoleProviderGateway.stream_chat`` does. Successive calls consume
    successive entries of ``payloads_per_call`` so an agent turn's N calls
    can be exercised.
    """

    payloads_per_call = ({"prompt_tokens": 100, "completion_tokens": 20},)

    def __init__(self):
        self.calls = 0

    async def stream_chat(self, resolution, messages, **kwargs):
        signals = kwargs.get("signals")
        index = self.calls
        self.calls += 1
        try:
            for chunk in ("hel", "lo"):
                yield chunk
            if signals is not None and index < len(self.payloads_per_call):
                payload = self.payloads_per_call[index]
                # A payload may itself arrive split across chunks (Anthropic).
                for fragment in payload if isinstance(payload, tuple) else (payload,):
                    signals.record_usage_payload(fragment)
        finally:
            if signals is not None:
                signals.close_usage_call()


@pytest.mark.asyncio
async def test_completed_message_carries_normalized_usage():
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=UsageEmittingGateway()
    )
    session = store.ensure_session(title="Chat 1")

    result = await controller.submit_draft("hi")
    assert result.accepted

    messages = store.messages_for_session(session.id)
    assistant = messages[-1]
    assert assistant.status == "complete"
    assert assistant.usage is not None
    assert assistant.usage.uncached_input == 100
    assert assistant.usage.output == 20
    assert assistant.usage.partial is False
    assert assistant.usage.provider  # attributed from resolution


#
# Final-review F1/F2/F3/F7: usage capture on every terminal path
#
class _UsageRecordingPersistence:
    """Minimal persistence that records the usage_json it is handed."""

    def __init__(self):
        self.created = []
        self.updated = []
        self._counter = 0
        self.console_library_policy_repository = SimpleNamespace(read=self._read_policy)
        self.console_dispatch_repository = self
        self._policy_snapshot = None
        self._checkpoint = None

    def _read_policy(self, conversation_id):
        del conversation_id
        return SimpleNamespace(durable_policy=object(), snapshot=self._policy_snapshot)

    def _cas_state(self, transition):
        checkpoint = self._checkpoint
        if checkpoint is None:
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.NOT_FOUND, None, None, None
            )
        checkpoint = replace(
            checkpoint,
            state=transition.new_state,
            checkpoint_revision=checkpoint.checkpoint_revision + 1,
            assistant_message_version=checkpoint.assistant_message_version + 1,
            attempt_id=transition.new_attempt_id,
        )
        self._checkpoint = checkpoint
        return ConsoleDispatchWriteResult(
            ConsoleDispatchResultStatus.COMMITTED,
            checkpoint,
            checkpoint.assistant_message_version,
            "fake-payload-hash",
        )

    cas_state = _cas_state

    def settle_with_assistant(self, settlement):
        checkpoint = self._checkpoint
        if checkpoint is None:
            return ConsoleDispatchWriteResult(
                ConsoleDispatchResultStatus.NOT_FOUND, None, None, None
            )
        self.updated.append(
            {
                "message_id": settlement.assistant_message_id,
                "content": settlement.content,
                "usage_json": settlement.usage_json,
            }
        )
        self._checkpoint = None
        return ConsoleDispatchWriteResult(
            ConsoleDispatchResultStatus.COMMITTED,
            None,
            checkpoint.assistant_message_version + 1,
            "fake-terminal-hash",
        )

    def create_conversation(self, **kwargs):
        return "conv-usage"

    def commit_durable_turn(self, *, acceptance, policy_candidate, conversation_kwargs):
        """Model atomic acceptance while retaining usage-write observations."""
        del conversation_kwargs
        self._policy_snapshot = ConsoleLibraryPolicySnapshot(
            auto_retrieve=policy_candidate.auto_retrieve,
            assistant_access=policy_candidate.assistant_access,
            policy_revision=1,
            source="durable",
        )
        self.created.extend(
            (
                {
                    "conversation_id": acceptance.conversation_id,
                    "sender": "user",
                    "content": acceptance.user_content,
                    "message_id": acceptance.user_message_id,
                },
                {
                    "conversation_id": acceptance.conversation_id,
                    "sender": "assistant",
                    "content": "",
                    "message_id": acceptance.assistant_message_id,
                },
            )
        )
        checkpoint = ConsoleDispatchCheckpoint(
            assistant_message_id=acceptance.assistant_message_id,
            user_message_id=acceptance.user_message_id,
            conversation_id=acceptance.conversation_id,
            preparation_id=acceptance.preparation_id,
            attempt_id=acceptance.attempt_id,
            state=ConsoleDispatchCheckpointState.ACCEPTED,
            checkpoint_revision=1,
            user_message_version=1,
            assistant_message_version=1,
            origin=acceptance.origin,
            queue_entry_id=acceptance.queue_entry_id,
            frozen_authority=acceptance.frozen_authority,
            resolved_destination=acceptance.resolved_destination,
            reconstructability=acceptance.reconstructability,
        )
        self._checkpoint = checkpoint
        return checkpoint

    def create_message(self, **kwargs):
        self.created.append(kwargs)
        self._counter += 1
        return f"msg-{self._counter}"

    def update_message_content(self, **kwargs):
        self.updated.append(kwargs)
        return True

    def usage_values(self):
        return [
            kwargs.get("usage_json")
            for kwargs in (*self.created, *self.updated)
            if kwargs.get("usage_json") is not None
        ]


class _GatewayDrivingBridge:
    """Stub agent bridge that dispatches through the gateway exactly as the
    real ``ConsoleAgentBridge`` does.

    The load-bearing detail is `console_agent_bridge.py`'s own seam: it adds
    ``signals=`` to the gateway call ONLY when ``provider_stream_signals`` is
    non-None. The controller used to forward ``None`` on this (default!)
    path, so nothing was ever captured for the agent runtime -- finding F1.
    """

    def __init__(self, gateway, store, *, calls_per_turn=1):
        self._gateway = gateway
        self._store = store
        self._calls_per_turn = calls_per_turn
        self.signals_seen = "never-called"

    def run_reply(self, **kwargs):
        self.signals_seen = kwargs.get("provider_stream_signals")
        stream_kwargs = {}
        if self.signals_seen is not None:
            stream_kwargs["signals"] = self.signals_seen
        assistant_message_id = kwargs["assistant_message_id"]

        async def _drain():
            text = ""
            for _ in range(self._calls_per_turn):
                async for chunk in self._gateway.stream_chat(
                    kwargs["resolution"], kwargs["agent_messages"], **stream_kwargs
                ):
                    self._store.append_stream_chunk(assistant_message_id, chunk)
                    text += chunk
            return text

        final_text = asyncio.run(_drain())
        return "run-usage", RunOutcome(status=RUN_DONE, steps=[], final_text=final_text)


@pytest.mark.asyncio
async def test_agent_path_attaches_and_persists_usage():
    """F1 regression: the DEFAULT send path (agent runtime on, bridge wired)
    captured NOTHING because the controller only built stream signals for
    citation repair. Every real send took this path.
    """
    persistence = _UsageRecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    gateway = UsageEmittingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True
    )
    bridge = _GatewayDrivingBridge(gateway, store)
    controller._agent_bridge = bridge
    session = _arm_session(store)

    result = await controller.submit_draft("hi")
    assert result.accepted

    assert bridge.signals_seen not in (None, "never-called"), (
        "the agent bridge must receive a real signals object, not None"
    )
    assistant = store.messages_for_session(session.id)[-1]
    assert assistant.status == "complete"
    assert assistant.usage is not None
    assert assistant.usage.uncached_input == 100
    assert assistant.usage.output == 20
    assert assistant.usage.partial is False
    assert any('"uncached_input": 100' in value for value in persistence.usage_values())


@pytest.mark.asyncio
async def test_agent_turn_sums_usage_across_provider_calls():
    """F2 regression at the turn level: an agent turn makes N provider calls.
    Raw-payload key-merging made call 2's 900 prompt_tokens sit next to call
    1's stale cached_tokens=4096 -> uncached_input 0 and a phantom cache read.
    Correct: normalize per call, then SUM the disjoint buckets.
    """

    class TwoCallGateway(UsageEmittingGateway):
        payloads_per_call = (
            {
                "prompt_tokens": 5000,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 4096},
            },
            {"prompt_tokens": 900, "completion_tokens": 30},
        )

    store = ConsoleChatStore()
    gateway = TwoCallGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True
    )
    controller._agent_bridge = _GatewayDrivingBridge(gateway, store, calls_per_turn=2)
    session = _arm_session(store)

    assert (await controller.submit_draft("hi")).accepted

    usage = store.messages_for_session(session.id)[-1].usage
    assert usage is not None
    assert usage.uncached_input == 1804  # (5000-4096) + 900
    assert usage.cache_read == 4096  # call 1 only -- never re-billed for call 2
    assert usage.output == 40


@pytest.mark.asyncio
async def test_stopped_stream_persists_partial_input_usage():
    """F3 regression: ``stop_active_run`` finalizes the message BEFORE the
    cancelled task attaches usage, and the second ``_mark_stream_stopped``
    takes the read-back branch -- so nothing ever persisted the tokens the
    provider had already billed. Anthropic-shaped: the input side arrives at
    ``message_start``, long before any output tokens exist.
    """

    class StalledAnthropicGateway(StreamingGateway):
        def __init__(self):
            self.started = asyncio.Event()
            self.never_release = asyncio.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
            signals = kwargs.get("signals")
            try:
                if signals is not None:
                    signals.record_usage_payload(
                        {"input_tokens": 3571, "cache_read_input_tokens": 6656}
                    )
                self.started.set()
                yield "partial"
                await self.never_release.wait()
                yield "ignored"
            finally:
                if signals is not None:
                    signals.close_usage_call()

    persistence = _UsageRecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    gateway = StalledAnthropicGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    task = asyncio.create_task(controller.submit_draft("hello"))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert controller.stop_active_run() is True
    result = await asyncio.wait_for(task, timeout=1)
    assert result.accepted

    stopped = [
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
    ][-1]
    assert stopped.status == "stopped"
    assert stopped.usage is not None
    assert stopped.usage.uncached_input == 3571
    assert stopped.usage.cache_read == 6656
    assert stopped.usage.partial is True

    persisted = persistence.usage_values()
    assert persisted, "the stopped turn's usage never reached persistence"
    assert '"uncached_input": 3571' in persisted[-1]
    assert '"partial": true' in persisted[-1]


class _RaisingOnUsageWritePersistence(_UsageRecordingPersistence):
    """Like ``_UsageRecordingPersistence``, but its content update raises
    once the write actually carries a ``usage_json`` payload -- simulating
    a SQLite/persistence exception during the stop-path's usage-only
    terminal flush."""

    def update_message_content(self, **kwargs):
        if kwargs.get("usage_json") is not None:
            raise RuntimeError("simulated persistence failure during usage flush")
        return super().update_message_content(**kwargs)


@pytest.mark.asyncio
async def test_stop_path_usage_attach_survives_a_persistence_exception():
    """Qodo round (Finding 1): ``_attach_stream_usage`` is documented "must
    never fail a send", but it used to only catch ``KeyError`` around
    ``store.set_message_usage``. Since that call now persists immediately
    for an already-terminal message (the stop-path flush, F3), ANY
    exception the persistence layer raises during that flush -- not just a
    missing message -- must not escape into stop/cancel control flow. A
    persistence adapter whose ``update_message_content`` raises
    ``RuntimeError`` specifically on the usage-carrying write proves the
    broadened ``except Exception`` swallows it and the stop outcome (status,
    content) is unaffected.
    """

    class StalledAnthropicGateway(StreamingGateway):
        def __init__(self):
            self.started = asyncio.Event()
            self.never_release = asyncio.Event()

        async def stream_chat(self, resolution, messages, **kwargs):
            signals = kwargs.get("signals")
            try:
                if signals is not None:
                    signals.record_usage_payload(
                        {"input_tokens": 3571, "cache_read_input_tokens": 6656}
                    )
                self.started.set()
                yield "partial"
                await self.never_release.wait()
                yield "ignored"
            finally:
                if signals is not None:
                    signals.close_usage_call()

    persistence = _RaisingOnUsageWritePersistence()
    store = ConsoleChatStore(persistence=persistence)
    gateway = StalledAnthropicGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway)

    task = asyncio.create_task(controller.submit_draft("hello"))
    await asyncio.wait_for(gateway.started.wait(), timeout=1)
    await asyncio.sleep(0)

    assert controller.stop_active_run() is True
    # Must not raise: the RuntimeError from the persistence layer's usage
    # write must be swallowed inside `_attach_stream_usage`, not propagate
    # out through the stream task.
    result = await asyncio.wait_for(task, timeout=1)
    assert result.accepted

    stopped = [
        message
        for message in store.messages_for_session(store.active_session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
    ][-1]
    assert stopped.status == "stopped"
    assert stopped.content == "partial"
    # The in-memory attach still happened (the send itself never failed) --
    # only the DURABLE write behind it raised and was swallowed.
    assert stopped.usage is not None
    assert stopped.usage.uncached_input == 3571


@pytest.mark.asyncio
async def test_billed_turn_without_visible_content_still_records_usage():
    """F7 (decided): a turn that reported usage but emitted no content -- a
    refusal, or a stream that ended after the usage chunk -- cost real money.
    The spec's "total = money actually spent" beats "failed sends produce no
    usage row", which is about transport failures where nothing was billed.
    """

    class ContentlessBilledGateway(StreamingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            signals = kwargs.get("signals")
            try:
                if signals is not None:
                    signals.record_usage_payload(
                        {"prompt_tokens": 812, "completion_tokens": 0}
                    )
                return
                yield  # pragma: no cover -- makes this an async generator
            finally:
                if signals is not None:
                    signals.close_usage_call()

    persistence = _UsageRecordingPersistence()
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(
        store=store, provider_gateway=ContentlessBilledGateway()
    )
    session = store.ensure_session(title="Chat 1")

    assert (await controller.submit_draft("hi")).accepted

    assistant = store.messages_for_session(session.id)[-1]
    assert assistant.status == "failed"
    assert assistant.usage is not None
    assert assistant.usage.uncached_input == 812
    assert assistant.usage.partial is True

    # Durable acceptance creates the intentionally empty assistant owner row;
    # terminal usage is then written onto that exact row even without content.
    assert any('"uncached_input": 812' in value for value in persistence.usage_values())
    assert [entry["sender"] for entry in persistence.created] == [
        "user",
        "assistant",
    ]


# --- Cost-ticker PR3: payload-fingerprint baseline + cache TTL --------------


@pytest.mark.asyncio
async def test_dispatch_records_fingerprint_baseline_and_cache_snapshot():
    class CacheUsageGateway(StreamingGateway):
        async def resolve_for_send(self, selection):
            resolution = await super().resolve_for_send(selection)
            resolution.provider = "anthropic"
            resolution.prompt_caching = True
            return resolution

        async def stream_chat(self, resolution, messages, **kwargs):
            signals = kwargs.get("signals")
            yield "hi"
            if signals is not None:
                signals.record_usage_payload(
                    {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "cache_creation_input_tokens": 900,
                    }
                )

    store = ConsoleChatStore()
    # Fix round 1, Finding 1: the baseline now comes from the DISPATCHED
    # resolution ("anthropic"/"test-model"), not from `self.provider`/
    # `self.model` -- so those must be pre-set to match here (mirroring
    # ordinary, non-racing usage, where the selection handed to
    # `resolve_for_send` is built from these same fields and so already
    # agrees with what comes back). `compute_current_fingerprint` reads
    # `self.provider`/`self.model` directly; the dedicated race test below
    # (`test_baseline_uses_dispatched_resolution_not_racing_controller_
    # fields`) covers the case where they deliberately diverge.
    controller = ConsoleChatController(
        store=store,
        provider_gateway=CacheUsageGateway(),
        provider="anthropic",
        model="test-model",
    )
    session = store.ensure_session(title="Chat 1")
    assert controller.payload_fingerprint_baseline(session.id) is None

    result = await controller.submit_draft("hello")
    assert result.accepted

    baseline = controller.payload_fingerprint_baseline(session.id)
    assert baseline is not None
    warm_until, had_activity = controller.cache_ttl_snapshot(session.id)
    assert had_activity is True
    assert warm_until is not None  # monotonic deadline stamped

    current = controller.compute_current_fingerprint(session.id)
    from tldw_chatbook.Chat.console_cost_tracker import fingerprint_break_reason

    assert fingerprint_break_reason(baseline, current) is None


@pytest.mark.asyncio
async def test_baseline_uses_dispatched_resolution_not_racing_controller_fields():
    """Fix round 1, Finding 1 (Critical): the baseline's provider/model must
    come from the RESOLUTION actually being dispatched, not
    `self.provider`/`self.model` -- those are controller-wide mutable
    fields shared across every fleet session, so a provider/model switch
    racing the awaits between `resolve_for_send` and the dispatch choke
    point (e.g. a DIFFERENT session's send flipping them in between) must
    not leak into THIS call's recorded baseline.
    """
    from tldw_chatbook.Chat.console_cost_tracker import fingerprint_payload

    class RacingProviderGateway(StreamingGateway):
        async def resolve_for_send(self, selection):
            resolution = await super().resolve_for_send(selection)
            # The pair this call is ACTUALLY dispatching.
            resolution.provider = "anthropic"
            resolution.model = "claude-real-dispatched-model"
            return resolution

    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store, provider_gateway=RacingProviderGateway()
    )
    session = store.ensure_session(title="Chat 1")

    original_resolve = controller.provider_gateway.resolve_for_send

    async def _racing_resolve(selection):
        resolution = await original_resolve(selection)
        # Simulate the race: something (another fleet session's own send)
        # flips the controller-wide mutable fields between resolve_for_send
        # returning and the dispatch choke point recording the baseline.
        controller.provider = "llama_cpp"
        controller.model = "wrong-racing-model"
        return resolution

    controller.provider_gateway.resolve_for_send = _racing_resolve

    result = await controller.submit_draft("hello")
    assert result.accepted

    baseline = controller.payload_fingerprint_baseline(session.id)
    assert baseline is not None

    dispatched_provider_model = fingerprint_payload(
        "anthropic", "claude-real-dispatched-model", []
    ).provider_model
    racing_provider_model = fingerprint_payload(
        "llama_cpp", "wrong-racing-model", []
    ).provider_model
    assert baseline.provider_model == dispatched_provider_model
    assert baseline.provider_model != racing_provider_model


@pytest.mark.asyncio
async def test_baseline_ignores_dispatch_time_substitution_and_stays_comparable():
    """Fix round 1, Finding 2 (Important, corrected design): the record site
    must fingerprint the SAME raw store view `compute_current_fingerprint`
    reads (a fresh `_provider_messages_for_session` call), not the
    `provider_messages` parameter in scope at the dispatch choke point.
    Every caller has already run its own per-send transforms (skill
    substitution here; chat-dictionary/world-info/RAG folding are the same
    shape) on that parameter before passing it in, so fingerprinting the
    parameter directly would compare a transformed payload against
    `compute_current_fingerprint`'s untransformed one and falsely report
    "earlier history changed" immediately after a completely ordinary send.
    """
    from tldw_chatbook.Chat.console_cost_tracker import fingerprint_break_reason

    store = ConsoleChatStore()
    # provider="llama_cpp" matches StreamingGateway's default resolution
    # already; model must be pre-set to match its "test-model" too (Finding
    # 1: the baseline reads the resolution's model, `compute_current_
    # fingerprint` reads `self.model` -- see the sibling test above for the
    # dedicated race case).
    controller = ConsoleChatController(
        store=store, provider_gateway=StreamingGateway(), model="test-model"
    )
    session = store.ensure_session(title="Chat 1")

    async def _substitute_final_turn(provider_messages):
        # Stand-in for skill/chat-dictionary/world-info substitution: the
        # ephemeral payload for this turn differs from what the store
        # actually holds (the raw text the user typed is what's persisted).
        transformed = [dict(row) for row in provider_messages]
        for row in reversed(transformed):
            if row.get("role") == "user":
                row["content"] = "SUBSTITUTED CONTENT -- not what the user typed"
                break
        return transformed, None, (), (), ""

    controller._apply_skill_substitution = _substitute_final_turn

    result = await controller.submit_draft("hello")
    assert result.accepted

    baseline = controller.payload_fingerprint_baseline(session.id)
    assert baseline is not None
    current = controller.compute_current_fingerprint(session.id)
    # No break immediately after an ordinary send -- the substitution never
    # touched the store, so both sides read the same raw view.
    assert fingerprint_break_reason(baseline, current) is None


# ---------------------------------------------------------------------------
# Prompt-history recording (TASK-1364)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_accepted_send_records_cleaned_draft_to_prompt_history(tmp_path):
    """AC4: an accepted send lands in the shared JSONL history exactly once."""
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    history_path = tmp_path / "prompt_history.jsonl"
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    controller.prompt_history = PromptHistory(history_path)

    result = await controller.submit_draft("record this prompt")
    assert result.accepted
    assert controller.prompt_history.size == 1
    entry = await controller.prompt_history.get_entry(-1)
    assert entry["input"] == "record this prompt"

    # Persisted as JSONL (one object per line, real timestamp).
    lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    persisted = json.loads(lines[0])
    assert persisted["input"] == "record this prompt"
    assert persisted["timestamp"] > 0


@pytest.mark.asyncio
async def test_blocked_send_records_nothing_to_prompt_history(tmp_path):
    """Refused/blocked sends never reach the history (validation failures)."""
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    history_path = tmp_path / "prompt_history.jsonl"
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=BlockedGateway())
    controller.prompt_history = PromptHistory(history_path)

    result = await controller.submit_draft("blocked prompt")
    assert not result.accepted
    assert controller.prompt_history.size == 0
    assert not history_path.exists()


@pytest.mark.asyncio
async def test_empty_and_whitespace_drafts_record_nothing(tmp_path):
    """Attachment-only (empty cleaned draft) sends skip recording."""
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    controller.prompt_history = PromptHistory(tmp_path / "prompt_history.jsonl")

    await controller._record_prompt_history("")
    await controller._record_prompt_history("   \n  ")
    assert controller.prompt_history.size == 0


@pytest.mark.asyncio
async def test_submit_without_prompt_history_configured_is_a_noop():
    """Controllers with no history wired (tests, embedders) send as before."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    assert controller.prompt_history is None
    result = await controller.submit_draft("hello")
    assert result.accepted


# -- task-1337: per-run Library/RAG provider factory seam --


class _AllowedLibraryCoordinator:
    def register_holder(self, *_args, **_kwargs):
        return None

    def unregister_holder(self, *_args, **_kwargs):
        return None

    async def capture_for_execution(self, _session_id):
        return ConsoleLibraryPolicySnapshot(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
            policy_revision=1,
            source="durable",
        )


class _ControllerLibraryService:
    def invoke(self, _name, _arguments):
        return {"items": [], "total": 0}


@pytest.mark.asyncio
async def test_run_agent_reply_threads_library_provider_from_factory():
    """The controller resolves the injected `library_provider_factory` exactly
    once per run, on the main loop, and hands the resulting provider to the
    bridge's run_reply alongside the other per-run providers."""
    store = ConsoleChatStore()
    store.library_policy_coordinator = _AllowedLibraryCoordinator()
    gateway = RecordingStreamingGateway()
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    provider = LibraryToolProvider(_ControllerLibraryService())
    factory_calls = []

    def factory(context):
        factory_calls.append(context)
        return provider

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=True,
        library_provider_factory=factory,
    )
    bridge_calls = []

    def run_reply(**kwargs):
        bridge_calls.append(kwargs)
        return "run-test", RunOutcome(status=RUN_DONE, steps=[], final_text="ok")

    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)
    _arm_session(store)

    await controller.submit_draft("hello")

    assert len(factory_calls) == 1
    assert len(bridge_calls) == 1
    assert bridge_calls[0]["library_provider"] is provider
    assert provider.authenticates_builtin_authority(
        bridge_calls[0]["library_authority"]
    )


@pytest.mark.asyncio
async def test_run_agent_reply_without_factory_passes_no_library_provider():
    """Default construction (no factory) keeps the pre-task-1337 handoff:
    run_reply receives `library_provider=None`."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, agent_runtime_enabled=True
    )
    bridge_calls = []

    def run_reply(**kwargs):
        bridge_calls.append(kwargs)
        return "run-test", RunOutcome(status=RUN_DONE, steps=[], final_text="ok")

    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)
    _arm_session(store)

    await controller.submit_draft("hello")

    assert len(bridge_calls) == 1
    assert bridge_calls[0]["library_provider"] is None


@pytest.mark.asyncio
async def test_library_provider_factory_refreshes_per_run_without_rebuilding_bridge():
    """Per-run freshness issues a new provider/authority on the cached bridge."""
    store = ConsoleChatStore()
    store.library_policy_coordinator = _AllowedLibraryCoordinator()
    gateway = RecordingStreamingGateway()
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    first_provider = LibraryToolProvider(_ControllerLibraryService())
    second_provider = LibraryToolProvider(_ControllerLibraryService())
    offerings = [first_provider, second_provider]

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=True,
        library_provider_factory=lambda _context: offerings.pop(0),
    )
    bridge_calls = []

    def run_reply(**kwargs):
        bridge_calls.append(kwargs)
        return "run-test", RunOutcome(status=RUN_DONE, steps=[], final_text="ok")

    cached_bridge = SimpleNamespace(run_reply=run_reply)
    controller._agent_bridge = cached_bridge
    _arm_session(store)

    await controller.submit_draft("first")
    await controller.submit_draft("second")

    assert len(bridge_calls) == 2
    assert bridge_calls[0]["library_provider"] is first_provider
    assert bridge_calls[1]["library_provider"] is second_provider
    assert first_provider.authenticates_builtin_authority(
        bridge_calls[0]["library_authority"]
    )
    assert second_provider.authenticates_builtin_authority(
        bridge_calls[1]["library_authority"]
    )
    assert controller._agent_bridge is cached_bridge


@pytest.mark.asyncio
async def test_library_provider_factory_failure_degrades_to_no_provider():
    """A raising factory must never break a send: the run proceeds with
    `library_provider=None` (no Library tools that run)."""
    store = ConsoleChatStore()
    gateway = RecordingStreamingGateway()

    def factory(_context):
        raise RuntimeError("config exploded")

    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=True,
        library_provider_factory=factory,
    )
    bridge_calls = []

    def run_reply(**kwargs):
        bridge_calls.append(kwargs)
        return "run-test", RunOutcome(status=RUN_DONE, steps=[], final_text="ok")

    controller._agent_bridge = SimpleNamespace(run_reply=run_reply)
    _arm_session(store)

    result = await controller.submit_draft("hello")

    assert result.accepted
    assert bridge_calls[0]["library_provider"] is None


# ---------------------------------------------------------------------------
# task-1337, plan Task 8: Console MCP bypass prevention
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compose_mcp_provider_excludes_console_shadowed_builtin_names():
    """The Console-composed MCP provider drops exactly the 29 shadowed raw
    names (24 descriptor tools + 5 legacy readers) from the
    `builtin:tldw_chatbook` source -- the Console serves Library retrieval
    through its own direct/RAG provider (either mode), so the MCP copies
    would be an ungoverned duplicate. Same-named external/local profile
    tools stay; the unrelated built-in stays; the exclusion set is
    mode-independent (`_compose_mcp_provider` takes no mode argument)."""
    from Tests.Agents.test_mcp_tool_provider import (
        FakeMCPService,
        _catalog_record,
        _tool_dict,
    )
    from tldw_chatbook.Chat.console_chat_controller import (
        CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS,
    )
    from tldw_chatbook.Library.library_tool_contract import LIBRARY_TOOL_DESCRIPTORS

    # The exclusion set is exactly descriptor names + the five legacy names;
    # the legacy names must NEVER join the shared descriptor table.
    assert CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS == frozenset(
        set(LIBRARY_TOOL_DESCRIPTORS)
        | {
            "search_rag",
            "search_notes",
            "search_conversations",
            "get_conversation_history",
            "export_conversation",
        }
    )
    assert len(CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS) == 29
    assert "search_rag" not in LIBRARY_TOOL_DESCRIPTORS

    inventory = {
        "tools": [
            *(
                _tool_dict(name)
                for name in sorted(CONSOLE_MCP_BUILTIN_RAW_NAME_EXCLUSIONS)
            ),
            _tool_dict("chat_with_llm"),
        ]
    }
    service = FakeMCPService(
        inventory=inventory,
        catalog_records=[_catalog_record("docs", [_tool_dict("library_list_media")])],
    )
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    controller.app = SimpleNamespace(unified_mcp_service=service)

    provider = await controller._compose_mcp_provider()

    assert provider is not None
    names = {entry.name for entry in provider.list_catalog()}
    assert names == {
        "mcp__tldw_chatbook__chat_with_llm",
        "mcp__docs__library_list_media",
    }


# -----------------------------------------------------------------------------
# PR3a-1 Task 6b (audit F3): a surviving child's spend must not vanish silently
# -----------------------------------------------------------------------------
#
# The agent path attaches usage exactly ONCE, the instant `run_reply` returns.
# A fleet child that outlives its turn keeps streaming into the SAME
# `ConsoleProviderStreamSignals`, so every payload it closes out afterwards is
# appended to an object nobody reads again: the user is billed, and the chip
# and the message row never show it. Re-attaching needs a "last child done"
# signal the bridge does not emit (PR 3a-2 builds it for auto-wake), so 3a-1's
# job is to make the loss OBSERVABLE, not to fix it.


@pytest.mark.asyncio
async def test_a_survivors_post_turn_spend_is_readable_not_silently_dropped():
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderStreamSignals,
    )

    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    signals = ConsoleProviderStreamSignals()
    signals.record_usage_payload({"prompt_tokens": 100, "completion_tokens": 20})
    signals.close_usage_call()
    resolution = SimpleNamespace(provider="openai", model="gpt-4o")

    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="done")
    await controller._finalize_agent_reply(
        placeholder.id,
        session.id,
        outcome,
        variant_mode=False,
        stream_signals=signals,
        resolution=resolution,
    )

    assert store.get_message(placeholder.id).usage.total_tokens == 120
    assert controller.unattributed_fleet_tokens(session.id) == 0

    # The turn is over. The survivor makes one more provider call.
    signals.record_usage_payload({"prompt_tokens": 40, "completion_tokens": 5})
    signals.close_usage_call()

    # Pinned as the KNOWN 3a-1 limitation, not as desirable: the message
    # row is deliberately not re-attached here (that is 3a-2's signal).
    assert store.get_message(placeholder.id).usage.total_tokens == 120

    assert controller.unattributed_fleet_tokens(session.id) == 45, (
        "the survivor's spend was billed and is visible nowhere"
    )


@pytest.mark.asyncio
async def test_re_attaching_the_same_signals_is_idempotent():
    """Recorded for PR 3a-2, which will do exactly this on last-child-done.

    `_attach_stream_usage` recomputes the TOTAL from every payload and
    `set_message_usage` REPLACES -- so a second attach is a replace, not an
    add. 3a-2 inherits a safe path; this pins it so a later refactor to
    accumulate-in-place cannot quietly double-bill.
    """
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderStreamSignals,
    )

    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    placeholder = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )

    signals = ConsoleProviderStreamSignals()
    signals.record_usage_payload({"prompt_tokens": 100, "completion_tokens": 20})
    signals.close_usage_call()
    resolution = SimpleNamespace(provider="openai", model="gpt-4o")

    outcome = RunOutcome(status=RUN_DONE, steps=[], final_text="done")
    await controller._finalize_agent_reply(
        placeholder.id,
        session.id,
        outcome,
        variant_mode=False,
        stream_signals=signals,
        resolution=resolution,
    )
    assert store.get_message(placeholder.id).usage.total_tokens == 120

    signals.record_usage_payload({"prompt_tokens": 40, "completion_tokens": 5})
    signals.close_usage_call()
    controller._attach_stream_usage(placeholder.id, signals, resolution, partial=False)

    assert store.get_message(placeholder.id).usage.total_tokens == 165, (
        "a second attach must REPLACE with the recomputed total, not add"
    )


@pytest.mark.asyncio
async def test_unattributed_fleet_tokens_is_zero_for_an_unwatched_session():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=StreamingGateway())
    session = store.ensure_session()
    assert controller.unattributed_fleet_tokens(session.id) == 0
    assert controller.unattributed_fleet_tokens("no-such-session") == 0
