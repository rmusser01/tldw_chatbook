"""Controller-level tests for `/rewind` "Summarize up to here" (SP2, Task 3).

Covers ``ConsoleChatController.summarize_up_to`` (gates, span construction,
rolling re-summarize, provider call, storage) and the dispatch-choke-point
``_apply_context_summary_compaction`` (the leak rule: compact only when the
boundary message is present in the payload). Reuses the fake-gateway harness
shape from ``test_console_regenerate_branching.py``.
"""

import asyncio
from dataclasses import replace

import pytest

from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleVariantSet,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleMemorySelectionRecord,
    MemorySelectionKind,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_history_budget import bound_messages_to_window
from tldw_chatbook.Chat.console_prepared_request import (
    PreparedConsoleRequest,
    PreparedProviderRequest,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionResult,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationResult,
    ContinuationRound,
    ProviderContinuationCheckpoint,
    dump_provider_continuation_json,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.console_provider_doubles import provider_resolution


class SummaryGateway:
    """Fake gateway that returns a fixed summary and captures the sent payload."""

    def __init__(
        self,
        summary: str = "SUMMARY TEXT",
        ready: bool = True,
        *,
        context_window_tokens: int = 50_000,
    ) -> None:
        self.summary = summary
        self.ready = ready
        self.captured_messages = None
        self.captured_auxiliary = None
        self.calls = 0
        self.context_window_tokens = context_window_tokens
        self.block_auxiliary = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def resolve_for_send(self, selection):
        destination = provider_resolution(
            ready=True,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
        ).resolved_destination
        return ConsoleProviderResolution(
            ready=self.ready,
            provider="llama_cpp",
            model="test-model",
            base_url="http://127.0.0.1:9099",
            max_tokens=512,
            visible_copy="" if self.ready else "Provider blocked: no key.",
            resolved_destination=destination if self.ready else None,
        )

    async def stream_chat(self, resolution, messages, **kwargs):
        self.captured_messages = messages
        for chunk in _as_chunks(self.summary):
            if chunk:
                yield chunk

    def prepare_chat_request(
        self,
        resolution,
        messages,
        *,
        tools=None,
        apply_safety_window=True,
        **_kwargs,
    ):
        semantic = (
            messages
            if isinstance(messages, PreparedConsoleRequest)
            else build_console_request(messages, tools=tools or ())
        )
        return prepare_provider_request(
            semantic,
            wire_style="single_preamble",
            model=resolution.model or "gpt-test",
            provider=resolution.provider,
            capacity=resolve_request_capacity(
                context_window_tokens=self.context_window_tokens,
                requested_response_tokens=resolution.max_tokens or 512,
            ),
            count_fn=lambda rows, _model: sum(
                len(str(row.get("content", "")).split()) + 2 for row in rows
            ),
            apply_safety_window=apply_safety_window,
        )

    async def complete_auxiliary(self, request):
        self.calls += 1
        self.captured_auxiliary = request
        self.started.set()
        if self.block_auxiliary:
            await self.release.wait()
        return AuxiliaryCompletionResult(
            provider=request.resolution.provider,
            model=request.resolution.model or "gpt-test",
            text=self.summary,
            usage=ProviderUsage(
                uncached_input=20,
                output=max(1, len(self.summary.split())),
                provider=request.resolution.provider,
                model=request.resolution.model or "gpt-test",
            ),
        )


def _as_chunks(text: str):
    # Emit in two pieces to exercise chunk accumulation (mirrors a real stream).
    if not text:
        return []
    mid = max(1, len(text) // 2)
    return [text[:mid], text[mid:]]


def _seed_conversation(store, session_id):
    """Append U1/A1/U2/A2/U3/A3 and return the six messages."""
    u1 = store.append_message(session_id, role=ConsoleMessageRole.USER, content="q1")
    a1 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a1"
    )
    u2 = store.append_message(session_id, role=ConsoleMessageRole.USER, content="q2")
    a2 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a2"
    )
    u3 = store.append_message(session_id, role=ConsoleMessageRole.USER, content="q3")
    a3 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a3"
    )
    return u1, a1, u2, a2, u3, a3


def _seed_durable_conversation(store, session_id):
    """Append the same three exchanges through the durable write path."""
    rows = []
    for role, content in (
        (ConsoleMessageRole.USER, "q1 " + "detail " * 20),
        (ConsoleMessageRole.ASSISTANT, "a1 " + "detail " * 20),
        (ConsoleMessageRole.USER, "q2 " + "detail " * 20),
        (ConsoleMessageRole.ASSISTANT, "a2 " + "detail " * 20),
        (ConsoleMessageRole.USER, "q3 " + "detail " * 20),
        (ConsoleMessageRole.ASSISTANT, "a3 " + "detail " * 20),
    ):
        rows.append(
            store.append_message(
                session_id,
                role=role,
                content=content,
                persist=True,
            )
        )
    return tuple(rows)


def _durable_controller(tmp_path, *, gateway=None):
    db = CharactersRAGDB(tmp_path / "rewind-summary.sqlite", "rewind-summary")
    store = ConsoleChatStore(persistence=ChatPersistenceService(db))
    session = store.create_session()
    store.persist_session_if_needed(session.id)
    resolved_gateway = gateway or SummaryGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=resolved_gateway,
    )
    return controller, store, session, resolved_gateway, db


def _completed_tool_checkpoint(
    final_content: str,
    *,
    revision: int = 1,
    city: str = "Paris",
    result: str = "sunny",
) -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=revision,
        provider="moonshot",
        protocol="chat_completions",
        model="kimi-k3",
        api_base_url="https://api.moonshot.ai/v1",
        state="complete",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("tool reasoning",),
                calls=(
                    ContinuationCall(
                        call_id="call_weather",
                        name="weather_lookup",
                        arguments=f'{{"city":"{city}"}}',
                        state="completed",
                        result=ContinuationResult(result),
                    ),
                ),
            ),
            ContinuationRound(
                assistant_content=final_content,
                reasoning_blocks=("final reasoning",),
                calls=(),
            ),
        ),
    )


def _install_terminal_continuation(owner, db, checkpoint, *, expected_version: int):
    canonical = dump_provider_continuation_json(checkpoint)
    assert canonical is not None
    assert owner.persisted_message_id is not None
    assert db.update_provider_continuation(
        message_id=owner.persisted_message_id,
        expected_message_version=expected_version,
        provider_continuation_json=canonical,
        content=checkpoint.rounds[-1].assistant_content,
        assistant_generation_state="complete",
    )
    owner.provider_continuation = checkpoint
    owner.provider_continuation_message_version = expected_version + 1
    owner.provider_continuation_actions_enabled = False
    owner.assistant_generation_state = "complete"
    owner.content = checkpoint.rounds[-1].assistant_content


@pytest.mark.asyncio
async def test_summarize_from_commits_exact_inclusive_manual_range(tmp_path):
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="S")
    )
    rows = _seed_durable_conversation(store, session.id)

    result = await controller.summarize_from(rows[2].id)

    assert result.accepted is True
    assert gateway.calls == 1
    payload = gateway.captured_auxiliary.messages[1]["content"]
    assert '"content":"q1 ' not in payload
    assert '"content":"a1 ' not in payload
    assert all(f'"content":"{text} ' in payload for text in ("q2", "a2", "q3", "a3"))
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    memories = controller._context_repository.list_active_memories(conversation_id)
    assert len(memories) == 1
    scope = controller._context_repository.load_memory_scope(memories[0].memory_id)
    assert scope is not None
    assert scope.coverage_kind.value == "range"
    assert scope.selection_anchor_message_id == rows[2].persisted_message_id
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_manual_summary_position_zero_attachment_label_does_not_false_stale(
    tmp_path,
):
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="S")
    )
    u1 = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="q1 " + "detail " * 20,
        attachments=(
            MessageAttachment(
                data=b"png-facts",
                mime_type="image/png",
                display_name="facts.png",
                position=0,
            ),
        ),
        persist=True,
    )
    for role, content in (
        (ConsoleMessageRole.ASSISTANT, "a1 " + "detail " * 20),
        (ConsoleMessageRole.USER, "q2 " + "detail " * 20),
        (ConsoleMessageRole.ASSISTANT, "a2 " + "detail " * 20),
    ):
        store.append_message(
            session.id, role=role, content=content, persist=True
        )

    result = await controller.summarize_from(u1.id)

    assert result.accepted is True
    assert gateway.calls == 1
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    assert len(controller._context_repository.list_active_memories(conversation_id)) == 1


@pytest.mark.asyncio
async def test_manual_summary_projects_durable_tool_call_result_envelope(tmp_path):
    controller, store, session, gateway, db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="S")
    )
    rows = _seed_durable_conversation(store, session.id)
    owner = store._nodes_by_session[session.id][rows[1].id]
    checkpoint = _completed_tool_checkpoint(rows[1].content)
    _install_terminal_continuation(
        owner,
        db,
        checkpoint,
        expected_version=1,
    )

    result = await controller.summarize_up_to(rows[2].id)

    assert result.accepted is True
    payload = gateway.captured_auxiliary.messages[1]["content"]
    assert '"tool_calls":[{"function"' in payload
    assert '"id":"call_weather"' in payload
    assert '"tool_call_id":"call_weather"' in payload
    assert '"content":"sunny"' in payload


@pytest.mark.asyncio
async def test_manual_summary_rejects_stale_runtime_continuation_before_call(tmp_path):
    controller, store, session, gateway, db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="S")
    )
    rows = _seed_durable_conversation(store, session.id)
    owner = store._nodes_by_session[session.id][rows[1].id]
    runtime_checkpoint = _completed_tool_checkpoint(rows[1].content)
    _install_terminal_continuation(
        owner,
        db,
        runtime_checkpoint,
        expected_version=1,
    )
    durable_checkpoint = _completed_tool_checkpoint(
        "new durable answer",
        revision=2,
        city="Rome",
        result="rainy",
    )
    canonical = dump_provider_continuation_json(durable_checkpoint)
    assert canonical is not None
    assert owner.persisted_message_id is not None
    assert db.update_provider_continuation(
        message_id=owner.persisted_message_id,
        expected_message_version=2,
        provider_continuation_json=canonical,
        content="new durable answer",
        assistant_generation_state="complete",
    )

    result = await controller.summarize_up_to(rows[2].id)

    assert result.accepted is False
    assert gateway.calls == 0
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    assert controller._context_repository.list_active_memories(conversation_id) == ()
    assert (
        controller._context_repository.list_active_memory_selections(conversation_id)
        == ()
    )
    assert store.session_context_summary(session.id) == (None, None)


def _seed_fenced_durable_conversation(store, session_id, db):
    rows = [
        store.append_message(
            session_id,
            role=ConsoleMessageRole.USER,
            content="q1 " + "detail " * 20,
            attachments=(
                MessageAttachment(
                    data=b"initial-facts",
                    mime_type="image/png",
                    display_name="facts.png",
                    position=0,
                ),
            ),
            persist=True,
        )
    ]
    for role, content in (
        (ConsoleMessageRole.ASSISTANT, "a1 " + "detail " * 20),
        (ConsoleMessageRole.USER, "q2 " + "detail " * 20),
        (ConsoleMessageRole.ASSISTANT, "a2 " + "detail " * 20),
        (ConsoleMessageRole.USER, "q3 " + "detail " * 20),
        (ConsoleMessageRole.ASSISTANT, "a3 " + "detail " * 20),
    ):
        rows.append(
            store.append_message(
                session_id,
                role=role,
                content=content,
                persist=True,
            )
        )
    owner = store._nodes_by_session[session_id][rows[1].id]
    _install_terminal_continuation(
        owner,
        db,
        _completed_tool_checkpoint(rows[1].content),
        expected_version=1,
    )
    return tuple(rows)


def _manual_persistence_state(controller, store, session, db):
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    repository = controller._context_repository
    memories = repository.list_active_memories(conversation_id)
    return (
        memories,
        tuple(repository.load_memory_scope(memory.memory_id) for memory in memories),
        repository.list_active_memory_selections(conversation_id),
        store.session_context_summary(session.id),
        db.get_conversation_context_summary(conversation_id),
    )


def _mutate_manual_fence(
    name,
    *,
    controller,
    store,
    session,
    rows,
    db,
    monkeypatch,
):
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    nodes = store._nodes_by_session[session.id]
    if name == "session":
        store.create_session(activate=True)
    elif name == "cursor":
        assert db.set_conversation_active_cursor(
            conversation_id,
            active_leaf_message_id=rows[-2].persisted_message_id,
            before_message_id=None,
        )
    elif name == "ordered_lineage_leaf":
        store._active_leaf_by_session[session.id] = rows[3].id
        store._recompute_active_path(session.id)
    elif name == "payload_revision":
        store._bump_payload_revision(session.id)
    elif name == "identity_revision":
        session.identity_revision += 1
    elif name == "conversation_policy":
        session.context_policy_overrides = ConsoleContextPolicyOverrides(
            summary_max_tokens=257
        )
    elif name == "global_policy":
        monkeypatch.setattr(
            controller,
            "_global_context_policy_overrides",
            lambda: ConsoleContextPolicyOverrides(summary_max_tokens=257),
        )
    elif name == "persisted_policy_revision":
        assert (
            controller._context_repository.save_policy(
                conversation_id,
                ConsoleContextPolicyOverrides(summary_max_tokens=257),
            )
            == 1
        )
    elif name == "provider_model_configuration":
        session.settings = ConsoleSessionSettings(
            provider="llama_cpp",
            model="changed-model",
            base_url="http://127.0.0.1:9191",
        )
    elif name == "prompt_digest":
        monkeypatch.setattr(
            "tldw_chatbook.Chat.console_chat_controller.get_internal_prompt",
            lambda _prompt_id: "Changed rewind summary prompt.",
        )
    elif name == "effective_selection_head":
        controller._context_repository.insert_memory_selection(
            ConsoleMemorySelectionRecord(
                sequence=1,
                selection_id="concurrent-reset",
                conversation_id=conversation_id,
                activation_message_id=rows[-1].persisted_message_id,
                selected_memory_id=None,
                event_kind=MemorySelectionKind.RESET,
                suppresses_legacy=True,
                created_at="2026-08-29T00:00:00+00:00",
            )
        )
    elif name == "legacy_digest":
        store.set_session_context_summary(session.id, "new legacy", rows[1].id)
    elif name == "message_version":
        assert rows[2].persisted_message_id is not None
        assert db.update_message(
            rows[2].persisted_message_id,
            {"content": rows[2].content},
            expected_version=1,
            preserve_descendants=True,
        )
    elif name == "parent":
        nodes[rows[3].id].parent_message_id = rows[0].persisted_message_id
    elif name == "status_visibility":
        nodes[rows[3].id].status = "failed"
    elif name == "deletion":
        assert rows[0].persisted_message_id is not None
        store.persistence.delete_message_subtree(
            message_id=rows[0].persisted_message_id
        )
    elif name == "variant":
        nodes[rows[3].id].variants = ConsoleVariantSet.from_contents(
            turn_id=rows[3].id,
            contents=[rows[3].content, "changed selected variant"],
            selected_index=1,
        )
    elif name == "attachment":
        ConsoleChatStore._set_message_attachments(
            nodes[rows[0].id],
            (
                MessageAttachment(
                    data=b"changed-facts",
                    mime_type="image/png",
                    display_name="facts.png",
                    position=0,
                ),
            ),
        )
    elif name == "tool_envelope":
        owner = nodes[rows[1].id]
        checkpoint = owner.provider_continuation
        assert checkpoint is not None
        first_round = checkpoint.rounds[0]
        changed_call = replace(
            first_round.calls[0],
            result=ContinuationResult("changed runtime result"),
        )
        owner.provider_continuation = replace(
            checkpoint,
            rounds=(
                replace(first_round, calls=(changed_call,)),
                *checkpoint.rounds[1:],
            ),
        )
    elif name == "start_anchor":
        nodes[rows[4].id].persisted_message_id = rows[2].persisted_message_id
    elif name == "end_anchor":
        nodes[rows[-1].id].persisted_message_id = rows[3].persisted_message_id
    elif name == "durable_content":
        nodes[rows[2].id].content += " changed"
    else:  # pragma: no cover - parameter table and hook must stay in lockstep
        raise AssertionError(name)


@pytest.mark.asyncio
async def test_summarize_up_to_preserves_raw_prefix_larger_than_12k(tmp_path):
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path,
        gateway=SummaryGateway(summary="S", context_window_tokens=80_000),
    )
    earliest = "EARLIEST_COVERED_UNIT"
    latest = "LATEST_COVERED_UNIT"
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content=earliest + " " + "alpha " * 2_500,
        persist=True,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="first answer " + "answer " * 100,
        persist=True,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content=latest + " " + "beta " * 100,
        persist=True,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="latest answer " + "answer " * 100,
        persist=True,
    )
    target = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="target prompt",
        persist=True,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="target answer",
        persist=True,
    )

    result = await controller.summarize_up_to(target.id)

    assert result.accepted is True
    assert gateway.calls == 1
    raw_input = gateway.captured_auxiliary.messages[1]["content"]
    assert len(raw_input) > 12_000
    assert earliest in raw_input
    assert latest in raw_input


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fence_name",
    [
        "session",
        "cursor",
        "ordered_lineage_leaf",
        "payload_revision",
        "identity_revision",
        "conversation_policy",
        "global_policy",
        "persisted_policy_revision",
        "provider_model_configuration",
        "prompt_digest",
        "effective_selection_head",
        "legacy_digest",
        "message_version",
        "parent",
        "status_visibility",
        "deletion",
        "variant",
        "attachment",
        "tool_envelope",
        "start_anchor",
        "end_anchor",
        "durable_content",
    ],
)
async def test_blocked_manual_summary_discards_every_controller_fence_mutation(
    tmp_path,
    monkeypatch,
    fence_name,
):
    gateway = SummaryGateway(summary="S")
    gateway.block_auxiliary = True
    gateway.release.clear()
    controller, store, session, _gateway, db = _durable_controller(
        tmp_path,
        gateway=gateway,
    )
    rows = _seed_fenced_durable_conversation(store, session.id, db)

    pending = asyncio.create_task(controller.summarize_up_to(rows[4].id))
    await gateway.started.wait()
    _mutate_manual_fence(
        fence_name,
        controller=controller,
        store=store,
        session=session,
        rows=rows,
        db=db,
        monkeypatch=monkeypatch,
    )
    state_after_external_mutation = _manual_persistence_state(
        controller, store, session, db
    )
    gateway.release.set()
    result = await pending

    assert result.accepted is False
    assert "changed while summarizing" in result.visible_copy
    assert gateway.calls == 1
    assert (
        _manual_persistence_state(controller, store, session, db)
        == state_after_external_mutation
    )


@pytest.mark.asyncio
async def test_old_effective_memory_remains_selected_while_manual_call_awaits(tmp_path):
    gateway = SummaryGateway(summary="S")
    controller, store, session, _gateway, db = _durable_controller(
        tmp_path,
        gateway=gateway,
    )
    rows = _seed_durable_conversation(store, session.id)
    store.set_session_context_summary(session.id, "legacy memory", rows[1].id)
    first = await controller.summarize_up_to(rows[2].id)
    assert first.accepted is True
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    old_memory = controller._context_repository.list_active_memories(conversation_id)[0]
    old_snapshots = controller._durable_context_snapshots(session.id)
    assert old_snapshots is not None
    old_fences = controller._manual_branch_fences(
        session_id=session.id,
        snapshots=old_snapshots,
    )
    assert old_fences is not None
    assert old_fences[0].memory_id == old_memory.memory_id
    assert old_fences[1].memory_id == old_memory.memory_id

    gateway.block_auxiliary = True
    gateway.started.clear()
    gateway.release.clear()
    pending = asyncio.create_task(controller.summarize_up_to(rows[4].id))
    await gateway.started.wait()

    during_snapshots = controller._durable_context_snapshots(session.id)
    assert during_snapshots is not None
    during_fences = controller._manual_branch_fences(
        session_id=session.id,
        snapshots=during_snapshots,
    )
    assert during_fences == old_fences
    assert store.session_context_summary(session.id) == (
        "legacy memory",
        rows[1].id,
    )
    assert len(controller._context_repository.list_active_memories(conversation_id)) == 1
    assert len(
        controller._context_repository.list_active_memory_selections(conversation_id)
    ) == 1

    gateway.release.set()
    result = await pending

    assert result.accepted is True
    new_snapshots = controller._durable_context_snapshots(session.id)
    assert new_snapshots is not None
    new_fences = controller._manual_branch_fences(
        session_id=session.id,
        snapshots=new_snapshots,
    )
    assert new_fences is not None
    assert new_fences[0].memory_id != old_memory.memory_id
    assert new_fences[1].memory_id != old_memory.memory_id
    assert store.session_context_summary(session.id) == (
        "legacy memory",
        rows[1].id,
    )


# --------------------------------------------------------------------------
# summarize_up_to gates + storage
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_up_to_commits_manual_prefix_without_legacy_write(tmp_path):
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="S")
    )
    rows = _seed_durable_conversation(store, session.id)

    result = await controller.summarize_up_to(rows[4].id)

    assert result.accepted is True
    assert gateway.calls == 1
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    memory = controller._context_repository.list_active_memories(conversation_id)[0]
    scope = controller._context_repository.load_memory_scope(memory.memory_id)
    assert scope is not None
    assert scope.coverage_kind.value == "prefix"
    assert scope.selection_anchor_message_id == rows[4].persisted_message_id
    assert memory.boundary_message_id == rows[3].persisted_message_id
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_up_to_uses_exact_strict_complete_prefix(tmp_path):
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="S")
    )
    rows = _seed_durable_conversation(store, session.id)

    result = await controller.summarize_up_to(rows[2].id)

    assert result.accepted is True
    payload = gateway.captured_auxiliary.messages[1]["content"]
    assert '"content":"q1 ' in payload
    assert '"content":"a1 ' in payload
    assert all(
        f'"content":"{text} ' not in payload for text in ("q2", "a2", "q3", "a3")
    )


@pytest.mark.asyncio
async def test_summarize_rejects_non_provider_visible_prefix_without_call(tmp_path):
    """task-2391 fix-now (audit follow-up): a committed voice turn whose
    transcript came back empty persists a real placeholder as CONTENT
    ("(no speech detected)") -- UI chrome written so the row could exist at
    all, not something the user said. `summarize_up_to` sends the raw span
    straight to a real provider call (`_collect_summary_completion`), so
    leaving the placeholder in would fabricate a user turn in the
    SUMMARIZER's context too -- the same defect `_provider_message_
    payloads` had before its fix, one layer removed."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.UI.Console_Modules.realtime import (
        CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
    )

    controller, store, session, gateway, _db = _durable_controller(tmp_path)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content=CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
        metadata=MessageMetadata(engine="realtime", transcript_status="empty"),
        persist=True,
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="a1", persist=True
    )
    u2 = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="q2", persist=True
    )
    store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="a2", persist=True
    )

    result = await controller.summarize_up_to(u2.id)

    assert result.accepted is False
    assert gateway.calls == 0


@pytest.mark.asyncio
async def test_summarize_nothing_before_target_when_only_prior_is_empty_transcript(
    tmp_path,
):
    """The "nothing to summarize" gate must see the empty-transcript row as
    absent too -- otherwise it would proceed to send an empty span to the
    provider instead of the honest block."""
    from tldw_chatbook.Chat.message_metadata import MessageMetadata
    from tldw_chatbook.UI.Console_Modules.realtime import (
        CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
    )

    controller, store, session, gateway, _db = _durable_controller(tmp_path)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content=CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER,
        metadata=MessageMetadata(engine="realtime", transcript_status="empty"),
        persist=True,
    )
    u2 = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="q2", persist=True
    )

    result = await controller.summarize_up_to(u2.id)

    assert result.accepted is False
    assert gateway.calls == 0
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_provider_not_ready_blocks_and_stores_nothing(tmp_path):
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(ready=False)
    )
    rows = _seed_durable_conversation(store, session.id)

    result = await controller.summarize_up_to(rows[2].id)

    assert result.accepted is False
    assert gateway.calls == 0
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_non_user_target_blocks_and_stores_nothing():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    _u1, a1, *_rest = _seed_conversation(store, session.id)

    result = await controller.summarize_up_to(a1.id)

    assert result.accepted is False
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_off_path_target_blocks_and_stores_nothing():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)

    # Move the active leaf back so u2 falls off the active path.
    store.set_active_leaf(session.id, a1.id)
    assert u2.id not in store.active_path_message_ids(session.id)

    result = await controller.summarize_up_to(u2.id)

    assert result.accepted is False
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_incomplete_first_prompt_blocks_without_call(tmp_path):
    controller, store, session, gateway, _db = _durable_controller(tmp_path)
    u1 = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="only prompt", persist=True
    )

    result = await controller.summarize_up_to(u1.id)

    assert result.accepted is False
    assert gateway.calls == 0
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_empty_reply_stores_nothing(tmp_path):
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="")
    )
    rows = _seed_durable_conversation(store, session.id)

    result = await controller.summarize_up_to(rows[4].id)

    assert result.accepted is False
    assert gateway.calls == 1
    assert store.session_context_summary(session.id) == (None, None)


@pytest.mark.asyncio
async def test_summarize_up_to_never_folds_or_rewrites_legacy_memory(tmp_path):
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="S2")
    )
    rows = _seed_durable_conversation(store, session.id)
    store.set_session_context_summary(session.id, "S1", rows[1].id)

    result = await controller.summarize_up_to(rows[4].id)

    assert result.accepted is True
    payload = gateway.captured_auxiliary.messages[1]["content"]
    assert "S1" not in payload
    assert all(
        f'"content":"{text} ' in payload for text in ("q1", "a1", "q2", "a2")
    )
    assert store.session_context_summary(session.id) == ("S1", rows[1].id)


# --------------------------------------------------------------------------
# choke-point compaction + THE LEAK RULE
# --------------------------------------------------------------------------


def _payload_texts(messages):
    texts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.append(
                "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            )
    return texts


@pytest.mark.asyncio
async def test_compaction_folds_summary_and_drops_pre_boundary_rows():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    controller.system_prompt = "You are helpful."
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)
    store.set_session_context_summary(session.id, "S", u3.id)

    # Compaction anchors the boundary by native id, so the payload must be
    # built id-annotated (as every real send path does).
    payload = controller._provider_messages_for_session(session.id, annotate_ids=True)
    compacted = controller._apply_context_summary_compaction(session.id, payload)

    texts = _payload_texts(compacted)
    # Pre-boundary turns gone, boundary + tail kept.
    assert "q1" not in texts and "a1" not in texts
    assert "q2" not in texts and "a2" not in texts
    assert "q3" in texts and "a3" in texts
    # Summary folded into the leading system prefix.
    assert compacted[0]["role"] == "system"
    assert "You are helpful." in compacted[0]["content"]
    assert "[Summary of earlier conversation]" in compacted[0]["content"]
    assert "S" in compacted[0]["content"]

    # The trimmer preserves the leading system prefix (summary survives).
    bound = bound_messages_to_window(
        compacted, model="test-model", provider="llama_cpp", response_reservation=256
    )
    assert bound.messages[0]["role"] == "system"
    assert "[Summary of earlier conversation]" in bound.messages[0]["content"]


@pytest.mark.asyncio
async def test_compaction_creates_system_message_when_payload_has_none():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)
    store.set_session_context_summary(session.id, "S", u3.id)

    payload = controller._provider_messages_for_session(session.id, annotate_ids=True)
    assert payload[0]["role"] != "system"  # no system prompt set

    compacted = controller._apply_context_summary_compaction(session.id, payload)

    assert compacted[0]["role"] == "system"
    assert "[Summary of earlier conversation]" in compacted[0]["content"]
    assert "S" in compacted[0]["content"]
    texts = _payload_texts(compacted)
    assert "q3" in texts and "q1" not in texts


@pytest.mark.asyncio
async def test_leak_rule_pre_boundary_payload_is_byte_identical():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_conversation(store, session.id)

    # Payload for regenerating a PRE-boundary message ends before the boundary.
    pre_boundary_payload = controller._provider_messages_for_session(
        session.id, before_message_id=a1.id, annotate_ids=True
    )

    store.set_session_context_summary(session.id, "S", u3.id)
    compacted = controller._apply_context_summary_compaction(
        session.id,
        controller._provider_messages_for_session(
            session.id, before_message_id=a1.id, annotate_ids=True
        ),
    )

    # The boundary (u3) id is absent from this ancestors-only payload, so
    # compaction is a no-op -- byte-identical to the no-summary payload.
    assert compacted == pre_boundary_payload


@pytest.mark.asyncio
async def test_dangling_boundary_leaves_payload_untouched():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    _seed_conversation(store, session.id)
    # A boundary id that is not a live message (branch switch / deletion).
    store.set_session_context_summary(session.id, "S", "ghost-native-id")

    payload = controller._provider_messages_for_session(session.id, annotate_ids=True)
    compacted = controller._apply_context_summary_compaction(session.id, payload)

    assert compacted == payload


# --------------------------------------------------------------------------
# duplicate-content leak (reviewer repro) + id-anchoring + key stripping
# --------------------------------------------------------------------------


def _seed_duplicate_content(store, session_id):
    """U1/A1/U2/A2/U3(/A3) where U1 and U3 share the exact text "continue"."""
    u1 = store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="continue"
    )
    a1 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a1"
    )
    u2 = store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="different"
    )
    a2 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a2"
    )
    u3 = store.append_message(
        session_id, role=ConsoleMessageRole.USER, content="continue"
    )
    a3 = store.append_message(
        session_id, role=ConsoleMessageRole.ASSISTANT, content="a3"
    )
    return u1, a1, u2, a2, u3, a3


@pytest.mark.asyncio
async def test_leak_rule_duplicate_content_pre_boundary_no_false_fire():
    """Reviewer repro: a byte-identical EARLIER duplicate of the boundary's
    text must NOT false-fire compaction on a pre-boundary payload.

    U1 and the boundary U3 both say "continue". Regenerating pre-boundary A1
    builds an ancestors-only ``[U1]`` payload where the boundary U3 is ABSENT.
    First-occurrence content matching wrongly anchored on U1 and injected the
    summary of LATER turns; id-anchored compaction leaves the payload
    byte-identical to the no-summary payload.
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_duplicate_content(store, session.id)

    baseline = controller._provider_messages_for_session(
        session.id, before_message_id=a1.id, annotate_ids=True
    )
    store.set_session_context_summary(session.id, "S", u3.id)
    compacted = controller._apply_context_summary_compaction(
        session.id,
        controller._provider_messages_for_session(
            session.id, before_message_id=a1.id, annotate_ids=True
        ),
    )

    # No summary folded, no rows dropped -- the LATER-turn summary never reaches
    # this EARLIER point's context.
    assert compacted == baseline
    assert not any(
        "[Summary of earlier conversation]" in text
        for text in _payload_texts(compacted)
    )


@pytest.mark.asyncio
async def test_compaction_anchors_on_boundary_id_not_duplicate_text():
    """Same duplicate-text tree, but the FULL active-path payload DOES contain
    the real boundary U3. Compaction must anchor on U3 by native id (dropping
    U1/A1/U2/A2) even though the earlier U1 shares U3's exact text -- content
    matching would wrongly anchor on U1 and drop nothing.
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session()
    u1, a1, u2, a2, u3, a3 = _seed_duplicate_content(store, session.id)
    store.set_session_context_summary(session.id, "S", u3.id)

    payload = controller._provider_messages_for_session(session.id, annotate_ids=True)
    compacted = controller._apply_context_summary_compaction(session.id, payload)

    texts = _payload_texts(compacted)
    # Everything strictly before the real boundary U3 is dropped: the earlier
    # duplicate "continue" (U1) and the intervening turns are gone.
    assert "different" not in texts
    assert "a1" not in texts and "a2" not in texts
    assert texts.count("continue") == 1  # only the boundary U3 survives
    assert "a3" in texts
    # Summary folded into a leading system row.
    assert compacted[0]["role"] == "system"
    assert "[Summary of earlier conversation]" in compacted[0]["content"]
    assert "S" in compacted[0]["content"]


class _SkillsFake:
    """Minimal fake skills service: resolves `$do-it` to a fixed inline
    render. Mirrors the shape of `test_console_skill_substitution.py`'s
    `_Skills` fake, trimmed to only what this regression needs.
    """

    async def get_context(self, *, mode="local"):
        return {
            "available_skills": [
                {
                    "name": "do-it",
                    "description": "d",
                    "user_invocable": True,
                    "trust_blocked": False,
                }
            ],
            "blocked_skills": [],
        }

    async def execute_skill(self, name, *, mode="local", args=None):
        return {
            "skill_name": name,
            "rendered_prompt": f"RENDERED[{args}]",
            "allowed_tools": None,
            "execution_mode": "inline",
            "fork_output": None,
        }


@pytest.mark.asyncio
async def test_compaction_anchors_after_skill_substitution_inline_rewrite():
    """Regression (review finding): `_apply_skill_substitution`'s non-fork
    rewrite paths must preserve the original row's private keys (via a
    ``{**row, ...}`` spread), exactly like chat-dictionary/world-info do --
    otherwise, when the compaction boundary IS the final user row AND its
    content also resolves to a skill, the inline rewrite silently drops
    ``NATIVE_MESSAGE_ID_KEY`` and the choke point's id match misses (fails
    SAFE to full history, but compaction never applies).
    """
    store = ConsoleChatStore()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SummaryGateway(),
        provider="llama_cpp",
        model="test-model",
        skills_service=_SkillsFake(),
    )
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="a1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q2")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="a2")
    # The boundary is the final user row, and its content resolves to a
    # skill -- the exact overlap the review finding calls out.
    u3 = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="$do-it go"
    )
    store.set_session_context_summary(session.id, "S", u3.id)

    payload = controller._provider_messages_for_session(session.id, annotate_ids=True)
    (
        substituted,
        refuse,
        notes,
        bindings,
        block,
    ) = await controller._apply_skill_substitution(payload)
    assert refuse is None
    assert bindings == ("do-it",)
    assert substituted[-1]["content"] == "RENDERED[go]"

    compacted = controller._apply_context_summary_compaction(session.id, substituted)

    texts = _payload_texts(compacted)
    # Compaction anchored on the (id-preserved) boundary row: pre-boundary
    # turns are dropped and the summary is folded in.
    assert "q1" not in texts and "a1" not in texts
    assert "q2" not in texts and "a2" not in texts
    assert "RENDERED[go]" in texts
    assert compacted[0]["role"] == "system"
    assert "[Summary of earlier conversation]" in compacted[0]["content"]
    assert "S" in compacted[0]["content"]


@pytest.mark.asyncio
async def test_native_message_id_key_stripped_before_provider():
    """The private id-threading key must never reach the provider: after a
    normal compacted send, no captured gateway payload row carries it.
    """
    store = ConsoleChatStore()
    gateway = SummaryGateway(summary="reply")
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
    )
    session = store.create_session(ephemeral=True)
    _u1, _a1, _u2, _a2, u3, _a3 = _seed_conversation(store, session.id)
    store.set_session_context_summary(session.id, "S", u3.id)

    result = await controller.submit_draft("next question")
    assert result.accepted is True

    assert gateway.captured_messages is not None
    captured = (
        gateway.captured_messages.messages
        if isinstance(gateway.captured_messages, PreparedProviderRequest)
        else gateway.captured_messages
    )
    assert all("_native_message_id" not in row for row in captured)
    # Sanity: compaction genuinely ran on this send (summary folded), so the
    # strip assertion above is not vacuous.
    assert any(
        row["role"] == "system"
        and "[Summary of earlier conversation]" in row.get("content", "")
        for row in captured
    )


def test_compacted_summary_precedes_run_local_startup_rider(tmp_path):
    from tldw_chatbook.Agents.agent_models import AgentConfig
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.project_instruction_resolver import (
        InstructionSource,
        StartupInstructionCandidate,
    )
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    calls = []
    service = AgentService(
        AgentRunsDB(tmp_path / "runs.db", client_id="test"),
        ToolCatalogRegistry(),
        chat_call=lambda **kwargs: (
            calls.append(kwargs) or {"choices": [{"message": {"content": "done"}}]}
        ),
        startup_instruction_candidate=StartupInstructionCandidate(
            binding_id="b",
            binding_root=tmp_path,
            locator_fingerprint="f" * 64,
            dispatch_started_wall_ns=1,
            source=InstructionSource(
                canonical_path=tmp_path / "AGENTS.md",
                relative_path="AGENTS.md",
                scope=".",
                kind="standard",
                body="REWIND_RIDER_SENTINEL",
                byte_count=21,
                digest="d" * 64,
            ),
            outcomes=(),
        ),
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
    )
    service.run_turn(
        conversation_id="c",
        messages=[
            {
                "role": "system",
                "content": "[Summary of earlier conversation]\nCOMPACTED",
            },
            {"role": "user", "content": "continue"},
        ],
        config=AgentConfig(
            model="gpt-4o-mini", system_prompt="system", native_tools=False
        ),
        api_endpoint="openai",
    )
    payload = calls[0]["messages_payload"]
    summary_index = next(
        i for i, row in enumerate(payload) if "COMPACTED" in str(row.get("content"))
    )
    rider_index = next(
        i
        for i, row in enumerate(payload)
        if "REWIND_RIDER_SENTINEL" in str(row.get("content"))
    )
    assert summary_index < rider_index


# ---------------------------------------------------------------------------
# task-548: the inspector next-send preview mirrors boundary compaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_reflects_boundary_compaction():
    """With an active summary, build_context_snapshot compacts like a real send."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session(title="Chat 1")

    u1 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="old-q")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="old-a")
    u2 = store.append_message(session.id, role=ConsoleMessageRole.USER, content="new-q")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="new-a")
    store.set_session_context_summary(session.id, "COMPACT-SUMMARY", u2.id)

    snapshot = await controller.build_context_snapshot(draft="")
    rows = snapshot.next_send_payload["messages"]

    # Pre-boundary turns replaced; boundary tail intact.
    contents = [row.get("content") or "" for row in rows]
    assert not any("old-q" in c or "old-a" in c for c in contents)
    assert any("new-q" in c for c in contents)
    assert any("new-a" in c for c in contents)
    # Summary folded into the leading system row AND the duplicated field.
    assert rows[0]["role"] == "system"
    assert "COMPACT-SUMMARY" in rows[0]["content"]
    assert any(
        "COMPACT-SUMMARY" in (row.get("content") or "")
        for row in snapshot.next_send_payload["system"]
    )
    # AC #2: the private id-threading key never reaches the preview.
    assert not any("_native_message_id" in row for row in rows)
    _ = u1


@pytest.mark.asyncio
async def test_snapshot_without_summary_unchanged_and_key_free():
    """No stored summary: preview shows full history and no private keys."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session(title="Chat 1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="a1")

    snapshot = await controller.build_context_snapshot(draft="next")
    rows = snapshot.next_send_payload["messages"]

    contents = [row.get("content") or "" for row in rows]
    assert any("q1" in c for c in contents)
    assert any("a1" in c for c in contents)
    assert not any("_native_message_id" in row for row in rows)


@pytest.mark.asyncio
async def test_snapshot_with_dangling_boundary_shows_full_history():
    """A dangling boundary leaves the preview un-compacted (leak rule parity)."""
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=SummaryGateway())
    session = store.ensure_session(title="Chat 1")
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="q1")
    store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="a1")
    store.set_session_context_summary(session.id, "GHOST-SUMMARY", "ghost-native-id")

    snapshot = await controller.build_context_snapshot(draft="")
    rows = snapshot.next_send_payload["messages"]

    contents = [row.get("content") or "" for row in rows]
    assert any("q1" in c for c in contents)
    assert any("a1" in c for c in contents)
    assert not any("GHOST-SUMMARY" in c for c in contents)
