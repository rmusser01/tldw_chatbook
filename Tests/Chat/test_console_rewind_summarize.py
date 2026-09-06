"""Controller-level tests for `/rewind` "Summarize up to here" (SP2, Task 3).

Covers ``ConsoleChatController.summarize_up_to`` (gates, span construction,
rolling re-summarize, provider call, storage) and the typed effective-memory
projection used by every preview and dispatch path. Reuses the fake-gateway
harness shape from ``test_console_regenerate_branching.py``.
"""

import asyncio
from dataclasses import replace

import pytest

from tldw_chatbook.Chat import console_context_compaction as context_compaction
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleVariantSet,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_context_compaction import (
    NO_LEGACY_MEMORY,
    DurableMessageSnapshot,
    prefix_digest,
    select_effective_memory,
)
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleMemoryRecord,
    ConsoleMemoryScopeRecord,
    ConsoleMemorySelectionRecord,
    MemoryCoverageKind,
    MemoryOriginKind,
    MemorySelectionKind,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_prepared_request import (
    PreparedConsoleRequest,
    PreparedProviderRequest,
    build_console_request,
    prepare_provider_request,
    resolve_request_capacity,
    thaw_json,
)
from tldw_chatbook.Chat.console_provider_gateway import (
    AuxiliaryCompletionResult,
    ConsoleProviderGateway,
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
from tldw_chatbook.Chat.thinking_blocks import (
    DisplayableThinkingBlock,
    ThinkingEnvelope,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from Tests.console_provider_doubles import provider_resolution
from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
)


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


class VisualSummaryGateway(SummaryGateway):
    """Summary fake whose exact accounting makes each image visibly expensive."""

    IMAGE_TOKEN_COST = 10_000

    def __init__(
        self,
        summary: str = "S",
        *,
        context_window_tokens: int = 100_000,
    ) -> None:
        super().__init__(
            summary=summary,
            context_window_tokens=context_window_tokens,
        )
        self.prepared_requests: list[PreparedProviderRequest] = []

    @classmethod
    def _count_exact_payload(cls, rows, _model):
        total = 0
        for row in rows:
            total += 2
            content = row.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        total += len(str(part.get("text", "")).split())
                    elif part.get("type") == "image_url":
                        total += cls.IMAGE_TOKEN_COST
            else:
                total += len(str(content).split())
        return total

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
        prepared = prepare_provider_request(
            semantic,
            wire_style="single_preamble",
            model=resolution.model or "gpt-test",
            provider=resolution.provider,
            capacity=resolve_request_capacity(
                context_window_tokens=self.context_window_tokens,
                requested_response_tokens=resolution.max_tokens or 512,
            ),
            count_fn=self._count_exact_payload,
            apply_safety_window=apply_safety_window,
        )
        self.prepared_requests.append(prepared)
        return prepared


class ControllerDispatchGateway(SummaryGateway):
    """Prepare with the real gateway and capture its exact provider kwargs."""

    def __init__(self, provider: str, *, summary: str = "RANGE-MEMORY") -> None:
        super().__init__(summary=summary)
        self.provider = provider
        self.execution_key = provider
        self.model = "Qwen3.8-27B" if provider == "llama_cpp" else "gpt-test"
        self.base_url = (
            "http://127.0.0.1:9099"
            if provider == "llama_cpp"
            else "https://api.openai.com/v1"
        )
        self._exact = ConsoleProviderGateway(environ={})
        self.dispatched_prepared: PreparedProviderRequest | None = None
        self.dispatched_kwargs: dict | None = None
        self.prepare_thinking_sidecars: list[tuple] = []

    async def resolve_for_send(self, selection):
        destination = provider_resolution(
            ready=True,
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
        ).resolved_destination
        return ConsoleProviderResolution(
            ready=True,
            provider=self.provider,
            execution_key=self.execution_key,
            model=self.model,
            base_url=self.base_url,
            max_tokens=512,
            streaming=False,
            visible_copy="",
            resolved_destination=destination,
            thinking_stream_disposition=(
                "displayable" if self.provider == "llama_cpp" else "ignored"
            ),
            thinking_round_trip_version=(1 if self.provider == "llama_cpp" else None),
        )

    def prepare_chat_request(self, resolution, messages, **kwargs):
        self.prepare_thinking_sidecars.append(tuple(kwargs.get("thinking_sidecar", ())))
        return self._exact.prepare_chat_request(resolution, messages, **kwargs)

    def capture_dispatch(
        self,
        resolution: ConsoleProviderResolution,
        prepared: PreparedProviderRequest,
    ) -> None:
        self.dispatched_prepared = prepared
        self.dispatched_kwargs = self._exact._chat_api_kwargs_from_prepared(
            resolution, prepared
        )

    async def stream_chat(self, resolution, messages, **_kwargs):
        prepared = (
            messages
            if isinstance(messages, PreparedProviderRequest)
            else self.prepare_chat_request(resolution, messages)
        )
        self.capture_dispatch(resolution, prepared)
        yield "done"


class ControllerDispatchAgentBridge:
    """Drive the real serializer from the controller's agent handoff."""

    def __init__(self, gateway: ControllerDispatchGateway) -> None:
        self.gateway = gateway
        self.received_messages: list[dict] = []
        self.received_thinking_sidecar = ()

    def run_reply(self, **kwargs):
        from tldw_chatbook.Agents.agent_models import RUN_DONE, RunOutcome

        self.received_messages = list(kwargs["agent_messages"])
        self.received_thinking_sidecar = tuple(kwargs["thinking_sidecar"])
        messages = [
            {
                "role": "system",
                "content": kwargs["session_system_prompt"],
            },
            *self.received_messages,
        ]
        prepared = self.gateway.prepare_chat_request(
            kwargs["resolution"],
            messages,
            thinking_sidecar=self.received_thinking_sidecar,
            thinking_policy=kwargs["thinking_policy"],
            thinking_owner_key=(
                "_native_message_id" if self.received_thinking_sidecar else None
            ),
        )
        self.gateway.capture_dispatch(kwargs["resolution"], prepared)
        return "controller-memory-run", RunOutcome(
            status=RUN_DONE,
            steps=[],
            final_text="done",
        )


def _as_chunks(text: str):
    # Emit in two pieces to exercise chunk accumulation (mirrors a real stream).
    if not text:
        return []
    mid = max(1, len(text) // 2)
    return [text[:mid], text[mid:]]


def test_effective_memory_selection_carries_the_validated_scope() -> None:
    snapshots = tuple(
        DurableMessageSnapshot(
            message_id=message_id,
            version=1,
            role=role,
            content=message_id,
        )
        for message_id, role in (
            ("u1", "user"),
            ("a1", "assistant"),
            ("u2", "user"),
            ("a2", "assistant"),
            ("u3", "user"),
            ("a3", "assistant"),
        )
    )
    memory = ConsoleMemoryRecord(
        memory_id="range-memory",
        conversation_id="conversation-1",
        boundary_message_id="a2",
        captured_leaf_message_id="a3",
        lineage_json='["u1","a1","u2","a2","u3","a3"]',
        summary_text="Range facts.",
        provider="openai",
        model="gpt-test",
        prompt_id="console.rewind_summarize",
        prompt_revision=1,
        prompt_digest="p" * 64,
        selected_units_json="[]",
        summarized_prefix_digest=prefix_digest(snapshots[:4]),
        input_tokens=40,
        output_tokens=10,
        before_tokens=100,
        after_tokens=50,
        created_at="2026-08-28T00:00:00Z",
    )
    scope = ConsoleMemoryScopeRecord(
        memory_id=memory.memory_id,
        conversation_id=memory.conversation_id,
        coverage_kind=MemoryCoverageKind.RANGE,
        origin_kind=MemoryOriginKind.MANUAL_REWIND,
        selection_anchor_message_id="u2",
    )
    selection = ConsoleMemorySelectionRecord(
        sequence=1,
        selection_id="range-selection",
        conversation_id=memory.conversation_id,
        activation_message_id="a3",
        selected_memory_id=memory.memory_id,
        event_kind=MemorySelectionKind.SELECT,
        suppresses_legacy=True,
        created_at="2026-08-28T00:00:00Z",
    )

    result = select_effective_memory(
        memory.conversation_id,
        snapshots,
        memories=(memory,),
        scopes=(scope,),
        selection_candidates=(selection,),
        legacy=NO_LEGACY_MEMORY,
    )

    assert result.scope == scope


def _effective_projection_memory(
    coverage: MemoryCoverageKind,
    *,
    conversation_id: str = "conversation-1",
):
    snapshots = tuple(
        DurableMessageSnapshot(
            message_id=message_id,
            version=1,
            role=role,
            content=message_id,
        )
        for message_id, role in (
            ("u1", "user"),
            ("a1", "assistant"),
            ("u2", "user"),
            ("a2", "assistant"),
            ("u3", "user"),
            ("a3", "assistant"),
        )
    )
    boundary = "a2" if coverage is MemoryCoverageKind.RANGE else "a1"
    memory = ConsoleMemoryRecord(
        memory_id=f"{coverage.value}-memory",
        conversation_id=conversation_id,
        boundary_message_id=boundary,
        captured_leaf_message_id="a3",
        lineage_json='["u1","a1","u2","a2","u3","a3"]',
        summary_text=f"{coverage.value} facts.",
        provider="openai",
        model="gpt-test",
        prompt_id="console.rewind_summarize",
        prompt_revision=1,
        prompt_digest="p" * 64,
        selected_units_json="[]",
        summarized_prefix_digest=prefix_digest(
            snapshots[:4] if coverage is MemoryCoverageKind.RANGE else snapshots[:2]
        ),
        input_tokens=40,
        output_tokens=10,
        before_tokens=100,
        after_tokens=50,
        created_at="2026-08-28T00:00:00Z",
    )
    scope = ConsoleMemoryScopeRecord(
        memory_id=memory.memory_id,
        conversation_id=conversation_id,
        coverage_kind=coverage,
        origin_kind=(
            MemoryOriginKind.MANUAL_REWIND
            if coverage is MemoryCoverageKind.RANGE
            else MemoryOriginKind.AUTOMATIC
        ),
        selection_anchor_message_id=(
            "u2" if coverage is MemoryCoverageKind.RANGE else None
        ),
    )
    selection = ConsoleMemorySelectionRecord(
        sequence=1,
        selection_id=f"{coverage.value}-selection",
        conversation_id=conversation_id,
        activation_message_id="a3",
        selected_memory_id=memory.memory_id,
        event_kind=MemorySelectionKind.SELECT,
        suppresses_legacy=(coverage is MemoryCoverageKind.RANGE),
        created_at="2026-08-28T00:00:00Z",
    )
    return select_effective_memory(
        conversation_id,
        snapshots,
        memories=(memory,),
        scopes=(scope,),
        selection_candidates=(selection,),
        legacy=NO_LEGACY_MEMORY,
    )


_PERSISTED_ID = "_tldw_persisted_message_id"
_PERSISTED_CONVERSATION = "_tldw_persisted_conversation_id"


def _projection_row(message_id: str, role: str, **extra):
    return {
        "role": role,
        "content": message_id,
        _PERSISTED_ID: message_id,
        _PERSISTED_CONVERSATION: "conversation-1",
        **extra,
    }


@pytest.mark.parametrize(
    ("coverage", "expected_ids"),
    [
        (MemoryCoverageKind.PREFIX, ["system", "u2", "a2", "u3", "a3"]),
        (MemoryCoverageKind.RANGE, ["system", "u1", "a1", "u3", "a3"]),
    ],
)
def test_project_effective_memory_applies_exact_prefix_or_inclusive_range(
    coverage,
    expected_ids,
) -> None:
    rows = [
        {"role": "system", "content": "system"},
        *(
            _projection_row(message_id, role)
            for message_id, role in (
                ("u1", "user"),
                ("a1", "assistant"),
                ("u2", "user"),
                ("a2", "assistant"),
                ("u3", "user"),
                ("a3", "assistant"),
            )
        ),
    ]

    projected = context_compaction.project_effective_memory(
        rows, _effective_projection_memory(coverage)
    )

    assert [row["content"] for row in projected.rows] == expected_ids
    assert len(projected.memory) == 1
    assert projected.memory[0]["role"] == "system"
    assert coverage.value + " facts." in projected.memory[0]["content"]


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        (
            "missing-start",
            lambda rows: [row for row in rows if row.get(_PERSISTED_ID) != "u2"],
        ),
        (
            "missing-end",
            lambda rows: [row for row in rows if row.get(_PERSISTED_ID) != "a2"],
        ),
        (
            "reversed",
            lambda rows: [rows[0], rows[1], rows[2], rows[4], rows[3], *rows[5:]],
        ),
        (
            "cross-conversation",
            lambda rows: [
                (
                    {**row, _PERSISTED_CONVERSATION: "conversation-2"}
                    if row.get(_PERSISTED_ID) == "a2"
                    else row
                )
                for row in rows
            ],
        ),
        (
            "off-lineage",
            lambda rows: [
                (
                    {**row, _PERSISTED_ID: "sibling-a2"}
                    if row.get(_PERSISTED_ID) == "a2"
                    else row
                )
                for row in rows
            ],
        ),
        (
            "duplicate-anchor",
            lambda rows: [*rows, dict(rows[3])],
        ),
    ],
)
def test_project_effective_memory_invalid_anchor_tables_fail_open_raw(
    case,
    mutate,
) -> None:
    del case
    original = [
        {"role": "system", "content": "system"},
        _projection_row("u1", "user"),
        _projection_row("a1", "assistant"),
        _projection_row("u2", "user"),
        _projection_row("a2", "assistant"),
        _projection_row("u3", "user"),
    ]
    rows = mutate(original)

    projected = context_compaction.project_effective_memory(
        rows, _effective_projection_memory(MemoryCoverageKind.RANGE)
    )

    assert projected.rows == tuple(rows)
    assert projected.memory == ()


def test_project_effective_memory_removes_only_removed_rows_sidecars() -> None:
    retained = _projection_row(
        "u3",
        "user",
        attachment={"url": "retained"},
        custom_wire_field={"keep": True},
    )
    removed = _projection_row(
        "a2",
        "assistant",
        thinking="REMOVED-THINKING",
        continuation={"secret": "REMOVED-CONTINUATION"},
        attachment={"url": "REMOVED-ATTACHMENT"},
        tool_calls=[{"id": "REMOVED-TOOL"}],
    )
    rows = [
        {"role": "system", "content": "system"},
        _projection_row("u1", "user"),
        _projection_row("a1", "assistant"),
        _projection_row("u2", "user"),
        removed,
        retained,
    ]

    projected = context_compaction.project_effective_memory(
        rows, _effective_projection_memory(MemoryCoverageKind.RANGE)
    )

    assert projected.rows == tuple(rows[:3] + [retained])
    assert projected.rows[-1] is retained
    assert all("REMOVED" not in repr(row) for row in projected.rows)


@pytest.mark.asyncio
async def test_untyped_memory_adapter_preserves_newest_first_order(tmp_path) -> None:
    controller, store, session, _gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="BASE-MEMORY")
    )
    rows = _seed_durable_conversation(store, session.id)
    assert (await controller.summarize_from(rows[2].id)).accepted is True
    repository = controller._context_repository
    assert repository is not None
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    base_memory = repository.list_active_memories(conversation_id)[0]
    oldest = replace(
        base_memory,
        memory_id="oldest-memory",
        summary_text="OLDEST-MEMORY",
        created_at="2026-08-28T00:00:00Z",
    )
    newest = replace(
        base_memory,
        memory_id="newest-memory",
        summary_text="NEWEST-MEMORY",
        created_at="2026-08-29T00:00:00Z",
    )

    class UntypedMemoryAdapter:
        def list_active_memories(self, requested_conversation_id):
            assert requested_conversation_id == conversation_id
            return (newest, oldest)

    controller._context_repository = UntypedMemoryAdapter()
    snapshots = controller._durable_context_snapshots(session.id)
    assert snapshots is not None

    effective = controller._select_session_effective_memory(
        session.id, conversation_id, snapshots
    )

    assert effective.memory is not None
    assert effective.memory.summary_text == "NEWEST-MEMORY"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "position", "applies"),
    [
        ("retry", "before", False),
        ("regenerate", "inside", False),
        ("continue", "inside", False),
        ("edit", "inside", False),
        ("continue", "at-end", True),
        ("edit", "after", True),
    ],
)
async def test_controller_operations_obey_range_activation_boundary(
    tmp_path,
    operation,
    position,
    applies,
) -> None:
    gateway = SummaryGateway(summary="RANGE-MEMORY")
    controller, store, session, _gateway, _db = _durable_controller(
        tmp_path, gateway=gateway
    )
    rows = _seed_durable_conversation(store, session.id)
    assert (await controller.summarize_from(rows[2].id)).accepted is True
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF),
    )

    if operation == "retry":
        anchor = store._message_or_raise(rows[1].id)
        anchor.status = "pending"
        anchor.assistant_generation_state = "pending"
        store.mark_message_failed(anchor.id)
        result = await controller.retry_message(anchor.id)
    elif operation == "regenerate":
        result = await controller.regenerate_message(rows[3].id)
    elif operation == "continue":
        anchor = rows[3] if position == "inside" else rows[5]
        result = await controller.continue_from_message(anchor.id)
    else:
        anchor = rows[4]
        if position == "after":
            anchor = store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="q4 after range",
                persist=True,
            )
        result = await controller.edit_and_resend_message(
            anchor.id, f"edited {position}"
        )

    assert result.accepted is True
    prepared = gateway.captured_messages
    assert isinstance(prepared, PreparedProviderRequest)
    wire = "\n".join(
        [prepared.system_message or ""]
        + [str(row.get("content", "")) for row in prepared.messages_payload]
    )
    assert ("RANGE-MEMORY" in wire) is applies
    assert _PERSISTED_ID not in repr(prepared)
    assert _PERSISTED_CONVERSATION not in repr(prepared)
    assert "_native_message_id" not in repr(prepared)
    if applies:
        assert all(text not in wire for text in ("q2 ", "a2 ", "q3 ", "a3 "))
    else:
        assert "<chatbook_conversation_memory>" not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "wire_style"),
    [
        ("llama_cpp", "distinct_roles"),
        ("openai", "single_preamble"),
    ],
)
@pytest.mark.parametrize("dispatch_path", ["direct", "agent"])
async def test_controller_dispatch_projects_repository_memory_without_leaks(
    tmp_path,
    provider,
    wire_style,
    dispatch_path,
) -> None:
    gateway = ControllerDispatchGateway(provider)
    controller, store, session, _gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="RANGE-MEMORY")
    )
    controller.update_provider_selection(
        replace(
            controller._provider_selection(),
            provider=provider,
            explicit_model=gateway.model,
            configured_model=gateway.model,
            base_url=gateway.base_url,
            system_prompt="ORIGINAL-SYSTEM",
        )
    )
    bridge = None
    if dispatch_path == "agent":
        bridge = ControllerDispatchAgentBridge(gateway)
        controller._agent_bridge = bridge

    rows = _seed_durable_conversation(store, session.id)
    for owner, text in ((rows[1], "RETAINED-THINKING"), (rows[3], "REMOVED-THINKING")):
        stored = store._message_or_raise(owner.id)
        stored.thinking = ThinkingEnvelope(
            (
                DisplayableThinkingBlock(
                    block_id=f"block-{text.lower()}",
                    round_ordinal=0,
                    provider=provider,
                    model=gateway.model,
                    protocol="chat_completions",
                    source_format="start_anchored_think",
                    status="complete",
                    text=text,
                ),
            )
        )
    assert (await controller.summarize_from(rows[2].id)).accepted is True
    controller.provider_gateway = gateway
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF),
    )

    result = await controller.submit_draft("ACTIVE-REQUEST", session_id=session.id)

    assert result.accepted is True
    prepared = gateway.dispatched_prepared
    kwargs = gateway.dispatched_kwargs
    assert prepared is not None and kwargs is not None
    assert prepared.wire_style == wire_style
    assert kwargs.get("system_message") == prepared.system_message
    assert kwargs["messages_payload"] == [
        thaw_json(row) for row in prepared.messages_payload
    ]
    wire = "\n".join(
        [str(kwargs.get("system_message", ""))]
        + [str(row.get("content", "")) for row in kwargs["messages_payload"]]
    )
    assert wire.count("ORIGINAL-SYSTEM") == 1
    assert wire.count("RANGE-MEMORY") == 1
    assert "ACTIVE-REQUEST" in wire
    assert all(text not in wire for text in ("q2 ", "a2 ", "q3 ", "a3 "))
    assert "REMOVED-THINKING" not in wire
    assert _PERSISTED_ID not in repr(kwargs)
    assert _PERSISTED_CONVERSATION not in repr(kwargs)
    assert "_native_message_id" not in repr(kwargs)
    assert not any(
        row.get("role") == "user" and "RANGE-MEMORY" in str(row.get("content"))
        for row in kwargs["messages_payload"]
    )
    assert prepared.accounting.memory_tokens > 0
    final_sidecars = gateway.prepare_thinking_sidecars[-1]
    if provider == "llama_cpp":
        assert [sidecar.owner_message_id for sidecar in final_sidecars] == [rows[1].id]
        assert "RETAINED-THINKING" in wire
    else:
        assert final_sidecars == ()
    if bridge is not None:
        expected_sidecar_owners = [rows[1].id] if provider == "llama_cpp" else []
        assert [
            sidecar.owner_message_id for sidecar in bridge.received_thinking_sidecar
        ] == expected_sidecar_owners
        assert (
            sum(
                "RANGE-MEMORY" in str(row.get("content"))
                for row in bridge.received_messages
            )
            == 1
        )


@pytest.mark.asyncio
async def test_range_projection_is_shared_by_preflight_and_next_send_preview(
    tmp_path,
) -> None:
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="RANGE-MEMORY")
    )
    rows = _seed_durable_conversation(store, session.id)
    store.set_session_context_summary(session.id, "LEGACY-MEMORY", rows[1].id)
    assert (await controller.summarize_from(rows[2].id)).accepted is True
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF),
    )
    active = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="q4 active request",
        persist=True,
    )
    resolution = await gateway.resolve_for_send(controller._provider_selection())
    raw = controller._provider_messages_for_session(session.id, annotate_ids=True)
    snapshots = controller._durable_context_snapshots(session.id)
    assert snapshots is not None
    effective = controller._select_session_effective_memory(
        session.id,
        session.persisted_conversation_id,
        snapshots,
    )
    assert effective.kind is context_compaction.EffectiveMemoryKind.GENERATED_RANGE
    assert all(
        _PERSISTED_ID in row and _PERSISTED_CONVERSATION in row
        for row in raw
        if row.get("role") != "system"
    )

    preflight, blocked = await controller._apply_conversation_memory_preflight(
        session_id=session.id,
        resolution=resolution,
        provider_messages=raw,
        assistant_message_id=active.id,
        agent_tools_enabled=False,
    )
    snapshot = await controller.build_context_snapshot(
        "q4 preview", session_id=session.id
    )

    assert blocked is None
    preflight_text = "\n".join(str(row.get("content", "")) for row in preflight)
    preview_text = "\n".join(
        str(row.get("content", "")) for row in snapshot.next_send_payload["messages"]
    )
    for projected_text in (preflight_text, preview_text):
        assert "RANGE-MEMORY" in projected_text
        assert "LEGACY-MEMORY" not in projected_text
        assert "q1 " in projected_text and "a1 " in projected_text
        assert all(text not in projected_text for text in ("q2 ", "a2 ", "q3 ", "a3 "))
    assert "q4 active request" in preflight_text
    assert "q4 preview" in preview_text


@pytest.mark.asyncio
async def test_preview_duplicates_complete_system_and_memory_dispatch_block(
    tmp_path,
) -> None:
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="RANGE-MEMORY")
    )
    controller.update_provider_selection(
        replace(controller._provider_selection(), system_prompt="ORIGINAL-SYSTEM")
    )
    rows = _seed_durable_conversation(store, session.id)
    assert (await controller.summarize_from(rows[2].id)).accepted is True
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF),
    )
    resolution = await gateway.resolve_for_send(controller._provider_selection())
    projected, blocked = await controller._apply_conversation_memory_preflight(
        session_id=session.id,
        resolution=resolution,
        provider_messages=controller._provider_messages_for_session(
            session.id, annotate_ids=True
        ),
        assistant_message_id=rows[-1].id,
        agent_tools_enabled=False,
    )
    prepared = gateway.prepare_chat_request(
        resolution,
        projected,
        apply_safety_window=False,
    )
    snapshot = await controller.build_context_snapshot(
        "active preview", session_id=session.id
    )

    leading_system = []
    for row in snapshot.next_send_payload["messages"]:
        if row.get("role") != "system":
            break
        leading_system.append(row)
    assert len(leading_system) == 2
    assert snapshot.next_send_payload["system"] == leading_system
    assert prepared.system_message == "\n\n".join(
        str(row["content"]).strip() for row in leading_system
    )
    assert prepared.system_message.count("ORIGINAL-SYSTEM") == 1
    assert prepared.system_message.count("RANGE-MEMORY") == 1


@pytest.mark.asyncio
async def test_effective_legacy_memory_makes_automatic_and_compact_now_zero_call(
    tmp_path,
) -> None:
    controller, store, session, gateway, _db = _durable_controller(
        tmp_path, gateway=SummaryGateway(summary="MUST-NOT-BE-CALLED")
    )
    rows = _seed_durable_conversation(store, session.id)
    store.set_session_context_summary(session.id, "LEGACY-MEMORY", rows[1].id)
    store.set_session_context_policy_overrides(
        session.id,
        ConsoleContextPolicyOverrides(
            compaction_mode=ContextCompactionMode.AUTOMATIC,
            custom_budget_tokens=1,
        ),
    )
    active = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="active request",
        persist=True,
    )
    resolution = await gateway.resolve_for_send(controller._provider_selection())
    raw = controller._provider_messages_for_session(session.id, annotate_ids=True)

    projected, blocked = await controller._apply_conversation_memory_preflight(
        session_id=session.id,
        resolution=resolution,
        provider_messages=raw,
        assistant_message_id=active.id,
        agent_tools_enabled=False,
    )
    compacted, _copy = await controller.compact_context_now(session.id)

    assert blocked is None
    assert compacted is False
    assert gateway.calls == 0
    text = "\n".join(str(row.get("content", "")) for row in projected)
    assert "LEGACY-MEMORY" in text
    assert "q1 " not in text and "a1 " not in text
    assert "q2 " in text and "active request" in text


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


def _prepared_contains_visual(
    prepared: PreparedProviderRequest,
    expected_data_url: str,
) -> bool:
    for row in thaw_json(prepared.messages_payload):
        content = row.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part == {
                "type": "image_url",
                "image_url": {"url": expected_data_url},
            }:
                return True
    return False


def _seed_manual_visual_case(
    store: ConsoleChatStore,
    session_id: str,
    *,
    from_here: bool,
    image_only: bool,
):
    image_bytes = b"UNIQUE_IMAGE_FACT_7429"
    attachments = (
        (
            MessageAttachment(
                data=b"FIRST_IMAGE",
                mime_type="image/png",
                display_name="first.png",
                position=0,
            ),
            MessageAttachment(
                data=image_bytes,
                mime_type="image/png",
                display_name="fact.png",
                position=1,
            ),
        )
        if not image_only
        else (
            MessageAttachment(
                data=image_bytes,
                mime_type="image/png",
                display_name="fact.png",
                position=0,
            ),
        )
    )
    if from_here:
        for role, content in (
            (ConsoleMessageRole.USER, "earlier question " + "context " * 40),
            (ConsoleMessageRole.ASSISTANT, "earlier answer " + "context " * 40),
        ):
            store.append_message(
                session_id,
                role=role,
                content=content,
                persist=True,
            )
    visual_user = store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="" if image_only else "mixed visual question " + "detail " * 40,
        attachments=attachments,
        persist=True,
    )
    store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="visual answer " + "detail " * 80,
        persist=True,
    )
    if from_here:
        return visual_user
    anchor = store.append_message(
        session_id,
        role=ConsoleMessageRole.USER,
        content="retained question " + "context " * 40,
        persist=True,
    )
    store.append_message(
        session_id,
        role=ConsoleMessageRole.ASSISTANT,
        content="retained answer " + "context " * 40,
        persist=True,
    )
    return anchor


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
    monkeypatch,
):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.is_vision_capable",
        lambda _provider, _model: True,
    )
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
        store.append_message(session.id, role=role, content=content, persist=True)

    result = await controller.summarize_from(u1.id)

    assert result.accepted is True
    assert gateway.calls == 1
    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    assert (
        len(controller._context_repository.list_active_memories(conversation_id)) == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("from_here", [False, True], ids=["prefix", "range"])
@pytest.mark.parametrize("image_only", [True, False], ids=["image-only", "mixed"])
async def test_manual_summary_sends_exact_visual_through_auxiliary_and_accounting(
    tmp_path,
    monkeypatch,
    *,
    from_here: bool,
    image_only: bool,
):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.is_vision_capable",
        lambda _provider, _model: True,
    )
    gateway = VisualSummaryGateway()
    controller, store, session, _gateway, _db = _durable_controller(
        tmp_path,
        gateway=gateway,
    )
    anchor = _seed_manual_visual_case(
        store,
        session.id,
        from_here=from_here,
        image_only=image_only,
    )

    result = (
        await controller.summarize_from(anchor.id)
        if from_here
        else await controller.summarize_up_to(anchor.id)
    )

    expected_data_url = "data:image/png;base64,VU5JUVVFX0lNQUdFX0ZBQ1RfNzQyOQ=="
    assert result.accepted is True
    assert gateway.calls == 1
    assert gateway.captured_auxiliary is not None
    auxiliary_content = thaw_json(gateway.captured_auxiliary.messages[1]["content"])
    assert {
        "type": "image_url",
        "image_url": {"url": expected_data_url},
    } in auxiliary_content

    prepared_with_visual = [
        prepared
        for prepared in gateway.prepared_requests
        if _prepared_contains_visual(prepared, expected_data_url)
    ]
    canonical_before = next(
        prepared for prepared in prepared_with_visual if prepared.semantic.compactable
    )
    auxiliary_projection = next(
        prepared
        for prepared in prepared_with_visual
        if not prepared.semantic.compactable
    )
    canonical_after = next(
        prepared
        for prepared in gateway.prepared_requests
        if prepared.semantic.memory
        and not _prepared_contains_visual(prepared, expected_data_url)
    )
    assert (
        canonical_before.accounting.compactable_tokens
        >= VisualSummaryGateway.IMAGE_TOKEN_COST
    )
    assert (
        auxiliary_projection.accounting.total_input_tokens
        >= VisualSummaryGateway.IMAGE_TOKEN_COST
    )
    assert canonical_after.accounting.total_input_tokens < (
        VisualSummaryGateway.IMAGE_TOKEN_COST
    )

    conversation_id = session.persisted_conversation_id
    assert conversation_id is not None
    memory = controller._context_repository.list_active_memories(conversation_id)[0]
    attempts = controller._context_repository.list_auxiliary_attempts(conversation_id)
    durable_snapshot_repr = repr(controller._durable_context_snapshots(session.id))
    assert "UNIQUE_IMAGE_FACT_7429" not in memory.selected_units_json
    assert "VU5JUVVFX0lNQUdFX0ZBQ1RfNzQyOQ" not in memory.selected_units_json
    assert len(attempts) == 1
    assert "UNIQUE_IMAGE_FACT_7429" not in repr(attempts)
    assert "VU5JUVVFX0lNQUdFX0ZBQ1RfNzQyOQ" not in repr(attempts)
    assert "UNIQUE_IMAGE_FACT_7429" not in durable_snapshot_repr
    assert "VU5JUVVFX0lNQUdFX0ZBQ1RfNzQyOQ" not in durable_snapshot_repr
    assert "UNIQUE_IMAGE_FACT_7429" not in repr(gateway.captured_auxiliary)
    assert "VU5JUVVFX0lNQUdFX0ZBQ1RfNzQyOQ" not in repr(gateway.captured_auxiliary)


@pytest.mark.asyncio
async def test_manual_summary_refuses_visual_when_active_model_cannot_represent_it(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.is_vision_capable",
        lambda _provider, _model: False,
    )
    gateway = VisualSummaryGateway()
    controller, store, session, _gateway, _db = _durable_controller(
        tmp_path,
        gateway=gateway,
    )
    anchor = _seed_manual_visual_case(
        store,
        session.id,
        from_here=True,
        image_only=True,
    )

    result = await controller.summarize_from(anchor.id)

    assert result.accepted is False
    assert gateway.calls == 0
    assert "UNIQUE_IMAGE_FACT_7429" not in result.visible_copy
    assert "vision" in result.visible_copy.lower()


@pytest.mark.asyncio
async def test_manual_summary_refuses_nonvisual_attachment_before_dispatch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.is_vision_capable",
        lambda _provider, _model: True,
    )
    gateway = VisualSummaryGateway()
    controller, store, session, _gateway, _db = _durable_controller(
        tmp_path,
        gateway=gateway,
    )
    anchor = store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="summarize the attached document",
        attachments=(
            MessageAttachment(
                data=b"UNREPRESENTABLE_DOCUMENT_FACT_9241",
                mime_type="application/pdf",
                display_name="private.pdf",
                position=0,
            ),
        ),
        persist=True,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="document answer " + "detail " * 80,
        persist=True,
    )

    result = await controller.summarize_from(anchor.id)

    assert result.accepted is False
    assert gateway.calls == 0
    assert "UNREPRESENTABLE_DOCUMENT_FACT_9241" not in result.visible_copy
    assert "cannot safely" in result.visible_copy.lower()


@pytest.mark.asyncio
async def test_manual_summary_refuses_visual_when_exact_auxiliary_capacity_is_exceeded(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.is_vision_capable",
        lambda _provider, _model: True,
    )
    gateway = VisualSummaryGateway(context_window_tokens=10_000)
    controller, store, session, _gateway, _db = _durable_controller(
        tmp_path,
        gateway=gateway,
    )
    anchor = _seed_manual_visual_case(
        store,
        session.id,
        from_here=True,
        image_only=True,
    )

    result = await controller.summarize_from(anchor.id)

    assert result.accepted is False
    assert gateway.calls == 0
    assert "UNIQUE_IMAGE_FACT_7429" not in result.visible_copy
    assert "one call" in result.visible_copy.lower()


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
        store._native_parent_by_message[rows[3].id] = rows[0].id
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
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.is_vision_capable",
        lambda _provider, _model: True,
    )
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
    assert (
        len(controller._context_repository.list_active_memories(conversation_id)) == 1
    )
    assert (
        len(
            controller._context_repository.list_active_memory_selections(
                conversation_id
            )
        )
        == 1
    )

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
    assert all(f'"content":"{text} ' in payload for text in ("q1", "a1", "q2", "a2"))
    assert store.session_context_summary(session.id) == ("S1", rows[1].id)


# --------------------------------------------------------------------------
# choke-point compaction + THE LEAK RULE
# --------------------------------------------------------------------------


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


def test_compacted_summary_precedes_run_local_startup_rider(
    tmp_path, monkeypatch, request
):
    from tldw_chatbook.Agents.agent_models import AgentConfig
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.project_instruction_resolver import (
        InstructionSource,
        StartupInstructionCandidate,
    )
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Tools import workspace_file_roots
    from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

    workspace_db = WorkspaceDB(tmp_path / "workspaces.db")
    request.addfinalizer(workspace_db.close)
    registry = LocalWorkspaceRegistryService(workspace_db)
    monkeypatch.setattr(workspace_file_roots, "_registry_factory", lambda: registry)
    runs_db = AgentRunsDB(tmp_path / "runs.db", client_id="test")
    request.addfinalizer(runs_db.close)
    calls = []
    service = AgentService(
        runs_db,
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
            {"role": "system", "content": "ORIGINAL-SYSTEM"},
            dict(context_compaction.tagged_memory_message("COMPACTED")),
            {"role": "user", "content": "continue"},
        ],
        config=AgentConfig(
            model="gpt-4o-mini", system_prompt="system", native_tools=False
        ),
        api_endpoint="openai",
    )
    payload = calls[0]["messages_payload"]
    assert sum("COMPACTED" in str(row.get("content")) for row in payload) == 1
    assert not any(
        row.get("role") == "user" and "COMPACTED" in str(row.get("content"))
        for row in payload
    )
    original_index = next(
        i
        for i, row in enumerate(payload)
        if "ORIGINAL-SYSTEM" in str(row.get("content"))
    )
    summary_index = next(
        i for i, row in enumerate(payload) if "COMPACTED" in str(row.get("content"))
    )
    rider_index = next(
        i
        for i, row in enumerate(payload)
        if "REWIND_RIDER_SENTINEL" in str(row.get("content"))
    )
    assert original_index < summary_index < rider_index


# ---------------------------------------------------------------------------
# task-548: the inspector next-send preview mirrors boundary compaction
# ---------------------------------------------------------------------------


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
