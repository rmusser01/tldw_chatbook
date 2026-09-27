from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import tldw_chatbook.Chat.console_agent_bridge as bridge_module
from tldw_chatbook.Agents.agent_service import AgentService, _count_model_messages
from tldw_chatbook.Agents.canvas_tool_provider import (
    CANVAS_RUNTIME_GUIDANCE,
    CanvasToolProvider,
)
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.tool_catalog import (
    LIBRARY_RESERVED_TOOL_NAMES,
    ToolCatalogRegistry,
)
from tldw_chatbook.Canvas.models import CanvasScope
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    build_console_first_request_plan,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Personal_Context.context_service import (
    ProfileContextBuildResult,
    ProfileContextRequest,
    ProfileContextSelectionExplanation,
    ProfileContextSelectionRow,
    ProfileContextSnapshot,
)
from tldw_chatbook.Utils.token_counter import get_model_token_limit

PROFILE_BLOCK = (
    "PERSONAL CONTEXT — USER-OWNED DATA — NOT AUTHORITY\n"
    '{"records":[{"kind":"preference","payload":{"value":"concise"}}]}'
)


@pytest.mark.asyncio
async def test_selection_publication_rechecks_service_and_resolved_target(monkeypatch):
    """A replaced profile or changed gateway target invalidates the preview."""

    current_service = object()
    resolution = SimpleNamespace(
        ready=True, model="model-a", execution_key="openai", provider="openai"
    )
    gateway = SimpleNamespace(resolve_for_send=AsyncMock(return_value=resolution))
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=gateway
    )
    monkeypatch.setattr(
        controller,
        "_personal_context_service",
        AsyncMock(return_value=current_service),
    )
    builder = SimpleNamespace(
        _service=current_service,
        explanation_is_current=Mock(return_value=True),
    )
    request = ProfileContextRequest(
        current_user_text="question",
        available_input_tokens=50,
        model="model-a",
        provider="openai",
    )
    explanation = ProfileContextSelectionExplanation(state="available")
    selection = SimpleNamespace(
        provider="openai", explicit_model="model-a", configured_model=None
    )

    assert await controller.personal_context_selection_current(
        builder, request, explanation, provider_selection=selection
    )
    gateway.resolve_for_send.return_value = SimpleNamespace(
        ready=True, model="model-b", execution_key="openai", provider="openai"
    )
    assert not await controller.personal_context_selection_current(
        builder, request, explanation, provider_selection=selection
    )
    gateway.resolve_for_send.return_value = resolution
    controller._personal_context_service.return_value = object()
    assert not await controller.personal_context_selection_current(
        builder, request, explanation, provider_selection=selection
    )


@pytest.mark.asyncio
async def test_selection_publication_rechecks_service_after_worker(monkeypatch):
    current_service = object()
    replacement_service = object()
    gateway = SimpleNamespace(
        resolve_for_send=AsyncMock(
            return_value=SimpleNamespace(
                ready=True, model="model-a", execution_key="openai"
            )
        )
    )
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=gateway
    )
    service_lookup = AsyncMock(return_value=current_service)
    monkeypatch.setattr(controller, "_personal_context_service", service_lookup)
    started = threading.Event()
    release = threading.Event()

    def validate(*_args):
        started.set()
        assert release.wait(5)
        return True

    builder = SimpleNamespace(
        _service=current_service,
        explanation_is_current=validate,
    )
    request = ProfileContextRequest(
        current_user_text="question",
        available_input_tokens=50,
        model="model-a",
        provider="openai",
    )
    explanation = ProfileContextSelectionExplanation(state="available")
    selection = SimpleNamespace(
        provider="openai", explicit_model="model-a", configured_model=None
    )

    pending = asyncio.create_task(
        controller.personal_context_selection_current(
            builder, request, explanation, provider_selection=selection
        )
    )
    assert await asyncio.to_thread(started.wait, 5)
    service_lookup.return_value = replacement_service
    release.set()

    assert not await pending


class _ProfileContextBuilder:
    def __init__(self) -> None:
        self.requests = []
        self.snapshot = None

    def build_snapshot(self, request):
        self.requests.append(request)
        self.snapshot = ProfileContextSnapshot(
            generation=1,
            record_set_revision="record-v1",
            scope_id="scope-workspace",
            authority_revision="authority-v1",
            serialized_block=PROFILE_BLOCK,
            source_version_ids=("version-1",),
            estimated_tokens=20,
        )
        return self.snapshot


class _ExplainedProfileContextBuilder(_ProfileContextBuilder):
    def build_explained_snapshot(self, request):
        snapshot = self.build_snapshot(request)
        return ProfileContextBuildResult(
            snapshot,
            ProfileContextSelectionExplanation(
                state="available",
                rows=(
                    ProfileContextSelectionRow(
                        record_id="ELIGIBLE_DIAGNOSTIC_ID_ONLY",
                        disposition="selected",
                        priority_group=3,
                    ),
                ),
                request_id=id(request),
            ),
        )


def _real_preview_bridge() -> ConsoleAgentBridge:
    bridge = object.__new__(ConsoleAgentBridge)
    bridge._registry = ToolCatalogRegistry()
    bridge._allowed_tools = ()
    bridge._skills_service = None
    bridge._native_tools_enabled = lambda: True
    return bridge


def _plan(
    builder,
    *,
    block_override=None,
    workspace_id="workspace-42",
    turn_bundle_block="",
    library_provider=None,
    library_authority=None,
    canvas_provider=None,
    canvas_authority=None,
    **overrides,
):
    kwargs = dict(
        shared_registry=ToolCatalogRegistry(),
        shared_allowed_tools=(),
        context={},
        skills_present=False,
        mcp_provider=None,
        builtin_gate=None,
        local_provider=None,
        library_provider=library_provider,
        library_authority=library_authority,
        canvas_provider=canvas_provider,
        canvas_authority=canvas_authority,
        workspace_id=workspace_id,
        ephemeral=False,
        diff_sink=None,
        scratch_root=None,
        scratch_lease=None,
        resolution=SimpleNamespace(
            model="gpt-4o-mini", execution_key="openai", max_tokens=2048
        ),
        fallback_model="gpt-4o-mini",
        session_system_prompt="BASE",
        native_tools=True,
        turn_skill_bindings=(),
        turn_bundle_block=turn_bundle_block,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        agent_messages=[{"role": "user", "content": "question"}],
        profile_context_service=builder,
    )
    if block_override is not None:
        kwargs["personal_context_snapshot"] = block_override
    kwargs.update(overrides)
    return build_console_first_request_plan(**kwargs)


def test_first_request_plan_builds_one_snapshot_and_pins_exact_block() -> None:
    builder = _ProfileContextBuilder()

    plan = _plan(builder)

    assert len(builder.requests) == 1
    assert builder.requests[0].active_workspace_id == "workspace-42"
    assert plan.profile_context_snapshot is builder.snapshot
    assert plan.profile_context_snapshot.serialized_block == PROFILE_BLOCK
    assert plan.config.personal_context_block == PROFILE_BLOCK


def test_preview_bridge_returns_plain_snapshot_and_captures_diagnostic_sidecar() -> (
    None
):
    builder = _ExplainedProfileContextBuilder()
    captured = []
    bridge = _real_preview_bridge()

    snapshot = bridge.build_personal_context_preview_snapshot(
        workspace_id="workspace-42",
        ephemeral=False,
        resolution=SimpleNamespace(
            ready=True,
            provider="openai",
            execution_key="openai",
            model="gpt-4o-mini",
            max_tokens=2_048,
        ),
        fallback_model="gpt-4o-mini",
        session_system_prompt="BASE",
        agent_messages=[{"role": "user", "content": "question"}],
        profile_context_service=builder,
        selection_sink=lambda source, request, explanation: captured.append(
            (source, request, explanation)
        ),
    )

    assert type(snapshot) is ProfileContextSnapshot
    assert snapshot.serialized_block == PROFILE_BLOCK
    assert len(builder.requests) == len(captured) == 1
    assert captured[0][0] is builder
    assert captured[0][1] is builder.requests[0]
    assert captured[0][1].active_workspace_id == "workspace-42"
    assert captured[0][1].available_input_tokens > 0
    assert captured[0][2].rows[0].record_id == "ELIGIBLE_DIAGNOSTIC_ID_ONLY"
    assert "ELIGIBLE_DIAGNOSTIC_ID_ONLY" not in repr(snapshot)
    assert "ELIGIBLE_DIAGNOSTIC_ID_ONLY" not in str(asdict(snapshot))


def test_console_global_workspace_requests_only_global_profile_context() -> None:
    builder = _ProfileContextBuilder()

    _plan(builder, workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID)

    assert builder.requests[0].active_workspace_id is None


def test_first_request_profile_budget_reserves_disclosed_tool_protocol() -> None:
    builder = _ProfileContextBuilder()

    plan = _plan(builder)
    naive_required = _count_model_messages(
        [
            {"role": "system", "content": plan.config.system_prompt},
            {"role": "user", "content": "question"},
        ],
        "gpt-4o-mini",
        "openai",
    )
    naive_available = (
        get_model_token_limit("gpt-4o-mini", "openai") - 2_048 - naive_required
    )

    assert builder.requests[0].available_input_tokens < naive_available


def test_first_request_profile_budget_reserves_canvas_runtime_guidance(
    monkeypatch,
) -> None:
    class Coordinator:
        def is_scope_current(self, _scope):
            return True

        def list_canvases(self, _scope):
            return ()

        def read_canvas(self, _scope, _canvas_id):
            raise AssertionError("not invoked")

        def create_canvas(self, _scope, **_kwargs):
            raise AssertionError("not invoked")

        def update_canvas(self, _scope, **_kwargs):
            raise AssertionError("not invoked")

    provider = CanvasToolProvider(
        Coordinator(),
        scope=CanvasScope("session", "conversation", (), None, None, "run"),
        enabled_reader=lambda: True,
    )
    authority = provider.issue_registration_authority()
    captured_system_prompts: list[str] = []
    original_count = bridge_module._count_model_messages

    def capture(messages, model, provider_name, **kwargs):
        captured_system_prompts.append(str(messages[0].get("content", "")))
        return original_count(messages, model, provider_name, **kwargs)

    monkeypatch.setattr(bridge_module, "_count_model_messages", capture)

    _plan(
        _ProfileContextBuilder(),
        canvas_provider=provider,
        canvas_authority=authority,
    )

    assert any(CANVAS_RUNTIME_GUIDANCE in prompt for prompt in captured_system_prompts)


def test_first_request_profile_budget_reserves_the_injected_skill_bundle() -> None:
    without_bundle = _ProfileContextBuilder()
    with_bundle = _ProfileContextBuilder()

    _plan(without_bundle)
    _plan(with_bundle, turn_bundle_block="skill data " * 2_000)

    assert (
        with_bundle.requests[0].available_input_tokens
        < without_bundle.requests[0].available_input_tokens
    )


def test_preview_and_live_request_assembly_use_the_same_pinned_block(tmp_path) -> None:
    builder = _ProfileContextBuilder()
    plan = _plan(builder)
    service = AgentService(
        tmp_path / "unused.db", plan.registry, chat_call=lambda **_: {}
    )

    preview_request = service._build_model_request(
        plan.config,
        plan.api_endpoint,
        list(plan.schemas.runtime_schemas),
        list(plan.messages),
        plan.schemas.active_schemas,
    )

    assert len(builder.requests) == 1
    assert preview_request.messages[0]["content"].endswith(PROFILE_BLOCK)
    assert preview_request.messages[0]["content"].count(PROFILE_BLOCK) == 1


def test_empty_profile_keeps_existing_system_content_byte_identical() -> None:
    builder = _ProfileContextBuilder()
    builder.build_snapshot = lambda _request: ProfileContextSnapshot.empty()

    plan = _plan(builder)

    assert plan.config.personal_context_block == ""
    assert plan.config.system_prompt.startswith("BASE")


@pytest.mark.asyncio
async def test_non_agent_preview_does_not_build_or_display_profile(monkeypatch) -> None:
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SimpleNamespace(),
        agent_runtime_enabled=False,
    )
    calls = []

    async def build_profile(*_args, **_kwargs):
        calls.append(True)
        return ProfileContextSnapshot(
            generation=1,
            record_set_revision="v1",
            scope_id=None,
            authority_revision="a1",
            serialized_block=PROFILE_BLOCK,
            source_version_ids=("record-v1",),
            estimated_tokens=20,
        )

    monkeypatch.setattr(controller, "_build_personal_context_snapshot", build_profile)

    diagnostics = []
    snapshot = await controller.build_context_snapshot(
        draft="question",
        session_id=session.id,
        profile_selection_sink=lambda *values: diagnostics.append(values),
    )

    assert calls == []
    assert diagnostics == []
    assert PROFILE_BLOCK not in str(snapshot.next_send_payload)
    assert snapshot.personal_context_snapshot == ProfileContextSnapshot.empty()


@pytest.mark.asyncio
async def test_agent_next_send_uses_one_pinned_snapshot_without_double_append(
    monkeypatch,
) -> None:
    builder = _ProfileContextBuilder()

    class _PreviewBridge:
        def __init__(self) -> None:
            self.calls = []

        @staticmethod
        def native_tool_schemas():
            return []

        def build_personal_context_preview_snapshot(self, **kwargs):
            self.calls.append(kwargs)
            plan = _plan(
                kwargs["profile_context_service"],
                workspace_id=kwargs["workspace_id"],
                turn_bundle_block=kwargs["turn_bundle_block"],
            )
            return plan.profile_context_snapshot

    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    bridge = _PreviewBridge()
    resolution = SimpleNamespace(
        ready=True,
        provider="openai",
        execution_key="openai",
        model="gpt-4o-mini",
        max_tokens=2_048,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SimpleNamespace(
            resolve_for_send=AsyncMock(return_value=resolution)
        ),
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )

    raw_service = object()
    service_reads = []

    async def personal_context_service():
        service_reads.append(raw_service)
        return raw_service

    async def personal_context_builder(service=None):
        assert service is raw_service
        return builder

    async def compose_providers(**_kwargs):
        return None, None, None, None

    monkeypatch.setattr(
        controller, "_personal_context_builder", personal_context_builder
    )
    monkeypatch.setattr(
        controller, "_personal_context_service", personal_context_service
    )
    monkeypatch.setattr(
        controller, "_compose_agent_request_providers", compose_providers
    )

    snapshot = await controller.build_context_snapshot(
        draft="question", session_id=session.id
    )

    assert len(bridge.calls) == 1
    assert service_reads == [raw_service]
    assert len(builder.requests) == 1
    raw_available = (
        get_model_token_limit("gpt-4o-mini", "openai")
        - 2_048
        - _count_model_messages(
            [{"role": "user", "content": "question"}],
            "gpt-4o-mini",
            "openai",
        )
    )
    assert builder.requests[0].available_input_tokens < raw_available
    assert snapshot.personal_context_snapshot is builder.snapshot
    messages = snapshot.next_send_payload["messages"]
    assert (
        sum(str(row.get("content", "")).count(PROFILE_BLOCK) for row in messages) == 1
    )


@pytest.mark.asyncio
async def test_agent_next_send_collects_diagnostics_outside_snapshot_and_payload(
    monkeypatch,
) -> None:
    builder = _ExplainedProfileContextBuilder()
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    real_bridge = _real_preview_bridge()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SimpleNamespace(
            resolve_for_send=AsyncMock(
                return_value=SimpleNamespace(
                    ready=True,
                    provider="openai",
                    execution_key="openai",
                    model="gpt-4o-mini",
                    max_tokens=2_048,
                )
            )
        ),
        agent_bridge=SimpleNamespace(
            native_tool_schemas=real_bridge.native_tool_schemas,
            build_personal_context_preview_snapshot=(
                real_bridge.build_personal_context_preview_snapshot
            ),
        ),
        agent_runtime_enabled=True,
    )

    async def owner():
        return object()

    async def profile_builder(_service):
        return builder

    async def providers(**_kwargs):
        return None, None, None, None

    monkeypatch.setattr(controller, "_personal_context_service", owner)
    monkeypatch.setattr(controller, "_personal_context_builder", profile_builder)
    monkeypatch.setattr(controller, "_compose_agent_request_providers", providers)
    captured = []

    snapshot = await controller.build_context_snapshot(
        draft="question",
        session_id=session.id,
        profile_selection_sink=lambda source, request, explanation: captured.append(
            (source, request, explanation)
        ),
    )

    assert len(builder.requests) == len(captured) == 1
    assert captured[0][0] is builder
    assert captured[0][1] is builder.requests[0]
    assert captured[0][1].available_input_tokens > 0
    assert snapshot.personal_context_snapshot is builder.snapshot
    assert "ELIGIBLE_DIAGNOSTIC_ID_ONLY" not in str(asdict(snapshot))
    assert "ELIGIBLE_DIAGNOSTIC_ID_ONLY" not in str(snapshot.next_send_payload)
    assert "ELIGIBLE_DIAGNOSTIC_ID_ONLY" not in str(snapshot.current_messages)


@pytest.mark.asyncio
async def test_agent_next_send_reserves_the_live_library_schemas(
    monkeypatch, tmp_path,
) -> None:
    builder = _ProfileContextBuilder()
    library_provider = LibraryToolProvider(SimpleNamespace())
    library_authority = library_provider.issue_builtin_authority(
        reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
    )

    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    db = AgentRunsDB(tmp_path / "preview-runs.db")
    real_bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=object(),
        native_tools_enabled=lambda: True,
    )

    def preview(**kwargs):
        try:
            return real_bridge.build_personal_context_preview_snapshot(**kwargs)
        finally:
            db.close()

    bridge = SimpleNamespace(
        native_tool_schemas=real_bridge.native_tool_schemas,
        build_personal_context_preview_snapshot=Mock(wraps=preview),
    )
    resolution = SimpleNamespace(
        ready=True,
        provider="openai",
        execution_key="openai",
        model="gpt-4o-mini",
        max_tokens=2_048,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SimpleNamespace(
            resolve_for_send=AsyncMock(return_value=resolution)
        ),
        agent_bridge=bridge,
        agent_runtime_enabled=True,
        library_provider_factory=lambda _context: library_provider,
    )

    async def personal_context_builder(_service=None):
        return builder

    async def capture_authority(*_args, **_kwargs):
        return object()

    async def compose_providers(**_kwargs):
        return None, None, None, None

    monkeypatch.setattr(
        controller, "_personal_context_builder", personal_context_builder
    )
    monkeypatch.setattr(
        controller, "_capture_turn_library_authority", capture_authority
    )
    monkeypatch.setattr(
        controller,
        "_finalize_turn_execution_context",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        controller,
        "_library_provider_for_context",
        lambda _context: (library_provider, library_authority),
    )
    monkeypatch.setattr(
        controller, "_compose_agent_request_providers", compose_providers
    )

    try:
        snapshot = await controller.build_context_snapshot(
            draft="question", session_id=session.id
        )
        without_library = _ProfileContextBuilder()
        _plan(without_library)

        assert bridge.build_personal_context_preview_snapshot.call_count == 1
        preview_call = bridge.build_personal_context_preview_snapshot.call_args.kwargs
        assert preview_call["library_provider"] is library_provider
        assert preview_call["library_authority"] is library_authority
        assert (
            builder.requests[0].available_input_tokens
            < without_library.requests[0].available_input_tokens
        )
        assert snapshot.personal_context_snapshot is builder.snapshot
    finally:
        controller.begin_shutdown()
        real_bridge.close_all_progress()
        db.close()


@pytest.mark.asyncio
async def test_agent_next_send_uses_selected_project_root_for_local_schemas(
    monkeypatch,
    tmp_path,
) -> None:
    store = ConsoleChatStore()
    state = ProjectInstructionControlState(
        True,
        "binding-1",
        "f" * 64,
        None,
    )
    session = store.create_session(
        ephemeral=True,
        project_instruction_state=state,
    )
    from tldw_chatbook.Chat.console_chat_controller import (
        ProjectInstructionBindingSelection,
    )
    from Tests.console_provider_doubles import provider_resolution

    selected = ProjectInstructionBindingSelection(
        binding=SimpleNamespace(binding_id="binding-1"),
        root=tmp_path,
        locator_fingerprint="f" * 64,
        allow_write=True,
        root_identity=(),
    )
    captured = []
    bridge = SimpleNamespace(
        native_tool_schemas=list,
        build_personal_context_preview_snapshot=lambda **_kwargs: (
            ProfileContextSnapshot.empty()
        ),
    )
    resolution = provider_resolution(
        ready=True,
        provider="openai",
        execution_key="openai",
        model="gpt-4o-mini",
        max_tokens=2_048,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=SimpleNamespace(
            resolve_for_send=AsyncMock(return_value=resolution)
        ),
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    controller.app = SimpleNamespace(workspace_registry_service=object())

    async def personal_context_builder(_service=None):
        return _ProfileContextBuilder()

    async def compose_providers(**kwargs):
        captured.append(kwargs)
        return None, None, None, None

    monkeypatch.setattr(
        controller, "_personal_context_builder", personal_context_builder
    )
    monkeypatch.setattr(
        controller, "_compose_agent_request_providers", compose_providers
    )
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.resolve_project_instruction_binding",
        lambda _session, _registry: selected,
    )

    await controller.build_context_snapshot(draft="question", session_id=session.id)

    assert captured[0]["project_selection"] is selected


def _local_reasoning_rows():
    from tldw_chatbook.Chat.console_provider_gateway import ProviderThinkingDelta
    from tldw_chatbook.Chat.console_thinking_capture import ThinkingCapture

    capture = ThinkingCapture(assistant_owner_id="profile-budget-thought")
    capture.observe(
        ProviderThinkingDelta(
            text="Detailed tool reasoning " * 300,
            provider="local_vllm",
            model="reasoner",
            protocol="chat_completions",
            source_format="reasoning_content",
        )
    )
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "I checked.",
            "_tldw_call_thinking": capture.settle("complete").envelope,
        },
    ]


def _prepared_local_reasoning_rows(resolution, messages):
    import asyncio

    from tldw_chatbook.Chat.console_prepared_request import thaw_json
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.console_thinking_capture import consume_call_thinking

    owner_key = "_tldw_call_thinking_owner"
    rows, sidecars = consume_call_thinking(messages, owner_key=owner_key)
    gateway = ConsoleProviderGateway()
    try:
        prepared = gateway.prepare_chat_request(
            resolution,
            rows,
            thinking_sidecar=sidecars,
            thinking_owner_key=owner_key,
            apply_safety_window=False,
        )
        return [thaw_json(row) for row in prepared.messages]
    finally:
        asyncio.run(gateway.aclose())


@pytest.mark.parametrize("mode", ["off", "all", None])
def test_profile_capacity_counts_the_dispatched_reasoning_projection(
    tmp_path,
    monkeypatch,
    mode,
):
    from tldw_chatbook.Chat.console_history_budget import count_console_messages_tokens
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy

    limit = 20_000
    monkeypatch.setattr(bridge_module, "get_model_token_limit", lambda *_: limit)
    policy = ReasoningReplayPolicy(mode, "test") if mode else None
    resolution = ConsoleProviderResolution(
        provider="local_vllm",
        execution_key="local_vllm",
        base_url="",
        model="reasoner",
        ready=True,
        max_tokens=1024,
        local_structured_thinking=True,
        reasoning_replay=policy,
    )
    builder = _ProfileContextBuilder()

    def empty_snapshot(request):
        builder.requests.append(request)
        return ProfileContextSnapshot.empty()

    builder.build_snapshot = empty_snapshot
    plan = _plan(
        builder,
        resolution=resolution,
        native_tools=False,
        agent_messages=_local_reasoning_rows(),
    )
    service = AgentService(
        tmp_path / "unused.db", plan.registry, chat_call=lambda **_: {}
    )
    request = service._build_model_request(
        plan.config,
        plan.api_endpoint,
        list(plan.schemas.runtime_schemas),
        list(plan.messages),
        plan.schemas.active_schemas,
        plan.schemas.log_active,
    )
    wire_rows = _prepared_local_reasoning_rows(resolution, list(request.messages))
    assert any("reasoning_content" in row for row in wire_rows) is (mode != "off")
    assert builder.requests[0].available_input_tokens == (
        limit
        - resolution.max_tokens
        - count_console_messages_tokens(wire_rows, "reasoner")
    )


@pytest.mark.parametrize("mode", ["off", "all"])
def test_fenced_instruction_capacity_counts_the_dispatched_reasoning_projection(
    monkeypatch,
    mode,
):
    from dataclasses import replace

    from tldw_chatbook.Chat.console_history_budget import count_console_messages_tokens
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
    from tldw_chatbook.Chat.local_reasoning import ReasoningReplayPolicy

    rows = _local_reasoning_rows()
    policy = ReasoningReplayPolicy(mode, "test")
    resolution = ConsoleProviderResolution(
        provider="local_vllm",
        execution_key="local_vllm",
        base_url="",
        model="reasoner",
        ready=True,
        local_structured_thinking=True,
        reasoning_replay=policy,
    )
    wire_off = _prepared_local_reasoning_rows(
        replace(resolution, reasoning_replay=replace(policy, mode="off")),
        rows,
    )
    wire_all = _prepared_local_reasoning_rows(
        replace(resolution, reasoning_replay=replace(policy, mode="all")),
        rows,
    )
    reserve = 10
    limit = (
        reserve
        + (
            count_console_messages_tokens(wire_off, "reasoner")
            + count_console_messages_tokens(wire_all, "reasoner")
        )
        // 2
    )
    monkeypatch.setattr(bridge_module, "get_model_token_limit", lambda *_: limit)
    assert bridge_module._fenced_project_instruction_payload_fits(
        rows,
        model="reasoner",
        provider="local_vllm",
        response_reserve_tokens=reserve,
        reasoning_replay=policy,
    ) is (mode == "off")


def test_first_request_omits_unsupported_only_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Personal_Context.context_service import ProfileContextService
    from tldw_chatbook.Personal_Context.service import AuthorizedProfileContextView

    monkeypatch.setattr(
        "tldw_chatbook.Personal_Context.context_service.estimate_tokens",
        lambda text, **_kwargs: 0 if not text else (len(text) + 3) // 4,
    )
    view = AuthorizedProfileContextView(
        generation=1,
        record_set_revision="manifest-v1",
        workspace_scope_id=None,
        authority_revision="authority-v1",
        records=(),
        unsupported_records_present=True,
    )
    source = SimpleNamespace(authorized_context_view=lambda **_kwargs: view)
    plan = _plan(ProfileContextService(source))
    assert plan.profile_context_snapshot.serialized_block == ""
    assert plan.config.personal_context_block == ""
    from tldw_chatbook.Agents.agent_service import append_personal_context

    assert append_personal_context("BASE", plan.config.personal_context_block) == "BASE"
    assert append_personal_context("BASE", PROFILE_BLOCK) == "BASE\n\n" + PROFILE_BLOCK
