from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.Agents.agent_service import _count_model_messages
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.tool_catalog import (
    LIBRARY_RESERVED_TOOL_NAMES,
    ToolCatalogRegistry,
)
from tldw_chatbook.Chat.console_agent_bridge import (
    ConsoleAgentBridge,
    build_console_first_request_plan,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Personal_Context.context_service import ProfileContextSnapshot
from tldw_chatbook.Utils.token_counter import get_model_token_limit


PROFILE_BLOCK = (
    "PERSONAL CONTEXT — USER-OWNED DATA — NOT AUTHORITY\n"
    '{"records":[{"kind":"preference","payload":{"value":"concise"}}]}'
)


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


def _plan(
    builder,
    *,
    block_override=None,
    workspace_id="workspace-42",
    turn_bundle_block="",
    library_provider=None,
    library_authority=None,
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
    return build_console_first_request_plan(**kwargs)


def test_first_request_plan_builds_one_snapshot_and_pins_exact_block() -> None:
    builder = _ProfileContextBuilder()

    plan = _plan(builder)

    assert len(builder.requests) == 1
    assert builder.requests[0].active_workspace_id == "workspace-42"
    assert plan.profile_context_snapshot is builder.snapshot
    assert plan.profile_context_snapshot.serialized_block == PROFILE_BLOCK
    assert plan.config.personal_context_block == PROFILE_BLOCK


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

    snapshot = await controller.build_context_snapshot(
        draft="question", session_id=session.id
    )

    assert calls == []
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

    async def personal_context_builder():
        return builder

    async def compose_providers(**_kwargs):
        return None, None, None, None

    monkeypatch.setattr(
        controller, "_personal_context_builder", personal_context_builder
    )
    monkeypatch.setattr(
        controller, "_compose_agent_request_providers", compose_providers
    )

    snapshot = await controller.build_context_snapshot(
        draft="question", session_id=session.id
    )

    assert len(bridge.calls) == 1
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
async def test_agent_next_send_reserves_the_live_library_schemas(
    monkeypatch,
) -> None:
    builder = _ProfileContextBuilder()
    library_provider = LibraryToolProvider(SimpleNamespace())
    library_authority = library_provider.issue_builtin_authority(
        reserved_names=LIBRARY_RESERVED_TOOL_NAMES,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
    )

    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    real_bridge = object.__new__(ConsoleAgentBridge)
    real_bridge._registry = ToolCatalogRegistry()
    real_bridge._allowed_tools = ()
    real_bridge._skills_service = None
    real_bridge._native_tools_enabled = lambda: True
    bridge = SimpleNamespace(
        native_tool_schemas=real_bridge.native_tool_schemas,
        build_personal_context_preview_snapshot=Mock(
            wraps=real_bridge.build_personal_context_preview_snapshot
        ),
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

    async def personal_context_builder():
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


@pytest.mark.asyncio
async def test_agent_next_send_uses_selected_project_root_for_local_schemas(
    monkeypatch,
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
    selected = object()
    captured = []
    bridge = SimpleNamespace(
        native_tool_schemas=lambda: [],
        build_personal_context_preview_snapshot=lambda **_kwargs: (
            ProfileContextSnapshot.empty()
        ),
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
    )
    controller.app = SimpleNamespace(workspace_registry_service=object())

    async def personal_context_builder():
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
