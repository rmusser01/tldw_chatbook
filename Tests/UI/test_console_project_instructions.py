"""Console project-instruction display, chooser, notice, and preview contracts."""

from __future__ import annotations

import asyncio
import copy
import importlib
import inspect
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Footer, Static

from tldw_chatbook.Agents import agent_service
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    RunBudget,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_service import (
    RUN_LOG_PROMPT_SECTION,
    AgentService,
    RunLogRequestPlan,
)
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionOutcome,
    InstructionSource,
    StartupInstructionCandidate,
)
from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    ToolCatalogRegistry,
    initial_disclosure,
)
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ProjectInstructionDispatchNotice,
)
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
import tldw_chatbook.Chat.console_agent_bridge as bridge_module
import tldw_chatbook.Chat.console_chat_controller as controller_module
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, MessageAttachment
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)
from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding


def _ui_module():
    return importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_project_instructions"
    )


def _candidate(tmp_path: Path) -> StartupInstructionCandidate:
    body = "AUTOMATIC_BODY_ONLY_IN_EXPLICIT_PREVIEW"
    source = InstructionSource(
        canonical_path=tmp_path / "AGENTS.md",
        relative_path="AGENTS.md",
        scope=".",
        kind="standard",
        body=body,
        byte_count=len(body.encode()),
        digest="d" * 64,
    )
    return StartupInstructionCandidate(
        binding_id="binding-1",
        binding_root=tmp_path,
        locator_fingerprint="f" * 64,
        dispatch_started_wall_ns=1,
        source=source,
        outcomes=(),
    )


class _PreviewCatalogProvider:
    def __init__(self, name: str, source: str) -> None:
        self.name = name
        self.source = source

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=self.name,
                name=self.name,
                one_line_description=f"{self.source} tool",
                source=self.source,
            )
        ]

    def load_schema(self, tool_id):
        return ToolSchema(
            id=tool_id,
            name=self.name,
            description=f"{self.source} schema",
            parameters={"type": "object", "properties": {}},
        )

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content=f"{tool_id}:{args}")


def test_display_states_are_metadata_only():
    display = importlib.import_module("tldw_chatbook.Chat.console_display_state")
    build = display.build_console_project_instruction_state
    row = display.ConsoleProjectInstructionSourceRow(
        relative_source="AGENTS.md",
        scope=".",
        byte_count=12,
        outcome="active",
    )
    cases = [
        (
            ProjectInstructionControlState.legacy_disabled(),
            {},
            "Off",
        ),
        (
            ProjectInstructionControlState.new_session(),
            {},
            "Choose folder",
        ),
        (
            ProjectInstructionControlState(True, "b", "f" * 64, None),
            {"binding_label": "Repo", "locator_matches": True},
            "None",
        ),
        (
            ProjectInstructionControlState(True, "b", "f" * 64, None),
            {
                "binding_label": "Repo",
                "locator_matches": True,
                "sources": (row,),
            },
            "1 loaded",
        ),
        (
            ProjectInstructionControlState(True, "b", "f" * 64, None),
            {
                "binding_label": "Repo",
                "locator_matches": False,
                "warning_codes": ("binding_retargeted",),
            },
            "Warning",
        ),
    ]
    for control, kwargs, expected in cases:
        state = build(control, **kwargs)
        assert state.status == expected
        assert "body" not in state.__dataclass_fields__
        assert all("body" not in item.__dataclass_fields__ for item in state.sources)


@pytest.mark.asyncio
async def test_display_revalidates_binding_and_drops_stale_loaded_metadata(tmp_path):
    original = tmp_path / "original"
    retargeted = tmp_path / "retargeted"
    original.mkdir()
    retargeted.mkdir()
    binding = WorkspaceRuntimeBinding(
        workspace_id="workspace-1",
        binding_id="binding-1",
        binding_kind="local-filesystem",
        label="Repo [literal]",
        locator=str(retargeted),
        status="ready",
        metadata={"access": "rw"},
    )
    registry = SimpleNamespace(
        get_runtime_binding=lambda _binding_id: binding,
        list_runtime_bindings=lambda _workspace_id: (binding,),
    )
    control = ProjectInstructionControlState(
        True,
        binding.binding_id,
        controller_module.fingerprint_canonical_locator(str(original)),
        None,
    )
    session = SimpleNamespace(
        id="session-1",
        workspace_id="workspace-1",
        project_instruction_state=control,
    )
    store = SimpleNamespace(active_session_id=session.id, sessions=lambda: (session,))
    stale_metadata = SimpleNamespace(
        warning_codes=(),
        relative_source="AGENTS.md",
        scope=".",
        byte_count=99,
        outcome="active",
    )
    backend = SimpleNamespace(
        project_instruction_display_metadata=lambda _session_id: stale_metadata,
        _clear_project_instruction_delivery=Mock(),
    )
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: store
    controller._ensure_console_chat_controller_fn = lambda: backend
    controller.app_instance = SimpleNamespace(workspace_registry_service=registry)

    state = await controller._refresh_console_project_instruction_display_state(
        session.id
    )

    assert state.status == "Warning"
    assert state.binding_label == "Repo [literal]"
    assert state.locator_match == "mismatch"
    assert state.sources == ()
    assert state.warning_codes


@pytest.mark.asyncio
async def test_status_sync_never_resolves_authority_and_async_refresh_publishes(
    tmp_path, monkeypatch
):
    display = importlib.import_module("tldw_chatbook.Chat.console_display_state")
    root = tmp_path / "repo"
    root.mkdir()
    control = ProjectInstructionControlState(True, "binding-1", "f" * 64, None)
    store = ConsoleChatStore()
    session = store.create_session(
        workspace_id="workspace-1", project_instruction_state=control
    )
    binding = WorkspaceRuntimeBinding(
        workspace_id="workspace-1",
        binding_id="binding-1",
        binding_kind="local-filesystem",
        label="Repo [literal]",
        locator=str(root),
        status="ready",
        metadata={"access": "rw"},
    )
    registry = SimpleNamespace(get_runtime_binding=lambda _binding_id: binding)
    loaded = display.build_console_project_instruction_state(
        control,
        binding_label="old",
        locator_matches=None,
    )
    row = SimpleNamespace(sync_state=Mock())
    metadata = {
        "value": SimpleNamespace(
            warning_codes=(),
            relative_source="AGENTS.md",
            scope=".",
            byte_count=99,
            outcome="active",
        )
    }
    backend = SimpleNamespace(
        project_instruction_display_metadata=lambda _session_id: metadata["value"],
        _clear_project_instruction_delivery=Mock(),
    )
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: store
    controller._ensure_console_chat_controller_fn = lambda: backend
    controller.app_instance = SimpleNamespace(workspace_registry_service=registry)
    controller._screen = SimpleNamespace(query_one=lambda *_args, **_kwargs: row)
    controller._console_project_instruction_display_cache = {
        session.id: (control, loaded)
    }
    controller._console_project_instruction_refresh_inflight = {
        session.id: (control, metadata["value"])
    }
    started = threading.Event()
    release = threading.Event()

    def slow_resolve(_session, _registry):
        started.set()
        assert release.wait(2)
        return SimpleNamespace(binding=binding)

    monkeypatch.setattr(
        "tldw_chatbook.UI.Console_Modules.session.resolve_project_instruction_binding",
        slow_resolve,
    )

    refresh = asyncio.create_task(
        controller._refresh_console_project_instruction_display_state(session.id)
    )
    assert await asyncio.to_thread(started.wait, 1)

    # The 5 Hz UI tick consumes only the cached DTO; the resolver remains
    # blocked in its disposable worker and is never called a second time.
    controller._sync_console_project_instruction_status_row()
    row.sync_state.assert_called_once_with(loaded)

    metadata["value"] = None
    release.set()
    state = await refresh
    assert state.binding_label == "Repo [literal]"
    assert state.locator_match == "match"
    assert state.sources == ()
    assert controller._build_console_project_instruction_display_state() == state


def test_status_sync_schedules_bounded_async_authority_refresh():
    display = importlib.import_module("tldw_chatbook.Chat.console_display_state")
    control = ProjectInstructionControlState(True, "binding-1", "f" * 64, None)
    store = ConsoleChatStore()
    session = store.create_session(project_instruction_state=control)
    cached = display.build_console_project_instruction_state(control)
    row = SimpleNamespace(sync_state=Mock())
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: store
    controller._screen = SimpleNamespace(query_one=lambda *_args, **_kwargs: row)
    controller._console_project_instruction_display_cache = {
        session.id: (control, cached)
    }
    controller._request_console_project_instruction_display_refresh = Mock()

    controller._sync_console_project_instruction_status_row()

    row.sync_state.assert_called_once_with(cached)
    controller._request_console_project_instruction_display_refresh.assert_called_once_with(
        session.id
    )


@pytest.mark.asyncio
async def test_authority_refresh_is_snapshot_driven_and_explicitly_invalidatable():
    display = importlib.import_module("tldw_chatbook.Chat.console_display_state")
    control = ProjectInstructionControlState(True, "binding-1", "f" * 64, None)
    store = ConsoleChatStore()
    session = store.create_session(project_instruction_state=control)
    state = display.build_console_project_instruction_state(control)
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: store
    controller._ensure_console_chat_controller_fn = lambda: SimpleNamespace(
        project_instruction_display_metadata=lambda _session_id: None
    )
    controller._console_project_instruction_refresh_inflight = {}
    controller._console_project_instruction_refresh_completed = {}
    controller._refresh_console_project_instruction_display_state = AsyncMock(
        return_value=state
    )
    controller._active_native_console_session = lambda: None
    tasks = []
    controller._screen = SimpleNamespace(
        run_worker=lambda coroutine, **_kwargs: tasks.append(
            asyncio.create_task(coroutine)
        )
    )

    controller._request_console_project_instruction_display_refresh(session.id)
    await tasks[-1]
    controller._request_console_project_instruction_display_refresh(session.id)
    await asyncio.sleep(0)
    assert (
        controller._refresh_console_project_instruction_display_state.await_count == 1
    )

    changed = ProjectInstructionControlState(True, "binding-2", "e" * 64, None)
    store.set_session_project_instruction_state(session.id, changed)
    controller._request_console_project_instruction_display_refresh(session.id)
    await tasks[-1]
    assert (
        controller._refresh_console_project_instruction_display_state.await_count == 2
    )

    controller._request_console_project_instruction_display_refresh(
        session.id, force=True
    )
    await tasks[-1]
    assert (
        controller._refresh_console_project_instruction_display_state.await_count == 3
    )


def test_disposable_preview_matches_live_exact_request_when_source_is_omitted(
    tmp_path, monkeypatch
):
    candidate = _candidate(tmp_path)
    registry = ToolCatalogRegistry()
    registry.register_provider(BuiltinToolProvider())
    config = AgentConfig(
        model="gpt-4o-mini",
        system_prompt="system",
        allowed_tools=(),
        budget=RunBudget(max_subagents=0),
        native_tools=False,
        response_reserve_tokens=10,
    )
    active, _offer_find_load = initial_disclosure(registry, config.budget)
    active = tuple(schema for schema in active if schema.name in config.allowed_tools)
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a, **_k: 100)
    monkeypatch.setattr(agent_service, "count_tokens_messages", lambda *_a, **_k: 20)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_a, **_k: 0)
    monkeypatch.setattr(agent_service, "_count_model_messages", lambda *_a, **_k: 95)
    preview_consent = Mock(side_effect=AssertionError("preview requested consent"))
    preview_service = AgentService(
        AgentRunsDB(tmp_path / "preview.db", client_id="test"),
        registry,
        chat_call=Mock(side_effect=AssertionError("preview called provider")),
        confirm_project_instruction_dispatch=preview_consent,
    )
    build_exact = getattr(preview_service, "build_project_instruction_request", None)

    assert callable(build_exact)
    preview_request, preview_snapshot = build_exact(
        candidate=candidate,
        config=config,
        api_endpoint="openai",
        runtime_schemas=[],
        messages=[{"role": "user", "content": "question"}],
        active_schemas=active,
    )

    calls = []

    def chat_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "done"}}]}

    live_service = AgentService(
        AgentRunsDB(tmp_path / "live.db", client_id="test"),
        registry,
        chat_call=chat_call,
        startup_instruction_candidate=candidate,
        confirm_project_instruction_dispatch=lambda _snapshot: "proceed",
    )
    live_service.run_turn(
        conversation_id="conversation",
        messages=[{"role": "user", "content": "question"}],
        config=config,
        api_endpoint="openai",
    )

    assert [item.code for item in preview_snapshot.primary_delivery.outcomes] == [
        "omitted_token_budget"
    ]
    assert "AUTOMATIC_BODY_ONLY_IN_EXPLICIT_PREVIEW" not in str(
        preview_request.messages
    )
    assert list(preview_request.messages) == calls[0]["messages_payload"]
    assert list(preview_request.tools) == calls[0].get("tools", [])
    preview_consent.assert_not_called()


@pytest.mark.asyncio
async def test_preview_fails_closed_when_run_log_binding_is_uncertain(
    tmp_path, monkeypatch
):
    candidate = _candidate(tmp_path)
    control = ProjectInstructionControlState(True, "binding-1", "f" * 64, None)
    store = ConsoleChatStore()
    session = store.create_session(project_instruction_state=control)
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="question")
    resolution = SimpleNamespace(
        ready=True,
        provider="openai",
        execution_key="openai",
        model="model",
        max_tokens=20,
    )
    gateway = SimpleNamespace(resolve_for_send=AsyncMock(return_value=resolution))
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "uncertain-log.db", client_id="test"),
        store=store,
        provider_gateway=gateway,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        model="model",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    selection = SimpleNamespace(
        binding=SimpleNamespace(binding_id="binding-1"),
        root=tmp_path,
        locator_fingerprint="f" * 64,
        allow_write=True,
        root_identity=controller_module._capture_project_root_identity(tmp_path),
    )
    controller.app = SimpleNamespace(
        workspace_registry_service=SimpleNamespace(
            get_runtime_binding=lambda _binding_id: selection.binding
        ),
        unified_mcp_service=None,
    )
    monkeypatch.setattr(
        bridge_module,
        "build_run_log_request_plan",
        lambda: RunLogRequestPlan(True, True, 1),
    )
    monkeypatch.setattr(
        controller_module,
        "resolve_project_instruction_binding",
        lambda _session, _registry: selection,
    )
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        lambda _resolver, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        controller_module,
        "project_instruction_authority_snapshot_is_current",
        lambda **_kwargs: True,
    )

    snapshot = await controller.build_context_snapshot("", session_id=session.id)

    preview = snapshot.project_instruction_preview
    assert preview is not None
    assert preview.outcomes == ("preview_uncertain_run_log_binding",)
    assert preview.warning_codes == ("preview_uncertain_run_log_binding",)
    assert "AUTOMATIC_BODY_ONLY_IN_EXPLICIT_PREVIEW" not in str(
        preview.next_send_payload
    )


@pytest.mark.asyncio
async def test_controller_preview_uses_live_destination_fresh_tools_and_raw_admission(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        bridge_module,
        "build_run_log_request_plan",
        lambda: RunLogRequestPlan(False, True, 1),
    )
    secret = "api_key=" + "s" * 80
    candidate = _candidate(tmp_path)
    control = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-1",
        working_folder_locator_fingerprint="f" * 64,
        project_instruction_notice_key=None,
    )
    store = ConsoleChatStore()
    session = store.create_session(project_instruction_state=control)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content=f"question {secret}",
    )
    resolution = SimpleNamespace(
        ready=True,
        provider="friendly-alias",
        execution_key="openai",
        model="resolved-model",
        max_tokens=20,
        base_url="https://example.invalid/v1",
    )

    class Gateway:
        def __init__(self) -> None:
            self.resolve_for_send = AsyncMock(return_value=resolution)

    gateway = Gateway()
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "preview-plan.db", client_id="test"),
        store=store,
        provider_gateway=gateway,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="stale-display-provider",
        model="stale-model",
        max_tokens=20,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    selection = SimpleNamespace(
        binding=SimpleNamespace(binding_id="binding-1"),
        root=tmp_path,
        locator_fingerprint="f" * 64,
        allow_write=True,
        root_identity=controller_module._capture_project_root_identity(tmp_path),
    )
    controller.app = SimpleNamespace(
        workspace_registry_service=SimpleNamespace(
            get_runtime_binding=lambda _binding_id: selection.binding
        ),
        unified_mcp_service=None,
    )
    monkeypatch.setattr(
        controller_module,
        "resolve_project_instruction_binding",
        lambda _session, _registry: selection,
    )
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        lambda _resolver, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        controller_module,
        "project_instruction_authority_snapshot_is_current",
        lambda **_kwargs: True,
    )
    mcp_provider = _PreviewCatalogProvider("mcp__srv__search", "mcp")
    local_provider = _PreviewCatalogProvider("fs_read", "local")
    controller._compose_mcp_provider = AsyncMock(return_value=mcp_provider)
    controller._compose_local_provider = Mock(return_value=(local_provider, None))

    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a, **_k: 200)

    def count_messages(messages, *_args, **_kwargs):
        rendered = str(messages)
        if "AUTOMATIC_BODY_ONLY_IN_EXPLICIT_PREVIEW" in rendered:
            return 20
        return 190 if secret in rendered else 10

    monkeypatch.setattr(agent_service, "count_tokens_messages", count_messages)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_a, **_k: 0)

    snapshot = await controller.build_context_snapshot("", session_id=session.id)

    preview = snapshot.project_instruction_preview
    assert preview is not None
    assert "omitted_token_budget" in preview.outcomes
    assert preview.next_send_payload["model"] == "resolved-model"
    assert secret not in str(preview.next_send_payload)
    assert "AUTOMATIC_BODY_ONLY_IN_EXPLICIT_PREVIEW" not in str(
        preview.next_send_payload
    )
    native_names = {
        item["function"]["name"] for item in preview.next_send_payload["tools"]
    }
    assert {"fs_read", "mcp__srv__search"} <= native_names
    assert "search_run_log" not in native_names
    assert "install_skill" not in native_names
    assert "run_skill_script" not in native_names
    assert (
        RUN_LOG_PROMPT_SECTION
        not in preview.next_send_payload["messages"][0]["content"]
    )
    gateway.resolve_for_send.assert_awaited_once()
    controller._compose_mcp_provider.assert_awaited_once_with(
        session.id, publish_counts=False
    )
    controller._compose_local_provider.assert_called_once()


@pytest.mark.asyncio
async def test_controller_preview_applies_live_skill_turn_before_admission(
    tmp_path, monkeypatch
):
    """A triggering skill must change the disposable request exactly as live."""
    monkeypatch.setattr(
        bridge_module,
        "build_run_log_request_plan",
        lambda: RunLogRequestPlan(False, True, 1),
    )

    class Skills:
        async def get_context(self, *, mode="local"):
            return {
                "available_skills": [
                    {
                        "name": "code-review",
                        "description": "Review code",
                        "user_invocable": True,
                        "trust_blocked": False,
                        "disable_model_invocation": False,
                    }
                ],
                "blocked_skills": [],
            }

        async def execute_skill(self, name, *, mode="local", args=None):
            assert (name, mode, args) == ("code-review", "local", "the diff")
            return {
                "rendered_prompt": "RENDERED SKILL TURN",
                "execution_mode": "inline",
                "allowed_tools": None,
                "reference_files": [
                    {"path": "references/checklist.md", "size": 12, "is_text": True}
                ],
            }

    candidate = _candidate(tmp_path)
    control = ProjectInstructionControlState(True, "binding-1", "f" * 64, None)
    store = ConsoleChatStore()
    session = store.create_session(project_instruction_state=control)
    store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="$code-review the diff"
    )
    resolution = SimpleNamespace(
        ready=True,
        provider="alias",
        execution_key="openai",
        model="resolved-model",
        max_tokens=20,
    )
    gateway = SimpleNamespace(resolve_for_send=AsyncMock(return_value=resolution))
    skills = Skills()
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "skill-preview.db", client_id="test"),
        store=store,
        provider_gateway=gateway,
        skills_service=skills,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
        skills_service=skills,
    )
    selection = SimpleNamespace(
        binding=SimpleNamespace(binding_id="binding-1"),
        root=tmp_path,
        locator_fingerprint="f" * 64,
        allow_write=True,
        root_identity=controller_module._capture_project_root_identity(tmp_path),
    )
    controller.app = SimpleNamespace(
        workspace_registry_service=SimpleNamespace(
            get_runtime_binding=lambda _binding_id: selection.binding
        ),
        unified_mcp_service=None,
    )
    monkeypatch.setattr(
        controller_module,
        "resolve_project_instruction_binding",
        lambda _session, _registry: selection,
    )
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        lambda _resolver, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        controller_module,
        "project_instruction_authority_snapshot_is_current",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a, **_k: 100)
    monkeypatch.setattr(
        agent_service,
        "_count_model_messages",
        lambda messages, *_a, **_k: 95 if "$code-review" in str(messages) else 20,
    )
    monkeypatch.setattr(agent_service, "count_tokens_messages", lambda *_a, **_k: 5)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_a, **_k: 0)

    snapshot = await controller.build_context_snapshot("", session_id=session.id)

    preview = snapshot.project_instruction_preview
    assert preview is not None
    assert "omitted_token_budget" not in preview.outcomes
    rendered = str(preview.next_send_payload)
    assert "$code-review" not in rendered
    assert "RENDERED SKILL TURN" in rendered
    assert "Bundled files" in rendered
    assert "AUTOMATIC_BODY_ONLY_IN_EXPLICIT_PREVIEW" in rendered
    tool_names = {
        item["function"]["name"] for item in preview.next_send_payload["tools"]
    }
    assert {"code-review", "skill_file"} <= tool_names


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["disable", "remove", "retarget", "replace_root"])
async def test_preview_revalidates_authority_after_awaited_composition(
    tmp_path, monkeypatch, race
):
    root = tmp_path / "root"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    fingerprint = controller_module.fingerprint_canonical_locator(str(root))
    candidate = replace(
        _candidate(root), binding_root=root, locator_fingerprint=fingerprint
    )
    binding = WorkspaceRuntimeBinding(
        workspace_id="workspace-1",
        binding_id="binding-1",
        binding_kind="local-filesystem",
        label="Repo",
        locator=str(root),
        status="ready",
        metadata={"access": "rw"},
    )

    class Registry:
        current = binding

        def get_runtime_binding(self, _binding_id):
            return self.current

    registry = Registry()
    control = ProjectInstructionControlState(True, "binding-1", fingerprint, None)
    store = ConsoleChatStore()
    session = store.create_session(
        workspace_id="workspace-1", project_instruction_state=control
    )
    resolution = SimpleNamespace(
        ready=True, provider="openai", execution_key="openai", model="m", max_tokens=20
    )
    gateway = SimpleNamespace(resolve_for_send=AsyncMock(return_value=resolution))

    async def compose(**_kwargs):
        return None, Mock(), None, None

    bridge = SimpleNamespace(
        build_project_instruction_preview_request=Mock(
            return_value=(
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": candidate.source.body}],
                },
                SimpleNamespace(
                    startup_source_metadata=None,
                    startup_source=candidate.source,
                    primary_delivery=SimpleNamespace(outcomes=()),
                    warning_codes=(),
                ),
            )
        )
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    controller.app = SimpleNamespace(workspace_registry_service=registry)
    controller._compose_agent_request_providers = compose
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        lambda _resolver, **_kwargs: candidate,
    )
    loop = asyncio.get_running_loop()

    def mutate_after_authority_check():
        if race == "disable":
            store.set_session_project_instruction_state(
                session.id, ProjectInstructionControlState.legacy_disabled()
            )
        elif race == "remove":
            registry.current = None
        elif race == "retarget":
            registry.current = replace(binding, locator=str(other))
        else:
            root.rmdir()
            other.rename(root)

    def authority_check(**_kwargs):
        loop.call_soon_threadsafe(mutate_after_authority_check)
        return True

    monkeypatch.setattr(
        controller_module,
        "project_instruction_authority_snapshot_is_current",
        authority_check,
    )

    preview = await controller._build_project_instruction_preview_for_session(
        session.id,
        {"messages": [{"role": "user", "content": "question"}]},
        [{"role": "user", "content": "question"}],
    )

    assert preview is None


@pytest.mark.asyncio
async def test_parked_session_preview_does_not_publish_global_mcp_counts(
    tmp_path, monkeypatch
):
    candidate = StartupInstructionCandidate(
        binding_id="binding-1",
        binding_root=tmp_path,
        locator_fingerprint="f" * 64,
        dispatch_started_wall_ns=1,
        source=None,
        outcomes=(),
    )
    control = ProjectInstructionControlState(True, "binding-1", "f" * 64, None)
    store = ConsoleChatStore()
    parked = store.create_session(project_instruction_state=control)
    store.create_session(
        project_instruction_state=ProjectInstructionControlState.legacy_disabled()
    )
    resolution = SimpleNamespace(
        ready=True, provider="openai", execution_key="openai", model="m", max_tokens=20
    )
    gateway = SimpleNamespace(resolve_for_send=AsyncMock(return_value=resolution))
    bridge = SimpleNamespace(
        build_project_instruction_preview_request=Mock(
            return_value=(
                {"model": "m", "messages": [{"role": "user", "content": "q"}]},
                SimpleNamespace(
                    startup_source_metadata=None,
                    startup_source=None,
                    primary_delivery=SimpleNamespace(outcomes=()),
                    warning_codes=(),
                ),
            )
        )
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    app = SimpleNamespace(
        workspace_registry_service=SimpleNamespace(
            get_runtime_binding=lambda _binding_id: selection.binding
        ),
        unified_mcp_service=object(),
        console_mcp_tool_count=41,
        console_mcp_not_connected_count=7,
    )
    controller.app = app
    selection = SimpleNamespace(
        binding=SimpleNamespace(binding_id="binding-1"),
        root=tmp_path,
        locator_fingerprint="f" * 64,
        allow_write=True,
        root_identity=controller_module._capture_project_root_identity(tmp_path),
    )
    monkeypatch.setattr(
        controller_module,
        "resolve_project_instruction_binding",
        lambda _session, _registry: selection,
    )
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        lambda _resolver, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        controller_module,
        "project_instruction_authority_snapshot_is_current",
        lambda **_kwargs: True,
    )

    async def compose_mcp(_session_id, *, publish_counts=True):
        if publish_counts:
            controller._publish_mcp_inspector_counts(1, 2)
        return None

    controller._compose_mcp_provider = AsyncMock(side_effect=compose_mcp)

    preview = await controller._build_project_instruction_preview_for_session(
        parked.id,
        {"messages": [{"role": "user", "content": "q"}]},
        [{"role": "user", "content": "q"}],
    )

    assert preview is not None
    assert app.console_mcp_tool_count == 41
    assert app.console_mcp_not_connected_count == 7
    controller._compose_mcp_provider.assert_awaited_once_with(
        parked.id, publish_counts=False
    )


def test_controller_has_no_candidate_or_body_cache():
    controller = ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=Mock(),
    )
    assert "_project_instruction_candidates" not in vars(controller)
    assert all(
        "body" not in name and "candidate" not in name
        for name in vars(controller)
        if name.startswith("_project_instruction")
    )


@pytest.mark.parametrize("race", ["disable", "remove", "retarget"])
def test_setup_choice_cancels_if_state_or_binding_changes_while_modal_is_open(
    tmp_path, race
):
    root = tmp_path / "root"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    initial_binding = WorkspaceRuntimeBinding(
        workspace_id="workspace-1",
        binding_id="binding-1",
        binding_kind="local-filesystem",
        label="Repo",
        locator=str(root),
        status="ready",
        metadata={"access": "rw"},
    )

    class Registry:
        bindings = {initial_binding.binding_id: initial_binding}

        def list_runtime_bindings(self, workspace_id):
            return tuple(
                binding
                for binding in self.bindings.values()
                if binding.workspace_id == workspace_id
            )

        def get_runtime_binding(self, binding_id):
            return self.bindings.get(binding_id)

    registry = Registry()
    store = ConsoleChatStore()
    session = store.create_session(
        workspace_id="workspace-1",
        project_instruction_state=ProjectInstructionControlState.new_session(),
    )
    expected_state = copy.deepcopy(session.project_instruction_state)
    options = controller_module.list_project_instruction_bindings(session, registry)
    expected_selection = options[0]
    if race == "disable":
        session.project_instruction_state = (
            ProjectInstructionControlState.legacy_disabled()
        )
    elif race == "remove":
        registry.bindings.clear()
    else:
        registry.bindings[initial_binding.binding_id] = WorkspaceRuntimeBinding(
            workspace_id="workspace-1",
            binding_id="binding-1",
            binding_kind="local-filesystem",
            label="Repo",
            locator=str(other),
            status="ready",
            metadata={"access": "rw"},
        )
    original_setter = store.set_session_project_instruction_state
    setter = Mock(wraps=original_setter)
    store.set_session_project_instruction_state = setter
    commit = getattr(
        controller_module, "commit_project_instruction_setup_decision", None
    )

    assert callable(commit)
    action, selection = commit(
        store=store,
        session_id=session.id,
        registry=registry,
        expected_state=expected_state,
        expected_options=options,
        action="select",
        binding_id=expected_selection.binding.binding_id,
    )

    assert action == "cancel"
    assert selection is None
    setter.assert_not_called()


@pytest.mark.asyncio
async def test_controller_preview_does_not_advance_live_session_state(
    tmp_path, monkeypatch
):
    candidate = StartupInstructionCandidate(
        binding_id="binding-1",
        binding_root=tmp_path,
        locator_fingerprint="f" * 64,
        dispatch_started_wall_ns=1,
        source=None,
        outcomes=(InstructionOutcome("AGENTS.md", ".", "omitted_token_budget"),),
    )
    control = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-1",
        working_folder_locator_fingerprint="f" * 64,
        project_instruction_notice_key="n" * 64,
    )
    store = ConsoleChatStore()
    session = store.create_session(project_instruction_state=control)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    consent = Mock(side_effect=AssertionError("preview acknowledged notice"))
    exact_builder = Mock(
        return_value=(
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "question"}],
            },
            SimpleNamespace(
                startup_source_metadata=None,
                startup_source=None,
                primary_delivery=SimpleNamespace(outcomes=candidate.outcomes),
                warning_codes=("omitted_token_budget",),
            ),
        )
    )
    resolution = SimpleNamespace(
        ready=True,
        provider="openai",
        execution_key="openai",
        model="gpt-test",
        max_tokens=20,
    )
    gateway = SimpleNamespace(resolve_for_send=AsyncMock(return_value=resolution))
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=SimpleNamespace(
            build_project_instruction_preview_request=exact_builder
        ),
        agent_runtime_enabled=True,
        confirm_project_instruction_dispatch=consent,
    )
    selection = SimpleNamespace(
        binding=SimpleNamespace(binding_id="binding-1"),
        root=tmp_path,
        locator_fingerprint="f" * 64,
        allow_write=True,
        root_identity=controller_module._capture_project_root_identity(tmp_path),
    )
    controller.app = SimpleNamespace(
        workspace_registry_service=SimpleNamespace(
            get_runtime_binding=lambda _binding_id: selection.binding
        ),
        unified_mcp_service=None,
    )
    controller._run_state_histories[session.id] = []
    monkeypatch.setattr(
        controller_module,
        "resolve_project_instruction_binding",
        lambda _session, _registry: selection,
    )
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        lambda _resolver, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        controller_module,
        "project_instruction_authority_snapshot_is_current",
        lambda **_kwargs: True,
    )
    state_setter = Mock(side_effect=AssertionError("preview changed controls"))
    monkeypatch.setattr(store, "set_session_project_instruction_state", state_setter)

    before_messages = copy.deepcopy(store.messages_for_session(session.id))
    before_control = copy.deepcopy(session.project_instruction_state)
    before_display = dict(controller._project_instruction_display)
    before_run_states = dict(controller._run_states)
    before_run_history = copy.deepcopy(controller._run_state_histories)
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "question"}],
        "admission": {"warning": "omitted_token_budget"},
    }
    preview = await controller._build_project_instruction_preview_for_session(
        session.id,
        payload,
        [{"role": "user", "content": "question"}],
    )

    assert preview is not None
    assert preview.outcomes == ("omitted_token_budget",)
    assert preview.warning_codes == ("omitted_token_budget",)
    assert preview.next_send_payload == {**payload, "system": []}
    assert store.messages_for_session(session.id) == before_messages
    assert session.project_instruction_state == before_control
    assert controller._project_instruction_display == before_display
    assert controller._run_states == before_run_states
    assert controller._run_state_histories == before_run_history
    consent.assert_not_called()
    state_setter.assert_not_called()
    exact_builder.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "runtime_enabled", "bridge_present"),
    [
        ("response_prefill", True, True),
        ("character", True, True),
        ("missing_bridge", True, False),
        ("runtime_disabled", False, True),
    ],
)
async def test_context_preview_never_reads_project_instructions_when_agent_dispatch_is_bypassed(
    tmp_path, monkeypatch, mode, runtime_enabled, bridge_present
):
    control = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-1",
        working_folder_locator_fingerprint="f" * 64,
        project_instruction_notice_key=None,
    )
    store = ConsoleChatStore()
    session = store.create_session(
        assistant_kind="character" if mode == "character" else "generic",
        project_instruction_state=control,
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    if mode == "response_prefill":
        store.set_session_one_shot_prefill(session.id, "prefilled response")

    bridge = (
        SimpleNamespace(
            build_project_instruction_preview_request=Mock(),
            native_tool_schemas=lambda: [],
        )
        if bridge_present
        else None
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=Mock(),
        agent_bridge=bridge,
        agent_runtime_enabled=runtime_enabled,
    )
    controller.app = SimpleNamespace(workspace_registry_service=object())
    binding_resolver = Mock(
        side_effect=AssertionError("bypassed dispatch resolved project binding")
    )
    startup_reader = Mock(
        side_effect=AssertionError("bypassed dispatch read AGENTS.md")
    )
    monkeypatch.setattr(
        controller_module, "resolve_project_instruction_binding", binding_resolver
    )
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        startup_reader,
    )

    snapshot = await controller.build_context_snapshot("", session_id=session.id)

    assert snapshot.project_instruction_preview is None
    binding_resolver.assert_not_called()
    startup_reader.assert_not_called()


@pytest.mark.asyncio
async def test_context_preview_without_bound_textual_app_degrades_only_preview():
    store = ConsoleChatStore()
    session = store.create_session(
        project_instruction_state=ProjectInstructionControlState.new_session()
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    controller = ConsoleChatController(store=store, provider_gateway=Mock())

    snapshot = await controller.build_context_snapshot(draft="next")

    assert "error" not in snapshot.next_send_payload
    assert snapshot.project_instruction_preview is None


@pytest.mark.asyncio
async def test_context_snapshot_stays_on_captured_session_after_active_switch():
    store = ConsoleChatStore()
    captured = store.create_session(
        title="Captured",
        project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
    )
    store.append_message(
        captured.id,
        role=ConsoleMessageRole.USER,
        content="captured transcript",
    )
    active = store.create_session(
        title="Active",
        project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
    )
    store.append_message(
        active.id,
        role=ConsoleMessageRole.USER,
        content="wrong active transcript",
    )
    controller = ConsoleChatController(store=store, provider_gateway=Mock())
    kwargs = (
        {"session_id": captured.id}
        if "session_id"
        in inspect.signature(controller.build_context_snapshot).parameters
        else {}
    )

    snapshot = await controller.build_context_snapshot(draft="captured draft", **kwargs)

    assert [message.content for message in snapshot.current_messages] == [
        "captured transcript"
    ]
    assert snapshot.next_send_payload["messages"][-1]["content"] == "captured draft"


@pytest.mark.asyncio
async def test_captured_context_uses_own_session_destination_model_system_and_reserve(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        bridge_module,
        "build_run_log_request_plan",
        lambda: RunLogRequestPlan(False, True, 1),
    )
    candidate = _candidate(tmp_path)
    control = ProjectInstructionControlState(True, "binding-1", "f" * 64, None)
    captured_settings = ConsoleSessionSettings(
        provider="openrouter",
        model="captured-model",
        max_tokens=70,
        system_prompt="CAPTURED SYSTEM",
    )
    active_settings = ConsoleSessionSettings(
        provider="openai",
        model="active-model",
        max_tokens=10,
        system_prompt="ACTIVE SYSTEM",
    )
    store = ConsoleChatStore()
    captured = store.create_session(
        settings=captured_settings, project_instruction_state=control
    )
    store.append_message(
        captured.id, role=ConsoleMessageRole.USER, content="captured question"
    )
    store.create_session(
        settings=active_settings,
        project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
    )

    class Gateway:
        def __init__(self):
            self.selections = []

        async def resolve_for_send(self, selection):
            self.selections.append(selection)
            return SimpleNamespace(
                ready=True,
                provider=selection.provider,
                execution_key=f"exec-{selection.provider}",
                model=selection.explicit_model,
                max_tokens=selection.max_tokens,
            )

    gateway = Gateway()
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "captured-preview.db", client_id="test"),
        store=store,
        provider_gateway=gateway,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="openai",
        model="active-model",
        max_tokens=10,
        system_prompt="ACTIVE SYSTEM",
        agent_bridge=bridge,
        agent_runtime_enabled=True,
    )
    selection = SimpleNamespace(
        binding=SimpleNamespace(binding_id="binding-1"),
        root=tmp_path,
        locator_fingerprint="f" * 64,
        allow_write=True,
        root_identity=controller_module._capture_project_root_identity(tmp_path),
    )
    controller.app = SimpleNamespace(
        workspace_registry_service=SimpleNamespace(
            get_runtime_binding=lambda _binding_id: selection.binding
        ),
        unified_mcp_service=None,
    )
    monkeypatch.setattr(
        controller_module,
        "resolve_project_instruction_binding",
        lambda _session, _registry: selection,
    )
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        lambda _resolver, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        controller_module,
        "project_instruction_authority_snapshot_is_current",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a, **_k: 100)
    monkeypatch.setattr(agent_service, "_count_model_messages", lambda *_a, **_k: 20)
    monkeypatch.setattr(agent_service, "count_tokens_messages", lambda *_a, **_k: 15)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_a, **_k: 0)

    snapshot = await controller.build_context_snapshot("", session_id=captured.id)

    preview = snapshot.project_instruction_preview
    assert preview is not None
    assert gateway.selections[0].provider == "openrouter"
    assert gateway.selections[0].explicit_model == "captured-model"
    assert gateway.selections[0].max_tokens == 70
    assert preview.next_send_payload["model"] == "captured-model"
    assert "CAPTURED SYSTEM" in preview.next_send_payload["messages"][0]["content"]
    assert "ACTIVE SYSTEM" not in str(preview.next_send_payload)
    assert "omitted_token_budget" in preview.outcomes


@pytest.mark.asyncio
async def test_captured_context_uses_provider_fallbacks_for_images_and_admission(
    tmp_path, monkeypatch
):
    candidate = _candidate(tmp_path)
    control = ProjectInstructionControlState(True, "binding-1", "f" * 64, None)
    captured_settings = ConsoleSessionSettings(provider="llama_cpp")
    active_settings = ConsoleSessionSettings(
        provider="openai", model="active-text", max_tokens=10
    )
    store = ConsoleChatStore()
    captured = store.create_session(
        settings=captured_settings, project_instruction_state=control
    )
    attachment = MessageAttachment(b"old-image", "image/png", "old.png", 0)
    store.append_message(
        captured.id,
        role=ConsoleMessageRole.USER,
        content="captured question",
        attachments=(attachment,),
    )
    store.create_session(
        settings=active_settings,
        project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
    )
    app_config = {
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9191/v1/chat/completions",
                "model": "captured-vision",
                "max_tokens": 70,
            }
        },
        "console": {},
    }

    class Gateway:
        def __init__(self):
            self.selections = []

        async def resolve_for_send(self, selection):
            self.selections.append(selection)
            return SimpleNamespace(
                ready=True,
                provider=selection.provider,
                execution_key=selection.provider,
                model=selection.explicit_model or selection.configured_model,
                max_tokens=selection.max_tokens,
                base_url=selection.base_url,
            )

    gateway = Gateway()
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(tmp_path / "fallback-preview.db", client_id="test"),
        store=store,
        provider_gateway=gateway,
    )
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="openai",
        model="active-text",
        max_tokens=10,
        agent_bridge=bridge,
        agent_runtime_enabled=True,
        provider_config=lambda: app_config,
    )
    selection = SimpleNamespace(
        binding=SimpleNamespace(binding_id="binding-1"),
        root=tmp_path,
        locator_fingerprint="f" * 64,
        allow_write=True,
        root_identity=controller_module._capture_project_root_identity(tmp_path),
    )
    controller.app = SimpleNamespace(
        workspace_registry_service=SimpleNamespace(
            get_runtime_binding=lambda _binding_id: selection.binding
        ),
        unified_mcp_service=None,
    )
    monkeypatch.setattr(
        bridge_module,
        "build_run_log_request_plan",
        lambda: RunLogRequestPlan(False, True, 1),
    )
    monkeypatch.setattr(
        controller_module,
        "resolve_project_instruction_binding",
        lambda _session, _registry: selection,
    )
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        lambda _resolver, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        controller_module,
        "project_instruction_authority_snapshot_is_current",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        controller_module,
        "is_vision_capable",
        lambda provider, model: (provider, model) == ("llama_cpp", "captured-vision"),
    )
    monkeypatch.setattr(controller_module, "max_history_images", lambda *_a: 2)
    monkeypatch.setattr(agent_service, "get_model_token_limit", lambda *_a, **_k: 100)
    monkeypatch.setattr(agent_service, "_count_model_messages", lambda *_a, **_k: 20)
    monkeypatch.setattr(agent_service, "count_tokens_messages", lambda *_a, **_k: 15)
    monkeypatch.setattr(agent_service, "estimate_tokens", lambda *_a, **_k: 0)

    snapshot = await controller.build_context_snapshot(
        "draft",
        attachments=(MessageAttachment(b"new-image", "image/png", "new.png", 0),),
        session_id=captured.id,
    )

    preview = snapshot.project_instruction_preview
    assert preview is not None
    captured_selection = gateway.selections[0]
    assert captured_selection.provider == "llama_cpp"
    assert captured_selection.configured_model == "captured-vision"
    assert captured_selection.base_url == "http://127.0.0.1:9191"
    assert captured_selection.max_tokens == 70
    assert (
        str(preview.next_send_payload).count("[image: data redacted for preview]") == 2
    )
    assert "[image omitted]" not in str(preview.next_send_payload)
    assert "omitted_token_budget" in preview.outcomes


@pytest.mark.asyncio
async def test_setup_callback_is_scoped_to_the_owning_session():
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: SimpleNamespace(
        sessions=lambda: []
    )
    push_screen_wait = AsyncMock(
        side_effect=AssertionError("mounted for closed session")
    )
    controller._screen = SimpleNamespace(
        app=SimpleNamespace(push_screen_wait=push_screen_wait)
    )

    result = await controller._select_project_instruction_binding(
        "closed-session", (), "binding_unavailable"
    )

    assert result == ("cancel", None)
    push_screen_wait.assert_not_called()


def test_notice_callback_fails_closed_when_main_loop_is_unavailable():
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: SimpleNamespace(
        sessions=lambda: [SimpleNamespace(id="session-a")]
    )
    controller.app_instance = SimpleNamespace(
        call_from_thread=Mock(side_effect=RuntimeError("loop closed"))
    )

    decision = controller._confirm_project_instruction_dispatch(
        SimpleNamespace(session_id="session-a")
    )

    assert decision == "cancel"


def test_notice_timeout_fails_closed_and_dismisses_owning_modal():
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: SimpleNamespace(
        sessions=lambda: [SimpleNamespace(id="session-a")]
    )
    mounted = []

    def push_screen(modal, callback):
        modal.dismiss = Mock()
        mounted.append((modal, callback))

    controller._screen = SimpleNamespace(app=SimpleNamespace(push_screen=push_screen))
    controller.app_instance = SimpleNamespace(
        call_from_thread=lambda callback: callback()
    )
    controller._project_instruction_notice_timeout_seconds = 0.0

    decision = controller._confirm_project_instruction_dispatch(
        SimpleNamespace(session_id="session-a")
    )

    assert decision == "cancel"
    assert len(mounted) == 1
    mounted[0][0].dismiss.assert_called_once_with("cancel")


def test_notice_observes_captured_stop_event_after_run_map_cleanup():
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: SimpleNamespace(
        sessions=lambda: [SimpleNamespace(id="session-a")]
    )
    mounted = threading.Event()
    modal_holder = []

    def push_screen(modal, callback):
        modal.dismiss = Mock(side_effect=lambda decision: callback(decision))
        modal_holder.append(modal)
        mounted.set()

    cancel_event = threading.Event()
    active_events = {"session-a": cancel_event}
    controller._screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=push_screen),
        _console_chat_controller=SimpleNamespace(_active_cancel_events=active_events),
    )
    controller.app_instance = SimpleNamespace(
        call_from_thread=lambda callback: callback()
    )
    controller._project_instruction_notice_timeout_seconds = 0.25
    result = []
    worker = threading.Thread(
        target=lambda: result.append(
            controller._confirm_project_instruction_dispatch(
                SimpleNamespace(session_id="session-a")
            )
        )
    )
    worker.start()
    assert mounted.wait(1)

    stopped_at = time.monotonic()
    cancel_event.set()
    active_events.pop("session-a")
    worker.join(1)

    assert not worker.is_alive()
    assert time.monotonic() - stopped_at < 0.15
    assert result == ["cancel"]
    modal_holder[0].dismiss.assert_called_once_with("cancel")


class _ModalHarness(App):
    def compose(self) -> ComposeResult:
        yield Static("background")


@pytest.mark.asyncio
async def test_status_row_sync_uses_short_status_first_copy():
    display = importlib.import_module("tldw_chatbook.Chat.console_display_state")
    ui = _ui_module()
    initial = display.build_console_project_instruction_state(
        ProjectInstructionControlState.legacy_disabled()
    )
    warning = display.build_console_project_instruction_state(
        ProjectInstructionControlState(True, "binding", "f" * 64, None),
        binding_label="Repo",
        locator_matches=False,
        warning_codes=("binding_retargeted",),
    )

    class RowHarness(App):
        def compose(self) -> ComposeResult:
            yield ui.ConsoleProjectInstructionStatusRow(initial)

    app = RowHarness()
    async with app.run_test(size=(30, 5)) as pilot:
        row = app.query_one(ui.ConsoleProjectInstructionStatusRow)
        row.sync_state(warning)
        await pilot.pause()
        button = row.query_one(Button)
        assert str(button.label) == "Warning · Project"
        assert button.region.width <= 30


@pytest.mark.asyncio
async def test_setup_modal_selects_only_eligible_binding_and_has_true_footer():
    ui = _ui_module()
    options = (
        ui.ProjectInstructionBindingOption("ready", "Ready repo", True),
        ui.ProjectInstructionBindingOption(
            "stale", "Stale [repo]", False, "Folder is unavailable"
        ),
    )
    app = _ModalHarness()
    result = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(ui.ProjectInstructionSetupModal(options), result.append)
        await pilot.pause()
        modal = app.screen
        assert modal.query_one(Footer)
        stale = modal.query_one("#console-project-binding-1", Button)
        assert stale.disabled
        rendered = " ".join(str(item.renderable) for item in modal.query(Static))
        assert "Stale [repo]" in str(stale.label)
        assert "Folder is unavailable" in rendered
        await pilot.click("#console-project-binding-1")
        await pilot.pause()
        assert app.screen is modal
        await pilot.click("#console-project-binding-0")
        await pilot.pause()
    assert result[0].action == "select"
    assert result[0].binding_id == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "action"),
    (("d", "disable"), ("c", "cancel")),
)
async def test_setup_modal_disable_and_cancel_bindings_are_exact(key, action):
    ui = _ui_module()
    app = _ModalHarness()
    results = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            ui.ProjectInstructionSetupModal(
                (ui.ProjectInstructionBindingOption("one", "Repo", True),)
            ),
            results.append,
        )
        await pilot.pause()
        await pilot.press(key)
        await pilot.pause()
    assert results == [ui.ProjectInstructionSetupResult(action)]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40)])
async def test_notice_modal_is_usable_at_supported_sizes(size):
    ui = _ui_module()
    notice = ProjectInstructionDispatchNotice(
        session_id="session-a",
        destination_label="OpenAI (https://api.example)",
        relative_source="AGENTS[1].md",
        scope=".",
        byte_count=42,
        outcomes=("omitted_token_budget",),
        warning_codes=("omitted_token_budget",),
    )
    app = _ModalHarness()
    decisions = []
    async with app.run_test(size=size) as pilot:
        app.push_screen(ui.ProjectInstructionNoticeModal(notice), decisions.append)
        await pilot.pause()
        modal = app.screen
        assert modal.query_one(Footer)
        assert modal.query_one("#console-project-notice-proceed", Button).region.width
        visible_copy = " ".join(
            str(widget.renderable) for widget in modal.query(Static)
        )
        assert "AGENTS[1].md" in visible_copy
        assert "[1]" in visible_copy
        assert "deeper AGENTS.md" in visible_copy
        assert "omitted_token_budget" in visible_copy
        await pilot.press("p")
        await pilot.pause()
    assert decisions == ["proceed"]


@pytest.mark.asyncio
async def test_notice_modal_cancel_and_disable_bindings_match_actions():
    ui = _ui_module()
    notice = ProjectInstructionDispatchNotice(
        session_id="session-a",
        destination_label="Provider",
        relative_source=None,
        scope=".",
        byte_count=0,
        outcomes=(),
        warning_codes=(),
    )
    for key, expected in (("c", "cancel"), ("d", "disable")):
        app = _ModalHarness()
        decisions = []
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(ui.ProjectInstructionNoticeModal(notice), decisions.append)
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
        assert decisions == [expected]
