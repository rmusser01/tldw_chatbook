"""Disposable Canvas budgeting through the production Inspector and dispatch."""

from __future__ import annotations

import asyncio
import copy
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from tldw_profile_core import PreferencePayload

import tldw_chatbook.Chat.console_agent_bridge as bridge_module
from Tests.console_provider_doubles import provider_resolution
from Tests.private_profile import private_profile_test
from tldw_chatbook.Agents.agent_models import ToolCatalogEntry
from tldw_chatbook.Agents.canvas_tool_provider import CanvasToolProvider
from tldw_chatbook.Canvas.models import CanvasScope
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_canvas_controller import ConsoleCanvasController
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Personal_Context.context_service import ProfileContextService
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.repository import PersonalContextRepository
from tldw_chatbook.Personal_Context.runtime_policy import AgentAuthority
from tldw_chatbook.Personal_Context.service import PersonalContextService


@pytest.fixture
def preview_rig(tmp_path, monkeypatch):
    """Real private profile, Canvas owner, controller, bridge and request assembly."""
    repository = PersonalContextRepository(
        tmp_path / "profile.db", key_protector=InMemoryProfileKeyProtector()
    )
    profile = PersonalContextService(repository)
    profile.create_profile()
    profile.set_runtime_enabled(True)
    scope = profile.list_scopes()[0]
    profile.set_scope_authority(scope.scope_id, AgentAuthority.READ_ONLY)
    record = profile.create_manual_record(
        scope_id=scope.scope_id,
        payload=PreferencePayload(
            subject="response.detail", polarity="like", value="concise " * 150
        ),
        semantic_key={"namespace": "preference", "subject": "response.detail"},
        controls={"sync_mode": "syncable", "agent_visibility": "agent_visible"},
    )
    builder = ProfileContextService(profile)
    requests = []
    original_build = builder.build_explained_snapshot

    def build(request):
        requests.append(request)
        return original_build(request)

    monkeypatch.setattr(builder, "build_explained_snapshot", build)
    canvas = ConsoleCanvasController()
    store = ConsoleChatStore(canvas_turn_controller=canvas)
    session = store.create_session(ephemeral=True)
    resolution = provider_resolution(
        provider="openai", execution_key="openai", model="gpt-4o-mini", max_tokens=2048
    )
    sent = []

    async def resolve(_selection):
        return resolution

    async def stream(_resolution, messages, **kwargs):
        sent.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        yield "done"

    gateway = SimpleNamespace(resolve_for_send=resolve, stream_chat=stream)
    db = AgentRunsDB(tmp_path / "runs.db")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=lambda: True,
    )
    preview_inputs = []
    original_preview = bridge.build_personal_context_preview_snapshot

    def preview(**kwargs):
        preview_inputs.append(kwargs)
        return original_preview(**kwargs)

    monkeypatch.setattr(bridge, "build_personal_context_preview_snapshot", preview)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        agent_bridge=bridge,
        canvas_enabled_reader=lambda: True,
    )

    async def raw_profile():
        return profile

    async def context_builder(_service=None):
        return builder

    async def no_external_providers(**_kwargs):
        return None, None, None, None

    monkeypatch.setattr(controller, "_personal_context_service", raw_profile)
    monkeypatch.setattr(controller, "_personal_context_builder", context_builder)
    monkeypatch.setattr(
        controller, "_compose_agent_request_providers", no_external_providers
    )
    try:
        yield SimpleNamespace(
            profile=profile,
            record=record,
            builder=builder,
            requests=requests,
            canvas=canvas,
            store=store,
            session=session,
            resolution=resolution,
            bridge=bridge,
            controller=controller,
            db=db,
            sent=sent,
            preview_inputs=preview_inputs,
        )
    finally:
        controller.begin_shutdown()
        bridge.close_all_progress()
        store.end_app_runtime()
        db.close()
        repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("native", "limit", "draft", "selected", "schema_case"),
    [
        (True, 128000, "question", True, "healthy"),
        (True, 16000, "question " * 8000, False, "healthy"),
        (False, 32000, "question " * 24000, False, "healthy"),
        (True, 6000, "question", False, "healthy"),
        (True, 128000, "question", True, "unreadable"),
        (True, 128000, "question", True, "canvas_denied"),
    ],
    ids=[
        "native",
        "native-tight",
        "fenced-tight",
        "discovery",
        "unreadable",
        "canvas-denied",
    ],
)
@private_profile_test
async def test_canvas_inspector_reserves_the_dispatched_request_budget(
    request,
    monkeypatch,
    native,
    limit,
    draft,
    selected,
    schema_case,
):
    """Omitting Canvas from the controller/bridge preview overselects memory."""
    rig = request.getfixturevalue("preview_rig")
    rig.bridge._native_tools_enabled = lambda: native
    if schema_case == "unreadable":

        class UnreadableProvider:
            def list_catalog(self):
                return [
                    ToolCatalogEntry("local:broken", "broken", "Unreadable", "local")
                ]

            def load_schema(self, _tool_id):
                raise OSError("schema unavailable")

            def invoke(self, *_args):
                raise AssertionError("Inspection must not execute tools")

        async def providers(**_kwargs):
            return None, None, UnreadableProvider(), None

        monkeypatch.setattr(
            rig.controller, "_compose_agent_request_providers", providers
        )
    if schema_case == "canvas_denied":
        original_context = rig.controller.resolve_turn_execution_context

        def context(session_id):
            return replace(
                original_context(session_id),
                persona_policy_rules=tuple(
                    {"rule_kind": "mcp_tool", "rule_name": name, "allowed": False}
                    for name in (
                        "canvas_create",
                        "canvas_update",
                        "canvas_list",
                        "canvas_read",
                        "canvas_guide",
                    )
                ),
            )

        def denied_guide(*_args):
            raise AssertionError("Denied Canvas schemas must not read guides")

        monkeypatch.setattr(rig.controller, "resolve_turn_execution_context", context)
        monkeypatch.setattr(
            "tldw_chatbook.Canvas.authoring.canvas_authoring_guide", denied_guide
        )
    for module in (
        "tldw_chatbook.Chat.console_agent_bridge",
        "tldw_chatbook.Agents.agent_service",
        "tldw_chatbook.Chat.console_history_budget",
    ):
        monkeypatch.setattr(f"{module}.get_model_token_limit", lambda *_args: limit)
    captured = []
    plans = []
    original_plan = bridge_module.build_console_first_request_plan

    def plan(**kwargs):
        result = original_plan(**kwargs)
        plans.append(result)
        return result

    monkeypatch.setattr(bridge_module, "build_console_first_request_plan", plan)

    def no_execution(*_args, **_kwargs):
        raise AssertionError("Disposable inspection must not acquire Canvas authority")

    before = copy.deepcopy(rig.store.messages_for_session(rig.session.id))
    with monkeypatch.context() as guard:
        guard.setattr(rig.canvas, "register_run", no_execution)
        guard.setattr(rig.canvas, "capture_selected_scope", no_execution)
        guard.setattr(CanvasToolProvider, "issue_registration_authority", no_execution)
        snapshot = await rig.controller.build_context_snapshot(
            draft=draft,
            session_id=rig.session.id,
            profile_selection_sink=lambda *args: captured.append(args),
        )
    assert rig.canvas._runs == rig.canvas._assistant_runs == {}
    assert rig.store.messages_for_session(rig.session.id) == before
    assert len(rig.preview_inputs) == len(rig.requests) == len(captured) == 1
    assert not plans[0].registry.is_canvas_reversible_conversation_local_mutation(
        "canvas_create"
    )
    assert not plans[0].registry.invoke_by_name("canvas_create", {}).ok
    preview_budget = rig.requests[0].available_input_tokens
    inputs = rig.preview_inputs[0]

    user = rig.store.append_message(
        rig.session.id, role=ConsoleMessageRole.USER, content=draft
    )
    assistant = rig.store.append_message(
        rig.session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    scope = CanvasScope(
        rig.session.id, rig.session.id, (user.id, assistant.id), None, None, "live-run"
    )
    run = rig.canvas.register_run(
        scope, assistant_message_id=assistant.id, temporary=True
    )
    provider = CanvasToolProvider(run, scope=scope, enabled_reader=lambda: True)
    live_requests = []
    live_builder = ProfileContextService(rig.profile)
    original_build = live_builder.build_snapshot

    def build_live(context_request):
        live_requests.append(context_request)
        return original_build(context_request)

    monkeypatch.setattr(live_builder, "build_snapshot", build_live)
    _run_id, outcome = await asyncio.to_thread(
        rig.bridge.run_reply,
        conversation_id=rig.session.id,
        session_id=rig.session.id,
        resolution=rig.resolution,
        assistant_message_id=assistant.id,
        model="gpt-4o-mini",
        session_system_prompt=inputs["session_system_prompt"],
        agent_messages=inputs["agent_messages"],
        should_cancel=lambda: False,
        canvas_provider=provider,
        canvas_authority=provider.issue_registration_authority(),
        profile_context_service=live_builder,
        scratch_root=inputs["scratch_root"],
        scratch_lease=inputs["scratch_lease"],
        local_provider=inputs["local_provider"],
        virtual_cli_provider=inputs["virtual_cli_provider"],
        raw_shell_provider=inputs["raw_shell_provider"],
        profile_provider=inputs["profile_provider"],
        persona_policy_rules=inputs.get("persona_policy_rules"),
        work_chain_id=rig.db.automatic_work.create_chain(
            rig.session.id, root_submission_id="canvas-preview-test"
        ),
    )
    assert outcome.status == "done", outcome
    assert len(rig.sent) == len(live_requests) == 1
    assert preview_budget == live_requests[0].available_input_tokens
    assert plans[0].schemas == plans[1].schemas
    if limit == 6000 or schema_case == "unreadable":
        assert plans[0].schemas.offer_find_load
        assert any(
            schema.name == "find_tools" and "canvas_create" in schema.description
            for schema in plans[0].schemas.runtime_schemas
        )
    elif schema_case != "canvas_denied":
        assert {schema.name for schema in plans[0].schemas.active_schemas} >= {
            "canvas_create",
            "canvas_update",
            "canvas_guide",
        }
    assert snapshot.personal_context_snapshot.source_version_ids == (
        (rig.record.version_id,) if selected else ()
    )
    if selected:
        assert (
            snapshot.personal_context_snapshot.serialized_block
            in rig.sent[0][0][0]["content"]
        )
    else:
        assert '"value":"concise ' not in rig.sent[0][0][0]["content"]
    assert rig.sent[0][0][-1]["content"] == inputs["agent_messages"][-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["close", "disable", "replacement", "durable_replacement"]
)
@private_profile_test
async def test_canvas_changes_during_project_preview_discard_snapshot(
    request, monkeypatch, change
):
    rig = request.getfixturevalue("preview_rig")
    if change == "durable_replacement":
        rig.session.ephemeral = False
    enabled = [True]
    rig.controller._canvas_enabled_reader = lambda: enabled[0]
    started, release = asyncio.Event(), asyncio.Event()

    async def held_project(*_args, **_kwargs):
        started.set()
        await release.wait()

    monkeypatch.setattr(
        rig.controller, "_build_project_instruction_preview_for_session", held_project
    )
    captured = []
    pending = asyncio.create_task(
        rig.controller.build_context_snapshot(
            draft="question",
            session_id=rig.session.id,
            profile_selection_sink=lambda *args: captured.append(args),
        )
    )
    await asyncio.wait_for(started.wait(), 10)
    if change == "close":
        rig.canvas.close_runtime()
    elif change == "disable":
        enabled[0] = False
    elif change == "replacement":
        rig.store.canvas_turn_controller = ConsoleCanvasController()
    else:
        rig.store.restore_state(
            sessions=[replace(rig.session)], active_session_id=rig.session.id
        )
    release.set()
    snapshot = await pending
    assert snapshot.personal_context_snapshot.serialized_block == ""
    assert captured == []
    assert '"value":"concise ' not in str(snapshot.next_send_payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "close",
        "disable",
        "enable",
        "replacement",
        "incarnation",
        "branch",
        "owner_available",
    ],
)
@private_profile_test
async def test_canvas_changes_discard_late_profile_preview(
    request, monkeypatch, change
):
    """A completed worker cannot publish a budget captured under stale Canvas state."""
    rig = request.getfixturevalue("preview_rig")
    enabled = [change != "enable"]
    rig.controller._canvas_enabled_reader = lambda: enabled[0]
    if change == "owner_available":
        rig.store.canvas_turn_controller = None
    started = threading.Event()
    release = threading.Event()
    original = rig.bridge.build_personal_context_preview_snapshot

    def held_preview(**kwargs):
        result = original(**kwargs)
        started.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(
        rig.bridge, "build_personal_context_preview_snapshot", held_preview
    )
    captured = []
    pending = asyncio.create_task(
        rig.controller.build_context_snapshot(
            draft="question",
            session_id=rig.session.id,
            profile_selection_sink=lambda *args: captured.append(args),
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        if change == "close":
            rig.canvas.close_runtime()
        elif change in {"disable", "enable"}:
            enabled[0] = not enabled[0]
        elif change == "replacement":
            rig.store.canvas_turn_controller = ConsoleCanvasController()
        elif change == "incarnation":
            rig.canvas.activate_session(rig.session.id)
        elif change == "owner_available":
            rig.store.canvas_turn_controller = rig.canvas
        else:
            rig.store.append_message(
                rig.session.id, role=ConsoleMessageRole.USER, content="later"
            )
    finally:
        release.set()
        snapshot = await pending
    assert snapshot.personal_context_snapshot.serialized_block == ""
    assert captured == []
    assert rig.canvas._runs == rig.canvas._assistant_runs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["disabled", "unavailable", "closed", "inactive", "reader_error"]
)
@private_profile_test
async def test_unavailable_canvas_keeps_preview_without_execution(request, state):
    rig = request.getfixturevalue("preview_rig")
    if state == "disabled":
        rig.controller._canvas_enabled_reader = lambda: False
    elif state == "unavailable":
        rig.store.canvas_turn_controller = None
    elif state == "closed":
        rig.canvas.close_runtime()
    elif state == "inactive":
        rig.canvas.discard_session(rig.session.id)
    else:

        def unavailable():
            raise RuntimeError("unavailable")

        rig.controller._canvas_enabled_reader = unavailable
    captured = []
    snapshot = await rig.controller.build_context_snapshot(
        draft="question",
        session_id=rig.session.id,
        profile_selection_sink=lambda *args: captured.append(args),
    )
    assert rig.canvas._runs == rig.canvas._assistant_runs == {}
    if state in {"disabled", "unavailable"}:
        assert snapshot.personal_context_snapshot.source_version_ids == (
            rig.record.version_id,
        )
        assert len(captured) == 1
        assert rig.preview_inputs[0]["canvas_profile_snapshot"] is None
    else:
        assert snapshot.personal_context_snapshot.serialized_block == ""
        assert captured == []


@pytest.mark.asyncio
@private_profile_test
async def test_diagnostic_sink_failure_preserves_profile_snapshot(request):
    """Inspection diagnostics cannot change the independently built model block."""
    rig = request.getfixturevalue("preview_rig")

    def broken_sink(*_args):
        raise RuntimeError("diagnostic unavailable")

    snapshot = await rig.controller.build_context_snapshot(
        draft="question",
        session_id=rig.session.id,
        profile_selection_sink=broken_sink,
    )
    assert snapshot.personal_context_snapshot.source_version_ids == (
        rig.record.version_id,
    )


@pytest.mark.asyncio
@private_profile_test
async def test_canvas_preview_does_not_load_persona_denied_schema(request, monkeypatch):
    """Persona-denied schemas stay unread even when Canvas contributes preview data."""
    rig = request.getfixturevalue("preview_rig")

    class DeniedProvider:
        def list_catalog(self):
            return [
                ToolCatalogEntry(
                    "local:blocked", "blocked", "Unavailable schema", "local"
                )
            ]

        def load_schema(self, _tool_id):
            raise AssertionError("A denied schema must not be fetched")

        def invoke(self, _tool_id, _args):
            raise AssertionError("Preview must not invoke tools")

    async def providers(**_kwargs):
        return None, None, DeniedProvider(), None

    original_context = rig.controller.resolve_turn_execution_context

    def context(session_id):
        return replace(
            original_context(session_id),
            persona_policy_rules=(
                {"rule_kind": "mcp_tool", "rule_name": "blocked", "allowed": False},
            ),
        )

    monkeypatch.setattr(rig.controller, "resolve_turn_execution_context", context)
    monkeypatch.setattr(rig.controller, "_compose_agent_request_providers", providers)
    snapshot = await rig.controller.build_context_snapshot(
        draft="question", session_id=rig.session.id
    )
    assert snapshot.personal_context_snapshot.source_version_ids == (
        rig.record.version_id,
    )
