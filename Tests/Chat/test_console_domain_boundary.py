"""Causal controls for the explicit in-process domain and presentation route."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Chat import console_runtime as runtime_source
from tldw_chatbook.Chat.console_configuration_preparation import (
    ConsoleTurnCaptureSelection,
    _source_references,
    capture_received_configuration_owned,
)
from tldw_chatbook.Chat.console_domain_services import (
    ConsoleDomainServices,
    ConsoleDomainSources,
)
from tldw_chatbook.Chat.console_presentation_port import ConsolePresentationPort
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository

pytestmark = pytest.mark.bootstrap_profile


class _Parent:
    """Presentation fixture whose domain-source slots are intentionally unusable."""

    def __init__(self):
        self.app_config = {"console": {"native_tool_calls": True}}
        self.notifications = []
        self.pending = []

    def __getattr__(self, name):
        if name in {
            "chachanotes_db",
            "workspace_registry_service",
            "skills_scope_service",
            "local_skills_service",
            "unified_mcp_service",
            "services",
            "local_character_persona_service",
            "prompt_scope_service",
        }:
            raise AssertionError(f"Domain work read parent App source: {name}")
        raise AttributeError(name)

    def notify(self, text, *, severity):
        self.notifications.append((text, severity))

    def call_from_thread(self, callback, *args):
        self.pending.append((callback, args))

    def flush(self):
        pending, self.pending = self.pending, []
        for callback, args in pending:
            callback(*args)


@pytest_asyncio.fixture
async def domain_case(tmp_path, request):
    parent = _Parent()
    media = None
    workspace_db = None
    full_agent_before = None
    if getattr(request, "param", None) == "control_mcp":
        from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config

        full_agent_before = get_cli_setting("tools", "full_agent_tools_enabled", False)
        assert save_setting_to_cli_config("tools", "full_agent_tools_enabled", True)
    if getattr(request, "param", None) in {
        "textual",
        "control",
        "control_saved",
        "control_settings",
        "control_trace",
        "control_mcp",
        "control_index",
    }:
        from textual.app import App

        parent = App()
        parent.app_config = {"console": {"native_tool_calls": True}}
    database = CharactersRAGDB(tmp_path / "domain.sqlite", client_id="domain-boundary")
    sources = ConsoleDomainSources(
        configuration={
            "console": {"native_tool_calls": False},
            "chat_defaults": {"user_display_name": "Domain user"},
        },
        chat_database=database,
        prompt_database=database,
        visual_repository=VisualIdentityRepository(database),
        scratch=ConsoleScratchSpaceManager(),
        prompt_history=SimpleNamespace(),
    )
    if getattr(request, "param", None) == "cold_trust":
        from tldw_chatbook import app_service_wiring as wiring
        from tldw_chatbook.Widgets import compact_model_bar as metadata
        from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
        from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService

        local = LocalSkillsService(
            store_dir=wiring.default_local_skills_store_dir(wiring.get_user_data_dir())
        )
        scope = SkillsScopeService(local_service=local)
        sources = replace(
            sources, local_skills=local, skills_scope=scope, skill_wiring_source=wiring
        )
        assert metadata is not None
    if getattr(request, "param", None) in {
        "control",
        "control_saved",
        "control_settings",
        "control_trace",
        "control_mcp",
        "control_index",
    }:
        from tldw_chatbook.MCP.permission_store import MCPPermissionStore
        from tldw_chatbook.MCP.unified_control_plane_service import (
            UnifiedMCPControlPlaneService,
        )
        from tldw_chatbook.config import get_user_data_dir

        from tldw_chatbook.MCP.local_store import LocalMCPStore
        from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
        from tldw_chatbook.runtime_policy.bootstrap import (
            RuntimePolicyContext,
            default_runtime_policy_path,
        )
        from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
        from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore

        policy_store = RuntimeSourceStateStore(default_runtime_policy_path())
        policy_context = RuntimePolicyContext(
            state=policy_store.load(), store=policy_store
        )
        policy_enforcer = ServicePolicyEnforcer.from_runtime_policy_context(
            policy_context
        )
        storage_provider = None
        if getattr(request, "param", None) == "control_mcp":
            from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
            from tldw_chatbook.app_services import ProfileStorage

            media = MediaDatabase(tmp_path / "media.sqlite", client_id="control-mcp")
            storage = ProfileStorage(chachanotes=database, media=media)
            storage_provider = lambda: storage
            sources = replace(sources, storage=storage)
            sources.configuration["console"]["native_tool_calls"] = True
        local_service = LocalMCPControlService(
            store=LocalMCPStore(get_user_data_dir() / "local_mcp_store.json"),
            policy_enforcer=policy_enforcer,
            storage_provider=storage_provider,
        )
        control_plane = UnifiedMCPControlPlaneService(
            target_store=None,
            context_store=None,
            local_service=local_service,
            server_service=None,
        )
        control_plane._permission_store = MCPPermissionStore(
            get_user_data_dir() / "mcp_permissions.json"
        )
        sources = replace(
            sources, control_plane=control_plane, policy_enforcer=policy_enforcer
        )
    if getattr(request, "param", None) == "control_index":
        from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
        from tldw_chatbook.Workspaces.registry_service import (
            LocalWorkspaceRegistryService,
        )

        workspace_db = WorkspaceDB(tmp_path / "workspaces.db")
        registry = LocalWorkspaceRegistryService(workspace_db)
        registry.create_workspace(workspace_id="index-control", name="Index control")
        project = tmp_path / "project"
        project.mkdir()
        (project / "example.txt").write_text(
            "original scanner evidence", encoding="utf-8"
        )
        binding = registry.add_folder_binding("index-control", project)
        sources = replace(sources, workspace_registry=registry)
    if getattr(request, "param", None) in {
        "control_saved",
        "control_trace",
        "control_settings",
    }:
        from tldw_chatbook.Chat.chat_conversation_scope_service import (
            ChatConversationScopeService,
        )
        from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

        sources = replace(
            sources,
            conversation_scope_service=ChatConversationScopeService(
                local_service=ChatConversationService(database),
                server_service=None,
            ),
        )
        # The original parent archive action resolves this same concrete owner.
        parent.chat_conversation_scope_service = sources.conversation_scope_service
    if getattr(request, "param", None) == "control_settings":
        from tldw_chatbook.Chat.console_settings_durability import (
            ConsoleSettingsDurabilityOwner,
        )

        sources = replace(
            sources, settings_durability_owner=ConsoleSettingsDurabilityOwner()
        )
    presentation = ConsolePresentationPort(parent)
    domain = ConsoleDomainServices(
        sources, runtime_source=runtime_source, presentation=presentation
    )
    runtime = runtime_source.ConsoleRuntime(parent, domain_services=domain)
    gateway = (
        runtime.ensure_provider_gateway()
        if getattr(request, "param", None) in {"control_trace", "control_mcp"}
        else ConsoleProviderGateway(config_provider=domain.provider_config)
    )
    store = runtime.ensure_chat_store()
    bridge = None
    if getattr(request, "param", None) == "control_mcp":
        bridge = runtime.ensure_agent_bridge(
            store_factory=runtime.ensure_chat_store,
            provider_gateway_factory=runtime.ensure_provider_gateway,
        )
        assert bridge is not None
    controller = runtime.ensure_chat_controller(
        store=store,
        provider_gateway=gateway,
        agent_runtime_enabled=bridge is not None,
        agent_bridge=bridge,
        provider="deepseek",
        model="deepseek-chat",
    )
    session = store.create_session(
        title="Domain source",
        workspace_id="index-control" if workspace_db is not None else "global",
    )
    case = SimpleNamespace(
        parent=parent,
        sources=sources,
        domain=domain,
        runtime=runtime,
        controller=controller,
        store=store,
        session=session,
        database=database,
        presentation=presentation,
    )
    if workspace_db is not None:
        case.index_binding = binding
        case.index_root = project
    try:
        if getattr(request, "param", None) in {
            "textual",
            "control",
            "control_saved",
            "control_settings",
            "control_trace",
            "control_mcp",
            "control_index",
        }:
            async with parent.run_test() as pilot:
                case.pilot = pilot
                yield case
        else:
            yield case
    finally:
        await runtime.dispose()
        await gateway.aclose()
        database.close()
        if media is not None:
            media.close()
        if workspace_db is not None:
            workspace_db.close()
        if full_agent_before is not None:
            assert save_setting_to_cli_config(
                "tools", "full_agent_tools_enabled", full_agent_before
            )


def _selection(case):
    return ConsoleTurnCaptureSelection(
        provider_selection=case.controller._provider_selection_for_session(
            case.session.id
        ),
        presentation_context=case.controller._presentation_context_for(case.session.id),
        rag_defaults=None,
        tool_configuration=None,
        skill_workspace_id=case.store.session_workspace_id(case.session.id),
        project_bindings_eligible=False,
        agent_runtime_enabled=case.controller._agent_runtime_enabled,
    )


@pytest.mark.asyncio
async def test_selected_capture_uses_supplied_domain_sources_without_app_discovery(
    domain_case,
):
    case = domain_case
    calls = []
    selected_code = ConsoleDomainServices.provider_config.__code__
    capture_code = runtime_source._provider_config_for_domain.__code__

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is selected_code:
            calls.append(("domain", frame.f_locals["self"]))
        if event == "call" and frame.f_code is capture_code:
            calls.append(("configuration", frame.f_locals["config"]))

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        captured = await capture_received_configuration_owned(
            case.controller, case.session.id, selection=_selection(case)
        )
    finally:
        sys.setprofile(previous)
    refs = _source_references(None, case.store, case.controller, case.session)
    assert refs.chat_database is case.database
    assert refs.prompt_database is case.database
    assert refs.app_config is case.sources.configuration
    assert case.controller.domain_services is case.domain
    assert case.runtime._execution_service.domain_services is case.domain
    assert ("domain", case.domain) in calls
    assert ("configuration", case.sources.configuration) in calls
    assert captured.session_id == case.session.id
    assert captured.tool_configuration["native_tool_calls_enabled"] is False
    assert case.controller._global_user_display_name() == "Domain user"
    assert case.parent.app_config["console"]["native_tool_calls"] is True
    assert not case.controller._preparation_reads
    assert not case.runtime._preparation_reads


@pytest.mark.asyncio
async def test_selected_source_replacement_after_original_worker_wait_refuses(
    domain_case,
):
    case = domain_case
    entered, release = threading.Event(), threading.Event()
    code = ConsoleScratchSpaceManager.snapshot.__code__
    worker = []

    def observe(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is code
            and frame.f_locals.get("self") is case.sources.scratch
        ):
            worker.append(threading.current_thread())
            entered.set()
            assert release.wait(8), "original snapshot was not released"

    previous, thread_profile = sys.getprofile(), threading.getprofile()
    task = None
    threading.setprofile_all_threads(observe)
    try:
        task = asyncio.create_task(
            capture_received_configuration_owned(
                case.controller, case.session.id, selection=_selection(case)
            )
        )
        for _ in range(400):
            if entered.is_set() or task.done():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set(), "selected original snapshot did not run"
        selected_reads = tuple(
            read
            for read in case.controller._preparation_reads
            if read.session_id == case.session.id
        )
        assert selected_reads
        assert all(
            read in case.runtime._preparation_reads
            and not read.callback_returned
            and not read.retired.done()
            for read in selected_reads
        )
        case.domain.sources = replace(
            case.sources, configuration={"console": {"native_tool_calls": True}}
        )
        release.set()
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            await task
        assert worker and all(item is not threading.current_thread() for item in worker)
        assert all(
            not read.callback_returned
            and read.retired.done()
            and read._producer.done()
            and isinstance(read._producer.exception(), RecoveryRequired)
            and read not in case.controller._preparation_reads
            and read not in case.runtime._preparation_reads
            for read in selected_reads
        )
        assert not case.controller._preparation_reads
        assert all(
            read.session_id != case.session.id
            for read in case.runtime._preparation_reads
        )
        from tldw_chatbook.Chat.console_preparation_reads import drain_preparation_reads

        await drain_preparation_reads(case.runtime._preparation_reads)
        assert not case.runtime._preparation_reads
    finally:
        release.set()
        if task is not None:
            with contextlib.suppress(BaseException):
                await task
        threading.setprofile_all_threads(thread_profile)
        sys.setprofile(previous)
        case.domain.sources = case.sources


def test_selected_original_helper_body_change_is_refused(domain_case, monkeypatch):
    case = domain_case
    original = runtime_source._provider_config_for_domain

    def replaced(config):
        return {}

    monkeypatch.setattr(original, "__code__", replaced.__code__)
    with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
        case.domain.provider_config()


@pytest.mark.asyncio
async def test_selected_received_sources_refuse_graph_and_parent_replacement(
    domain_case,
):
    from tldw_chatbook.Chat.console_received_dispatch import (
        DomainReceivedPreparationSource,
        received_preparation_source,
        require_received_source,
    )

    case = domain_case
    source = received_preparation_source(case.runtime)
    assert type(source) is DomainReceivedPreparationSource
    assert source.domain_services is case.domain
    assert source.configuration is case.sources.configuration
    record = SimpleNamespace(session_id=case.session.id, received_claim=None)
    require_received_source(case.runtime, record, source, inputs=False)
    case.domain.sources = replace(case.sources, prompt_scope=object())
    try:
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            require_received_source(case.runtime, record, source, inputs=False)
    finally:
        case.domain.sources = case.sources
    configuration = case.parent.app_config
    case.parent.app_config = dict(configuration)
    try:
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            require_received_source(case.runtime, record, source, inputs=False)
    finally:
        case.parent.app_config = configuration
    require_received_source(case.runtime, record, source, inputs=False)


@pytest.mark.asyncio
async def test_selected_interrupt_projection_rejects_stale_same_view_visit(domain_case):
    from Tests.Chat.test_console_runtime_lifetime import _View

    case = domain_case
    cards = []
    view = _View({"set_pending_approval": cards.append})
    first = case.runtime.attach_view(view)
    host = case.controller._interrupt_host
    assert host.presentation is case.presentation
    host._marshal_pending_approval({"round_id": "first"}, fire_summary=False)
    second = case.runtime.attach_view(view)
    assert second != first
    case.parent.flush()
    assert cards == [], "a queued projection from the previous visit was adopted"
    host._marshal_pending_approval({"round_id": "second"}, fire_summary=False)
    case.parent.flush()
    assert cards == [{"round_id": "second"}]
    case.presentation.clear_decision("approval")
    replacement = _View({"set_pending_approval": cards.append})
    case.runtime.attach_view(replacement)
    case.parent.flush()
    assert cards == [{"round_id": "second"}], "a stale clear erased a successor card"


@pytest.mark.asyncio
async def test_selected_remount_uses_current_session_and_original_fifo(domain_case):
    from Tests.Chat.test_console_runtime_lifetime import _View

    case = domain_case
    cards = []
    view = _View({"set_pending_question": cards.append})
    case.runtime.attach_view(view)
    host = case.controller._interrupt_host
    background = case.store.create_session(
        title="Background", workspace_id="global", activate=False
    )
    host.park_round_payload(
        "question", "one", {"round_id": "one", "session_id": case.session.id}
    )
    host.park_round_payload(
        "question", "two", {"round_id": "two", "session_id": case.session.id}
    )
    host.remount_head("question", background.id)
    case.parent.flush()
    assert cards == []
    host.remount_head("question", case.session.id)
    case.parent.flush()
    assert cards[-1]["round_id"] == "one"
    host.unpark_round_payload("question", "one")
    host.remount_head("question", case.session.id)
    case.parent.flush()
    assert cards[-1]["round_id"] == "two"
    assert case.controller._pending_question_lock is host.lock
    assert case.controller._parked_question_payloads is host.payloads["question"]
    host.unpark_round_payload("question", "two")


def test_selected_notice_failure_is_advisory(domain_case, monkeypatch):
    case = domain_case

    def fail(*args, **kwargs):
        raise RuntimeError("unavailable presentation")

    monkeypatch.setattr(case.parent, "notify", fail)
    assert case.presentation.notice("fixed notice") is False
    assert case.domain.sources is case.sources
    assert case.runtime._chat_store is case.store


@pytest.mark.asyncio
async def test_selected_gateway_uses_explicit_configuration_and_database(domain_case):
    case = domain_case
    gateway = case.runtime.ensure_provider_gateway()
    assert gateway is case.runtime.ensure_provider_gateway()
    assert gateway._config_provider.__self__ is case.domain
    assert gateway._config_provider() == case.domain.provider_config()


def test_selected_live_control_plane_reads_actual_permission_owner(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Agents.plain_chat_tool_surface import ControlPlaneToolKillSwitch
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    service = UnifiedMCPControlPlaneService(
        target_store=None, context_store=None, local_service=None, server_service=None
    )
    permission = MCPPermissionStore(tmp_path / "permissions.json")
    service._permission_store = permission
    switch = ControlPlaneToolKillSwitch(service)
    assert switch() is False
    permission.set_kill_switch(True)
    assert switch() is True
    permission.set_kill_switch(False)
    assert switch() is False

    def failed():
        raise RuntimeError("read failure")

    monkeypatch.setattr(permission, "get_kill_switch", failed)
    assert switch() is True
    monkeypatch.setattr(service, "get_kill_switch", lambda: False)
    assert switch() is True
    assert ControlPlaneToolKillSwitch(None)() is False


@pytest.fixture
def plugin_case(tmp_path):
    from tldw_chatbook import config
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    database = WorkspaceDB(tmp_path / "workspace.sqlite", "domain-plugin")
    registry = LocalWorkspaceRegistryService(database)
    local = LocalSkillsService(store_dir=tmp_path / "local")
    sources = ConsoleDomainSources(
        configuration={},
        local_skills=local,
        workspace_registry=registry,
        data_root_selector=config.get_user_data_dir,
    )
    domain = ConsoleDomainServices(
        sources,
        runtime_source=runtime_source,
        presentation=ConsolePresentationPort(_Parent()),
    )
    try:
        yield SimpleNamespace(
            domain=domain,
            sources=sources,
            registry=registry,
            local=local,
            config=config,
        )
    finally:
        database.close()


def test_selected_plugin_stays_lazy_and_preserves_resident_owner(plugin_case):
    from tldw_chatbook.Chat.console_domain_services import (
        _ready_domain_published_plugin,
    )
    from tldw_chatbook.Plugins.service import PluginService

    case = plugin_case
    calls = []
    code = case.sources.data_root_selector.__code__

    def observe(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is code
            and frame.f_back.f_code
            is ConsoleDomainServices.build_plugin_service.__code__
        ):
            calls.append(True)

    prior = sys.getprofile()
    sys.setprofile(observe)
    try:
        assert _ready_domain_published_plugin(case.local, case.domain) is None
        assert calls == []
        facade = case.local.plugin_service
    finally:
        sys.setprofile(prior)
    assert type(facade) is PluginService and len(calls) == 1
    assert facade is case.domain.published_plugin_service()
    assert facade._thread is None
    assert facade.workspace_lookup.__self__ is case.registry
    assert facade.workspace_lookup.__func__ is type(case.registry).get_workspace
    assert _ready_domain_published_plugin(case.local, case.domain) is facade
    facade.workspace_lookup = lambda identity: None
    with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
        case.domain.build_plugin_service()
    with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
        _ready_domain_published_plugin(case.local, case.domain)


@pytest.mark.asyncio
@pytest.mark.parametrize("domain_case", ["cold_trust"], indirect=True)
async def test_selected_cold_trust_publishes_original_domain_winner(domain_case):
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_console_received_configuration_preparation,
        capture_console_skill_catalog_owned,
        finish_console_skill_catalog_owned,
        require_received_configuration_preparation,
    )
    from tldw_chatbook.Chat.console_domain_skill_trust import (
        DomainSkillTrustSource,
        build_domain_skill_trust,
    )

    case = domain_case
    local = case.sources.local_skills
    local.skills_dir.mkdir(parents=True, exist_ok=True)
    skill = local.skills_dir / "domain-witness"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: domain-witness\ndescription: Domain witness\n---\nRead a supplied source.\n",
        encoding="utf-8",
    )
    local._save_index(
        {"domain-witness": {"name": "domain-witness", "description": "Domain witness"}}
    )
    preparation = capture_console_received_configuration_preparation(
        None, case.store, case.controller, session_id=case.session.id
    )
    assert preparation is not None
    assert type(preparation.trust_source) is DomainSkillTrustSource
    assert preparation.trust_source.domain_services is case.domain
    assert case.domain._trust is local._trust_service is None

    def current():
        require_received_configuration_preparation(
            preparation, None, case.store, case.controller, session_id=case.session.id
        )

    reads, observers = (
        case.controller._preparation_reads,
        (case.runtime._preparation_reads,),
    )
    catalog = await capture_console_skill_catalog_owned(
        None,
        case.store,
        case.controller,
        session_id=case.session.id,
        turn_id="domain-turn",
        skill_workspace_id="global",
        preparation=preparation,
        reads=reads,
        observers=observers,
        require_current=current,
    )
    assert catalog.records is not None and catalog.maximum is None
    observed = []
    code = build_domain_skill_trust.__code__

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is code:
            observed.append(
                (frame.f_locals["_source_wiring"], threading.current_thread())
            )

    previous, thread_profile = sys.getprofile(), threading.getprofile()
    threading.setprofile_all_threads(observe)
    try:
        prepared = await finish_console_skill_catalog_owned(
            catalog, reads=reads, observers=observers, require_current=current
        )
    finally:
        threading.setprofile_all_threads(thread_profile)
        sys.setprofile(previous)
    assert observed and all(
        wiring is case.sources.skill_wiring_source
        and thread is not threading.current_thread()
        for wiring, thread in observed
    )
    assert case.domain._trust is not None and local._trust_service is case.domain._trust
    publication = vars(local).get("_console_native_skill_trust_publication")
    assert publication is not None and publication.source.domain_services is case.domain
    assert publication.local_winner is case.domain._trust
    assert prepared.catalog is catalog
    current()


def test_selected_runtime_source_replacement_cannot_supply_its_own_checker(domain_case):
    case = domain_case
    case.domain.runtime_source = SimpleNamespace(
        _domain_callback_definitions_current=lambda: True,
        _provider_config_for_domain=lambda config: {"foreign": True},
    )
    try:
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            case.domain.provider_config()
    finally:
        case.domain.runtime_source = runtime_source


@pytest.mark.asyncio
async def test_selected_presentation_refuses_replaced_parent_and_view_callbacks(
    domain_case, monkeypatch
):
    from Tests.Chat.test_console_runtime_lifetime import _View

    case = domain_case
    cards, foreign = [], []
    case.runtime.attach_view(_View({"set_pending_approval": cards.append}))
    case.presentation.present_decision("approval", {"round_id": "old-slot"})
    with monkeypatch.context() as patch:
        patch.setattr(case.controller, "set_pending_approval", foreign.append)
        case.parent.flush()
    assert cards == foreign == []
    case.presentation.present_decision("approval", {"round_id": "old-parent-callback"})
    with monkeypatch.context() as patch:
        patch.setattr(
            type(case.parent), "call_from_thread", lambda *args: foreign.append(args)
        )
        case.parent.flush()
    assert cards == foreign == []
    case.presentation.present_decision("approval", {"round_id": "current"})
    case.parent.flush()
    assert cards == [{"round_id": "current"}]


def test_selected_plugin_resident_replacement_cannot_hide_behind_local_facade(
    domain_case, tmp_path
):
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Chat.console_configuration_preparation import (
        _same_configuration_references,
    )

    case = domain_case
    local_facade = PluginService(
        tmp_path / "local-facade", workspace_lookup=lambda identity: None
    )
    local = LocalSkillsService(
        store_dir=tmp_path / "local", plugin_service=local_facade
    )
    sources = replace(case.sources, local_skills=local)
    domain = ConsoleDomainServices(
        sources, runtime_source=runtime_source, presentation=case.presentation
    )
    creator = SimpleNamespace(
        domain_services=domain,
        _visual_identity_repository=case.sources.visual_repository,
        _scratch_spaces=case.sources.scratch,
        _provider_config=domain.provider_config,
    )
    before = domain.configuration_sources(case.store, creator, case.session)
    assert before.plugin is local_facade
    domain._plugin = PluginService(
        tmp_path / "new-resident", workspace_lookup=lambda identity: None
    )
    after = domain.configuration_sources(case.store, creator, case.session)
    assert after.plugin is local_facade
    assert not _same_configuration_references(before, after)


@pytest.mark.asyncio
@pytest.mark.parametrize("domain_case", ["textual"], indirect=True)
async def test_selected_real_textual_remount_and_worker_preserve_visit(domain_case):
    from textual.app import App
    from Tests.Chat.test_console_runtime_lifetime import _View

    case = domain_case
    cards = []
    view = _View({"set_pending_question": cards.append})
    generation = case.runtime.attach_view(view)
    host = case.controller._interrupt_host
    host.park_round_payload(
        "question",
        "real-round",
        {"round_id": "real-round", "session_id": case.session.id},
    )
    with pytest.raises(RuntimeError, match="different thread"):
        App.call_from_thread(case.parent, lambda: None)
    case.runtime.finish_view_reconciliation(view, generation)
    cards.clear()
    host._remount_session_kinds(case.session.id)
    assert cards[-1]["round_id"] == "real-round"
    cards.clear()
    await asyncio.to_thread(host.remount_head, "question", case.session.id)
    assert cards[-1]["round_id"] == "real-round"
    cards.clear()
    entered, release = threading.Event(), threading.Event()
    original = App.call_from_thread.__code__

    def observe(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is original
            and frame.f_locals.get("self") is case.parent
        ):
            entered.set()
            assert release.wait(8), "original Textual worker marshal was not released"

    previous, thread_profile = sys.getprofile(), threading.getprofile()
    task = None
    threading.setprofile_all_threads(observe)
    try:
        task = asyncio.create_task(
            asyncio.to_thread(host.remount_head, "question", case.session.id)
        )
        for _ in range(400):
            if entered.is_set() or task.done():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        case.runtime.attach_view(view)
        release.set()
        await task
        assert cards == []
    finally:
        release.set()
        if task is not None:
            await task
        threading.setprofile_all_threads(thread_profile)
        sys.setprofile(previous)
        host.unpark_round_payload("question", "real-round")


@pytest.mark.parametrize("changed", ["getter_body", "property_body", "permission_body"])
def test_selected_live_switch_refuses_defining_body_change(
    tmp_path, monkeypatch, changed
):
    from tldw_chatbook.Agents.plain_chat_tool_surface import ControlPlaneToolKillSwitch
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    service = UnifiedMCPControlPlaneService(
        target_store=None, context_store=None, local_service=None, server_service=None
    )
    permission = MCPPermissionStore(tmp_path / "switch.json")
    service._permission_store = permission
    permission.set_kill_switch(True)
    switch = ControlPlaneToolKillSwitch(service)
    assert switch() is True

    def false_body(self):
        return False

    def absent_property(self):
        return None

    if changed == "getter_body":
        target, replacement = UnifiedMCPControlPlaneService.get_kill_switch, false_body
    elif changed == "property_body":
        target, replacement = (
            UnifiedMCPControlPlaneService.permission_store.fget,
            absent_property,
        )
    else:
        target, replacement = MCPPermissionStore.get_kill_switch.__wrapped__, false_body
    with monkeypatch.context() as patch:
        patch.setattr(target, "__code__", replacement.__code__)
        assert switch() is True
    assert permission.get_kill_switch() is True and switch() is True
    permission.set_kill_switch(False)
    assert switch() is False


@pytest.mark.parametrize(
    "changed", ["registry_binding", "registry_body", "selector_body"]
)
def test_selected_plugin_refuses_changed_original_constructor_callbacks(
    plugin_case, monkeypatch, changed
):
    case = plugin_case

    def foreign(self):
        return None

    def foreign_selector():
        return None

    if changed == "registry_binding":
        monkeypatch.setattr(type(case.registry), "get_workspace", foreign)
    elif changed == "registry_body":
        monkeypatch.setattr(
            type(case.registry).get_workspace, "__code__", foreign.__code__
        )
    else:
        monkeypatch.setattr(
            inspect.unwrap(case.sources.data_root_selector),
            "__code__",
            foreign_selector.__code__,
        )
    with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
        case.domain.build_plugin_service()
    assert case.domain._plugin is case.local._plugin_service is None


@pytest.mark.asyncio
async def test_selected_plugin_refuses_source_change_during_original_selector(
    plugin_case,
):
    case = plugin_case
    entered, release = threading.Event(), threading.Event()
    code = case.sources.data_root_selector.__code__

    def observe(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is code
            and frame.f_back.f_code
            is ConsoleDomainServices.build_plugin_service.__code__
        ):
            entered.set()
            assert release.wait(8), "original data-root selector was not released"

    previous, thread_profile = sys.getprofile(), threading.getprofile()
    task = None
    threading.setprofile_all_threads(observe)
    try:
        task = asyncio.create_task(asyncio.to_thread(case.domain.build_plugin_service))
        for _ in range(400):
            if entered.is_set() or task.done():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        case.domain.sources = replace(case.sources, plugin_mcp_mapping=object())
        release.set()
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            await task
        assert case.domain._plugin is case.local._plugin_service is None
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        threading.setprofile_all_threads(thread_profile)
        sys.setprofile(previous)
        case.domain.sources = case.sources
