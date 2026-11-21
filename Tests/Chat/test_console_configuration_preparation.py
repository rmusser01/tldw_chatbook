"""Selected-value, affinity and optional-domain controls for owned capture."""

from __future__ import annotations

import threading
from dataclasses import replace
from functools import partial
from types import MethodType, SimpleNamespace

import pytest

from Tests.Chat import test_console_configuration_capture as controls

catalog_controls = controls.catalog_controls
catalog_store = controls.catalog_store
local_root = controls.local_root
mcp_sources = controls.mcp_sources
snapshot_case = controls.snapshot_case
runtime_case = controls.runtime_case
pytestmark = pytest.mark.bootstrap_profile


def _selection(case, **changes):
    from tldw_chatbook.Chat.console_configuration_preparation import (
        ConsoleTurnCaptureSelection,
    )

    selected = ConsoleTurnCaptureSelection(
        provider_selection=case.controller._provider_selection_for_session(
            case.session.id
        ),
        presentation_context=case.controller._presentation_context_for(case.session.id),
        rag_defaults=None,
        tool_configuration=None,
        skill_workspace_id=case.session.workspace_id,
        project_bindings_eligible=False,
        agent_runtime_enabled=False,
    )
    return replace(selected, **changes)


def test_capture_selection_detaches_nested_selected_values():
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_configuration_preparation import (
        ConsoleTurnCaptureSelection,
    )

    rag = {"source_types": ["notes"], "top_k": 4}
    tools = {"local_tools_enabled": False, "nested": {"limit": 2}}
    selected = ConsoleTurnCaptureSelection(
        ConsoleProviderSelection(provider="deepseek", explicit_model="deepseek-chat"),
        None,
        rag,
        tools,
        None,
        False,
        False,
    )
    rag["source_types"].append("media")
    tools["nested"]["limit"] = 7
    assert selected.rag_defaults["source_types"] == ("notes",)
    assert selected.tool_configuration["nested"]["limit"] == 2
    with pytest.raises(TypeError):
        selected.tool_configuration["local_tools_enabled"] = True


@pytest.mark.parametrize(
    "kind",
    ["cold", "custom_mcp", "custom_plugin", "custom_consent", "background_persona"],
)
def test_ineligible_source_is_targeted_and_does_not_construct_cold_owner(
    runtime_case,
    monkeypatch,
    kind,
    tmp_path,
):
    from tldw_chatbook.Chat.console_configuration_preparation import (
        standard_console_configuration_sources,
    )
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    case = runtime_case
    app = case.app
    target = case.session
    calls = []
    if kind == "cold":

        class ColdApp:
            _skills_scope_service = None

            @property
            def skills_scope_service(self):
                calls.append("constructed")
                raise AssertionError("eligibility invoked a cold getter")

        app = ColdApp()
    elif kind == "custom_mcp":
        app = SimpleNamespace(unified_mcp_service=SimpleNamespace())
    elif kind == "custom_plugin":
        local = LocalSkillsService(
            store_dir=tmp_path / "skills",
            plugin_service=SimpleNamespace(
                capture_maximum=lambda *_: calls.append("plugin")
            ),
        )
        app = SimpleNamespace(local_skills_service=local)
    elif kind == "custom_consent":
        app = SimpleNamespace(
            change_review_consent_service=SimpleNamespace(
                admit_turn=lambda *_: calls.append("consent")
            )
        )
    else:
        # Keep an earlier non-persona slot. Eligibility must select THIS
        # background target rather than the first/foreground session.
        target = case.store.create_session(
            title="Background persona",
            assistant_kind="persona",
            assistant_id="persona-one",
            ephemeral=True,
            activate=False,
        )
        app = SimpleNamespace(
            local_character_persona_service=SimpleNamespace(
                get_persona_profile=lambda *_: calls.append("persona")
            )
        )
    assert (
        standard_console_configuration_sources(
            app, case.store, case.controller, session_id=target.id
        )
        is False
    )
    assert calls == []


@pytest.mark.asyncio
async def test_ready_local_skill_capture_runs_original_body_on_worker_and_matches_snapshot(
    runtime_case,
    tmp_path,
    monkeypatch,
):
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_console_turn_configuration_owned,
        standard_console_configuration_sources,
    )
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    case = runtime_case
    # Deliberately use the service's documented trust-service-less test mode;
    # this exercises real stored skill metadata without constructing keyring.
    local = LocalSkillsService(
        store_dir=tmp_path / "ready-skills",
        allow_untrusted_without_trust_service=True,
    )
    imported = await local.import_skill(
        name="capture-worker-fixture",
        content="---\nname: capture-worker-fixture\ndescription: Capture parity fixture\n---\nFixture instructions.\n",
    )
    assert imported["name"] == "capture-worker-fixture"
    monkeypatch.setattr(case.app, "local_skills_service", local, raising=False)
    selected = _selection(case)
    expected = case.controller.resolve_runtime_turn_configuration_snapshot(
        case.session.id
    )
    observed = []
    original = LocalSkillsService._visible_records

    def record_thread(self, *args, **kwargs):
        observed.append(threading.get_ident())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LocalSkillsService, "_visible_records", record_thread)
    assert standard_console_configuration_sources(
        case.app, case.store, case.controller, session_id=case.session.id
    )
    loop_thread = threading.get_ident()
    current_checks = []

    def loop_current():
        current_checks.append(threading.get_ident())
        assert threading.get_ident() == loop_thread

    actual = await capture_console_turn_configuration_owned(
        case.app,
        case.store,
        case.session.id,
        selection=selected,
        creator=case.controller,
        reads=case.controller._preparation_reads,
        observers=(case.runtime._preparation_reads,),
        require_current=loop_current,
    )
    # Each independent skill capture intentionally creates a fresh attempt nonce.
    expected_skill = dict(expected.skill_context_maximum)
    actual_skill = dict(actual.skill_context_maximum)
    expected_skill.pop("plugin_run_id", None)
    actual_skill.pop("plugin_run_id", None)
    assert actual_skill == expected_skill
    assert actual_skill["available_skills"]
    assert "capture-worker-fixture" in {
        row["name"] for row in actual_skill["available_skills"]
    }
    # Under a managed runtime, legacy synchronous capture reads only closed MCP
    # metadata (11d71e5db5). runtime_case's loop projection installs a
    # policy_enforcer and a custom manifest_provider, so the metadata is open
    # and the legacy capture defers MCP with an empty ceiling; the owned one
    # must match. The closed case is the next test.
    from tldw_chatbook.MCP.console_snapshot import _mcp_metadata_is_closed

    assert not _mcp_metadata_is_closed(case.service)
    assert expected.mcp_capture_pending and not expected.mcp_definition_maximum
    assert (
        actual.mcp_definition_capture == expected.mcp_definition_capture == "captured"
    )
    for name in actual.__dataclass_fields__:
        if name != "skill_context_maximum":
            assert getattr(actual, name) == getattr(expected, name), name
    assert observed and all(thread != loop_thread for thread in observed)
    assert current_checks
    assert (
        case.controller._preparation_reads == case.runtime._preparation_reads == set()
    )


@pytest.mark.asyncio
async def test_closed_mcp_metadata_gives_owned_and_legacy_capture_one_ceiling(
    runtime_case,
):
    import tldw_chatbook.MCP.server  # noqa: F401 - load the original closed manifest source
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_console_turn_configuration_owned,
    )
    from tldw_chatbook.MCP.console_snapshot import _mcp_metadata_is_closed
    from tldw_chatbook.MCP.local_control_service import _default_manifest_provider

    case = runtime_case
    case.local.policy_enforcer = None
    case.local.manifest_provider = _default_manifest_provider
    assert _mcp_metadata_is_closed(case.service)
    expected = case.controller.resolve_runtime_turn_configuration_snapshot(
        case.session.id
    )
    actual = await capture_console_turn_configuration_owned(
        case.app,
        case.store,
        case.session.id,
        selection=_selection(case),
        creator=case.controller,
        reads=case.controller._preparation_reads,
        observers=(case.runtime._preparation_reads,),
        require_current=lambda: None,
    )
    assert not expected.mcp_capture_pending and expected.mcp_definition_maximum
    # Only the owned capture can prove its definition set complete; the
    # synchronous one always reports incomplete. Each capture also mints its
    # own skill attempt nonce. Every other field must agree.
    assert actual.mcp_definition_complete and not expected.mcp_definition_complete
    skills = [dict(c.skill_context_maximum) for c in (actual, expected)]
    for skill in skills:
        skill.pop("plugin_run_id", None)
    assert skills[0] == skills[1]
    differing = {
        name
        for name in actual.__dataclass_fields__
        if getattr(actual, name) != getattr(expected, name)
    }
    assert differing <= {"mcp_definition_complete", "skill_context_maximum"}, differing


@pytest.mark.asyncio
async def test_runtime_absent_provider_config_retains_original_default_and_one_budget_read(
    runtime_case,
    monkeypatch,
):
    from tldw_chatbook.Chat import console_agent_bridge
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_console_turn_configuration_owned,
    )

    case = runtime_case
    monkeypatch.setattr(case.controller, "_provider_config", None)
    case.app.app_config = {"console": {"native_tool_calls": False}}
    selected = _selection(case)
    observed = []
    original = console_agent_bridge.console_run_budget

    def budget():
        observed.append(threading.get_ident())
        return original()

    monkeypatch.setattr(console_agent_bridge, "console_run_budget", budget)
    result = await capture_console_turn_configuration_owned(
        case.app,
        case.store,
        case.session.id,
        selection=selected,
        creator=case.controller,
        reads=case.controller._preparation_reads,
        require_current=lambda: None,
    )
    assert result.tool_configuration["native_tool_calls_enabled"] is True
    assert len(observed) == 1
    assert observed[0] != threading.get_ident()


def test_stock_builtin_loader_is_exact_partial_and_reads_current_config(
    runtime_case,
    tmp_path,
    monkeypatch,
):
    from functools import partial

    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook.Chat.console_configuration_preparation import (
        standard_console_configuration_sources,
    )
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    case = runtime_case
    loader = partial(wiring._read_app_disabled_builtin_skills, case.app)
    local = LocalSkillsService(
        store_dir=tmp_path / "stock-loader", builtin_disabled_loader=loader
    )
    monkeypatch.setattr(case.app, "local_skills_service", local, raising=False)
    # The app registered this loader when it built its skills stack.
    monkeypatch.setattr(
        case.app, "_local_skills_callbacks", (None, None, loader), raising=False
    )
    assert standard_console_configuration_sources(
        case.app,
        case.store,
        case.controller,
        session_id=case.session.id,
    )
    case.app.app_config = {"skills": {"disabled_builtins": ["web-research"]}}
    assert loader() == frozenset({"web-research"})
    case.app.app_config = {"skills": {"disabled_builtins": []}}
    assert loader() == frozenset()


@pytest.mark.parametrize("kind", ["custom", "wrong_app", "keywords"])
def test_custom_builtin_loader_does_not_borrow_stock_provenance(
    runtime_case,
    tmp_path,
    monkeypatch,
    kind,
):
    from functools import partial

    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook.Chat.console_configuration_preparation import (
        standard_console_configuration_sources,
    )
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    case = runtime_case
    calls = []
    if kind == "custom":

        def custom():
            calls.append("called")
            return frozenset()

        # A claimed module name alone must never move an injected callback.
        custom.__module__ = wiring.__name__
        loader = custom
    elif kind == "wrong_app":
        loader = partial(
            wiring._read_app_disabled_builtin_skills, SimpleNamespace(app_config={})
        )
    else:
        loader = partial(wiring._read_app_disabled_builtin_skills, app=case.app)
    local = LocalSkillsService(
        store_dir=tmp_path / kind, builtin_disabled_loader=loader
    )
    monkeypatch.setattr(case.app, "local_skills_service", local, raising=False)
    # The app registered a stock loader; the injected one is not it.
    registered = partial(wiring._read_app_disabled_builtin_skills, case.app)
    monkeypatch.setattr(
        case.app, "_local_skills_callbacks", (None, None, registered), raising=False
    )
    assert not standard_console_configuration_sources(
        case.app,
        case.store,
        case.controller,
        session_id=case.session.id,
    )
    assert calls == []


def test_exact_stock_cold_plugin_factory_is_not_called_for_published_read(
    runtime_case,
    tmp_path,
    monkeypatch,
):
    from types import MethodType

    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook.Chat.console_configuration_capture import (
        capture_skill_context_maximum,
    )
    from tldw_chatbook.Chat.console_configuration_preparation import (
        _ready_published_plugin,
        standard_console_configuration_sources,
    )
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    case = runtime_case
    monkeypatch.setattr(case.app, "_plugin_service", None, raising=False)
    factory = MethodType(wiring.ServiceWiringMixin._build_plugin_service, case.app)
    # The app registered this factory when it built its skills stack.
    monkeypatch.setattr(
        case.app, "_local_skills_callbacks", (None, factory, None), raising=False
    )
    local = LocalSkillsService(
        store_dir=tmp_path / "cold-plugin", plugin_service_factory=factory
    )
    monkeypatch.setattr(case.app, "local_skills_service", local, raising=False)
    assert _ready_published_plugin(local, case.app) is None
    assert standard_console_configuration_sources(
        case.app, case.store, case.controller, session_id=case.session.id
    )
    captured = capture_skill_context_maximum(case.app, _plugin_service=None)
    assert captured["available_skills"] == []
    assert local._plugin_service is None and case.app._plugin_service is None


@pytest.mark.parametrize("local_ready", [False, True])
def test_published_plugin_read_preserves_exact_app_or_local_facade(
    runtime_case,
    tmp_path,
    monkeypatch,
    local_ready,
):
    from types import MethodType

    from tldw_chatbook import app_service_wiring as wiring
    from tldw_chatbook.Chat.console_configuration_preparation import (
        _ready_published_plugin,
    )
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    case = runtime_case
    published = PluginService(tmp_path / "app-facade", workspace_lookup=lambda *_: None)
    existing = (
        PluginService(tmp_path / "local-facade", workspace_lookup=lambda *_: None)
        if local_ready
        else None
    )
    monkeypatch.setattr(case.app, "_plugin_service", published, raising=False)
    factory = MethodType(wiring.ServiceWiringMixin._build_plugin_service, case.app)
    monkeypatch.setattr(
        case.app, "_local_skills_callbacks", (None, factory, None), raising=False
    )
    local = LocalSkillsService(
        store_dir=tmp_path / "published",
        plugin_service=existing,
        plugin_service_factory=factory,
    )
    assert _ready_published_plugin(local, case.app) is (
        existing if local_ready else published
    )
    assert local._plugin_service is existing
    assert published._thread is None


@pytest.mark.parametrize("replacement", ["factory", "getter"])
def test_sync_custom_plugin_source_keeps_original_getter_and_loop_affinity(
    runtime_case,
    tmp_path,
    monkeypatch,
    replacement,
):
    from tldw_chatbook.Chat.console_configuration_capture import (
        capture_skill_context_maximum,
    )
    from tldw_chatbook.Chat.console_configuration_preparation import (
        standard_console_configuration_sources,
    )
    from tldw_chatbook.Plugins.service import PluginService
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    case = runtime_case
    calls = []
    custom = SimpleNamespace(
        capture_maximum=lambda _workspace: {
            "available_skills": [{"name": "custom-retained"}]
        }
    )

    def original_dynamic_source(*_args):
        calls.append(threading.get_ident())
        return custom

    if replacement == "factory":
        local = LocalSkillsService(
            store_dir=tmp_path / replacement,
            plugin_service_factory=original_dynamic_source,
        )
    else:
        # A ready real facade must not bypass a replaced class getter.
        cached = PluginService(tmp_path / "cached", workspace_lookup=lambda *_: None)
        local = LocalSkillsService(
            store_dir=tmp_path / replacement, plugin_service=cached
        )
        monkeypatch.setattr(
            LocalSkillsService, "plugin_service", property(original_dynamic_source)
        )
    monkeypatch.setattr(case.app, "local_skills_service", local, raising=False)
    # An unregistered factory is refused before it runs. A replaced class
    # getter is not inspected (ADR-356 rule 4): the ready slot decides.
    assert standard_console_configuration_sources(
        case.app, case.store, case.controller, session_id=case.session.id
    ) is (replacement == "getter")
    assert calls == []
    captured = capture_skill_context_maximum(case.app)
    assert [row["name"] for row in captured["available_skills"]] == ["custom-retained"]
    assert calls == [threading.get_ident()]


@pytest.mark.asyncio
async def test_received_public_capture_uses_detached_selected_values(runtime_case):
    case = runtime_case
    selected = _selection(
        case,
        rag_defaults={"top_k": 7, "source_types": ["notes"]},
        tool_configuration={"local_tools_enabled": False, "direct_library_tools": False},
    )
    captured = await case.controller.capture_turn_configuration_snapshot(
        case.session.id, selection=selected
    )
    assert captured.session_id == case.session.id
    assert captured.provider_selection == selected.provider_selection
    assert captured.rag_defaults["top_k"] == 7
    assert captured.rag_defaults["source_types"] == ("notes",)
    assert captured.tool_configuration["local_tools_enabled"] is False
    assert captured.tool_configuration["direct_library_tools"] is False
    assert not case.controller._preparation_reads
    assert not case.runtime._preparation_reads


@pytest.mark.asyncio
async def test_prepared_skills_without_selected_source_refuses_before_capture(runtime_case):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    case = runtime_case
    with pytest.raises(RecoveryRequired):
        await case.controller.capture_turn_configuration_snapshot(
            case.session.id, _prepared_skills=object()
        )
    assert not case.controller._preparation_reads


@pytest.mark.asyncio
async def test_received_profile_depth_capture_uses_lightweight_metadata():
    """Use the actual selected-capture owner with a cold real profile-depth read."""
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.console_configuration_preparation import (
        ConsoleTurnCaptureSelection,
    )

    app = SimpleNamespace(app_config=config.load_settings())
    store = ConsoleChatStore()
    session = store.create_session(
        title="Cold metadata capture",
        workspace_id="global",
        ephemeral=True,
        settings=ConsoleSessionSettings(provider="openai", model="synthetic-metadata"),
    )
    runtime = ConsoleRuntime(app)
    runtime.set_chat_store(store)
    controller = runtime.ensure_chat_controller(
        store=store,
        provider_gateway=SimpleNamespace(),
        agent_runtime_enabled=False,
    )
    selection = ConsoleTurnCaptureSelection(
        ConsoleProviderSelection(
            provider="openai", explicit_model="synthetic-metadata"
        ),
        None,
        None,
        {"local_tools_enabled": False, "direct_library_tools": False},
        session.workspace_id,
        False,
        False,
    )
    try:
        captured = await controller.capture_turn_configuration_snapshot(
            session.id,
            selection=selection,
        )
        assert captured.session_id == session.id
        assert captured.provider_selection == selection.provider_selection
        assert captured.rag_defaults["top_k"] == 15
        assert not controller._preparation_reads and not runtime._preparation_reads
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("store_state", ["bound", "unbound", "replaced_reader"])
async def test_unbound_mcp_store_is_no_mcp_source_on_the_runtime_route(
    runtime_case, monkeypatch, tmp_path, store_state
):
    """TASK-40714: the runtime route captures an unbound store as no MCP.

    The store is built unbound, with pinned I/O off at its bind and outside
    the profile's data folder (Windows binds: it has pinned I/O, TASK-41308).
    A stock-typed service with a replaced reader is still refused.
    """
    from tldw_chatbook.Backup_Recovery import mcp_source_participants, raw_participants
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_console_turn_configuration_owned,
        standard_console_configuration_sources,
    )
    from tldw_chatbook.MCP.local_store import LocalMCPStore

    case = runtime_case
    service = case.app.unified_mcp_service
    if store_state != "bound":
        real_bind = mcp_source_participants.bind

        def bind_without_pinned_io(source):
            with monkeypatch.context() as patch:
                patch.setattr(raw_participants, "_pinned_io_available", lambda: False)
                return real_bind(source)

        monkeypatch.setattr(mcp_source_participants, "bind", bind_without_pinned_io)
        store = LocalMCPStore(tmp_path / "local_mcp_store.json")
        assert store not in mcp_source_participants._BINDINGS
        monkeypatch.setattr(service.local_service, "store", store)
    if store_state == "replaced_reader":
        original = service.get_kill_switch
        monkeypatch.setattr(
            service, "get_kill_switch", lambda *a, **k: original(*a, **k), raising=False
        )
    eligible = standard_console_configuration_sources(
        case.app, case.store, case.controller, session_id=case.session.id
    )
    assert eligible is (store_state != "replaced_reader")
    if store_state != "unbound":
        return
    result = await capture_console_turn_configuration_owned(
        case.app,
        case.store,
        case.session.id,
        selection=_selection(case),
        creator=case.controller,
        reads=case.controller._preparation_reads,
        require_current=lambda: None,
    )
    assert dict(result.mcp_definition_maximum) == {}
    assert result.mcp_definition_sources == ()
    assert result.mcp_definition_complete is True


# --- TASK-41644 review: bound-method checks without __func__ identity ---


class _AlwaysEqual:
    """A callable that claims to equal every function (an impostor)."""

    def __eq__(self, other):
        return True

    def __call__(self, *args, **kwargs):
        raise AssertionError("the impostor must never run")


class _ImpostorSlot:
    """A class-level replacement that binds the impostor on every read."""

    def __eq__(self, other):
        return True  # so a `== slot` compare would wrongly accept the impostor

    def __get__(self, instance, owner):
        return MethodType(_AlwaysEqual(), instance)


class _MountedOwner:
    def _build_console_turn_execution_context(self, session_id):
        raise AssertionError("not called by the capture under test")

    def _build_console_turn_capture_selection(self, session_id, *, scratch_owner):
        return SimpleNamespace(rag_defaults=None)


class _MountedScreen:
    def _console_runtime(self):
        raise AssertionError("not called by the capture under test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attack",
    [
        "none",
        "instance_builder",
        "instance_selector",
        "instance_runtime",
        "class_builder",
        "class_runtime",
    ],
)
async def test_mounted_capture_refuses_an_always_equal_bound_impostor(
    monkeypatch, attack
):
    """``==`` lets the function answer for itself, so it cannot decide alone."""
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat import console_configuration_preparation as prep
    from tldw_chatbook.UI.Console_Modules import turn_admission

    owner, screen, app = _MountedOwner(), _MountedScreen(), SimpleNamespace()
    owner._scratch_snapshot_provider = partial(lambda screen, session_id: None, screen)
    screen._session, screen.app_instance = owner, app
    controller = SimpleNamespace(app=app, store=SimpleNamespace(), _scratch_spaces=object())
    monkeypatch.setattr(prep, "standard_console_configuration_sources", lambda *a, **k: True)
    monkeypatch.setattr(
        turn_admission,
        "stock_mounted_capture",
        lambda owner: (owner._build_console_turn_capture_selection, lambda adapter: True),
    )
    monkeypatch.setattr(turn_admission, "policy_producers_current", lambda: True)
    holder, name = {
        "instance_builder": (owner, "_build_console_turn_execution_context"),
        "instance_selector": (owner, "_build_console_turn_capture_selection"),
        "instance_runtime": (screen, "_console_runtime"),
        "class_builder": (_MountedOwner, "_build_console_turn_execution_context"),
        "class_runtime": (_MountedScreen, "_console_runtime"),
    }.get(attack, (None, None))

    async def capture(controller, session_id, *, require_current, **_):
        require_current()  # the stock seam is current
        if holder is owner or holder is screen:
            setattr(holder, name, MethodType(_AlwaysEqual(), holder))
        elif holder is not None:
            monkeypatch.setattr(holder, name, _ImpostorSlot())
        require_current()
        return "captured"

    monkeypatch.setattr(prep, "capture_runtime_configuration_owned", capture)
    captured = prep.capture_mounted_configuration_owned(
        controller,
        "session",
        owner,
        owner._build_console_turn_execution_context,
        require_current=lambda: None,
    )
    if attack == "none":
        assert await captured == "captured"
    else:
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            await captured


def test_composition_mcp_capture_refuses_a_rebound_checker(monkeypatch):
    """The registered checker decides; a different module binding is refused.

    Rebinding the registration tuple and the module attribute together is
    accepted, as is an in-place ``__code__`` swap: ADR-356 rule 4 takes the
    registration and current state, not code identity (TASK-41644).
    """
    from tldw_chatbook.Agents import mcp_tool_provider as provider_source
    from tldw_chatbook.Chat import console_chat_controller as controller_source
    from tldw_chatbook.Chat import console_configuration_preparation as prep
    from tldw_chatbook.MCP import console_snapshot

    def registered(controller):
        return True

    monkeypatch.setattr(controller_source, "_stock_console_tool_composition_current", registered)
    monkeypatch.setattr(controller_source, "_CONSOLE_TOOL_COMPOSITION_CHECKER", (registered,))
    monkeypatch.setattr(provider_source, "controller_factory_current", lambda factory: True)
    monkeypatch.setattr(provider_source, "console_preparation_pipeline_current", lambda source: True)
    monkeypatch.setattr(console_snapshot, "standard_console_catalog_sources", lambda service: True)
    creator = controller_source.ConsoleChatController.__new__(
        controller_source.ConsoleChatController
    )
    assert prep._composition_mcp_capture_current(creator, object())
    monkeypatch.setattr(
        controller_source, "_stock_console_tool_composition_current", lambda controller: True
    )
    assert not prep._composition_mcp_capture_current(creator, object())


def test_same_bound_attribute_fails_closed_without_an_instance_dict():
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat.console_configuration_preparation import (
        _same_bound_attribute,
    )

    class Slotted:
        __slots__ = ()

        def method(self):
            return None

    holder = Slotted()
    with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
        _same_bound_attribute(holder, "method", MethodType(Slotted.method, holder), Slotted.method)


class _SiteTwoScreen:
    def __init__(self, app, store):
        self.app_instance, self._store = app, store

    def _ensure_console_chat_store(self):
        return self._store

    def _build(self, session_id):
        return None

    _current_chat_store_accessor = _chat_store_accessor = _build
    _provider_readiness_app_config_fn = _build_provider_selection_fn = _build
    _scratch_snapshot_provider = _rag_source_types_accessor = _rag_top_k_accessor = _build


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ["none", "instance_callback"])
async def test_controller_capture_snapshot_refuses_a_per_instance_impostor(
    monkeypatch, attack
):
    """Site 2: the screen callbacks it records are re-read through the same helper."""
    import threading

    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Chat import console_configuration_preparation as prep
    from tldw_chatbook.MCP import console_snapshot
    from tldw_chatbook.UI.Console_Modules import turn_admission

    app = SimpleNamespace(app_config={}, unified_mcp_service=object())
    session = SimpleNamespace(id="s")
    store = SimpleNamespace(
        sessions=lambda: [session],
        session_settings_revision=lambda sid: 1,
        session_workspace_id=lambda sid: None,
    )
    screen = _SiteTwoScreen(app, store)
    provider = screen._build
    controller = SimpleNamespace(
        app=app, store=store, _turn_context_provider=None, _scratch_spaces=object(),
        _provider_config=object(), _disposed=False,
        _shutdown_requested=threading.Event(),
        _provider_selection_for_session=lambda sid: "selection",
    )
    monkeypatch.setattr(prep, "_mcp_attachment_source", lambda *a: None)
    monkeypatch.setattr(prep, "_require_mcp_attachment_source", lambda *a: None)
    monkeypatch.setattr(config, "current_config_identity", lambda: "id")
    monkeypatch.setattr(turn_admission, "is_stock_session_provider", lambda p: True)
    monkeypatch.setattr(console_snapshot, "standard_console_sources", lambda s: True)

    async def capture(controller, session_id, owner, provider, *, require_current):
        require_current()
        if attack != "none":
            screen._rag_top_k_accessor = MethodType(_AlwaysEqual(), screen)
        require_current()
        return "captured"

    monkeypatch.setattr(prep, "capture_mounted_configuration_owned", capture)
    captured = prep._controller_capture_turn_configuration_snapshot(
        controller, "s", context_provider=provider
    )
    if attack == "none":
        assert await captured == "captured"
    else:
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            await captured



@pytest.mark.asyncio
async def test_installed_skill_trust_read_runs_outside_the_workspace_operation(
    runtime_case, tmp_path, monkeypatch
):
    """#3347: a user-installed skill made every Console send die before submit.

    The trust read (`SkillTrustService.status_for_skill`) takes its own storage
    operation on the skills and trust roots. Capture held the Workspace DB's
    repository operation around it, so storage admission refused the nested
    path (`operation_path_outside_scope`) and the turn ended with no message.
    A built-in-only catalog never reads trust, and the trust-service-less test
    mode never takes storage, so neither showed it.
    """
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_console_turn_configuration_owned,
        standard_console_configuration_sources,
    )
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Library.library_rag_state import library_rag_profile_top_k
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
    from tldw_chatbook.Skills_Interop.skill_trust_store import (
        FileSkillTrustGenerationMarkerStore,
        SkillTrustStore,
    )
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    case = runtime_case
    case.app.workspace_registry_service = LocalWorkspaceRegistryService(
        WorkspaceDB(
            config.get_user_data_dir() / "installed-skill-capture.sqlite",
            client_id="installed-skill-capture",
        )
    )
    root = tmp_path / "skills"
    trust = SkillTrustService(
        skills_dir=root / "skills",
        trust_store=SkillTrustStore(
            store_dir=root / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(
                root / "marker.json", store_dir=root
            ),
        ),
    )
    local = LocalSkillsService(store_dir=root, trust_service=trust)
    await local.import_skill(
        name="release-notes",
        content="---\nname: release-notes\ndescription: Draft release notes\n---\nBody.\n",
    )
    monkeypatch.setattr(case.app, "local_skills_service", local, raising=False)
    assert standard_console_configuration_sources(
        case.app, case.store, case.controller, session_id=case.session.id
    )

    captured = await capture_console_turn_configuration_owned(
        case.app,
        case.store,
        case.session.id,
        selection=_selection(case),
        creator=case.controller,
        reads=case.controller._preparation_reads,
        observers=(case.runtime._preparation_reads,),
        require_current=lambda: None,
    )

    skills = captured.skill_context_maximum
    listed = {row["name"] for row in (*skills["available_skills"], *skills["blocked_skills"])}
    assert "release-notes" in listed
    # The RAG profile read was refused the same way and fell back silently.
    assert captured.rag_defaults["top_k"] == library_rag_profile_top_k() != 5
