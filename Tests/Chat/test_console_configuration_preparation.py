"""Selected-value, affinity and optional-domain controls for owned capture."""

from __future__ import annotations

from dataclasses import replace
import threading
from types import SimpleNamespace

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
    # Legacy synchronous capture still reads MCP; stock temporary sessions
    # intentionally capture an empty ceiling because they never compose MCP.
    assert case.store.session_is_ephemeral(case.session.id)
    assert expected.mcp_definition_maximum
    assert (
        actual.mcp_definition_capture == expected.mcp_definition_capture == "captured"
    )
    expected = replace(
        expected, mcp_tool_maximum=frozenset(), mcp_definition_maximum={}
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
async def test_runtime_absent_provider_config_retains_original_default_and_one_budget_read(
    runtime_case,
    monkeypatch,
):
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_console_turn_configuration_owned,
    )
    from tldw_chatbook.Chat import console_agent_bridge

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
    from tldw_chatbook.Chat.console_configuration_preparation import (
        _ready_published_plugin,
        standard_console_configuration_sources,
    )
    from tldw_chatbook.Chat.console_configuration_capture import (
        capture_skill_context_maximum,
    )
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

    case = runtime_case
    monkeypatch.setattr(case.app, "_plugin_service", None, raising=False)
    factory = MethodType(wiring._STOCK_PLUGIN_SERVICE_FACTORY[0], case.app)
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
    local = LocalSkillsService(
        store_dir=tmp_path / "published",
        plugin_service=existing,
        plugin_service_factory=MethodType(
            wiring._STOCK_PLUGIN_SERVICE_FACTORY[0], case.app
        ),
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
    assert not standard_console_configuration_sources(
        case.app, case.store, case.controller, session_id=case.session.id
    )
    captured = capture_skill_context_maximum(case.app)
    assert [row["name"] for row in captured["available_skills"]] == ["custom-retained"]
    assert calls == [threading.get_ident()]
