"""One screen-free configuration producer preserves owning-session semantics."""

import asyncio
import builtins
import gc
import threading
from types import SimpleNamespace
import weakref

import pytest
import pytest_asyncio

from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

catalog_controls = snapshot_controls.catalog_controls
catalog_store = snapshot_controls.catalog_store
local_root = snapshot_controls.local_root
mcp_sources = snapshot_controls.mcp_sources
snapshot_case = snapshot_controls.snapshot_case


@pytest_asyncio.fixture
async def runtime_case(snapshot_case):
    case = snapshot_case
    case.app.app_config = {"console": {"native_tool_calls": "false"}}
    case.app.console_prompt_history_factory = lambda: SimpleNamespace()
    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(provider="deepseek", model="deepseek-chat"),
    )
    case.store.create_session(title="Foreground", workspace_id="other-workspace")
    runtime = ConsoleRuntime(case.app)
    runtime.set_chat_store(case.store)
    controller = runtime.ensure_chat_controller(
        store=case.store,
        provider_gateway=SimpleNamespace(),
        agent_runtime_enabled=False,
    )
    assert controller._turn_context_provider is None
    case.controller = controller
    case.runtime = runtime
    snapshot_controls._loop_projection(case)
    try:
        yield case
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
async def test_default_runtime_capture_does_not_import_console_session(
    runtime_case, monkeypatch
):
    case = runtime_case
    original_import = builtins.__import__
    forbidden = "tldw_chatbook.UI.Console_Modules.session"
    attempts = []

    def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == forbidden or (
            name == "tldw_chatbook.UI.Console_Modules" and "session" in fromlist
        ):
            attempts.append(name)
            raise AssertionError("default runtime capture imported Console session UI")
        return original_import(name, globals, locals, fromlist, level)

    # Intercept the import operation, including cached-module requests. A finder
    # or sys.modules check would miss the old unconditional import after fixtures.
    probe = snapshot_controls._MaximumProbe(case.source, case.permissions)
    with monkeypatch.context() as patch, probe.installed():
        patch.setattr(builtins, "__import__", checked_import)
        captured = await case.controller.capture_turn_configuration_snapshot(
            case.session.id
        )
    assert attempts == []
    assert captured.session_id == case.session.id
    assert case.store.active_session_id != case.session.id
    assert captured.provider_selection.provider == "deepseek"
    assert captured.session_settings == case.store.effective_session_settings(
        case.session.id
    )
    assert case.store.session_is_ephemeral(case.session.id)
    assert captured.mcp_definition_capture == "captured"
    assert captured.mcp_tool_maximum == frozenset()
    assert dict(captured.mcp_definition_maximum) == {}
    assert len(probe.permission_threads) == len(probe.read_threads) == 0
    assert not probe.leases


@pytest.mark.asyncio
async def test_default_capture_releases_detached_view_while_native_worker_is_held(
    runtime_case,
):
    from Tests.Chat.test_console_configuration_worker_lifetime import (
        _OriginalConfigurationWorkspaceRead,
    )
    from tldw_chatbook import config
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    case = runtime_case
    database = WorkspaceDB(
        config.get_user_data_dir() / "detached-view-capture.sqlite",
        client_id="detached-view-capture",
    )
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(workspace_id="capture-owned", name="Capture owned")
    case.app.workspace_registry_service = registry
    case.session.workspace_id = "capture-owned"
    database.close()

    class View:
        def console_view_hooks(self):
            return {}

    view = View()
    reference = weakref.ref(view)
    generation = case.runtime.attach_view(view)
    # MCP is now deferred/empty here. Retain the actual stock Workspace reader
    # so this still proves view detachment while native capture is outstanding.
    probe = _OriginalConfigurationWorkspaceRead(registry)
    with probe.installed():
        task = asyncio.create_task(
            case.controller.capture_turn_configuration_snapshot(case.session.id)
        )
        try:
            deadline = asyncio.get_running_loop().time() + 4
            while not probe.entered.is_set() and not task.done():
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.01)
            assert probe.entered.is_set()
            assert probe.thread is not threading.current_thread()
            assert probe.live_at_entry and not probe.release_timed_out
            assert case.runtime.detach_view(view, generation)
            del view
            gc.collect()
            assert reference() is None
            assert not task.done()
            assert probe.lease in storage_admission._live_leases
            assert not probe.retired()
            probe.release.set()
            captured = await task
            assert captured.session_id == case.session.id
            assert captured.provider_selection.provider == "deepseek"
            assert not probe.release_timed_out
            assert probe.retired()
        finally:
            await catalog_controls._settle(task, probe)
            database.close()


def test_shared_producer_captures_complete_detached_snapshot(snapshot_case, tmp_path):
    from dataclasses import fields, replace
    from uuid import UUID

    from tldw_chatbook.Character_Chat.emote_directives import (
        CharacterEmoteAssetReference,
        CharacterEmoteRunSnapshot,
    )
    from tldw_chatbook.Chat.attachment_core import max_history_images
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_configuration_capture import (
        capture_console_turn_configuration,
    )
    from tldw_chatbook.Chat.console_dispatch_checkpoint import (
        ConsoleLibraryItemScopeSnapshot,
    )
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
        ConsoleLibraryPolicySnapshot,
    )
    from tldw_chatbook.Chat.console_roleplay_identity import ConsolePresentationContext
    from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSnapshot
    from tldw_chatbook.Chat.console_turn_context import (
        ConsoleCharacterAuthoritySnapshot,
        ConsoleProjectAuthoritySnapshot,
        ConsoleProjectBindingSnapshot,
        ConsoleTurnConfigurationSnapshot,
    )
    from tldw_chatbook.Chat.rag_scope import RagScope, ScopeItem
    from tldw_chatbook.model_capabilities import is_vision_capable
    from tldw_chatbook.Workspaces import SkippedReviewRoot

    case = snapshot_case
    session = case.session
    settings = ConsoleSessionSettings(provider="deepseek", model="deepseek-chat")
    case.store.replace_session_settings(session.id, settings)
    case.store.create_session(title="Foreground", workspace_id="other-workspace")
    session.assistant_kind = "character"
    session.runtime_backend = "local"
    session.assistant_id = "7"
    session.assistant_authority_id = "character:7"
    session.character_id = 7
    session.identity_revision = 3
    session.rag_scope_holder.set(
        RagScope(
            (ScopeItem("note", "note-a"), ScopeItem("media", "media-a")), "selected"
        )
    )
    policy = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=4,
        source="durable",
    )
    session.library_policy_holder.snapshot = policy
    roots, aliases = [str(tmp_path / "review")], ["selected-root"]
    skipped = [
        SkippedReviewRoot(alias="pending-root", reason="Preparing change history")
    ]
    review_calls, skill_calls, character_calls, policy_calls = [], [], [], []

    def admit(workspace_id):
        review_calls.append(workspace_id)
        return SimpleNamespace(
            ready_roots=roots, ready_aliases=aliases, skipped_roots=skipped
        )

    allowed = {"name": "builtin-skill", "trust_blocked": False, "tags": ["selected"]}
    blocked = {"name": "blocked-skill", "trust_blocked": True}
    plugin = {"name": "plugin-skill", "definition_digest": "frozen-plugin"}

    def plugin_maximum(workspace_id):
        skill_calls.append(workspace_id)
        return {"available_skills": [plugin]}

    graph = {
        "pack": {"id": 11},
        "version": {"id": 13},
        "assets": [{"id": 17, "expression_key": "happy"}],
    }

    def active_pack(actor_kind, actor_id):
        character_calls.append((actor_kind, actor_id))
        return graph

    case.app.change_review_consent_service = SimpleNamespace(admit_turn=admit)
    case.app.local_skills_service = SimpleNamespace(
        _visible_records=lambda: {
            "a": {"source": "builtin", "summary": allowed},
            "b": {"source": "local", "summary": blocked},
        },
        _summary_for_record=lambda record: record["summary"],
        plugin_service=SimpleNamespace(capture_maximum=plugin_maximum),
    )
    case.app.workspace_registry_service = SimpleNamespace(
        get_workspace=lambda workspace_id: policy_calls.append(workspace_id)
    )
    case.app.local_character_persona_service = SimpleNamespace(
        get_persona_profile=lambda persona_id: policy_calls.append(persona_id)
    )
    selection = ConsoleProviderSelection(
        provider="openai",
        explicit_model="gpt-4o",
        temperature=0.23,
        top_p=0.8,
        min_p=0.1,
        top_k=19,
        max_tokens=321,
        seed=17,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        reasoning_effort="low",
        reasoning_summary="concise",
        verbosity="low",
        thinking_effort="medium",
        thinking_budget_tokens=400,
        streaming=False,
        system_prompt="selected prompt",
    )
    scratch = ConsoleScratchSnapshot(tmp_path / "scratch", "selected-token", (7, 11))
    presentation = ConsolePresentationContext(
        user_name="Selected user",
        assistant_kind="character",
        character_name="Selected character",
        revision=3,
    )
    binding = ConsoleProjectBindingSnapshot(
        binding_id="selected-binding",
        workspace_id=session.workspace_id,
        display_name="Selected folder",
        root=str(tmp_path / "selected"),
        locator_fingerprint="selected-locator",
        exclusions=("private",),
    )
    project = ConsoleProjectAuthoritySnapshot(
        workspace_id=session.workspace_id,
        enabled=True,
        working_folder_binding_id=binding.binding_id,
        selected=binding,
        options=(binding,),
    )
    rag = {"source_types": ["notes"], "top_k": 9}
    tools = {"local_tools_enabled": True, "nested": {"allowed": ["fs_read"]}}
    rules = [{"rule_kind": "mcp_tool", "rule_name": "write", "allowed": False}]
    maximum = {"local:one::first": "a" * 64}
    inputs = dict(
        provider_selection=selection,
        scratch_space=scratch,
        presentation_context=presentation,
        rag_defaults=rag,
        tool_configuration=tools,
        project_authority=project,
        skill_workspace_id=session.workspace_id,
        character_repository=SimpleNamespace(get_active_actor_pack=active_pack),
        tool_policy_profile_id="selected-policy",
        persona_policy_rules=rules,
        mcp_definition_maximum=maximum,
    )
    expected = ConsoleTurnConfigurationSnapshot.capture(
        session_id=session.id,
        provider_selection=selection,
        scratch_space=scratch,
        session_settings=settings,
        workspace_roots=roots,
        change_review_root_aliases=aliases,
        change_review_skipped_roots=skipped,
        presentation_context=presentation,
        library_policy_maximum=policy,
        library_scope_maximum=ConsoleLibraryItemScopeSnapshot(
            ("note-a",), ("media-a",), False
        ),
        project_authority=project,
        character_authority=ConsoleCharacterAuthoritySnapshot(
            identity_revision=3,
            runtime_backend="local",
            assistant_id="7",
            assistant_authority_id="character:7",
            local_character_id=7,
            emote_snapshot=CharacterEmoteRunSnapshot(
                actor_id=7,
                pack_id=11,
                pack_version_id=13,
                states=("happy",),
                assets=(CharacterEmoteAssetReference("happy", "happy", 17),),
            ),
        ),
        prompt_transform_inputs={
            "conversation_id": None,
            "dictionary_entries": (),
            "world_books": (),
            "world_enabled": False,
        },
        skill_context_maximum={
            "plugin_run_id": "pending:normalized",
            "backend": "local",
            "available_skills": [allowed, plugin],
            "blocked_skills": [blocked],
            "context_text": "- builtin-skill\n- plugin-skill",
        },
        mcp_tool_maximum=maximum,
        mcp_definition_maximum=maximum,
        capabilities={
            "vision": is_vision_capable("openai", "gpt-4o"),
            "max_history_images": max_history_images("openai", "gpt-4o"),
        },
        rag_defaults=rag,
        tool_configuration=tools,
        provider_payload_settings={
            "streaming": False,
            "temperature": 0.23,
            "top_p": 0.8,
            "min_p": 0.1,
            "top_k": 19,
            "max_tokens": 321,
            "seed": 17,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.3,
            "reasoning_effort": "low",
            "reasoning_summary": "concise",
            "verbosity": "low",
            "thinking_effort": "medium",
            "thinking_budget_tokens": 400,
        },
        persona_policy_rules=rules,
        tool_policy_profile_id="selected-policy",
    )
    probe = snapshot_controls._MaximumProbe(case.source, case.permissions)
    with probe.installed():
        captured = capture_console_turn_configuration(
            case.app, case.store, session.id, **inputs
        )
        later = capture_console_turn_configuration(
            case.app, case.store, session.id, **inputs
        )
        explicit_none = capture_console_turn_configuration(
            case.app,
            case.store,
            session.id,
            **(inputs | {"tool_policy_profile_id": None, "persona_policy_rules": None}),
        )
    assert not probe.permission_threads and not probe.read_threads
    first_nonce = captured.skill_context_maximum["plugin_run_id"]
    later_nonce = later.skill_context_maximum["plugin_run_id"]
    assert first_nonce != later_nonce
    for nonce in (first_nonce, later_nonce):
        assert nonce.startswith("pending:")
        UUID(nonce.removeprefix("pending:"))
    normalized = replace(
        captured,
        skill_context_maximum=dict(captured.skill_context_maximum)
        | {"plugin_run_id": "pending:normalized"},
    )
    assert {
        field.name: getattr(normalized, field.name) for field in fields(normalized)
    } == {field.name: getattr(expected, field.name) for field in fields(expected)}
    assert explicit_none.tool_policy_profile_id == "default"
    assert explicit_none.persona_policy_rules == ()
    assert policy_calls == []
    assert review_calls == skill_calls == [session.workspace_id] * 3
    assert character_calls == [("character", 7)] * 3

    roots.append(str(tmp_path / "later"))
    aliases.clear()
    skipped.clear()
    allowed["tags"].append("later")
    plugin["definition_digest"] = "later"
    graph["assets"][0]["id"] = 99
    rag["source_types"].append("media")
    tools["nested"]["allowed"].append("fs_write")
    rules[0]["allowed"] = True
    maximum["local:other::second"] = "b" * 64
    session.rag_scope_holder.set(RagScope((ScopeItem("note", "later-note"),), "later"))
    assert normalized == expected
    assert captured.rag_defaults["source_types"] == ("notes",)
    assert captured.tool_configuration["nested"]["allowed"] == ("fs_read",)
    assert captured.skill_context_maximum["available_skills"][0]["tags"] == (
        "selected",
    )
    assert captured.character_authority.emote_snapshot.assets[0].asset_id == 17
    with pytest.raises(TypeError):
        captured.mcp_definition_maximum["local:other::second"] = "b" * 64


@pytest.mark.asyncio
async def test_both_adapters_use_original_producer_and_preserve_distinct_inputs(
    runtime_case,
):
    import sys

    from tldw_chatbook.Chat.console_configuration_capture import (
        capture_console_turn_configuration,
    )
    from tldw_chatbook.Library.library_rag_state import library_rag_profile_top_k
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController

    case = runtime_case
    skill_workspaces = []
    case.app.local_skills_service = SimpleNamespace(
        _visible_records=lambda: {},
        _summary_for_record=lambda record: record,
        plugin_service=SimpleNamespace(
            capture_maximum=lambda workspace_id: skill_workspaces.append(workspace_id)
            or {"available_skills": []}
        ),
    )
    runtime_top_k = library_rag_profile_top_k()
    mounted_config = {
        "console": {
            "native_tool_calls": True,
            "agent_runtime": True,
            "direct_library_tools": 0,
        }
    }
    owner = ConsoleSessionController.__new__(ConsoleSessionController)
    owner.app_instance = case.app
    owner._provider_readiness_app_config_fn = lambda: mounted_config
    owner._build_provider_selection_fn = case.controller._provider_selection_for_session
    owner._current_chat_store_accessor = lambda: case.store
    owner._chat_store_accessor = lambda: case.store
    owner._rag_source_types_accessor = lambda: ["notes"]
    owner._rag_top_k_accessor = lambda: runtime_top_k + 5
    owner._scratch_snapshot_provider = case.controller._scratch_spaces.snapshot
    owner._resolve_turn_tool_policy_profile_id = lambda workspace_id: "mounted-policy"
    owner._resolve_turn_persona_policy_rules = lambda session_id: (
        {"rule_name": "mounted"},
    )
    calls, returns = [], []
    code = capture_console_turn_configuration.__code__

    def observe(frame, event, arg):
        if frame.f_code is code:
            if event == "call":
                calls.append(
                    (
                        frame.f_locals["app"],
                        frame.f_locals["store"],
                        frame.f_locals["session_id"],
                    )
                )
            elif event == "return":
                returns.append(arg)

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        runtime_snapshot = case.controller.resolve_runtime_turn_configuration_snapshot(
            case.session.id, mcp_definition_maximum={}
        )
        mounted_snapshot = owner._build_console_turn_execution_context(
            case.session.id, mcp_definition_maximum={}
        )
    finally:
        sys.setprofile(previous)
    assert calls == [(case.app, case.store, case.session.id)] * 2
    assert len(returns) == 2
    assert returns[0] is runtime_snapshot and returns[1] is mounted_snapshot
    assert skill_workspaces == [case.session.workspace_id, None]
    assert runtime_snapshot.rag_defaults == {"top_k": runtime_top_k}
    assert mounted_snapshot.rag_defaults == {
        "source_types": ("notes",),
        "top_k": runtime_top_k + 5,
    }
    assert runtime_snapshot.tool_configuration["agent_runtime_enabled"] is False
    assert mounted_snapshot.tool_configuration["agent_runtime_enabled"] is True
    assert runtime_snapshot.tool_configuration["native_tool_calls_enabled"] is False
    assert mounted_snapshot.tool_configuration["native_tool_calls_enabled"] is True
    assert mounted_snapshot.tool_configuration["direct_library_tools"] is True
    assert runtime_snapshot.tool_policy_profile_id == "default"
    assert mounted_snapshot.tool_policy_profile_id == "mounted-policy"
    assert mounted_snapshot.persona_policy_rules == ({"rule_name": "mounted"},)
    assert runtime_snapshot.session_settings == mounted_snapshot.session_settings
    assert runtime_snapshot.provider_selection == mounted_snapshot.provider_selection


@pytest.mark.asyncio
@pytest.mark.parametrize("builder", ["selection", "execution"])
@pytest.mark.parametrize(
    "resident_enabled, raw_enabled",
    [(False, True), (None, False), (None, True)],
    ids=["resident-disables", "absent-raw-disabled", "absent-raw-enabled"],
)
async def test_mounted_builders_use_only_resident_runtime_flag_override(
    runtime_case,
    builder,
    resident_enabled,
    raw_enabled,
):
    """Original adapters share the live flag but retain other readiness inputs."""
    from copy import deepcopy
    from functools import partial

    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.UI.Console_Modules.wiring import _stock_console_scratch_snapshot

    case = runtime_case
    resident_console = {"native_tool_calls": False, "exchange_capture": True}
    if resident_enabled is not None:
        resident_console["agent_runtime"] = resident_enabled
    case.app.app_config = {"console": resident_console}
    readiness_config = {
        "console": {
            "agent_runtime": raw_enabled,
            "native_tool_calls": True,
            "exchange_capture": False,
        }
    }
    resident_before = deepcopy(case.app.app_config)
    readiness_before = deepcopy(readiness_config)
    # Reuse the adapter setup above, retaining the original policy methods and
    # original scratch callback so the real selection builder accepts its source.
    owner = ConsoleSessionController.__new__(ConsoleSessionController)
    owner.app_instance = case.app
    owner._provider_readiness_app_config_fn = lambda: readiness_config
    owner._build_provider_selection_fn = case.controller._provider_selection_for_session
    owner._current_chat_store_accessor = lambda: case.store
    owner._chat_store_accessor = lambda: case.store
    owner._rag_source_types_accessor = lambda: ["notes"]
    owner._rag_top_k_accessor = lambda: 17
    screen = SimpleNamespace(
        _session=owner,
        app_instance=case.app,
        _console_runtime=lambda: case.runtime,
    )
    owner._scratch_snapshot_provider = partial(_stock_console_scratch_snapshot, screen)
    if builder == "selection":
        captured = owner._build_console_turn_capture_selection(
            case.session.id,
            scratch_owner=case.runtime._scratch_spaces,
        )
        assert captured is not None, "Original mounted selection source was refused"
    else:
        captured = owner._build_console_turn_execution_context(
            case.session.id,
            mcp_definition_maximum={},
        )

    expected = raw_enabled if resident_enabled is None else resident_enabled
    # Only the runtime flag overrides readiness input; merging the resident
    # console dictionary would incorrectly alter the other two captured flags.
    assert captured.tool_configuration["native_tool_calls_enabled"] is True
    assert captured.tool_configuration["exchange_capture_enabled"] is False
    assert captured.provider_selection.provider == "deepseek"
    assert case.app.app_config == resident_before
    assert readiness_config == readiness_before
    assert captured.tool_configuration["agent_runtime_enabled"] is expected
    if builder == "selection":
        assert captured.agent_runtime_enabled is expected
        assert captured.project_bindings_eligible is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["configured", "explicit"])
async def test_custom_provider_keeps_sync_shape_affinity_and_live_errors(
    runtime_case, route
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot

    case = runtime_case
    expected = ConsoleTurnConfigurationSnapshot.capture(
        session_id=case.session.id,
        provider_selection=ConsoleProviderSelection(
            provider="custom", explicit_model="selected"
        ),
        rag_defaults={"custom": ["unchanged"]},
    )
    loop, thread = asyncio.get_running_loop(), threading.current_thread()
    calls = []
    failure = RuntimeError("custom capture failed")

    def custom(*args, **kwargs):
        calls.append(
            (args, kwargs, asyncio.get_running_loop(), threading.current_thread())
        )
        return expected

    def changed(*args, **kwargs):
        calls.append(
            (args, kwargs, asyncio.get_running_loop(), threading.current_thread())
        )
        raise failure

    async def capture(provider):
        if route == "configured":
            case.controller._turn_context_provider = provider
            return await case.controller.capture_turn_configuration_snapshot(
                case.session.id
            )
        return await case.controller.capture_turn_configuration_snapshot(
            case.session.id, context_provider=provider
        )

    probe = snapshot_controls._MaximumProbe(case.source, case.permissions)
    with probe.installed():
        assert await capture(custom) is expected
        with pytest.raises(RuntimeError) as caught:
            await capture(changed)
    assert caught.value is failure
    assert calls == [((case.session.id,), {}, loop, thread)] * 2
    assert not probe.permission_threads and not probe.read_threads


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["type", "session"])
async def test_custom_provider_retains_result_validation(runtime_case, invalid):
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnConfigurationSnapshot

    case = runtime_case
    value = (
        object()
        if invalid == "type"
        else ConsoleTurnConfigurationSnapshot.capture(
            session_id="another-session",
            provider_selection=ConsoleProviderSelection(provider="custom"),
        )
    )
    calls = []

    def custom(session_id):
        calls.append(session_id)
        return value

    probe = snapshot_controls._MaximumProbe(case.source, case.permissions)
    with probe.installed(), pytest.raises(
        TypeError if invalid == "type" else ValueError
    ):
        await case.controller.capture_turn_configuration_snapshot(
            case.session.id, context_provider=custom
        )
    assert calls == [case.session.id]
    assert not probe.permission_threads and not probe.read_threads
