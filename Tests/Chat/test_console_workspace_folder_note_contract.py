"""TASK-33940.1: the workspace note must name folders the way the tools accept.

The agent's only description of a workspace's bound folders is the
system-prompt note ``workspace_context_note`` builds. It used to list folders
relative to the app's launch directory, a coordinate system no file tool
resolves; a live gpt-4.1-mini run called ``fs_list {"path": "myproj"}`` and got
``not a directory: myproj``. These tests pin the contract end to end: the note
the real first-request plan carries must name each folder by the
``root_alias`` the run's fs_* tools advertise, and that exact call must reach
the folder through a real tool worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.Chat.console_agent_bridge import build_console_first_request_plan
from tldw_chatbook.Chat.console_chat_controller import (
    capture_project_instruction_authority,
    capture_run_admitted_workspace_roots,
    list_project_instruction_bindings,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools import workspace_file_roots as roots_module
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService

WORKSPACE_ID = "ws-proj"


@dataclass
class _WorkspaceRun:
    tmp_path: Path
    launch: Path
    folders: dict[str, Path]
    service: LocalWorkspaceRegistryService
    session: Any
    authority: Any


@pytest.fixture(autouse=True)
def _isolated_roots(monkeypatch):
    monkeypatch.setattr(roots_module, "_default_registry_instance", None)
    yield


def _workspace_run(
    tmp_path: Path,
    monkeypatch,
    *,
    names: tuple[str, ...] = ("myproj",),
    inside_launch: bool = False,
) -> _WorkspaceRun:
    tmp_path = tmp_path.resolve()
    launch = tmp_path / "launch" / "tldw_chatbook"
    launch.mkdir(parents=True)
    parent = launch / "work" if inside_launch else tmp_path / "code"
    service = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.db", client_id="note-contract")
    )
    service.create_workspace(workspace_id=WORKSPACE_ID, name="My Project")
    folders: dict[str, Path] = {}
    for name in names:
        folder = parent / name
        (folder / "src").mkdir(parents=True)
        (folder / f"{name}-README.md").write_text(f"# {name}\n", encoding="utf-8")
        service.add_folder_binding(WORKSPACE_ID, folder, allow_write=True)
        folders[name] = folder
    monkeypatch.setattr(roots_module, "_registry_factory", lambda: service)
    monkeypatch.setattr(roots_module, "_LAUNCH_CWD", str(launch))
    session = ConsoleChatStore().create_session(workspace_id=WORKSPACE_ID)
    authority = capture_project_instruction_authority(session, service)
    return _WorkspaceRun(tmp_path, launch, folders, service, session, authority)


def _local_provider(run: _WorkspaceRun, *, project_selection=None) -> LocalToolProvider:
    scratch = run.tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    admitted = capture_run_admitted_workspace_roots(
        session=run.session,
        registry=run.service,
        project_selection=project_selection,
        status_cache=None,
    )
    return LocalToolProvider(
        workspace_root=scratch,
        result_redaction_root=scratch,
        resolve_state=lambda _hub: EffectiveToolState(
            state="allow", origin="global_default"
        ),
        admitted_roots=admitted,
    )


def _plan(run: _WorkspaceRun, provider: LocalToolProvider):
    options = tuple(run.authority.options)
    return build_console_first_request_plan(
        shared_registry=ToolCatalogRegistry(),
        shared_allowed_tools=(),
        context={},
        skills_present=False,
        mcp_provider=None,
        builtin_gate=None,
        local_provider=provider,
        library_provider=None,
        library_authority=None,
        workspace_id=WORKSPACE_ID,
        workspace_read_binding_ids=tuple(item.binding_id for item in options),
        workspace_write_binding_ids=tuple(
            item.binding_id for item in options if item.allow_write
        ),
        workspace_binding_authority=options,
        ephemeral=False,
        diff_sink=None,
        scratch_root=None,
        scratch_lease=None,
        resolution=SimpleNamespace(
            model="gpt-4.1-mini", execution_key="openai", max_tokens=2_048
        ),
        fallback_model="gpt-4.1-mini",
        session_system_prompt="",
        native_tools=True,
        turn_skill_bindings=(),
        turn_bundle_block="",
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        agent_messages=[{"role": "user", "content": "What files are here?"}],
    )


def _advertised_aliases(provider: LocalToolProvider) -> tuple[str, ...]:
    schema = provider.load_schema("fs_list")
    return tuple(schema.parameters["properties"]["root_alias"]["enum"])


def _note_line_for(note: str, alias: str) -> str:
    lines = [line for line in note.splitlines() if alias in line]
    assert len(lines) == 1, f"alias {alias!r} must appear on exactly one note line"
    return lines[0]


@pytest.mark.parametrize("inside_launch", [False, True], ids=["outside", "inside"])
@pytest.mark.parametrize(
    "names", [("myproj",), ("myproj", "docs")], ids=["one-folder", "two-folders"]
)
@private_profile_test
def test_every_folder_the_note_names_is_listed_by_fs_list_with_the_alias_it_gives(
    request, tmp_path, monkeypatch, names, inside_launch
) -> None:
    run = _workspace_run(
        tmp_path, monkeypatch, names=names, inside_launch=inside_launch
    )
    provider = _local_provider(run)
    note = _plan(run, provider).config.workspace_context_note

    aliases = _advertised_aliases(provider)
    assert len(aliases) == len(names)
    by_alias = {
        item.binding_id: Path(item.root).name for item in run.authority.options
    }
    for alias in aliases:
        folder_name = by_alias[alias]
        assert folder_name in _note_line_for(note, alias)
        result = provider.invoke("fs_list", {"root_alias": alias, "path": "."})
        assert result.ok, result.error
        assert f"{folder_name}-README.md" in result.content


@private_profile_test
def test_note_never_frames_paths_relative_to_the_launch_directory(
    request, tmp_path, monkeypatch
) -> None:
    run = _workspace_run(tmp_path, monkeypatch)
    note = _plan(run, _local_provider(run)).config.workspace_context_note

    assert "launch directory" not in note
    assert "Launched from" not in note
    assert str(run.tmp_path) not in note  # never an absolute host path


@private_profile_test
def test_note_lists_only_the_working_folder_when_a_selection_narrows_the_run(
    request, tmp_path, monkeypatch
) -> None:
    run = _workspace_run(tmp_path, monkeypatch, names=("myproj", "docs"))
    selection = next(
        item
        for item in list_project_instruction_bindings(run.session, run.service)
        if item.root.name == "myproj"
    )
    provider = _local_provider(run, project_selection=selection)
    note = _plan(run, provider).config.workspace_context_note

    (alias,) = _advertised_aliases(provider)
    assert "myproj" in _note_line_for(note, alias)
    unselected = next(
        item.binding_id
        for item in run.authority.options
        if Path(item.root).name == "docs"
    )
    assert unselected not in note
    assert "docs" not in note


@private_profile_test
def test_note_says_bound_folders_need_path_tools_when_the_run_has_none(
    request, tmp_path, monkeypatch
) -> None:
    run = _workspace_run(tmp_path, monkeypatch)
    options = tuple(run.authority.options)
    note = roots_module.workspace_context_note(
        WORKSPACE_ID,
        binding_authority=options,
        status_cache=None,
        path_tool_aliases=(),
    )

    assert options[0].binding_id not in note
    assert "fs_*" in note  # names what is missing, not a path to try


@private_profile_test
def test_note_says_builtin_relative_paths_resolve_in_private_scratch(
    request, tmp_path, monkeypatch
) -> None:
    run = _workspace_run(tmp_path, monkeypatch)
    note = roots_module.workspace_context_note(
        WORKSPACE_ID,
        binding_authority=tuple(run.authority.options),
        status_cache=None,
        scratch_relative_tools=("read_file", "list_directory"),
    )

    scratch_lines = [line for line in note.splitlines() if "private scratch" in line]
    assert len(scratch_lines) == 1
    assert "read_file" in scratch_lines[0]
    assert "list_directory" in scratch_lines[0]


@private_profile_test
def test_first_request_guides_project_exploration(request, tmp_path, monkeypatch) -> None:
    """Missing guidance must fail at the real first-request boundary."""
    run = _workspace_run(tmp_path, monkeypatch)
    provider = _local_provider(run)
    plan = _plan(run, provider)
    note = " ".join(plan.config.workspace_context_note.split())

    assert "before saying project information is unavailable" in note
    (alias,) = _advertised_aliases(provider)
    assert alias in note
    result = provider.invoke("fs_list", {"root_alias": alias, "path": "."})
    assert result.ok, result.error
    assert "myproj-README.md" in result.content


@pytest.mark.parametrize("read_tools", [(), ("git_status",)], ids=["empty", "git-only"])
@private_profile_test
def test_guidance_requires_workspace_read_tools(
    request, tmp_path, monkeypatch, read_tools
) -> None:
    run = _workspace_run(tmp_path, monkeypatch)
    (alias,) = _advertised_aliases(_local_provider(run))
    note = roots_module.workspace_context_note(
        WORKSPACE_ID,
        binding_authority=tuple(run.authority.options),
        path_tool_aliases=(alias,),
        workspace_read_tools=read_tools,
    )

    assert alias in note
    assert "before saying project information is unavailable" not in " ".join(note.split())


@private_profile_test
def test_goal_scope_removing_readers_removes_exploration_promise(
    request, tmp_path, monkeypatch
) -> None:
    from tldw_chatbook.Agents import automatic_work_runtime
    from tldw_chatbook.Agents.goal_models import GoalPolicy, GoalToolScope
    from tldw_chatbook.Chat.console_goal_runs import GoalIterationAuthorization

    run = _workspace_run(tmp_path, monkeypatch)
    # Exercise real scope narrowing without dispatching/accepting a durable goal.
    goal = object.__new__(GoalIterationAuthorization)
    goal.goal = SimpleNamespace(
        request=SimpleNamespace(
            tool_scope=GoalToolScope(catalog_tools=("local:git_status",))
        ),
        policy=GoalPolicy(),
    )
    monkeypatch.setattr(goal, "remaining_seconds", lambda: 60.0)
    monkeypatch.setattr(
        automatic_work_runtime, "current_automatic_work", lambda: SimpleNamespace(goal=goal)
    )
    plan = _plan(run, _local_provider(run))

    assert plan.allowed_tools == ("git_status",)
    assert not {"fs_list", "fs_glob", "fs_grep", "fs_read"}.intersection(plan.allowed_tools)
    note = " ".join(plan.config.workspace_context_note.split())
    assert "before saying project information is unavailable" not in note


@private_profile_test
def test_custom_operating_prompt_keeps_workspace_guidance(
    request, tmp_path, monkeypatch
) -> None:
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Internal_Prompts import resolver

    run = _workspace_run(tmp_path, monkeypatch)
    original = resolver._config_value

    def custom_config_value(key):
        if key == "internal_prompts.agents.console_agent_operating":
            return "CUSTOM OPERATING PROMPT"
        return original(key)

    monkeypatch.setattr(resolver, "_config_value", custom_config_value)
    plan = _plan(run, _local_provider(run))
    service = AgentService(AgentRunsDB(tmp_path / "runs.db"), plan.registry)
    model_request = service._build_model_request(
        plan.config,
        plan.api_endpoint,
        list(plan.schemas.runtime_schemas),
        plan.messages,
        plan.schemas.active_schemas,
    )
    prompt = " ".join(model_request.messages[0]["content"].split())

    assert "CUSTOM OPERATING PROMPT" in prompt
    assert "before saying project information is unavailable" in prompt


@private_profile_test
async def test_index_scope_preparation_runs_off_loop_and_note_tracks_final_tools(
    tmp_path, request, monkeypatch
):
    import threading
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Workspaces.project_index import runtime
    from tldw_chatbook.Workspaces.project_index.models import (
        ProjectIndexKey,
        ProjectIndexScope,
        ProjectIndexStatus,
        locator_fingerprint,
        root_identity_fingerprint,
    )
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

    run = _workspace_run(tmp_path, monkeypatch)
    admitted = capture_run_admitted_workspace_roots(
        session=run.session, registry=run.service
    )
    main = threading.current_thread()
    seen = []

    def key_for(app, workspace, binding, root):
        seen.append(threading.current_thread())
        return ProjectIndexKey(
            "data",
            workspace,
            binding,
            locator_fingerprint(root),
            root_identity_fingerprint(capture_directory_chain(root)),
        )

    from tldw_chatbook.Workspaces.project_index.runtime import project_index_registry_connections
    service = SimpleNamespace(
        registry_connection_scope=lambda: project_index_registry_connections(run.service.db),
        status=lambda key: ProjectIndexStatus(
            ProjectIndexScope(key, 1, 1), "Ready", 1, 1, 1, None, None, 0, 0
        )
    )
    monkeypatch.setattr(runtime, "get_project_index_service", lambda app: service)
    monkeypatch.setattr(runtime, "project_index_key", key_for)
    controller = object.__new__(ConsoleChatController)
    controller.app = SimpleNamespace(
        workspace_registry_service=run.service, unified_mcp_service=None
    )
    controller.store = SimpleNamespace(sessions=lambda: [run.session])

    async def mcp(*args, **kwargs):
        return None

    captured = {}

    def local(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    controller._compose_mcp_provider = mcp
    controller._compose_local_provider = local
    await controller._compose_agent_request_providers(
        session_id=run.session.id,
        project_selection=None,
        project_authority_guard=None,
        admitted_roots=admitted,
    )
    assert seen and all(thread is not main for thread in seen)
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.participants import _retire_current_thread_caches
    import time

    pause = storage._begin_local_pause()
    try:
        _retire_current_thread_caches(pause)
        assert pause.drain(time.monotonic() + 2), "scope-preparation thread leaked registry admission"
    finally:
        pause.resume()
    assert captured["project_index_service"] is service
    scopes = captured["project_index_scopes"]
    p = LocalToolProvider(
        workspace_root=tmp_path,
        admitted_roots=admitted,
        project_index_service=service,
        project_index_scopes=scopes,
    )
    plan = _plan(run, p)
    assert "fs_search_index" in plan.config.workspace_context_note
    p._specs.pop("fs_search_index")
    assert "fs_search_index" not in _plan(run, p).config.workspace_context_note


@pytest.mark.parametrize("automatic", [0, 1], ids=["never", "automatic"])
@pytest.mark.parametrize("access", [0, 1], ids=["blocked", "allowed"])
@private_profile_test
async def test_enabled_index_preserves_saved_library_policy_and_scope(
    tmp_path, request, monkeypatch, automatic, access
):
    import asyncio
    import time
    from Tests.Chat.test_console_workspace_guidance import _project_case, _policy_rows
    from tldw_chatbook.Workspaces.project_index.models import ProjectIndexPolicy
    from tldw_chatbook.Workspaces.project_index.runtime import (
        get_project_index_service,
        project_index_key,
        close_project_index_service,
    )

    case = await _project_case(
        tmp_path, monkeypatch, native=True, automatic=automatic, access=access
    )
    try:
        app = case.controller.app
        service = await asyncio.to_thread(get_project_index_service, app)
        key = await asyncio.to_thread(
            project_index_key, app, "owned", case.aliases["owned"], tmp_path / "owned"
        )

        def build():
            status = service.status(key)
            status = service.enable(
                key, ProjectIndexPolicy(("",), (".md",)), expected_scope=status.scope
            )
            service.refresh(
                key,
                expected_scope=status.scope,
                expected_job_revision=status.job_revision,
            )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if service.status(key).state == "Ready":
                    return
                time.sleep(0.02)
            pytest.fail("index did not become Ready")

        await asyncio.to_thread(build)
        before = _policy_rows(case)
        scope = case.session.rag_scope_holder.scope
        case.registry.set_active_workspace("visible")
        case.gateway.release.set()
        result = await case.controller.submit_draft(
            "Summarize this project", session_id=case.session.id
        )
        assert result.accepted, result.visible_copy
        payload = "\n".join(
            str(row.get("content", "")) for row in case.gateway.messages_seen[0]
        )
        assert "fs_search_index" in payload
        assert case.aliases["owned"] in payload
        assert _policy_rows(case) == before
        assert case.session.rag_scope_holder.scope == scope
        assert len(case.service.calls) == automatic
    finally:
        await close_project_index_service(case.controller.app)
        await case.controller.shutdown()
        case.runs.close()
        case.db.close()
        case.workspace_db.close()
