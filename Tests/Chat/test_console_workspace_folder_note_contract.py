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
    # Same precedent as Tests/Agents/test_local_tool_provider.py's
    # ``_default_specs_without_config_reads``: provider construction reads the
    # web-deep-search gate through get_cli_setting, which trips the known
    # sandboxed-config Backup_Recovery failure. Pin every gate to its default.
    def _default(_section, _key=None, default=None):
        return default

    monkeypatch.setattr(
        "tldw_chatbook.Agents.local_tool_provider.get_cli_setting", _default
    )
    # The plan's prompt resolver reads config lazily through the module.
    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", _default)
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
def test_every_folder_the_note_names_is_listed_by_fs_list_with_the_alias_it_gives(
    tmp_path, monkeypatch, names, inside_launch
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


def test_note_never_frames_paths_relative_to_the_launch_directory(
    tmp_path, monkeypatch
) -> None:
    run = _workspace_run(tmp_path, monkeypatch)
    note = _plan(run, _local_provider(run)).config.workspace_context_note

    assert "launch directory" not in note
    assert "Launched from" not in note
    assert str(run.tmp_path) not in note  # never an absolute host path


def test_note_lists_only_the_working_folder_when_a_selection_narrows_the_run(
    tmp_path, monkeypatch
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


def test_note_says_bound_folders_need_path_tools_when_the_run_has_none(
    tmp_path, monkeypatch
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


def test_note_says_builtin_relative_paths_resolve_in_private_scratch(
    tmp_path, monkeypatch
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
