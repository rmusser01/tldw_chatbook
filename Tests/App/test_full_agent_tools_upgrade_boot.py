"""A real app boot runs the one-time Full agent tools upgrade (TASK-34100.5.3.2).

The unit suite (``Tests/Agents/test_full_agent_tools_upgrade.py``) pins the
decision; this file pins the wiring: the app's staggered boot worker runs it
after the UI is ready, against the app's own workspaces database, the runs
database the app finds beside its own ChaChaNotes file, and the profile's
real config file, and mirrors a written key into ``app_config``.

Every test waits for the boot's UI to settle before leaving ``run_test``
(review round 1, F1-1): leaving as soon as the marker or the key landed left
the Console still mounting, and under load teardown raised
``NoMatches: 'SelectOverlay' on Select(id='recovery-scope')`` -- the
early-exit race recorded for TASK-32632/TASK-32794.
"""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app

MARKER = "full_agent_tools_upgrade"


def _raw_config() -> dict:
    from tldw_chatbook.config import read_cli_config_snapshot

    serialized = read_cli_config_snapshot().serialized
    return tomllib.loads(serialized or "")


def _config_bytes() -> bytes:
    from tldw_chatbook.config import get_cli_config_path

    return Path(get_cli_config_path()).read_bytes()


def _profile_dir(tmp_path: Path) -> Path:
    data = tmp_path / "profile-data"
    data.mkdir(mode=0o700)
    return data.resolve(strict=True)


def _marker_recorded(data: Path) -> bool:
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    db = WorkspaceDB(data / "workspaces.sqlite", client_id="upgrade-probe")
    try:
        return db.is_agent_backfill_complete(MARKER)
    finally:
        db.close()


def _app_with_file_backed_chachanotes(data: Path):
    """A factory app whose ChaChaNotes is a real file in ``data``.

    The factory patches ChaChaNotes to ``None``; a real profile has a file,
    and the Console's runs database is ``agent_runs.db`` beside it -- the
    path the upgrade must find (review round 1, F2-2). Without a ChaChaNotes
    file the upgrade cannot locate the runs database and fails closed (F3-4).
    """
    from Tests.UI.test_console_fleet_wake_wiring import _attached_real_databases
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    app = _build_test_app(user_data_dir=data)
    chachanotes = CharactersRAGDB(str(data / "ChaChaNotes.db"), "upgrade-boot-test")
    # Retired by the shared async owner fixture: runtimes first, then the DB.
    _attached_real_databases.append((app, chachanotes))
    app.chachanotes_db = chachanotes
    return app


async def _until(pilot, predicate, *, seconds: float = 30.0) -> bool:
    for _ in range(int(seconds / 0.05)):
        if predicate():
            return True
        await pilot.pause(0.05)
    return predicate()


async def _settle_boot_ui(app, pilot, *, seconds: float = 60.0) -> None:
    """Wait until the boot has finished mounting what it mounts.

    The Console composes some controls after first paint. A widget a query
    can find has not necessarily composed its own children
    (lessons-textual, TASK-36382), so the bar is ``is_mounted`` on every
    ``Select`` on screen -- the widget whose half-composed teardown raised --
    and a widget count that has stopped changing, held across several polls.
    """
    from textual.widgets import Select

    def snapshot() -> tuple[bool, int]:
        screen = app.screen
        selects = list(screen.query(Select))
        mounted = bool(getattr(app, "_ui_ready", False)) and all(
            select.is_mounted for select in selects
        )
        return mounted, len(screen.query("*"))

    stable, last_count = 0, -1
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while loop.time() < deadline:
        mounted, count = snapshot()
        stable = stable + 1 if mounted and count == last_count else 0
        last_count = count
        if stable >= 6:
            break
        await pilot.pause(0.1)
    assert stable >= 6, "the boot UI never settled"
    await pilot.pause()


async def _boot_and_settle(app, pilot, predicate, what: str) -> None:
    assert await _until(pilot, predicate), what
    await _settle_boot_ui(app, pilot)


@pytest.mark.asyncio
@private_profile_test
async def test_an_upgraded_profile_with_a_workspace_binding_boots_with_full_agent_tools(
    request, tmp_path
):
    from tldw_chatbook.Agents.plain_chat_tool_surface import light_surface_configured
    from tldw_chatbook.config import save_setting_to_cli_config
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import (
        LocalWorkspaceRegistryService,
    )

    # A profile from before the light surface: a saved config without the
    # key, and a workspace with a folder binding.
    assert save_setting_to_cli_config("splash_screen", "enabled", False)
    data = _profile_dir(tmp_path)
    seed = WorkspaceDB(data / "workspaces.sqlite", client_id="seed")
    try:
        registry = LocalWorkspaceRegistryService(seed)
        registry.create_workspace(workspace_id="ws-project", name="Project")
        folder = tmp_path / "project"
        folder.mkdir()
        registry.add_folder_binding("ws-project", folder)
        # Review v5r1 (F1): a real pre-upgrade profile already has the
        # workspace agent backfill's own row in the same table.
        seed.mark_agent_backfill_complete()
    finally:
        seed.close()
    assert "full_agent_tools_enabled" not in _raw_config().get("tools", {})
    assert light_surface_configured() is True

    app = _app_with_file_backed_chachanotes(data)
    async with app.run_test() as pilot:
        await _boot_and_settle(
            app,
            pilot,
            lambda: app.app_config.get("tools", {}).get(
                "full_agent_tools_enabled"
            )
            is True,
            "the written key never reached app.app_config",
        )

    assert _raw_config()["tools"]["full_agent_tools_enabled"] is True
    assert light_surface_configured() is False
    assert _marker_recorded(data)


@pytest.mark.asyncio
@private_profile_test
async def test_a_sub_agent_definition_beside_the_apps_chachanotes_keeps_full_agent_tools(
    request, tmp_path
):
    """Review round 1 (F2-2): three of the four signals reach production only
    through the runs database the app finds beside its ChaChaNotes file."""
    from tldw_chatbook.Agents.agent_models import AgentDefinition
    from tldw_chatbook.Agents.plain_chat_tool_surface import light_surface_configured
    from tldw_chatbook.config import save_setting_to_cli_config
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    assert save_setting_to_cli_config("splash_screen", "enabled", False)
    data = _profile_dir(tmp_path)
    runs = AgentRunsDB(data / "agent_runs.db", client_id="seed")
    try:
        runs.create_agent_definition(
            AgentDefinition(
                name="researcher",
                description="Finds sources.",
                instructions="Search, then summarize.",
            )
        )
    finally:
        runs.close()

    app = _app_with_file_backed_chachanotes(data)
    async with app.run_test() as pilot:
        await _boot_and_settle(
            app,
            pilot,
            lambda: app.app_config.get("tools", {}).get(
                "full_agent_tools_enabled"
            )
            is True,
            "the boot never turned Full agent tools on for the definition",
        )

    assert _raw_config()["tools"]["full_agent_tools_enabled"] is True
    assert light_surface_configured() is False
    assert _marker_recorded(data)


@pytest.mark.asyncio
@private_profile_test
async def test_a_profile_without_agent_use_boots_light_and_its_config_is_not_rewritten(
    request, tmp_path
):
    from tldw_chatbook.Agents.plain_chat_tool_surface import light_surface_configured
    from tldw_chatbook.config import save_setting_to_cli_config

    assert save_setting_to_cli_config("splash_screen", "enabled", False)
    data = _profile_dir(tmp_path)
    before = _config_bytes()

    app = _app_with_file_backed_chachanotes(data)
    async with app.run_test() as pilot:
        await _boot_and_settle(
            app,
            pilot,
            lambda: _marker_recorded(data),
            "the boot never recorded that the upgrade ran",
        )

    assert _config_bytes() == before
    assert "full_agent_tools_enabled" not in _raw_config().get("tools", {})
    assert light_surface_configured() is True
