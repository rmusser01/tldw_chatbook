"""A real app boot runs the one-time Full agent tools upgrade (TASK-34100.5.3.2).

The unit suite (``Tests/Agents/test_full_agent_tools_upgrade.py``) pins the
decision; this file pins the wiring: the app's staggered boot worker runs it
after the UI is ready, against the app's own workspaces database and the
profile's real config file, and mirrors a written key into ``app_config``.
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


async def _settle(pilot, predicate, *, seconds: float = 15.0) -> bool:
    for _ in range(int(seconds / 0.05)):
        if predicate():
            return True
        await pilot.pause(0.05)
    return predicate()


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
    finally:
        seed.close()
    assert "full_agent_tools_enabled" not in _raw_config().get("tools", {})
    assert light_surface_configured() is True

    app = _build_test_app(user_data_dir=data)
    async with app.run_test() as pilot:
        written = await _settle(
            pilot,
            lambda: _raw_config().get("tools", {}).get("full_agent_tools_enabled")
            is True,
        )
        assert written, "the boot never wrote full_agent_tools_enabled = true"
        assert await _settle(
            pilot,
            lambda: app.app_config.get("tools", {}).get("full_agent_tools_enabled")
            is True,
            seconds=2.0,
        ), "the written key never reached app.app_config"

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

    app = _build_test_app(user_data_dir=data)
    async with app.run_test() as pilot:
        recorded = await _settle(pilot, lambda: _marker_recorded(data))
        await asyncio.sleep(0)

    assert _config_bytes() == before
    assert "full_agent_tools_enabled" not in _raw_config().get("tools", {})
    assert light_surface_configured() is True
    assert recorded, "the boot never recorded that the upgrade ran"
