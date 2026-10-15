"""Canonical Save keeps literal argv and its consent identity unchanged."""

import json
import tomllib

import pytest
from textual.widgets import TextArea

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _wait_for_selector,
)
from tldw_chatbook.Agents.hook_permissions import HookPermissions
from tldw_chatbook.Agents.run_hooks import HookSpec, fingerprint_hook
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

hook_file = _hook_file

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_canonical_hooks_save_preserves_literal_x64_argv_and_fingerprint(
    hook_file,
):
    command = [
        r"C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe",
        r"--literal=\x41",
        "-c",
        "pass",
    ]
    expected = fingerprint_hook(
        HookSpec(event="PostToolUse", command=tuple(command), timeout_s=5)
    )
    host = DestinationHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 40)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({"category": "hooks"})
        await _wait_for_selector(screen, pilot, "#settings-hooks-command")
        before = hook_file.read_bytes()
        screen.query_one("#settings-hooks-command", TextArea).load_text(
            json.dumps(command)
        )
        await pilot.pause()
        assert hook_file.read_bytes() == before
        screen.query_one("#settings-hooks-save").focus()
        await pilot.press("enter")
        await host.workers.wait_for_complete()
        await pilot.pause()
        saved = tomllib.loads(hook_file.read_text())["hooks"]["hook"][0]
        assert saved["command"] == command
        assert not screen._category_has_unsaved_changes(SettingsCategoryId.HOOKS)
        snapshot = HookPermissions().snapshot()
        rows = [row for row in snapshot.rows if row.entry and row.entry.spec]
        assert len(rows) == 1
        assert rows[0].entry.spec.command == tuple(command)
        assert fingerprint_hook(rows[0].entry.spec) == expected
        assert rows[0].state == "pending"
