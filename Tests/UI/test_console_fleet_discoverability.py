"""Fleet-UX expert review F2 (task-1232): parallel-agents discoverability.

At rest, nothing communicated that each Console tab runs its own agent, in
parallel, under a cap: F1 Help covered panes/transcript/composer shortcuts
only (zero mentions of "agent"/"parallel"/"approval"/"workspace"), the
footer omitted Alt+W/Alt+1..9, and the capability only taught itself after
the user had already, accidentally, run two agents at once.

This file covers the three fixes:
  1. F1 Help gains an "Agents" section (tabs=agents, live cap, approval
     flow, marker legend, hotkeys, screen-scope caveat).
  2. A one-time dismissible coach-mark on the first second-tab creation.
  3. Alt+W/Alt+1..9 reachable from Help (the footer is a single-line,
     non-wrapping Static already ~120 chars for its 7 existing hints at
     the narrowest width this suite tests Console against -- see the
     "Agents & fleet" shortcut group's docstring in chat_screen.py for the
     full crowding judgment).
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_FLEET_MARKER_LEGEND,
    CONSOLE_WORKBENCH_SHORTCUT_GROUPS,
    _console_workbench_agents_notes,
)
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


async def _wait_for_active_session_change(
    store, pilot, previous_session_id, *, attempts: int = 40
) -> str:
    """Wait for the Console store to activate a session other than
    `previous_session_id` and return its id."""
    for _ in range(attempts):
        active_session_id = store.active_session_id
        if active_session_id is not None and active_session_id != previous_session_id:
            return active_session_id
        await pilot.pause(0.05)
    raise AssertionError(
        "Console active session did not change. "
        f"previous={previous_session_id!r}; active={store.active_session_id!r}"
    )


# --- AC#1: F1 Help gains an "Agents" section -------------------------------


def test_console_workbench_agents_notes_covers_required_content():
    """The pure content builder covers every AC#1 bullet."""
    notes = _console_workbench_agents_notes(7)
    joined = " ".join(notes)

    assert "own agent" in joined
    assert "background" in joined
    assert "7 runs in parallel" in joined
    assert "Settings > Console Behavior" in joined
    assert "ask before running" in joined
    assert "◆" in joined
    assert CONSOLE_FLEET_MARKER_LEGEND in notes
    assert "● running" in CONSOLE_FLEET_MARKER_LEGEND
    assert "◆ needs approval" in CONSOLE_FLEET_MARKER_LEGEND
    assert "✓ finished" in CONSOLE_FLEET_MARKER_LEGEND
    assert "✗ failed" in CONSOLE_FLEET_MARKER_LEGEND
    assert "Leaving Console cancels" in joined


def test_console_help_map_includes_agents_and_fleet_hotkeys():
    """Alt+W and Alt+1..9 (task-1232 AC#1/#3) are reachable from F1 Help."""
    groups = dict(CONSOLE_WORKBENCH_SHORTCUT_GROUPS)
    assert "Agents & fleet" in groups
    fleet_shortcuts = dict(groups["Agents & fleet"])
    assert fleet_shortcuts["Alt+W"] == "switch workspace"
    assert fleet_shortcuts["Alt+1..9"] == "jump to tab 1-9"
    assert "Ctrl+T" in fleet_shortcuts
    assert "Ctrl+K" in fleet_shortcuts


@pytest.mark.asyncio
async def test_console_f1_help_agents_section_reads_the_live_parallel_cap(
    monkeypatch,
):
    """End-to-end: F1 opens a help panel whose body reads the LIVE cap, not
    the baked-in default, and covers every required Agents-section bullet.
    """
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        controller = console._ensure_console_chat_controller()
        # A distinctive, non-default cap proves the help copy is read live
        # (the default is 3 -- see CONSOLE_DEFAULT_MAX_PARALLEL_RUNS).
        monkeypatch.setattr(
            type(controller), "max_parallel_runs", property(lambda self: 7)
        )
        await console.action_show_workbench_help()
        await pilot.pause()

        assert isinstance(host.screen_stack[-1], WorkbenchHelpPanel)
        body = _static_text(
            host.screen_stack[-1].query_one("#workbench-help-body", Static)
        )
        assert "Agents:" in body
        assert "Each Console tab runs its own agent" in body
        assert "7 runs in parallel" in body
        assert "Settings > Console Behavior" in body
        assert "Built-in tools ask before running" in body
        assert CONSOLE_FLEET_MARKER_LEGEND in body
        assert "Leaving Console cancels" in body
        assert "Alt+W" in body and "switch workspace" in body
        assert "Alt+1..9" in body and "jump to tab 1-9" in body


# --- AC#2: one-time coach-mark on first second-tab creation ----------------


@pytest.mark.asyncio
async def test_fleet_coachmark_shows_on_first_second_tab_creation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session_a_id = store.active_session_id

        banner = console.query_one("#console-fleet-coachmark")
        assert banner.display is False

        # Real "new tab" action -- mirrors how a user opens a second tab.
        await pilot.click("#console-new-chat-tab")
        await _wait_for_active_session_change(store, pilot, session_a_id)
        await pilot.pause()

        assert banner.display is True
        text = _static_text(console.query_one("#console-fleet-coachmark-text", Static))
        assert "own agent" in text
        assert "3 in parallel" in text  # CONSOLE_DEFAULT_MAX_PARALLEL_RUNS
        assert "Settings > Console Behavior" in text


@pytest.mark.asyncio
async def test_fleet_coachmark_dismiss_hides_banner_and_persists_flag_in_memory():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session_a_id = store.active_session_id

        await pilot.click("#console-new-chat-tab")
        await _wait_for_active_session_change(store, pilot, session_a_id)
        await pilot.pause()
        banner = console.query_one("#console-fleet-coachmark")
        assert banner.display is True

        await pilot.click("#console-fleet-coachmark-dismiss")
        await pilot.pause()

        assert banner.display is False
        onboarding = app.app_config.get("console", {}).get("onboarding", {})
        assert onboarding.get("fleet_coachmark_seen") is True
        assert console._console_fleet_coachmark_seen() is True


@pytest.mark.asyncio
async def test_fleet_coachmark_does_not_reappear_for_a_third_tab_after_dismiss():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        session_a_id = store.active_session_id

        await pilot.click("#console-new-chat-tab")
        session_b_id = await _wait_for_active_session_change(
            store, pilot, session_a_id
        )
        await pilot.pause()
        await pilot.click("#console-fleet-coachmark-dismiss")
        await pilot.pause()
        banner = console.query_one("#console-fleet-coachmark")
        assert banner.display is False

        await pilot.click("#console-new-chat-tab")
        await _wait_for_active_session_change(store, pilot, session_b_id)
        await pilot.pause()

        assert banner.display is False


@pytest.mark.asyncio
async def test_fleet_coachmark_seen_flag_survives_restart_via_real_config_seam(
    monkeypatch, tmp_path
):
    """Drives the REAL config seam (not a mocked in-memory dict): dismiss
    persists to an actual TOML file via `save_setting_to_cli_config`, and a
    fresh "restarted" screen reading that same file never shows the banner.
    """
    from tldw_chatbook import config as config_module

    config_path = tmp_path / "fleet-coachmark-config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_settings(force_reload=True)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        app = _build_test_app()
        host = ConsoleHarness(app)

        async with host.run_test(size=(160, 48)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            session_a_id = store.active_session_id

            await pilot.click("#console-new-chat-tab")
            await _wait_for_active_session_change(store, pilot, session_a_id)
            await pilot.pause()
            assert console.query_one("#console-fleet-coachmark").display is True

            await pilot.click("#console-fleet-coachmark-dismiss")
            await pilot.pause()
            await pilot.app.workers.wait_for_complete()

        # Real disk read (force_reload bypasses the in-process cache) --
        # proves the write actually landed on the tmp config file, not just
        # in the in-memory app_config dict.
        fresh_config = config_module.load_settings(force_reload=True)
        onboarding = fresh_config.get("console", {}).get("onboarding", {})
        assert onboarding.get("fleet_coachmark_seen") is True

        # "Restart": a brand-new app/screen instance whose app_config is
        # seeded from the persisted disk config, exactly like real boot.
        restarted_app = _build_test_app()
        restarted_app.app_config = fresh_config
        restarted_host = ConsoleHarness(restarted_app)
        async with restarted_host.run_test(size=(160, 48)) as pilot:
            console = restarted_host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            store = console._ensure_console_chat_store()
            session_a_id = store.active_session_id

            await pilot.click("#console-new-chat-tab")
            await _wait_for_active_session_change(store, pilot, session_a_id)
            await pilot.pause()

            assert console.query_one("#console-fleet-coachmark").display is False
    finally:
        config_module.load_settings(force_reload=True)
        config_module.load_cli_config_and_ensure_existence(force_reload=True)
