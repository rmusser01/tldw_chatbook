"""Console screen-instance reuse contracts (TASK-31520).

The chat route is reusable: one ChatScreen per app run, suspended on
switch-away, resumed on return. These pin the behaviors the 2026-09-04
audit gated enablement on:

1. same-instance resume;
2. suspend stops the Console's per-visit timers (transcript-sync 0.2s,
   fleet-survivor 1s, cost-TTL 10s, draft-spend one-shot) -- Textual only
   auto-cancels timers on real removal;
3. runs survive navigation: the runtime view stays attached across a
   suspend, and `confirm_navigation` no longer gates a lossless switch;
4. resume re-arms what suspend quiesced (auto-speak wiring; the sync
   timer when a run is still in flight).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route


def _scratch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    data = tmp_path / "data"
    config = tmp_path / "config"
    for sub in (home, data, config):
        sub.mkdir(parents=True, exist_ok=True)
    config_file = config / "tldw_cli" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[first_run]\nsetup_completed = true\n\n[splash_screen]\nenabled = false\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_file))
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "console_screen_reuse")
    return home


async def _boot_settled(app, pilot) -> None:
    while not getattr(app, "_ui_ready", False):
        await asyncio.sleep(0.01)
    for _ in range(20):
        await asyncio.sleep(0.05)
        await pilot.pause()


async def _press_until_screen(pilot, key: str, expected: str) -> None:
    deadline = asyncio.get_running_loop().time() + 30.0
    await pilot.press(key)
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        if type(pilot.app.screen).__name__ == expected:
            break
    assert type(pilot.app.screen).__name__ == expected
    for _ in range(6):
        await asyncio.sleep(0.05)
        await pilot.pause()


def test_chat_route_is_flagged_reusable() -> None:
    route = resolve_screen_route("chat")
    assert route is not None and route.reusable is True


@pytest.mark.ui
@pytest.mark.asyncio
async def test_console_reuse_timer_quiescence_and_runtime_attachment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One journey pins reuse, timer quiescence, and runtime survival.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Pytest fixture providing the scratch profile root.
    """
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        runtime = console._console_runtime()
        assert runtime is not None and runtime.view is console

        # Arm the interval timers the way an active run would.
        console._start_console_transcript_sync_timer()
        console._start_console_cost_ttl_timer()
        assert console._console_transcript_sync_timer is not None

        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        assert console._console_transcript_sync_timer is None, (
            "suspend must stop the 0.2s transcript-sync poll -- Textual "
            "does not auto-cancel a suspended installed screen's timers"
        )
        assert console._console_cost_ttl_timer is None
        assert getattr(console._fleet, "_console_fleet_survivor_timer", None) is None
        # THE crux of the audit: the runtime view must stay attached --
        # a suspend-time detach permanently kills the prompt queue
        # (begin_visit never re-fires for a reused screen).
        assert runtime.view is console, (
            "leave_console_runtime must NOT run at suspend: runs and "
            "approvals survive navigation under reuse"
        )

        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        assert app.screen is console, (
            "chat is a reusable route: returning must resume the installed "
            "instance"
        )
        assert runtime.view is console


@pytest.mark.ui
@pytest.mark.asyncio
async def test_console_resume_restarts_sync_timer_for_active_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run still in flight when the user returns gets its poll back.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Pytest fixture providing the scratch profile root.
    """
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        controller = console._console_chat_controller
        assert controller is not None

        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        assert console._console_transcript_sync_timer is None

        monkeypatch.setattr(controller, "in_flight_run_count", lambda: 1)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        assert console._console_transcript_sync_timer is not None, (
            "resume must restart the transcript-sync poll while a run is "
            "in flight -- no other path restarts it"
        )
        # And an idle console does NOT get a pointless poll on resume.
        monkeypatch.setattr(controller, "in_flight_run_count", lambda: 0)
        console._stop_console_transcript_sync_timer()
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        assert console._console_transcript_sync_timer is None


@pytest.mark.ui
@pytest.mark.asyncio
async def test_console_confirm_navigation_no_longer_gates_switches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Tab switches are lossless under reuse; quit still confirms.

    Args:
        monkeypatch: Pytest fixture for scoped attribute patching.
        tmp_path: Pytest fixture providing the scratch profile root.
    """
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app, pilot)
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        console = app.screen
        assert await console.confirm_navigation() is True, (
            "leaving Console cancels nothing under reuse -- the busy-fleet "
            "navigation dialog gates a lossless action"
        )
        # The quit path still delegates to the loss confirmation.
        import inspect as _inspect

        source = _inspect.getsource(type(console).confirm_quit)
        assert "confirm_quit" in source and "_session" in source, (
            "confirm_quit must keep delegating -- app exit really does "
            "cancel everything"
        )
