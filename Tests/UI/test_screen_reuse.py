"""Screen-instance reuse contracts for reusable routes (TASK-24452).

The app hands ``switch_screen`` a fresh instance on every navigation --
except routes whose ``ScreenRoute.reusable`` flag is set, whose screen is
constructed once, installed, and re-switched to on later visits (Textual
suspends installed screens instead of unmounting them). These tests pin:

1. the reuse itself (same instance across a leave-and-return cycle, no
   widget re-mint) -- the warm-visit guarantee the flag exists for;
2. per-visit dashboard refresh moving to ``on_screen_resume``;
3. runtime-identity scoping (a local<->server flip must not resume the
   other identity's live widget state);
4. the opt-in boundary: a route WITHOUT the flag keeps fresh instances,
   because reuse changes on_mount/on_unmount frequency and every route
   must opt in only after auditing that (see the flag's docstring).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from Tests.console_resource_fixtures import (
    close_owned_console_resources as close_owned_console_resources,
    close_owned_console_test_apps as close_owned_console_test_apps,
)
from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route


def _build_test_app():
    from tldw_chatbook.app import TldwCli

    return TldwCli()


@pytest.fixture(autouse=True)
def close_owned_real_app_notifications(
    request, monkeypatch, close_owned_console_resources, close_owned_console_test_apps
):
    """Register the additional notification store owned by each real app."""
    build_app = request.module._build_test_app

    def build_owned_app():
        app = build_app()
        close_owned_console_resources.callback(app.client_notifications_db.close)
        return app

    monkeypatch.setattr(request.module, "_build_test_app", build_owned_app)


def _scratch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point every config/data seam at a scratch tree with setup completed."""
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
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "screen_reuse")
    return home


async def _boot_settled(app):
    while not getattr(app, "_ui_ready", False):
        await asyncio.sleep(0.01)


async def _settle(pilot, passes: int = 6, interval: float = 0.05) -> None:
    for _ in range(passes):
        await asyncio.sleep(interval)
        await pilot.pause()


async def _press_until_screen(pilot, key: str, expected: str) -> None:
    deadline = asyncio.get_running_loop().time() + 30.0
    await pilot.press(key)
    while asyncio.get_running_loop().time() < deadline:
        await pilot.pause()
        if type(pilot.app.screen).__name__ == expected:
            break
    assert type(pilot.app.screen).__name__ == expected, (
        f"never arrived at {expected}; stuck on "
        f"{type(pilot.app.screen).__name__}"
    )
    await _settle(pilot)


def test_home_route_is_flagged_reusable() -> None:
    """The registry carries the enablement this suite exercises."""
    route = resolve_screen_route("home")
    assert route is not None and route.reusable is True


@pytest.mark.ui
@pytest.mark.asyncio
async def test_reusable_route_returns_the_same_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Leave-and-return to Home resumes the SAME screen instance.

    Also pins the anti-goal of the whole task: a warm visit must not
    re-mint the widget tree, so the tree size is compared across visits
    (a re-mint would build a second full tree before the first's removal).
    """
    _scratch_env(monkeypatch, tmp_path)

    app = _build_test_app()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app)
        await _settle(pilot, passes=20)

        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        first_home = app.screen
        first_widget_count = len(list(first_home.walk_children()))

        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        assert first_home.is_attached is False or first_home is not app.screen

        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        second_home = app.screen
        assert second_home is first_home, (
            "Home is a reusable route: returning to it must resume the "
            "installed instance, not construct a new one"
        )
        second_widget_count = len(list(second_home.walk_children()))
        assert second_widget_count <= first_widget_count * 1.5, (
            "warm Home visit grew the widget tree "
            f"({first_widget_count} -> {second_widget_count}) -- "
            "re-minting is what reuse exists to prevent"
        )


@pytest.mark.ui
@pytest.mark.asyncio
async def test_resume_retriggers_home_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Returning to Home re-runs the per-visit refreshers.

    With reuse, ``on_mount`` fires once per app run -- ``on_screen_resume``
    is what keeps revisits from showing the previous visit's dashboard.
    Deleting Home's resume hook must fail THIS test, not a user report.
    """
    _scratch_env(monkeypatch, tmp_path)

    app = _build_test_app()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app)
        await _settle(pilot, passes=20)

        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        home = app.screen

        calls: list[str] = []
        monkeypatch.setattr(
            home,
            "_refresh_home_active_work_cache",
            lambda: calls.append("active-work"),
        )
        monkeypatch.setattr(
            home,
            "_refresh_home_content_snapshot",
            lambda: calls.append("content"),
        )
        monkeypatch.setattr(
            home,
            "_refresh_home_chatbook_artifact_snapshot",
            lambda: calls.append("artifact"),
        )

        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")
        assert not calls, "suspending Home must not trigger its refreshers"
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        assert set(calls) == {"active-work", "content", "artifact"}, (
            f"resume re-triggered {sorted(set(calls))}; expected all three "
            "per-visit refreshers"
        )


@pytest.mark.ui
@pytest.mark.asyncio
async def test_identity_flip_invalidates_the_cached_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cached instance is scoped to the runtime identity that built it."""
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.UI.Navigation.screen_state_store import RuntimeIdentity

    app = _build_test_app()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app)
        await _settle(pilot, passes=20)

        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        cached_home = app.screen
        await _press_until_screen(pilot, "ctrl+2", "ChatScreen")

        other_identity = RuntimeIdentity(
            active_source="server", active_server_id="scope-flip-probe"
        )
        assert (
            app._reusable_navigation_screen("home", other_identity) is None
        ), "an identity flip must not hand back the other identity's screen"
        assert "home" not in getattr(app, "_reusable_screen_instances", {}), (
            "the stale entry must leave the cache, not linger for the next "
            "same-identity lookup to resurrect"
        )

        # And the next navigation builds a FRESH instance rather than
        # resuming the dropped one.
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        assert app.screen is not cached_home


@pytest.mark.ui
@pytest.mark.asyncio
async def test_non_reusable_route_still_gets_fresh_instances(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reuse stays opt-in: an unflagged route constructs per visit.

    Settings is the probe (Console was, until its TASK-31520 audit flipped
    it): flipping a route to reuse without dispositioning its
    ``on_unmount`` teardown is exactly the change this guard makes loud.
    """
    _scratch_env(monkeypatch, tmp_path)

    route = resolve_screen_route("settings")
    assert route is not None and route.reusable is False

    app = _build_test_app()
    async with app.run_test(size=(170, 48)) as pilot:
        await _boot_settled(app)
        await _settle(pilot, passes=20)

        await _press_until_screen(pilot, "f9", "SettingsScreen")
        first_settings = app.screen
        await _press_until_screen(pilot, "ctrl+1", "HomeScreen")
        await _press_until_screen(pilot, "f9", "SettingsScreen")
        assert app.screen is not first_settings
