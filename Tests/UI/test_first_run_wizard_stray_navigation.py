"""First-run wizard must survive a stray navigation (TASK-31807).

Regression for a release-gate defect: with the splash screen enabled, the
setup wizard occasionally mounted and then self-dismissed to Home with zero
user input, leaving ``setup_started`` persisted so the user was neither shown
setup nor re-offered it cleanly.

Root cause (established by reproduction, see the task's Implementation Notes):
the wizard is a modal pushed over the initial screen via a
``call_after_refresh``. During splash teardown the app's global key bindings
are already live on the just-mounted initial screen, so a shell-destination
key (F9=settings, F10=research, ctrl+N... -> ``action_shell_destination``)
that leaks in -- the splash's own ``on_key`` consumes only the FIRST key --
posts a ``NavigateToScreen``. Handling that navigation runs
``_dismiss_navigation_overlays``, which ``dismiss(None)``s the wizard: a
navigation-driven teardown of onboarding that no user asked for.

These tests drive the real app through ``run_test`` and fire that exact
trigger (both the shell-destination action a leaked key invokes and a raw
posted ``NavigateToScreen``). Before the fix the wizard was torn down to the
initial screen; after it, the navigation is ignored and the wizard stays up.
Splash is left disabled here on purpose: the defect is navigation-driven, so
it reproduces without the (untestable-in-run_test) splash timing -- the splash
is only how the stray key gets its window in the field.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen


def _prepare_clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_var, path_name in (
        ("HOME", "home"),
        ("XDG_CONFIG_HOME", "xdg-config"),
        ("XDG_DATA_HOME", "xdg-data"),
        ("XDG_CACHE_HOME", "xdg-cache"),
    ):
        path = tmp_path / path_name
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(env_var, str(path))


def _splash_off_cli_setting(section: str, key: str, default=None):
    if section == "splash_screen" and key == "enabled":
        return False
    return default


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 30.0,
    interval_seconds: float = 0.02,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition not met within {timeout_seconds:.1f}s")


def _fresh_wizard_app():
    app = _build_test_app(first_run_setup_completed=False)
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    return app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trigger",
    ["posted_navigate", "shell_destination_key"],
    ids=["raw-NavigateToScreen", "leaked-shell-destination-key"],
)
async def test_stray_navigation_does_not_dismiss_first_run_wizard(
    trigger: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A navigation arriving while the wizard is up must not dismiss it.

    Fires the stray navigation two ways: a raw ``NavigateToScreen`` post and
    the ``action_shell_destination`` a leaked F9 keypress invokes. Both must
    leave the wizard mounted (the guard ignores them); before the fix either
    tore the wizard down to Home.

    Args:
        trigger: Which stray-navigation path to exercise ("posted_navigate"
            or "shell_destination_key").
        monkeypatch: Pytest fixture used to sandbox the config lookups.
        tmp_path: Pytest fixture providing the clean HOME/XDG sandbox dirs.
    """
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _fresh_wizard_app()

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_splash_off_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            # The wizard is up over Home; onboarding has begun.
            assert app.current_tab == "home"

            # Fire the stray navigation the way the field does.
            if trigger == "posted_navigate":
                app.post_message(NavigateToScreen("settings"))
            else:
                # Exactly what a leaked F9 keypress runs.
                app.action_shell_destination("settings")

            # Give any navigation worker ample time to run.
            for _ in range(60):
                await pilot.pause(0.02)
                if type(app.screen).__name__ != "FirstRunSetupWizard":
                    break

            # The wizard must still be up -- the stray navigation is ignored,
            # not allowed to tear onboarding down.
            assert type(app.screen).__name__ == "FirstRunSetupWizard", (
                f"stray navigation ({trigger}) dismissed the wizard to "
                f"{type(app.screen).__name__}"
            )
            assert app.current_tab == "home"


@pytest.mark.asyncio
async def test_navigation_resumes_once_the_wizard_leaves_the_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guard must not outlive the wizard: once it is off the stack, a
    navigation posted afterwards proceeds normally.

    Args:
        monkeypatch: Pytest fixture used to sandbox the config lookups.
        tmp_path: Pytest fixture providing the clean HOME/XDG sandbox dirs.
    """
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _fresh_wizard_app()

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_splash_off_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            # Dismiss the wizard the way its own controls do (direct dismiss),
            # then confirm a subsequent navigation is honored.
            app.screen.dismiss(None)
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            app.post_message(NavigateToScreen("settings"))
            await _wait_until(
                pilot, lambda: app.current_tab == "settings"
            )
            assert app.current_tab == "settings"


# ---------------------------------------------------------------------------
# Isolated unit tests for the guard predicate in
# TldwCli._handle_screen_navigation_locked (no app.run_test): pin the
# stack-scan + silent-False contract directly, independent of the full app.
# ---------------------------------------------------------------------------


class _StubScreen:
    """A minimal stand-in for a screen on the navigation stack.

    Args:
        blocks: When True, carries ``blocks_stray_navigation = True`` exactly
            as ``FirstRunSetupWizard`` does; otherwise the attribute is absent
            (matching an ordinary screen).
    """

    def __init__(self, blocks: bool = False) -> None:
        if blocks:
            self.blocks_stray_navigation = True


class _StubApp:
    """Duck-typed host exposing only what the guard reads before it acts.

    Args:
        stack: The value returned for ``_screen_stack``.
    """

    _initial_screen_pushed = True

    def __init__(self, stack: list[object]) -> None:
        self._screen_stack = stack
        self.notify = MagicMock()
        self._resolve_screen_navigation_target = MagicMock()


@pytest.mark.asyncio
async def test_guard_returns_false_and_does_not_resolve_when_gate_on_stack() -> None:
    """A blocking gate anywhere on the stack short-circuits to a silent False.

    Confirms the guard neither resolves a navigation target nor notifies the
    user -- the stray navigation is ignored, not surfaced as a failure.
    """
    app = _StubApp([_StubScreen(), _StubScreen(blocks=True)])

    result = await TldwCli._handle_screen_navigation_locked(
        app, SimpleNamespace(screen_name="settings")
    )

    assert result is False
    app._resolve_screen_navigation_target.assert_not_called()
    app.notify.assert_not_called()


@pytest.mark.asyncio
async def test_guard_lets_navigation_through_when_no_gate_on_stack() -> None:
    """With no blocking gate, the guard passes control on to target resolution.

    Uses a sentinel raised from ``_resolve_screen_navigation_target`` to prove
    execution reached past the guard (rather than returning False early).
    """
    app = _StubApp([_StubScreen(), _StubScreen()])
    sentinel = RuntimeError("reached target resolution")
    app._resolve_screen_navigation_target.side_effect = sentinel

    with pytest.raises(RuntimeError, match="reached target resolution"):
        await TldwCli._handle_screen_navigation_locked(
            app, SimpleNamespace(screen_name="settings")
        )

    app._resolve_screen_navigation_target.assert_called_once_with("settings")
