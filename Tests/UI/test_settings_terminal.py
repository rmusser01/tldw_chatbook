"""Mounted Settings contracts for the persistent Terminal authority gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Checkbox, Static

from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_configuration_hub import StyledSettingsDestinationHarness
from Tests.UI.test_settings_raw_cli import (
    _open_privacy,
    _published_snapshot,
    _wait_for_save,
    _wait_until,
)
from tldw_chatbook.Terminal.session_manager import TerminalArmResult
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


TERMINAL_DISCLOSURE = (
    "The Terminal shell and every program run with the same OS permissions as "
    "Chatbook.",
    "Programs may read, modify, or delete any accessible data, use the network, "
    "invoke credentialed clients, or exhaust machine resources.",
    "Chatbook starts Terminal from a scrubbed environment, but normal shell profile "
    "and startup files run and may restore secrets, credentials, agents, proxies, "
    "aliases, environment variables, or arbitrary commands.",
    "Shells and programs may save history, files, logs, caches, and other side "
    "effects outside Chatbook.",
    "Your active workspace—or your home directory when no workspace is selected—is "
    "only the starting directory. Terminal is not sandboxed or confined there.",
    "Closing, disarming, or quitting Chatbook attempts bounded cleanup, but "
    "deliberately detached processes may survive and cleanup may remain unproven.",
    "Terminal content is user-only and is not sent to a model.",
)


class RecordingTerminalRuntime:
    """Small Settings spy; manager concurrency is covered by Terminal tests."""

    def __init__(self, *, armed: bool, session_count: int) -> None:
        self.armed = armed
        self.disclosure_acknowledged = armed
        self.arm_calls: list[bool] = []
        self.disarm_calls = 0
        self._projections = tuple(
            SimpleNamespace(session_id=f"session-{index}")
            for index in range(session_count)
        )

    def arm(self, *, acknowledge_disclosure: bool = False) -> TerminalArmResult:
        self.arm_calls.append(acknowledge_disclosure)
        if not self.disclosure_acknowledged and not acknowledge_disclosure:
            return TerminalArmResult(disclosure_required=True)
        self.disclosure_acknowledged = True
        self.armed = True
        return TerminalArmResult(armed=True)

    def disarm(self) -> None:
        self.disarm_calls += 1
        self.armed = False

    def projections(self) -> tuple[object, ...]:
        return self._projections


@pytest.mark.asyncio
async def test_terminal_control_uses_shared_unlock_and_an_independent_launch_arm():
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        terminal_button = screen.query_one("#settings-terminal-arm", Button)

        assert str(checkbox.label) == "Allow raw CLI and Terminal access on this device"
        assert "Unlocked, not armed" in str(
            screen.query_one("#settings-terminal-state", Static).content
        )
        assert str(terminal_button.label) == "Arm Terminal"
        assert terminal_button.disabled is False

        assert app.raw_cli_runtime.arm().armed is True
        screen._refresh_raw_cli_state()
        assert app.terminal_session_manager.armed is False
        assert "Unlocked, not armed" in str(
            screen.query_one("#settings-terminal-state", Static).content
        )

        terminal_button.press()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, lambda: host.screen is screen)

        assert app.terminal_session_manager.armed is True
        assert app.raw_cli_runtime.armed is True
        app.raw_cli_runtime.disarm()
        screen._refresh_raw_cli_state()
        assert app.terminal_session_manager.armed is True

    fresh_app = _build_test_app(
        config_overrides={"console": {"raw_cli_permitted": True}}
    )
    assert fresh_app.raw_cli_runtime.armed is False
    assert fresh_app.terminal_session_manager.armed is False
    assert fresh_app.terminal_session_manager.disclosure_acknowledged is False


@pytest.mark.asyncio
async def test_terminal_disclosure_is_static_once_per_launch_and_armed_state_is_red(
    monkeypatch,
):
    monkeypatch.setenv("TASK22512_TERMINAL_SECRET", "must-not-appear-in-disclosure")
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    real_arm = app.terminal_session_manager.arm
    arm_calls: list[bool] = []

    def tracked_arm(*, acknowledge_disclosure: bool = False):
        arm_calls.append(acknowledge_disclosure)
        return real_arm(acknowledge_disclosure=acknowledge_disclosure)

    monkeypatch.setattr(app.terminal_session_manager, "arm", tracked_arm)
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        terminal_button = screen.query_one("#settings-terminal-arm", Button)
        terminal_button.press()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))

        dialog = host.screen
        assert dialog.title == "Arm Terminal for this launch"
        assert dialog.message == "\n\n".join(TERMINAL_DISCLOSURE)
        assert "must-not-appear-in-disclosure" not in dialog.message
        confirm_button = dialog.query_one("#confirm-button", Button)
        assert confirm_button.region.width > 0
        assert dialog.screen.region.contains_region(confirm_button.region)
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, lambda: host.screen is screen)

        card = screen.query_one("#settings-raw-cli-card")
        assert arm_calls == [True]
        assert app.terminal_session_manager.armed is True
        assert card.has_class("settings-raw-cli-armed")
        error_color = card.query_one("#settings-raw-cli-state", Static).styles.color
        armed_border = card.styles.border
        assert all(
            kind == "solid" and color == error_color
            for kind, color in (
                armed_border.top,
                armed_border.right,
                armed_border.bottom,
                armed_border.left,
            )
        )
        assert card.styles.background == error_color.with_alpha(0.1)
        assert "ARMED — HOST TERMINAL" in str(
            screen.query_one("#settings-terminal-state", Static).content
        )
        assert "HOST TERMINAL - FULL USER ACCESS" in str(
            screen.query_one("#settings-terminal-host-access", Static).content
        )
        assert "terminal_armed" not in app.app_config.get("console", {})
        assert "terminal_armed" not in vars(screen)
        assert "terminal_armed" not in repr(screen.save_state())
        assert all(
            "terminal_armed" not in draft.values
            for draft in screen._settings_drafts.values()
        )

        screen.query_one("#settings-terminal-arm", Button).press()
        await pilot.pause()
        assert app.terminal_session_manager.armed is False

        screen.query_one("#settings-terminal-arm", Button).press()
        await pilot.pause()
        assert host.screen is screen
        assert app.terminal_session_manager.armed is True
        assert arm_calls == [True, False]


@pytest.mark.asyncio
async def test_terminal_disarm_confirms_one_cleanup_cohort_without_revoking_raw(
    monkeypatch,
):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    assert app.raw_cli_runtime.arm().armed is True
    terminal = RecordingTerminalRuntime(armed=True, session_count=2)
    app.terminal_session_manager = terminal
    notifications: list[tuple[str, str]] = []
    monkeypatch.setattr(
        StyledSettingsDestinationHarness,
        "notify",
        lambda _self, message, *, severity="information", **_kwargs: (
            notifications.append((message, severity))
        ),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        terminal_button = screen.query_one("#settings-terminal-arm", Button)
        assert str(terminal_button.label) == "Disarm Terminal"

        terminal_button.press()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        dialog = host.screen
        assert dialog.title == "Disarm Terminal and close sessions?"
        assert "2 live Terminal sessions" in dialog.message
        dialog_depth = len(host.screen_stack)
        screen.handle_terminal_arm_pressed(Button.Pressed(terminal_button))
        await pilot.pause()
        assert host.screen is dialog
        assert len(host.screen_stack) == dialog_depth
        assert terminal.disarm_calls == 0
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, lambda: host.screen is screen)

        assert terminal.disarm_calls == 1
        assert terminal.armed is False
        assert app.raw_cli_runtime.armed is True
        assert any(
            "pending or unproven" in message.lower() for message, _ in notifications
        )

        terminal.armed = True
        notifications.clear()
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = False
        await pilot.pause()
        locked_values = dict(app.app_config)
        locked_values["console"] = dict(app.app_config["console"])
        locked_values["console"]["raw_cli_permitted"] = False
        monkeypatch.setattr(
            SettingsScreen,
            "_save_raw_cli_permitted_value",
            staticmethod(lambda _value: (True, _published_snapshot(locked_values))),
        )

        screen.action_settings_save_category()
        await _wait_for_save(pilot)

        assert terminal.disarm_calls == 2
        assert terminal.armed is False
        assert app.raw_cli_runtime.armed is False
        assert app.app_config["console"]["raw_cli_permitted"] is False
        assert any(
            "pending or unproven" in message.lower() for message, _ in notifications
        )
        assert "Locked" in str(
            screen.query_one("#settings-terminal-state", Static).content
        )
