"""Mounted contracts for the Settings raw CLI danger gate (TASK-18926 Task 4)."""

from __future__ import annotations

from pathlib import Path
import time

import pytest
from textual.app import App
from textual.widgets import Button, Checkbox, Static

from Tests.UI.test_destination_shells import (
    _active_destination_screen,
    _visible_text,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import _click_settings_category
from Tests.UI.test_settings_configuration_hub import StyledSettingsDestinationHarness
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


RAW_CLI_DISCLOSURE = (
    "full OS-user file, process, and network authority",
    "credential-file access remains possible despite environment scrubbing",
    "Command text and bounded output may persist in local run logs",
    "detached descendants may survive cleanup",
    "This is not a sandbox and is not limited to your workspace.",
)

_AGENTIC_TERMINAL_CSS_PATH = (
    Path(__file__).parents[2] / "tldw_chatbook/css/components/_agentic_terminal.tcss"
)
_AGENTIC_TERMINAL_CSS = _AGENTIC_TERMINAL_CSS_PATH.read_text(encoding="utf-8")
_RAW_CLI_CSS = _AGENTIC_TERMINAL_CSS.split(
    "/* RAW CLI DANGER GATE START */",
    1,
)[1].split("/* RAW CLI DANGER GATE END */", 1)[0]


class RawCliStyledSettingsHarness(StyledSettingsDestinationHarness):
    """Mount the app bundle plus the exact owned raw CLI source block."""

    CSS_PATH = StyledSettingsDestinationHarness.CSS_PATH
    CSS = _RAW_CLI_CSS


async def _wait_until(pilot, predicate, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            await pilot.pause()
            return
        await pilot.pause(0.01)
    assert predicate()


async def _open_privacy(pilot):
    await _click_settings_category(pilot, SettingsCategoryId.PRIVACY_SECURITY.value)
    return _active_destination_screen(pilot.app)


async def _wait_for_save(pilot) -> None:
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


def _painted_relative_luminance(color) -> float:
    triplet = color.get_truecolor()

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _painted_contrast(first, second) -> float:
    lighter, darker = sorted(
        (_painted_relative_luminance(first), _painted_relative_luminance(second)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_style_of_text(app: App, button: Button):
    needle = str(button.label)
    strips = list(app.screen._compositor.render_strips())
    for y in range(button.region.y, button.region.y + button.region.height):
        if y >= len(strips):
            break
        segments = list(strips[y]._segments)
        row_text = "".join(segment.text for segment in segments)
        index = row_text.find(needle)
        if index == -1:
            continue
        x = 0
        for segment in segments:
            if x + len(segment.text) > index:
                return segment.style
            x += len(segment.text)
    return None


@pytest.mark.asyncio
async def test_raw_cli_unlock_and_arm_are_separate_confirmed_gates():
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    host = RawCliStyledSettingsHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)

        assert screen.query("#settings-raw-cli-card"), (
            "Privacy & Security must mount the raw CLI danger gate"
        )
        card = screen.query_one("#settings-raw-cli-card")
        text = _visible_text(card)
        for disclosure in RAW_CLI_DISCLOSURE:
            assert disclosure in text
        assert "sandboxed" not in text.lower()
        assert "confined" not in text.lower()
        assert "Draft — save with s" in _visible_text(screen)
        assert "Applies immediately." in text
        assert screen.query_one("#settings-privacy-security-card")
        assert screen.query_one("#settings-check-privacy")
        assert screen.query_one("#settings-open-provider-credentials")
        assert screen.query_one("#settings-open-advanced-config")
        ownership = screen._ownership_record(SettingsCategoryId.PRIVACY_SECURITY)
        assert ownership.owns_config_sections == ("console.raw_cli_permitted",)

        checkbox = card.query_one("#settings-raw-cli-permitted", Checkbox)
        arm_button = card.query_one("#settings-raw-cli-arm", Button)
        assert checkbox.value is False
        assert "Locked" in str(
            card.query_one("#settings-raw-cli-state", Static).content
        )
        assert arm_button.disabled is True
        assert "Save unlock first" in str(arm_button.label)

        checkbox.value = True
        await pilot.pause()
        draft = screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY]
        assert draft.values == {"console.raw_cli_permitted": True}
        assert app.app_config["console"]["raw_cli_permitted"] is False
        assert "Locked" in str(
            card.query_one("#settings-raw-cli-state", Static).content
        )
        assert arm_button.disabled is True
        assert "Save unlock first" in str(arm_button.label)

        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        unlock_dialog = host.screen
        assert unlock_dialog.title == "Unlock raw CLI host access"
        await _wait_until(
            pilot,
            lambda: getattr(host.focused, "id", None) == "cancel-button",
        )
        await pilot.press("y")
        await pilot.pause()
        assert host.screen is unlock_dialog
        assert app.app_config["console"]["raw_cli_permitted"] is False
        assert app.raw_cli_runtime.armed is False
        await pilot.press("escape")
        await _wait_until(pilot, lambda: host.screen is screen)

        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_for_save(pilot)

        assert type(app.app_config["console"]["raw_cli_permitted"]) is bool
        assert app.app_config["console"]["raw_cli_permitted"] is True
        assert app.raw_cli_runtime.permitted is True
        assert SettingsCategoryId.PRIVACY_SECURITY not in screen._settings_drafts
        assert "Unlocked, not armed" in str(
            card.query_one("#settings-raw-cli-state", Static).content
        )
        arm_button = card.query_one("#settings-raw-cli-arm", Button)
        assert arm_button.disabled is False

        size_before = arm_button.region.size
        arm_button.focus()
        await _wait_until(pilot, lambda: host.focused is arm_button)
        assert arm_button.region.size == size_before
        assert await pilot.hover(arm_button)
        await pilot.pause()
        assert arm_button.region.size == size_before

        arm_button.press()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        arm_dialog = host.screen
        assert arm_dialog.title == "Arm raw CLI for this launch"
        await _wait_until(
            pilot,
            lambda: getattr(host.focused, "id", None) == "cancel-button",
        )
        await pilot.press("enter")
        await _wait_until(pilot, lambda: host.screen is screen)
        assert app.raw_cli_runtime.armed is False

        screen.query_one("#settings-raw-cli-arm", Button).press()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, lambda: host.screen is screen)
        await pilot.pause(0.05)

        card = screen.query_one("#settings-raw-cli-card")
        assert app.raw_cli_runtime.armed is True
        assert card.has_class("settings-raw-cli-armed")
        armed_rule = _RAW_CLI_CSS.split(
            ".settings-raw-cli-danger.settings-raw-cli-armed", 1
        )[1].split("}", 1)[0]
        assert "border: solid $error;" in armed_rule
        assert "background: $error 10%;" in armed_rule
        assert "ARMED — HOST ACCESS" in str(
            card.query_one("#settings-raw-cli-state", Static).content
        )
        assert "raw_cli_armed" not in app.app_config.get("console", {})
        assert "raw_cli_armed" not in vars(screen)
        assert "raw_cli_armed" not in repr(screen.save_state())
        assert all(
            "raw_cli_armed" not in draft.values
            for draft in screen._settings_drafts.values()
        )


@pytest.mark.asyncio
async def test_raw_cli_unlock_uses_ordinary_settings_revert():
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    host = RawCliStyledSettingsHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        assert screen.query("#settings-raw-cli-card"), "raw CLI card is absent"
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = True
        await pilot.pause()
        assert screen._category_has_unsaved_changes(SettingsCategoryId.PRIVACY_SECURITY)

        screen.action_settings_revert_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, lambda: host.screen is screen)

        assert checkbox.value is False
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.PRIVACY_SECURITY
        )
        assert app.app_config["console"]["raw_cli_permitted"] is False


@pytest.mark.asyncio
async def test_raw_cli_disarm_is_immediate_and_saved_lock_starts_cleanup(monkeypatch):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    assert app.raw_cli_runtime.arm().armed is True
    host = RawCliStyledSettingsHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        assert screen.query("#settings-raw-cli-card"), "raw CLI card is absent"
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = False
        await pilot.pause()
        draft = screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY]
        assert draft.values == {"console.raw_cli_permitted": False}
        assert app.raw_cli_runtime.armed is True

        disarm_button = screen.query_one("#settings-raw-cli-arm", Button)
        assert str(disarm_button.label) == "Disarm now"
        assert disarm_button.disabled is False
        disarm_button.press()
        await pilot.pause()

        assert app.raw_cli_runtime.armed is False
        assert screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY] is draft
        assert draft.values == {"console.raw_cli_permitted": False}
        assert app.app_config["console"]["raw_cli_permitted"] is True
        recovery = screen.query_one("#settings-raw-cli-arm", Button)
        assert recovery.disabled is True
        assert "Save unlock first" in str(recovery.label)

        assert app.raw_cli_runtime.arm().armed is True
        screen._refresh_raw_cli_state()
        original_disarm = app.raw_cli_runtime.disarm
        disarm_calls: list[tuple[str, ...]] = []

        def tracked_disarm() -> tuple[str, ...]:
            result = original_disarm()
            disarm_calls.append(result)
            return result

        monkeypatch.setattr(app.raw_cli_runtime, "disarm", tracked_disarm)
        screen.action_settings_save_category()
        await _wait_for_save(pilot)

        assert disarm_calls == [()]
        assert app.raw_cli_runtime.armed is False
        assert type(app.app_config["console"]["raw_cli_permitted"]) is bool
        assert app.app_config["console"]["raw_cli_permitted"] is False
        assert SettingsCategoryId.PRIVACY_SECURITY not in screen._settings_drafts
        assert "Locked" in str(
            screen.query_one("#settings-raw-cli-state", Static).content
        )


@pytest.mark.asyncio
async def test_failed_raw_cli_lock_save_keeps_saved_authority_and_draft(monkeypatch):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    assert app.raw_cli_runtime.arm().armed is True
    host = RawCliStyledSettingsHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        assert screen.query("#settings-raw-cli-card"), "raw CLI card is absent"
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = False
        await pilot.pause()

        monkeypatch.setattr(
            SettingsScreen,
            "_save_raw_cli_permitted_value",
            staticmethod(lambda _value: (False, None)),
        )
        screen.action_settings_save_category()
        await _wait_for_save(pilot)

        assert app.raw_cli_runtime.armed is True
        assert app.app_config["console"]["raw_cli_permitted"] is True
        assert screen._category_has_unsaved_changes(SettingsCategoryId.PRIVACY_SECURITY)
        assert screen.query_one("#settings-raw-cli-permitted", Checkbox).value is False
        assert "ARMED — HOST ACCESS" in str(
            screen.query_one("#settings-raw-cli-state", Static).content
        )


@pytest.mark.asyncio
async def test_raw_cli_disclosure_wraps_and_disabled_arm_paints_above_contrast_floor():
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    host = RawCliStyledSettingsHarness(app, "settings")

    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open_privacy(pilot)
        assert screen.query("#settings-raw-cli-card"), "raw CLI card is absent"
        card = screen.query_one("#settings-raw-cli-card")
        text = _visible_text(card)
        for disclosure in RAW_CLI_DISCLOSURE:
            assert disclosure in text

        body = screen.query_one("#settings-detail-pane-body")
        arm_button = card.query_one("#settings-raw-cli-arm", Button)
        assert body.max_scroll_y > 0
        assert any(
            static.region.height > 1
            for static in card.query(".settings-raw-cli-disclosure")
        )
        body.scroll_to_widget(arm_button, animate=False, force=True)
        await pilot.pause()
        assert arm_button.region.width > 0
        assert screen.region.contains_region(arm_button.region)

        painted = _painted_style_of_text(host, arm_button)
        assert painted is not None
        assert painted.color is not None and painted.bgcolor is not None
        ratio = _painted_contrast(painted.color, painted.bgcolor)
        assert ratio >= 3.0, f"disabled Arm label paints at only {ratio:.2f}:1"
