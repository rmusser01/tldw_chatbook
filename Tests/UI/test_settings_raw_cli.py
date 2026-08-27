"""Mounted contracts for the Settings raw CLI danger gate (TASK-18926 Task 4)."""

from __future__ import annotations

import threading
import time
from unittest.mock import AsyncMock

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
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from tldw_chatbook.config import RuntimeConfigSnapshot
from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
    AudioCppModelLibraryRequest,
)
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
    SpeechTTSSettingsPanel,
)


RAW_CLI_DISCLOSURE = (
    "Commands run with the same OS permissions as Chatbook.",
    "Commands can read, modify, or delete any accessible file, including Chatbook's "
    "config and permission store.",
    "Commands can access the network, invoke credentialed clients, launch background "
    "processes, and exhaust machine resources.",
    "The environment is scrubbed, but commands can still read credential files and "
    "other user data.",
    "Cancellation attempts to terminate the owned process group/job; deliberately "
    "detached descendants may survive.",
    "Command text and bounded output may persist in local run logs.",
    "This is not a sandbox and is not limited to your workspace.",
)


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


def _published_snapshot(
    values: dict, *, generation: int | None = None
) -> RuntimeConfigSnapshot:
    if generation is None:
        generation = settings_screen_module.get_runtime_config_snapshot().generation
    return RuntimeConfigSnapshot(generation, values)


def _install_runtime_generation_state(
    monkeypatch, snapshot: RuntimeConfigSnapshot
) -> None:
    monkeypatch.setattr(
        settings_screen_module,
        "get_runtime_config_snapshot",
        lambda **_kwargs: snapshot,
    )

    def guarded(expected_generation: int, action) -> bool:
        if expected_generation != snapshot.generation:
            return False
        return action() is True

    monkeypatch.setattr(
        settings_screen_module,
        "run_if_runtime_config_generation_current",
        guarded,
        raising=False,
    )


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
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)

        assert screen.query("#settings-raw-cli-card"), (
            "Privacy & Security must mount the raw CLI danger gate"
        )
        card = screen.query_one("#settings-raw-cli-card")
        locked_border = card.styles.border
        locked_background = card.styles.background
        error_color = card.query_one("#settings-raw-cli-state", Static).styles.color
        assert (
            str(card.query_one("#settings-raw-cli-title", Static).content)
            == "DANGER!!! RAW CLI HOST ACCESS"
        )
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
        assert unlock_dialog.message == "\n\n".join(RAW_CLI_DISCLOSURE)
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
        assert arm_dialog.message == "\n\n".join(RAW_CLI_DISCLOSURE)
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
        armed_border = card.styles.border
        armed_edges = (
            armed_border.top,
            armed_border.right,
            armed_border.bottom,
            armed_border.left,
        )
        assert armed_border != locked_border
        assert all(kind == "solid" for kind, _color in armed_edges)
        assert all(color == error_color for _kind, color in armed_edges)
        armed_background = card.styles.background
        assert armed_background != locked_background
        assert armed_background == error_color.with_alpha(0.1)
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
async def test_raw_cli_confirmation_actions_are_serialized(monkeypatch):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    loaded_config = dict(app.app_config)
    loaded_config["console"] = dict(app.app_config["console"])
    loaded_config["console"]["raw_cli_permitted"] = True
    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(lambda _value: (True, _published_snapshot(loaded_config))),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = True
        await pilot.pause()

        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        unlock_dialog = host.screen
        unlock_depth = len(host.screen_stack)
        screen.action_settings_save_category()
        await pilot.pause()
        assert host.screen is unlock_dialog
        assert len(host.screen_stack) == unlock_depth
        assert screen._raw_cli_unlock_confirmation_pending is True

        await pilot.press("escape")
        await _wait_until(pilot, lambda: host.screen is screen)
        assert screen._raw_cli_unlock_confirmation_pending is False

        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_for_save(pilot)
        assert screen._raw_cli_unlock_confirmation_pending is False

        arm_button = screen.query_one("#settings-raw-cli-arm", Button)
        arm_button.press()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        arm_dialog = host.screen
        arm_depth = len(host.screen_stack)
        screen.handle_raw_cli_arm_pressed(Button.Pressed(arm_button))
        await pilot.pause()
        assert host.screen is arm_dialog
        assert len(host.screen_stack) == arm_depth
        assert screen._raw_cli_arm_confirmation_pending is True

        await pilot.press("escape")
        await _wait_until(pilot, lambda: host.screen is screen)
        assert screen._raw_cli_arm_confirmation_pending is False

        arm_button.press()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, lambda: host.screen is screen)
        assert screen._raw_cli_arm_confirmation_pending is False
        assert app.raw_cli_runtime.armed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation", ("unlock", "arm"))
async def test_raw_cli_confirmation_pending_clears_when_push_fails(
    monkeypatch, confirmation: str
):
    permitted = confirmation == "arm"
    app = _build_test_app(
        config_overrides={"console": {"raw_cli_permitted": permitted}}
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)

        def failed_push(*_args, **_kwargs):
            raise RuntimeError("push failed")

        monkeypatch.setattr(host, "push_screen", failed_push)
        if confirmation == "unlock":
            screen.query_one("#settings-raw-cli-permitted", Checkbox).value = True
            await pilot.pause()
            with pytest.raises(RuntimeError, match="push failed"):
                screen.action_settings_save_category()
            assert screen._raw_cli_unlock_confirmation_pending is False
        else:
            arm_button = screen.query_one("#settings-raw-cli-arm", Button)
            with pytest.raises(RuntimeError, match="push failed"):
                screen.handle_raw_cli_arm_pressed(Button.Pressed(arm_button))
            assert screen._raw_cli_arm_confirmation_pending is False


@pytest.mark.asyncio
async def test_raw_cli_unlock_uses_ordinary_settings_revert():
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    host = StyledSettingsDestinationHarness(app, "settings")

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
    host = StyledSettingsDestinationHarness(app, "settings")

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
    host = StyledSettingsDestinationHarness(app, "settings")

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
        assert screen._raw_cli_save_pending is False


@pytest.mark.asyncio
async def test_successful_raw_cli_write_with_snapshot_failure_fails_closed(
    monkeypatch,
):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    assert app.raw_cli_runtime.arm().armed is True
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = False
        await pilot.pause()
        writes: list[dict] = []

        class SuccessfulAdapter:
            def save_sections(self, section_values):
                writes.append(section_values)
                return True

        def failed_snapshot(**_kwargs):
            raise RuntimeError("post-write snapshot failed")

        monkeypatch.setattr(
            settings_screen_module,
            "SettingsConfigAdapter",
            SuccessfulAdapter,
        )
        monkeypatch.setattr(
            settings_screen_module,
            "get_runtime_config_snapshot",
            failed_snapshot,
        )
        screen.action_settings_save_category()
        await _wait_for_save(pilot)

        assert writes == [{"console": {"raw_cli_permitted": False}}]
        assert app.app_config["console"]["raw_cli_permitted"] is False
        assert app.raw_cli_runtime.armed is False
        assert SettingsCategoryId.PRIVACY_SECURITY not in screen._settings_drafts
        assert checkbox.value is False
        assert screen._raw_cli_save_pending is False


@pytest.mark.asyncio
async def test_unstable_raw_cli_generation_uses_exact_bounded_attempts_and_disarms(
    monkeypatch,
):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    assert app.raw_cli_runtime.arm().armed is True
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = False
        await pilot.pause()
        snapshot_calls: list[int] = []
        guarded_generations: list[int] = []

        def changing_snapshot(**_kwargs) -> RuntimeConfigSnapshot:
            generation = 101 + len(snapshot_calls)
            snapshot_calls.append(generation)
            values = dict(app.app_config)
            values["console"] = dict(app.app_config["console"])
            values["console"]["raw_cli_permitted"] = True
            return RuntimeConfigSnapshot(generation, values)

        def refuse_generation(expected_generation: int, _action) -> bool:
            guarded_generations.append(expected_generation)
            return False

        monkeypatch.setattr(
            settings_screen_module,
            "get_runtime_config_snapshot",
            changing_snapshot,
        )
        monkeypatch.setattr(
            settings_screen_module,
            "run_if_runtime_config_generation_current",
            refuse_generation,
        )
        screen._raw_cli_save_pending = True
        screen._apply_raw_cli_save_result(True, None, False)
        await pilot.pause()

        assert snapshot_calls == [101, 102, 103]
        assert guarded_generations == [101, 102, 103]
        assert app.app_config["console"]["raw_cli_permitted"] is False
        assert app.raw_cli_runtime.armed is False
        assert SettingsCategoryId.PRIVACY_SECURITY not in screen._settings_drafts
        assert checkbox.value is False
        assert screen._raw_cli_save_pending is False


@pytest.mark.asyncio
async def test_pending_raw_cli_save_preserves_a_newer_clean_draft(monkeypatch):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    loaded_config = dict(app.app_config)
    loaded_config["console"] = dict(app.app_config["console"])
    loaded_config["console"]["raw_cli_permitted"] = True
    save_started = threading.Event()
    release_save = threading.Event()
    save_calls: list[bool] = []

    def blocked_save(value: bool) -> tuple[bool, RuntimeConfigSnapshot | None]:
        save_calls.append(value)
        save_started.set()
        assert release_save.wait(timeout=3)
        return True, _published_snapshot(loaded_config)

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(blocked_save),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = True
        await pilot.pause()
        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, save_started.is_set)

        try:
            screen.action_settings_save_category()
            await pilot.pause()
            assert host.screen is screen
            assert save_calls == [True]

            checkbox.value = False
            await pilot.pause()
            draft = screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY]
            assert not draft.is_dirty
            screen.action_settings_save_category()
            await pilot.pause()
            assert save_calls == [True]
            assert screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY] is draft
            assert checkbox.value is False
        finally:
            release_save.set()
            await _wait_for_save(pilot)

        assert screen._raw_cli_save_pending is False
        assert app.app_config["console"]["raw_cli_permitted"] is True
        assert draft.originals == {"console.raw_cli_permitted": True}
        assert draft.values == {"console.raw_cli_permitted": False}
        assert draft.is_dirty
        assert checkbox.value is False


@pytest.mark.asyncio
async def test_pending_raw_cli_save_vetoes_real_navigation_until_arrival(monkeypatch):
    app = _build_test_app(
        configured_default="settings",
        config_overrides={"console": {"raw_cli_permitted": False}},
    )
    loaded_config = dict(app.app_config)
    loaded_config["console"] = dict(app.app_config["console"])
    loaded_config["console"]["raw_cli_permitted"] = True
    save_started = threading.Event()
    release_save = threading.Event()
    save_calls: list[bool] = []

    def blocked_save(value: bool) -> tuple[bool, RuntimeConfigSnapshot | None]:
        save_calls.append(value)
        save_started.set()
        assert release_save.wait(timeout=3)
        return True, _published_snapshot(loaded_config)

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(blocked_save),
    )

    async with app.run_test(size=(120, 35)) as pilot:
        await _wait_until(
            pilot,
            lambda: isinstance(app.screen, SettingsScreen),
            timeout=3,
        )
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        flush_outcomes: list[bool] = []
        real_flush = screen.flush_pending_work

        async def tracked_flush() -> bool:
            outcome = await real_flush()
            flush_outcomes.append(outcome)
            return outcome

        monkeypatch.setattr(screen, "flush_pending_work", tracked_flush)
        assert screen._start_raw_cli_save(True) is True
        await _wait_until(pilot, save_started.is_set)

        try:
            request = AudioCppModelLibraryRequest("raw-save-route", 1)
            app._audio_cpp_settings_model_library_request = request
            screen._speech_tts_model_library_route_token = request.token
            screen.post_message(
                NavigateToScreen(
                    "llm",
                    {"view": "curated", "consumer": "audio_cpp"},
                )
            )
            await _wait_until(pilot, lambda: flush_outcomes == [False])
            assert app.screen is screen
            assert screen.is_mounted

            assert screen._start_raw_cli_save(True) is False
            assert save_calls == [True]
        finally:
            release_save.set()
            await _wait_until(
                pilot,
                lambda: not screen._raw_cli_save_pending,
                timeout=3,
            )

        screen.query_one("#settings-category-speech-tts", Button).press()
        await _wait_until(
            pilot,
            lambda: bool(screen.query("#settings-speech-tts-panel")),
            timeout=8,
        )
        panel = screen.query_one(
            "#settings-speech-tts-panel",
            SpeechTTSSettingsPanel,
        )
        panel.confirm_leave = AsyncMock(return_value=False)

        screen.post_message(NavigateToScreen("home"))
        await _wait_until(pilot, lambda: len(flush_outcomes) == 2)
        assert flush_outcomes == [False, False]
        assert panel.confirm_leave.await_count == 1
        assert app.screen is screen
        assert screen.is_mounted

        panel.confirm_leave.return_value = True
        screen.post_message(NavigateToScreen("home"))
        await _wait_until(pilot, lambda: app.screen is not screen, timeout=3)
        assert type(app.screen).__name__ == "HomeScreen"
        assert flush_outcomes == [False, False, True]
        assert panel.confirm_leave.await_count == 2
        assert save_calls == [True]


@pytest.mark.asyncio
async def test_stale_raw_cli_unlock_snapshot_reconciles_latest_lock_and_disarms(
    monkeypatch,
):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    worker_values = dict(app.app_config)
    worker_values["console"] = dict(app.app_config["console"])
    worker_values["console"]["raw_cli_permitted"] = True
    latest_values = dict(worker_values)
    latest_values["console"] = dict(worker_values["console"])
    latest_values["console"]["raw_cli_permitted"] = False
    worker_snapshot = _published_snapshot(worker_values, generation=41)
    latest_snapshot = _published_snapshot(latest_values, generation=42)
    save_started = threading.Event()
    release_save = threading.Event()

    def blocked_save(_value: bool) -> tuple[bool, RuntimeConfigSnapshot | None]:
        save_started.set()
        assert release_save.wait(timeout=3)
        return True, worker_snapshot

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(blocked_save),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = True
        await pilot.pause()
        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, save_started.is_set)

        _install_runtime_generation_state(monkeypatch, latest_snapshot)
        app.app_config["console"]["raw_cli_permitted"] = True
        assert app.raw_cli_runtime.arm().armed is True
        release_save.set()
        await _wait_for_save(pilot)

        assert app.app_config["console"]["raw_cli_permitted"] is False
        assert app.raw_cli_runtime.armed is False
        assert SettingsCategoryId.PRIVACY_SECURITY not in screen._settings_drafts
        assert checkbox.value is False


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_authority", (None, "true"), ids=("missing", "text"))
async def test_successful_raw_cli_save_with_invalid_latest_authority_fails_closed(
    monkeypatch,
    invalid_authority,
):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    invalid_values = dict(app.app_config)
    invalid_values["console"] = dict(app.app_config["console"])
    if invalid_authority is None:
        invalid_values["console"].pop("raw_cli_permitted", None)
    else:
        invalid_values["console"]["raw_cli_permitted"] = invalid_authority
    invalid_snapshot = _published_snapshot(invalid_values, generation=47)
    save_started = threading.Event()
    release_save = threading.Event()

    def blocked_save(_value: bool) -> tuple[bool, RuntimeConfigSnapshot | None]:
        save_started.set()
        assert release_save.wait(timeout=3)
        return True, invalid_snapshot

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(blocked_save),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = True
        await pilot.pause()
        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, save_started.is_set)

        _install_runtime_generation_state(monkeypatch, invalid_snapshot)
        app.app_config["console"]["raw_cli_permitted"] = True
        assert app.raw_cli_runtime.arm().armed is True
        release_save.set()
        await _wait_for_save(pilot)

        assert app.app_config["console"]["raw_cli_permitted"] is False
        assert app.raw_cli_runtime.armed is False
        assert SettingsCategoryId.PRIVACY_SECURITY not in screen._settings_drafts
        assert checkbox.value is False


@pytest.mark.asyncio
async def test_stale_raw_cli_snapshot_reconciles_newer_true_without_clobbering(
    monkeypatch,
):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    worker_values = dict(app.app_config)
    worker_values["console"] = dict(app.app_config["console"])
    worker_values["console"]["raw_cli_permitted"] = True
    latest_values = dict(worker_values)
    latest_values["unrelated_snapshot_value"] = "must not replace app_config"
    worker_snapshot = _published_snapshot(worker_values, generation=51)
    latest_snapshot = _published_snapshot(latest_values, generation=52)
    save_started = threading.Event()
    release_save = threading.Event()

    def blocked_save(_value: bool) -> tuple[bool, RuntimeConfigSnapshot | None]:
        save_started.set()
        assert release_save.wait(timeout=3)
        return True, worker_snapshot

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(blocked_save),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = True
        await pilot.pause()
        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, save_started.is_set)

        _install_runtime_generation_state(monkeypatch, latest_snapshot)
        app.app_config["unrelated_runtime_change"] = {"marker": "preserve me"}
        app.app_config["console"]["concurrent_runtime_marker"] = "preserve console"
        release_save.set()
        await _wait_for_save(pilot)

        assert app.app_config["console"]["raw_cli_permitted"] is True
        assert app.app_config["console"]["concurrent_runtime_marker"] == (
            "preserve console"
        )
        assert app.app_config["unrelated_runtime_change"] == {"marker": "preserve me"}
        assert "unrelated_snapshot_value" not in app.app_config
        assert SettingsCategoryId.PRIVACY_SECURITY not in screen._settings_drafts


@pytest.mark.asyncio
async def test_newer_checkbox_edit_survives_stale_generation_reconciliation(
    monkeypatch,
):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    worker_values = dict(app.app_config)
    worker_values["console"] = dict(app.app_config["console"])
    worker_values["console"]["raw_cli_permitted"] = True
    worker_snapshot = _published_snapshot(worker_values, generation=61)
    latest_snapshot = _published_snapshot(worker_values, generation=62)
    save_started = threading.Event()
    release_save = threading.Event()

    def blocked_save(_value: bool) -> tuple[bool, RuntimeConfigSnapshot | None]:
        save_started.set()
        assert release_save.wait(timeout=3)
        return True, worker_snapshot

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(blocked_save),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = True
        await pilot.pause()
        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, save_started.is_set)

        checkbox.value = False
        await pilot.pause()
        _install_runtime_generation_state(monkeypatch, latest_snapshot)
        release_save.set()
        await _wait_for_save(pilot)

        draft = screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY]
        assert app.app_config["console"]["raw_cli_permitted"] is True
        assert draft.originals == {"console.raw_cli_permitted": True}
        assert draft.values == {"console.raw_cli_permitted": False}
        assert draft.is_dirty
        assert checkbox.value is False


@pytest.mark.asyncio
async def test_raw_cli_revert_is_blocked_while_save_is_pending(monkeypatch):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    loaded_config = dict(app.app_config)
    loaded_config["console"] = dict(app.app_config["console"])
    loaded_config["console"]["raw_cli_permitted"] = True
    save_started = threading.Event()
    release_save = threading.Event()

    def blocked_save(_value: bool) -> tuple[bool, RuntimeConfigSnapshot | None]:
        save_started.set()
        assert release_save.wait(timeout=3)
        return True, _published_snapshot(loaded_config)

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(blocked_save),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = True
        await pilot.pause()
        draft = screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY]
        screen.action_settings_save_category()
        await _wait_until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        assert await pilot.click("#confirm-button")
        await _wait_until(pilot, save_started.is_set)

        try:
            screen.action_settings_revert_category()
            await pilot.pause()
            assert host.screen is screen
            assert screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY] is draft
        finally:
            release_save.set()
            await _wait_for_save(pilot)


@pytest.mark.asyncio
async def test_raw_cli_save_preserves_unrelated_runtime_config_changes(monkeypatch):
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": True}})
    loaded_config = dict(app.app_config)
    loaded_config["console"] = dict(app.app_config["console"])
    loaded_config["console"]["raw_cli_permitted"] = False
    save_started = threading.Event()
    release_save = threading.Event()

    def blocked_save(_value: bool) -> tuple[bool, RuntimeConfigSnapshot | None]:
        save_started.set()
        assert release_save.wait(timeout=3)
        return True, _published_snapshot(loaded_config)

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(blocked_save),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = False
        await pilot.pause()
        screen.action_settings_save_category()
        await _wait_until(pilot, save_started.is_set)

        app.app_config["concurrent_runtime_change"] = {"marker": "preserve me"}
        app.app_config["console"]["concurrent_runtime_marker"] = "preserve console"
        release_save.set()
        await _wait_for_save(pilot)

        assert app.app_config["console"]["raw_cli_permitted"] is False
        assert (
            app.app_config["console"]["concurrent_runtime_marker"] == "preserve console"
        )
        assert app.app_config["concurrent_runtime_change"] == {"marker": "preserve me"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("saved_value", "dispatched_value", "newer_value"),
    ((False, True, False), (True, False, True)),
    ids=("unlock-arrival", "lock-arrival"),
)
async def test_raw_cli_save_arrival_preserves_a_newer_mounted_checkbox_edit(
    monkeypatch,
    saved_value: bool,
    dispatched_value: bool,
    newer_value: bool,
):
    app = _build_test_app(
        config_overrides={"console": {"raw_cli_permitted": saved_value}}
    )
    if saved_value:
        assert app.raw_cli_runtime.arm().armed is True
    loaded_config = dict(app.app_config)
    loaded_config["console"] = dict(app.app_config["console"])
    loaded_config["console"]["raw_cli_permitted"] = dispatched_value
    save_started = threading.Event()
    release_save = threading.Event()

    def blocked_save(value: bool) -> tuple[bool, RuntimeConfigSnapshot | None]:
        assert value is dispatched_value
        save_started.set()
        assert release_save.wait(timeout=3)
        return True, _published_snapshot(loaded_config)

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(blocked_save),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 35)) as pilot:
        screen = await _open_privacy(pilot)
        checkbox = screen.query_one("#settings-raw-cli-permitted", Checkbox)
        checkbox.value = dispatched_value
        await pilot.pause()

        screen.action_settings_save_category()
        if dispatched_value:
            await _wait_until(
                pilot, lambda: isinstance(host.screen, ConfirmationDialog)
            )
            assert await pilot.click("#confirm-button")
        await _wait_until(pilot, save_started.is_set)

        checkbox.value = newer_value
        await pilot.pause()
        release_save.set()
        await _wait_for_save(pilot)

        draft = screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY]
        assert app.app_config["console"]["raw_cli_permitted"] is dispatched_value
        assert draft.originals == {"console.raw_cli_permitted": dispatched_value}
        assert draft.values == {"console.raw_cli_permitted": newer_value}
        assert draft.is_dirty
        assert checkbox.value is newer_value
        assert app.raw_cli_runtime.armed is False


@pytest.mark.asyncio
async def test_raw_cli_disclosure_wraps_and_disabled_arm_paints_above_contrast_floor():
    app = _build_test_app(config_overrides={"console": {"raw_cli_permitted": False}})
    host = StyledSettingsDestinationHarness(app, "settings")

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
        assert body.content_region.contains_region(arm_button.region)

        painted = _painted_style_of_text(host, arm_button)
        assert painted is not None
        assert painted.color is not None and painted.bgcolor is not None
        ratio = _painted_contrast(painted.color, painted.bgcolor)
        assert ratio >= 3.0, f"disabled Arm label paints at only {ratio:.2f}:1"
