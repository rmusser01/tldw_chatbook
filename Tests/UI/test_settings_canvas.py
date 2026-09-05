"""Canonical F9 Settings Canvas policy controls."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from textual.widgets import Checkbox

from Tests.UI.test_destination_shells import _active_destination_screen, _visible_text
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import _click_settings_category
from Tests.UI.test_settings_configuration_hub import StyledSettingsDestinationHarness
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.config import RuntimeConfigSnapshot


async def test_privacy_settings_present_canvas_controls_status_and_read_only_quotas() -> (
    None
):
    app = _build_test_app(
        config_overrides={
            "canvas": {"enabled": True, "auto_open_on_create": False},
            "web_server": {"host": "127.0.0.1", "access_token": "DO-NOT-SHOW"},
        }
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 40)) as pilot:
        await _click_settings_category(pilot, SettingsCategoryId.PRIVACY_SECURITY.value)
        screen = _active_destination_screen(pilot.app)
        card = screen.query_one("#settings-canvas-card")
        text = _visible_text(card)

        assert card.query_one("#settings-canvas-enabled", Checkbox).value is True
        assert card.query_one("#settings-canvas-auto-open", Checkbox).value is False
        assert "Strict zero-egress runtime" in text
        assert "Loopback only" in text
        assert "Configured served posture" in text
        assert "Effective hard quotas — read-only" in text
        assert "HTML document" in text and "512 KiB" in text
        assert "Runtime memory" in text and "32 MiB" in text
        assert "DO-NOT-SHOW" not in text
        assert not card.query("Input")

        enabled = card.query_one("#settings-canvas-enabled", Checkbox)
        enabled.focus()
        await pilot.press("space")
        await pilot.pause()
        draft = screen._settings_drafts[SettingsCategoryId.PRIVACY_SECURITY]
        assert draft.values["canvas.enabled"] is False


async def test_canvas_disable_save_persists_both_values_and_revokes_live_runtime(
    monkeypatch,
) -> None:
    app = _build_test_app(
        config_overrides={
            "canvas": {"enabled": True, "auto_open_on_create": True},
            "console": {"raw_cli_permitted": False},
        }
    )
    runtime = SimpleNamespace(
        latch_canvas_disabled=Mock(),
        apply_canvas_policy=AsyncMock(),
    )
    app.console_runtime = runtime
    captured: list[tuple[bool, dict[str, bool]]] = []

    def save_privacy(raw_cli: bool, canvas: dict[str, bool]):
        captured.append((raw_cli, dict(canvas)))
        values = dict(app.app_config)
        values["canvas"] = dict(canvas)
        values["console"] = {"raw_cli_permitted": raw_cli}
        return True, RuntimeConfigSnapshot(17, values)

    monkeypatch.setattr(
        SettingsScreen,
        "_save_raw_cli_permitted_value",
        staticmethod(save_privacy),
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.run_if_runtime_config_generation_current",
        lambda _generation, action: action(),
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 40)) as pilot:
        await _click_settings_category(pilot, SettingsCategoryId.PRIVACY_SECURITY.value)
        screen = _active_destination_screen(pilot.app)
        screen.query_one("#settings-canvas-enabled", Checkbox).value = False
        screen.query_one("#settings-canvas-auto-open", Checkbox).value = False
        await pilot.pause()

        screen.action_settings_save_category()
        for _attempt in range(100):
            if not screen._raw_cli_save_pending:
                break
            await asyncio.sleep(0.01)
            await pilot.pause()

        assert captured == [
            (
                False,
                {"enabled": False, "auto_open_on_create": False},
            )
        ]
        assert app.app_config["canvas"] == {
            "enabled": False,
            "auto_open_on_create": False,
        }
        runtime.latch_canvas_disabled.assert_called_once_with()
        runtime.apply_canvas_policy.assert_awaited_once_with()
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.PRIVACY_SECURITY
        )


async def test_accepted_disable_stays_latched_when_config_is_reenabled_before_callback(
    monkeypatch,
) -> None:
    app = _build_test_app(
        config_overrides={
            "canvas": {"enabled": True, "auto_open_on_create": True},
            "console": {"raw_cli_permitted": False},
        }
    )
    runtime = SimpleNamespace(
        latch_canvas_disabled=Mock(),
        apply_canvas_policy=AsyncMock(),
    )
    app.console_runtime = runtime
    stale = RuntimeConfigSnapshot(
        17,
        {
            **app.app_config,
            "canvas": {"enabled": False, "auto_open_on_create": False},
        },
    )
    current = RuntimeConfigSnapshot(
        18,
        {
            **app.app_config,
            "canvas": {"enabled": True, "auto_open_on_create": True},
        },
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.get_runtime_config_snapshot",
        lambda *args, **kwargs: current,
    )
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.run_if_runtime_config_generation_current",
        lambda generation, action: action() if generation == 18 else False,
    )
    host = StyledSettingsDestinationHarness(app, "settings")

    async with host.run_test(size=(120, 40)) as pilot:
        await _click_settings_category(pilot, SettingsCategoryId.PRIVACY_SECURITY.value)
        screen = _active_destination_screen(pilot.app)
        screen.query_one("#settings-canvas-enabled", Checkbox).value = False
        await pilot.pause()

        screen._apply_raw_cli_save_result(
            True,
            stale,
            False,
            {"enabled": False, "auto_open_on_create": False},
        )
        await pilot.pause()

        runtime.latch_canvas_disabled.assert_called_once_with()
        runtime.apply_canvas_policy.assert_awaited_once_with()
        assert app.app_config["canvas"] == {
            "enabled": True,
            "auto_open_on_create": True,
        }
        assert screen.query_one("#settings-canvas-enabled", Checkbox).value is True
