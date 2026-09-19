"""Instant model-refresh controls retain edits and publish truthful receipts."""

import copy
import threading

import pytest
from textual.widgets import Button, Checkbox, Input, Static

from Tests.UI.test_settings_configuration_hub import _open_settings_category
from Tests.UI.test_settings_model_catalog_toggles import _build_settings_app
from Tests.UI.test_settings_provider_keyboard_journeys import (
    ProviderSettingsHarness,
    _edit,
    _settle,
    _tab_to,
)
from tldw_chatbook.UI.Screens import settings_screen as settings_module
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

MASTER = "#settings-model-catalog-auto-refresh"
HOURS = "#settings-model-catalog-stale-hours"
STATUS = "#settings-model-catalog-save-status"
RETRY = "#settings-model-catalog-retry"


class ConfigWriter:
    def __init__(self, *, failure=None, gate=False):
        self.config = {"model_catalog": {"refresh_consent_recorded": False}}
        self.calls = []
        self.failure = failure
        self.gate = gate
        self.started = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def save(self, sections):
        with self.lock:
            self.calls.append(copy.deepcopy(sections))
            self.active += 1
            self.max_active = max(self.active, self.max_active)
            first = len(self.calls) == 1
        try:
            if self.gate and first:
                self.started.set()
                assert self.release.wait(5), "writer gate was not released"
            if self.failure == "raise":
                raise OSError("fixture-secret must stay out of the UI")
            if self.failure == "false":
                return False
            for key, values in sections.items():
                self.config.setdefault(key, {}).update(copy.deepcopy(values))
            return True
        finally:
            with self.lock:
                self.active -= 1


async def _open(host, pilot):
    await _open_settings_category(pilot, "#settings-category-providers-models")
    await _settle(host, pilot)
    return host.screen


def _install(monkeypatch, writer):
    monkeypatch.setattr(
        settings_module, "load_settings", lambda: copy.deepcopy(writer.config)
    )
    monkeypatch.setattr(settings_module, "save_settings_to_cli_config", writer.save)


def _status(screen):
    return str(screen.query_one(STATUS, Static).renderable)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_catalog_keyboard_failure_retry_and_fractional_interval(
    theme, size, monkeypatch
):
    writer = ConfigWriter(failure="false")
    _install(monkeypatch, writer)
    app = _build_settings_app()
    host = ProviderSettingsHarness(app, "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = await _open(host, pilot)
        assert not writer.calls
        await _tab_to(host, pilot, MASTER)
        await pilot.press("space")
        await _settle(host, pilot)
        assert "not saved" in _status(screen).lower()
        assert not screen.query_one(MASTER, Checkbox).value
        screen.mutate_reactive(SettingsScreen.active_category)
        await _settle(host, pilot)
        assert not screen.query_one(MASTER, Checkbox).value
        assert "not saved" in _status(screen).lower()
        assert len(writer.calls) == 1
        writer.failure = None
        await _tab_to(host, pilot, RETRY)
        await pilot.press("enter")
        await _settle(host, pilot)
        assert writer.config["model_catalog"]["auto_refresh_enabled"] is False
        assert "saved" in _status(screen).lower()
        assert not screen.query_one(RETRY, Button).display
        await _edit(host, pilot, HOURS, "0.5")
        assert writer.config["model_catalog"]["stale_after_hours"] == 0.5
        assert writer.config["model_catalog"]["refresh_consent_recorded"] is False
        assert all(set(call) == {"model_catalog"} for call in writer.calls)
        assert all(
            "refresh_consent_recorded" not in call["model_catalog"]
            for call in writer.calls
        )


@pytest.mark.asyncio
async def test_catalog_exception_preserves_choices_and_redacts_message(monkeypatch):
    writer = ConfigWriter(failure="raise")
    _install(monkeypatch, writer)
    host = ProviderSettingsHarness(_build_settings_app(), "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        screen.query_one("#settings-mc-write-openai", Checkbox).value = True
        await _settle(host, pilot)
        assert "not saved" in _status(screen).lower()
        assert "fixture-secret" not in _status(screen)
        assert screen.query_one("#settings-mc-write-openai", Checkbox).value
        writer.failure = None
        screen.query_one(RETRY, Button).press()
        await _settle(host, pilot)
        assert writer.config["model_catalog"]["write_to_config"] == ["OpenAI"]


@pytest.mark.asyncio
@pytest.mark.parametrize("latest", ["12", "24"])
async def test_catalog_serializes_rapid_edits_including_saved_value_roundtrip(
    latest, monkeypatch
):
    writer = ConfigWriter(gate=True)
    _install(monkeypatch, writer)
    host = ProviderSettingsHarness(_build_settings_app(), "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open(host, pilot)
        screen.query_one(HOURS, Input).value = "6"
        try:
            for _ in range(100):
                if writer.started.is_set():
                    break
                await pilot.pause(0.01)
            assert writer.started.is_set()
            screen.query_one(HOURS, Input).value = latest
            await pilot.pause()
            screen.mutate_reactive(SettingsScreen.active_category)
            await pilot.pause()
            after_rebuild = screen.query_one(HOURS, Input).value
        finally:
            writer.release.set()
        await _settle(host, pilot)
        assert after_rebuild == latest
        assert writer.config["model_catalog"]["stale_after_hours"] == int(latest)
        assert writer.max_active == 1
        assert "saved" in _status(screen).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["", "abc", "-1", "nan", "inf", "-inf"])
async def test_catalog_invalid_intervals_do_not_write_and_explain_recovery(
    value, monkeypatch
):
    writer = ConfigWriter()
    _install(monkeypatch, writer)
    host = ProviderSettingsHarness(_build_settings_app(), "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = await _open(host, pilot)
        screen.query_one(HOURS, Input).value = value
        await _settle(host, pilot)
        assert not writer.calls
        assert "hours" in _status(screen).lower()
        assert "not saved" in _status(screen).lower()
        screen.mutate_reactive(SettingsScreen.active_category)
        await _settle(host, pilot)
        assert screen.query_one(HOURS, Input).value == value
        screen.query_one(HOURS, Input).value = "6"
        await _settle(host, pilot)
        assert writer.config["model_catalog"]["stale_after_hours"] == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (120, 35), (80, 24)])
async def test_catalog_checkbox_labels_are_fully_painted(size, theme, monkeypatch):
    writer = ConfigWriter()
    _install(monkeypatch, writer)
    host = ProviderSettingsHarness(_build_settings_app(), "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = await _open(host, pilot)
        controls = list(screen.query("#settings-model-catalog-group Checkbox"))
        for checkbox in controls:
            await _tab_to(host, pilot, f"#{checkbox.id}")
            region, clip = screen._compositor.visible_widgets[checkbox]
            visible = region.intersection(clip)
            strips = list(screen._compositor.render_strips())
            painted = " ".join(
                strips[row].crop(visible.x, visible.right).text
                for row in range(visible.y, visible.bottom)
            )
            assert str(checkbox.label) in painted, (checkbox.id, painted)
        assert not writer.calls
