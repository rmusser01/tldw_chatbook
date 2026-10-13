"""Settings ▸ Providers & Models ▸ Session summary on quit: when its two
controls save, and what they show afterwards (review of #3057, R2 F8,
ruling C8).

The duration wrote config.toml on every keystroke from unqueued threads,
the field kept "99" after 30 was saved, and the Inspector said the two
controls were staged for Save. The duration now saves when the field is
submitted or left, the checkbox still applies at once, the field and the
disclosure title show what was saved, and both controls read as
instant-apply.
"""

from __future__ import annotations

import threading

import pytest
from textual.widgets import Checkbox, Collapsible, Input

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import _FAKE_KEY, _SIZE, _open_providers
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_keyboard_journeys import _settle
from tldw_chatbook.UI.Screens import settings_screen

_DISCLOSURE = "settings-advanced-session-summary"
_TITLE = "Session summary on quit · applies immediately · "


class _Config:
    """The [session_summary] table, with every write to it recorded."""

    def __init__(self, monkeypatch) -> None:
        self.saved = {"enabled": False, "duration_seconds": 3}
        self.writes: list[dict] = []
        self.fail = False
        real_get = settings_screen.get_cli_setting

        def get(section, key=None, default=None):
            if section == "session_summary":
                return self.saved.get(key, default)
            return real_get(section, key, default)

        def save(section_values) -> bool:
            self.writes.append(dict(section_values["session_summary"]))
            if self.fail:
                raise OSError("config.toml is read-only")
            self.saved.update(section_values["session_summary"])
            return True

        monkeypatch.setattr(settings_screen, "get_cli_setting", get)
        monkeypatch.setattr(settings_screen, "save_settings_to_cli_config", save)


def _app():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": _FAKE_KEY}}
    return app


async def _open_session_summary(host, pilot):
    screen = await _open_providers(host, pilot)
    screen.query_one(f"#{_DISCLOSURE}", Collapsible).collapsed = False
    await _settle(host, pilot)
    return (
        screen,
        screen.query_one("#settings-session-summary-enabled", Checkbox),
        screen.query_one("#settings-session-summary-duration", Input),
    )


async def _type_duration(pilot, field: Input, text: str) -> None:
    field.focus()
    await pilot.pause()
    await pilot.press("end", *["backspace"] * len(field.value), *text)
    await pilot.pause()
    assert field.value == text


def _title(screen) -> str:
    return str(screen.query_one(f"#{_DISCLOSURE}", Collapsible).title)


@pytest.mark.asyncio
@private_profile_test
async def test_the_duration_saves_on_enter_and_shows_what_was_saved(
    request, monkeypatch
):
    config = _Config(monkeypatch)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen, checkbox, field = await _open_session_summary(host, pilot)

        checkbox.focus()
        await pilot.pause()
        await pilot.press("space")
        await _settle(host, pilot)
        assert config.writes == [{"enabled": True, "duration_seconds": 3}]
        assert _title(screen) == f"{_TITLE}On · 3 s"

        await _type_duration(pilot, field, "99")
        await _settle(host, pilot)
        assert len(config.writes) == 1, "typing writes nothing"

        await pilot.press("enter")
        await _settle(host, pilot)
        assert config.writes[1:] == [{"enabled": True, "duration_seconds": 30}]
        assert field.value == "30"
        assert _title(screen) == f"{_TITLE}On · 30 s"
        assert not screen._settings_drafts


@pytest.mark.asyncio
@private_profile_test
async def test_the_duration_saves_when_the_field_is_left(request, monkeypatch):
    config = _Config(monkeypatch)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        _screen, _checkbox, field = await _open_session_summary(host, pilot)
        await _type_duration(pilot, field, "7")
        assert config.writes == []

        await pilot.press("tab")
        await _settle(host, pilot)

        assert not field.has_focus
        assert config.writes == [{"enabled": False, "duration_seconds": 7}]
        assert field.value == "7"


@pytest.mark.asyncio
@private_profile_test
async def test_a_failed_save_puts_the_saved_values_back(request, monkeypatch):
    config = _Config(monkeypatch)
    config.fail = True
    host = _SettingsCssHarness(_app(), "settings")
    toasts: list[str] = []

    async with host.run_test(size=_SIZE) as pilot:
        screen, checkbox, field = await _open_session_summary(host, pilot)
        monkeypatch.setattr(
            screen, "notify", lambda message, **_options: toasts.append(message)
        )
        await _type_duration(pilot, field, "12")
        await pilot.press("enter")
        await _settle(host, pilot)

        assert config.writes == [{"enabled": False, "duration_seconds": 12}]
        assert field.value == "3"
        assert not checkbox.value
        assert _title(screen) == f"{_TITLE}Off"
        assert toasts == ["Could not save session summary settings"]


@pytest.mark.asyncio
@private_profile_test
async def test_the_inspector_says_both_controls_apply_at_once(request, monkeypatch):
    _Config(monkeypatch)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen, checkbox, field = await _open_session_summary(host, pilot)
        for control, name, key in (
            (checkbox, "Session summary on quit", "session_summary.enabled"),
            (field, "Summary duration", "session_summary.duration_seconds"),
        ):
            control.focus()
            await pilot.pause()
            rows = dict(screen._provider_field_guidance_rows())
            assert rows["Focused setting"] == name
            assert rows["Saved as"] == key
            assert rows["Save"] == settings_screen.INSTANT_APPLY_BEHAVIOR_COPY


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(("hand_edited", "used"), [(99, "30"), ("abc", "3")])
async def test_a_hand_edited_duration_is_composed_as_a_quit_reads_it(
    request, monkeypatch, hand_edited, used
):
    """Fix round 1, finding 8: the title was built from the saved values as
    a quit reads them ("On · 30 s") over a field composed from the raw one
    (99), which corrected itself only when the field was left."""
    config = _Config(monkeypatch)
    config.saved = {"enabled": True, "duration_seconds": hand_edited}
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen, checkbox, field = await _open_session_summary(host, pilot)

        assert checkbox.value is True
        assert field.value == used
        assert _title(screen) == f"{_TITLE}On · {used} s"
        assert config.writes == []


@pytest.mark.asyncio
@private_profile_test
async def test_a_write_the_checkbox_started_leaves_newer_typed_text_alone(
    request, monkeypatch
):
    """Fix round 1, finding 14: every finished write put the duration it
    carried back in the field, also over text typed since the checkbox
    started it (a slow disk, an encrypted config): "3" then "5" became 3."""
    config = _Config(monkeypatch)
    release = threading.Event()
    fake_save = settings_screen.save_settings_to_cli_config

    def held_save(section_values) -> bool:
        release.wait(10)
        return fake_save(section_values)

    monkeypatch.setattr(settings_screen, "save_settings_to_cli_config", held_save)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        _screen, checkbox, field = await _open_session_summary(host, pilot)
        checkbox.focus()
        await pilot.pause()
        await pilot.press("space")  # the write starts, carrying 3 s
        await pilot.pause()
        field.focus()
        await pilot.pause()
        await pilot.press("end", "5")
        await pilot.pause()
        assert field.value == "35"

        release.set()
        await _settle(host, pilot)

        assert config.writes == [{"enabled": True, "duration_seconds": 3}]
        assert checkbox.value
        assert field.value == "35"

        # Committed, the typed text is saved and shown as a quit reads it.
        await pilot.press("enter")
        await _settle(host, pilot)
        assert config.writes[1:] == [{"enabled": True, "duration_seconds": 30}]
        assert field.value == "30"


@pytest.mark.asyncio
@private_profile_test
async def test_a_write_the_checkbox_started_leaves_an_emptied_field_alone(
    request, monkeypatch
):
    """Fix round 2, finding 3: an empty field reads as the default 3, so a
    write that sent 3 took it for the duration it carried and put "3" back
    mid-edit: "5" typed next became "35", and Enter saved 30."""
    config = _Config(monkeypatch)
    release = threading.Event()
    fake_save = settings_screen.save_settings_to_cli_config

    def held_save(section_values) -> bool:
        release.wait(10)
        return fake_save(section_values)

    monkeypatch.setattr(settings_screen, "save_settings_to_cli_config", held_save)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        _screen, checkbox, field = await _open_session_summary(host, pilot)
        checkbox.focus()
        await pilot.pause()
        await pilot.press("space")  # the write starts, carrying 3 s
        await pilot.pause()
        field.focus()
        await pilot.pause()
        await pilot.press("end", "backspace")
        await pilot.pause()
        assert field.value == ""

        release.set()
        await _settle(host, pilot)
        assert config.writes == [{"enabled": True, "duration_seconds": 3}]
        assert field.value == ""

        await pilot.press("5", "enter")
        await _settle(host, pilot)
        assert config.writes[1:] == [{"enabled": True, "duration_seconds": 5}]
        assert field.value == "5"

        # Emptied and committed, the field reads the 3 s it saved (C8).
        await pilot.press("backspace", "enter")
        await _settle(host, pilot)
        assert config.writes[2:] == [{"enabled": True, "duration_seconds": 3}]
        assert field.value == "3"
