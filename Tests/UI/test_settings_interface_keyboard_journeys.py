"""Interface settings retain visible keyboard controls under production CSS."""

import pytest
from textual.widgets import Button, Checkbox, Input

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_provider_keyboard_journeys import _settle, _tab_to
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from Tests.UI.theme_editor_helpers import open_theme_editor


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(190, 55), (80, 24)])
@pytest.mark.parametrize("category", ["Appearance", "Theme", "Splash Screen"])
@private_profile_test
async def test_interface_controls_are_keyboard_reachable_and_painted(
    request, theme, size, category
):
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, category)
        # TASK-32948: Theme opens on the picker; Clone swaps the editor in and
        # its controls are walked here. The picker's own controls join this
        # walk once its layout lands (plan Task 7 bounds the columns).
        scopes = {
            "Appearance": ["#settings-appearance-card"],
            "Theme": ["#settings-theme-editor-view"],
            "Splash Screen": ["#settings-splash-card"],
        }[category]
        failures = []
        for scope in scopes:
            if scope == "#settings-theme-editor-view":
                await open_theme_editor(host, pilot)
            root = host.screen.query_one(scope)
            controls = [
                control
                for control in root.query(
                    "Button, Input, Select, Checkbox, OptionList, Tree"
                )
                if not control.disabled and control.display
            ]
            assert controls
            for control in controls:
                # Select internals are focus proxies, not separate fields.
                if not control.id or control.has_class("select-overlay"):
                    continue
                try:
                    # Scoped: picker and editor both carry #settings-theme-clone/-new.
                    await _tab_to(host, pilot, f"{scope} #{control.id}")
                    painted = _painted(host, control)
                    if isinstance(control, Button):
                        assert str(control.label) in painted, (control.id, painted)
                    if isinstance(control, Input) and control.value:
                        assert control.value in painted, (
                            control.id,
                            control.value,
                            painted,
                        )
                    if isinstance(control, Checkbox):
                        assert "X" in painted, (control.id, painted)
                    if control.id == "settings-theme-preset-target":
                        assert "Primary" in painted, painted
                    if control.id == "settings-splash-default-select":
                        assert "Random" in painted, painted
                except AssertionError as exc:
                    failures.append(str(exc))
        if category == "Theme":
            swatch = await _tab_to(host, pilot, "#settings-theme-preset-Blues-0")
            await pilot.press("enter")
            await _settle(host, pilot)
            editor = host.screen.query_one("#settings-theme-editor")
            assert (
                editor.current_theme_data["primary"] == editor.COLOR_PRESETS["Blues"][0]
            )
            final_control = swatch
        else:
            final_control = host.screen.focused
        await pilot.resize_terminal(
            80 if size[0] == 190 else 190, 24 if size[0] == 190 else 55
        )
        await _settle(host, pilot)
        assert host.screen.focused is final_control
        try:
            await _tab_to(host, pilot, f"#{final_control.id}")
        except AssertionError as exc:
            failures.append(str(exc))
        assert not failures, "\n".join(failures)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@private_profile_test
async def test_appearance_keyboard_preview_revert_and_real_save(request, theme):
    import tomllib
    from pathlib import Path

    from Tests.UI.test_settings_provider_keyboard_journeys import _edit, _revert
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

    app = _build_test_app()
    host = _StyledDestinationHarness(app, "settings")
    host.theme = theme
    path = Path(config.get_cli_config_path())
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Appearance")
        before = path.read_bytes()
        screen = host.screen
        original = screen.query_one("#settings-appearance-font-size", Input).value
        await _edit(host, pilot, "#settings-appearance-font-size", "99")
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert screen._category_has_unsaved_changes(SettingsCategoryId.APPEARANCE)
        assert path.read_bytes() == before
        await _revert(host, pilot, discard=False)
        assert screen.query_one("#settings-appearance-font-size", Input).value == "99"
        await _revert(host, pilot, discard=True)
        assert (
            screen.query_one("#settings-appearance-font-size", Input).value == original
        )
        assert path.read_bytes() == before
        await _edit(host, pilot, "#settings-appearance-font-size", "18")
        await _tab_to(host, pilot, "#settings-preview-appearance")
        await pilot.press("enter")
        await _settle(host, pilot)
        assert path.read_bytes() == before
        await _category(host, pilot, "Overview")
        await _category(host, pilot, "Appearance")
        assert screen.query_one("#settings-appearance-font-size", Input).value == "18"
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert not screen._category_has_unsaved_changes(SettingsCategoryId.APPEARANCE)
        saved = tomllib.loads(path.read_text())
        assert saved["web_server"]["font_size"] == 18
        prior = tomllib.loads(before.decode())
        for key in prior.keys() | saved.keys():
            if key not in {"general", "web_server", "appearance", "library"}:
                assert saved.get(key) == prior.get(key), key
