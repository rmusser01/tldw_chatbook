from types import SimpleNamespace

import pytest
from textual import on
from textual.app import ComposeResult
from textual.widgets import Button, OptionList

from Tests.private_profile import private_profile_test
from Tests.textual_test_harness import IsolatedWidgetTestApp
from tldw_chatbook.css.Themes import theme_catalog as tc
from tldw_chatbook.css.Themes.themes import ALL_THEMES
from tldw_chatbook.Widgets.settings_theme_picker import ThemePicker
from tldw_chatbook.Widgets.theme_preview import ThemePreview


def _app(*widgets):
    def compose() -> ComposeResult:
        yield from widgets

    return IsolatedWidgetTestApp(compose)


@pytest.mark.asyncio
@private_profile_test
async def test_theme_preview_paints_rows_from_colours(request):
    preview = ThemePreview("pv")
    async with _app(preview).run_test(size=(80, 20)) as pilot:
        preview.paint({"panel": "#112233", "foreground": "#EEEEEE", "accent": "#FF8800", "background": "#000000"})
        await pilot.pause()
        rail = preview.query_one("#pv-rail")
        assert rail.styles.background.hex.upper() == "#112233"
        assert "[ Send ]" in str(preview.query_one("#pv-accent").render())


@pytest.fixture
def config_writes(monkeypatch, tmp_path):
    calls = []
    state = {"launch": "textual-dark"}

    def fake_apply(mutation):
        calls.append(mutation)
        state["launch"] = mutation["general"]["default_theme"]
        return SimpleNamespace(file_replaced=True, caches_reloaded=True)

    monkeypatch.setattr(tc, "_apply_config_mutation", fake_apply)
    monkeypatch.setattr(tc, "current_launch_default", lambda: state["launch"])
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.settings_theme_picker.get_user_themes_dir", lambda: tmp_path
    )
    # The picker imported the name, so patch its copy too.
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.settings_theme_picker.current_launch_default", lambda: state["launch"]
    )
    return calls


async def _picker_app(size=(160, 45)):
    picker = ThemePicker(id="settings-theme-picker")
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    return app, picker


def _option_ids(picker):
    lst = picker.query_one("#settings-theme-list", OptionList)
    return [lst.get_option_at_index(i).id for i in range(lst.option_count)]


@pytest.mark.asyncio
@private_profile_test
async def test_rows_show_markers_as_words(request, config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        lst = picker.query_one("#settings-theme-list", OptionList)
        active = next(lst.get_option_at_index(i) for i in range(lst.option_count) if lst.get_option_at_index(i).id == app.theme)
        assert "active" in str(active.prompt) and "launch" in str(active.prompt)


@pytest.mark.asyncio
@private_profile_test
async def test_filter_narrows_and_enter_uses(request, config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        picker.query_one("#settings-theme-filter").focus()
        await pilot.press(*"apricot")
        await pilot.pause()
        assert "apricot" in _option_ids(picker)
        assert "nord" not in _option_ids(picker)
        await pilot.press("enter")          # filter -> list
        await pilot.pause()
        assert app.focused.id == "settings-theme-list"
        await pilot.press("enter")          # Use
        await pilot.pause()
        assert app.theme == picker.highlighted_id
        assert config_writes[-1] == {"general": {"default_theme": app.theme}}
        assert picker.query_one("#settings-theme-revert", Button).display


@pytest.mark.asyncio
@private_profile_test
async def test_filter_no_match_enter_is_inert(request, config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        before = app.theme
        picker.query_one("#settings-theme-filter").focus()
        await pilot.press(*"zzzqqq", "enter")
        await pilot.pause()
        assert app.theme == before and config_writes == []
        assert picker.query_one("#settings-theme-empty").display
        assert "No themes match 'zzzqqq'" in str(picker.query_one("#settings-theme-empty").render())


@pytest.mark.asyncio
@private_profile_test
async def test_highlight_repaints_preview_not_app(request, config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        before = app.theme
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        await pilot.press("down", "down")
        await pilot.pause()
        assert app.theme == before
        assert picker.highlighted_id != before
        assert (
            display_name_for(picker, picker.highlighted_id)
            in str(picker.query_one("#settings-theme-card-title").render())
        )


def display_name_for(picker, theme_id):
    return next(e.display_name for e in picker.entries if e.id == theme_id)


@pytest.mark.asyncio
@private_profile_test
async def test_ansi_theme_preview_paints_resolved_rgb(request, config_writes):
    # ansi-dark's colours resolve to ANSI names ("ansi_default", ...), which
    # have no RGB of their own; theme_catalog._colour_hex falls back to
    # #808080 for those. Regression guard for fix round 1: highlighting an
    # ANSI theme used to leave ThemePreview showing the PREVIOUS theme's
    # colours, since Color.parse("ANSI_DEFAULT") silently failed inside
    # ThemePreview.paint's own try/except.
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.highlighted = lst.get_option_index("ansi-dark")
        await pilot.pause()
        entry = next(e for e in picker.entries if e.id == "ansi-dark")
        panel = dict(entry.colours)["panel"]
        rail = picker.query_one("#settings-theme-picker-preview-rail")
        assert rail.styles.background.hex.upper() == panel


@pytest.mark.asyncio
@private_profile_test
async def test_try_then_use_then_revert_restores_original(request, config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        original = app.theme
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        await pilot.press("down", "t")      # Try
        await pilot.pause()
        tried = app.theme
        assert tried != original and config_writes == []
        await pilot.press("down", "enter")  # Use another
        await pilot.pause()
        picker.query_one("#settings-theme-revert", Button).press()
        await pilot.pause()
        assert app.theme == original
        assert config_writes[-1] == {"general": {"default_theme": "textual-dark"}}
        assert not picker.query_one("#settings-theme-revert", Button).display


@pytest.mark.asyncio
@private_profile_test
async def test_up_on_first_row_returns_to_filter(request, config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        lst.highlighted = lst.get_option_index(_option_ids(picker)[1])  # first enabled row
        await pilot.press("up")
        await pilot.pause()
        assert app.focused.id == "settings-theme-filter"


@pytest.mark.asyncio
@private_profile_test
async def test_markers_follow_external_theme_change(request, config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        app.theme = "nord"                  # e.g. the palette
        await pilot.pause()
        assert next(e for e in picker.entries if e.id == "nord").is_active


class _CaptureEditApp(IsolatedWidgetTestApp):
    def __init__(self, compose):
        super().__init__(compose)
        self.edits: list[tuple[str, str]] = []

    @on(ThemePicker.EditRequested)
    def _capture(self, message: ThemePicker.EditRequested) -> None:
        self.edits.append((message.mode, message.theme_id))


@pytest.mark.asyncio
@private_profile_test
async def test_clone_and_new_post_edit_requested(request, config_writes):
    picker = ThemePicker(id="settings-theme-picker")

    def compose() -> ComposeResult:
        yield picker

    app = _CaptureEditApp(compose)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        picker.query_one("#settings-theme-list", OptionList).focus()
        await pilot.press("c", "n")
        await pilot.pause()
        assert [mode for mode, _ in app.edits] == ["clone", "new"]
        assert app.edits[0][1] == picker.highlighted_id


@pytest.mark.asyncio
@private_profile_test
async def test_use_toast_when_persist_fails(request, monkeypatch, config_writes):
    monkeypatch.setattr(tc, "_apply_config_mutation", lambda m: SimpleNamespace(file_replaced=False, caches_reloaded=False))
    app, picker = await _picker_app()
    notes = []
    app.notify = lambda message, **kw: notes.append(message)
    async with app.run_test(size=(160, 45)) as pilot:
        picker.query_one("#settings-theme-list").focus()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert any("launch default was not saved" in n for n in notes)
