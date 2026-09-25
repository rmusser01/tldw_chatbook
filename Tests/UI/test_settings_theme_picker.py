from types import SimpleNamespace

import pytest
from textual import on
from textual.app import ComposeResult
from textual.content import Content
from textual.theme import Theme
from textual.widgets import Button, OptionList

from Tests.private_profile import private_profile_test
from Tests.textual_test_harness import IsolatedWidgetTestApp
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.css.Themes import theme_catalog as tc
from tldw_chatbook.css.Themes.themes import ALL_THEMES
from tldw_chatbook.Widgets.settings_theme_editor import THEMES_UNAVAILABLE_LABEL
from tldw_chatbook.Widgets.settings_theme_picker import ThemePicker
from tldw_chatbook.Widgets.theme_preview import ThemePreview

_FILE_ACTION_BUTTON_IDS = (
    "#settings-theme-picker-edit",
    "#settings-theme-picker-rename",
    "#settings-theme-picker-delete",
    "#settings-theme-picker-export",
)


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


async def _picker_app(size=(160, 45), list_user_names=None):
    picker = ThemePicker(id="settings-theme-picker", list_user_names=list_user_names)
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
        # First enabled row (an empty YOUR THEMES now shows a disabled "(none yet)").
        lst.highlighted = lst.get_option_index(next(i for i in _option_ids(picker) if i))
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
        self.renames: list[str] = []
        self.deletes: list[str] = []
        self.exports: list[str] = []
        self.imports = 0

    @on(ThemePicker.EditRequested)
    def _capture(self, message: ThemePicker.EditRequested) -> None:
        self.edits.append((message.mode, message.theme_id))

    @on(ThemePicker.RenameRequested)
    def _capture_rename(self, message: ThemePicker.RenameRequested) -> None:
        self.renames.append(message.theme_id)

    @on(ThemePicker.DeleteRequested)
    def _capture_delete(self, message: ThemePicker.DeleteRequested) -> None:
        self.deletes.append(message.theme_id)

    @on(ThemePicker.ExportRequested)
    def _capture_export(self, message: ThemePicker.ExportRequested) -> None:
        self.exports.append(message.theme_id)

    @on(ThemePicker.ImportRequested)
    def _capture_import(self, message: ThemePicker.ImportRequested) -> None:
        self.imports += 1


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


@pytest.mark.asyncio
@private_profile_test
async def test_use_toast_warns_when_cache_reload_fails(request, monkeypatch, config_writes):
    # Fix round 1 (TASK-32948 Task 5): the picker's Use toast must carry the
    # same cache-refresh-failed warning as the palette's -- both persist
    # through use_theme/use_theme_toast, one shared toast (spec §4).
    monkeypatch.setattr(tc, "_apply_config_mutation", lambda m: SimpleNamespace(file_replaced=True, caches_reloaded=False))
    app, picker = await _picker_app()
    notes = []
    app.notify = lambda message, **kw: notes.append((message, kw.get("severity")))
    async with app.run_test(size=(160, 45)) as pilot:
        picker.query_one("#settings-theme-list").focus()
        await pilot.press("down", "enter")  # Use (persisted, cache reload fails)
        await pilot.pause()
        assert any(
            "is now your theme" in message
            and "configuration refresh failed — reopen Settings to refresh" in message
            and severity == "warning"
            for message, severity in notes
        )


@pytest.mark.asyncio
@private_profile_test
async def test_mouse_click_highlights_but_does_not_use(request, config_writes):
    # Spec D2: highlight only repaints the preview. OptionList's own click
    # handler highlights AND selects (= Use); the picker's must only highlight.
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        before = app.theme
        lst = picker.query_one("#settings-theme-list", OptionList)
        start = lst.highlighted
        for y in range(1, 6):  # the first visible non-header, non-highlighted row
            await pilot.click("#settings-theme-list", offset=(3, y))
            await pilot.pause()
            if lst.highlighted != start:
                break
        assert lst.highlighted != start, "no clickable row found"
        assert app.theme == before and config_writes == []
        assert not picker.query_one("#settings-theme-revert", Button).display
        assert display_name_for(picker, picker.highlighted_id) in str(
            picker.query_one("#settings-theme-card-title").render()
        )


@pytest.mark.asyncio
@private_profile_test
async def test_revert_warns_when_launch_default_not_restored(request, monkeypatch, config_writes):
    app, picker = await _picker_app()
    notes = []
    app.notify = lambda message, **kw: notes.append(message)
    async with app.run_test(size=(160, 45)) as pilot:
        original = app.theme
        picker.query_one("#settings-theme-list").focus()
        await pilot.press("down", "enter")  # Use (persisted)
        await pilot.pause()
        monkeypatch.setattr(tc, "_apply_config_mutation", lambda m: SimpleNamespace(file_replaced=False, caches_reloaded=False))
        picker.query_one("#settings-theme-revert", Button).press()
        await pilot.pause()
        assert app.theme == original
        assert "Reverted the theme; the launch default was not restored" in notes


@pytest.mark.asyncio
@private_profile_test
async def test_pending_revert_survives_a_new_picker_instance(request, config_writes):
    # R9: the pane is recomposed on every category switch; the Revert chip
    # must come back with the new picker (spec §5 "rest of the session").
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        original = app.theme
        picker.query_one("#settings-theme-list").focus()
        await pilot.press("down", "enter")  # Use
        await pilot.pause()
        await picker.remove()
        fresh = ThemePicker(id="settings-theme-picker")
        await app.screen.mount(fresh)
        await pilot.pause()
        revert = fresh.query_one("#settings-theme-revert", Button)
        assert revert.display and str(revert.label) == f"Revert to {tc.display_name(original)}"
        revert.press()
        await pilot.pause()
        assert app.theme == original and not revert.display


@pytest.mark.asyncio
@private_profile_test
async def test_revert_label_names_the_launch_default_when_it_differs(request, config_writes):
    # Task 5 (TASK-32948 PR 2): a persisted Use captures where a Revert would
    # land -- normally just the previously-active theme, but when the active
    # theme and the (already-persisted) launch default disagree, name both.
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        app.theme = "nord"  # active diverges from the launch default (textual-dark)
        await pilot.pause()
        picker.highlighted_id = "apricot"
        picker.use_highlighted()
        await pilot.pause()
        revert = picker.query_one("#settings-theme-revert", Button)
        assert str(revert.label) == (
            f"Revert to {tc.display_name('nord')} (launch: {tc.display_name('textual-dark')})"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_revert_label_stays_plain_for_a_try_even_when_launch_differs(request, config_writes):
    # A Try never persists, so the chip never needs the launch-default
    # qualifier even when active and launch disagree.
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        app.theme = "nord"
        await pilot.pause()
        picker.highlighted_id = "apricot"
        picker.try_highlighted()
        await pilot.pause()
        revert = picker.query_one("#settings-theme-revert", Button)
        assert str(revert.label) == f"Revert to {tc.display_name('nord')}"


@pytest.mark.asyncio
@private_profile_test
async def test_yours_actions_only_show_for_your_themes(request, config_writes):
    app, picker = await _picker_app(list_user_names=lambda: {"mine"})
    app.register_theme(Theme(name="mine", primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.highlighted = lst.get_option_index("mine")
        await pilot.pause()
        for button_id in _FILE_ACTION_BUTTON_IDS:
            assert picker.query_one(button_id, Button).display, button_id
        lst.highlighted = lst.get_option_index("nord")
        await pilot.pause()
        for button_id in _FILE_ACTION_BUTTON_IDS:
            assert not picker.query_one(button_id, Button).display, button_id


@pytest.mark.asyncio
@private_profile_test
async def test_rename_delete_export_edit_post_requests(request, config_writes):
    picker = ThemePicker(id="settings-theme-picker", list_user_names=lambda: {"mine"})

    def compose() -> ComposeResult:
        yield picker

    app = _CaptureEditApp(compose)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    app.register_theme(Theme(name="mine", primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        lst.highlighted = lst.get_option_index("mine")
        await pilot.pause()
        await pilot.press("r", "delete", "e")
        await pilot.pause()
        picker.query_one("#settings-theme-picker-export", Button).press()
        await pilot.pause()
        assert app.renames == ["mine"]
        assert app.deletes == ["mine"]
        assert app.edits == [("edit", "mine")]
        assert app.exports == ["mine"]


@pytest.mark.asyncio
@private_profile_test
async def test_keys_ignored_for_catalog_themes(request, config_writes):
    picker = ThemePicker(id="settings-theme-picker", list_user_names=lambda: {"mine"})

    def compose() -> ComposeResult:
        yield picker

    app = _CaptureEditApp(compose)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    app.register_theme(Theme(name="mine", primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        lst.highlighted = lst.get_option_index("nord")
        await pilot.pause()
        await pilot.press("r", "delete", "e")
        await pilot.pause()
        assert app.renames == [] and app.deletes == [] and app.edits == []


@pytest.mark.asyncio
@private_profile_test
async def test_pause_row_disables_file_actions(request, config_writes):
    def raiser():
        raise RecoveryRequired("x")

    app, picker = await _picker_app(list_user_names=raiser)
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        assert picker.files_available is False
        lst = picker.query_one("#settings-theme-list", OptionList)
        labels = [str(lst.get_option_at_index(i).prompt) for i in range(lst.option_count)]
        assert THEMES_UNAVAILABLE_LABEL in labels
        for button_id in _FILE_ACTION_BUTTON_IDS:
            button = picker.query_one(button_id, Button)
            assert button.disabled, button_id
            assert button.tooltip == THEMES_UNAVAILABLE_LABEL, button_id
        before = app.theme
        lst.highlighted = lst.get_option_index("nord")
        lst.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.theme == "nord" and app.theme != before


@pytest.mark.asyncio
@private_profile_test
async def test_empty_your_themes_says_none_yet(request, config_writes):
    """Spec §9 / task-32945: an empty YOUR THEMES group shows an inert
    "(none yet)" row (the retired editor tree used to own this)."""
    app, picker = await _picker_app(list_user_names=set)
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        lst = picker.query_one("#settings-theme-list", OptionList)
        prompts = [str(lst.get_option_at_index(i).prompt) for i in range(3)]
        assert prompts[:2] == ["YOUR THEMES", "(none yet)"]
        assert lst.get_option_at_index(1).disabled
        # The filter hides it: no match is not the same as no themes.
        picker.query_one("#settings-theme-filter").value = "nord"
        await pilot.pause()
        prompts = [str(lst.get_option_at_index(i).prompt) for i in range(lst.option_count)]
        assert "(none yet)" not in prompts


@pytest.mark.asyncio
@private_profile_test
async def test_pause_keeps_last_known_your_themes(request, config_writes):
    """R27 (5): a pause after a good listing must not relabel your themes
    as shipped -- the last successfully listed names keep their origin."""
    state = {"paused": False}

    def lister():
        if state["paused"]:
            raise RecoveryRequired("x")
        return {"mine"}

    app, picker = await _picker_app(list_user_names=lister)
    app.register_theme(Theme(name="mine", primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        state["paused"] = True
        picker.refresh_catalog()
        await pilot.pause()
        assert picker.files_available is False
        origin = {e.id: e.origin for e in picker.entries}["mine"]
        assert origin == "yours"


@pytest.mark.asyncio
@private_profile_test
async def test_lister_oserror_reads_as_no_user_themes(request, config_writes):
    """R27 (6): an OSError from the lister is "no user themes", not a crash
    and not a backup/recovery pause."""

    def broken():
        raise OSError(13, "Permission denied")

    app, picker = await _picker_app(list_user_names=broken)
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        assert picker.files_available is True
        assert not [e for e in picker.entries if e.origin == "yours"]
        assert picker.entries  # the catalog still lists


# -- PR 3 Task 1: an unreadable file is listed, and only Delete works -----------

_UNREADABLE_DISABLED_IDS = (
    "#settings-theme-use",
    "#settings-theme-try",
    "#settings-theme-picker-clone",
    "#settings-theme-picker-new",  # R29
    "#settings-theme-picker-edit",
    "#settings-theme-picker-rename",
    "#settings-theme-picker-export",
)


@pytest.mark.asyncio
@private_profile_test
async def test_unreadable_file_is_listed_and_only_delete_works(request, config_writes):
    error = "missing [colors].primary [b]not markup[/b]"
    picker = ThemePicker(
        id="settings-theme-picker",
        list_user_names=set,
        list_unreadable=lambda: {"a": error},
    )

    def compose() -> ComposeResult:
        yield picker

    app = _CaptureEditApp(compose)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        before = app.theme
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        lst.highlighted = lst.get_option_index("a")
        await pilot.pause()
        assert "(unreadable)" in str(lst.get_option_at_index(lst.highlighted).prompt)
        title = picker.query_one("#settings-theme-card-title")
        assert str(title.render()) == "A (unreadable)"
        card_error = picker.query_one("#settings-theme-card-error")
        assert card_error.display and str(card_error.render()) == error
        for button_id in _UNREADABLE_DISABLED_IDS:
            button = picker.query_one(button_id, Button)
            assert button.disabled, button_id
            assert Content.from_markup(button.tooltip).plain == (
                f"This theme file can't be read: {error}"
            ), button_id
        delete = picker.query_one("#settings-theme-picker-delete", Button)
        assert delete.display and not delete.disabled

        await pilot.press("enter", "t", "c", "n", "e", "r")
        await pilot.pause()
        assert app.theme == before
        assert app.edits == [] and app.renames == []
        assert config_writes == []

        await pilot.press("delete")
        await pilot.pause()
        assert app.deletes == ["a"]

        # A readable theme hides the error line and re-enables the actions.
        lst.highlighted = lst.get_option_index("nord")
        await pilot.pause()
        assert not card_error.display
        assert not picker.query_one("#settings-theme-use", Button).disabled
        assert picker.query_one("#settings-theme-use", Button).tooltip is None


@pytest.mark.asyncio
@private_profile_test
async def test_unreadable_error_with_markup_renders_its_tooltip_literally(request, config_writes):
    """Fix round 1: the error quotes untrusted file content; a tooltip parses markup."""
    error = "unknown colour '[/mismatched]'"
    picker = ThemePicker(
        id="settings-theme-picker", list_user_names=set, list_unreadable=lambda: {"a": error}
    )
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.highlighted = lst.get_option_index("a")
        await pilot.pause()
        use = picker.query_one("#settings-theme-use", Button)
        assert "[/mismatched]" in Content.from_markup(use.tooltip).plain
        # The tooltip actually renders on hover.
        await pilot.hover("#settings-theme-picker-delete")
        await pilot.hover("#settings-theme-use")
        await pilot.pause(0.6)
        assert "[/mismatched]" in str(picker.query_one("#settings-theme-card-error").render())


def _capture_app(picker):
    def compose() -> ComposeResult:
        yield picker

    app = _CaptureEditApp(compose)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    return app


@pytest.mark.asyncio
@private_profile_test
async def test_import_button_and_i_key_post_import_requested(request, config_writes):
    """TASK-32948 PR 3 Task 2: Import… sits next to New; `i` on the list."""
    picker = ThemePicker(id="settings-theme-picker")
    app = _capture_app(picker)
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        button = picker.query_one("#settings-theme-picker-import", Button)
        assert str(button.label) == "Import…"
        assert button.parent is picker.query_one("#settings-theme-picker-new").parent
        picker.query_one("#settings-theme-list", OptionList).focus()
        await pilot.press("i")
        await pilot.click("#settings-theme-picker-import")
        await pilot.pause()
        assert app.imports == 2


@pytest.mark.asyncio
@private_profile_test
async def test_import_is_disabled_while_paused(request, config_writes):
    def raiser():
        raise RecoveryRequired("x")

    picker = ThemePicker(id="settings-theme-picker", list_user_names=raiser)
    app = _capture_app(picker)
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        button = picker.query_one("#settings-theme-picker-import", Button)
        assert button.disabled
        assert button.tooltip == THEMES_UNAVAILABLE_LABEL
        picker.query_one("#settings-theme-list", OptionList).focus()
        await pilot.press("i")
        await pilot.pause()
        assert app.imports == 0
