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


def _names(list_user_names):
    """Adapt a names-only lister to the picker's one ``list_themes`` hook."""
    return lambda: (list_user_names(), {})


async def _picker_app(size=(160, 45), list_user_names=None):
    picker = ThemePicker(
        id="settings-theme-picker",
        list_themes=_names(list_user_names) if list_user_names else None,
    )
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    return app, picker


async def _landed(app, pilot):
    """Let the picker's folder scan (a thread worker, TASK-32957) land."""
    await app.workers.wait_for_complete()
    await pilot.pause()


def _option_ids(picker):
    lst = picker.query_one("#settings-theme-list", OptionList)
    return [lst.get_option_at_index(i).id for i in range(lst.option_count)]


@pytest.mark.asyncio
@private_profile_test
async def test_rows_show_markers_as_words(request, config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        lst = picker.query_one("#settings-theme-list", OptionList)
        active = next(lst.get_option_at_index(i) for i in range(lst.option_count) if lst.get_option_at_index(i).id == app.theme)
        assert "active" in str(active.prompt) and "launch" in str(active.prompt)


@pytest.mark.asyncio
@private_profile_test
async def test_filter_narrows_and_enter_uses(request, config_writes):
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
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


@pytest.mark.asyncio
@private_profile_test
async def test_shared_name_shows_origin_once_and_filter_ignores_origin_words(request, config_writes):
    """P3 review M3/M4: a name shared across origins is labelled "X · built-in";
    the card title used to repeat the origin ("X · built-in  ·  light ·
    built-in") and the filter "built" listed just those few rows."""
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        shared = next(e for e in picker.entries if e.display_name.endswith(" · built-in"))
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        lst.highlighted = lst.get_option_index(shared.id)
        await pilot.pause()
        title = str(picker.query_one("#settings-theme-card-title").render())
        assert title.count("built-in") == 1, title
        picker.query_one("#settings-theme-filter").focus()
        await pilot.press(*"built")
        await pilot.pause()
        assert shared.id not in _option_ids(picker)


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
async def test_revert_leaves_a_missing_launch_default_alone(request, config_writes):
    # TASK-33061: the app started on a launch default that is not a
    # registered theme. Reverting a Use restores the active theme but must
    # not write the broken name back (the "Launch default missing" notice
    # would return), and the chip says the launch default stays put.
    tc._apply_config_mutation({"general": {"default_theme": "ghost_theme"}})
    config_writes.clear()
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        picker.highlighted_id = "apricot"
        picker.use_highlighted()
        await pilot.pause()
        revert = picker.query_one("#settings-theme-revert", Button)
        assert str(revert.label) == f"Revert to {tc.display_name('textual-dark')} (launch unchanged)"
        revert.press()
        await pilot.pause()
        assert app.theme == "textual-dark"
        assert config_writes == [{"general": {"default_theme": "apricot"}}]  # the Use only


@pytest.mark.asyncio
@private_profile_test
async def test_revert_chip_hides_when_it_would_change_nothing(request, config_writes):
    # TASK-33061: a pending Revert whose target is already active (and whose
    # launch default already matches) would do nothing -- no chip.
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        revert = picker.query_one("#settings-theme-revert", Button)
        picker.highlighted_id = "nord"
        picker.try_highlighted()
        await pilot.pause()
        assert revert.display
        app.theme = "textual-dark"  # e.g. the palette puts the original back
        await pilot.pause()
        assert not revert.display
        picker.highlighted_id = "apricot"
        picker.use_highlighted()  # persisted: launch default -> apricot
        await pilot.pause()
        assert revert.display
        app.theme = "textual-dark"  # active matches, launch default still differs
        await pilot.pause()
        assert revert.display
        tc._apply_config_mutation({"general": {"default_theme": "textual-dark"}})
        picker.refresh_catalog(rescan=False)
        await pilot.pause()
        assert not revert.display


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
        await _landed(app, pilot)
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
    picker = ThemePicker(id="settings-theme-picker", list_themes=lambda: ({"mine"}, {}))

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
    picker = ThemePicker(id="settings-theme-picker", list_themes=lambda: ({"mine"}, {}))

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
        await _landed(app, pilot)
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
        await _landed(app, pilot)
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
async def test_builtin_group_has_a_user_facing_title(request, config_writes):
    """TASK-33073: Textual's own themes are grouped as BUILT-IN, not by the
    framework's name."""
    app, picker = await _picker_app(list_user_names=set)
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        lst = picker.query_one("#settings-theme-list", OptionList)
        headers = [
            str(lst.get_option_at_index(i).prompt)
            for i in range(lst.option_count)
            if lst.get_option_at_index(i).id is None
        ]
        assert any(h.startswith("BUILT-IN") for h in headers), headers
        assert not any("TEXTUAL" in h for h in headers), headers


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
        await _landed(app, pilot)
        state["paused"] = True
        picker.refresh_catalog()
        await _landed(app, pilot)
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
        await _landed(app, pilot)
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
        list_themes=lambda: (set(), {"a": error}),
    )

    def compose() -> ComposeResult:
        yield picker

    app = _CaptureEditApp(compose)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        before = app.theme
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        lst.highlighted = lst.get_option_index("unreadable:a")
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

        app.notify.reset_mock()
        await pilot.press("enter", "t", "c", "n", "e", "r")
        await pilot.pause()
        assert app.theme == before
        assert app.edits == [] and app.renames == []
        assert config_writes == []
        # R40(d): each blocked key says why, once.
        notes = [Content.from_markup(c.args[0]).plain for c in app.notify.call_args_list]
        assert notes == [f"This theme file can't be read: {error}"] * 6

        await pilot.press("delete")
        await pilot.pause()
        assert app.deletes == ["unreadable:a"]

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
        id="settings-theme-picker", list_themes=lambda: (set(), {"a": error})
    )
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.highlighted = lst.get_option_index("unreadable:a")
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
        await _landed(app, pilot)
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

    picker = ThemePicker(id="settings-theme-picker", list_themes=_names(raiser))
    app = _capture_app(picker)
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        button = picker.query_one("#settings-theme-picker-import", Button)
        assert button.disabled
        assert button.tooltip == THEMES_UNAVAILABLE_LABEL
        picker.query_one("#settings-theme-list", OptionList).focus()
        await pilot.press("i")
        await pilot.pause()
        assert app.imports == 0


@pytest.mark.asyncio
@private_profile_test
async def test_revert_and_apply_failure_toasts_show_a_markup_name_literally(request, monkeypatch, config_writes):
    """R28: an exception text quoting a theme name goes through escape_markup."""
    app, picker = await _picker_app()
    notes = []
    app.notify = lambda message, **kw: notes.append(Content.from_markup(message).plain)
    async with app.run_test(size=(160, 45)) as pilot:
        picker.query_one("#settings-theme-list").focus()
        await pilot.press("down", "enter")  # Use (persisted): arms Revert
        await pilot.pause()

        def boom(*_args, **_kwargs):
            raise ValueError("theme 'x[/]' is gone")

        monkeypatch.setattr("tldw_chatbook.Widgets.settings_theme_picker.revert_theme", boom)
        picker.query_one("#settings-theme-revert", Button).press()
        await pilot.pause()
        assert "Could not revert the theme: theme 'x[/]' is gone" in notes
        monkeypatch.setattr("tldw_chatbook.Widgets.settings_theme_picker.use_theme", boom)
        picker.use_highlighted()
        await pilot.pause()
        assert any(n.endswith(": theme 'x[/]' is gone") and n.startswith("Could not apply") for n in notes)


@pytest.mark.asyncio
@private_profile_test
async def test_one_listing_call_per_refresh(request, config_writes):
    """R40(b): readable and unreadable come from ONE read of the themes dir."""
    calls = []

    def lister():
        calls.append(1)
        return {"mine"}, {"broken": "not valid TOML"}

    picker = ThemePicker(id="settings-theme-picker", list_themes=lister)
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        calls.clear()
        picker.refresh_catalog()
        await app.workers.wait_for_complete()  # TASK-32957: the scan is a worker
        assert len(calls) == 1
        assert "unreadable:broken" in _option_ids(picker)


# -- Qodo review fixes (TASK-32948) ------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_new_after_a_zero_match_filter_starts_from_the_active_theme(request, config_writes):
    """Qodo 4104047326 / 4107495882: New is enabled with nothing listed, so it
    must do something -- start from the theme on screen."""
    picker = ThemePicker(id="settings-theme-picker")

    def compose() -> ComposeResult:
        yield picker

    app = _CaptureEditApp(compose)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        picker.query_one("#settings-theme-filter").focus()
        await pilot.press(*"zzzzqq")
        await pilot.pause()
        assert picker.highlighted_id is None
        picker.query_one("#settings-theme-picker-new", Button).press()
        await pilot.pause()
        assert app.edits == [("new", str(app.theme))]
        # Clone/Edit still need a highlighted theme.
        picker.request_edit("clone")
        await pilot.pause()
        assert len(app.edits) == 1


@pytest.mark.asyncio
@private_profile_test
async def test_revert_warns_when_the_config_refresh_fails(request, monkeypatch, config_writes):
    """Qodo 4107495934: Revert surfaces ``caches_reloaded`` like Use does."""
    app, picker = await _picker_app()
    notes = []
    app.notify = lambda message, **kw: notes.append((message, kw.get("severity")))
    async with app.run_test(size=(160, 45)) as pilot:
        picker.query_one("#settings-theme-list").focus()
        await pilot.press("down", "enter")  # Use (persisted)
        await pilot.pause()
        monkeypatch.setattr(tc, "_apply_config_mutation", lambda m: SimpleNamespace(file_replaced=True, caches_reloaded=False))
        picker.query_one("#settings-theme-revert", Button).press()
        await pilot.pause()
        assert (f"Reverted the theme; {tc.CACHE_REFRESH_FAILED}", "warning") in notes


@pytest.mark.asyncio
@private_profile_test
async def test_launch_missing_notice_strips_control_characters(request, monkeypatch, config_writes):
    """Qodo 4109320405: a hand-edited config value reaches the terminal."""
    hostile = "gone\x1b]52;c;eA==\x07"
    monkeypatch.setattr("tldw_chatbook.Widgets.settings_theme_picker.current_launch_default", lambda: hostile)
    monkeypatch.setattr(tc, "current_launch_default", lambda: hostile)
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        notice = picker.query_one("#settings-theme-launch-missing")
        assert notice.display
        text = str(notice.render())
        assert "Launch default missing: gone?]52;c;eA==?" in text
        assert text.isprintable(), repr(text)


@pytest.mark.asyncio
@private_profile_test
async def test_palette_switch_reaches_app_theme_config_and_toast(request):
    """Qodo 4104047279: drive the real command palette (no mocked config
    write): pick a theme, then check the running theme, the persisted
    launch default in this test's private config.toml, and the toast."""
    from typing import ClassVar

    import toml
    from textual.app import App
    from textual.command import CommandPalette

    from tldw_chatbook import config
    from tldw_chatbook.app import ThemeProvider

    class _PaletteApp(App):
        COMMANDS: ClassVar = {ThemeProvider}

    app = _PaletteApp()
    notes = []
    real_notify = app.notify
    app.notify = lambda message, **kw: (notes.append((message, kw.get("severity"))), real_notify(message, **kw))
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(120, 40)) as pilot:
        app.action_command_palette()
        await pilot.pause()
        assert isinstance(app.screen, CommandPalette)
        await pilot.press(*"switch to monokai pro")
        for _ in range(20):
            await pilot.pause(0.05)
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause(0.05)
        assert app.theme == "monokai_pro"
    saved = toml.load(config._get_effective_config_path())
    assert saved["general"]["default_theme"] == "monokai_pro"
    assert ("Monokai Pro is now your theme (was: Textual Dark)", "information") in notes


@pytest.mark.asyncio
@private_profile_test
async def test_theme_switches_reuse_the_last_listing(request, config_writes):
    """Qodo 4107495860: a theme change only moves the active/launch markers,
    so it must not re-scan the themes folder (the backup-scoped scan costs
    ~5-6.6 ms per file -- a 254 ms median at 50 files, measured)."""
    calls = []

    def lister():
        calls.append(1)
        return {"mine"}, {}

    picker = ThemePicker(id="settings-theme-picker", list_themes=lister)
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        calls.clear()
        app.theme = "nord"  # e.g. the palette
        await pilot.pause()
        assert next(e for e in picker.entries if e.id == "nord").is_active
        picker.query_one("#settings-theme-list").focus()
        await pilot.press("down", "t")  # Try
        await pilot.pause()
        picker.query_one("#settings-theme-revert", Button).press()
        await pilot.pause()
        assert calls == []
        picker.refresh_catalog()  # an explicit refresh (after a file action) still scans
        await app.workers.wait_for_complete()
        assert calls == [1]


@pytest.mark.asyncio
@private_profile_test
async def test_revert_label_strips_control_characters_from_the_launch_default(
    request, monkeypatch, config_writes
):
    """Review follow-up 3: the chip names the launch default, which comes
    from a hand-editable config.toml -- ESC must not reach the terminal."""
    hostile = "gone\x1b]52;c;eA==\x07"
    monkeypatch.setattr("tldw_chatbook.Widgets.settings_theme_picker.current_launch_default", lambda: hostile)
    monkeypatch.setattr(tc, "current_launch_default", lambda: hostile)
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        app.theme = "nord"
        await pilot.pause()
        picker.highlighted_id = "apricot"
        picker.use_highlighted()
        await pilot.pause()
        label = str(picker.query_one("#settings-theme-revert", Button).label)
        # TASK-33061: a launch default that is no registered theme is never
        # written back, so the chip no longer names it at all.
        assert label.endswith("(launch unchanged)"), label
        assert label.isprintable(), repr(label)


# -- TASK-32957: the folder scan runs off the UI thread ----------------------


def _gated_lister(results):
    """A lister whose calls return ``results`` in order; a result that is a
    ``threading.Event`` blocks that call until the test sets it, then returns
    the next result (or raises it, for an exception)."""
    import threading

    queue = list(results)
    lock = threading.Lock()
    returned = []

    def lister():
        with lock:
            item = queue.pop(0)
            if isinstance(item, threading.Event):
                gate, item = item, queue.pop(0)
            else:
                gate = None
        if gate is not None:
            assert gate.wait(5), "test never released the gated scan"
        returned.append(item)
        if isinstance(item, Exception):
            raise item
        return item, {}

    lister.returned = returned
    return lister


def _yours(picker):
    return {e.id for e in picker.entries if e.origin == "yours"}


async def _until(pilot, predicate):
    for _ in range(100):
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError("condition never held")


@pytest.mark.asyncio
@private_profile_test
async def test_first_open_shows_loading_row_and_stays_responsive(request, config_writes):
    """AC#2: with the first scan in flight, the registered themes are listed,
    YOUR THEMES says it's loading, and keys still move the highlight."""
    import threading

    from tldw_chatbook.Widgets.settings_theme_picker import THEMES_LOADING_LABEL

    gate = threading.Event()
    lister = _gated_lister([gate, {"mine"}])
    picker = ThemePicker(id="settings-theme-picker", list_themes=lister)
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    app.register_theme(Theme(name="mine", primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        lst = picker.query_one("#settings-theme-list", OptionList)
        labels = [str(lst.get_option_at_index(i).prompt) for i in range(lst.option_count)]
        assert THEMES_LOADING_LABEL in labels
        assert "nord" in _option_ids(picker) and not _yours(picker)
        lst.focus()
        before = picker.highlighted_id
        await pilot.press("down")
        await pilot.pause()
        assert picker.highlighted_id != before  # the UI thread is free
        gate.set()
        await _landed(app, pilot)
        await _until(pilot, lambda: _yours(picker) == {"mine"})
        labels = [str(lst.get_option_at_index(i).prompt) for i in range(lst.option_count)]
        assert THEMES_LOADING_LABEL not in labels


@pytest.mark.asyncio
@private_profile_test
async def test_stale_scan_never_overwrites_a_newer_listing(request, config_writes):
    """AC#3: a slow scan that lands after a newer one (e.g. the rescan after
    a file action) is dropped, and the newer scan's highlight wins."""
    import threading

    gate = threading.Event()
    lister = _gated_lister([{"mine"}, gate, {"old"}, {"new"}])
    picker = ThemePicker(id="settings-theme-picker", list_themes=lister)
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    for name in ("mine", "old", "new"):
        app.register_theme(Theme(name=name, primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        assert _yours(picker) == {"mine"}
        picker.refresh_catalog(highlight="old")  # blocks in the worker
        await pilot.pause()
        # The last good listing stays up while the scan is in flight.
        assert _yours(picker) == {"mine"}
        picker.refresh_catalog(highlight="new")  # a newer scan lands first
        await _until(pilot, lambda: _yours(picker) == {"new"})
        assert picker.highlighted_id == "new"
        gate.set()  # now the stale scan finishes
        await _until(pilot, lambda: len(lister.returned) == 3)
        for _ in range(10):
            await pilot.pause(0.02)
        assert _yours(picker) == {"new"}
        assert picker.highlighted_id == "new"


@pytest.mark.asyncio
@private_profile_test
async def test_pause_during_background_scan_shows_unavailable(request, config_writes):
    """AC#4: RecoveryRequired raised inside the worker still yields the
    "Theme files unavailable" row, keeps your themes' origin from the last
    good listing, and blocks file actions."""
    import threading

    gate = threading.Event()
    lister = _gated_lister([{"mine"}, gate, RecoveryRequired("x")])
    picker = ThemePicker(id="settings-theme-picker", list_themes=lister)
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    app.register_theme(Theme(name="mine", primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        picker.refresh_catalog(highlight="mine")
        await pilot.pause()
        assert picker.files_available is True  # in flight: last good state
        gate.set()
        await _landed(app, pilot)
        await _until(pilot, lambda: picker.files_available is False)
        lst = picker.query_one("#settings-theme-list", OptionList)
        labels = [str(lst.get_option_at_index(i).prompt) for i in range(lst.option_count)]
        assert THEMES_UNAVAILABLE_LABEL in labels
        assert {e.id: e.origin for e in picker.entries}["mine"] == "yours"
        assert picker.highlighted_id == "mine"
        for button_id in _FILE_ACTION_BUTTON_IDS:
            assert picker.query_one(button_id, Button).disabled, button_id
        assert picker.query_one("#settings-theme-picker-import", Button).disabled


@pytest.mark.asyncio
@private_profile_test
async def test_a_move_during_the_first_scan_is_kept_when_it_lands(request, config_writes):
    """Qodo 4116061873: the open's highlight (the active theme) must not pull
    the cursor back once the user has moved it while the scan was in flight."""
    import threading

    gate = threading.Event()
    lister = _gated_lister([gate, {"mine"}])
    picker = ThemePicker(id="settings-theme-picker", list_themes=lister)
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    app.register_theme(Theme(name="mine", primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.focus()
        start = picker.highlighted_id
        await pilot.press("down")
        await pilot.pause()
        moved = picker.highlighted_id
        assert moved != start
        gate.set()
        await _landed(app, pilot)
        await _until(pilot, lambda: _yours(picker) == {"mine"})
        await pilot.pause()
        assert picker.highlighted_id == moved
        assert lst.get_option_at_index(lst.highlighted).id == moved


@pytest.mark.asyncio
@private_profile_test
async def test_a_request_still_lands_when_the_user_did_not_move(request, config_writes):
    """The other half of 4116061873: a caller's highlight for a theme the
    in-flight scan will list (e.g. after Import) is applied when it lands."""
    import threading

    gate = threading.Event()
    lister = _gated_lister([set(), gate, {"fresh"}])
    picker = ThemePicker(id="settings-theme-picker", list_themes=lister)
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    app.register_theme(Theme(name="fresh", primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        picker.refresh_catalog(highlight="fresh")
        await pilot.pause()
        gate.set()
        await _landed(app, pilot)
        await _until(pilot, lambda: _yours(picker) == {"fresh"})
        assert picker.highlighted_id == "fresh"


@pytest.mark.asyncio
@private_profile_test
async def test_unexpected_scan_failure_keeps_the_last_listing(request, config_writes):
    """Qodo 4116061876 + 4116061867: a non-OSError failure keeps the last good
    listing (it is not "no user themes"), and its log names the scan and the
    frames but never the exception's text (R16: it may carry a path)."""
    from loguru import logger

    records = []
    sink = logger.add(lambda message: records.append(str(message)), level="ERROR")
    try:
        lister = _gated_lister([{"mine"}, ValueError("/secret/themes/mine.toml")])
        picker = ThemePicker(id="settings-theme-picker", list_themes=lister)
        app = _app(picker)
        for theme in ALL_THEMES:
            app.register_theme(theme)
        app.register_theme(Theme(name="mine", primary="#336699"))
        async with app.run_test(size=(160, 45)) as pilot:
            await _landed(app, pilot)
            picker.refresh_catalog()
            await _landed(app, pilot)
            await _until(pilot, lambda: len(lister.returned) == 2)
            await pilot.pause()
            assert _yours(picker) == {"mine"}
            assert picker.files_available is True
    finally:
        logger.remove(sink)
    failures = [r for r in records if "Saved-theme listing failed" in r]
    assert failures, records
    assert "ValueError" in failures[0] and "scan 2" in failures[0]
    assert "lister" in failures[0]  # the frame that raised
    assert "/secret/" not in "".join(records)


# -- Qodo 4116061864: the scan-landing decisions, driven directly -------------


async def _mounted_picker(pilot_body):
    lister = _gated_lister([{"mine"}])
    picker = ThemePicker(id="settings-theme-picker", list_themes=lister)
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    for name in ("mine", "other"):
        app.register_theme(Theme(name=name, primary="#336699"))
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        assert _yours(picker) == {"mine"}
        await pilot_body(picker)


@pytest.mark.asyncio
@private_profile_test
async def test_apply_scan_drops_a_stale_generation(request, config_writes):
    async def body(picker):
        current = picker._scan_generation
        picker._apply_scan(current - 1, ("ok", {"other"}, {}))
        assert _yours(picker) == {"mine"}
        picker._apply_scan(current, ("ok", {"other"}, {}))
        assert _yours(picker) == {"other"}

    await _mounted_picker(body)


@pytest.mark.asyncio
@private_profile_test
async def test_apply_scan_outcomes_set_availability_and_listing(request, config_writes):
    async def body(picker):
        gen = picker._scan_generation
        picker._apply_scan(gen, ("paused", set(), {}))
        assert picker.files_available is False and _yours(picker) == {"mine"}
        picker._apply_scan(gen, ("failed", set(), {}))
        assert _yours(picker) == {"mine"}
        picker._apply_scan(gen, ("oserror", set(), {}))  # R27 (6)
        assert picker.files_available is True and _yours(picker) == set()
        picker._apply_scan(gen, ("ok", {"other"}, {}))
        assert picker.files_available is True and _yours(picker) == {"other"}

    await _mounted_picker(body)


# -- Critique #3 P3 wave (lane A) ---------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_unreadable_row_shows_its_error_in_place_of_the_preview(request, config_writes):
    """TASK-33067: an unreadable row painted every preview row #808080 on
    #808080 (1:1); the error line stands in for the preview instead."""
    picker = ThemePicker(id="settings-theme-picker", list_themes=lambda: (set(), {"a": "bad colour"}))
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.highlighted = lst.get_option_index("unreadable:a")
        await pilot.pause()
        assert picker.query_one("#settings-theme-card-error").display
        assert not picker.query_one(ThemePreview).display
        lst.highlighted = lst.get_option_index("nord")
        await pilot.pause()
        assert picker.query_one(ThemePreview).display
        assert not picker.query_one("#settings-theme-card-error").display


@pytest.mark.asyncio
@private_profile_test
async def test_no_match_filter_offers_clear_and_restores_the_previous_highlight(request, config_writes):
    """TASK-33069: spec §9 Clear filter chip, no stale preview, and clearing
    lands back on the theme highlighted before the filter -- not row 1."""
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.highlighted = lst.get_option_index("nord")
        await pilot.pause()
        clear = picker.query_one("#settings-theme-clear-filter", Button)
        assert not clear.display
        picker.query_one("#settings-theme-filter").focus()
        await pilot.press(*"zzzqqq")
        await pilot.pause()
        assert clear.display
        assert not picker.query_one(ThemePreview).display
        assert str(picker.query_one("#settings-theme-card-title").render()) == ""
        clear.press()
        await pilot.pause()
        assert picker.query_one("#settings-theme-filter").value == ""
        assert picker.highlighted_id == "nord"
        assert lst.highlighted == lst.get_option_index("nord")
        assert picker.query_one(ThemePreview).display
        assert not clear.display
        assert app.focused is lst


@pytest.mark.asyncio
@private_profile_test
async def test_clearing_a_no_match_filter_by_hand_falls_back_to_the_active_theme(request, config_writes):
    """TASK-33069 AC2: with no earlier highlight left to return to, clearing
    lands on the active theme, not the first row (an unreadable file here)."""
    picker = ThemePicker(id="settings-theme-picker", list_themes=lambda: (set(), {"a": "bad"}))
    app = _app(picker)
    for theme in ALL_THEMES:
        app.register_theme(theme)
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        picker.query_one("#settings-theme-filter").focus()
        await pilot.press(*"zzzqqq")
        await pilot.pause()
        picker._prefilter_highlight = None  # nothing to return to
        picker.query_one("#settings-theme-filter").value = ""
        await pilot.pause()
        assert picker.highlighted_id == str(app.theme)


@pytest.mark.asyncio
@private_profile_test
async def test_j_and_k_move_the_list_highlight(request, config_writes):
    """TASK-33072 AC2."""
    app, picker = await _picker_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await _landed(app, pilot)
        lst = picker.query_one("#settings-theme-list", OptionList)
        lst.highlighted = lst.get_option_index("nord")
        lst.focus()
        await pilot.pause()
        start = lst.highlighted
        await pilot.press("j")
        await pilot.pause()
        assert lst.highlighted > start and picker.highlighted_id != "nord"
        await pilot.press("k")
        await pilot.pause()
        assert lst.highlighted == start and picker.highlighted_id == "nord"


@pytest.mark.asyncio
@private_profile_test
async def test_row_truncates_only_the_name_to_fit(request):
    """TASK-33074: the name gives way (ellipsis); strip and markers stay."""
    from tldw_chatbook.Widgets.settings_theme_picker import _row

    entry = tc.ThemeEntry(
        id="x",
        display_name="A Very Long Theme Name That Keeps Going",
        origin="yours",
        dark=True,
        colours=tuple((key, "#112233") for key in tc.STRIP_KEYS),
        is_active=True,
        is_launch_default=False,
    )
    full = _row(entry).plain
    assert full.startswith("A Very Long Theme Name That Keeps Going")
    fitted = _row(entry, 30)
    assert fitted.cell_len <= 30
    assert "…" in fitted.plain and "▮" * len(tc.STRIP_KEYS) in fitted.plain and fitted.plain.endswith("active")


@pytest.mark.parametrize("width", [32, 26])
def test_row_tail_gives_way_before_the_name(width):
    """P3 review I2: active + launch + overrides used to leave the name as a
    bare ellipsis and still overflow the row. The tail shrinks first; the
    name keeps a readable prefix and "active" never drops."""
    from tldw_chatbook.Widgets.settings_theme_picker import _row

    entry = tc.ThemeEntry(
        id="textual-dark",
        display_name="Textual Dark",
        origin="yours",
        dark=True,
        colours=tuple((key, "#112233") for key in tc.STRIP_KEYS),
        is_active=True,
        is_launch_default=True,
        overrides="textual",
    )
    assert _row(entry).plain.endswith("active · launch · overrides built-in")
    fitted = _row(entry, width)
    assert fitted.cell_len <= width, fitted.plain
    assert fitted.plain.startswith("Textual D"), fitted.plain
    assert "active" in fitted.plain
