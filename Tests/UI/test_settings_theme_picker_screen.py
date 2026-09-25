"""TASK-32948: Settings > Theme opens on the picker; Clone/New swap in the editor."""

import pytest
from textual.widgets import ContentSwitcher

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_overview_search_journeys import _category
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness
from tldw_chatbook.css.Themes.themes import ALL_THEMES


def _host():
    host = _StyledDestinationHarness(_build_test_app(), "settings")
    for theme in ALL_THEMES:
        host.register_theme(theme)
    return host


@pytest.mark.asyncio
@private_profile_test
async def test_theme_category_opens_on_the_picker_without_state_banner(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-picker"
        assert not host.screen.query("#settings-category-state-banner")
        # R6: only the picker has Clone/New (PR 2 removed the editor's).
        for button_id in ("clone", "new"):
            assert len(host.screen.query(f"#settings-theme-{button_id}")) == 0
            assert len(host.screen.query(f"#settings-theme-picker-{button_id}")) == 1


@pytest.mark.asyncio
@private_profile_test
async def test_clone_opens_editor_and_back_returns(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        host.screen.query_one("#settings-theme-list").focus()
        await pilot.press("c")
        await pilot.pause(0.2)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-editor-view"
        editor = host.screen.query_one("#settings-theme-editor")
        assert editor.current_theme_name.endswith("_copy")
        assert not host.screen.query("#settings-theme-set-default")
        editor.is_modified = False
        await pilot.click("#settings-theme-back")
        await pilot.pause(0.2)
        assert pane.current == "settings-theme-picker"


@pytest.mark.asyncio
@private_profile_test
async def test_back_with_unsaved_edits_asks_first(request):
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        host.screen.query_one("#settings-theme-list").focus()
        await pilot.press("c")
        await pilot.pause(0.2)
        host.screen.query_one("#settings-theme-editor").is_modified = True
        await pilot.click("#settings-theme-back")
        await pilot.pause(0.2)
        assert isinstance(host.screen, ThemeLeaveModal)
        await pilot.press("escape")  # Stay
        await pilot.pause(0.2)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-editor-view"


@pytest.mark.asyncio
@private_profile_test
async def test_appearance_shows_read_only_theme_row_and_opens_picker(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Appearance")
        assert not host.screen.query("#settings-appearance-theme")
        summary = str(host.screen.query_one("#settings-appearance-theme-summary").render())
        assert "launch default" in summary and "active" in summary
        host.theme = "nord"  # e.g. from the palette, while Appearance is open
        await pilot.pause(0.2)
        assert "active: Nord" in str(host.screen.query_one("#settings-appearance-theme-summary").render())
        await pilot.click("#settings-appearance-open-theme")
        await pilot.pause(0.3)
        assert host.screen.query_one("#settings-theme-pane").current == "settings-theme-picker"


@pytest.mark.asyncio
@private_profile_test
async def test_open_theme_highlights_the_launch_default(request):
    # Task 5 (TASK-32948 PR 2, spec §8): Open Theme highlights the launch
    # default, not whatever happens to be active right now -- those can
    # differ (e.g. a palette Try/Use elsewhere in the session).
    from tldw_chatbook.Widgets.settings_theme_picker import ThemePicker

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Appearance")
        host.theme = "nord"  # active diverges from the launch default (textual-dark)
        await pilot.pause(0.2)
        await pilot.click("#settings-appearance-open-theme")
        await pilot.pause(0.3)
        picker = host.screen.query_one(ThemePicker)
        assert picker.highlighted_id == "textual-dark"


def test_appearance_save_sections_carry_no_default_theme():
    # Spec §8, structural: Appearance never writes general.default_theme --
    # not even the value it read. The config writer sets keys one by one
    # (config._apply_literal_mutation_unlocked), so an absent key leaves the
    # file's launch default alone.
    from dataclasses import replace

    from tldw_chatbook.UI.Screens import settings_appearance_defaults as sad

    values = replace(sad.SettingsAppearanceDefaults(), default_theme="textual-light")
    sections = sad.build_appearance_save_sections(
        {"general": {"default_theme": "nord", "palette_theme_limit": 3}}, values
    )
    assert "default_theme" not in sections["general"]
    assert "palette_theme_limit" in sections["general"]


async def _save_appearance(host, pilot, saved):
    from textual.widgets import Input

    await _category(host, pilot, "Appearance")
    screen = host.screen
    limit = screen.query_one("#settings-appearance-palette-theme-limit", Input)
    limit.value = "7" if limit.value != "7" else "6"
    screen.handle_appearance_palette_theme_limit_changed(Input.Changed(limit, limit.value))
    await pilot.pause()
    before = len(saved)
    await pilot.click("#settings-save-category")
    for _ in range(40):
        await pilot.pause(0.05)
        if len(saved) > before and "Appearance defaults saved." in str(
            screen.query_one("#settings-appearance-save-result").render()
        ):
            return
    raise AssertionError("Appearance save did not complete")


@pytest.fixture
def appearance_writes(monkeypatch):
    from tldw_chatbook.UI.Screens import settings_screen as settings_screen_module

    saved = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    return saved


@pytest.mark.asyncio
@private_profile_test
async def test_appearance_save_round_trip_keeps_launch_default(request, appearance_writes):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        config = host.screen.app_instance.app_config
        config.setdefault("general", {})["default_theme"] = "dracula"
        await _save_appearance(host, pilot, appearance_writes)
        assert "default_theme" not in appearance_writes[-1]["general"]
        assert config["general"]["default_theme"] == "dracula"


@pytest.mark.asyncio
@private_profile_test
async def test_launch_default_changed_survives_a_later_appearance_save(request, appearance_writes):
    # Task 6 review: the editor's Delete -> _save_launch_default path posts
    # LaunchDefaultChanged; a later Appearance Save must not clobber it.
    from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        host.screen.post_message(SettingsThemeEditor.LaunchDefaultChanged("nord"))
        await pilot.pause()
        config = host.screen.app_instance.app_config
        assert config["general"]["default_theme"] == "nord"
        await _save_appearance(host, pilot, appearance_writes)
        assert "default_theme" not in appearance_writes[-1]["general"]
        assert config["general"]["default_theme"] == "nord"


@pytest.mark.asyncio
@private_profile_test
async def test_revert_chip_survives_a_category_round_trip(request, monkeypatch):
    # R9 / spec §5: the pending Revert lasts for the session.
    from types import SimpleNamespace

    from textual.widgets import Button

    from tldw_chatbook.css.Themes import theme_catalog as tc

    writes = []
    monkeypatch.setattr(
        tc, "_apply_config_mutation",
        lambda m: writes.append(m) or SimpleNamespace(file_replaced=True, caches_reloaded=True),
    )
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        original, launch = host.theme, tc.current_launch_default()
        host.screen.query_one("#settings-theme-list").focus()
        await pilot.press("down", "enter")  # Use
        await pilot.pause(0.2)
        assert host.theme != original
        await _category(host, pilot, "Appearance")
        await _category(host, pilot, "Theme")
        revert = host.screen.query_one("#settings-theme-revert", Button)
        assert revert.display
        revert.press()
        await pilot.pause(0.2)
        assert host.theme == original
        assert writes[-1] == {"general": {"default_theme": launch}}


async def _dirty_editor_then_back(host, pilot, choice):
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    await _category(host, pilot, "Theme")
    host.screen.query_one("#settings-theme-list").focus()
    await pilot.press("c")  # Clone opens clean (TASK-32948 PR 2) ...
    await pilot.pause(0.2)
    await _edit_primary(host, pilot)  # ... so make a real edit
    screen = host.screen
    # custom_themes_path is the private profile's themes dir (the editor's
    # writer refuses paths outside the profile, so no tmp_path override).
    editor = screen.query_one("#settings-theme-editor")
    assert editor.is_modified and screen.theme_editor_modified
    await pilot.click("#settings-theme-back")
    await pilot.pause(0.2)
    assert isinstance(host.screen, ThemeLeaveModal)
    await pilot.click(choice)
    await host.workers.wait_for_complete()
    await pilot.pause(0.2)
    return screen, editor


@pytest.mark.asyncio
@private_profile_test
async def test_back_discard_clears_both_flags_and_shows_picker(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        screen, editor = await _dirty_editor_then_back(host, pilot, "#settings-theme-leave-discard")
        assert editor.is_modified is False and screen.theme_editor_modified is False
        assert screen.query_one("#settings-theme-pane", ContentSwitcher).current == "settings-theme-picker"
        assert not (editor.custom_themes_path / f"{editor.current_theme_name}.toml").exists()


@pytest.mark.asyncio
@private_profile_test
async def test_back_save_writes_the_theme_then_shows_picker(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        screen, editor = await _dirty_editor_then_back(host, pilot, "#settings-theme-leave-save")
        assert (editor.custom_themes_path / f"{editor.current_theme_name}.toml").exists()
        assert editor.is_modified is False
        assert screen.query_one("#settings-theme-pane", ContentSwitcher).current == "settings-theme-picker"


@pytest.mark.asyncio
@private_profile_test
async def test_back_save_refused_stays_in_editor_with_edits(request):
    from textual.widgets import Input

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        host.screen.query_one("#settings-theme-list").focus()
        await pilot.press("c")
        await pilot.pause(0.2)
        await _edit_primary(host, pilot)  # Clone opens clean; make it dirty
        name = host.screen.query_one("#settings-theme-name", Input)
        name.value = "textual-dark"  # built-in: Save refuses
        await pilot.pause()
        await pilot.click("#settings-theme-back")
        await pilot.pause(0.2)
        await pilot.click("#settings-theme-leave-save")
        await host.workers.wait_for_complete()
        await pilot.pause(0.2)
        screen = host.screen
        assert screen.query_one("#settings-theme-pane", ContentSwitcher).current == "settings-theme-editor-view"
        assert screen.query_one("#settings-theme-editor").is_modified
        assert screen.query_one("#settings-theme-name", Input).value == "textual-dark"
        themes_dir = screen.query_one("#settings-theme-editor").custom_themes_path
        assert not (themes_dir / "textual-dark.toml").exists()


def test_appearance_validation_ignores_theme():
    from dataclasses import replace

    from tldw_chatbook.UI.Screens import settings_appearance_defaults as sad

    values = replace(sad.SettingsAppearanceDefaults(), default_theme="")
    assert sad.validate_appearance_defaults(values).valid


# TASK-32948 Task 7: layout CSS, geometry and contrast at 80x24 and 190x55.
# R6: the picker's Clone/New buttons are `#settings-theme-picker-clone` /
# `#settings-theme-picker-new` (PR 2 removed the editor's own Clone/New).
PICKER_CONTROLS = (
    "#settings-theme-filter",
    "#settings-theme-list",
    "#settings-theme-picker-preview",
    "#settings-theme-use",
    "#settings-theme-try",
    "#settings-theme-picker-clone",
    "#settings-theme-picker-new",
    "#settings-theme-picker-import",
)

# TASK-32948 PR 2 Task 6: the four yours-only file-action buttons, visible
# only once a user theme is highlighted (`_show` gates on origin == "yours").
YOURS_ONLY_CONTROLS = (
    "#settings-theme-picker-edit",
    "#settings-theme-picker-rename",
    "#settings-theme-picker-delete",
    "#settings-theme-picker-export",
)

# The editor view behind Clone/New/Edit: header + Save/Save as replaced the
# tree (PR 2 Task 4).
EDITOR_CONTROLS = (
    "#settings-theme-back",
    "#settings-theme-name",
    "#settings-theme-save",
    "#settings-theme-save-as",
    "#settings-theme-apply",
)


def _visible(host, widget):
    geometry = host.screen._compositor.find_widget(widget)
    return geometry.region.intersection(geometry.clip)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(80, 24), (190, 55)])
@private_profile_test
async def test_every_picker_control_is_reachable(request, theme, size):
    host = _host()
    _saved_theme(host)
    async with host.run_test(size=size) as pilot:
        host.theme = theme
        await _highlight(host, pilot, "mine")
        for selector in PICKER_CONTROLS + YOURS_ONLY_CONTROLS:
            widget = host.screen.query_one(selector)
            widget.scroll_visible(animate=False)
            await pilot.pause(0.1)
            region = _visible(host, widget)
            assert region.height > 0 and region.width > 0, f"{selector} unreachable at {size}"
        lst = host.screen.query_one("#settings-theme-list")
        # The yours-only row sits below the list; scrolling to reach it (the
        # last control checked above) can leave the list itself scrolled
        # off-screen, so re-scroll it into view before measuring its rows.
        lst.scroll_visible(animate=False)
        await pilot.pause(0.1)
        assert _visible(host, lst).height >= 5, f"list shows <5 rows at {size}"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (190, 55)])
@private_profile_test
async def test_every_editor_control_is_reachable(request, size):
    from Tests.UI.theme_editor_helpers import open_theme_editor

    host = _host()
    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, "Theme")
        await open_theme_editor(host, pilot)
        for selector in EDITOR_CONTROLS:
            widget = host.screen.query_one(selector)
            widget.scroll_visible(animate=False)
            await pilot.pause(0.1)
            region = _visible(host, widget)
            assert region.height > 0 and region.width > 0, f"{selector} unreachable at {size}"


@pytest.mark.asyncio
@private_profile_test
async def test_appearance_summary_recomposes_after_launch_default_changes_elsewhere(
    request,
):
    """Task 6 review ⚠️: after a launch-default change made off-screen (what
    the editor's Delete does when the deleted theme WAS the launch default --
    ``SettingsThemeEditor._delete_user_theme`` falls back to
    ``_save_launch_default("textual-dark", ...)``), Appearance must show the
    new value on its next visit.

    This changes the launch default the same way Delete's fallback does --
    a real ``apply_settings_mutation_to_cli_config`` write, not an
    ``app.theme`` switch -- so Appearance's ``theme_changed_signal``
    subscription (which only fires on ACTIVE theme changes) cannot be what
    catches it; only recomposing the detail pane on category re-entry can.
    """
    from tldw_chatbook.config import apply_settings_mutation_to_cli_config

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Appearance")
        before = str(host.screen.query_one("#settings-appearance-theme-summary").render())
        assert "Textual Dark" in before or "textual-dark" in before

        result = apply_settings_mutation_to_cli_config({"general": {"default_theme": "nord"}})
        assert result.file_replaced

        await _category(host, pilot, "Theme")
        await _category(host, pilot, "Appearance")
        after = str(host.screen.query_one("#settings-appearance-theme-summary").render())
        assert "Nord" in after
        assert after != before



@pytest.mark.asyncio
@private_profile_test
async def test_search_for_an_editor_field_lands_on_the_picker_list(request):
    # The editor sits hidden behind the picker (ContentSwitcher); a search
    # landing must not put focus -- and keystrokes -- into a hidden Input.
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Primary color")
        await pilot.pause(0.2)
        screen = host.screen
        assert screen.query_one("#settings-theme-pane", ContentSwitcher).current == "settings-theme-picker"
        assert host.focused is not None and host.focused.id == "settings-theme-list"
        await pilot.press("z", "9")
        await pilot.pause(0.2)
        assert screen.query_one("#settings-theme-editor").is_modified is False
        assert screen.theme_editor_modified is False


@pytest.mark.asyncio
@private_profile_test
async def test_theme_help_copy_describes_the_picker(request):
    from textual.widgets import Button

    from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Overview")
        screen = host.screen
        notes = " ".join(screen._category_help_notes(SettingsCategoryId.THEME))
        assert "Save contract: Applies immediately." in notes
        assert "Use/Try apply at once; Save in the editor stores a theme file." in notes
        assert "Read-only" not in notes and "Review readiness" not in notes
        summary = screen._category_summary_by_id(SettingsCategoryId.THEME)
        assert "Full theme editor" not in summary.description
        button = screen.query_one("#settings-open-appearance", Button)
        assert "editor" not in str(button.label).lower()
        assert "editor" not in str(button.tooltip).lower()


# TASK-32948 PR 2 Task 3: the picker's file actions route through the
# editor's backup-scoped API; Rename prompts via RagProfileNameModal.
def _saved_theme(host, name="mine"):
    """Write ``name`` into the private profile's themes dir and register it
    (the test harness does not run the app's startup theme loader)."""
    from textual.theme import Theme

    from tldw_chatbook import config

    themes = config._get_effective_config_path().parent / "themes"
    themes.mkdir(exist_ok=True)
    path = themes / f"{name}.toml"
    path.write_text(
        f'[theme]\nname = "{name}"\ndark = true\n[colors]\nprimary = "#0099FF"\n',
        encoding="utf-8",
    )
    host.register_theme(Theme(name=name, primary="#0099FF", dark=True))
    return path


async def _highlight(host, pilot, theme_id):
    await _category(host, pilot, "Theme")
    lst = host.screen.query_one("#settings-theme-list")
    lst.highlighted = lst.get_option_index(theme_id)
    lst.focus()
    await pilot.pause(0.1)


def _picker_ids(host):
    return {e.id for e in host.screen.query_one("#settings-theme-picker").entries}


@pytest.mark.asyncio
@private_profile_test
async def test_rename_from_picker_prompts_and_renames(request):
    from textual.widgets import Input

    from tldw_chatbook.UI.Screens.settings_screen import RagProfileNameModal

    host = _host()
    path = _saved_theme(host)
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "mine")
        await pilot.press("r")
        await pilot.pause(0.2)
        assert isinstance(host.screen, RagProfileNameModal)
        host.screen.query_one("#settings-rag-profile-name-input", Input).value = "ours"
        await pilot.click("#settings-rag-profile-name-confirm")
        await pilot.pause(0.3)
        picker = host.screen.query_one("#settings-theme-picker")
        assert picker.highlighted_id == "ours"
        assert "ours" in _picker_ids(host) and "mine" not in _picker_ids(host)
        assert not path.exists() and (path.parent / "ours.toml").exists()


@pytest.mark.asyncio
@private_profile_test
async def test_delete_from_picker_confirms_and_rebuilds(request):
    host = _host()
    path = _saved_theme(host)
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "mine")
        await pilot.press("delete")
        await pilot.pause(0.2)
        await pilot.click("#confirm-button")  # "Delete theme"
        await pilot.pause(0.3)
        assert not path.exists()
        assert "mine" not in _picker_ids(host)


@pytest.mark.asyncio
@private_profile_test
async def test_edit_opens_saved_theme_in_editor(request):
    from textual.widgets import Input

    host = _host()
    _saved_theme(host)
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "mine")
        await pilot.press("e")
        await pilot.pause(0.2)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-editor-view"
        assert host.screen.query_one("#settings-theme-name", Input).value == "mine"
        assert host.screen.query_one("#settings-theme-editor").is_modified is False


@pytest.mark.asyncio
@private_profile_test
async def test_picker_lists_via_editor_scope(request, monkeypatch):
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Widgets.settings_theme_editor import (
        THEMES_UNAVAILABLE_LABEL,
        SettingsThemeEditor,
    )

    def paused(self):
        raise RecoveryRequired("x")

    monkeypatch.setattr(SettingsThemeEditor, "list_user_theme_names", paused)
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        picker = host.screen.query_one("#settings-theme-picker")
        assert picker.files_available is False
        lst = host.screen.query_one("#settings-theme-list")
        prompts = [str(lst.get_option_at_index(i).prompt) for i in range(lst.option_count)]
        assert THEMES_UNAVAILABLE_LABEL in prompts


# TASK-32948 PR 2 Task 4: the editor loses its tree and library buttons,
# gains a header and Save as; Save returns to the picker.
async def _edit_primary(host, pilot, colour="#123456"):
    from textual.widgets import Input

    host.screen.query_one("#settings-theme-color-primary", Input).value = colour
    await pilot.pause(0.1)


@pytest.mark.asyncio
@private_profile_test
async def test_category_leave_save_does_not_crash_on_saved_message(request):
    from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    host = _host()
    path = _saved_theme(host)
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "mine")
        await pilot.press("e")
        await pilot.pause(0.2)
        await _edit_primary(host, pilot)
        screen = host.screen
        assert screen.theme_editor_modified is True
        screen._select_category(SettingsCategoryId.APPEARANCE.value)
        await pilot.pause(0.2)
        assert isinstance(host.screen, ThemeLeaveModal)
        await pilot.click("#settings-theme-leave-save")
        await host.workers.wait_for_complete()
        await pilot.pause(0.3)
        assert screen.active_category == SettingsCategoryId.APPEARANCE.value
        assert "#123456" in path.read_text(encoding="utf-8")
    # run_test re-raises any handler exception on exit; reaching here means
    # the Saved message on the torn-down pane was swallowed.


@pytest.mark.asyncio
@private_profile_test
async def test_save_active_theme_reapplies_and_returns(request):
    from textual.color import Color

    host = _host()
    _saved_theme(host)
    async with host.run_test(size=(190, 55)) as pilot:
        host.theme = "mine"
        await _highlight(host, pilot, "mine")
        await pilot.press("e")
        await pilot.pause(0.2)
        await _edit_primary(host, pilot, "#AB1234")
        await pilot.click("#settings-theme-save")
        await pilot.pause(0.3)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-picker"
        assert host.screen.query_one("#settings-theme-picker").highlighted_id == "mine"
        assert host.theme == "mine"
        active = host.available_themes[host.theme]
        assert Color.parse(active.primary).hex.upper() == "#AB1234"
        # What the app actually paints: the stylesheet variables are only
        # refreshed by a re-apply (registration alone leaves them stale).
        # (Compared with Textual's own generation: it rounds #AB1234 to #AA1234.)
        painted = Color.parse(host.stylesheet._variables["primary"])
        expected = Color.parse(active.to_color_system().generate()["primary"])
        assert painted == expected
        assert painted != Color.parse("#0099FF")  # not the pre-save palette


@pytest.mark.asyncio
@private_profile_test
async def test_save_as_keeps_original(request):
    from textual.widgets import Input

    from tldw_chatbook.UI.Screens.settings_screen import RagProfileNameModal

    host = _host()
    path = _saved_theme(host)
    original = path.read_bytes()
    async with host.run_test(size=(190, 55)) as pilot:
        before = host.theme
        await _highlight(host, pilot, "mine")
        await pilot.press("e")
        await pilot.pause(0.2)
        await _edit_primary(host, pilot)
        await pilot.click("#settings-theme-save-as")
        await pilot.pause(0.2)
        assert isinstance(host.screen, RagProfileNameModal)
        name_input = host.screen.query_one("#settings-rag-profile-name-input", Input)
        assert name_input.value == "mine_copy"
        name_input.value = "mine2"
        await pilot.click("#settings-rag-profile-name-confirm")
        await pilot.pause(0.3)
        assert (path.parent / "mine2.toml").exists()
        assert "#123456" in (path.parent / "mine2.toml").read_text(encoding="utf-8")
        assert path.read_bytes() == original
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-picker"
        assert host.screen.query_one("#settings-theme-picker").highlighted_id == "mine2"
        assert host.theme == before  # saving a non-active theme applies nothing


@pytest.mark.asyncio
@private_profile_test
async def test_editor_has_no_tree_or_library_buttons(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        editor = host.screen.query_one("#settings-theme-editor")
        for selector in (
            "#settings-theme-tree",
            "#settings-theme-new",
            "#settings-theme-clone",
            "#settings-theme-delete",
            "#settings-theme-export",
        ):
            assert not editor.query(selector), selector
        for selector in ("#settings-theme-apply", "#settings-theme-save", "#settings-theme-save-as"):
            assert editor.query_one(selector)
        assert str(editor.query_one("#settings-theme-apply").label) == "Try"


@pytest.mark.asyncio
@private_profile_test
async def test_editor_header_names_the_source(request):
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "apricot")
        await pilot.press("c")
        await pilot.pause(0.2)
        header = host.screen.query_one("#settings-theme-editor-header")
        assert str(header.render()) == "Editing apricot_copy · copy of Apricot"


@pytest.mark.asyncio
@private_profile_test
async def test_clone_then_back_without_edits_does_not_prompt(request):
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "apricot")
        await pilot.press("c")
        await pilot.pause(0.2)
        assert host.screen.query_one("#settings-theme-editor").is_modified is False
        await pilot.click("#settings-theme-back")
        await pilot.pause(0.2)
        assert not isinstance(host.screen, ThemeLeaveModal)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        assert pane.current == "settings-theme-picker"


@pytest.mark.asyncio
@private_profile_test
async def test_save_buttons_disabled_while_theme_files_are_paused(request, monkeypatch):
    """R20 / spec §9: during a backup/recovery pause Save and Save as are
    disabled, with the reason as their tooltip."""
    from textual.widgets import Button

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.Widgets.settings_theme_editor import (
        THEMES_UNAVAILABLE_LABEL,
        SettingsThemeEditor,
    )

    def paused(self):
        raise RecoveryRequired("x")

    monkeypatch.setattr(SettingsThemeEditor, "list_user_theme_names", paused)
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "apricot")
        await pilot.press("c")
        await pilot.pause(0.2)
        for button_id in ("#settings-theme-save", "#settings-theme-save-as"):
            button = host.screen.query_one(button_id, Button)
            assert button.disabled, button_id
            assert button.tooltip == THEMES_UNAVAILABLE_LABEL, button_id


# Final-review fixes (TASK-32948).
def _saved_theme_file(host, stem, name, primary="#AB0001"):
    """A saved theme whose ``[theme].name`` differs from its file stem (R12)."""
    from textual.theme import Theme

    from tldw_chatbook import config

    themes = config._get_effective_config_path().parent / "themes"
    themes.mkdir(exist_ok=True)
    path = themes / f"{stem}.toml"
    path.write_text(
        f'[theme]\nname = "{name}"\ndark = true\n[colors]\nprimary = "{primary}"\n',
        encoding="utf-8",
    )
    host.register_theme(Theme(name=name, primary=primary, dark=True))
    return path


def _stateful_launch_default(monkeypatch, launch):
    """Fake config writes that the launch-default reads then see."""
    from types import SimpleNamespace

    from tldw_chatbook.css.Themes import theme_catalog as tc

    state = {"launch": launch}
    writes = []

    def fake_apply(mutation):
        writes.append(mutation)
        state["launch"] = mutation["general"]["default_theme"]
        return SimpleNamespace(file_replaced=True, caches_reloaded=True)

    monkeypatch.setattr(tc, "_apply_config_mutation", fake_apply)
    monkeypatch.setattr(tc, "current_launch_default", lambda: state["launch"])
    monkeypatch.setattr(
        "tldw_chatbook.Widgets.settings_theme_picker.current_launch_default",
        lambda: state["launch"],
    )
    return writes


async def _rename_highlighted(host, pilot, new):
    from textual.widgets import Input

    await pilot.press("r")
    await pilot.pause(0.2)
    host.screen.query_one("#settings-rag-profile-name-input", Input).value = new
    await pilot.click("#settings-rag-profile-name-confirm")
    await pilot.pause(0.3)


@pytest.mark.asyncio
@private_profile_test
async def test_edit_and_save_a_theme_whose_name_differs_from_its_file(request):
    """R12: Edit of a.toml (name "b") loads b; Save rewrites a.toml."""
    from textual.widgets import Input

    host = _host()
    path = _saved_theme_file(host, "a", "b")
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "b")
        await pilot.press("e")
        await pilot.pause(0.2)
        screen = host.screen
        assert screen.query_one("#settings-theme-name", Input).value == "b"
        assert screen.query_one("#settings-theme-color-primary", Input).value == "#AB0001"
        header = screen.query_one("#settings-theme-editor-header")
        assert str(header.render()) == "Editing b · saved theme"
        await _edit_primary(host, pilot)
        await pilot.click("#settings-theme-save")
        await pilot.pause(0.3)
        assert "#123456" in path.read_text(encoding="utf-8")
        assert not (path.parent / "b.toml").exists()


@pytest.mark.asyncio
@private_profile_test
async def test_clone_of_a_theme_whose_name_differs_from_its_file(request):
    from textual.widgets import Input

    host = _host()
    _saved_theme_file(host, "a", "b")
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "b")
        await pilot.press("c")
        await pilot.pause(0.2)
        screen = host.screen
        assert screen.query_one("#settings-theme-name", Input).value == "b_copy"
        assert screen.query_one("#settings-theme-color-primary", Input).value == "#AB0001"


@pytest.mark.asyncio
@private_profile_test
async def test_rename_of_non_active_launch_default_keeps_running_theme(request, monkeypatch):
    """Spec §7 step 4: renaming the launch default only rewrites config."""
    writes = _stateful_launch_default(monkeypatch, "mine")
    host = _host()
    _saved_theme(host)
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "nord")
        await pilot.press("t")  # Try nord
        await pilot.pause(0.2)
        assert host.theme == "nord"
        lst = host.screen.query_one("#settings-theme-list")
        lst.highlighted = lst.get_option_index("mine")
        await pilot.pause(0.1)
        await _rename_highlighted(host, pilot, "ours")
        assert host.theme == "nord"
        assert writes == [{"general": {"default_theme": "ours"}}]


@pytest.mark.asyncio
@private_profile_test
async def test_revert_follows_a_renamed_theme(request, monkeypatch):
    from textual.widgets import Button

    writes = _stateful_launch_default(monkeypatch, "mine")
    host = _host()
    _saved_theme(host)
    async with host.run_test(size=(190, 55)) as pilot:
        host.theme = "mine"
        await _highlight(host, pilot, "nord")
        await pilot.press("enter")  # Use nord
        await pilot.pause(0.2)
        lst = host.screen.query_one("#settings-theme-list")
        lst.highlighted = lst.get_option_index("mine")
        await pilot.pause(0.1)
        await _rename_highlighted(host, pilot, "mine2")
        revert = host.screen.query_one("#settings-theme-revert", Button)
        assert revert.display
        revert.press()
        await pilot.pause(0.2)
        assert host.theme == "mine2"
        assert writes[-1] == {"general": {"default_theme": "mine2"}}


@pytest.mark.asyncio
@private_profile_test
async def test_delete_drops_a_revert_that_names_the_deleted_theme(request, monkeypatch):
    from textual.widgets import Button

    _stateful_launch_default(monkeypatch, "mine")
    host = _host()
    _saved_theme(host)
    async with host.run_test(size=(190, 55)) as pilot:
        host.theme = "mine"
        await _highlight(host, pilot, "nord")
        await pilot.press("enter")  # Use nord
        await pilot.pause(0.2)
        lst = host.screen.query_one("#settings-theme-list")
        lst.highlighted = lst.get_option_index("mine")
        await pilot.pause(0.1)
        await pilot.press("delete")
        await pilot.pause(0.2)
        await pilot.click("#confirm-button")
        await pilot.pause(0.3)
        assert getattr(host, "theme_revert_change", None) is None
        assert not host.screen.query_one("#settings-theme-revert", Button).display


@pytest.mark.asyncio
@private_profile_test
async def test_escape_with_nothing_focused_leaves_the_editor(request):
    """R26: the first Esc releases field focus (task-1560), the second is Back."""
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "apricot")
        await pilot.press("c")
        await pilot.pause(0.2)
        pane = host.screen.query_one("#settings-theme-pane", ContentSwitcher)
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert host.focused is None
        assert pane.current == "settings-theme-editor-view"
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert pane.current == "settings-theme-picker"


@pytest.mark.asyncio
@private_profile_test
async def test_escape_with_unsaved_edits_asks_first(request):
    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "apricot")
        await pilot.press("c")
        await pilot.pause(0.2)
        await _edit_primary(host, pilot)
        await pilot.press("escape", "escape")
        await pilot.pause(0.2)
        assert isinstance(host.screen, ThemeLeaveModal)


@pytest.mark.asyncio
@private_profile_test
async def test_rename_prompt_escapes_markup_in_theme_name(request):
    """R27 (7): a hand-edited [theme].name like ``x[/]`` must not raise
    MarkupError in the Rename prompt's title."""
    from textual.widgets import Static

    from tldw_chatbook.UI.Screens.settings_screen import RagProfileNameModal

    host = _host()
    _saved_theme_file(host, "odd", "x[/]")
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "x[/]")
        await pilot.press("r")
        await pilot.pause(0.2)
        assert isinstance(host.screen, RagProfileNameModal)
        title = host.screen.query_one(".destination-section", Static)
        assert "x[/]" in str(title.render())


@pytest.mark.asyncio
@private_profile_test
async def test_delete_of_an_unreadable_file_removes_it(request):
    """PR 3 Task 1: an unreadable file is listed under Yours and deletable."""
    from tldw_chatbook import config

    themes = config._get_effective_config_path().parent / "themes"
    themes.mkdir(exist_ok=True)
    path = themes / "broken.toml"
    path.write_text("garbage [[ not toml", encoding="utf-8")
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "broken")
        picker = host.screen.query_one("#settings-theme-picker")
        entry = next(e for e in picker.entries if e.id == "broken")
        assert entry.origin == "yours" and entry.error == "not valid TOML"
        await pilot.press("delete")
        await pilot.pause(0.2)
        await pilot.click("#confirm-button")  # "Delete theme"
        await pilot.pause(0.3)
        assert not path.exists()
        assert "broken" not in _picker_ids(host)


@pytest.mark.asyncio
@private_profile_test
async def test_import_from_picker_prompts_imports_and_highlights(request, tmp_path_factory):
    """TASK-32948 PR 3 Task 2: `i` -> path prompt -> file lands in the
    profile's themes dir, registered and highlighted."""
    from textual.widgets import Input

    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens.settings_screen import RagProfileNameModal

    source = tmp_path_factory.mktemp("import-src") / "drop.toml"
    source.write_text(
        '[theme]\nname = "dropped"\ndark = true\n[colors]\nprimary = "#22AA88"\n',
        encoding="utf-8",
    )
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Theme")
        picker = host.screen.query_one("#settings-theme-picker")
        highlights: list[str | None] = []
        real_refresh = picker.refresh_catalog

        def spy(highlight=None):
            highlights.append(highlight)
            real_refresh(highlight=highlight)

        picker.refresh_catalog = spy
        host.screen.query_one("#settings-theme-list").focus()
        await pilot.press("i")
        await pilot.pause(0.2)
        assert isinstance(host.screen, RagProfileNameModal)
        host.screen.query_one("#settings-rag-profile-name-input", Input).value = f"'{source}'"
        await pilot.click("#settings-rag-profile-name-confirm")
        await pilot.pause(0.3)
        themes = config._get_effective_config_path().parent / "themes"
        assert (themes / "dropped.toml").exists()
        assert "dropped" in host.available_themes
        picker = host.screen.query_one("#settings-theme-picker")
        assert picker.highlighted_id == "dropped"
        assert "dropped" in _picker_ids(host)
        # R37: one refresh (ThemesChanged(highlight=)), not a second from the screen.
        assert highlights.count("dropped") == 1


# TASK-32948 PR 3 Task 3: launch-default-missing notice, and Export's Copy path.
@pytest.mark.asyncio
@private_profile_test
async def test_launch_default_missing_notice_pins_picker_and_appearance(request, monkeypatch):
    """Spec §9: a launch default that no theme is registered under shows in
    both the picker (above the list) and Appearance's read-only summary;
    a Use of a real theme clears the picker's notice."""
    writes = _stateful_launch_default(monkeypatch, "deleted_one")
    host = _host()
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Appearance")
        summary = str(host.screen.query_one("#settings-appearance-theme-summary").render())
        assert "launch default missing: deleted_one" in summary

        await _category(host, pilot, "Theme")
        notice = host.screen.query_one("#settings-theme-launch-missing")
        assert notice.display
        assert "Launch default missing: deleted_one" in str(notice.render())

        lst = host.screen.query_one("#settings-theme-list")
        lst.highlighted = lst.get_option_index("nord")
        lst.focus()
        await pilot.pause(0.1)
        await pilot.press("enter")  # Use nord
        await pilot.pause(0.2)
        assert writes[-1] == {"general": {"default_theme": "nord"}}
        assert not host.screen.query_one("#settings-theme-launch-missing").display


@pytest.mark.asyncio
@private_profile_test
async def test_export_shows_full_path_and_copy_path_copies_it(request, monkeypatch, tmp_path_factory):
    """Spec §7: after Export, the picker card shows the full export path
    (the §7 exception to R16's truncated paths) with a Copy path button;
    a fresh highlight clears the row."""
    from pathlib import Path

    downloads_home = tmp_path_factory.mktemp("export-home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: downloads_home))
    host = _host()
    _saved_theme(host)
    async with host.run_test(size=(190, 55)) as pilot:
        await _highlight(host, pilot, "mine")
        assert not host.screen.query_one("#settings-theme-export-result").display
        await pilot.click("#settings-theme-picker-export")
        await pilot.pause(0.3)
        export_path = downloads_home / "Downloads" / "mine_theme.toml"
        assert export_path.exists()
        result = host.screen.query_one("#settings-theme-export-path")
        assert str(export_path) in str(result.render())
        assert host.screen.query_one("#settings-theme-export-result").display

        copied = []
        monkeypatch.setattr(host, "copy_to_clipboard", lambda text: copied.append(text))
        await pilot.click("#settings-theme-copy-path")
        await pilot.pause(0.1)
        assert copied == [str(export_path)]

        lst = host.screen.query_one("#settings-theme-list")
        lst.highlighted = lst.get_option_index("nord")
        await pilot.pause(0.1)
        assert not host.screen.query_one("#settings-theme-export-result").display
