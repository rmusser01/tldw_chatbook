"""TASK-33241 / TASK-33242: Theme view switches and Settings dialogs skip
work they do not need.

Counts, not wall-clock: the timings live in the tasks' notes.
"""

import re
from pathlib import Path

import pytest
from textual.css.stylesheet import Stylesheet

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_overview_search_journeys import _category
from Tests.UI.test_settings_theme_picker_screen import _host
from Tests.UI.theme_editor_helpers import FULL_SCREEN_SIZE
from tldw_chatbook.UI.Screens.settings_screen import RagProfileNameModal, SettingsScreen
from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor
from tldw_chatbook.Widgets.settings_theme_picker import ThemePane, ThemePicker

_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


async def _theme_pane(host, pilot) -> ThemePane:
    await _category(host, pilot, "Theme")
    await host.workers.wait_for_complete()
    await pilot.pause()
    return host.screen.query_one(ThemePane)


def test_picker_class_styles_only_the_pane():
    """ThemePane restyles only itself when it toggles `-picker`, so no rule
    may use the class as an ANCESTOR (`.-picker #x`): that rule would go
    stale on every switch."""
    css = re.sub(r"/\*.*?\*/", "", _BUNDLE.read_text(encoding="utf-8"), flags=re.DOTALL)
    offenders = []
    for block in re.findall(r"([^{}]+)\{", css):
        for selector in block.split(","):
            compounds = selector.split()
            if any(re.search(r"\.-picker\b", c) for c in compounds[:-1]):
                offenders.append(selector.strip())
    assert offenders == []
    assert "#settings-theme-pane.-picker {" in css  # the guard still has a subject


@pytest.mark.asyncio
@private_profile_test
async def test_editor_to_picker_switch_restyles_only_the_pane(request, monkeypatch):
    """TASK-33241: Back/Save restyled the pane's whole subtree (~180 nodes,
    the 136-node editor included) -- ~165 ms at 211x44. Only the pane and
    the focus change restyle now, and the picker still fills the pane."""
    host = _host()
    async with host.run_test(size=FULL_SCREEN_SIZE) as pilot:
        pane = await _theme_pane(host, pilot)
        lst = pane.query_one("#settings-theme-list")
        list_height = lst.size.height
        pane.open_editor("textual-dark", "clone")
        await pilot.pause()
        assert not pane.has_class("-picker")

        restyled = []
        real = Stylesheet.update_nodes

        def counting(self, nodes, animate=False):
            nodes = list(nodes)
            restyled.extend(nodes)
            return real(self, nodes, animate)

        monkeypatch.setattr(Stylesheet, "update_nodes", counting)
        pane.show_picker()
        await pilot.pause()
        await host.workers.wait_for_complete()
        await pilot.pause()

        editor = pane.query_one("#settings-theme-editor", SettingsThemeEditor)
        assert pane in restyled and pane.has_class("-picker")
        # Only the blurred name Input (its :focus) of the editor's ~136 nodes.
        assert len(set(editor.walk_children(with_self=True)) & set(restyled)) <= 1
        assert len(restyled) < 10, len(restyled)
        assert lst.size.height == list_height > 24  # still the 1fr fill


@pytest.mark.asyncio
@private_profile_test
async def test_save_returns_to_the_picker_with_one_rescan(request, monkeypatch):
    """TASK-33241: Save used to rescan the themes folder twice (show_picker's
    refresh, then a second one for the highlight)."""
    host = _host()
    async with host.run_test(size=FULL_SCREEN_SIZE) as pilot:
        pane = await _theme_pane(host, pilot)
        pane.open_editor("textual-dark", "clone")
        await pilot.pause()
        calls = []
        real = ThemePicker.refresh_catalog

        def counting(self, highlight=None, *, rescan=True):
            calls.append((highlight, rescan))
            return real(self, highlight, rescan=rescan)

        monkeypatch.setattr(ThemePicker, "refresh_catalog", counting)
        pane.post_message(SettingsThemeEditor.Saved("textual-light"))
        await pilot.pause()
        assert calls == [("textual-light", True)]
        assert pane.current == "settings-theme-picker"


@pytest.mark.asyncio
@private_profile_test
async def test_closing_a_theme_dialog_skips_the_sync_rows_refresh(request, monkeypatch):
    """TASK-33242: every dialog close resumed Settings and re-ran the
    sync-rows refresh (~550 ms of backup-scoped DB work, rows Theme does not
    show). Control: a dialog over another category still refreshes."""
    calls = []
    monkeypatch.setattr(
        SettingsScreen,
        "_queue_sync_rows_refresh",
        lambda self: calls.append(self) or True,
    )
    host = _host()
    async with host.run_test(size=FULL_SCREEN_SIZE) as pilot:
        await _theme_pane(host, pilot)
        calls.clear()

        async def dialog_round_trip():
            host.push_screen(
                RagProfileNameModal(title="Rename", initial="x", confirm_label="Rename")
            )
            await pilot.pause()
            assert isinstance(host.screen, RagProfileNameModal)
            host.pop_screen()
            await pilot.pause()
            assert isinstance(host.screen, SettingsScreen)

        await dialog_round_trip()
        assert calls == []

        await _category(host, pilot, "Overview")
        calls.clear()
        await dialog_round_trip()
        assert len(calls) == 1
