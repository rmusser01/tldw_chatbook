"""TASK-2775: the F9 Settings screen has an About category.

Version/license/links were unreachable since TASK-1346 retired the legacy
ToolsSettingsWindow. About is a read-only ("view") category under
Troubleshooting: it shows the installed version, renders the real-markdown
About text, and opens http(s) links in the system browser with a notify.
"""

import webbrowser
from types import SimpleNamespace

import pytest
from textual.widgets import Markdown

from Tests.UI.test_destination_shells import DestinationHarness, _build_test_app
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Utils.about_text import ABOUT_MARKDOWN, get_app_version


def test_about_category_registered_read_only():
    screen = SettingsScreen(_build_test_app())
    summaries = {s.category: s for s in screen._category_summaries()}
    assert SettingsCategoryId.ABOUT in summaries
    assert summaries[SettingsCategoryId.ABOUT].title == "About"

    groups = dict(screen._category_groups())
    assert SettingsCategoryId.ABOUT in groups["Troubleshooting"]

    record = screen._ownership_record(SettingsCategoryId.ABOUT)
    assert record.writes_allowed is False
    # Explicit record, not the missing-record fallback.
    assert "matrix" not in record.boundary_copy.lower()

    assert screen._inspector_guidance(SettingsCategoryId.ABOUT)


def test_about_text_and_version_are_presentable():
    assert "[bold]" not in ABOUT_MARKDOWN and "[link=" not in ABOUT_MARKDOWN
    version = get_app_version()
    assert isinstance(version, str) and version


@pytest.mark.asyncio
async def test_about_pane_renders_version_and_markdown():
    app = DestinationHarness(_build_test_app(), "settings")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen_stack[-1]
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        screen.active_category = SettingsCategoryId.ABOUT.value
        await pilot.pause()
        await pilot.pause()

        card = screen.query_one("#settings-about-card")
        assert card is not None
        md = screen.query_one("#settings-about-markdown", Markdown)
        assert md.source == ABOUT_MARKDOWN
        visible = " ".join(
            str(getattr(w, "renderable", "")) for w in card.query("Static")
        )
        assert get_app_version() in visible
        assert "AGPLv3+" in visible


@pytest.mark.asyncio
async def test_about_link_policy(monkeypatch):
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    app = DestinationHarness(_build_test_app(), "settings")
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen_stack[-1]
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        notices = []
        monkeypatch.setattr(
            screen, "notify", lambda msg, **kw: notices.append((msg, kw))
        )
        screen._handle_about_link(
            SimpleNamespace(href="https://github.com/rmusser01/tldw", stop=lambda: None)
        )
        assert opened == ["https://github.com/rmusser01/tldw"]
        assert any("Opened in browser" in n[0] for n in notices)

        screen._handle_about_link(
            SimpleNamespace(href="file:///etc/passwd", stop=lambda: None)
        )
        assert opened == ["https://github.com/rmusser01/tldw"]
        assert any("unsupported scheme" in n[0] for n in notices)
