"""``FittedText``: literal text refitted to its own width (Roleplay frame B1).

The shared widget behind the Roleplay header's item label and chips. Small
host apps only (no destination screen), so this file runs in the UI fast lane.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal

import tldw_chatbook.app  # noqa: F401  -- collection-time import (lessons-testing-evidence: Tests/UI RecoveryRequired at setup)
from tldw_chatbook.UI.Workbench.workbench_widgets import FittedText

pytestmark = pytest.mark.asyncio


def _head(value: str, width: int) -> str:
    return value[:width]


class _Host(App):
    CSS = """
    #row { height: 1; }
    #label { width: 1fr; height: 1; }
    #chip { width: auto; height: 1; }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="row"):
            yield FittedText("[b]x[/] " + "y" * 60, _head, id="label")
            yield FittedText("", hide_when_empty=True, id="chip")


def _painted_row(app: App, y: int) -> str:
    return list(app.screen._compositor.render_strips())[y].text


async def test_text_is_literal_and_carries_no_action():
    """Markup-shaped text paints as typed (spec R33): no MarkupError, no meta."""
    app = _Host()
    async with app.run_test(size=(40, 3)) as pilot:
        await pilot.pause()
        row = _painted_row(app, 0)
        assert row.startswith("[b]x[/] yyy")
        for strip in app.screen._compositor.render_strips():
            for segment in strip:
                meta = segment.style.meta if segment.style is not None else {}
                assert "@click" not in meta


async def test_the_fit_follows_the_widget_width_on_every_resize():
    app = _Host()
    async with app.run_test(size=(40, 3)) as pilot:
        await pilot.pause()
        label = app.query_one("#label", FittedText)
        assert label.fitted_text == label.value[:40]
        await pilot.resize_terminal(25, 3)
        await pilot.pause()
        assert label.content_size.width == 25
        assert _painted_row(app, 0) == label.value[:25]


async def test_set_value_repaints_only_on_a_change(monkeypatch):
    app = _Host()
    async with app.run_test(size=(40, 3)) as pilot:
        await pilot.pause()
        label = app.query_one("#label", FittedText)
        calls = []
        original = label.refresh

        def counting_refresh(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(label, "refresh", counting_refresh)
        label.set_value(label.value)
        assert calls == []
        label.set_value("changed")
        assert calls == [{"layout": True}]


async def test_an_empty_chip_takes_no_space_until_it_has_text():
    app = _Host()
    async with app.run_test(size=(40, 3)) as pilot:
        await pilot.pause()
        chip = app.query_one("#chip", FittedText)
        assert chip.display is False
        chip.set_value("Unsaved")
        await pilot.pause()
        assert chip.display is True
        assert chip.region.width == len("Unsaved")
        chip.set_value("")
        await pilot.pause()
        assert chip.display is False


def test_the_widget_declares_no_css():
    """Rules belong to the owning destination's sheet (zero boot bytes)."""
    for name in ("DEFAULT_CSS", "CSS", "BUNDLED_CSS"):
        assert name not in FittedText.__dict__
