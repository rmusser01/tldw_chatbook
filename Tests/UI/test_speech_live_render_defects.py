"""Three defects that only a live run exposed.

157 unit tests passed while the Speech screen rendered six axis controls
showing no values and two progress bars pulsing at an idle user. The tests
asserted the things a test naturally reaches for -- the control exists, it is
focusable, its label is not truncated -- none of which is what was wrong.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ProgressBar

from tldw_chatbook.UI.Speech.speech_playground_model import AXIS_CONTROLS
from Tests.UI.test_speech_playground_pane import _PaneScreen, _build_layout_test_app
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane

BOX_DRAWING = set("▔▁▊▎│─┌┐└┘━╸╺")


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield SpeechPlaygroundPane(
            provider="audio_cpp",
            axis_values={"tts-voice-select": "Nova", "tts-format-select": "wav"},
        )


@pytest.mark.asyncio
async def test_the_axis_select_has_no_inner_border_under_the_real_css(
    monkeypatch: pytest.MonkeyPatch,
):
    """The defect: `border: none` on a Select does not reach its inner
    SelectCurrent, which keeps its own three-row bordered box. Clipped to
    `height: 1`, every axis painted its TOP BORDER and nothing else -- six
    controls showing neither value nor arrow.

    Two things this test must do that the ones which missed it did not.

    It runs under the REAL app CSS. The fix lives in the app-tier bundle, so
    a bare `App` harness cannot see it either way -- a bundle rule is
    invisible to a test that never loads the bundle.

    And it asserts the inner widget, because that is where the border is.
    A widget's own `render_line` is in its own coordinate space and never
    shows a border, so every self-oriented check called the broken render
    clean -- the same oracle failure that let "Export" ship as "Exp".
    """
    from textual.widgets._select import SelectCurrent

    app = _build_layout_test_app(monkeypatch)
    async with app.run_test(size=(200, 60)) as pilot:
        screen = _PaneScreen()
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        bordered = []
        for axis in AXIS_CONTROLS:
            for current in screen.query(f"#{axis}").results():
                for inner in current.query(SelectCurrent):
                    edges = inner.styles.border
                    if any(edge and edge[0] not in ("", "none") for edge in edges):
                        bordered.append(axis)

    assert not bordered, (
        f"inner SelectCurrent still bordered, so the value cannot show: "
        f"{bordered}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bar_id", ["#audio-progress-bar", "#generation-progress"]
)
async def test_progress_bars_declare_a_total(bar_id):
    """A ProgressBar with no `total` renders its indeterminate pulse, so an
    idle screen animates forever -- motion that says work is happening when
    none is."""
    app = _Harness()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        bar = app.query_one(bar_id, ProgressBar)
        assert bar.total is not None, f"{bar_id} will pulse indefinitely"


@pytest.mark.asyncio
async def test_nothing_reports_progress_before_anything_starts():
    """Both the player bar and the generation status start hidden, as the
    legacy screen had them. The rebuild dropped the `hidden` class and put a
    placeholder ETA (`--% --:--:--`) on an idle screen."""
    app = _Harness()
    async with app.run_test(size=(200, 60)) as pilot:
        await pilot.pause()
        await pilot.pause()
        for widget_id in ("#audio-progress-bar", "#generation-status-container"):
            widget = app.query_one(widget_id)
            assert not widget.display or widget.has_class("hidden"), (
                f"{widget_id} is reporting progress before anything started"
            )


class _StudioPrefsHarness(App[None]):
    """The Studio TTS Preferences pane under the real app bundle."""

    CSS_PATH = "../../tldw_chatbook/css/tldw_cli_modular.tcss"

    def compose(self) -> ComposeResult:
        """Mount the real Studio preferences pane.

        Returns:
            A ``ComposeResult`` yielding one ``SpeechSettingsPane``.
        """
        from tldw_chatbook.UI.Speech.speech_settings_pane import (
            SpeechSettingsPane,
        )

        yield SpeechSettingsPane()


@pytest.mark.asyncio
async def test_focused_exact_model_input_shows_its_text_under_the_real_css():
    """Typed text must stay visible while the exact-ID input is focused.

    The defect (TASK-15421 AC3): the bundle's global accessibility rule
    ``*:focus {{ outline: solid $ds-focus-accent }}`` paints the outline
    OVER the widget's outermost rendered lines (its own comment warns of
    exactly this), and a height-1 input's outermost line IS its only
    content line — so focusing the field replaced the typed value with
    box-drawing characters. `styles.border` probes stayed clean throughout
    (the border rules were a red herring); only a rendered frame shows it,
    which is why this lives in the only-a-live-run-exposed file.
    """
    from textual.widgets import Select

    app = _StudioPrefsHarness()
    async with app.run_test(size=(235, 52)) as pilot:
        for _ in range(30):
            await pilot.pause(0.02)
        app.query_one("#studio-tts-model-mode", Select).value = "exact"
        for _ in range(10):
            await pilot.pause(0.02)
        field = app.query_one("#studio-tts-model-id")
        field.focus()
        field.insert_text_at_cursor("studio-model")
        for _ in range(10):
            await pilot.pause(0.02)
        assert app.focused is field, "the exact-ID input never took focus"
        frame = app.export_screenshot()

    assert "studio-model" in frame, (
        "the focused exact-ID input's typed text is painted over"
    )
