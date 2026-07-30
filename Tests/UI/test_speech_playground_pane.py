"""The assembled Playground: fold position, truncation, and containment."""

from __future__ import annotations

import pytest
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Static

from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane


class _PaneScreen(Screen):
    """Hosts the pane on its own, under the real app CSS.

    The pane is not the mounted `playground` view yet: dev's
    `TTSPlaygroundWidget` keeps that while its profile presets and this
    pane's axis row are reconciled. So these mount the pane directly rather
    than navigating to Speech and asserting on whatever is there -- which
    would be testing the routing, not the pane.
    """

    def compose(self):
        body = Vertical(
            SpeechPlaygroundPane(id="speech-playground-pane"), id="lab-body"
        )
        # Inline, because the app-tier bundle outranks a test Screen's
        # DEFAULT_CSS. The Lab frame constrains its body to the viewport;
        # without that the pane is sized by its own content -- measured 236
        # cells wide in a 60-column terminal -- so nothing ever looks narrow
        # and the stacking rule never fires.
        body.styles.width = "100%"
        body.styles.height = "100%"
        yield body

    def on_mount(self) -> None:
        """Pin the pane to the viewport width once it exists."""
        pane = self.query_one("#speech-playground-pane")
        pane.styles.width = "100%"


async def _speech_screen(app):
    screen = _PaneScreen()
    await app.push_screen(screen)
    return screen


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_the_primary_action_is_above_the_fold(size):
    """The defect this phase exists to fix.

    `Generate Speech` rendered at y=60 in a 34-row viewport -- 21 rows below
    the fold, reachable only by scrolling ~2.5 screens.
    """
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()
        body = screen.query_one("#lab-body")
        # The bare id, not `workbench-action-tts-generate-btn`. This test
        # originally asserted the prefixed one, which is what `CommandStrip`
        # mounts -- so it passed against a button whose id the handler
        # (`event.button.id == "tts-generate-btn"`) could never match. A
        # visible, above-the-fold, permanently dead button.
        generate = screen.query_one("#tts-generate-btn", Button)
        assert body.region.contains_region(generate.region), (
            f"Generate below the fold at {size}: y={generate.region.y}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_no_control_is_clipped_by_its_container(size):
    """Containment, not just self-rendering.

    `render_line(0).text` reads a widget in its OWN coordinate space, so it
    reports a full label for a control the parent is clipping. That is how
    "Export" shipped rendering as "Exp": it sat at x=101..111 inside a pane
    ending at 107 and every self-oriented check called it clean. Assert the
    region is inside its parent.
    """
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()

        escaped = []
        for strip_id in ("#speech-playground-actions", "#speech-result-actions"):
            strip = screen.query_one(strip_id)
            for button in strip.query(Button):
                if not button.region.width:
                    continue
                if not strip.region.contains_region(button.region):
                    escaped.append((strip_id, str(button.label)))
        assert not escaped, f"clipped by container at {size}: {escaped}"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_no_chip_text_is_truncated(size):
    """The axes are what the user is comparing; they may not be cut off."""
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()
        for chip in screen.query(".speech-chip").results(Static):
            text = str(chip.renderable)
            assert text in chip.render_line(0).text, f"truncated at {size}: {text!r}"


@pytest.mark.asyncio
async def test_the_pane_scrolls_rather_than_clipping_when_stacked():
    """`1fr` children compress instead of overflowing, which clips content
    that should scroll. The pane must be genuinely taller than its viewport."""
    app = _build_test_app()
    # 60 cells, not 80. Hosted directly the pane gets the whole terminal
    # width, where inside the Lab frame it got the body minus rail and
    # inspector -- so 80 columns used to leave it under its own 64-cell
    # threshold and now does not. The test is about what the pane does when
    # it IS too narrow, so it is measured below the threshold.
    async with app.run_test(size=(60, 24)) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()
        pane = screen.query_one("#speech-playground-pane")
        assert pane.has_class("speech-split-stacked")
        assert pane.virtual_size.height > pane.container_size.height


@pytest.mark.asyncio
async def test_the_axes_and_the_text_input_are_both_present():
    """The comparison loop needs both: what you are varying, and what you
    are synthesizing."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _speech_screen(app)
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one("#tts-text-input")
        assert screen.query_one("#speech-axis-row")
        assert screen.query_one("#speech-result-history")
        assert screen.query_one("#speech-param-group")
