"""Non-compact bordered `Checkbox` renders zero content rows -- TASK-18960.

Found during TASK-17961's painted-frame verification: Settings > Workspaces'
"Show archived" `Checkbox` (`settings_screen.py`, NOT `compact=True`) renders
blank in EVERY state, blurred included -- a different defect from the
focus-outline family 17961 fixed (see that task's own
`test_compact_checkbox_focused_frame_shows_its_label` docstring, which
carves this exact widget out of its scope for that reason).

Root cause: `features/_conversations.tcss` carries a bare, unscoped
`Checkbox { width: 100%; margin-bottom: 0; height: 2; }` type selector that
reaches every `Checkbox` app-wide. `ToggleButton` (the base class behind
`Checkbox`/`RadioButton`) pins `border: tall` in its own DEFAULT_CSS -- two
rows of chrome, present even while blurred, unrelated to focus. `height: 2`
gives the whole widget only two rows total, so the border consumes both and
the label/glyph content row has nowhere to paint.

These tests deliberately load the **production** bundle
(`tldw_cli_modular.tcss`) on a bare, id-only harness -- a plain `App` with no
CSS at all cannot reproduce the defect, since the offending rule is the
cause. Painted-frame reads go through the real compositor
(`Screen._compositor.render_strips()`), not `Widget.renderable`/`.value`,
mirroring `Tests/UI/test_compact_focus_outline_render.py` (the harness and
`_rendered_text` helper below are a direct copy of that file's pattern --
kept in a separate file per TASK-18960's brief since this is a distinct bug
family: height-squeeze independent of focus, not outline-over-content-row).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Checkbox

# Repo-root resolution, matching Tests/UI/test_compact_focus_outline_render.py
# and Tests/UI/test_non_obscuring_focus_contract.py -- layout-stable
# regardless of how the package itself is installed/imported.
_BUNDLED_CSS_PATH = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css" / "tldw_cli_modular.tcss"
)
assert _BUNDLED_CSS_PATH.is_file(), (
    f"Production CSS bundle not found at {_BUNDLED_CSS_PATH} -- these tests "
    "must run against the real bundle or they cannot see the offending rule."
)


def _rendered_text(app: App) -> str:
    """Join every compositor strip's segment text into one blob.

    Same helper as `Tests/UI/test_compact_focus_outline_render.py`
    (`_rendered_text`) and `Tests/UI/test_console_session_tab_strip.py`:
    Textual 8.2.7 has no `App.export_text()`, so
    `Screen._compositor.render_strips()` is the only way to read what was
    ACTUALLY painted (post-CSS, post-clip), as opposed to inferring it from
    the un-clipped source string.

    Args:
        app: The running harness app whose current screen to read.

    Returns:
        The painted frame as newline-joined rows of plain segment text.
    """
    strips = app.screen._compositor.render_strips()
    return "\n".join("".join(segment.text for segment in strip) for strip in strips)


class ProductionCssWidgetHarness(App):
    """A single focusable widget under the real app stylesheet.

    No screen-local id/class rules can reach a bare widget like this, so
    whatever these tests observe is what EVERY non-compact `Checkbox` gets
    by default. `AUTO_FOCUS` is disabled so the widget starts genuinely
    blurred -- a real `pilot.press("tab")` is what moves focus onto it,
    mirroring how a user actually reaches it. Identical to the harness in
    `Tests/UI/test_compact_focus_outline_render.py`; replicated here rather
    than imported to keep this file's regression self-contained per
    TASK-18960's brief.
    """

    AUTO_FOCUS = None
    CSS_PATH = str(_BUNDLED_CSS_PATH)

    def __init__(self, widget_factory: Callable[[], Widget]) -> None:
        super().__init__()
        self._widget_factory = widget_factory

    def compose(self) -> ComposeResult:
        """Yield the single probe widget under the production stylesheet.

        Returns:
            The composed widget tree: exactly the one factory-built widget.
        """
        yield self._widget_factory()


@pytest.mark.asyncio
async def test_noncompact_checkbox_blurred_frame_shows_its_label() -> None:
    """AC#2 (blurred half). The regression: a bordered, non-compact
    `Checkbox` on the production bundle is squeezed to zero content rows by
    the unscoped `Checkbox { height: 2; }` rule -- `ToggleButton`'s own
    `border: tall` alone consumes both rows, blurred or not."""
    app = ProductionCssWidgetHarness(
        lambda: Checkbox("Show archived", True, id="probe-checkbox")
    )
    async with app.run_test(size=(60, 10)) as pilot:
        widget = app.query_one("#probe-checkbox", Checkbox)
        await pilot.pause()
        assert not widget.has_focus

        frame = _rendered_text(app)
        assert "Show archived" in frame, (
            "a blurred, non-compact Checkbox's label is not on screen -- "
            "the unscoped Checkbox{height:2} rule squeezes its bordered "
            f"content to zero rows; frame:\n{frame!r}"
        )


@pytest.mark.asyncio
async def test_noncompact_checkbox_focused_frame_shows_its_label() -> None:
    """AC#2 (focused half). Same defect, focused -- the height squeeze is
    focus-independent, unlike the TASK-17961 outline-over-content-row
    family, so this must fail identically to the blurred case above."""
    app = ProductionCssWidgetHarness(
        lambda: Checkbox("Show archived", True, id="probe-checkbox")
    )
    async with app.run_test(size=(60, 10)) as pilot:
        widget = app.query_one("#probe-checkbox", Checkbox)
        await pilot.press("tab")
        await pilot.pause()
        assert widget.has_focus

        frame = _rendered_text(app)
        assert "Show archived" in frame, (
            "a focused, non-compact Checkbox's label is not on screen -- "
            "the unscoped Checkbox{height:2} rule squeezes its bordered "
            f"content to zero rows; frame:\n{frame!r}"
        )
