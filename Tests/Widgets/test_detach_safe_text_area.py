"""TASK-32049: a TextArea repainted after detaching must not raise.

Textual clears a widget's component styles when it detaches; a screen repaint
already queued can still reach `TextArea.render_lines`, whose theme step asks
for `text-area--gutter` and raises KeyError. That is the fast-lane flake seen
across `Tests/UI/test_mcp_workbench.py`, reproduced here deterministically.
"""

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Region
from textual.widgets import TextArea

from tldw_chatbook.Widgets.detach_safe_text_area import DetachSafeTextArea


class _Host(App):
    def __init__(self, widget_type: type[TextArea]) -> None:
        super().__init__()
        self._widget_type = widget_type

    def compose(self) -> ComposeResult:
        yield self._widget_type('{"a": 1}', id="editor")


async def _render_after_removal(widget_type: type[TextArea]) -> list:
    app = _Host(widget_type)
    async with app.run_test(size=(60, 10)) as pilot:
        editor = app.query_one("#editor", TextArea)
        await pilot.pause()
        await editor.remove()
        return editor.render_lines(Region(0, 0, 20, 3))


@pytest.mark.asyncio
async def test_stock_text_area_raises_after_detach():
    """Negative control: proves the race is real on the pinned Textual."""
    try:
        await _render_after_removal(TextArea)
    except KeyError as exc:
        assert "text-area--gutter" in str(exc)
    else:
        pytest.fail(
            "Stock TextArea no longer raises when a detached widget is repainted: "
            "Textual appears to have fixed the detach race, so DetachSafeTextArea "
            "may now be removable (TASK-32049)."
        )


@pytest.mark.asyncio
async def test_detach_safe_text_area_renders_blank_after_detach():
    """A detached DetachSafeTextArea renders blank strips instead of raising."""
    lines = await _render_after_removal(DetachSafeTextArea)

    assert len(lines) == 3
    assert all(line.cell_length == 20 for line in lines)
    assert all(not line.text.strip() for line in lines)


@pytest.mark.asyncio
async def test_attached_detach_safe_text_area_renders_like_stock():
    """The guard must not change rendering while attached."""
    rendered = {}
    for widget_type in (TextArea, DetachSafeTextArea):
        app = _Host(widget_type)
        async with app.run_test(size=(60, 10)) as pilot:
            await pilot.pause()
            editor = app.query_one("#editor", TextArea)
            rendered[widget_type] = [
                line.text for line in editor.render_lines(Region(0, 0, 20, 3))
            ]
    assert rendered[DetachSafeTextArea] == rendered[TextArea]
