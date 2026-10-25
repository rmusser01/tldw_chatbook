"""Reattachment paint boundaries do not restart Textual's input thread."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from tldw_chatbook.Web_Server.console_web_driver import ConsoleWebDriver


async def test_reattach_waits_for_resize_and_full_paint_without_starting_input():
    callbacks, output = [], []
    screen = SimpleNamespace(
        refresh=lambda **kw: output.append(("paint", kw)),
        _compositor=SimpleNamespace(render_update=lambda **kw: ("full-frame", kw)),
    )
    app = SimpleNamespace(
        call_after_refresh=lambda callback: callbacks.append(callback),
        _on_resize=AsyncMock(
            side_effect=lambda event: output.append(("resize", event.size))
        ),
        _check_resize=lambda: output.append("resize-applied"),
        screen=screen,
        _background_screens=[],
        _display=lambda screen, frame: output.append(frame),
    )
    driver = object.__new__(ConsoleWebDriver)
    driver._app = app
    driver._key_thread = Mock()
    driver.write_meta = lambda data: output.append(data)
    driver.write = lambda data: output.append(data)
    driver.reattach(100, 30, "current")
    assert output == []
    await callbacks.pop(0)()
    assert output[0] == {"type": "console_attachment_begin", "generation": "current"}
    assert "\x1b[?1049h" in output
    assert "\x1b[?2004h" in output
    assert "\x1b[?1006h" in output
    assert ("resize", (100, 30)) in output
    callbacks.pop(0)()
    assert output[-1] == ("paint", {"repaint": True, "layout": True})
    callbacks.pop(0)()
    assert output[-2] == ("full-frame", {"full": True, "screen_stack": []})
    assert output[-1] == {"type": "console_attachment_ready", "generation": "current"}
    driver._key_thread.start.assert_not_called()
    assert "__GANGLION__" not in repr(output)


async def test_replaced_generation_cannot_publish_late_readiness():
    callbacks, output = [], []
    driver = object.__new__(ConsoleWebDriver)
    driver._app = SimpleNamespace(
        call_after_refresh=lambda callback: callbacks.append(callback),
        _on_resize=AsyncMock(),
        _check_resize=lambda: None,
        screen=SimpleNamespace(
            refresh=lambda **kw: None,
            _compositor=SimpleNamespace(render_update=lambda **kw: None),
        ),
        _background_screens=[],
        _display=lambda *args: None,
    )
    driver.write_meta = output.append
    driver.write = lambda _: None
    driver.reattach(80, 24, "first")
    await callbacks.pop(0)()
    driver.reattach(100, 30, "second")
    while callbacks:
        result = callbacks.pop(0)()
        if result is not None:
            await result
    assert {"type": "console_attachment_ready", "generation": "first"} not in output
    assert output[-1] == {"type": "console_attachment_ready", "generation": "second"}


def test_resize_during_partial_console_mount_does_not_consume_width_band():
    from textual.css.query import NoMatches
    from textual.events import Resize
    from textual.geometry import Size

    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = SimpleNamespace(
        _last_console_workspace_width_band=None,
        _request_console_control_bar_sync=lambda: None,
        app=SimpleNamespace(focused=None),
    )

    def query(selector):
        if selector == "#console-workspace-grid":
            return object()
        raise NoMatches("children are still mounting")

    screen.query_one = query
    ChatScreen._adapt_console_workspace_to_width(
        screen, Resize(Size(80, 24), Size(80, 24))
    )
    assert screen._last_console_workspace_width_band is None
