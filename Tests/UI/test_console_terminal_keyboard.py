from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.app import ComposeResult

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Terminal.screen_model import (
    SafeTerminalCell,
    SafeTerminalLine,
    SafeTerminalRun,
    TerminalScreenSnapshot,
)
from tldw_chatbook.Widgets.Console.console_terminal_workspace import (
    ConsoleTerminalInputRequested,
    TerminalViewport,
    terminal_key_bytes,
)


def _line(text: str) -> SafeTerminalLine:
    return SafeTerminalLine(
        runs=(
            SafeTerminalRun(
                cells=tuple(SafeTerminalCell(character, 1) for character in text)
            ),
        )
    )


class _KeyEvent:
    def __init__(self, key: str, character: str | None = None) -> None:
        self.key = key
        self.character = character
        self.stopped = False
        self.default_prevented = False

    def stop(self) -> None:
        self.stopped = True

    def prevent_default(self) -> None:
        self.default_prevented = True


class _ViewportApp(ConsolidatedCSSApp):
    def __init__(self, viewport: TerminalViewport) -> None:
        super().__init__()
        self.viewport = viewport
        self.inputs: list[bytes] = []

    def compose(self) -> ComposeResult:
        yield self.viewport

    def on_console_terminal_input_requested(
        self,
        message: ConsoleTerminalInputRequested,
    ) -> None:
        self.inputs.append(message.data)


@pytest.mark.parametrize(
    ("key", "character", "expected"),
    [
        ("tab", "\t", b"\t"),
        ("enter", "\r", b"\r"),
        ("pageup", None, b"\x1b[5~"),
        ("pagedown", None, b"\x1b[6~"),
        ("up", None, b"\x1b[A"),
        ("down", None, b"\x1b[B"),
        ("left", None, b"\x1b[D"),
        ("right", None, b"\x1b[C"),
        ("home", None, b"\x1b[H"),
        ("end", None, b"\x1b[F"),
        ("insert", None, b"\x1b[2~"),
        ("delete", None, b"\x1b[3~"),
        ("shift+tab", None, b"\x1b[Z"),
        ("backtab", None, b"\x1b[Z"),
        ("f2", None, b"\x1bOQ"),
        ("f3", None, b"\x1bOR"),
        ("f4", None, b"\x1bOS"),
        ("f5", None, b"\x1b[15~"),
        ("f7", None, b"\x1b[18~"),
        ("f8", None, b"\x1b[19~"),
        ("f9", None, b"\x1b[20~"),
        ("f10", None, b"\x1b[21~"),
        ("f11", None, b"\x1b[23~"),
        ("f12", None, b"\x1b[24~"),
        ("backspace", None, b"\x7f"),
        ("ctrl+c", None, b"\x03"),
        ("alt+x", "x", b"\x1bx"),
        ("a", "a", b"a"),
        ("é", "é", "é".encode()),
    ],
)
def test_terminal_key_encoding_is_byte_accurate(
    key: str,
    character: str | None,
    expected: bytes,
) -> None:
    assert terminal_key_bytes(key, character) == expected


@pytest.mark.parametrize("key", ["ctrl+p", "ctrl+q", "f1", "f6"])
def test_chatbook_global_keys_are_never_encoded(key: str) -> None:
    assert terminal_key_bytes(key, None) is None


def test_input_mode_forwards_terminal_keys_and_consumes_release_chord(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewport = TerminalViewport()
    posted: list[object] = []
    monkeypatch.setattr(viewport, "post_message", posted.append)

    for key, character, expected in (
        ("tab", "\t", b"\t"),
        ("shift+tab", None, b"\x1b[Z"),
        ("insert", None, b"\x1b[2~"),
        ("f2", None, b"\x1bOQ"),
        ("pageup", None, b"\x1b[5~"),
        ("pagedown", None, b"\x1b[6~"),
    ):
        event = _KeyEvent(key, character)
        viewport.on_key(event)
        assert event.stopped is True
        assert event.default_prevented is True
        message = posted.pop()
        assert isinstance(message, ConsoleTerminalInputRequested)
        assert message.data == expected

    release = _KeyEvent("ctrl+right_square_bracket")
    viewport.on_key(release)
    assert release.stopped is True
    assert release.default_prevented is True
    assert viewport.input_focused is False
    assert posted == []


@pytest.mark.parametrize("key", ["ctrl+p", "ctrl+q", "f1", "f6"])
def test_input_mode_bubbles_chatbook_globals_without_forwarding(
    key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    viewport = TerminalViewport()
    posted: list[object] = []
    monkeypatch.setattr(viewport, "post_message", posted.append)
    event = _KeyEvent(key)

    viewport.on_key(event)

    assert event.stopped is False
    assert event.default_prevented is False
    assert posted == []


@pytest.mark.asyncio
async def test_navigation_keys_move_locally_and_enter_returns_without_newline() -> None:
    viewport = TerminalViewport()
    app = _ViewportApp(viewport)
    async with app.run_test(size=(80, 24)):
        viewport.project(
            session_id="one",
            snapshot=TerminalScreenSnapshot(
                lines=(_line("live-1"), _line("live-2")),
                scrollback=tuple(_line(f"old-{index}") for index in range(6)),
                generation=1,
            ),
        )
        viewport.release_input()

        up = _KeyEvent("up")
        viewport.on_key(up)
        assert viewport.history_offset == 1
        assert up.stopped is True

        page_up = _KeyEvent("pageup")
        viewport.on_key(page_up)
        assert viewport.history_offset == 3

        home = _KeyEvent("home")
        viewport.on_key(home)
        assert viewport.history_offset == 6

        down = _KeyEvent("down")
        viewport.on_key(down)
        assert viewport.history_offset == 5

        end = _KeyEvent("end")
        viewport.on_key(end)
        assert viewport.history_offset == 0

        viewport.release_input()
        enter = _KeyEvent("enter", "\r")
        viewport.on_key(enter)
        assert viewport.input_focused is True
        assert app.inputs == []

        viewport.release_input()
        tab = _KeyEvent("tab", "\t")
        viewport.on_key(tab)
        assert tab.stopped is False
        assert app.inputs == []


@pytest.mark.asyncio
async def test_output_freezes_released_view_and_jump_live_clears_count() -> None:
    viewport = TerminalViewport()
    app = _ViewportApp(viewport)
    first = TerminalScreenSnapshot(
        lines=(_line("live-1"), _line("live-2")),
        scrollback=(_line("old-1"), _line("old-2")),
        generation=1,
    )
    async with app.run_test(size=(80, 24)):
        viewport.project(session_id="one", snapshot=first)
        viewport.release_input()
        viewport.scroll_up(1)
        before = viewport.renderable.plain

        viewport.project(
            session_id="one",
            snapshot=replace(
                first,
                scrollback=(*first.scrollback, _line("old-3")),
                lines=(_line("live-2"), _line("new-live")),
                generation=2,
                dirty_lines=(0, 1),
            ),
        )

        assert viewport.history_offset > 0
        assert viewport.new_output_count == 3
        assert viewport.renderable.plain == before

        viewport.jump_live()

        assert viewport.history_offset == 0
        assert viewport.new_output_count == 0
        assert "new-live" in viewport.renderable.plain


@pytest.mark.parametrize("key", ["up", "down", "pageup", "pagedown", "home", "end"])
@pytest.mark.asyncio
async def test_alternate_screen_local_history_is_a_clear_noop(key: str) -> None:
    viewport = TerminalViewport()
    app = _ViewportApp(viewport)
    async with app.run_test(size=(80, 24)):
        viewport.project(
            session_id="one",
            snapshot=TerminalScreenSnapshot(
                lines=(_line("vim"),),
                in_alternate=True,
                generation=1,
            ),
        )
        viewport.release_input()
        viewport.history_offset = 2
        viewport.new_output_count = 3

        viewport.on_key(_KeyEvent(key))

        assert viewport.history_offset == 2
        assert viewport.new_output_count == 3
        assert viewport.status_text == "Alternate screen has no local scrollback."


@pytest.mark.asyncio
async def test_mouse_wheel_enters_local_navigation_without_emitting_mouse_bytes() -> (
    None
):
    viewport = TerminalViewport()
    app = _ViewportApp(viewport)
    async with app.run_test(size=(80, 24)):
        viewport.project(
            session_id="one",
            snapshot=TerminalScreenSnapshot(
                lines=(_line("live"),),
                scrollback=(_line("old"),),
                generation=1,
            ),
        )
        event = SimpleNamespace(stop=lambda: None)

        viewport.on_mouse_scroll_up(event)

        assert viewport.input_focused is False
        assert viewport.history_offset == 1
        assert app.inputs == []
