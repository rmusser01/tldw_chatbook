"""Conversation filter and reader continuity with the production stylesheet set."""

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Input

from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _records():
    return [
        {
            "id": "chat-alpha",
            "title": "Alpha planning",
            "version": 1,
            "message_count": 24,
        },
        {"id": "chat-beta", "title": "Beta review", "version": 1, "message_count": 2},
    ]


def _painted(screen, widget):
    region = widget.region
    strips = list(screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


async def _open(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-conversations", Button).press()
    await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
    await _wait_for_condition(
        pilot,
        lambda: screen._conversations_state.reader_state.complete,
        message="reader did not finish loading",
    )
    return screen


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize(
    "selector", ["#library-conversations-filter", "#library-conversation-reader-find"]
)
async def test_resize_retains_conversation_query_editing(theme, selector):
    app = _build_test_app()
    _seed_conversations(app, _records())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        host.theme = theme
        screen = await _open(host, pilot)
        field = screen.query_one(selector, Input)
        field.focus()
        field.value = "unsubmitted query"
        await pilot.press("end", "shift+left", "shift+left")
        selection = field.selection
        loaded = screen._conversations_state.reader_state.loaded_id
        for size in ((80, 24), (170, 48), (170, 24)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            current = screen.query_one(selector, Input)
            assert current is field
            assert current.value == "unsubmitted query"
            assert current.selection == selection
            if (
                selector == "#library-conversations-filter"
                and not screen._conversations_state.reader_layout.items_open
            ):
                # The shell protects the reader by collapsing Items. Its grip
                # must own evacuation, then return to the same editor on Enter.
                grip = screen.query_one("#library-conversations-items-grip")
                assert screen.focused is grip
                await pilot.press("enter")
                await pilot.pause()
            assert screen.focused is field
            assert "query" in _painted(screen, field)
            assert screen._conversations_state.reader_state.loaded_id == loaded
        await pilot.press("x")
        assert field.value == "unsubmitted quex"


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_empty_conversation_filter_can_be_cleared_from_keyboard(theme, size):
    app = _build_test_app()
    _seed_conversations(app, _records())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        host.theme = theme
        screen = await _open(host, pilot)
        await pilot.resize_terminal(*size)
        if not screen._conversations_state.reader_layout.items_open:
            screen.query_one("#library-conversations-items-grip").focus()
            await pilot.press("enter")
        field = screen.query_one("#library-conversations-filter", Input)
        field.focus()
        field.value = "nothing-matches"
        await pilot.press("enter")
        await _wait_for_selector(
            screen, pilot, "#library-conversations-empty-clear-filter"
        )
        clear = screen.query_one("#library-conversations-empty-clear-filter", Button)
        clear.focus()
        await pilot.pause()
        assert "Clear filter" in _painted(screen, clear)
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is screen.query_one("#library-conversations-filter"),
            message="filter did not regain focus",
        )
        await pilot.press("b")
        assert screen.query_one("#library-conversations-filter", Input).value == "b"


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_conversation_find_and_new_focus_survive_resize(theme, monkeypatch):
    app = _build_test_app()
    _seed_conversations(app, _records())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(170, 48)) as pilot:
        host.theme = theme
        screen = await _open(host, pilot)
        find = screen.query_one("#library-conversation-reader-find", Input)
        find.focus()
        find.value = "Saved message 20"
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: bool(screen._conversations_state.reader_state.find_matches),
            message="find did not resolve a matching message",
        )
        reader = screen.query_one("#library-conversation-reader")
        messages = screen.query_one(
            "#library-conversation-reader-messages", VerticalScroll
        )
        match = next(
            row
            for row in screen.query(".library-conversation-reader-message")
            if row.message_id == "chat-alpha-message-20"
        )
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is match,
            message="match did not receive focus",
        )
        assert "Saved message 20" in _painted(screen, match)
        for size in ((80, 24), (170, 48), (170, 24)):
            await pilot.resize_terminal(*size)
            await pilot.pause()
            assert screen.query_one("#library-conversation-reader") is reader
            assert screen.focused is match
            assert "Saved message 20" in _painted(screen, match), (
                size,
                match.region,
                messages.region,
                messages.content_region,
                messages.scroll_offset,
                reader.region,
                reader.scroll_offset,
                reader.max_scroll_y,
            )
            assert messages.scroll_y > 0
        pending = []
        call_after_refresh = reader.call_after_refresh

        def hold_reveal(callback, *args, **kwargs):
            if callback == reader._reveal_reader_focus:
                pending.append((callback, args, kwargs))
                return True
            return call_after_refresh(callback, *args, **kwargs)

        monkeypatch.setattr(reader, "call_after_refresh", hold_reveal)
        await pilot.resize_terminal(80, 24)
        assert pending, "resize did not schedule the reader reveal"
        info = screen.query_one("#library-conversation-reader-info", Button)
        info.focus()
        await pilot.pause()
        assert screen.focused is info
        assert "Info" in _painted(screen, info)
        offset = reader.scroll_offset
        for callback, args, kwargs in pending:
            callback(*args, **kwargs)
        await pilot.pause()
        assert screen.focused is info
        assert "Info" in _painted(screen, info)
        assert reader.scroll_offset == offset
        await pilot.press("enter")
        assert screen._conversations_state.reader_state.mode == "info"
