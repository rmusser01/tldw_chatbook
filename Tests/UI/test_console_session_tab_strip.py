"""Console session tab strip: scroll overflow, streaming glyph, middle-click close."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual import events
from textual.app import App
from textual.containers import HorizontalScroll
from textual.content import Content
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import ConsoleRunMarker
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_session_surface import (
    ConsoleSessionSurface,
    ConsoleSessionTabStrip,
)


def _rendered_tooltip(button: Button) -> str:
    """Return a tab's tooltip as Textual will actually DISPLAY it.

    TASK-1233 review round 1: `Button.tooltip` is a markup *source*
    string, not the rendered text -- Textual's `Tooltip` widget parses it
    the same way `Content.from_markup` does, at display time. Asserting
    the raw attribute (as round 0 of this task did) can pass against a
    tooltip that is silently broken once rendered: an un-escaped literal
    "[" is read as a style-tag start, and an unrecognized tag name is
    DROPPED from the rendered text rather than shown literally. Render
    through the same path the app uses before asserting.
    """
    return Content.from_markup(button.tooltip or "").plain


class TabStripHost(App[None]):
    def __init__(self):
        super().__init__()
        self.pressed_ids: list[str] = []

    def compose(self):
        yield ConsoleSessionSurface(SimpleNamespace(notify=MagicMock()))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.pressed_ids.append(event.button.id or "")


def _sessions(count: int) -> list[ConsoleChatSession]:
    return [
        ConsoleChatSession(title=f"Session {i}", id=f"s{i}")
        for i in range(1, count + 1)
    ]


@pytest.mark.asyncio
async def test_tab_strip_is_horizontally_scrollable() -> None:
    app = TabStripHost()
    async with app.run_test(size=(80, 24)):
        strip = app.query_one("#console-native-tab-strip", HorizontalScroll)
        assert isinstance(strip, HorizontalScroll)


@pytest.mark.asyncio
async def test_streaming_session_tab_shows_run_glyph() -> None:
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        sessions = _sessions(2)
        await surface.sync_sessions(sessions=sessions, active_session_id="s1")
        await pilot.pause()

        await surface.sync_sessions(
            sessions=sessions,
            active_session_id="s1",
            streaming_session_id="s2",
        )
        await pilot.pause()

        streaming_tab = app.query_one("#console-session-tab-s2", Button)
        idle_tab = app.query_one("#console-session-tab-s1", Button)
        assert str(streaming_tab.label).startswith("●")
        assert not str(idle_tab.label).startswith("●")
        # TASK-1233: the tooltip decodes the ● glyph in context instead of
        # a bare "Run in progress." fragment.
        assert "agent running" in _rendered_tooltip(streaming_tab)

        # Glyph clears when the run ends.
        await surface.sync_sessions(
            sessions=sessions,
            active_session_id="s1",
            streaming_session_id=None,
        )
        await pilot.pause()
        assert not str(streaming_tab.label).startswith("●")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("marker", "expected_fragment"),
    [
        (ConsoleRunMarker.NONE, None),
        (ConsoleRunMarker.RUNNING, "— agent running."),
        (ConsoleRunMarker.NEEDS_APPROVAL, "— waiting for approval."),
        (ConsoleRunMarker.FINISHED_OK, "— finished — unseen."),
        (ConsoleRunMarker.FINISHED_FAILED, "— failed — unseen."),
    ],
)
async def test_tab_tooltip_decodes_marker_meaning(
    marker: ConsoleRunMarker, expected_fragment: str | None
) -> None:
    """TASK-1233 AC#1: a tab's tooltip decodes its fleet run-marker glyph in
    context instead of leaving the reader to infer meaning from shape alone.

    ``ConsoleRunMarker.NONE`` is the pinned no-regression case: the tooltip
    must be byte-for-byte the pre-task-1233 copy (no dangling suffix, no
    stray em dash).
    """
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        sessions = _sessions(2)
        await surface.sync_sessions(
            sessions=sessions,
            active_session_id="s1",
            run_markers={"s2": marker},
        )
        await pilot.pause()

        tab = app.query_one("#console-session-tab-s2", Button)
        tooltip = _rendered_tooltip(tab)
        if expected_fragment is None:
            assert tooltip == (
                "Switch to Console tab: Session 2. Middle-click closes the tab."
            )
        else:
            assert expected_fragment in tooltip


@pytest.mark.asyncio
async def test_tab_tooltip_escapes_markup_in_title() -> None:
    """A session title containing bracket tokens must render literally in
    the tooltip, not be interpreted as Rich markup (Textual's Tooltip is a
    Static with markup parsing on).

    TASK-1233 review round 1: asserts the RENDERED plain text (what a user
    actually sees), not the raw markup-source attribute -- an earlier round
    pinned the raw escaped-with-backslashes form, which would not have
    caught a sibling bug (a differently-assembled fragment left
    un-escaped) the way rendering it does.
    """
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        sessions = [ConsoleChatSession(title="[red]Alarm[/red]", id="s1")]
        await surface.sync_sessions(
            sessions=sessions,
            active_session_id="s1",
            run_markers={"s1": ConsoleRunMarker.NEEDS_APPROVAL},
        )
        await pilot.pause()

        tab = app.query_one("#console-session-tab-s1", Button)
        assert _rendered_tooltip(tab) == (
            "Active Console tab: [red]Alarm[/red] — waiting for approval."
            " Click again to rename. Middle-click closes the tab."
        )


@pytest.mark.asyncio
async def test_session_title_surfaces_project_control_whitespace_to_one_line() -> None:
    app = TabStripHost()
    raw_title = "Chat with Nyx\n\tAdmin\x00[/bold]"
    session = ConsoleChatSession(title=raw_title, id="s1")

    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        await surface.sync_sessions(sessions=[session], active_session_id="s1")
        await pilot.pause()

        tab = app.query_one("#console-session-tab-s1", Button)
        title = app.query_one("#console-transcript-title").renderable
        assert "\n" not in str(tab.label)
        assert "\t" not in str(tab.label)
        assert "Chat with Nyx Admi" in str(tab.label)
        assert "\n" not in str(title)
        assert "\t" not in str(title)
        assert "Nyx Admin?[/bold]" in str(title)
        assert session.title == raw_title


@pytest.mark.asyncio
async def test_middle_click_on_tab_presses_its_close_button() -> None:
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        await surface.sync_sessions(sessions=_sessions(2), active_session_id="s1")
        await pilot.pause()

        clicked = await pilot.click("#console-session-tab-s2", button=2)
        await pilot.pause()

        assert clicked
        assert "console-close-session-tab-s2" in app.pressed_ids


@pytest.mark.asyncio
async def test_active_tab_is_scrolled_into_view() -> None:
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        sessions = _sessions(8)
        await surface.sync_sessions(sessions=sessions, active_session_id="s8")
        await pilot.pause()
        await pilot.pause()

        strip = app.query_one("#console-native-tab-strip", HorizontalScroll)
        assert strip.scroll_x > 0


def _rendered_text(app: App) -> str:
    """Join every compositor strip's segment text into one blob.

    Textual 8.2.7 has no `App.export_text()`; `screen._compositor.render_strips()`
    is the way to read what was ACTUALLY rendered (post-clipping), as opposed to
    inferring it from styles or the un-clipped `Button.label` source string.
    """
    strips = app.screen._compositor.render_strips()
    return "\n".join("".join(segment.text for segment in strip) for strip in strips)


class _PaddedTabStripHost(TabStripHost):
    """`TabStripHost` plus the ONE production CSS rule this pin cares about.

    `TabStripHost` composes `ConsoleSessionSurface` bare -- it never loads
    the app's real stylesheet, so the tab-strip buttons' production
    `padding: 0 1` (`_agentic_terminal.tcss`, selector
    `#console-new-chat-tab, #console-new-temporary-tab`) never applies there.
    Without it, `CONSOLE_NEW_TAB_BUTTON_WIDTH`'s box math is short by the 2
    padding cells the real Console screen always reserves, so a bare
    `TabStripHost` cannot reproduce the "Temporar" clip this pin guards
    against. Reproducing just that one rule (rather than the whole bundle)
    keeps the test isolated to what it actually verifies.
    """

    CSS = """
    #console-new-chat-tab,
    #console-new-temporary-tab {
        padding: 0 1;
    }
    """


@pytest.mark.asyncio
async def test_new_tab_and_temporary_buttons_render_unclipped_labels() -> None:
    """RAG-47: at the old `CONSOLE_NEW_TAB_BUTTON_WIDTH = 12`, the fixed
    inline width, the production `padding: 0 1`, and Textual's own
    `line-pad: 1` box math left only 8 usable cells for "Temporary" (9
    chars), clipping it to "Temporar". Assert the actual COMPOSITED text
    (not the un-clipped `Button.label` source) shows both tab-strip
    buttons' full labels -- "New tab" (the shorter sibling sharing the
    same width constant) must still fit too.
    """
    app = _PaddedTabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        await surface.sync_sessions(sessions=_sessions(1), active_session_id="s1")
        await pilot.pause()

        rendered = _rendered_text(app)
        assert "Temporary" in rendered, (
            f"'Temporary' tab-strip button label was clipped:\n{rendered}"
        )
        assert "New tab" in rendered, (
            f"'New tab' tab-strip button label was clipped:\n{rendered}"
        )


# -- TASK-28028: wheel scrolling + overflow hints ---------------------------


async def _wheel(
    pilot, strip: ConsoleSessionTabStrip, *, down: bool, ctrl: bool = False
) -> None:
    """Post a real-shape SGR mouse-wheel event to the strip, as a terminal delivers it.

    ``Pilot`` has no wheel helper; posting the event message runs the same
    ``_on_mouse_scroll_*`` dispatch a real wheel travels through.
    """
    event_cls = events.MouseScrollDown if down else events.MouseScrollUp
    strip.post_message(
        event_cls(
            widget=strip,
            x=strip.region.x + 5,
            y=strip.region.y,
            delta_x=0,
            delta_y=1 if down else -1,
            button=0,
            shift=False,
            meta=False,
            ctrl=ctrl,
        )
    )
    await pilot.pause()


@pytest.mark.asyncio
async def test_plain_wheel_scrolls_strip_horizontally() -> None:
    """TASK-28028: a plain vertical wheel over the strip scrolls it horizontally.

    Textual 8.2.8's base wheel handler does nothing for a height-1
    ``HorizontalScroll`` (vertical scroll is disabled), so without this the
    only horizontal wheel paths are the undiscoverable shift/ctrl chords.
    """
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        # 10 tabs x 24 cells (tab 21 + close 3) + 26 cells of new-tab buttons
        # = 266 cells of content in an 80-cell strip: guaranteed overflow.
        await surface.sync_sessions(sessions=_sessions(10), active_session_id="s1")
        await pilot.pause()
        await pilot.pause()

        strip = app.query_one("#console-native-tab-strip", ConsoleSessionTabStrip)
        assert strip.scroll_x == 0

        await _wheel(pilot, strip, down=True)
        assert strip.scroll_x > 0, "wheel-down did not scroll the strip right"

        scroll_after_down = strip.scroll_x
        await _wheel(pilot, strip, down=False)
        assert strip.scroll_x < scroll_after_down, (
            "wheel-up did not scroll the strip back left"
        )


@pytest.mark.asyncio
async def test_wheel_at_left_edge_keeps_strip_at_zero() -> None:
    """A wheel-up at scroll position 0 must be a no-op, not a jump."""
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        await surface.sync_sessions(sessions=_sessions(10), active_session_id="s1")
        await pilot.pause()
        await pilot.pause()

        strip = app.query_one("#console-native-tab-strip", ConsoleSessionTabStrip)
        await _wheel(pilot, strip, down=False)
        assert strip.scroll_x == 0


@pytest.mark.asyncio
async def test_overflow_hints_follow_hidden_tabs_and_scroll_position() -> None:
    """TASK-28028: ‹ › hints appear only on the side with hidden tabs."""
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        await surface.sync_sessions(sessions=_sessions(10), active_session_id="s1")
        await pilot.pause()
        await pilot.pause()

        left = app.query_one("#console-tab-overflow-left", Static)
        right = app.query_one("#console-tab-overflow-right", Static)
        strip = app.query_one("#console-native-tab-strip", ConsoleSessionTabStrip)

        # At the far-left end: tabs hidden on the right only.
        assert right.styles.visibility == "visible", (
            "right hint missing with tabs hidden right"
        )
        assert left.styles.visibility == "hidden", (
            "left hint shown with nothing hidden left"
        )

        await _wheel(pilot, strip, down=True)
        assert left.styles.visibility == "visible", (
            "left hint missing after scrolling right"
        )


@pytest.mark.asyncio
async def test_overflow_hint_toggles_never_shift_the_strip_region() -> None:
    """Qodo PR #2327 review: hint toggles must not resize or move the strip.

    The hints hide via ``visibility`` (cells stay in the row's layout), not
    ``display: none`` (cells leave it). If this regresses, a hint appearing
    narrows the 1fr strip by a cell and visibly jumps the tabs at every
    threshold crossing. Pin the strip's region across a toggle cycle.
    """
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        await surface.sync_sessions(sessions=_sessions(10), active_session_id="s1")
        await pilot.pause()
        await pilot.pause()

        strip = app.query_one("#console-native-tab-strip", ConsoleSessionTabStrip)
        left = app.query_one("#console-tab-overflow-left", Static)

        region_before = strip.region
        assert left.styles.visibility == "hidden"

        # Wheel right until the left hint appears (a toggle happened).
        for _ in range(3):
            await _wheel(pilot, strip, down=True)
        assert left.styles.visibility == "visible"
        assert strip.scroll_x > 0
        assert strip.region == region_before, (
            "strip region moved/resized when a hint toggled: "
            f"{region_before} -> {strip.region}"
        )

        # And back: the reverse toggle must be just as inert.
        for _ in range(6):
            await _wheel(pilot, strip, down=False)
        assert left.styles.visibility == "hidden"
        assert strip.region == region_before


@pytest.mark.asyncio
async def test_overflow_hints_hidden_without_overflow() -> None:
    """A strip that fits shows neither hint."""
    app = TabStripHost()
    async with app.run_test(size=(80, 24)) as pilot:
        surface = app.query_one(ConsoleSessionSurface)
        await surface.sync_sessions(sessions=_sessions(1), active_session_id="s1")
        await pilot.pause()
        await pilot.pause()

        assert (
            app.query_one("#console-tab-overflow-left", Static).styles.visibility
            == "hidden"
        )
        assert (
            app.query_one("#console-tab-overflow-right", Static).styles.visibility
            == "hidden"
        )


def test_chat_screen_exposes_rail_body_height_seam() -> None:
    """TASK-28028: the wiring lambda's ``screen._console_rail_body_height()``
    must exist on ``ChatScreen`` -- the adaptive row-limit WIP shipped
    without it and every Console screen resume crashed with AttributeError.
    The functional leg is the switcher pair in
    ``test_console_native_chat_flow.py``, which drives the real screen.
    """
    assert callable(getattr(ChatScreen, "_console_rail_body_height", None))

