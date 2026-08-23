"""Per-click cost of the screen-level selection-menu dismissal (TASK-21119).

``ChatScreen._dismiss_console_selection_menus_outside_transcript`` runs on
BOTH ``on_mouse_down`` and ``on_click`` of the same physical press -- every
press anywhere on the Console (composer, buttons, rails). Before this task it
ran ``self.query(ConsoleTranscript)`` + ``self.query(ConsoleSelectionMenu)``
each time: four full-screen DOM walks per press on the largest-DOM screen in
the app (Docs/Design/2026-08-22-holistic-perf-review.md, finding 21119).

The counter probe below pins the idle cost at ZERO full-screen walks, and the
control arms pin the behaviour the optimization must not buy that zero with:
a mounted menu is still dismissed, and an active selection with NO menu (the
keyboard-selection mode state) is still cleared by a click outside.
"""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Static

from Tests.UI.test_console_left_rail import make_console_pilot
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_selection_menu import (
    ConsoleSelectionMenu,
    selection_menus_on_screen,
)
from tldw_chatbook.Widgets.Console.console_session_surface import ConsoleSessionSurface
from tldw_chatbook.Widgets.Console.console_transcript import (
    _LIVE_TRANSCRIPTS,
    ConsoleTranscript,
    ConsoleTranscriptMessage,
    console_transcripts_on_screen,
)
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)


class _BareModal(ModalScreen[None]):
    """A modal pushed above the Console, for the screen-scope arm."""

    def compose(self) -> ComposeResult:
        yield Static("modal body", id="bare-modal-body")


def _install_query_counter(screen) -> dict[str, int]:
    """Count full-screen ``query`` walks made through ``screen``.

    Shadows the bound method on the instance, so it also catches the
    transcript's own ``self.screen.query(...)`` menu lookups (the same
    screen object) -- every full-screen walk the press path can make.
    """
    counts = {"transcript": 0, "menu": 0, "total": 0}
    original = screen.query

    def counting_query(selector=None):
        counts["total"] += 1
        if selector is ConsoleTranscript:
            counts["transcript"] += 1
        elif selector is ConsoleSelectionMenu:
            counts["menu"] += 1
        return original(selector)

    screen.query = counting_query  # type: ignore[method-assign]
    return counts


async def _seed_row(pilot, transcript: ConsoleTranscript) -> ConsoleTranscriptMessage:
    """Put one selectable row in the transcript and return its widget."""
    transcript.set_messages(
        [
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER, content="selectable body text", id="row1"
            )
        ]
    )
    await transcript.refresh_messages()
    await pilot.pause()
    return pilot.app.screen.query_one(
        "#console-message-row1", ConsoleTranscriptMessage
    )


@pytest.mark.asyncio
async def test_idle_press_makes_no_full_screen_selection_walks():
    """No menu, no selection: a press outside the transcript walks nothing.

    Measured on dev ``ae817fefe`` before the fix, per PHYSICAL press:
    a composer press = 3 full-screen walks (the composer swallows the Click,
    so the handler runs once: ``ConsoleTranscript`` + the menu walk inside
    ``_remove_selection_menu`` + ``ConsoleSelectionMenu``); a rail press =
    6 (the handler runs on MouseDown *and* Click). Both must now be 0.
    """
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        assert not screen.query(ConsoleSelectionMenu)  # precondition: idle
        counts = _install_query_counter(screen)

        await pilot.click("#console-native-composer")
        await pilot.pause()

        assert counts["transcript"] == 0, counts
        assert counts["menu"] == 0, counts

        # The other half of the press: ``on_click`` calls the same handler
        # again whenever the target does not swallow the Click. Driven
        # directly so the count is independent of which rail widget stops
        # what (and of any side effect a real rail click would trigger).
        composer = screen.query_one("#console-native-composer", ConsoleComposerBar)
        counts.update(transcript=0, menu=0, total=0)
        screen._dismiss_console_selection_menus_outside_transcript(composer)
        screen._dismiss_console_selection_menus_outside_transcript(composer)
        await pilot.pause()

        assert counts["transcript"] == 0, counts
        assert counts["menu"] == 0, counts


@pytest.mark.asyncio
async def test_screen_mounted_menu_is_still_dismissed_by_an_outside_press():
    """Control arm: the production mount site (the screen) still folds."""
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        transcript = screen.query_one("#console-native-transcript", ConsoleTranscript)
        await screen.mount(ConsoleSelectionMenu(screen_x=2, screen_y=2, owner=transcript))
        await pilot.pause()
        assert screen.query_one(ConsoleSelectionMenu)

        await pilot.click("#console-native-composer")
        await pilot.pause()

        assert not screen.query(ConsoleSelectionMenu)


@pytest.mark.asyncio
async def test_active_selection_without_a_menu_is_still_cleared():
    """Control arm: keyboard-selection state (highlight, no menu) clears.

    ``s`` arms a selection with no menu mounted; a click on the composer is
    the user moving on, and the reverse-video strip must not survive it. A
    mounted-menu-only early return would leave the highlight painted.
    """
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        transcript = screen.query_one("#console-native-transcript", ConsoleTranscript)
        row = await _seed_row(pilot, transcript)
        transcript.selection_manager.begin_drag(row.id, 0)
        transcript.selection_manager.extend_drag(row.id, 5)
        row.set_selection_range(0, 5)
        transcript._selection_origin_row = row
        assert transcript.selection_manager.state.selection is not None

        await pilot.click("#console-native-composer")
        await pilot.pause()

        assert transcript.selection_manager.state.selection is None
        assert transcript._selection_origin_row is None
        assert row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_a_recomposed_transcript_replaces_the_old_one_with_no_bookkeeping():
    """The staleness arm: recompose swaps the transcript, the gate follows.

    A cached transcript reference would go stale exactly here (recompose
    tears the old widget out and builds a new one). Nothing invalidates
    anything: the registry is only a candidate set, and attachment is
    re-derived from the live DOM on every read, so the detached instance
    drops out and the fresh one is already in.
    """
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        old = screen.query_one("#console-native-transcript", ConsoleTranscript)
        assert console_transcripts_on_screen(screen) == [old]

        # The session surface owns the transcript surface, which builds the
        # transcript in its constructor -- recomposing THAT is what swaps the
        # widget out (the transcript surface re-yields its own instance).
        session_surface = old.parent.parent if old.parent is not None else None
        assert isinstance(session_surface, ConsoleSessionSurface)
        await session_surface.recompose()
        await pilot.pause()

        new = screen.query_one("#console-native-transcript", ConsoleTranscript)
        assert new is not old  # the recompose really did swap the widget
        assert console_transcripts_on_screen(screen) == [new]
        # The unmount hook pruned the corpse from the candidate set (it is
        # only an optimization -- the DOM check above is what makes the
        # result correct -- but an unfired hook is dead code).
        assert old not in _LIVE_TRANSCRIPTS
        assert new in _LIVE_TRANSCRIPTS

        # ...and the dismissal cleans the LIVE transcript, not the corpse.
        row = await _seed_row(pilot, new)
        new.selection_manager.begin_drag(row.id, 0)
        new.selection_manager.extend_drag(row.id, 4)
        row.set_selection_range(0, 4)
        new._selection_origin_row = row

        await pilot.click("#console-native-composer")
        await pilot.pause()

        assert new.selection_manager.state.selection is None
        assert row.get_selection_text() == ""


@pytest.mark.asyncio
async def test_the_registries_are_scoped_to_one_screen_not_the_whole_app():
    """The screen-scope arm (review round 1, MINOR-1).

    ``widget.screen is screen`` is what makes the registries equivalent to
    the ``screen.query(...)`` calls they replaced rather than "any attached
    widget anywhere in the app". Relaxing it to ``is not None`` passes every
    other test in the selection suites -- a modal pushed above the Console
    is the only shape that separates the two -- so this arm exists to kill
    that mutant: the Console must not see (or dismiss) a menu living on the
    modal above it, and the modal must not see the Console's transcript.
    """
    async with make_console_pilot() as pilot:
        console = pilot.app.screen
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        modal = _BareModal()
        await pilot.app.push_screen(modal)
        await pilot.pause()
        menu = ConsoleSelectionMenu(screen_x=2, screen_y=2)
        await modal.mount(menu)
        await pilot.pause()

        assert selection_menus_on_screen(modal) == [menu]
        assert selection_menus_on_screen(console) == []
        assert console_transcripts_on_screen(console) == [transcript]
        assert console_transcripts_on_screen(modal) == []

        # ...and the behavioural form: the Console's dismissal pass leaves a
        # foreign screen's menu mounted, exactly as the old query did.
        console._dismiss_console_selection_menus_outside_transcript(composer)
        await pilot.pause()
        assert menu.is_attached


@pytest.mark.asyncio
async def test_press_inside_the_transcript_still_leaves_the_menu_alone():
    """Control arm: the in-transcript guard is unchanged.

    The transcript (and the menu) own their in-area interaction; the screen
    handler must keep its hands off a press whose ancestor chain contains a
    transcript, gate or no gate -- and it must reach that conclusion without
    scanning either registry (review round 1, MINOR-2: the ancestor walk
    runs first, so the drag hot path costs what it did on base: nothing).
    """
    async with make_console_pilot() as pilot:
        screen = pilot.app.screen
        transcript = screen.query_one("#console-native-transcript", ConsoleTranscript)
        await screen.mount(ConsoleSelectionMenu(screen_x=2, screen_y=2, owner=transcript))
        await pilot.pause()
        menu = screen.query_one(ConsoleSelectionMenu)

        scans = {"menus": 0, "transcripts": 0}
        import tldw_chatbook.UI.Screens.chat_screen as chat_screen_module

        real_menus = chat_screen_module.selection_menus_on_screen
        real_transcripts = chat_screen_module.console_transcripts_on_screen

        def counting_menus(target):
            scans["menus"] += 1
            return real_menus(target)

        def counting_transcripts(target):
            scans["transcripts"] += 1
            return real_transcripts(target)

        chat_screen_module.selection_menus_on_screen = counting_menus
        chat_screen_module.console_transcripts_on_screen = counting_transcripts
        try:
            screen._dismiss_console_selection_menus_outside_transcript(transcript)
            await pilot.pause()
        finally:
            chat_screen_module.selection_menus_on_screen = real_menus
            chat_screen_module.console_transcripts_on_screen = real_transcripts

        assert menu.is_attached
        assert scans == {"menus": 0, "transcripts": 0}
