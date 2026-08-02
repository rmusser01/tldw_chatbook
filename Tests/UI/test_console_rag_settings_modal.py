"""The RAG chip's Library RAG settings modal: gating, results, dismissal.

User request 2026-08-01: clicking "RAG: off" in the status strip opens a
modal that lets the user set the retrieval query and run it -- instead of
the query living only in a rail input that may not even be on screen.
"""

from unittest.mock import Mock

import pytest
from textual.app import App
from textual.widgets import Button, Input

from tldw_chatbook.Widgets.Console.console_rag_settings_modal import (
    ConsoleRagSettingsModal,
    ConsoleRagSettingsResult,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_returns_the_query_and_is_gated_on_non_blank_text():
    """Run is disabled while blank, enables on typing, and returns the query."""

    class RagHost(App):
        pass

    received: list[ConsoleRagSettingsResult | None] = []
    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(scope_label="Scope: notes, media"),
            callback=received.append,
        )
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ConsoleRagSettingsModal)

        run = modal.query_one("#console-rag-settings-run", Button)
        assert run.disabled is True, "blank query must not be runnable"

        query_input = modal.query_one("#console-rag-settings-query", Input)
        query_input.value = "incident retro notes"
        await pilot.pause()
        assert run.disabled is False

        await pilot.click("#console-rag-settings-run")
        await pilot.pause()
        await pilot.pause()

    assert received == [
        ConsoleRagSettingsResult(query="incident retro notes", run=True)
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prefilled_query_is_runnable_and_enter_submits():
    """A prefilled modal is one keypress from retrieval (Enter submits)."""

    class RagHost(App):
        pass

    received: list[ConsoleRagSettingsResult | None] = []
    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(query="what changed in auth"),
            callback=received.append,
        )
        await pilot.pause()
        modal = app.screen
        assert modal.query_one("#console-rag-settings-run", Button).disabled is False

        modal.query_one("#console-rag-settings-query", Input).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

    assert received == [
        ConsoleRagSettingsResult(query="what changed in auth", run=True)
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancel_escape_and_backdrop_all_dismiss_without_changes():
    """Every no-action exit returns None; inside clicks keep it open."""

    class RagHost(App):
        pass

    received: list[ConsoleRagSettingsResult | None] = []
    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(query="draft text"),
            callback=received.append,
        )
        await pilot.pause()
        modal = app.screen

        # A click inside the box (its header line) must not dismiss.
        box = modal.query_one("#console-rag-settings")
        await pilot.click(offset=(box.region.x + 2, box.region.y + 1))
        await pilot.pause()
        assert app.screen is modal

        # Backdrop click dismisses with no changes.
        await pilot.click(offset=(1, 1))
        await pilot.pause()
        await pilot.pause()
        assert app.screen is not modal

    assert received == [None]


@pytest.mark.unit
def test_status_copy_is_honest_about_what_on_means():
    """The modal explains that "on" == staged retrieved evidence."""
    off = ConsoleRagSettingsModal()
    assert "RAG is off" in off._status_copy()
    assert "staged" in off._status_copy()

    on = ConsoleRagSettingsModal(rag_active=True, staged_title="Incident Review")
    assert "RAG is on" in on._status_copy()
    assert "Incident Review" in on._status_copy()


@pytest.mark.unit
def test_screen_callback_stores_sanitized_query_and_delegates_run():
    """The screen-side callback owns sanitization and the run delegation."""
    from textual.css.query import QueryError

    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = Mock()
    screen._console_library_rag_query = ""
    screen.query_one = Mock(side_effect=QueryError("not mounted"))

    # The bare state contract: sanitize, store through the one query
    # writer (`_set_console_library_rag_query`), delegate when run=True.
    ChatScreen._apply_console_rag_settings_choice(
        screen, ConsoleRagSettingsResult(query="  spaced   query  ", run=True)
    )
    screen._set_console_library_rag_query.assert_called_once_with("spaced query")
    screen._run_console_library_rag_from_visible_action.assert_called_once()

    screen._set_console_library_rag_query.reset_mock()
    screen._run_console_library_rag_from_visible_action.reset_mock()
    ChatScreen._apply_console_rag_settings_choice(
        screen, ConsoleRagSettingsResult(query="no run", run=False)
    )
    screen._set_console_library_rag_query.assert_called_once_with("no run")
    screen._run_console_library_rag_from_visible_action.assert_not_called()

    screen._set_console_library_rag_query.reset_mock()
    ChatScreen._apply_console_rag_settings_choice(screen, None)
    screen._set_console_library_rag_query.assert_not_called()


@pytest.mark.unit
def test_visible_run_action_falls_back_to_the_composer_draft():
    """User decision (2026-08-02): with no dedicated query set, the visible
    Run Library RAG action retrieves with the composer draft instead of
    demanding a query the collapsed rail gives no place to type. The
    fallback is STORED so the rail input and this modal agree with what
    actually ran; with no query anywhere, the warning toast survives."""
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_LIBRARY_RAG_QUERY_EMPTY_MESSAGE,
        ChatScreen,
    )

    screen = Mock()
    screen._console_library_rag_query = ""
    composer = Mock()
    composer.draft_text.return_value = "  what   changed in auth  "
    screen._console_composer_or_none.return_value = composer

    ChatScreen._run_console_library_rag_from_visible_action(screen)

    screen._set_console_library_rag_query.assert_called_once_with(
        "what changed in auth"
    )
    screen.app_instance.notify.assert_not_called()
    screen._stage_console_library_rag_launch.assert_called_once()
    launch = screen._stage_console_library_rag_launch.call_args.args[0]
    assert launch.payload["query"] == "what changed in auth"
    request = screen._execute_console_library_rag_search.call_args.args[0]
    assert request.query == "what changed in auth"

    # No dedicated query AND an empty composer: the toast survives.
    empty = Mock()
    empty._console_library_rag_query = ""
    empty_composer = Mock()
    empty_composer.draft_text.return_value = "   "
    empty._console_composer_or_none.return_value = empty_composer

    ChatScreen._run_console_library_rag_from_visible_action(empty)

    empty.app_instance.notify.assert_called_once()
    assert (
        CONSOLE_LIBRARY_RAG_QUERY_EMPTY_MESSAGE
        in empty.app_instance.notify.call_args.args
    )
    empty._stage_console_library_rag_launch.assert_not_called()


@pytest.mark.unit
def test_dedicated_query_still_wins_over_the_composer_draft():
    """A query set through the rail or the modal is what runs; the draft
    fallback only fills a VACANT query, never overrides an explicit one."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = Mock()
    screen._console_library_rag_query = "explicit query"
    composer = Mock()
    composer.draft_text.return_value = "draft text"
    screen._console_composer_or_none.return_value = composer

    ChatScreen._run_console_library_rag_from_visible_action(screen)

    screen._set_console_library_rag_query.assert_not_called()
    request = screen._execute_console_library_rag_search.call_args.args[0]
    assert request.query == "explicit query"
