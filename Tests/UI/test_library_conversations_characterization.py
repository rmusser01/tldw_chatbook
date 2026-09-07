"""Pre-extraction characterization pins for the Library Conversations subsystem.

task-5 of the library-decomposition-foundation plan: Tasks 6-8 move the
~68 ``LibraryScreen`` conversation methods out into a
``LibraryConversationsState`` dataclass plus two controllers (see
``backlog/docs/library-decomposition-recipe.md``). Before that move starts,
this file pins the CURRENT behavior of every conversations ``@on`` handler
that a plain ``grep -rl <method-name> Tests/`` (the brief's Step-1
enumeration script) reported as uncovered -- but only the ones that really
are unreached through the DOM after a closer look.

That closer look matters: the enumeration script is a name-grep, not a
behavior check, so several of its "UNCOVERED" hits (``show_library_
conversation_reader_info``, ``find_in_library_conversation``, ``library_
conversation_reader_messages_synced``, the select-toggle/-clear/-next
handlers, ``open_selected_conversation_in_console``) are already exercised
through real ``Button.Pressed``/``Input.Submitted`` presses in
``Tests/UI/test_library_conversation_reader.py`` and
``Tests/UI/test_library_multiselect_conversations.py`` -- just never by
literal method name. Verified per-handler via
``grep -rn '"#<button-id>"' Tests/UI/*.py`` plus a manual check for an
actual ``.press()``/Enter-submit call, not merely an id reference (e.g. a
``.disabled`` assertion). This file only adds tests for the handlers that
survive that second check as genuinely unreached:

- ``show_library_conversation_reader_read`` -- the sibling Info button IS
  pressed elsewhere; nothing presses Read to return to it.
- ``handle_library_conversations_select_all`` -- the id is referenced (label
  text, disabled-state) but never actually pressed.
- ``handle_library_conversations_previous`` -- same: referenced only for
  ``.disabled`` assertions, never pressed.
- ``use_selected_conversation_as_source`` -- see its test's docstring: the
  button it is bound to (``#library-conversation-use-source``) has zero
  compose sites anywhere in ``tldw_chatbook/``, a live finding recorded
  here rather than fixed (out of this task's scope).

Every test below drives the screen only through DOM queries/presses and
public screen attributes, per the recipe's byte-for-byte move discipline,
so it keeps working unmodified once Tasks 6-8 relocate the method bodies
into their controllers.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Button, Static

from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_SOURCE_PAGE_SIZES,
)
from Tests.UI.test_destination_shells import _link_library_items_to_active_workspace
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _conversation_records,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


@pytest.mark.asyncio
async def test_reader_read_button_returns_to_read_mode_after_info() -> None:
    """Characterization (pre-extraction): pins show_library_conversation_reader_read.

    The reader defaults to "read" mode, and the sibling Info button is
    pressed by ``test_library_conversation_reader.py``'s info-mode test --
    but that test never presses Read to come back, so the round trip was
    never actually driven through the DOM.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._conversations_state.reader_state.loaded_id == "chat-1"
                and screen._conversations_state.reader_state.complete
            ),
            message="Conversation reader never finished loading chat-1.",
        )
        assert screen._conversations_state.reader_state.mode == "read"

        screen.query_one("#library-conversation-reader-info", Button).press()
        await pilot.pause()
        assert screen._conversations_state.reader_state.mode == "info"

        screen.query_one("#library-conversation-reader-read", Button).press()
        await pilot.pause()

        assert screen._conversations_state.reader_state.mode == "read"
        messages = screen.query_one(
            "#library-conversation-reader-messages", VerticalScroll
        )
        info_body = screen.query_one(
            "#library-conversation-reader-info-body", Static
        )
        read_button = screen.query_one("#library-conversation-reader-read", Button)
        info_button = screen.query_one("#library-conversation-reader-info", Button)
        assert messages.display is True
        assert info_body.display is False
        assert read_button.has_class("-selected")
        assert not info_button.has_class("-selected")


@pytest.mark.asyncio
async def test_select_all_press_selects_every_rendered_row() -> None:
    """Characterization (pre-extraction): pins handle_library_conversations_select_all.

    The button's id is referenced elsewhere in the suite (label text,
    disabled-state assertions) but no existing test actually presses it.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-1")

        screen.query_one("#library-conversations-select-toggle", Button).press()
        await pilot.pause()
        assert screen._conversations_state.select_mode is True
        assert screen._conversations_state.row_selection.count == 0

        screen.query_one("#library-conversations-select-all", Button).press()
        await pilot.pause()

        assert screen._conversations_state.row_selection.count == 2
        assert screen._conversations_state.row_selection.is_selected("chat-1")
        assert screen._conversations_state.row_selection.is_selected("chat-2")
        assert screen._conversations_state.reader_state.bulk_selected_count == 2


@pytest.mark.asyncio
async def test_previous_page_press_retreats_the_conversation_page() -> None:
    """Characterization (pre-extraction): pins handle_library_conversations_previous.

    The button's id is referenced elsewhere only in ``.disabled`` assertions
    (stale-state gating); nothing presses it to actually page backward.
    """
    app = _build_test_app()
    page_size = LIBRARY_SOURCE_PAGE_SIZES["conversations"]
    _seed_conversations(app, _conversation_records(page_size + 2))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(
            screen, pilot, f"#library-conversation-row-{page_size - 1}"
        )
        assert screen._conversations_state.page == 1
        first_row_page_one = screen.query_one(
            "#library-conversation-row-0"
        ).conversation_id

        screen.query_one("#library-conversations-next", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._conversations_state.page == 2,
            message="Conversation page never advanced to 2.",
        )
        await pilot.pause()
        first_row_page_two = screen.query_one(
            "#library-conversation-row-0"
        ).conversation_id
        assert first_row_page_two != first_row_page_one

        screen.query_one("#library-conversations-previous", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._conversations_state.page == 1,
            message="Conversation page never retreated to 1.",
        )
        await pilot.pause()
        assert (
            screen.query_one("#library-conversation-row-0").conversation_id
            == first_row_page_one
        )


@pytest.mark.asyncio
async def test_use_as_source_delegates_identically_to_open_in_console() -> None:
    """Characterization (pre-extraction): pins use_selected_conversation_as_source.

    LIVE FINDING, recorded rather than fixed (out of task-5 scope): a
    repo-wide ``grep -rn "library-conversation-use-source" tldw_chatbook/``
    turns up exactly one hit -- this handler's own ``@on`` decorator line.
    The button it is bound to is never composed anywhere, so it cannot be
    pressed through the DOM/Pilot today; this test calls the screen method
    directly (the closest available "screen surface" invocation) to pin
    what it currently does. Both this handler and its sibling
    ``open_selected_conversation_in_console`` delegate, unconditionally and
    without any per-button distinction, to the same private
    ``_open_selected_conversation_handoff`` -- so today "Use as source"
    reports the action label "Use in Console", byte-for-byte identical to
    the Console button (see ``test_library_shell_open_in_console_triggers_
    handoff`` in ``Tests/UI/test_library_shell.py`` for the sibling's own
    pin).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("conversation", "chat-1", "Quarterly planning sync"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-open-console")
        await _wait_for_condition(
            pilot,
            lambda: screen._conversations_state.reader_state.loaded_actions_eligible,
            message="Selected conversation never became handoff-eligible.",
        )

        screen.use_selected_conversation_as_source(
            SimpleNamespace(stop=lambda: None)
        )
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    kwargs = app.open_chat_with_handoff.call_args.kwargs
    assert payload.source == "library"
    assert payload.item_type == "conversation"
    assert payload.source_id == "chat-1"
    assert kwargs.get("action_label") == "Use in Console"
