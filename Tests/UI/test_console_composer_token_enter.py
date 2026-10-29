"""Enter and collapsed draft tokens in the Console composer (TASK-33622.1).

A paste over the collapse threshold, an inserted text file and a ``/prompt``
insert sit in the draft as one collapsed token. Before this task Enter never
sent such a draft: each Enter stepped the FIRST token in the draft from
collapsed to a bare "Expand?" and then spliced it open, so one paste plus a
typed question took three Enters (live on dev 38c9d2edcf, 160x45).

The owner decided on 2026-10-27 that Enter expands a token only when the
caret is on it; otherwise Enter sends. The caret never sits inside a token
(it snaps to the token's edges), so "on it" means at either edge. These tests
drive the real ChatScreen and composer with real keys.
"""

from __future__ import annotations

import pytest
from textual.events import Paste

from Tests.UI.test_console_composer_keymap import APP_SIZE, _focused_composer
from Tests.UI.test_console_dictation import _ready_host
from Tests.UI.test_console_send_acknowledgement import (
    build,
    eager_tasks,
    press,
    ready_console,
    until,
)

# Mounted Console tests retain the configuration admitted at collection time.
pytestmark = pytest.mark.bootstrap_profile

LOG = "\n".join(f"2026-10-29 worker log line {index}: restarted" for index in range(5))
FILE_TEXT = "Photosynthesis turns light into chemical energy. " * 3
FILE_LABEL = "📄 photosynthesis.md · 1 KB"


def _paste_chip(text: str) -> str:
    return f"Pasted text | {len(text)} characters | Expand"


def _paste_confirm(text: str) -> str:
    return f"Expand pasted text ({len(text)} characters)? Enter: yes · Esc: no"


def _type(host, text: str) -> None:
    names = {" ": "space", "?": "question_mark"}
    for char in text:
        press(host, names.get(char, char), char)


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["paste", "file"])
async def test_one_enter_sends_a_token_and_typed_question_with_the_full_text(token):
    """AC#1/#5: paste (or insert a file), type a question, press Enter once.

    Exactly one provider dispatch, carrying the token's full content. On the
    pre-fix build the Enter armed "Expand?" on the token instead and nothing
    reached the provider.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("")
            await pilot.pause()
            gateway.validation_release.set()
            if token == "paste":
                assert len(LOG) > composer.paste_collapse_threshold
                body = LOG
                # Through the app, as the terminal driver delivers a paste.
                host.post_message(Paste(LOG))
            else:
                body = FILE_TEXT
                composer.insert_file_segment(FILE_TEXT, FILE_LABEL)
            await until(lambda: composer.draft_text() == body)
            # One collapsed token, nothing of its body visible.
            assert composer.has_paste_segments()
            assert body[:20] not in composer._display_draft_text()
            _type(host, " why")
            expected = body + " why"
            await until(lambda: composer.draft_text() == expected)

            press(host, "enter", "\r")
            try:
                await until(lambda: gateway.stream_calls >= 1, timeout=10)
            except AssertionError:
                pytest.fail(
                    "one Enter did not send; the draft shows "
                    f"{composer._display_draft_text()!r}"
                )
            await pilot.pause(0.3)

            assert gateway.stream_calls == 1
            store = console._ensure_console_chat_store()
            users = [
                message.content
                for message in store.messages_for_session(store.active_session_id)
                if message.role.value == "user"
            ]
            assert users == [expected]


@pytest.mark.asyncio
async def test_enter_advances_only_the_token_the_caret_is_on():
    """AC#2: with two tokens, Enter steps the caret's token, never the first.

    The pre-fix Enter always stepped the first collapsed token in the draft.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        first = "A" * (composer.paste_collapse_threshold + 1)
        second = "B" * (composer.paste_collapse_threshold + 2)
        composer.insert_pasted_text(first)
        composer.insert_text(" and ")
        composer.insert_pasted_text(second)
        await pilot.pause()
        draft = first + " and " + second
        assert composer.draft_text() == draft
        assert composer.cursor_index == len(draft)

        await pilot.press("enter")
        await pilot.pause()

        assert composer._display_draft_text() == (
            f"{_paste_chip(first)} and {_paste_confirm(second)}"
        )
        assert composer.draft_text() == draft

        await pilot.press("enter")
        await pilot.pause()

        assert composer._display_draft_text() == f"{_paste_chip(first)} and {second}"
        assert composer.draft_text() == draft


@pytest.mark.asyncio
async def test_enter_with_the_caret_before_a_token_arms_that_token():
    """AC#2/#3: the caret at a token's left edge is on that token too."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "question: ")
        pasted = "P" * (composer.paste_collapse_threshold + 1)
        composer.move_cursor_home()
        composer.insert_pasted_text(pasted)
        composer.move_cursor_home()
        await pilot.pause()
        assert composer.cursor_index == 0

        await pilot.press("enter")
        await pilot.pause()

        assert composer._display_draft_text() == (f"{_paste_confirm(pasted)}question: ")


@pytest.mark.asyncio
async def test_escape_returns_a_pending_confirm_to_the_collapsed_chip():
    """AC#4: Esc answers "no" to a pending expand confirm."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        pasted = "P" * (composer.paste_collapse_threshold + 1)
        composer.insert_pasted_text(pasted)
        await pilot.press("enter")
        await pilot.pause()
        assert composer._display_draft_text() == _paste_confirm(pasted)

        await pilot.press("escape")
        await pilot.pause()

        assert composer._display_draft_text() == _paste_chip(pasted)
        assert not composer.has_pending_paste_confirmation()
        assert composer.draft_text() == pasted
        assert host.focused is composer


@pytest.mark.asyncio
async def test_a_file_token_keeps_its_name_through_the_confirm():
    """AC#3/#4: the chip ends in Expand and the confirm names the file."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        composer.insert_file_segment(FILE_TEXT, FILE_LABEL)
        await pilot.pause()
        assert composer._display_draft_text() == f"{FILE_LABEL} | Expand"

        await pilot.press("enter")
        await pilot.pause()

        assert composer._display_draft_text() == (
            f"Expand {FILE_LABEL}? Enter: yes · Esc: no"
        )
        assert composer.draft_text() == FILE_TEXT


@pytest.mark.asyncio
async def test_clicking_a_token_puts_the_caret_on_it_so_enter_answers_the_confirm():
    """AC#3: a click arms the token and moves the caret onto it.

    Without the caret move, the Enter after a click on a token far from the
    caret would send instead of answering the confirm the chip shows.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        pasted = "P" * (composer.paste_collapse_threshold + 1)
        composer.insert_pasted_text(pasted)
        composer.insert_text(" and a tail")
        await pilot.pause()
        assert composer.cursor_index == len(pasted) + len(" and a tail")

        await pilot.click("#console-command-visible-text", offset=(4, 0))
        await pilot.pause()

        assert composer._display_draft_text() == (
            f"{_paste_confirm(pasted)} and a tail"
        )
        assert composer.cursor_index == len(pasted)

        await pilot.press("enter")
        await pilot.pause()

        assert composer._display_draft_text() == f"{pasted} and a tail"
