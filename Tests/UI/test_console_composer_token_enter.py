"""Enter and collapsed draft tokens in the Console composer (TASK-33622.1).

A paste over the collapse threshold, an inserted text file and a ``/prompt``
insert sit in the draft as one collapsed token. Before this task Enter never
sent such a draft: each Enter stepped the FIRST token in the draft from
collapsed to a bare "Expand?" and then spliced it open, so one paste plus a
typed question took three Enters (live on dev 38c9d2edcf, 160x45).

The owner decided on 2026-10-27 that Enter expands a token only when the
caret is on it; otherwise Enter sends. The caret never sits inside a token
(it snaps to the token's edges). The first reading counted either edge, so a
draft that ENDED in a token still took three Enters. The owner ruled on
2026-10-29 (TASK-37800): paste then Enter sends. Only a caret on a token or
immediately before it (its left edge) expands it; a caret right after a
token does not. These tests drive the real ChatScreen and composer with real
keys.
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
    names = {" ": "space", "?": "question_mark", ":": "colon"}
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

    Left from the end of the draft puts the caret on the second token (its
    left edge). The pre-fix Enter always stepped the first collapsed token.
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
        await pilot.press("left")
        await pilot.pause()
        assert composer.cursor_index == len(first + " and ")

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
        await pilot.press("left")
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
        await pilot.press("left")

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


async def _send_once_and_read_users(host, pilot, gateway, console, composer) -> list:
    """Press Enter once; fail unless exactly one dispatch went out."""
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
    return [
        message.content
        for message in store.messages_for_session(store.active_session_id)
        if message.role.value == "user"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["trailing", "leading"])
async def test_one_enter_sends_a_paste_whose_edge_line_break_parts_caret_and_chip(edge):
    """AC#1/#2: a line break at the paste's edge puts the caret off the chip.

    Text copied as whole lines usually ends in a newline. The chip draws that
    break outside itself, so the caret at the paste's end sits at the start of
    the empty row below the chip, not next to it, and one Enter sends. Review
    of f93748e746: that Enter armed "Expand pasted text (...)?" instead.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("")
            await pilot.pause()
            gateway.validation_release.set()
            body = LOG + "\n" if edge == "trailing" else "\n" + LOG
            host.post_message(Paste(body))
            await until(lambda: composer.draft_text() == body)
            if edge == "leading":
                press(host, "home")
                await until(lambda: composer.cursor_index == 0)
            chip = _paste_chip(body)
            assert composer._display_draft_text() == (
                f"{chip}\n" if edge == "trailing" else f"\n{chip}"
            )

            users = await _send_once_and_read_users(
                host, pilot, gateway, console, composer
            )

            assert users == [body]


@pytest.mark.asyncio
async def test_a_click_on_a_chip_ending_in_a_line_break_still_owns_the_next_enter():
    """AC#3: the click's caret lands after the break, and Enter answers yes."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        pasted = "P" * (composer.paste_collapse_threshold + 1) + "\n"
        composer.insert_pasted_text(pasted)
        composer.insert_text("tail")
        await pilot.pause()
        assert composer._display_draft_text() == f"{_paste_chip(pasted)}\ntail"

        await pilot.click("#console-command-visible-text", offset=(4, 0))
        await pilot.pause()
        assert composer._display_draft_text() == f"{_paste_confirm(pasted)}\ntail"

        await pilot.press("enter")
        await pilot.pause()

        assert composer._display_draft_text() == f"{pasted}tail"
        assert composer.draft_text() == f"{pasted}tail"


@pytest.mark.asyncio
async def test_moving_the_caret_off_a_pending_confirm_answers_no_and_enter_sends():
    """AC#2/#4: the caret leaving a confirm puts the chip back; Enter then sends.

    Right from the token's left edge jumps the token to its right edge, which
    is still on it; the next Right leaves it.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("")
            await pilot.pause()
            gateway.validation_release.set()
            host.post_message(Paste(LOG))
            await until(lambda: composer.draft_text() == LOG)
            _type(host, " why")
            expected = LOG + " why"
            await until(lambda: composer.draft_text() == expected)
            press(host, "home")
            await until(lambda: composer.cursor_index == 0)

            press(host, "enter", "\r")
            await until(composer.has_pending_paste_confirmation)
            press(host, "right")
            await until(lambda: composer.cursor_index == len(LOG))
            await pilot.pause()
            assert composer._display_draft_text() == f"{_paste_confirm(LOG)} why"

            press(host, "right")
            await until(lambda: composer.cursor_index == len(LOG) + 1)
            await pilot.pause()
            assert composer._display_draft_text() == f"{_paste_chip(LOG)} why"
            assert not composer.has_pending_paste_confirmation()
            assert gateway.stream_calls == 0

            users = await _send_once_and_read_users(
                host, pilot, gateway, console, composer
            )

            assert users == [expected]


PROMPT_NAME = "Triage"
PROMPT_BODY = "Triage this incident: list the failing worker, the cause and a fix. " * 2
QUESTION = "Summarize this log: "


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flow",
    ["paste", "question_then_paste", "prompt_insert", "file_insert", "after_mid_token"],
)
async def test_one_enter_sends_with_the_caret_right_after_a_token(
    flow, tmp_path, monkeypatch
):
    """TASK-37800 (owner ruling 2026-10-29): paste then Enter sends.

    A caret right after a collapsed token is not on it. The four flows the
    ruling names leave the caret after a token that ends the draft; the
    fifth moves it to just after a token mid-draft. Each sends on one Enter,
    carrying the full text. The pre-change build armed the expand confirm
    instead (and dev stepped the token to a bare "Expand?").
    """
    host, gateway, _timeline = build()
    if flow == "prompt_insert":
        from Tests.UI.test_console_command_composer import (
            _real_prompt_scope_service,
        )

        db, service = _real_prompt_scope_service(tmp_path)
        db.add_prompt(
            name=PROMPT_NAME,
            author="",
            details="",
            system_prompt="",
            user_prompt=PROMPT_BODY,
            keywords=[],
        )
        host.app_instance.prompt_scope_service = service
    if flow == "file_insert":
        import tldw_chatbook.Chat.attachment_core as attachment_core

        original = attachment_core.load_processed_file

        async def _rooted(file_path, *, allowed_root=None):
            # The file lives in tmp_path, outside $HOME: widen the safety root.
            return await original(file_path, allowed_root=str(tmp_path))

        monkeypatch.setattr(attachment_core, "load_processed_file", _rooted)
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("")
            await pilot.pause()
            gateway.validation_release.set()
            if flow == "paste":
                expected = LOG
                host.post_message(Paste(LOG))
            elif flow == "question_then_paste":
                expected = QUESTION + LOG
                _type(host, QUESTION)
                await until(lambda: composer.draft_text() == QUESTION)
                host.post_message(Paste(LOG))
            elif flow == "prompt_insert":
                expected = PROMPT_BODY
                assert len(PROMPT_BODY) > composer.paste_collapse_threshold
                composer.load_draft(f"/prompt {PROMPT_NAME}")
                await pilot.pause()
                # The command's own Enter: it replaces the draft with the body.
                press(host, "enter", "\r")
            elif flow == "file_insert":
                path = tmp_path / "photosynthesis.md"
                path.write_text(FILE_TEXT)
                # The real "insert a text file" path: the file's handler
                # shapes the inlined text, so read back what it inserted.
                await console._process_console_attachment(str(path))
                await until(
                    lambda: "photosynthesis.md" in composer._display_draft_text()
                )
                expected = composer.draft_text()
                assert FILE_TEXT.strip() in expected
            else:
                expected = LOG + " why"
                host.post_message(Paste(LOG))
                await until(lambda: composer.draft_text() == LOG)
                _type(host, " why")
                await until(lambda: composer.draft_text() == expected)
                for _ in " why":
                    press(host, "left")
                await until(lambda: composer.cursor_index == len(LOG))
            await until(lambda: composer.draft_text() == expected)
            await pilot.pause(0.2)
            if flow != "after_mid_token":
                assert composer.cursor_index == len(expected)
            # One collapsed token stands for the body; it is not armed.
            assert composer.has_paste_segments()
            assert "| Expand" in composer._display_draft_text()
            assert not composer.has_pending_paste_confirmation()
            assert gateway.stream_calls == 0

            users = await _send_once_and_read_users(
                host, pilot, gateway, console, composer
            )

            assert users == [expected]


@pytest.mark.asyncio
async def test_left_onto_a_trailing_token_then_enter_still_expands_it():
    """TASK-37800: Left from the end reaches a trailing token to expand it.

    The ruling keeps the token's left edge as "on" it: Enter asks, a second
    Enter puts the full text in place, and neither sends. Once expanded, the
    next Enter sends the same full text.
    """
    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, composer = await ready_console(host, pilot, gateway)
            composer.load_draft("")
            await pilot.pause()
            gateway.validation_release.set()
            _type(host, QUESTION)
            await until(lambda: composer.draft_text() == QUESTION)
            host.post_message(Paste(LOG))
            expected = QUESTION + LOG
            await until(lambda: composer.draft_text() == expected)
            press(host, "left")
            await until(lambda: composer.cursor_index == len(QUESTION))

            press(host, "enter", "\r")
            await until(composer.has_pending_paste_confirmation)
            await pilot.pause()
            assert composer._display_draft_text() == QUESTION + _paste_confirm(LOG)

            press(host, "enter", "\r")
            await until(lambda: composer._display_draft_text() == expected)
            await pilot.pause(0.3)
            assert gateway.stream_calls == 0
            assert composer.draft_text() == expected

            press(host, "end")
            await until(lambda: composer.cursor_index == len(expected))
            users = await _send_once_and_read_users(
                host, pilot, gateway, console, composer
            )

            assert users == [expected]
