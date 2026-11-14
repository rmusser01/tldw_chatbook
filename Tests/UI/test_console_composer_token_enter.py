"""Enter and collapsed draft tokens in the Console composer (TASK-33622.1).

A paste over the collapse threshold, an inserted text file and a ``/prompt``
insert sit in the draft as one collapsed token. Before TASK-33622.1 Enter
never sent such a draft: each Enter stepped the FIRST token in the draft from
collapsed to a bare "Expand?" and then spliced it open, so one paste plus a
typed question took three Enters (live on dev 38c9d2edcf, 160x45).

Two caret rules followed (owner, 2026-10-27 and 2026-10-29): Enter expanded
a token when the caret was on it, later only right before it. Each left a
flow that still took three Enters or that only the mouse could expand: a
question typed in front of a paste (TASK-37871), a paste that starts with a
line break (TASK-37870).

Owner decision 2026-10-30: ENTER ALWAYS SENDS. Enter never expands a token,
wherever the caret is and whatever state the token is in. A dedicated key
(Ctrl+E, ``test_console_composer_token_expand_key.py``) expands one, and a
click still does. These tests drive the real ChatScreen and composer with
real keys, and every flow sends on ONE Enter with the full text.
"""

from __future__ import annotations

import pytest
from textual.events import Paste

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
PROMPT_NAME = "Triage"
PROMPT_BODY = "Triage this incident: list the failing worker, the cause and a fix. " * 2
QUESTION = "Summarize this log: "


def _assert_collapsed(composer, body: str) -> None:
    """A collapsed token stands for ``body``; none of its text is shown.

    Copy-agnostic on purpose: the chip and question copy are pinned in
    ``test_console_composer_token_expand_key.py``, so a red here is always
    the send behaviour, never a hint's wording.
    """
    assert composer.has_paste_segments()
    shown = composer._display_draft_text()
    assert body.strip()[:20] not in shown, shown
    assert "characters" in shown or "📄" in shown, shown


def _type(host, text: str) -> None:
    names = {" ": "space", "?": "question_mark", ":": "colon"}
    for char in text:
        press(host, names.get(char, char), char)


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

            users = await _send_once_and_read_users(
                host, pilot, gateway, console, composer
            )

            assert users == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize("edge", ["trailing", "leading"])
async def test_one_enter_sends_a_paste_whose_edge_line_break_parts_caret_and_chip(edge):
    """AC#1/#2: a paste with a line break at an edge sends on one Enter.

    Text copied as whole lines usually ends in a newline, and a ``/prompt``
    or Library insert after a draft starts with one. The chip draws that
    break outside itself. Review of f93748e746: the trailing case armed
    "Expand pasted text (...)?" instead of sending.
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
            _assert_collapsed(composer, body)
            # The break is drawn outside the chip, on its own row.
            shown = composer._display_draft_text()
            assert (
                shown.endswith("\n") if edge == "trailing" else shown.startswith("\n")
            )

            users = await _send_once_and_read_users(
                host, pilot, gateway, console, composer
            )

            assert users == [body]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flow",
    [
        "paste",
        "question_then_paste",
        "paste_then_question",
        "question_in_front_of_paste",
        "prompt_insert",
        "file_insert",
        "leading_line_break_paste",
        "caret_before_a_trailing_token",
        "caret_after_a_mid_draft_token",
    ],
)
async def test_one_enter_sends_wherever_the_caret_is(flow, tmp_path, monkeypatch):
    """Owner decision 2026-10-30: Enter always sends, one Enter, full text.

    Each flow leaves a collapsed token in the draft with the caret right
    after it, right before it, or away from it, and one Enter sends the
    whole draft. Red before the decision: ``question_in_front_of_paste``
    (TASK-37871: paste, Home, type a question) and
    ``caret_before_a_trailing_token`` showed the expand question instead of
    sending; dev stepped the first token to a bare "Expand?" in every flow.
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
            caret = None  # None: the caret ends the draft
            if flow == "paste":
                expected = LOG
                host.post_message(Paste(LOG))
            elif flow == "question_then_paste":
                expected = QUESTION + LOG
                _type(host, QUESTION)
                await until(lambda: composer.draft_text() == QUESTION)
                host.post_message(Paste(LOG))
            elif flow == "paste_then_question":
                expected = LOG + " why"
                host.post_message(Paste(LOG))
                await until(lambda: composer.draft_text() == LOG)
                _type(host, " why")
            elif flow == "question_in_front_of_paste":
                expected = "Q: " + LOG
                host.post_message(Paste(LOG))
                await until(lambda: composer.draft_text() == LOG)
                press(host, "home")
                await until(lambda: composer.cursor_index == 0)
                _type(host, "Q: ")
                caret = len("Q: ")  # right before the token
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
            elif flow == "leading_line_break_paste":
                # TASK-37870: the break is drawn outside the chip, so no caret
                # position ever sat right before it.
                expected = QUESTION + "\n" + LOG
                _type(host, QUESTION)
                await until(lambda: composer.draft_text() == QUESTION)
                host.post_message(Paste("\n" + LOG))
                await until(lambda: composer.draft_text() == expected)
                press(host, "left")
                caret = len(QUESTION)  # right before the token
            elif flow == "caret_before_a_trailing_token":
                expected = QUESTION + LOG
                _type(host, QUESTION)
                await until(lambda: composer.draft_text() == QUESTION)
                host.post_message(Paste(LOG))
                await until(lambda: composer.draft_text() == expected)
                press(host, "left")
                caret = len(QUESTION)
            else:
                expected = LOG + " why"
                host.post_message(Paste(LOG))
                await until(lambda: composer.draft_text() == LOG)
                _type(host, " why")
                await until(lambda: composer.draft_text() == expected)
                for _ in " why":
                    press(host, "left")
                caret = len(LOG)  # right after the token
            await until(lambda: composer.draft_text() == expected)
            want = len(expected) if caret is None else caret
            await until(lambda: composer.cursor_index == want)
            await pilot.pause(0.2)
            # One collapsed token stands for the body; it is not armed.
            _assert_collapsed(composer, {"prompt_insert": PROMPT_BODY}.get(flow, LOG))
            assert not composer.has_pending_paste_confirmation()
            assert gateway.stream_calls == 0

            users = await _send_once_and_read_users(
                host, pilot, gateway, console, composer
            )

            assert users == [expected]


@pytest.mark.asyncio
async def test_enter_sends_while_a_clicked_token_asks_to_expand():
    """Owner decision 2026-10-30: a pending expand question never owns Enter.

    A click asks "Expand pasted text (N characters)? Ctrl+E: yes · Esc: no";
    Enter still sends the whole draft, the token in full. Before the
    decision that Enter answered yes and spliced the paste open, sending
    nothing.
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

            await pilot.click("#console-command-visible-text", offset=(4, 0))
            await until(composer.has_pending_paste_confirmation)
            await pilot.pause()
            # The question replaces the chip; the paste's text is still hidden.
            _assert_collapsed(composer, LOG)
            assert composer._display_draft_text().endswith(" why")
            assert gateway.stream_calls == 0

            users = await _send_once_and_read_users(
                host, pilot, gateway, console, composer
            )

            assert users == [expected]
