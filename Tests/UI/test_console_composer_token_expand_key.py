"""The expand key for collapsed draft tokens (owner decision 2026-10-30).

Enter always sends (``test_console_composer_token_enter.py``). A collapsed
paste, an inserted text file or a ``/prompt`` insert is expanded by a
dedicated key instead, Ctrl+E, which every chip names in its own hint
("Pasted text | N characters | Ctrl+E expand"). It expands the token next to
the caret, from either side of it, and a click still expands a token.

Ctrl+E was chosen against the Console's key map and ADR-031: bare letters
type into the draft; Ctrl+O, Ctrl+G, Ctrl+K, Ctrl+T, Ctrl+Y and Ctrl+Z are
Console actions; Ctrl+C/V/X/S/D/A/R/W are reserved or terminal-convention
keys; Alt chords type characters on macOS terminals; Ctrl+Shift+<letter>
reaches the app as plain Ctrl+<letter> without the Kitty protocol. Ctrl+E is
bound nowhere in the Console, its "E" is the chip's "expand", and terminals
and tmux send it as the C0 byte 0x05. An editor's built-in terminal can keep
it for itself (VS Code on Windows and Linux binds it to Quick Open); a click
still expands a token there.

Before this decision a paste that starts with a line break could be
expanded only by a click (TASK-37870): its break is drawn outside the chip,
so no caret position counted as on it.
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

LOG = "\n".join(f"2026-10-30 worker log line {index}: restarted" for index in range(5))
FILE_TEXT = "Photosynthesis turns light into chemical energy. " * 3
FILE_LABEL = "📄 photosynthesis.md · 1 KB"
PROMPT_NAME = "Triage"
PROMPT_BODY = "Triage this incident: list the failing worker, the cause and a fix. " * 2
QUESTION = "Summarize this log: "
TAIL = " tail"


def _paste_chip(text: str) -> str:
    return f"Pasted text | {len(text)} characters | Ctrl+E expand"


def _paste_confirm(text: str) -> str:
    return f"Expand pasted text ({len(text)} characters)? Ctrl+E: yes · Esc: no"


def _type(host, text: str) -> None:
    names = {" ": "space", "?": "question_mark", ":": "colon"}
    for char in text:
        press(host, names.get(char, char), char)


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("kind", ["paste", "file", "prompt", "leading_break_paste"])
async def test_the_expand_key_expands_each_kind_of_token_from_either_side(
    kind, side, tmp_path, monkeypatch
):
    """Ctrl+E with the caret right before or right after a token expands it.

    Every kind of token -- a paste, a file inserted through the real
    attachment path, a real ``/prompt`` insert, and a paste that starts with
    a line break (TASK-37870) -- shows its full text in place after ONE
    press. The draft text and the caret do not move, and nothing sends.
    Before the decision Ctrl+E did nothing at all.
    """
    host, gateway, _timeline = build()
    if kind == "prompt":
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
    if kind == "file":
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
            prefix = QUESTION
            if kind == "prompt":
                prefix = ""
                assert len(PROMPT_BODY) > composer.paste_collapse_threshold
                composer.load_draft(f"/prompt {PROMPT_NAME}")
                await pilot.pause()
                # The command's own Enter: it replaces the draft with the body.
                press(host, "enter", "\r")
                await until(lambda: composer.draft_text() == PROMPT_BODY)
            else:
                _type(host, QUESTION)
                await until(lambda: composer.draft_text() == QUESTION)
                if kind == "paste":
                    host.post_message(Paste(LOG))
                    await until(lambda: composer.draft_text() == QUESTION + LOG)
                elif kind == "leading_break_paste":
                    host.post_message(Paste("\n" + LOG))
                    await until(lambda: composer.draft_text() == QUESTION + "\n" + LOG)
                else:
                    path = tmp_path / "photosynthesis.md"
                    path.write_text(FILE_TEXT)
                    await console._process_console_attachment(str(path))
                    await until(
                        lambda: "photosynthesis.md" in composer._display_draft_text()
                    )
            body = composer.draft_text()[len(prefix) :]
            assert body.strip()
            _type(host, TAIL)
            draft = prefix + body + TAIL
            await until(lambda: composer.draft_text() == draft)
            # Left over the tail lands right after the token; one more Left
            # jumps the token to its left edge.
            presses = len(TAIL) + (1 if side == "before" else 0)
            for _ in range(presses):
                press(host, "left")
            caret = len(prefix) + (0 if side == "before" else len(body))
            await until(lambda: composer.cursor_index == caret)
            await pilot.pause(0.2)
            shown = composer._display_draft_text()
            # Collapsed (copy-agnostic: the hint is pinned on its own below,
            # so a red here is the key's behaviour, not the chip's wording).
            assert shown.startswith(prefix) and shown.endswith(TAIL)
            assert body.strip()[:30] not in shown

            press(host, "ctrl+e")
            try:
                await until(lambda: composer._display_draft_text() == draft, timeout=5)
            except AssertionError:
                pytest.fail(
                    f"Ctrl+E did not expand the token; the draft shows "
                    f"{composer._display_draft_text()!r}"
                )
            await pilot.pause(0.2)

            assert composer.draft_text() == draft
            assert composer.cursor_index == caret
            assert not composer.has_pending_paste_confirmation()
            assert gateway.stream_calls == 0
            assert host.focused is composer


@pytest.mark.asyncio
async def test_the_expand_key_takes_the_token_under_the_caret_then_the_one_before():
    """Ctrl+E expands the token right after the caret, else the one before it.

    Two pastes in a row are parted by the line break the composer adds
    between them. Left from the end puts the caret right before the second
    paste: Ctrl+E expands it, not the first. Left again puts the caret right
    after the first paste, and Ctrl+E expands that one.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        first = "A" * (composer.paste_collapse_threshold + 1)
        second = "B" * (composer.paste_collapse_threshold + 2)
        composer.insert_pasted_text(first)
        composer.insert_pasted_text(second)
        await pilot.pause()
        draft = composer.draft_text()
        assert draft.startswith(first) and draft.endswith(second)
        await pilot.press("left")
        await pilot.pause()
        assert composer.cursor_index == len(draft) - len(second)
        shown = composer._display_draft_text()
        assert first not in shown and second not in shown

        await pilot.press("ctrl+e")
        await pilot.pause()

        shown = composer._display_draft_text()
        assert second in shown and first not in shown, shown

        await pilot.press("left")
        await pilot.pause()
        assert composer.cursor_index == len(first)

        await pilot.press("ctrl+e")
        await pilot.pause()

        assert composer._display_draft_text() == draft
        assert composer.draft_text() == draft
        assert composer.cursor_index == len(first)


@pytest.mark.asyncio
async def test_the_expand_key_expands_the_nearest_token_when_none_touches_the_caret():
    """Paste, type a question, Ctrl+E: the chip's hint still holds.

    With no token next to the caret, Ctrl+E expands the collapsed token
    nearest to it, so the hint every chip shows works after typing too.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        first = "A" * (composer.paste_collapse_threshold + 1)
        second = "B" * (composer.paste_collapse_threshold + 2)
        composer.insert_pasted_text(first)
        composer.insert_text(" and a longer middle ")
        composer.insert_pasted_text(second)
        composer.insert_text(" why")
        await pilot.pause()
        draft = first + " and a longer middle " + second + " why"
        assert composer.cursor_index == len(draft)

        await pilot.press("ctrl+e")
        await pilot.pause()

        shown = composer._display_draft_text()
        assert second in shown and first not in shown, shown
        assert shown.endswith(f" and a longer middle {second} why")
        assert composer.draft_text() == draft
        assert composer.cursor_index == len(draft)


@pytest.mark.asyncio
@pytest.mark.parametrize("expand", ["expand_key", "second_click"])
async def test_undo_right_after_an_expand_collapses_the_token_and_keeps_the_typing(
    expand,
):
    """Ctrl+Z after an expand puts the chip back; what was typed stays.

    Readline users press Ctrl+E for end-of-line, and with no token beside the
    caret it expands the nearest one, so an accidental expand must be cheap to
    take back. An expand is its own undo step: one Ctrl+Z collapses the token
    again and keeps " why"; redo expands it again. Before this the expand
    recorded nothing, and Ctrl+Z reverted the typed run before it instead.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        pasted = "P" * (composer.paste_collapse_threshold + 1)
        composer.insert_pasted_text(pasted)
        await pilot.pause()
        await pilot.press("space", "w", "h", "y")
        await pilot.pause()
        draft = pasted + " why"
        assert composer.draft_text() == draft
        collapsed = composer._display_draft_text()
        assert collapsed.endswith(" why") and pasted not in collapsed

        if expand == "expand_key":
            await pilot.press("ctrl+e")
        else:
            await pilot.click("#console-command-visible-text", offset=(4, 0))
            await pilot.pause()
            assert composer.has_pending_paste_confirmation()
            await pilot.click("#console-command-visible-text", offset=(4, 0))
        await pilot.pause()
        assert composer._display_draft_text() == draft

        await pilot.press("ctrl+z")
        await pilot.pause()

        assert composer.draft_text() == draft, "undo lost the typed run"
        assert composer._display_draft_text() == collapsed
        assert not composer.has_pending_paste_confirmation()

        await pilot.press("ctrl+shift+z")
        await pilot.pause()
        assert composer._display_draft_text() == draft

        await pilot.press("ctrl+z", "ctrl+z")
        await pilot.pause()
        assert composer.draft_text() == pasted
        assert pasted not in composer._display_draft_text()


@pytest.mark.asyncio
async def test_the_expand_key_in_a_draft_without_tokens_changes_nothing():
    """Ctrl+E is not a printable key: with no token it leaves the draft alone."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot, "hello there")
        composer.move_cursor_home()
        await pilot.pause()

        await pilot.press("ctrl+e")
        await pilot.pause()

        assert composer.draft_text() == "hello there"
        assert composer.cursor_index == 0
        assert host.focused is composer


@pytest.mark.asyncio
async def test_every_collapsed_chip_names_the_expand_key():
    """A paste chip and a file chip both end in "Ctrl+E expand"."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        pasted = "P" * (composer.paste_collapse_threshold + 1)
        composer.insert_pasted_text(pasted)
        composer.insert_text(" and ")
        composer.insert_file_segment(FILE_TEXT, FILE_LABEL)
        await pilot.pause()

        assert composer._display_draft_text() == (
            f"{_paste_chip(pasted)} and {FILE_LABEL} | Ctrl+E expand"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["second_click", "expand_key"])
async def test_a_click_still_expands_a_token(answer):
    """A click asks to expand; a second click or Ctrl+E answers yes.

    The question names the token and the keys that answer it. Before the
    decision it named Enter, which now always sends.
    """
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        pasted = "P" * (composer.paste_collapse_threshold + 1)
        composer.insert_pasted_text(pasted)
        composer.insert_text(" and a tail")
        await pilot.pause()

        await pilot.click("#console-command-visible-text", offset=(4, 0))
        await pilot.pause()

        assert composer._display_draft_text() == (
            f"{_paste_confirm(pasted)} and a tail"
        )
        # The click also puts the caret beside the token it asks about.
        assert composer.cursor_index == len(pasted)

        if answer == "second_click":
            await pilot.click("#console-command-visible-text", offset=(4, 0))
        else:
            await pilot.press("ctrl+e")
        await pilot.pause()

        assert composer._display_draft_text() == f"{pasted} and a tail"
        assert composer.draft_text() == f"{pasted} and a tail"


@pytest.mark.asyncio
async def test_escape_returns_a_clicked_confirm_to_the_collapsed_chip():
    """AC#4: Esc answers "no" to a pending expand question."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        pasted = "P" * (composer.paste_collapse_threshold + 1)
        composer.insert_pasted_text(pasted)
        await pilot.pause()
        await pilot.click("#console-command-visible-text", offset=(4, 0))
        await pilot.pause()
        assert composer._display_draft_text() == _paste_confirm(pasted)

        await pilot.press("escape")
        await pilot.pause()

        assert composer._display_draft_text() == _paste_chip(pasted)
        assert not composer.has_pending_paste_confirmation()
        assert composer.draft_text() == pasted
        assert host.focused is composer


@pytest.mark.asyncio
async def test_a_file_token_keeps_its_name_through_the_clicked_confirm():
    """AC#4: the question after a click names the file and the keys."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        composer.insert_file_segment(FILE_TEXT, FILE_LABEL)
        await pilot.pause()
        assert composer._display_draft_text() == f"{FILE_LABEL} | Ctrl+E expand"

        await pilot.click("#console-command-visible-text", offset=(4, 0))
        await pilot.pause()

        assert composer._display_draft_text() == (
            f"Expand {FILE_LABEL}? Ctrl+E: yes · Esc: no"
        )
        assert composer.draft_text() == FILE_TEXT


@pytest.mark.asyncio
async def test_moving_the_caret_off_a_clicked_confirm_answers_no():
    """AC#4: the caret leaving the token it asks about puts the chip back."""
    _, host = _ready_host()
    async with host.run_test(size=APP_SIZE) as pilot:
        composer = await _focused_composer(host, pilot)
        pasted = "P" * (composer.paste_collapse_threshold + 1)
        composer.insert_pasted_text(pasted)
        composer.insert_text(" and a tail")
        await pilot.pause()
        await pilot.click("#console-command-visible-text", offset=(4, 0))
        await pilot.pause()
        assert composer.has_pending_paste_confirmation()
        assert composer.cursor_index == len(pasted)

        await pilot.press("right")
        await pilot.pause()

        assert composer.cursor_index == len(pasted) + 1
        assert composer._display_draft_text() == f"{_paste_chip(pasted)} and a tail"
        assert not composer.has_pending_paste_confirmation()


def test_f1_help_names_the_expand_key():
    """F1's Composer group lists Ctrl+E beside Enter."""
    from tldw_chatbook.UI.Screens.chat_screen import CONSOLE_WORKBENCH_SHORTCUT_GROUPS

    composer_keys = dict(dict(CONSOLE_WORKBENCH_SHORTCUT_GROUPS)["Composer"])
    assert "Ctrl+E" in composer_keys
    assert "expand" in composer_keys["Ctrl+E"]
    assert "Enter" in composer_keys
    assert "expand" not in composer_keys["Enter"]
