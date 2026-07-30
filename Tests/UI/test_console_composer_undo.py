"""Console composer undo/redo (TASK-1281).

Undo/redo history lives inside `ConsoleComposerBar` as a stack of (draft
text, cursor index) snapshots. Every user-intent mutation -- typing, paste,
attachment/file segments, dictation insertion, backspace/delete/Ctrl+W, and
an explicit Ctrl+U clear -- records the PRE-mutation state before it changes
anything. Consecutive single-character *printable* inserts (ordinary typing)
coalesce into one undo entry so a single Ctrl+Z reverts a whole typed run;
any other mutation kind, a deletion, or a cursor reposition closes that run.

`ChatScreen` wires Ctrl+Z/Ctrl+Shift+Z into its existing `on_key` whitelist
(next to Ctrl+U, never `BINDINGS`), scopes history per Console session via
`export_undo_history`/`restore_undo_history` around
`_sync_console_session_draft`'s switch path, and re-persists the resulting
draft to the console chat store after every undo/redo -- mirroring how
`_insert_console_dictation` re-persists after inserting a transcript --
so the store and the visible composer can never split-brain.

Pure composer-level tests below build an unmounted `ConsoleComposerBar()`
directly (no App needed), following the pattern already established in
`test_console_composer_cursor.py`. Pilot-driven tests crib their App/harness
fixtures from `test_console_dictation.py`/`test_console_dictation_streaming.py`.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


# ---------------------------------------------------------------------------
# Pure composer-level tests: typed-run coalescing, mutation-kind boundaries,
# redo, depth cap, export/restore -- no App/pilot needed.
# ---------------------------------------------------------------------------


def test_composer_undo_reverts_a_typed_run_as_one_entry():
    """Five separate single-character inserts (one per keypress) coalesce."""
    composer = ConsoleComposerBar()
    for character in "hello":
        composer.insert_text(character)
    assert composer.draft_text() == "hello"

    assert composer.undo() is True
    assert composer.draft_text() == ""
    # Nothing left to undo -- the whole run collapsed into one entry.
    assert composer.undo() is False


def test_composer_undo_restores_cursor_position_not_just_text():
    composer = ConsoleComposerBar()
    composer.load_draft("hello world")
    composer.move_cursor_left()  # cursor now at 10, between "worl" and "d"
    composer.insert_text("X")
    assert composer.draft_text() == "hello worlXd"
    assert composer.cursor_index == 11

    assert composer.undo() is True
    assert composer.draft_text() == "hello world"
    assert composer.cursor_index == 10


def test_composer_cursor_reposition_between_keystrokes_breaks_the_run():
    composer = ConsoleComposerBar()
    for character in "ab":
        composer.insert_text(character)
    composer.move_cursor_left()  # reposition -- closes the "ab" run
    for character in "cd":
        composer.insert_text(character)
    assert composer.draft_text() == "acdb"

    assert composer.undo() is True
    assert composer.draft_text() == "ab"
    assert composer.undo() is True
    assert composer.draft_text() == ""
    assert composer.undo() is False


def test_composer_paste_breaks_typed_run_coalescing():
    composer = ConsoleComposerBar()
    for character in "ab":
        composer.insert_text(character)
    composer.insert_pasted_text("PASTE")
    for character in "cd":
        composer.insert_text(character)
    assert composer.draft_text() == "abPASTEcd"

    # Undo #1 reverts only the "cd" typed run.
    assert composer.undo() is True
    assert composer.draft_text() == "abPASTE"
    # Undo #2 reverts only the paste.
    assert composer.undo() is True
    assert composer.draft_text() == "ab"
    # Undo #3 reverts the "ab" typed run.
    assert composer.undo() is True
    assert composer.draft_text() == ""
    assert composer.undo() is False


def test_composer_undo_reverts_paste_insertion():
    composer = ConsoleComposerBar()
    composer.insert_text("before ")
    composer.insert_pasted_text("a pasted chunk")
    assert composer.draft_text() == "before a pasted chunk"

    assert composer.undo() is True
    assert composer.draft_text() == "before "


def test_composer_undo_reverts_file_segment_insertion():
    composer = ConsoleComposerBar()
    composer.insert_text("before ")
    composer.insert_file_segment("full file contents", label="notes.md")
    assert "full file contents" in composer.draft_text()

    assert composer.undo() is True
    assert composer.draft_text() == "before "


def test_composer_undo_reverts_backspace_and_ctrl_w():
    composer = ConsoleComposerBar()
    composer.load_draft("delete this word")

    composer.delete_word_left()
    assert composer.draft_text() == "delete this "
    assert composer.undo() is True
    assert composer.draft_text() == "delete this word"

    composer.move_cursor_end()
    composer.delete_left()
    assert composer.draft_text() == "delete this wor"
    assert composer.undo() is True
    assert composer.draft_text() == "delete this word"

    composer.move_cursor_home()
    composer.delete_right()
    assert composer.draft_text() == "elete this word"
    assert composer.undo() is True
    assert composer.draft_text() == "delete this word"


def test_composer_clear_draft_record_history_true_is_undoable():
    """The Ctrl+U path: `clear_draft(record_history=True)`."""
    composer = ConsoleComposerBar()
    composer.load_draft("keep this")

    composer.clear_draft(record_history=True)
    assert composer.draft_text() == ""

    assert composer.undo() is True
    assert composer.draft_text() == "keep this"
    assert composer.cursor_index == len("keep this")


def test_composer_clear_draft_default_does_not_record_history():
    """Every other `clear_draft()` call site (session switch, post-send,
    restore-then-replace flows) uses the default and must NOT be undoable."""
    composer = ConsoleComposerBar()
    composer.load_draft("gone")

    composer.clear_draft()
    assert composer.draft_text() == ""
    assert composer.undo() is False


def test_composer_load_draft_never_records_history():
    composer = ConsoleComposerBar()
    composer.insert_text("x")
    composer.load_draft("programmatic replacement")
    assert composer.draft_text() == "programmatic replacement"

    # The "x" run is still the only thing on the stack; load_draft added
    # nothing (and didn't touch what was already there either).
    assert composer.undo() is True
    assert composer.draft_text() == ""


def test_composer_redo_reapplies_undone_mutation():
    composer = ConsoleComposerBar()
    for character in "hi":
        composer.insert_text(character)

    assert composer.undo() is True
    assert composer.draft_text() == ""

    assert composer.redo() is True
    assert composer.draft_text() == "hi"
    assert composer.cursor_index == 2


def test_composer_fresh_edit_after_undo_clears_redo_stack():
    composer = ConsoleComposerBar()
    composer.insert_text("a")
    composer.insert_text("b")
    assert composer.draft_text() == "ab"

    assert composer.undo() is True
    assert composer.draft_text() == ""

    # A fresh edit after the undo must drop the "redo ab" entry.
    composer.insert_text("c")
    assert composer.draft_text() == "c"

    assert composer.redo() is False
    assert composer.draft_text() == "c"


def test_composer_undo_on_empty_stack_is_silent_noop():
    composer = ConsoleComposerBar()
    assert composer.undo() is False
    assert composer.draft_text() == ""


def test_composer_redo_on_empty_stack_is_silent_noop():
    composer = ConsoleComposerBar()
    composer.insert_text("x")
    assert composer.redo() is False
    assert composer.draft_text() == "x"


def test_composer_undo_stack_capped_at_100_dropping_oldest():
    composer = ConsoleComposerBar()
    # Each `insert_pasted_text` call is always its own boundary (never
    # coalesced), so 105 calls push 105 candidate entries.
    for index in range(105):
        composer.insert_pasted_text(f"p{index} ")
    assert len(composer._undo_stack) == 100


def test_composer_export_and_restore_undo_history_round_trips():
    composer = ConsoleComposerBar()
    composer.insert_text("a")
    composer.insert_text("b")
    composer.insert_pasted_text("P")
    exported = composer.export_undo_history()

    fresh = ConsoleComposerBar()
    fresh.restore_undo_history(exported)
    assert fresh.export_undo_history() == exported

    # The export is a real copy, not aliased to the live stacks.
    composer.insert_text("c")
    assert composer.export_undo_history() != exported


def test_composer_restore_undo_history_none_gives_empty_stacks():
    composer = ConsoleComposerBar()
    composer.insert_text("x")
    composer.restore_undo_history(None)
    assert composer.undo() is False


# ---------------------------------------------------------------------------
# Screen-level key routing (AC1) and store/session integration -- pilot tests.
# ---------------------------------------------------------------------------


def test_ctrl_z_and_ctrl_shift_z_are_not_registered_as_bindings():
    """AC1: routed through `on_key`'s whitelist, never `BINDINGS`."""
    keys = {binding.key for binding in chat_screen_module.ChatScreen.BINDINGS}
    assert "ctrl+z" not in keys
    assert "ctrl+shift+z" not in keys
    assert "ctrl+shift+Z" not in keys


@pytest.mark.asyncio
async def test_console_ctrl_z_reverts_typed_run_and_ctrl_shift_z_redoes_it():
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        for character in "hi":
            composer.insert_text(character)
        assert composer.draft_text() == "hi"

        await pilot.press("ctrl+z")
        await pilot.pause()
        assert composer.draft_text() == ""

        await pilot.press("ctrl+shift+z")
        await pilot.pause()
        assert composer.draft_text() == "hi"


@pytest.mark.asyncio
async def test_console_ctrl_u_clear_is_undoable_via_ctrl_z():
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep me")
        composer.focus()
        await pilot.pause()

        await pilot.press("ctrl+u")
        await pilot.pause()
        assert composer.draft_text() == ""

        await pilot.press("ctrl+z")
        await pilot.pause()
        assert composer.draft_text() == "keep me"
        assert composer.cursor_index == len("keep me")


@pytest.mark.asyncio
async def test_console_dictation_insertion_is_undoable():
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        composer.load_draft("hello world")
        composer.move_cursor_end()
        await pilot.pause(0.1)

        console._insert_console_dictation(
            origin_session_id=session_id, transcript="there"
        )
        assert composer.draft_text() == "hello world there"
        assert store.session_draft(session_id) == "hello world there"

        console._console_composer_undo()
        assert composer.draft_text() == "hello world"
        assert store.session_draft(session_id) == "hello world"


@pytest.mark.asyncio
async def test_console_undo_and_redo_keep_store_in_sync_with_composer():
    """Store consistency: `store.session_draft(...)` matches the composer
    after every undo/redo, mirroring `_insert_console_dictation`'s own
    re-persist (line ~5192)."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        composer.focus()
        await pilot.pause()

        for character in "abc":
            composer.insert_text(character)
        assert composer.draft_text() == "abc"

        console._console_composer_undo()
        assert composer.draft_text() == ""
        assert store.session_draft(session_id) == composer.draft_text()

        console._console_composer_redo()
        assert composer.draft_text() == "abc"
        assert store.session_draft(session_id) == composer.draft_text()


@pytest.mark.asyncio
async def test_console_undo_redo_empty_stack_is_silent_noop_via_screen():
    """No toast, no bell, no store write on an empty stack (AC-adjacent)."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        composer.load_draft("untouched")
        composer.focus()
        await pilot.pause()
        draft_before = store.session_draft(session_id)

        await pilot.press("ctrl+z")
        await pilot.pause()
        assert composer.draft_text() == "untouched"
        assert store.session_draft(session_id) == draft_before

        await pilot.press("ctrl+shift+z")
        await pilot.pause()
        assert composer.draft_text() == "untouched"


@pytest.mark.asyncio
async def test_console_undo_history_scoped_per_session_never_leaks():
    """AC4: editing in session A, switching to B, must not let Ctrl+Z touch
    B's draft or apply A's history; switching back to A, undo still works."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()

        session_a = store.ensure_session(title="Session A")
        composer.focus()
        await pilot.pause()
        composer.load_draft("")
        console._sync_console_session_draft()  # settle tracker onto A

        for character in "hello":
            composer.insert_text(character)
        assert composer.draft_text() == "hello"

        # Switch to a brand-new session B.
        session_b = store.create_session(title="Session B")
        console._sync_console_session_draft()
        assert store.active_session_id == session_b.id
        assert composer.draft_text() == ""

        # Ctrl+Z in B must be a no-op: no history was ever recorded there,
        # and it must never reach for A's stack.
        console._console_composer_undo()
        assert composer.draft_text() == ""

        # Edit in B, to prove undo undoes B's OWN edit, not A's.
        composer.insert_text("world")
        console._console_composer_undo()
        assert composer.draft_text() == ""

        # Switch back to A: its original undo entry must still be intact.
        store.switch_session(session_a.id)
        console._sync_console_session_draft()
        assert composer.draft_text() == "hello"

        console._console_composer_undo()
        assert composer.draft_text() == ""
        assert store.session_draft(session_a.id) == ""
