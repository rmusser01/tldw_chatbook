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
from textual.widgets import Input, Static

from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _configure_native_ready_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_composer_bar import _DraftHistorySnapshot


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


def test_composer_load_draft_never_records_history_and_wipes_stale_scope_history():
    """`load_draft` never records its own replacement as an undo entry --
    AND (review F4) it wipes whatever history the previous scope left
    behind, since load_draft always represents a scope change (a session
    switch or a launch-context prefill), never an edit. Leaving a prior
    scope's stack live let one undo after a prefill return an unrelated
    older state that had nothing to do with the new scope."""
    composer = ConsoleComposerBar()
    composer.insert_text("x")
    composer.load_draft("programmatic replacement")
    assert composer.draft_text() == "programmatic replacement"

    # The "x" run's entry must NOT survive the scope change.
    assert composer.undo() is False
    assert composer.draft_text() == "programmatic replacement"

    # And the new scope records its own edits normally.
    composer.insert_text("!")
    assert composer.undo() is True
    assert composer.draft_text() == "programmatic replacement"


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
# Review fix-round (2026-07-30): F3, F4, F6 -- pure composer-level, pinned
# reproductions of the reviewer's own scenarios.
# ---------------------------------------------------------------------------


def test_composer_coalescing_reset_after_restore_stashed_draft():
    """F3, exact reviewer repro: `_coalescing_active` used to survive
    `stash_draft_for_send`/`restore_stashed_draft`'s non-recording clear,
    so the first typed character after a rejected-send round trip silently
    coalesced into (and was swallowed by) the pre-send typed run instead of
    recording its own entry -- one Ctrl+Z skipped straight past "hello" to
    empty instead of landing on "hello" first."""
    composer = ConsoleComposerBar()
    for character in "hello":
        composer.insert_text(character)
    assert composer.draft_text() == "hello"

    stash = composer.stash_draft_for_send()
    composer.restore_stashed_draft(stash)
    assert composer.draft_text() == "hello"

    composer.insert_text("X")
    assert composer.draft_text() == "helloX"

    # The "X" run must be its own undo entry -- one undo reverts only it.
    assert composer.undo() is True
    assert composer.draft_text() == "hello"


def test_composer_coalescing_reset_after_clear_draft():
    """F3: the same failure shape via the plain (non-recording) `clear_draft()`.

    Uses a non-empty prior draft (via `load_draft`, itself non-recording)
    so the stale pre-clear entry's pre-state ("existing") is distinguishable
    from the correct one (the empty draft right after the clear) -- a
    variant of this test that opens the coalescing run on a fresh, empty
    composer would pass either way, since both the stale and the correct
    entry happen to share the same "" pre-state there.
    """
    composer = ConsoleComposerBar()
    composer.load_draft("existing")
    composer.insert_text("Z")  # opens a coalescing run, pre="existing"
    composer.clear_draft()  # default record_history=False
    composer.insert_text("c")
    composer.insert_text("d")
    assert composer.draft_text() == "cd"

    # "cd" must be its own entry (pre="" -- the state right after the
    # clear) -- not merged into the stale "existingZ" run, whose pre-state
    # is "existing", a different (and wrong) string to land back on.
    assert composer.undo() is True
    assert composer.draft_text() == ""


def test_composer_load_draft_wipes_stale_scope_history():
    """F4, exact reviewer repro: the launch-context prefill `load_draft`
    call site left the PREVIOUS scope's undo stack live -- one undo after
    the prefill returned an unrelated older state that had nothing to do
    with the new scope."""
    composer = ConsoleComposerBar()
    for character in "typo":
        composer.insert_text(character)
    for _ in range(4):
        composer.delete_left()
    assert composer.draft_text() == ""

    composer.load_draft("PREFILLED SUGGESTED PROMPT")

    assert composer.undo() is False
    assert composer.draft_text() == "PREFILLED SUGGESTED PROMPT"


def test_composer_undo_history_capped_by_total_characters_not_just_entry_count():
    """F6, measured reviewer repro: a large inlined attachment followed by
    ordinary pastes multiplied to >20,000,000 retained characters across
    just 21 entries -- nowhere near the 100-entry depth cap. The stacks
    must also be bounded by a total character budget."""
    composer = ConsoleComposerBar()
    composer.insert_file_segment("x" * 1_000_000, label="big.txt")
    for index in range(20):
        composer.insert_pasted_text(f"paste-{index} ")

    total_chars = sum(len(entry.text) for entry in composer._undo_stack)
    assert total_chars <= composer.UNDO_HISTORY_CHAR_BUDGET
    assert len(composer._undo_stack) <= composer.UNDO_HISTORY_DEPTH_CAP


def test_composer_restore_undo_history_enforces_char_budget_too():
    """F6: the budget also applies to a history handed in externally
    (session-switch restore), not just to entries recorded live -- a
    banked history built before this eviction existed (or handed in from
    elsewhere) must not bypass it."""
    composer = ConsoleComposerBar()
    big_text = "y" * 1_500_000
    bloated_undo = [
        _DraftHistorySnapshot(text=big_text, cursor_index=0),
        _DraftHistorySnapshot(text=big_text, cursor_index=0),
        _DraftHistorySnapshot(text=big_text, cursor_index=0),
    ]
    composer.restore_undo_history((bloated_undo, []))

    total_chars = sum(len(entry.text) for entry in composer._undo_stack)
    assert total_chars <= composer.UNDO_HISTORY_CHAR_BUDGET


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


# ---------------------------------------------------------------------------
# Review fix-round (2026-07-30): F1 (HIGH), F2, F5, N1 -- pinned
# reproductions of the reviewer's own scenarios, screen/pilot level.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_undo_during_switch_settle_window_does_not_apply_or_corrupt():
    """F1 (HIGH), exact reviewer repro: `store.active_session_id` can change
    (via `controller.switch_session`) before `_console_visible_draft_
    session_id` catches up inside the deferred `_sync_console_session_
    draft` -- the TASK-339 settle window. Ctrl+Z during that window used to
    apply session A's undo history to the composer and then persist the
    result to the store under session B's (now-active) id, permanently
    overwriting B's own draft once the deferred swap finally landed."""
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

        for character in "aaa-secret":
            composer.insert_text(character)
        composer.insert_pasted_text(" MORE")
        assert composer.draft_text() == "aaa-secret MORE"

        session_b = store.create_session(title="Session B")
        store.set_session_draft(session_b.id, "bbb-b-own-draft")
        # Settle window open: the store already considers B active, but the
        # composer still visibly shows A (`_sync_console_session_draft`
        # has not run since B was created, so `_console_visible_draft_
        # session_id` has not caught up yet).
        assert console._console_visible_draft_session_id == session_a.id
        assert store.active_session_id == session_b.id

        console._console_composer_undo()

        # Nothing may move while the window is open: the composer still
        # shows A's untouched text, and B's own draft in the store must
        # survive completely untouched.
        assert composer.draft_text() == "aaa-secret MORE"
        assert store.session_draft(session_b.id) == "bbb-b-own-draft"

        # Now let the deferred swap actually run: A's real (untouched)
        # draft reaches the store as A's draft, and switching to B shows
        # B's own draft -- never A's leaked text.
        console._sync_console_session_draft()
        assert store.session_draft(session_a.id) == "aaa-secret MORE"
        assert composer.draft_text() == "bbb-b-own-draft"


@pytest.mark.asyncio
async def test_console_redo_during_switch_settle_window_does_not_apply():
    """F1 (HIGH): the same settle-window guard for redo."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()

        session_a = store.ensure_session(title="Session A")
        composer.focus()
        await pilot.pause()
        composer.load_draft("")
        console._sync_console_session_draft()

        composer.insert_text("x")
        composer.undo()  # arm the redo stack
        assert composer.draft_text() == ""

        session_b = store.create_session(title="Session B")
        store.set_session_draft(session_b.id, "b-own-draft")
        assert console._console_visible_draft_session_id == session_a.id
        assert store.active_session_id == session_b.id

        console._console_composer_redo()

        assert composer.draft_text() == ""
        assert store.session_draft(session_b.id) == "b-own-draft"


@pytest.mark.asyncio
async def test_console_undo_after_accepted_send_does_not_resurrect_sent_content():
    """F2, exact reviewer repro: the pre-send mutations stayed reachable on
    the undo stack after an accepted send, so Ctrl+Z resurrected already-
    sent content back into the composer AND re-persisted it into the store
    as the session's "live" draft."""
    gateway = CapturingGateway()
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        composer.focus()
        await pilot.pause(0.1)

        for character in "PASTED-SECRET":
            composer.insert_text(character)
        composer.insert_pasted_text(" q?")
        assert composer.draft_text() == "PASTED-SECRET q?"

        await pilot.press("enter")
        await _wait_for_text(console, pilot, "accepted")
        assert composer.draft_text() == ""

        await pilot.press("ctrl+z")
        await pilot.pause(0.1)

        assert composer.draft_text() == ""
        assert store.session_draft(store.active_session_id) == ""


@pytest.mark.asyncio
async def test_console_background_dictation_drops_stale_session_history():
    """F5, exact reviewer repro: a dictation transcript that lands via the
    store-only branch (the origin session isn't the visible one) left that
    session's banked undo/redo history stale relative to the store draft it
    is re-paired with on switch-in -- one Ctrl+Z after switching back
    destroyed the dictated text AND the whole pre-existing draft in a
    single step. The banked history must be dropped instead, making the
    dictated text simply not undoable (safe) rather than destructively so."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()

        session_a = store.ensure_session(title="Session A")
        composer.focus()
        await pilot.pause()
        composer.load_draft("")
        console._sync_console_session_draft()

        for character in "hello":
            composer.insert_text(character)
        assert composer.draft_text() == "hello"

        store.create_session(title="Session B")
        console._sync_console_session_draft()
        assert composer.draft_text() == ""

        # Dictation finishes for A while B is visible -- the store-only branch.
        console._insert_console_dictation(
            origin_session_id=session_a.id, transcript="dictated words"
        )
        assert store.session_draft(session_a.id) == "hello dictated words"

        store.switch_session(session_a.id)
        console._sync_console_session_draft()
        assert composer.draft_text() == "hello dictated words"

        # Nothing to undo: the store-only mutation isn't recorded, and the
        # stale pre-dictation history was dropped rather than left live.
        console._console_composer_undo()
        assert composer.draft_text() == "hello dictated words"


@pytest.mark.asyncio
async def test_console_prompt_append_undo_removes_separator_newline_too():
    """N1: `_insert_prompt_text_into_composer(replace=False)` used to record
    the separator newline and the pasted body as two separate undo entries,
    so one Ctrl+Z left a stray blank line behind. They must undo together."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("existing draft")
        await pilot.pause()

        assert console._insert_prompt_text_into_composer("resolved body", replace=False)
        assert composer.draft_text() == "existing draft\nresolved body"

        assert composer.undo() is True
        assert composer.draft_text() == "existing draft"


# ---------------------------------------------------------------------------
# Re-review fix round (2026-07-30): NEW-1, NEW-2, NEW-5 -- pinned
# reproductions of the reviewer's own re-review scenarios.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_composer_delete_left_uninitialized_fallback_is_undoable():
    """NEW-1: `delete_left`'s uninitialized-segments fallback used to
    record a snapshot immediately discarded by the `load_draft()` shortcut
    it then called (F4 made `load_draft` unconditionally wipe history),
    silently making that one deletion path non-undoable. The mounted
    composer normally initializes segments immediately (`compose()` calls
    `load_draft`), so this state is only reachable defensively; forced
    directly here, matching how the reviewer reproduced it."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer._segments = []
        composer._segments_initialized = False
        composer.query_one("#console-command-input", Input).value = "legacy"

        composer.delete_left()
        assert composer.draft_text() == "legac"

        assert composer.undo() is True
        assert composer.draft_text() == "legacy"


@pytest.mark.asyncio
async def test_console_undo_redo_re_collapses_over_threshold_restored_segment():
    """NEW-2 (MEDIUM, severity-corrected F7): a restored over-threshold
    segment used to flatten into one giant LITERAL segment, so
    `_refresh_visible_draft` ran its O(n^2) wrap/render path against the
    full text on every undo/redo -- measured by the reviewer at up to 283s
    frozen for a 2.4 MB restored draft, and 2.89s for a realistic
    one-keystroke repro (attach 200 KB, type one character, undo). The
    restored segment must instead re-collapse into the same paste-token
    display a fresh over-threshold paste gets -- a BEHAVIOR pin (what
    paints), not a timing pin.

    Gated on `UNDO_RECOLLAPSE_CHAR_THRESHOLD` (review W-1), NOT the
    cosmetic `paste_collapse_threshold` -- see
    `test_console_undo_of_ordinary_typed_text_stays_literal_not_a_paste_token`
    for the regression an earlier version of this fix introduced by using
    the wrong (much smaller) threshold."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        threshold = composer.UNDO_RECOLLAPSE_CHAR_THRESHOLD
        big_text = "x" * (threshold + 500)

        composer.insert_file_segment(big_text, label="big.txt")
        composer.focus()
        await pilot.pause()

        composer.insert_text("!")
        assert composer.undo() is True

        # The painted composer must show the collapsed token, not the
        # literal payload -- painting the literal is exactly what froze
        # the UI thread.
        painted = visible_draft.render_line(0).text.rstrip()
        assert f"Pasted Text: {len(big_text)} Characters" in painted
        assert big_text not in painted

        # Sanity: canonical text still round-trips the FULL content --
        # store persistence and send must see the whole payload, not the
        # short display token.
        assert composer.draft_text() == big_text

        # Redo lands back on the "!"-appended state, also over threshold --
        # must repaint collapsed too, not expanded.
        assert composer.redo() is True
        painted_after_redo = visible_draft.render_line(0).text.rstrip()
        assert "Pasted Text:" in painted_after_redo
        assert big_text not in painted_after_redo
        assert composer.draft_text() == big_text + "!"


def test_composer_undo_redo_does_not_double_collapse_already_collapsed_snapshot():
    """NEW-2 cross-check (requested explicitly): undoing back to a state
    that itself contained an over-threshold paste -- already collapsed at
    INSERTION time, not by the undo/redo re-collapse -- must show that
    same token correctly: not a token wrapping a token, and not a wrong
    character count. Confirms the re-collapse operates on the segment's
    real underlying text, never on a previously-rendered display string.

    The payload must exceed `UNDO_RECOLLAPSE_CHAR_THRESHOLD` (review W-1),
    not merely `paste_collapse_threshold`: a paste collapsed at insertion
    (over the small cosmetic threshold) but under the large recollapse
    threshold is EXPECTED to come back as ordinary literal text after an
    undo -- that's the whole point of W-1's fix -- so testing "stays
    collapsed" needs a payload big enough to still cross the real gate."""
    composer = ConsoleComposerBar()
    threshold = composer.UNDO_RECOLLAPSE_CHAR_THRESHOLD
    big_text = "y" * (threshold + 100)

    composer.insert_pasted_text(big_text)  # collapsed at insertion, not by undo/redo
    composer.insert_text("!")  # a second, unrelated edit -- its own entry

    assert composer.undo() is True
    assert composer.draft_text() == big_text
    assert [segment.collapse_state for segment in composer._segments] == ["collapsed"]
    # The segment's real text is the raw payload, never the display string
    # ("Pasted Text: N Characters") -- a token-of-a-token would fail this.
    assert composer._segments[0].text == big_text

    assert composer.undo() is True
    assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_refused_send_preserves_undo_history(monkeypatch):
    """NEW-5: a REFUSED send (`result.accepted=False`) must leave a
    background session's banked undo/redo history intact -- nothing was
    actually sent, so there is nothing to treat as a history barrier.
    Previously `_console_undo_histories.pop(session_id, None)` ran
    unconditionally after the accept-gated clear block, dropping the
    banked history even on a refusal."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()

        session_a = store.ensure_session(title="Session A")
        composer.focus()
        await pilot.pause()
        composer.load_draft("")
        console._sync_console_session_draft()

        for character in "hello":
            composer.insert_text(character)
        assert composer.draft_text() == "hello"

        # Switch away, banking A's history.
        store.create_session(title="Session B")
        console._sync_console_session_draft()
        assert composer.draft_text() == ""
        assert session_a.id in console._console_undo_histories

        controller = console._ensure_console_chat_controller()

        async def _refused(draft, *, session_id=None):
            return ConsoleSubmitResult(accepted=False, should_clear_draft=False)

        monkeypatch.setattr(controller, "submit_draft", _refused)

        await console._submit_console_native_draft(
            "attempted body", session_id=session_a.id
        )

        # The refusal must NOT have dropped A's banked history.
        assert session_a.id in console._console_undo_histories

        store.switch_session(session_a.id)
        console._sync_console_session_draft()
        assert composer.draft_text() == "hello"

        console._console_composer_undo()
        assert composer.draft_text() == ""


# ---------------------------------------------------------------------------
# Wave-2 re-review fix round (2026-07-30): W-1 (HIGH) and W-2 (LOW) -- pinned
# reproductions of the reviewer's own wave-2 scenarios.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_undo_of_ordinary_typed_text_stays_literal_not_a_paste_token():
    """W-1 (HIGH): the NEW-2 re-collapse fix reused `paste_collapse_
    threshold` (shipped default 50 chars) as the undo-restore gate, so
    undo of ordinary TYPED draft text over 50 characters repainted as an
    opaque "Pasted Text: N Characters" token -- `has_paste_segments()`
    became True for text that was never pasted, and one Backspace then
    deleted the WHOLE restored draft in a single step (a collapsed token
    deletes as one unit). This lands squarely on the AC's own flagship
    recovery path: Ctrl+U on a real message, then Ctrl+Z, must still show
    (and let you edit character-by-character) the recovered text."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        message = (
            "Please summarise the attached report and list the three "
            "biggest risks that stand out to you."
        )
        # Comfortably over the cosmetic paste threshold (50), comfortably
        # under the dedicated recollapse threshold (20,000).
        assert 50 < len(message) < composer.UNDO_RECOLLAPSE_CHAR_THRESHOLD

        composer.load_draft(message)
        composer.focus()
        await pilot.pause()

        # The AC's own flagship recovery path: Ctrl+U, then Ctrl+Z.
        composer.clear_draft(record_history=True)
        assert composer.draft_text() == ""
        assert composer.undo() is True

        assert composer.draft_text() == message
        assert composer.has_paste_segments() is False
        painted = visible_draft.render_line(0).text.rstrip()
        assert message[:20] in painted
        assert "Pasted Text:" not in painted

        # One Backspace removes exactly one character -- not the token
        # that a collapsed segment would delete as an atomic unit.
        composer.delete_left()
        assert composer.draft_text() == message[:-1]


def test_composer_undo_snaps_cursor_out_of_a_collapsed_restored_token():
    """W-1: `_apply_history_snapshot` used to restore the raw
    `snapshot.cursor_index` verbatim even when the restored segment
    collapsed, leaving the caret INSIDE a token -- no other code path can
    produce that state, since paste-token caret movement always treats a
    token as atomic (reviewer repro: a 120-char draft with the caret
    reporting 60 inside a now-collapsed segment, where typing then
    inserted at canonical offset 0, not 60)."""
    composer = ConsoleComposerBar()
    threshold = composer.UNDO_RECOLLAPSE_CHAR_THRESHOLD
    base = "A" * (threshold + 100)  # even length: midpoint splits it exactly

    composer.load_draft(base)
    composer.position_cursor_from_display_index(len(base) // 2)
    assert composer.cursor_index == len(base) // 2

    composer.insert_text("Z")
    assert composer.undo() is True

    # The restored segment is over threshold -> collapsed; the caret must
    # land on a real token boundary (0 or the token's end), never inside.
    # The pre-undo caret (the exact midpoint) is nearer the end here.
    assert composer.cursor_index == len(base)
    assert composer.draft_text() == base

    # And the reported position is genuinely where the next edit lands.
    composer.insert_text("Q")
    assert composer.draft_text() == base + "Q"


# ---------------------------------------------------------------------------
# TASK-1500: Ctrl+Y as a Kitty-protocol-independent redo alias. Ctrl+Shift+Z
# collapses to plain Ctrl+Z at the wire on terminals without the Kitty
# keyboard protocol (Terminal.app, stock iTerm2), making redo unreachable
# there. Ctrl+Y is the C0 control EM (0x19) -- textual's `_ansi_sequences`
# maps it to `Keys.ControlY` == "ctrl+y" everywhere, Kitty or not.
# ---------------------------------------------------------------------------


def test_ctrl_y_is_not_registered_as_a_binding():
    """Routed through `on_key`'s whitelist, never `BINDINGS`, matching how
    Ctrl+Z/Ctrl+Shift+Z are wired (TASK-1281)."""
    keys = {binding.key for binding in chat_screen_module.ChatScreen.BINDINGS}
    assert "ctrl+y" not in keys


@pytest.mark.asyncio
async def test_console_ctrl_z_then_ctrl_y_restores_the_undone_edit():
    """AC1/AC2: Ctrl+Y redoes exactly like Ctrl+Shift+Z does, and both keys
    keep working side by side -- Ctrl+Y is an addition, not a replacement."""
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

        await pilot.press("ctrl+y")
        await pilot.pause()
        assert composer.draft_text() == "hi"

        # AC2: Ctrl+Shift+Z must still work too.
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert composer.draft_text() == ""

        await pilot.press("ctrl+shift+z")
        await pilot.pause()
        assert composer.draft_text() == "hi"


@pytest.mark.asyncio
async def test_console_ctrl_y_on_empty_redo_stack_matches_ctrl_shift_z_behavior():
    """Consistency matters more than the choice itself: whatever
    Ctrl+Shift+Z does on an empty redo stack (silent no-op composer-side,
    no store write -- see `test_console_undo_redo_empty_stack_is_silent_
    noop_via_screen`), Ctrl+Y must do identically."""
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

        await pilot.press("ctrl+y")
        await pilot.pause()
        assert composer.draft_text() == "untouched"
        assert store.session_draft(session_id) == draft_before


def test_composer_help_documents_ctrl_y_alongside_ctrl_shift_z_for_redo():
    """AC3: wherever undo/redo is documented, both redo keys must be listed."""
    groups = dict(chat_screen_module.CONSOLE_WORKBENCH_SHORTCUT_GROUPS)
    assert "Composer" in groups
    composer = groups["Composer"]
    flat = " ".join(f"{key} {label}" for key, label in composer).lower()

    assert "undo" in flat
    assert "redo" in flat
    assert "ctrl+y" in flat
    assert "ctrl+shift+z" in flat


@pytest.mark.asyncio
async def test_console_undo_recollapses_regardless_of_collapse_large_pastes_setting():
    """W-2 (LOW): the re-collapse used to also gate on
    `collapse_large_pastes_enabled` -- a cosmetic paste-display preference
    -- so a user with `collapse_large_pastes = False` got NEW-2's
    multi-second freeze back in full. The performance guard must hold
    regardless of that preference; only collapse-ON-PASTE cosmetics should
    respect it."""
    _, host = _ready_host()

    async with host.run_test(size=(140, 42)) as pilot:
        console = await _mounted_console(host, pilot)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        composer.collapse_large_pastes = False
        threshold = composer.UNDO_RECOLLAPSE_CHAR_THRESHOLD
        big_text = "z" * (threshold + 500)

        composer.insert_file_segment(big_text, label="big.txt")
        composer.focus()
        await pilot.pause()

        composer.insert_text("!")
        assert composer.undo() is True

        painted = visible_draft.render_line(0).text.rstrip()
        assert f"Pasted Text: {len(big_text)} Characters" in painted
        assert big_text not in painted
        assert composer.draft_text() == big_text
