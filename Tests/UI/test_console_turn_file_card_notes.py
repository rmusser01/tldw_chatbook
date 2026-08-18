"""Turn file card: note UI on hunks (TASK-16800 Task 4).

REAL provider stack -- same fixture pattern as
``Tests/UI/test_change_review_screen.py::review_fixture`` (real
``ChangeTurnTracker``/``ShadowRepoService``, a FILE-BACKED ``AgentRunsDB``,
the real ``AgentRunsChangeReviewProvider``) and the real CSS bundle host
from ``Tests/UI/test_console_turn_file_card.py`` -- the fixture-invented-
shapes trap has bitten this repo five separate times, so no fake provider
shapes are hand-rolled here.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input

from tldw_chatbook.css import build_css
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.UI.Screens.change_review_screen import (
    AgentRunsChangeReviewProvider,
)
from tldw_chatbook.Widgets.Console.console_turn_file_card import (
    ConsoleTurnFileCard,
)
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.change_turn_tracker import ChangeTurnTracker

_CSS_DIR = Path(build_css.__file__).parent
_SELF, _SCOPED = build_css.screen_css_paths(_CSS_DIR)

MARKER = "✎ Edited 1 file  +1 −1 — review with `v`"

CONV_ID = "conv-1"


def _record_turn(db, tracker, root, run_id: str, mutate) -> None:
    """One real tracked turn: baseline, mutate the tree, end, store rows."""
    handle = tracker.begin_turn([root])
    handle.await_baseline()
    mutate()
    for rec in tracker.end_turn(handle):
        db.record_change_snapshot(
            run_id=run_id,
            root=rec.root,
            baseline_sha=rec.baseline_sha,
            end_sha=rec.end_sha,
            files_changed=rec.files_changed,
            adds=rec.adds,
            dels=rec.dels,
            tracking_error=rec.tracking_error,
            untracked_oversize=rec.untracked_oversize,
            nested_repos=rec.nested_repos,
        )


@pytest.fixture()
def notes_fixture(tmp_path):
    """One real tracked turn touching a single file (`a.py`, one hunk)."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.py").write_text("line1\nline2\nline3\n")

    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id=CONV_ID, agent_kind="primary")

    def mutate():
        (root / "a.py").write_text("line1\nCHANGED\nline3\n")

    _record_turn(db, tracker, root, run_id, mutate)

    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=CONV_ID
    )
    return db, service, provider, root, run_id


class _Host(App):
    CSS_PATH = [str(_SELF), str(_CSS_DIR / "tldw_cli_modular.tcss"), str(_SCOPED)]

    def __init__(self, provider_factory, run_id: str) -> None:
        super().__init__()
        self._provider_factory = provider_factory
        self._run_id = run_id

    def compose(self) -> ComposeResult:
        yield ConsoleTurnFileCard(
            MARKER, self._run_id, self._provider_factory, id="card-under-test"
        )


async def _settled_card(pilot):
    card = pilot.app.query_one("#card-under-test", ConsoleTurnFileCard)
    for _ in range(60):
        if card.query(".console-turn-file-row"):
            break
        await pilot.pause(0.02)
    return card


async def _expand_first_row(pilot, card):
    """Press the first row open and return its diff body once displayed.

    Focus+keyboard-Enter (not a positional click) -- matches
    ``test_console_turn_file_card.py``'s own proven-stable expand
    mechanism, which is layout-independent (a click's screen coordinates
    can race a not-yet-settled compositor under heavier parallel test
    load; focus+Enter only needs the widget reference).
    """
    row = card.query(".console-turn-file-row").first()
    row.focus()
    await pilot.press("enter")
    for _ in range(60):
        bodies = card.query(".console-turn-file-diff")
        if bodies and bodies.first().display:
            return bodies.first()
        await pilot.pause(0.02)
    raise AssertionError("diff body never displayed")


async def _open_note_input(pilot, body):
    """Press the hunk's `note` button and return the opened Input."""
    note_btn = body.query_one(".console-turn-file-note-btn", Button)
    note_btn.focus()
    await pilot.press("enter")
    for _ in range(60):
        inputs = body.query(".console-turn-file-note-input")
        if inputs:
            return inputs.first()
        await pilot.pause(0.02)
    raise AssertionError("note input never opened")


def _note_text(note_row) -> str:
    """The visible text of a mounted `.console-turn-file-note` row.

    The row is a `Horizontal` container -- its own `render()` is a Blank
    placeholder (containers paint children, not self-content, the same
    lesson `test_console_turn_file_card.py` already documents for the
    diff body); the text lives on the `.console-turn-file-note-text`
    child Static.
    """
    from textual.widgets import Static

    return str(note_row.query_one(".console-turn-file-note-text", Static).render())


@pytest.mark.asyncio
async def test_note_save_persists_and_renders_anchored(notes_fixture):
    """Enter in the note input persists an anchored ``change_notes`` row
    (spec §1/§3: run_id/root/path/hunk_index/hunk_header/hunk_excerpt) and
    renders the `.console-turn-file-note` row in place of the input.
    """
    db, service, provider, root, run_id = notes_fixture

    async with _Host(lambda: provider, run_id).run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)

        note_input = await _open_note_input(pilot, body)
        note_input.value = "use the cached value here"
        note_input.focus()
        await pilot.press("enter")

        note_row = None
        for _ in range(60):
            rows = body.query(".console-turn-file-note")
            if rows:
                note_row = rows.first()
                break
            await pilot.pause(0.02)
        assert note_row is not None, "note row never rendered after save"
        assert "use the cached value here" in _note_text(note_row)
        assert not body.query(".console-turn-file-note-input"), (
            "input must be replaced by the note row"
        )

        rows = db.notes_for_run(run_id)
        assert len(rows) == 1
        row = rows[0]
        assert row["run_id"] == run_id
        assert row["root"] == str(root)
        assert row["path"] == "a.py"
        assert row["hunk_index"] == 0
        assert row["hunk_header"].startswith("@@")
        assert row["note"] == "use the cached value here"
        assert row["delivered_at"] is None


@pytest.mark.asyncio
async def test_escape_cancels_note_input_without_saving(notes_fixture):
    """Escape unmounts the open note input; nothing is persisted."""
    db, service, provider, root, run_id = notes_fixture

    async with _Host(lambda: provider, run_id).run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)

        note_input = await _open_note_input(pilot, body)
        note_input.value = "abandoned draft"
        note_input.focus()
        await pilot.pause()

        await pilot.press("escape")
        for _ in range(60):
            if not body.query(".console-turn-file-note-input"):
                break
            await pilot.pause(0.02)
        assert not body.query(".console-turn-file-note-input")
        assert not body.query(".console-turn-file-note")
        assert db.notes_for_run(run_id) == []


@pytest.mark.asyncio
async def test_delete_removes_pending_note(notes_fixture):
    """The ✕ button on a pending note deletes it, off-thread, and its row
    is removed from the card.
    """
    db, service, provider, root, run_id = notes_fixture

    async with _Host(lambda: provider, run_id).run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)

        note_input = await _open_note_input(pilot, body)
        note_input.value = "delete me"
        note_input.focus()
        await pilot.press("enter")

        delete_btn = None
        for _ in range(60):
            buttons = body.query(".console-turn-file-note-delete")
            if buttons:
                delete_btn = buttons.first()
                break
            await pilot.pause(0.02)
        assert delete_btn is not None, "delete button never rendered"
        assert len(db.notes_for_run(run_id)) == 1

        delete_btn.focus()
        await pilot.press("enter")
        for _ in range(60):
            if not body.query(".console-turn-file-note"):
                break
            await pilot.pause(0.02)
        assert not body.query(".console-turn-file-note")
        assert db.notes_for_run(run_id) == []


@pytest.mark.asyncio
async def test_delete_press_on_note_delivered_behind_cards_back_notifies_instead_of_noop(
    notes_fixture,
):
    """Doc-honesty fix (final-review fix wave): a live card is reused in
    place across transcript syncs and never reloads its own notes, so a
    note delivered elsewhere while this card stays open still shows its
    stale ✕ button. `delete_change_note` correctly refuses to delete a
    delivered note (returns False), but a silent no-op there would look
    like a bug to the user -- pressing ✕ must now notify instead, and the
    row must survive untouched.
    """
    db, service, provider, root, run_id = notes_fixture

    async with _Host(lambda: provider, run_id).run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)

        note_input = await _open_note_input(pilot, body)
        note_input.value = "will be delivered behind this card's back"
        note_input.focus()
        await pilot.press("enter")

        delete_btn = None
        for _ in range(60):
            buttons = body.query(".console-turn-file-note-delete")
            if buttons:
                delete_btn = buttons.first()
                break
            await pilot.pause(0.02)
        assert delete_btn is not None, "delete button never rendered"
        note_id = db.notes_for_run(run_id)[0]["id"]

        # Delivered BEHIND the card's back -- the card is never told and
        # keeps rendering the pending ✕, exactly the documented live-card
        # scenario (Docs/User_Guide/console/agent-runs-and-tools.md).
        db.mark_notes_delivered([note_id])

        notify_calls: list[tuple[tuple, dict]] = []
        pilot.app.notify = lambda *a, **kw: notify_calls.append((a, kw))

        delete_btn.focus()
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert notify_calls, (
            "pressing delete on an already-delivered note must notify "
            "the user, not silently no-op"
        )
        message = notify_calls[0][0][0]
        assert "already sent" in message.lower()
        assert notify_calls[0][1].get("severity") == "warning"
        # Nothing was actually deleted -- the row (and its ✕) survives.
        assert body.query(".console-turn-file-note")
        assert body.query(".console-turn-file-note-delete")
        stored = db.notes_for_run(run_id)
        assert len(stored) == 1
        assert stored[0]["delivered_at"] is not None


@pytest.mark.asyncio
async def test_delivered_note_renders_sent_without_delete(notes_fixture):
    """A note stamped `delivered_at` (via `mark_notes_delivered` directly,
    per spec) renders a `sent` marker and carries no delete affordance --
    delivered notes are record.
    """
    db, service, provider, root, run_id = notes_fixture

    note_id = db.add_change_note(
        run_id=run_id,
        root=str(root),
        path="a.py",
        hunk_index=0,
        hunk_header="@@ -1,3 +1,3 @@",
        hunk_excerpt="-line2\n+CHANGED",
        note="already delivered feedback",
    )
    db.mark_notes_delivered([note_id])

    async with _Host(lambda: provider, run_id).run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)

        note_row = None
        for _ in range(60):
            rows = body.query(".console-turn-file-note")
            if rows:
                note_row = rows.first()
                break
            await pilot.pause(0.02)
        assert note_row is not None, "delivered note never rendered on expand"
        text = _note_text(note_row)
        assert "already delivered feedback" in text
        assert "sent" in text
        assert not body.query(".console-turn-file-note-delete")


@pytest.mark.asyncio
async def test_resume_round_trip_new_card_instance_shows_note(notes_fixture):
    """A brand-new card instance over the SAME DB shows a note written
    directly to it -- resume amnesia guard (spec §1: the anchor is stable
    across resume since snapshot rows are immutable).
    """
    db, service, provider, root, run_id = notes_fixture

    note_id = db.add_change_note(
        run_id=run_id,
        root=str(root),
        path="a.py",
        hunk_index=0,
        hunk_header="@@ -1,3 +1,3 @@",
        hunk_excerpt="-line2\n+CHANGED",
        note="pending across resume",
    )

    fresh_provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=CONV_ID
    )
    async with _Host(lambda: fresh_provider, run_id).run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)

        note_row = None
        for _ in range(60):
            rows = body.query(".console-turn-file-note")
            if rows:
                note_row = rows.first()
                break
            await pilot.pause(0.02)
        assert note_row is not None, "resumed note never rendered"
        assert "pending across resume" in _note_text(note_row)
        # Undelivered -- keeps its delete affordance.
        assert body.query(".console-turn-file-note-delete")

    assert db.notes_for_run(run_id)[0]["id"] == note_id


@pytest.mark.asyncio
async def test_note_input_survives_sync_tick_and_selection_move(notes_fixture):
    """Live-safety guard (spec §3, review finding #6): a note input with
    typed-but-unsaved text must survive a transcript sync tick (the SAME
    messages re-applied) and a selection move -- the final-review-wave
    ``_update_row_widget`` reuse branch protects card identity across
    both; this pins that the note input (a live descendant untouched by
    that method) rides along instead of being rebuilt out from under the
    user, exactly like an expanded diff already is (Task 3).
    """
    from Tests.UI.test_console_native_transcript import MutableTranscriptHarness
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    db, service, provider, root, run_id = notes_fixture

    card_message = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content=MARKER,
        id="m-card",
        change_review_run_id=run_id,
    )
    other_message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="hello", id="m-other"
    )

    app = MutableTranscriptHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_change_review_provider_factory(lambda: provider)
        transcript.set_messages([card_message, other_message])
        await transcript.refresh_messages()

        card = transcript.query_one(
            f"#console-turn-file-card-{card_message.id}", ConsoleTurnFileCard
        )
        for _ in range(60):
            if card.query(".console-turn-file-row"):
                break
            await pilot.pause(0.02)
        assert card.query(".console-turn-file-row"), "card rows never loaded"

        transcript.scroll_home(animate=False)
        await pilot.pause()
        # A CLICK, not focus+Enter: `ConsoleTranscript` is itself focusable
        # and owns its OWN "enter" binding ("confirm_selection", for
        # keyboard row navigation) -- inside a real transcript host,
        # `row.focus()` racing that container's own focus handling can
        # leave "enter" landing on the transcript rather than the row
        # `Button`, selecting the message instead of expanding the row.
        # A click always resolves to exactly the `Button.Pressed` this
        # card's own `on_button_pressed` expects, matching how the
        # existing header-click assertion in
        # `test_console_native_transcript.py` also uses `pilot.click`
        # rather than focus+Enter for interactions inside this same host.
        row = card.query(".console-turn-file-row").first()
        await pilot.click(row)
        body = None
        for _ in range(60):
            bodies = card.query(".console-turn-file-diff")
            if bodies and bodies.first().display:
                body = bodies.first()
                break
            await pilot.pause(0.02)
        assert body is not None, "diff body never displayed"
        # Same click-not-focus+Enter reasoning as the row expand above --
        # the note button is likewise a descendant of `ConsoleTranscript`.
        note_btn = body.query_one(".console-turn-file-note-btn", Button)
        await pilot.click(note_btn)
        note_input = None
        for _ in range(60):
            inputs = body.query(".console-turn-file-note-input")
            if inputs:
                note_input = inputs.first()
                break
            await pilot.pause(0.02)
        assert note_input is not None, "note input never opened"
        note_input.value = "half-typed feedback"
        note_input.focus()
        await pilot.pause()

        # Sync tick: the SAME messages re-applied.
        transcript.set_messages([card_message, other_message])
        await transcript.refresh_messages()
        card_after_sync = transcript.query_one(
            f"#console-turn-file-card-{card_message.id}", ConsoleTurnFileCard
        )
        assert card_after_sync is card, "sync tick rebuilt the card widget"
        surviving = card.query_one(".console-turn-file-note-input", Input)
        assert surviving is note_input, "sync tick rebuilt the note input"
        assert surviving.is_mounted
        assert surviving.value == "half-typed feedback"

        # Selection move onto the card row.
        transcript.selected_message_id = card_message.id
        await transcript.refresh_messages()
        card_after_select = transcript.query_one(
            f"#console-turn-file-card-{card_message.id}", ConsoleTurnFileCard
        )
        assert card_after_select is card, "selection move rebuilt the card widget"
        surviving = card.query_one(".console-turn-file-note-input", Input)
        assert surviving.is_mounted
        assert surviving.value == "half-typed feedback"

        # And a further move OFF the card row.
        transcript.selected_message_id = other_message.id
        await transcript.refresh_messages()
        card_after_deselect = transcript.query_one(
            f"#console-turn-file-card-{card_message.id}", ConsoleTurnFileCard
        )
        assert card_after_deselect is card, "deselection rebuilt the card widget"
        surviving = card.query_one(".console-turn-file-note-input", Input)
        assert surviving.is_mounted
        assert surviving.value == "half-typed feedback"


@pytest.mark.asyncio
async def test_note_button_second_press_does_not_double_mount_input(notes_fixture):
    """Pressing ``✎ note`` again while an input is already open on that
    hunk must NOT mount a second input -- the existing one is refocused
    (value intact) instead, matching ``_open_note_input``'s "already
    open" branch.
    """
    db, service, provider, root, run_id = notes_fixture

    async with _Host(lambda: provider, run_id).run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)

        note_input = await _open_note_input(pilot, body)
        note_input.value = "typed before second press"

        note_btn = body.query_one(".console-turn-file-note-btn", Button)
        note_btn.focus()
        await pilot.press("enter")
        await pilot.pause(0.1)

        inputs = list(body.query(".console-turn-file-note-input"))
        assert len(inputs) == 1, "a second press must not mount a second input"
        assert inputs[0] is note_input, "second press must reuse the same input"
        assert inputs[0].value == "typed before second press"
        assert pilot.app.focused is inputs[0], (
            "second press must refocus the existing input"
        )


@pytest.mark.asyncio
async def test_note_input_enter_submits_inside_real_transcript_not_select(
    notes_fixture,
):
    """The highest-risk interaction: a real Enter keypress INSIDE the note
    ``Input`` while it is mounted inside a REAL ``ConsoleTranscript`` host
    must submit the note -- not fall through to the transcript's own
    ``"enter" -> confirm_selection`` binding.

    Textual resolves the FOCUSED widget's own binding
    (``Input``'s built-in ``enter -> submit``) before walking up to any
    ancestor's BINDINGS, so this is expected to work by construction --
    but nothing pinned it before this test, and the row/note-button-open
    steps in this same file needed a click instead of focus+Enter
    specifically because of a DIFFERENT widget (``Button``) racing
    ``ConsoleTranscript``'s focus handling. This test proves the ``Input``
    itself is not subject to the same race: it polls for ``app.focused
    is note_input`` (never a fixed sleep) before pressing Enter, so a
    genuine focus-race would show up as this assertion timing out rather
    than as a silently-wrong pass.
    """
    from Tests.UI.test_console_native_transcript import MutableTranscriptHarness
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    db, service, provider, root, run_id = notes_fixture

    card_message = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content=MARKER,
        id="m-card",
        change_review_run_id=run_id,
    )
    other_message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="hello", id="m-other"
    )

    app = MutableTranscriptHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_change_review_provider_factory(lambda: provider)
        transcript.set_messages([card_message, other_message])
        await transcript.refresh_messages()

        card = transcript.query_one(
            f"#console-turn-file-card-{card_message.id}", ConsoleTurnFileCard
        )
        for _ in range(60):
            if card.query(".console-turn-file-row"):
                break
            await pilot.pause(0.02)
        assert card.query(".console-turn-file-row"), "card rows never loaded"

        transcript.scroll_home(animate=False)
        await pilot.pause()

        # Click (not focus+Enter) to expand the row and open the note
        # input -- the `Button` presses race `ConsoleTranscript`'s own
        # "enter" binding, per the earlier live-safety test's comment.
        # The interaction THIS test pins starts only once the `Input`
        # itself is confirmed focused, below.
        row = card.query(".console-turn-file-row").first()
        await pilot.click(row)
        body = None
        for _ in range(60):
            bodies = card.query(".console-turn-file-diff")
            if bodies and bodies.first().display:
                body = bodies.first()
                break
            await pilot.pause(0.02)
        assert body is not None, "diff body never displayed"

        note_btn = body.query_one(".console-turn-file-note-btn", Button)
        await pilot.click(note_btn)
        note_input = None
        for _ in range(60):
            inputs = body.query(".console-turn-file-note-input")
            if inputs:
                note_input = inputs.first()
                break
            await pilot.pause(0.02)
        assert note_input is not None, "note input never opened"

        # Wait until the Input has ACTUALLY gained focus -- not a fixed
        # pause racing the transition -- before typing/submitting.
        for _ in range(120):
            if pilot.app.focused is note_input:
                break
            await pilot.pause(0.02)
        assert pilot.app.focused is note_input, (
            "note input never actually gained focus after the click"
        )

        note_input.value = "enter must submit, not select"
        await pilot.press("enter")

        note_row = None
        for _ in range(60):
            rows = body.query(".console-turn-file-note")
            if rows:
                note_row = rows.first()
                break
            await pilot.pause(0.02)
        assert note_row is not None, (
            "Enter inside a focused note input must persist+render the "
            "note, not merely toggle transcript row selection"
        )
        assert "enter must submit, not select" in _note_text(note_row)
        assert not body.query(".console-turn-file-note-input"), (
            "input must be replaced by the note row"
        )

        # The transcript must NOT have treated this Enter as
        # confirm_selection on the card's row.
        assert not card.has_class("console-turn-file-card-selected"), (
            "Enter in the note input must not select the transcript row"
        )
        assert transcript.selected_message_id != card_message.id

        rows = db.notes_for_run(run_id)
        assert len(rows) == 1
        assert rows[0]["note"] == "enter must submit, not select"


@pytest.mark.asyncio
async def test_degrade_provider_add_change_note_raises_no_crash(notes_fixture):
    """A provider whose `add_change_note` raises must not crash the app:
    the input stays mounted (nothing lost) and a warning is logged --
    the card's absolute "no exception escapes an `on_*` handler" rule.
    """
    db, service, provider, root, run_id = notes_fixture

    class _RaisingAddNoteProvider(AgentRunsChangeReviewProvider):
        def add_change_note(self, **kwargs):
            raise RuntimeError("simulated DB write failure")

    raising_provider = _RaisingAddNoteProvider(
        db=db, service=service, conversation_id=CONV_ID
    )

    async with _Host(lambda: raising_provider, run_id).run_test(
        size=(120, 40)
    ) as pilot:
        card = await _settled_card(pilot)
        body = await _expand_first_row(pilot, card)

        note_input = await _open_note_input(pilot, body)
        note_input.value = "this will fail to save"
        note_input.focus()
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert pilot.app.is_running, (
            "a raising add_change_note must not crash the app"
        )
        assert card.is_mounted

        surviving = body.query(".console-turn-file-note-input")
        assert surviving, "input must stay mounted after a save failure"
        assert surviving.first().value == "this will fail to save"
        assert not body.query(".console-turn-file-note")
        assert db.notes_for_run(run_id) == []


@pytest.mark.asyncio
async def test_two_windows_same_root_path_note_scoped_to_its_own_hunk_header(
    tmp_path,
):
    """Regression (final-review fix wave): a run's ``change_snapshots`` can
    hold rows from TWO windows on the SAME root+path -- the turn's own
    window and its surviving sub-agents' post-turn window
    (``console_agent_bridge.py``'s ``_close_post_turn_change_window``,
    same real-provider two-window pattern as
    ``test_console_turn_file_card.py``'s
    ``test_real_provider_two_windows_on_same_root_no_duplicates_own_diffs``)
    -- each producing its OWN ``TurnFileEntry`` with its OWN diff, even
    when BOTH windows touch the exact same file.

    Pre-fix, ``_mount_hunk_blocks`` matched existing notes to a hunk by
    ``(root, path)`` + ``hunk_index`` alone: a note saved on one window's
    hunk 0 would ALSO render under the other window's hunk 0 of the same
    file -- wrong diff, wrong entry. The fix additionally requires
    ``note["hunk_header"] == hunk.header``, which the two windows' hunks
    never share (their diffs are against different baselines).

    Real stack (``ChangeTurnTracker``/``ShadowRepoService``/
    ``AgentRunsDB``/``AgentRunsChangeReviewProvider``) over a ``tmp_path``
    FILE-backed DB -- required for the same reason as the sibling
    two-window test: the card reads off ``asyncio.to_thread``, a
    different OS thread than the one that wrote the rows, and
    ``AgentRunsDB`` holds one connection PER THREAD.
    """
    from tldw_chatbook.Chat.console_agent_bridge import (
        CHANGE_KIND_SUBAGENT_POST_TURN,
        CHANGE_KIND_TURN,
    )

    root = tmp_path / "root"
    root.mkdir()
    lines = [f"line{i}\n" for i in range(1, 11)]
    (root / "shared.py").write_text("".join(lines))

    service = ShadowRepoService(data_dir=tmp_path / "appdata")
    tracker = ChangeTurnTracker(service=service)
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    run_id = db.create_run(conversation_id=CONV_ID, agent_kind="primary")

    def _record_window(kind: str, mutate) -> None:
        handle = tracker.begin_turn([root])
        handle.await_baseline()
        mutate()
        for rec in tracker.end_turn(handle):
            db.record_change_snapshot(
                run_id=run_id,
                root=rec.root,
                baseline_sha=rec.baseline_sha,
                end_sha=rec.end_sha,
                files_changed=rec.files_changed,
                adds=rec.adds,
                dels=rec.dels,
                tracking_error=rec.tracking_error,
                untracked_oversize=rec.untracked_oversize,
                nested_repos=rec.nested_repos,
                kind=kind,
            )

    def _mutate_turn() -> None:
        edited = lines[:]
        edited[1] = "line2-TURN\n"
        (root / "shared.py").write_text("".join(edited))

    def _mutate_post_turn() -> None:
        edited = lines[:]
        edited[1] = "line2-TURN\n"
        edited[8] = "line9-POST\n"
        (root / "shared.py").write_text("".join(edited))

    # Window 1: the turn's own window -- recorded first, matching
    # production's insertion order.
    _record_window(CHANGE_KIND_TURN, _mutate_turn)
    # Window 2: the post-turn window -- same root, same run_id, same
    # FILE, recorded second, against window 1's end state as its baseline.
    _record_window(CHANGE_KIND_SUBAGENT_POST_TURN, _mutate_post_turn)

    provider = AgentRunsChangeReviewProvider(
        db=db, service=service, conversation_id=CONV_ID
    )

    async with _Host(lambda: provider, run_id).run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        rows: list = []
        for _ in range(120):
            rows = list(card.query(".console-turn-file-row"))
            if len(rows) >= 2:
                break
            await pilot.pause(0.02)
        assert len(rows) == 2, "row count must equal one entry per window"
        labels = [str(row.render()) for row in rows]
        # Sanity: both entries really are the SAME file -- exactly the
        # shape that collides under the pre-fix (root, path)+hunk_index
        # filter.
        assert "shared.py" in labels[0], labels
        assert "shared.py" in labels[1], labels

        async def _expand(index: int):
            rows[index].focus()
            await pilot.press("enter")
            for _ in range(60):
                bodies = list(card.query(".console-turn-file-diff"))
                if bodies[index].display:
                    return bodies[index]
                await pilot.pause(0.02)
            raise AssertionError(f"diff body {index} never displayed")

        turn_body = await _expand(0)
        turn_hunk_text = str(turn_body.query_one(".console-turn-file-hunk").render())
        assert "line2-TURN" in turn_hunk_text
        assert "line9-POST" not in turn_hunk_text

        note_btn = turn_body.query_one(".console-turn-file-note-btn", Button)
        note_btn.focus()
        await pilot.press("enter")
        note_input = None
        for _ in range(60):
            inputs = turn_body.query(".console-turn-file-note-input")
            if inputs:
                note_input = inputs.first()
                break
            await pilot.pause(0.02)
        assert note_input is not None, "note input never opened on the turn window's hunk"
        note_input.value = "belongs to the TURN window's hunk only"
        note_input.focus()
        await pilot.press("enter")
        for _ in range(60):
            if turn_body.query(".console-turn-file-note"):
                break
            await pilot.pause(0.02)
        assert turn_body.query(".console-turn-file-note"), (
            "note never rendered on the turn window's own entry"
        )

        post_turn_body = await _expand(1)
        post_turn_hunk_text = str(
            post_turn_body.query_one(".console-turn-file-hunk").render()
        )
        assert "line9-POST" in post_turn_hunk_text
        assert "line2-TURN" not in post_turn_hunk_text
        # Sanity: the two windows' hunk headers for this same file really
        # do differ -- confirming this fixture exercises the collision
        # shape and isn't accidentally passing because both hunks happen
        # to share one header.
        assert "@@" in turn_hunk_text and "@@" in post_turn_hunk_text
        turn_header_line = next(
            line for line in turn_hunk_text.splitlines() if line.startswith("@@")
        )
        post_turn_header_line = next(
            line for line in post_turn_hunk_text.splitlines() if line.startswith("@@")
        )
        assert turn_header_line != post_turn_header_line

        await pilot.pause(0.1)
        post_turn_notes = list(post_turn_body.query(".console-turn-file-note"))
        assert post_turn_notes == [], (
            "a note saved on the turn window's hunk must not bleed into "
            "the post-turn window's same-index hunk of the same file"
        )

        # Persisted exactly once, anchored to the turn window's own hunk
        # header.
        stored = db.notes_for_run(run_id)
        assert len(stored) == 1
        assert stored[0]["path"] == "shared.py"
        assert stored[0]["hunk_index"] == 0


@pytest.mark.asyncio
async def test_note_input_swallows_up_down_arrow_keys_no_selection_move(
    notes_fixture,
):
    """Regression (final-review fix wave): the card's ``on_key`` reclaimed
    only enter/escape from a focused note input's ancestors; up/down
    bubbled past it to ``ConsoleTranscript.on_key``, which moves ROW
    SELECTION on those keys -- so a user typing a note who happened to
    press an arrow key would silently move the transcript's selected row
    mid-edit. An ``Input`` has no cursor-navigation use for either key on
    a single-line field, so both must be pure no-op reclaims here.

    Driven inside a REAL ``ConsoleTranscript`` host (not the standalone
    ``_Host``) -- the bug is specifically about the bubble race with that
    ancestor's own ``on_key``, matching this file's other
    real-transcript-host tests
    (``test_note_input_survives_sync_tick_and_selection_move``,
    ``test_note_input_enter_submits_inside_real_transcript_not_select``).
    """
    from Tests.UI.test_console_native_transcript import MutableTranscriptHarness
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleChatMessage,
        ConsoleMessageRole,
    )
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    db, service, provider, root, run_id = notes_fixture

    card_message = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content=MARKER,
        id="m-card",
        change_review_run_id=run_id,
    )
    other_message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="hello", id="m-other"
    )

    app = MutableTranscriptHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.set_change_review_provider_factory(lambda: provider)
        transcript.set_messages([card_message, other_message])
        await transcript.refresh_messages()

        card = transcript.query_one(
            f"#console-turn-file-card-{card_message.id}", ConsoleTurnFileCard
        )
        for _ in range(60):
            if card.query(".console-turn-file-row"):
                break
            await pilot.pause(0.02)
        assert card.query(".console-turn-file-row"), "card rows never loaded"

        transcript.scroll_home(animate=False)
        await pilot.pause()

        # Click, not focus+Enter -- `ConsoleTranscript` owns its own
        # "enter" binding and can race a `Button.focus()` (same reasoning
        # as this file's other real-transcript tests).
        row = card.query(".console-turn-file-row").first()
        await pilot.click(row)
        body = None
        for _ in range(60):
            bodies = card.query(".console-turn-file-diff")
            if bodies and bodies.first().display:
                body = bodies.first()
                break
            await pilot.pause(0.02)
        assert body is not None, "diff body never displayed"

        note_btn = body.query_one(".console-turn-file-note-btn", Button)
        await pilot.click(note_btn)
        note_input = None
        for _ in range(60):
            inputs = body.query(".console-turn-file-note-input")
            if inputs:
                note_input = inputs.first()
                break
            await pilot.pause(0.02)
        assert note_input is not None, "note input never opened"

        for _ in range(120):
            if pilot.app.focused is note_input:
                break
            await pilot.pause(0.02)
        assert pilot.app.focused is note_input, (
            "note input never actually gained focus after the click"
        )

        note_input.value = "typed before arrow keys"
        assert transcript.selected_message_id is None, (
            "baseline: nothing selected before the arrow-key presses"
        )

        await pilot.press("down")
        await pilot.pause()
        assert transcript.selected_message_id is None, (
            "down inside a focused note input must not move transcript "
            "row selection"
        )
        assert pilot.app.focused is note_input, (
            "down must not move focus off the note input"
        )
        assert note_input.value == "typed before arrow keys"

        await pilot.press("up")
        await pilot.pause()
        assert transcript.selected_message_id is None, (
            "up inside a focused note input must not move transcript row "
            "selection"
        )
        assert pilot.app.focused is note_input, (
            "up must not move focus off the note input"
        )
        assert note_input.value == "typed before arrow keys"
