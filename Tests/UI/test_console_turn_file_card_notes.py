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
