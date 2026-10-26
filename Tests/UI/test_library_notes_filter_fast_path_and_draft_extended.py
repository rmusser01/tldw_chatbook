"""More arms of the Notes filter fast path, a PR-lane census file of its own
(it boots four times; ``test_library_notes_filter_fast_path_and_draft.py``
holds the first four arms).

- TASK-34000.28 (N-21) fix round 1, M6: following ``‹ <origin>`` back to an
  origin deleted meanwhile warns, returns to the list and writes nothing.
- TASK-34000.40 (N-31) ruling T5-b: a sibling list filter (Prompts) takes the
  same "no after esc: keys when Escape lands in another text field" rule
  from the shared footer transform.
- Fix round 2, R2-2/R2-3: a pane key during a held Down hand-off withdraws it
  -- F6 (a key ``on_key`` sees) and Shift+F6 (a priority binding it does not)
  -- and the landing leaves focus on the user's own widget; so do a dialog
  over Library and, boot-free, the priority Tab pair and the wheel.
- Fix round 2, R2-4: with nothing applied, Down from the empty box lands on
  the browse tree's first row (a folder here), exactly as after a clear; an
  empty tree leaves focus in the box.
- PR #3259 CI fix round 1: ``/`` pressed right after Back from a note, while
  the list's source snapshot is still loading, keeps the filter -- Back's
  "focus the row you came from" hand-off, which waits for that snapshot,
  yields to the user's own move instead of landing over it; and Down's
  landing on the new result no longer loses a race to the box's own queued
  focus event.

Same harness as the first file: the real ``LibraryScreen`` with the
production stylesheet over a real ChaChaNotes database.
"""

from __future__ import annotations

import asyncio
from functools import partial
from types import SimpleNamespace

import pytest
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _armed_editor,
    _open_notes,
    _type,
    _until,
)
from Tests.UI.test_library_notes_filter_fast_path_and_draft import (
    INDEX,
    RECIPES,
    THESIS,
    _first_tree_row,
    _focused_id,
    _footer_withholds_after_esc,
    _box,
    _HeldSearch,
    _list,
    _mounted,
    _notes_ready,
    _painted_in,
    _painted_of,
    _Profile,
    _row_titled,
    _shown,
    _slash_to_filter,
    _status,
    _type_into,
    _wait,
)
from Tests.UI.test_library_shell import (
    LibraryGlobalKeyProductionCSSHarness,
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _notes_rows(db) -> list[tuple]:
    return sorted(
        (row["id"], row["version"], row["deleted"])
        for row in db.execute_query("SELECT id, version, deleted FROM notes").fetchall()
    )


async def test_the_way_back_to_a_deleted_origin_warns_and_writes_nothing(
    tmp_path, monkeypatch
) -> None:
    """M6: Index -> its note link -> Recipes shows ``‹ Index``; Index is then
    deleted elsewhere. Pressing ``‹`` warns, returns to the list, writes no
    notes row, and leaves focus on a live control a real key moves on from."""
    host, seeded = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        _row_titled(screen, INDEX).press()
        await _armed_editor(screen, pilot, seeded.index_id)
        screen.query_one("#library-note-preview", Button).press()
        await _wait(
            pilot,
            lambda: _shown(screen, "#library-note-preview-region")
            and RECIPES in _painted_of(host, screen, "#library-note-preview-body"),
            "Preview to paint the link",
        )
        await pilot.click("#library-note-preview-region")
        await _wait(
            pilot,
            lambda: _focused_id(screen) == "library-note-preview-region",
            "the Preview region to hold focus",
        )
        await pilot.press("enter")
        await _armed_editor(screen, pilot, seeded.recipes_id)
        await _wait(
            pilot,
            lambda: bool(screen.query("#library-note-link-back"))
            and f"‹ {INDEX}" in _painted_in(host, screen.query("#library-note-link-back").first()),
            "'‹ Index — start here' on Recipes",
        )

        toasts: list[str] = []
        app = host.app_instance
        real_notify = app.notify

        def _recording_notify(message, *args, **kwargs):
            toasts.append(str(message))
            return real_notify(message, *args, **kwargs)

        monkeypatch.setattr(app, "notify", _recording_notify)
        db = seeded.profile.db
        index = db.get_note_by_id(seeded.index_id)
        assert db.soft_delete_note(seeded.index_id, index["version"]) is True
        before = _notes_rows(db)

        await pilot.click("#library-note-link-back")
        await _wait(
            pilot,
            lambda: "That note is no longer available." in toasts
            and screen._notes_state.view == "list",
            lambda: f"the missing-origin warning and the list (toasts {toasts!r})",
        )
        await pilot.pause(0.3)
        assert _notes_rows(db) == before, "following the way back wrote to the notes table"
        assert screen.focused is not None and screen.focused.is_attached

        # A real key still works: "/" reaches the filter.
        await _slash_to_filter(screen, pilot)


class _HeldSnapshot:
    """The screen's real source-snapshot read, held until ``release`` is set,
    so a key can land while Back's list refresh is still in flight (the
    window a slow CI runner widens: PR #3259 shard 9)."""

    def __init__(self, real) -> None:
        self._real = real
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.answered = asyncio.Event()

    async def __call__(self):
        self.entered.set()
        await self.release.wait()
        snapshot = await self._real()
        # The worker applies the answer and schedules Back's hand-off in the
        # same step it resumes in, before the test's next turn.
        self.answered.set()
        return snapshot


async def _after_the_snapshot_lands(screen, pilot, held: _HeldSnapshot) -> None:
    """Release the held read; return once everything its answer scheduled ran.

    The answer schedules Back's hand-off with ``call_after_refresh`` -- at
    once when it changed nothing on screen, or from the entry reconcile it
    always queues (``_complete_library_entry_reconcile``) -- so the wait is
    for that reconcile to finish, then for a marker scheduled the same way
    after it (one FIFO queue): a hand-off that would land has landed.
    """
    held.release.set()
    await _wait(pilot, held.answered.is_set, "the held snapshot to answer")
    await _wait(
        pilot,
        lambda: screen._library_entry_reconcile_pending is None,
        "the snapshot's entry reconcile to finish",
    )
    landed = asyncio.Event()
    screen.call_after_refresh(landed.set)
    await _wait(pilot, landed.is_set, "one refresh after the reconcile")


async def _back_with_the_snapshot_held(screen, pilot, monkeypatch, real) -> _HeldSnapshot:
    """Escape from the open note with its list refresh held in flight."""
    held = _HeldSnapshot(real)
    monkeypatch.setattr(screen, "_list_local_source_snapshot", held)
    await pilot.press("escape")
    await _wait(pilot, held.entered.is_set, "Back's list snapshot refresh")
    await _wait(
        pilot,
        lambda: screen._notes_state.view == "list" and _box(screen) is not None,
        "Back to the list",
    )
    assert screen._notes_state.pending_focus_waits_for_snapshot, (
        "sanity: Back's hand-off should still be waiting for the snapshot"
    )
    return held


async def test_slash_right_after_back_keeps_the_filter_when_the_list_snapshot_lands(
    tmp_path, monkeypatch
) -> None:
    """Back from a note, then ``/`` before the list's snapshot refresh lands.

    Back arms a hand-off -- focus the row the note was opened from -- that
    waits for the source snapshot its exit refetches. The user's own move in
    the meantime must win: when the held snapshot is released, focus stays in
    the filter and the next keys type there. Every focus write after ``/`` is
    recorded, so "no row took focus" is a counted claim. Control, same boot:
    with no move, the released hand-off still focuses the row (a recorded
    write: a canvas sync already in flight can restore its own older capture
    over it afterwards, a separate race measured on the base too -- 5 of 16).

    Base (``ddb9693e3b``, and dev ``38623632e9``): the released snapshot's
    hand-off focused the note row over the filter -- the shard-9 red of
    ``test_down_with_an_unsubmitted_draft_applies_it_and_lands_on_the_new_result``.
    """
    host, seeded = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        real = screen._list_local_source_snapshot

        focus_writes: list[object] = []
        real_set_focus = screen.set_focus

        def _recording_set_focus(widget, *args, **kwargs):
            focus_writes.append(widget)
            return real_set_focus(widget, *args, **kwargs)

        monkeypatch.setattr(screen, "set_focus", _recording_set_focus)

        def _note_rows_focused() -> list[object]:
            return [
                target
                for target in focus_writes
                if getattr(target, "has_class", lambda _c: False)("library-notes-row")
            ]

        # Control: nothing moved, so the released hand-off focuses Thesis.
        _row_titled(screen, THESIS).press()
        await _armed_editor(screen, pilot, seeded.thesis_id)
        held = await _back_with_the_snapshot_held(screen, pilot, monkeypatch, real)
        focus_writes.clear()
        await _after_the_snapshot_lands(screen, pilot, held)
        await _wait(
            pilot,
            lambda: any(
                getattr(row, "note_id", "") == seeded.thesis_id
                for row in _note_rows_focused()
            ),
            lambda: f"Back's hand-off to the Thesis row (focus {_focused_id(screen)!r})",
        )

        # The user moves first: ``/`` before the snapshot lands.
        await _wait(
            pilot,
            lambda: (row := _row_titled(screen, THESIS)) is not None
            and row.is_attached
            and screen.focused is not None,
            "the list to settle with the Thesis row mounted",
        )
        _row_titled(screen, THESIS).press()
        await _armed_editor(screen, pilot, seeded.thesis_id)
        held = await _back_with_the_snapshot_held(screen, pilot, monkeypatch, real)
        await _slash_to_filter(screen, pilot)
        focus_writes.clear()
        await _after_the_snapshot_lands(screen, pilot, held)
        # A recompose in flight can park focus on nothing for a frame, so the
        # filter is waited for; the recorder catches a row that took focus
        # even for one frame, through the typing below.
        await _wait(
            pilot,
            lambda: _focused_id(screen) == "library-notes-filter",
            lambda: (
                "focus to stay in the filter once the snapshot landed "
                f"(focus {_focused_id(screen)!r})"
            ),
        )
        # A real key still works: the next keys type into the filter.
        await _type_into(screen, pilot, "rec", "rec")
        rows = [getattr(row, "id", row) for row in _note_rows_focused()]
        assert rows == [], f"a note row took focus after `/`: {rows}"


async def test_down_lands_on_the_result_even_when_the_box_focus_event_is_handled_first(
    tmp_path, monkeypatch
) -> None:
    """``/``, ``thes``, Down: the landing on the new result's first row must
    not depend on winning a race (PR #3259 CI fix round 1).

    The landing runs in the painted result's canvas-sync ``then``, right
    after that sync restored focus to the box. Its list-entry arm tried only
    after the next refresh, and when the box's queued DescendantFocus was
    handled first -- a slow runner's order -- it read as the user leaving the
    list and disarmed the landing: focus stayed in the box (1 of 10 runs of
    ``test_down_with_an_unsubmitted_draft_applies_it_and_lands_on_the_new_result``
    under CPU load). Here the arm's after-refresh attempt is held until that
    event has been handled, which makes the slow order certain.

    Base (``ddb9693e3b``): focus stays in the filter.
    """
    host, seeded = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        await _slash_to_filter(screen, pilot)
        await _type_into(screen, pilot, "thes", "thes")

        sequence: list[tuple[str, str]] = []
        held_attempts: list[int] = []
        real_attempt = screen._focus_library_list_entry_if_current
        real_arm = LibraryScreen._arm_library_list_entry_focus
        real_focus_event = LibraryScreen.on_descendant_focus

        def _arm(self, *args, **kwargs):
            sequence.append(("arm", ""))
            return real_arm(self, *args, **kwargs)

        def _focus_event(self, event):
            sequence.append(("focus-event", str(getattr(event.widget, "id", "") or "")))
            return real_focus_event(self, event)

        monkeypatch.setattr(LibraryScreen, "_arm_library_list_entry_focus", _arm)
        monkeypatch.setattr(LibraryScreen, "on_descendant_focus", _focus_event)
        monkeypatch.setattr(
            screen, "_focus_library_list_entry_if_current", held_attempts.append
        )
        await pilot.press("down")

        def _box_event_after_the_arm() -> bool:
            if ("arm", "") not in sequence:
                return False
            after = sequence[sequence.index(("arm", "")) + 1 :]
            return ("focus-event", "library-notes-filter") in after

        await _wait(
            pilot,
            lambda: bool(held_attempts) and _box_event_after_the_arm(),
            lambda: f"the landing's arm and the box's queued focus event ({sequence})",
        )
        for generation in list(held_attempts):
            real_attempt(generation)

        def _landed() -> bool:
            focused = screen.focused
            return bool(
                screen._notes_state.filter == "thes"
                and focused is not None
                and focused.is_attached
                and focused.has_class("library-notes-row")
                and getattr(focused, "note_id", "") == seeded.thesis_id
                and "filter: thes · 1 result" in _status(host, screen)
            )

        await _wait(
            pilot,
            _landed,
            lambda: (
                f"Down to land on the thes result (focus {_focused_id(screen)!r}, "
                f"status {_status(host, screen)!r})"
            ),
        )


async def test_a_sibling_list_filter_withholds_after_esc_keys_too(tmp_path) -> None:
    """Ruling T5-b through the shared transform: Escape from the Prompts filter
    lands in the rail's search box as well, so its typing footer names no
    "after esc:" keys (it read ``after esc: / focus search`` before)."""
    host, _seeded = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        screen.query_one("#library-row-browse-prompts", Button).press()
        await _wait(
            pilot,
            lambda: _shown(screen, "#library-prompts-filter"),
            "the Prompts filter",
        )
        # The Prompts canvas recomposes once its (empty) browse answers; a
        # click landing on the box that recompose replaces is lost (the
        # TASK-34000.85 shape), so click again until the box holds focus.
        for _ in range(5):
            await pilot.click("#library-prompts-filter")
            try:
                await _until(
                    pilot,
                    lambda: _focused_id(screen) == "library-prompts-filter",
                    "",
                    timeout=1.0,
                )
                break
            except AssertionError:
                continue
        assert _focused_id(screen) == "library-prompts-filter", _focused_id(screen)
        await _footer_withholds_after_esc(host, screen, pilot, what="Prompts filter")


async def test_a_pane_key_during_a_pending_down_keeps_the_users_focus(
    tmp_path, monkeypatch
) -> None:
    """R2-2, then R2-3: Down over a draft with the search held, then a pane key.

    Shift+F6 (a priority binding, which never reaches ``on_key``) moves focus
    to the rail, and the landing must not pull it onto a result row; nor may
    a landing behind a dialog (``on_screen_suspend``). With a
    note open, F6 (the app's global binding, which ``on_key`` sees) moves
    focus to the note's Save -- a control with a portable focus role -- and
    after the landing focus is on that same control (by its semantic
    identity: the landing's sync rebuilds the pane and replays it).
    """
    app = _build_test_app()
    _seed_conversations(app, [])
    seeded = _Profile(tmp_path)
    app.chachanotes_db = seeded.profile.db
    app.notes_scope_service = seeded.profile.scope_service
    app.notes_service = seeded.profile.interop
    host = LibraryGlobalKeyProductionCSSHarness(app, screen=LibraryScreen(app))
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)

        async def _held_down(text: str):
            await pilot.click("#library-notes-filter")
            await _wait(
                pilot,
                lambda: _focused_id(screen) == "library-notes-filter",
                "a click into the filter",
            )
            await pilot.press("end", *["backspace"] * 8)
            await _type_into(screen, pilot, text, text)
            held = _HeldSearch(seeded.profile.scope_service)
            monkeypatch.setattr(
                seeded.profile.scope_service, "search_note_tree_placements", held
            )
            await pilot.press("down")
            await _wait(pilot, held.entered.is_set, f"the held {text!r} search")
            return held

        # R2-2: Shift+F6, a priority binding that never reaches on_key, to the
        # rail; the landing must not pull focus onto a result row.
        held = await _held_down("thes")
        await pilot.press("shift+f6")
        await _wait(
            pilot,
            lambda: screen.focused is not None
            and screen._library_notes_focus_stage(screen.focused) == "rail",
            lambda: f"Shift+F6 to the rail (focus {_focused_id(screen)!r})",
        )
        held.release.set()
        await _wait(
            pilot,
            lambda: "filter: thes · 1 result" in _status(host, screen),
            lambda: f"the released 'thes' result (status {_status(host, screen)!r})",
        )
        await pilot.pause(0.3)
        focused = screen.focused
        assert focused is not None and not focused.has_class("library-notes-row"), (
            _focused_id(screen)
        )
        assert screen._library_notes_focus_stage(focused) == "rail", _focused_id(screen)

        # R2-2: a dialog over Library (Ctrl+Q's is a priority binding too)
        # withdraws it as well -- the landing must not move focus behind it.
        held = await _held_down("outl")
        await host.push_screen(_Dialog())
        await _wait(pilot, lambda: isinstance(host.screen, _Dialog), "the dialog")
        held.release.set()
        await _wait(
            pilot,
            lambda: screen._notes_state.filter == "outl"
            and screen._library_notes_filter_row_handoff is None,
            lambda: "the 'outl' landing behind the dialog (hand-off "
            f"{screen._library_notes_filter_row_handoff!r})",
        )
        await pilot.pause(0.3)
        host.pop_screen()
        await _wait(pilot, lambda: host.screen is screen, "the return from the dialog")
        await pilot.pause(0.5)
        focused = screen.focused
        assert focused is None or not focused.has_class("library-notes-row"), (
            _focused_id(screen)
        )

        # R2-3: with a note open, F6 to its Save -- a control with a portable
        # role -- and after the landing focus is on that same control.
        _row_titled(screen, THESIS).press()
        await _armed_editor(screen, pilot, seeded.thesis_id)
        held = await _held_down("index")
        await pilot.press("f6")
        await _wait(
            pilot,
            lambda: _focused_id(screen) == "library-note-save",
            lambda: f"F6 to the note's Save (focus {_focused_id(screen)!r})",
        )
        role = screen._library_notes_semantic_role(screen.focused)
        assert role == "save", role
        held.release.set()
        await _wait(
            pilot,
            lambda: "filter: index · 2 results" in _status(host, screen),
            lambda: f"the released 'index' result (status {_status(host, screen)!r})",
        )
        await pilot.pause(0.3)
        assert _focused_id(screen) == "library-note-save", _focused_id(screen)
        assert screen._library_notes_semantic_role(screen.focused) == role


class _Dialog(ModalScreen[None]):
    """A stand-in for any dialog the app opens over Library."""

    def compose(self) -> ComposeResult:
        yield Button("OK")


class _FolderProfile(_Profile):
    """The first file's profile plus one folder, so the tree's first row is a
    folder row above every note."""

    def __init__(self, tmp_path) -> None:
        super().__init__(tmp_path)
        repository = LocalNoteFolderRepository(self.profile.db)
        self.journal_id = repository.create_folder(
            name="Journal", parent_id=None
        ).folder_id
        repository.attach_manual(folder_id=self.journal_id, note_id=self.thesis_id)


async def test_down_from_an_empty_box_lands_on_the_first_tree_row_applied_or_not(
    tmp_path,
) -> None:
    """R2-4 (ruling T5-a, one rule): Down from an empty box -- never applied,
    or just cleared -- lands on the browse tree's first row, here the
    ``▸ Journal`` folder row above every note."""
    app = _build_test_app()
    _seed_conversations(app, [])
    seeded = _FolderProfile(tmp_path)
    app.chachanotes_db = seeded.profile.db
    app.notes_scope_service = seeded.profile.scope_service
    app.notes_service = seeded.profile.interop
    host = LibraryProductionCSSHarness(app, screen=LibraryScreen(app))
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes(screen, pilot)
        await _wait(
            pilot,
            lambda: INDEX in _list(host, screen) and "Journal" in _list(host, screen),
            "the Journal folder row and the unfiled notes",
        )

        def _on_the_journal_row() -> bool:
            first = _first_tree_row(screen)
            return bool(
                first is not None
                and first.has_class("library-notes-folder-row")
                and "Journal" in str(first.label)
                and screen.focused is first
            )

        # Never applied: the box is empty, nothing is filtered.
        await _slash_to_filter(screen, pilot)
        assert screen._notes_state.filter == ""
        await pilot.press("down")
        await _wait(
            pilot,
            _on_the_journal_row,
            lambda: (
                f"Down from the empty box to Journal (focus {_focused_id(screen)!r})"
            ),
        )

        # Applied, then emptied: the clear lands on the same row.
        await _slash_to_filter(screen, pilot)
        await _type(pilot, "index")
        await pilot.press("enter")
        await _wait(
            pilot,
            lambda: "filter: index · 2 results" in _status(host, screen),
            lambda: f"the index result (status {_status(host, screen)!r})",
        )
        await _slash_to_filter(screen, pilot)
        await pilot.press("end", *["backspace"] * 5)
        await pilot.press("down")
        await _wait(
            pilot,
            lambda: screen._notes_state.filter == ""
            and INDEX in _list(host, screen)
            and _on_the_journal_row(),
            lambda: (
                f"Down over the emptied box to Journal "
                f"(focus {_focused_id(screen)!r})"
            ),
        )


async def test_down_from_an_empty_box_over_an_empty_tree_leaves_focus_in_the_box() -> (
    None
):
    """R2-4, empty tree (screen-level): no row to land on, so the key is not
    claimed and focus stays where it is, in the box."""
    box = SimpleNamespace(value="")
    screen = SimpleNamespace(
        query_one=lambda *_a, **_k: box,
        _safe_text=lambda value, max_length: value[:max_length],
        _notes_state=SimpleNamespace(filter=""),
        _library_notes_filter_loading=lambda: False,
        _first_library_notes_tree_row=lambda: None,
        _first_library_notes_note_row=lambda: pytest.fail(
            "an empty box asked for a note"
        ),
    )
    assert LibraryScreen._library_notes_filter_down(screen) is False


@pytest.mark.parametrize(
    ("site", "event"),
    [
        ("action_focus_next", None),
        ("action_focus_previous", None),
        ("on_mouse_scroll_down", SimpleNamespace(widget=None)),
        ("on_mouse_scroll_up", SimpleNamespace(widget=None)),
    ],
)
async def test_input_on_key_never_sees_withdraws_a_pending_down(site, event) -> None:
    """R2-2 (screen-level): the note fields' PRIORITY Tab/Shift+Tab reach the
    screen's focus actions without passing ``on_key``, and the wheel has its
    own handlers -- each withdraws a pending Down's move to the first row,
    and (PR #3259 CI fix round 1) Back's pending hand-off to its row."""
    screen = SimpleNamespace(
        _library_notes_filter_row_handoff=("thes", True),
        _notes_state=SimpleNamespace(
            pending_focus_identity=object(),
            pending_focus_waits_for_snapshot=True,
            pending_focus_generation=7,
        ),
        _library_pending_list_entry_focus=False,
        _mark_library_notes_user_interaction=lambda: None,
        _advance_library_ordinary_emergency_user_interaction=lambda _w: None,
        _move_library_screen_focus=lambda _d: None,
    )
    screen._withdraw_library_notes_pending_focus_handoffs = partial(
        LibraryScreen._withdraw_library_notes_pending_focus_handoffs, screen
    )
    handler = getattr(LibraryScreen, site)
    handler(screen) if event is None else handler(screen, event)
    assert screen._library_notes_filter_row_handoff == ("thes", False)
    assert screen._notes_state.pending_focus_identity is None
    assert screen._notes_state.pending_focus_waits_for_snapshot is False
    assert screen._notes_state.pending_focus_generation is None
