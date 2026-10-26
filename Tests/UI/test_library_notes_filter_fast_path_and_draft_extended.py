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

Same harness as the first file: the real ``LibraryScreen`` with the
production stylesheet over a real ChaChaNotes database.
"""

from __future__ import annotations

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
    own handlers -- each withdraws a pending Down's move to the first row."""
    screen = SimpleNamespace(
        _library_notes_filter_row_handoff=("thes", True),
        _library_pending_list_entry_focus=False,
        _mark_library_notes_user_interaction=lambda: None,
        _advance_library_ordinary_emergency_user_interaction=lambda _w: None,
        _move_library_screen_focus=lambda _d: None,
    )
    screen._withdraw_library_notes_filter_row_handoff = partial(
        LibraryScreen._withdraw_library_notes_filter_row_handoff, screen
    )
    handler = getattr(LibraryScreen, site)
    handler(screen) if event is None else handler(screen, event)
    assert screen._library_notes_filter_row_handoff == ("thes", False)
