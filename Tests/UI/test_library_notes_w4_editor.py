"""Library ▸ Notes critique #3, wave 4 -- ``editor`` group.

Tasks 32537, 32539, 32548, 32550, 32551, 32556: focus, identity, keys and
Preview inside the note editor. See ``backlog/tasks/task-<id>*.md`` for the
acceptance criteria; each test is named after the task it pins.

Every assertion below reads production output -- the real
``_library_notes_footer_shortcuts`` tier, the real mounted widgets, the real
tree projection -- never a value the test just handed the code.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Widgets.Library.library_notes_canvas import render_preview_source


#: Two notes that share a title, with ids whose first four characters differ
#: -- the short-id fall-back ``note_row_tiebreak_labels`` uses when two rows
#: also share a folder, an age and a clock (task-32254).
_DUPLICATE_NOTES = [
    {
        "id": "a1b2-reading-one",
        "title": "Reading list",
        "content": "- Deep work\n",
        "version": 1,
    },
    {
        "id": "c3d4-reading-two",
        "title": "Reading list",
        "content": "# Reading list (second copy)\n\n- A Pattern Language\n",
        "version": 1,
    },
]


def _build_notes_host(notes=None) -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=notes
        if notes is not None
        else [
            {
                "id": "note-1",
                "title": "Research Note",
                "content": "Body text\n",
                "version": 1,
            }
        ],
    )
    return LibraryHarness(app)


async def _open_notes_list(screen, pilot):
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    await pilot.pause()


def _note_rows(screen) -> list[Button]:
    rows = list(screen.query(".library-notes-row"))
    assert rows, "No note row Button found"
    return rows


async def _open_note(screen, pilot, index: int = 0):
    _note_rows(screen)[index].press()
    await _wait_for_selector(screen, pilot, "#library-note-body")
    await pilot.pause()


async def _open_first_note(screen, pilot):
    await _open_notes_list(screen, pilot)
    await _open_note(screen, pilot, 0)


async def _open_preview(screen, pilot):
    screen.query_one("#library-note-preview", Button).press()
    await pilot.pause()
    await pilot.pause()


def _chips(screen) -> dict[str, str]:
    """The real footer tier for the visible Notes state, as a mapping."""
    return dict(screen._library_notes_footer_shortcuts())


# --- task-32537: the Preview tier names every Tab stop ---------------------


#: The Enter action of each control Tab reaches from inside Preview, in Tab
#: order. The labels are production's own
#: (``_LIBRARY_NOTE_EDITOR_ENTER_LABELS``); this test asserts the FOOTER
#: reaches them, which before the fix it never did.
_PREVIEW_TAB_STOPS = (
    ("library-note-back", "back to list"),
    ("library-note-edit", "edit note"),
    ("library-note-preview", "preview note"),
    ("library-note-context", "show info"),
    ("library-note-save", "save note"),
    ("library-note-use-in-console", "use in Console"),
)


@pytest.mark.asyncio
async def test_preview_tier_names_every_tab_stop():
    """AC#1/#3: every Tab stop in Preview shows its "enter …" chip, and
    pgup/pgdn stays advertised beside it.

    Live at dev 2f97a42c9a (editor-00-32537-preview-tab6-footer-frozen):
    Tab x6 moved the heavy focus box across ‹ Notes, Edit, Preview, Info,
    Save and Use in Console while the footer read "pgup/pgdn scroll | esc
    back to notes" on every one of them -- so the sixth blind Enter fired
    Use in Console with nothing having said so.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_preview(screen, pilot)

        assert screen._library_notes_focus_region() == "preview"

        for widget_id, expected in _PREVIEW_TAB_STOPS:
            screen.query_one(f"#{widget_id}", Button).focus()
            await pilot.pause()
            chips = _chips(screen)
            assert chips.get("enter") == expected, (
                f"Preview footer with #{widget_id} focused: {chips}"
            )
            assert chips.get("pgup/pgdn") == "scroll", chips


@pytest.mark.asyncio
async def test_preview_footer_advertises_escape_to_the_list():
    """AC#2 (doc branch): Escape from Preview goes to the LIST, and the
    footer says "esc back to notes" rather than promising a step back to
    Edit that does not happen.

    Verified live at dev 2f97a42c9a: Escape from Preview left the editor
    entirely. The guide (notes.md "Escape" row) already describes that; this
    pins the behaviour and the chip together so the two cannot drift apart.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_preview(screen, pilot)

        assert _chips(screen).get("esc") == "back to notes"

        screen.query_one("#library-note-preview-region").focus()
        await pilot.pause()
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.view == "list",
            message=lambda: (
                "Escape from Preview left the view at "
                f"{screen._notes_state.view!r}"
            ),
        )


# --- task-32539: focus after a confirmed delete ----------------------------


async def _confirm_delete_of_the_open_note(screen, pilot):
    """Info ▸ Delete ▸ confirm, through the real buttons."""
    screen.query_one("#library-note-context", Button).press()
    await pilot.pause()
    screen.query_one("#library-note-context-delete", Button).press()
    await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")
    screen.query_one("#library-note-delete-confirm", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-delete-undo")
    await pilot.pause()


@pytest.mark.asyncio
async def test_focus_lands_on_undo_after_a_confirmed_delete():
    """AC#1/#3: focus parks on the receipt's Undo, the button carries the
    shape-based action class, and the list footer names it.

    Live at dev 2f97a42c9a (editor-00-32539-post-delete-no-focus): after the
    confirmation the receipt appeared with Undo / Dismiss and NOTHING was
    focused -- the next Tab restarted at the toolbar's "New".
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _confirm_delete_of_the_open_note(screen, pilot)

        undo = screen.query_one("#library-notes-delete-undo", Button)
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is undo,
            message=lambda: (
                "Focus after a confirmed delete is "
                f"{getattr(screen.focused, 'id', None)!r}, not Undo"
            ),
        )
        assert "library-canvas-action" in undo.classes, undo.classes
        assert _chips(screen).get("enter") == "undo delete", _chips(screen)


@pytest.mark.asyncio
async def test_undo_costs_one_enter_after_delete():
    """AC#2: the recovery action is one keystroke from the post-delete
    state (it used to be "/" then Tab x6)."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _confirm_delete_of_the_open_note(screen, pilot)

        service = screen.app_instance.notes_scope_service
        assert service.delete_calls, "The delete never reached the service"

        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: bool(service.restore_calls),
            message="One Enter on the post-delete state did not restore",
        )


# --- task-32548: the open note carries its list row's tie-break ------------


@pytest.mark.asyncio
async def test_opening_the_second_of_two_same_titled_notes_shows_its_tiebreak_in_the_editor():
    """AC#1/#2: the editor header carries the same suffix the list row
    shows, so the reader can tell which "Reading list" is open.

    Live at dev 2f97a42c9a (editor-00-32548-editor-header-no-tiebreak): the
    list read "Reading list · Unfiled · 2m · #8a41" and the open editor read
    "Reading list".
    """
    host = _build_notes_host(_DUPLICATE_NOTES)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        rows = _note_rows(screen)
        assert len(rows) == 2, [str(row.label) for row in rows]
        row_labels = [str(row.label).strip() for row in rows]
        suffixes = [label.rsplit(" · ", 1)[-1] for label in row_labels]
        assert suffixes[0] != suffixes[1], row_labels
        assert all(suffix.startswith("#") for suffix in suffixes), row_labels

        await _open_note(screen, pilot, 1)

        heading = screen.query_one("#library-note-editor-title", Static)
        assert str(heading.renderable) == f"Reading list · {suffixes[1]}", (
            f"Editor header is {str(heading.renderable)!r}; the row it was "
            f"opened from reads {row_labels[1]!r}"
        )

        screen.query_one("#library-note-context", Button).press()
        await pilot.pause()
        info_heading = screen.query_one("#library-note-context-title", Static)
        assert str(info_heading.renderable) == f"Reading list · {suffixes[1]}"


@pytest.mark.asyncio
async def test_a_title_that_does_not_repeat_keeps_a_plain_editor_header():
    """AC#1: the suffix is spent only where two visible notes actually
    collide -- an ordinary note's header is untouched."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        heading = screen.query_one("#library-note-editor-title", Static)
        assert str(heading.renderable) == "Research Note"


# --- task-32550: toolbar keyboard nits -------------------------------------


async def _submit_filter(screen, pilot, text: str):
    filter_input = screen.query_one("#library-notes-filter", Input)
    filter_input.focus()
    await pilot.pause()
    filter_input.value = text
    await pilot.pause()
    await pilot.press("enter")
    await _wait_for_condition(
        pilot,
        lambda: screen._notes_state.filter == text,
        message=lambda: f"Filter never settled on {text!r}",
    )
    await pilot.pause()


@pytest.mark.asyncio
async def test_slash_on_a_filtered_list_selects_the_existing_text():
    """AC#1: "/" on an already-filtered list selects the stale query, so the
    next keystroke replaces it instead of landing in front of it.

    Verified live at dev 2f97a42c9a: filter "Reading", focus away, "/", type
    "SCAL" -> the box read "SCAL". This pins the behaviour against a
    regression (Textual's ``select_on_focus``, which
    ``SelectAllOnFocusingClickInput`` relies on, is what supplies it).
    """
    host = _build_notes_host(_DUPLICATE_NOTES)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        await _submit_filter(screen, pilot, "Reading")

        _note_rows(screen)[0].focus()
        await pilot.pause()
        await pilot.press("/")
        await pilot.pause()

        filter_input = screen.query_one("#library-notes-filter", Input)
        assert screen.focused is filter_input
        assert filter_input.value == "Reading"
        assert filter_input.selection == (0, len(filter_input.value)), (
            f"'/' left the selection at {filter_input.selection}"
        )

        await pilot.press("x")
        await pilot.pause()
        assert filter_input.value == "x", (
            f"Typing after '/' produced {filter_input.value!r}"
        )


@pytest.mark.asyncio
async def test_tab_counts_to_toolbar_buttons_are_pinned_per_filter_state():
    """AC#2/#3: the Tab count to a toolbar button DOES depend on the filter
    state, because a filter disables Sort and a disabled Button leaves the
    focus chain. Both counts are pinned here and named in notes.md, which is
    the AC's second branch.

    Live at dev 2f97a42c9a (editor-00-32550-tab4-filtered-export): "/" then
    Tab x4 lands on "Add from files…" unfiltered and on "Export" filtered.
    """
    host = _build_notes_host(_DUPLICATE_NOTES)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        async def tab_four_from_the_filter() -> str:
            screen.query_one("#library-notes-filter", Input).focus()
            await pilot.pause()
            for _ in range(4):
                await pilot.press("tab")
                await pilot.pause()
            return str(getattr(screen.focused, "id", "") or "")

        assert await tab_four_from_the_filter() == "library-notes-add-from-files"

        await _submit_filter(screen, pilot, "Reading")
        assert screen.query_one("#library-notes-sort", Button).disabled, (
            "This pin assumes a filter disables Sort"
        )
        assert await tab_four_from_the_filter() == "library-notes-export"


# --- task-32551: Preview renders the title once ----------------------------


@pytest.mark.asyncio
async def test_preview_shows_a_leading_h1_equal_to_the_title_once():
    """AC#1/#3: a body starting "# <the note's own title>" is not rendered
    underneath the title line Preview already shows.

    Live at dev 2f97a42c9a (editor-00-32551-preview-title-twice): the seeded
    "Markdown showcase" note printed its title as the pane title AND again
    as the rendered H1.
    """
    notes = [
        {
            "id": "note-1",
            "title": "Markdown showcase",
            "content": "# Markdown showcase\n\n## Headings\n\nProse.\n",
            "version": 1,
        }
    ]
    host = _build_notes_host(notes)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_preview(screen, pilot)

        heading = screen.query_one("#library-note-preview-body-title", Static)
        assert str(heading.renderable) == "Markdown showcase"

        body = screen.query_one("#library-note-preview-body")
        assert "# Markdown showcase" not in body.source, body.source
        assert "## Headings" in body.source, body.source


def test_a_leading_h1_that_is_not_the_title_survives_preview():
    """AC#1: only an EXACT repeat of the title is dropped; the author's own
    first heading is left alone."""
    body = "# Reading list (second copy)\n\n- A Pattern Language\n"
    assert render_preview_source(body, title="Reading list") == body


def test_a_second_h1_equal_to_the_title_survives_preview():
    """AC#1: the rule is scoped to the OPENING heading, not to every line
    that happens to match the title."""
    body = "Intro.\n\n# Notes\n"
    assert render_preview_source(body, title="Notes") == body


# --- task-32556: a whitespace-only title leaves a receipt, not a ghost -----


async def _start_a_blank_note(screen, pilot):
    await _open_notes_list(screen, pilot)
    await pilot.press("ctrl+n")
    await _wait_for_selector(screen, pilot, "#library-note-title")
    await _wait_for_condition(
        pilot,
        lambda: bool(screen._notes_state.session_blank_id),
        message="Ctrl+N never produced a session blank note",
    )
    await pilot.pause()
    return screen._notes_state.session_blank_id


@pytest.mark.asyncio
async def test_a_whitespace_only_title_then_escape_shows_the_discard_receipt():
    """AC#1: keystrokes went in and the row vanished, so something has to
    say so.

    Live at dev 2f97a42c9a: New note, three spaces in Title, Escape -- no
    toast, no receipt, and ``sqlite3`` showed the row already ``deleted=1``.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _start_a_blank_note(screen, pilot)

        title = screen.query_one("#library-note-title", Input)
        title.focus()
        await pilot.pause()
        title.value = "   "
        await pilot.pause()

        screen.app_instance.notify = MagicMock()
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.view == "list",
            message="Escape never returned to the list",
        )

        messages = [
            call.args[0] for call in screen.app_instance.notify.call_args_list
        ]
        assert "Empty note discarded" in messages, messages


@pytest.mark.asyncio
async def test_the_list_never_paints_a_row_for_a_note_being_discarded():
    """AC#2: the discarded note must never reappear as a list row.

    Live at dev 2f97a42c9a: the list kept painting "Untitled · now" for a
    row whose database record was already ``deleted=1``, because the GC
    removed the flat source record and never reconciled the folder tree the
    rows are actually projected from.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        blank_id = await _start_a_blank_note(screen, pilot)

        title = screen.query_one("#library-note-title", Input)
        title.focus()
        await pilot.pause()
        title.value = "   "
        await pilot.pause()

        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.view == "list",
            message="Escape never returned to the list",
        )

        def ghost_rows() -> list[str]:
            projection = screen._build_library_notes_tree_projection()
            rows = getattr(projection, "rows", ()) if projection else ()
            mounted = [
                str(getattr(row, "note_id", "") or "")
                for row in screen.query(".library-notes-row")
            ]
            return [
                row.note_id
                for row in rows
                if row.kind == "note" and row.note_id == blank_id
            ] + [note_id for note_id in mounted if note_id == blank_id]

        for _ in range(6):
            assert ghost_rows() == [], (
                f"The discarded note {blank_id!r} is still projected as a row"
            )
            await pilot.pause()


@pytest.mark.asyncio
async def test_a_background_sync_does_not_evict_the_post_delete_focus_intent():
    """task-32539: the fix above only holds while nothing else re-syncs the
    canvas first.

    In production every delete also starts a Trash reload worker, which ends
    in a target-less ``_sync_library_canvas``. ``queue_after_recompose``
    REPLACES, so that background sync used to evict "focus the receipt's
    Undo" and install its own default identity restore -- captured when
    nothing was focused. Reproduced live at 235x52 (the harness fake carries
    no ``list_deleted_notes``, so the worker never ran there and the bug was
    invisible in tests); this pin gives the fake that seam for one test so
    the race is on the real route.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        service = screen.app_instance.notes_scope_service

        async def list_deleted_notes(**kwargs):
            return {
                "items": [
                    {"id": note_id, "title": note.get("title"), "version": 2}
                    for note_id, note in service.deleted_notes.items()
                ],
                "total": len(service.deleted_notes),
            }

        service.list_deleted_notes = list_deleted_notes

        await _open_first_note(screen, pilot)
        await _confirm_delete_of_the_open_note(screen, pilot)

        undo = screen.query_one("#library-notes-delete-undo", Button)
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is undo,
            message=lambda: (
                "The Trash reload evicted the post-delete focus intent; "
                f"focus is {getattr(screen.focused, 'id', None)!r}"
            ),
        )
        # The reload really did land -- otherwise this pins nothing.
        assert screen._notes_state.trash is not None
        assert _chips(screen).get("enter") == "undo delete"
