"""Library ▸ Notes riders wave -- backlinks group (task-32145).

Info → Properties lists the notes whose bodies link to the open note through
the ``[label](note://<id>)`` form the Obsidian importer writes, and each entry
opens that note. See ``backlog/tasks/task-32145*.md``.
"""

from __future__ import annotations

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
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    library_note_backlink_header,
)

#: The seeded vault shape: "hub" is linked from two notes, "lonely" from none.
_NOTES = [
    {"id": "hub", "title": "Zettelkasten — overview", "content": "The hub.", "version": 1},
    {
        "id": "review",
        "title": "Library review",
        "content": "See [Zettelkasten](note://hub) for the method.",
        "version": 1,
    },
    {
        "id": "daily",
        "title": "Daily 2026-09-07",
        "content": "Follow-up on [overview](note://hub).",
        "version": 1,
    },
    {"id": "lonely", "title": "Lonely note", "content": "Nothing points here.", "version": 1},
]


def _build_notes_host() -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_NOTES)
    return LibraryHarness(app)


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


async def _open_note_in_info(screen, pilot, note_id: str) -> None:
    """Open Notes, open ``note_id``, and switch the work pane to Info."""
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    await pilot.pause()
    rows = [
        row
        for row in screen.query(".library-notes-row")
        if str(getattr(row, "note_id", "") or "") == note_id
    ]
    assert rows, f"no row for {note_id!r}"
    rows[0].press()
    await _wait_for_selector(screen, pilot, "#library-note-body")
    await pilot.pause()
    screen.query_one("#library-note-context", Button).press()
    await pilot.pause()


async def _wait_for_backlink_header(screen, pilot, expected: str) -> None:
    await _wait_for_condition(
        pilot,
        lambda: bool(screen.query("#library-note-context-backlinks-title"))
        and _static_text(
            screen.query_one("#library-note-context-backlinks-title", Static)
        )
        == expected,
        message=lambda: f"backlink header never became {expected!r}",
    )


@pytest.mark.asyncio
async def test_info_lists_every_note_that_links_to_the_open_note():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_in_info(screen, pilot, "hub")

        await _wait_for_backlink_header(screen, pilot, "Linked from (2)")
        entries = list(screen.query(".library-note-backlink"))
        assert [str(entry.label) for entry in entries] == [
            "Daily 2026-09-07",
            "Library review",
        ]
        assert [getattr(entry, "note_id", "") for entry in entries] == [
            "daily",
            "review",
        ]


@pytest.mark.asyncio
async def test_info_says_so_when_nothing_links_to_the_open_note():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_in_info(screen, pilot, "lonely")

        await _wait_for_backlink_header(
            screen, pilot, "Linked from (0) — no notes link here yet"
        )
        assert not list(screen.query(".library-note-backlink"))


@pytest.mark.asyncio
async def test_activating_a_backlink_opens_that_note():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_in_info(screen, pilot, "hub")
        await _wait_for_backlink_header(screen, pilot, "Linked from (2)")

        entries = list(screen.query(".library-note-backlink"))
        target = next(
            entry for entry in entries if getattr(entry, "note_id", "") == "review"
        )
        target.press()
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-note-title", Input).value
            == "Library review",
            message="the backlinked note never opened",
        )
        assert screen._notes_state.selected_note_id == "review"


@pytest.mark.asyncio
async def test_late_arriving_backlinks_paint_without_a_recompose():
    """The load lands after the editor is composed, and often while the
    reader owns a field -- when a recompose is deferred by design
    (task-32062). The presentation sync must repaint the rows on its own.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_in_info(screen, pilot, "hub")
        await _wait_for_backlink_header(screen, pilot, "Linked from (2)")

        screen.query_one("#library-note-body").focus()
        await pilot.pause()
        screen._notes_state.backlinks = (("lonely", "Lonely note"),)
        screen._apply_library_note_presentation_state()
        await pilot.pause()

        assert _static_text(
            screen.query_one("#library-note-context-backlinks-title", Static)
        ) == "Linked from (1)"
        entries = list(screen.query(".library-note-backlink"))
        assert [getattr(entry, "note_id", "") for entry in entries] == ["lonely"]


def test_the_header_only_claims_zero_once_the_query_has_answered():
    """A pending or failed lookup must not read as a verified zero."""
    assert library_note_backlink_header((), "loading") == "Linked from — checking…"
    assert library_note_backlink_header((), "failed") == "Linked from — couldn't check"
    assert (
        library_note_backlink_header((), "ready")
        == "Linked from (0) — no notes link here yet"
    )
    assert library_note_backlink_header((("a", "A"),), "ready") == "Linked from (1)"


@pytest.mark.asyncio
async def test_a_failed_backlink_lookup_does_not_claim_there_are_none():
    host = _build_notes_host()

    async def _boom(**kwargs):
        raise RuntimeError("backlink query exploded")

    host.app_instance.notes_scope_service.list_note_backlinks = _boom
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_in_info(screen, pilot, "hub")

        await _wait_for_backlink_header(screen, pilot, "Linked from — couldn't check")
        assert not list(screen.query(".library-note-backlink"))


@pytest.mark.asyncio
async def test_backlinks_are_requested_bounded_and_for_the_open_note():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_in_info(screen, pilot, "hub")
        await _wait_for_backlink_header(screen, pilot, "Linked from (2)")

        calls = host.app_instance.notes_scope_service.backlink_calls
        assert calls, "the backlink seam was never asked"
        assert calls[-1]["note_id"] == "hub"
        assert calls[-1]["scope"] == "local_note"
        assert isinstance(calls[-1]["limit"], int) and calls[-1]["limit"] <= 51


@pytest.mark.asyncio
async def test_a_backlink_result_landing_mid_recompose_does_not_kill_the_app():
    """The load is a worker, so an exception here takes the whole app down.

    Reproduced twice on plain dev by the wave-3 editor-keys group: opening a
    note raised ``NoMatches: '#library-note-work-authority'`` out of
    ``_load_library_note_backlinks``'s presentation sync and killed the app.
    The work pane is mounted but its CHILDREN are between removal and
    remount, which ``apply_session_state`` assumed could not happen --
    ``_apply_post_compose_state`` already guards the same shape for the
    editor fields, and this path does not go through it.

    ``remove_children()`` is that DOM shape exactly: mounted pane, no
    composed children. The state the sync stores is what the pending
    recompose paints from, so it must still be recorded.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_in_info(screen, pilot, "hub")
        await _wait_for_backlink_header(screen, pilot, "Linked from (2)")

        pane = screen.query_one("#library-note-work-pane")
        await pane.remove_children()

        await screen._notes_controller._load_library_note_backlinks("hub")

        assert pane.presentation_state is not None
        assert pane.presentation_state.backlinks_status == "ready"
        assert [row[0] for row in pane.presentation_state.backlinks] == [
            "daily",
            "review",
        ]

        # Dropped, not lost: the pending recompose paints the stored state
        # onto the surfaces it mounts, so the rows still reach Info.
        pane.refresh(recompose=True)
        await _wait_for_backlink_header(screen, pilot, "Linked from (2)")
        assert [
            getattr(entry, "note_id", "")
            for entry in screen.query(".library-note-backlink")
        ] == ["daily", "review"]
