"""TASK-34000.18: Library Media's Read later, Move to trash and Edit metadata
act on the item opened by link, not on row 1.

The review (nl-v-l-03, j2) imported a folder, chose "Open in Library" on a
non-newest item and pressed Read later: the Reader showed that item, the
list cursor (``▸``) stayed on row 1, and Read later saved row 1 -- three
times out of three. Edit metadata › Save wrote the opened item's URL onto
row 1's record (``UNIQUE constraint failed: Media.url``), and the trash
confirm named no item at all.

Cause (verified on 32caa91c1a): every by-link route (``_navigate_to_media``
from the import receipt, the landing hub's recent "Open", the deep-link
entry) hands ``_open_library_item_by_id`` a BARE id, which was written
straight into ``selected_media_id`` while only the Reader session got the
canonical ``local:media:N``. Page rows are always canonical, so when the
browse page landed the pure builder fell back to row 0 and the actions --
which read the selection, not the Reader -- acted on row 0.

Real app stylesheet (``LibraryProductionCSSHarness``), the production-shaped
exact-page fake (``StaticLibraryMediaScopeService`` records every write),
painted text through the compositor, ids production-shaped (``"1"`` ..
``"5"``, what ``LocalMediaReadingService`` returns) so the Reader's payload
id, the row model's id and the by-link id are one spelling. One boot per
test; the whole file stays inside the 35 s lane budget.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

SIZE = (160, 45)

#: The review's five-item import, newest first on the page: row 0 is the
#: "team handbook", the item every action used to act on.
TITLES = {
    "1": "Team handbook",
    "2": "Paper retrieval practice",
    "3": "Meeting notes",
    "4": "Reading list",
    "5": "Glossary",
}


def _five_media_items() -> list[dict[str, object]]:
    return [
        {
            "id": record_id,
            "title": title,
            "type": "document",
            "last_modified": f"2026-07-{10 - int(record_id):02d}T08:00:00Z",
            "author": "Review fixture",
            "url": f"file:///import-docs/{title.lower().replace(' ', '-')}.md",
            "keywords": ["import"],
            "content": f"{title}: a few paragraphs of body text for the Reader.",
            "version": 1,
        }
        for record_id, title in TITLES.items()
    ]


def _host():
    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, _two_conversations(), media=_five_media_items())
    return LibraryProductionCSSHarness(app), app, app.media_reading_scope_service


def _painted(host, region) -> str:
    strips = list(host.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text.rstrip()
        for y in range(region.y, min(region.bottom, len(strips)))
    )


def _rows(screen) -> list[Button]:
    return [
        button
        for button in screen.query("#library-media-canvas Button")
        if str(getattr(button, "id", "") or "").startswith("library-media-row-")
    ]


def _highlighted_rows(screen) -> list[Button]:
    return [row for row in _rows(screen) if getattr(row, "_library_media_selected", False)]


async def _open_by_link(screen, pilot, record_id: str) -> str:
    """Open ``record_id`` the way the by-link routes do and wait for BOTH the
    Reader detail and the browse page to land (the page landing is where the
    selection used to be overwritten)."""
    canonical = f"local:media:{record_id.rsplit(':', 1)[-1]}"
    await screen._open_library_item_by_id("media", record_id)
    await _wait_for_selector(screen, pilot, "#library-media-row-0")
    browse = screen._library_media_browse_controller

    def landed() -> bool:
        session = screen._media_state.reader_session
        return (
            session.loaded_id == canonical
            and session.pending_request is None
            and not browse.loading
            and browse.applied_scope is not None
            and browse.applied_scope == browse.requested_scope
            and len(_rows(screen)) == 5
        )

    await _wait_for_condition(
        pilot,
        landed,
        message=lambda: (
            f"open-by-link of {record_id!r} never settled: session="
            f"{screen._media_state.reader_session!r} loading={browse.loading!r} "
            f"rows={len(_rows(screen))}"
        ),
    )
    await pilot.pause()
    await pilot.pause()
    return canonical


async def _open_more_strip(screen, pilot) -> None:
    await pilot.click("#library-media-reader-more")
    await _wait_for_selector(screen, pilot, "#library-media-delete")
    await pilot.pause()


def _reader_title(screen) -> str:
    return str(screen.query_one("#library-media-viewer-title", Static).renderable)


async def test_open_by_bare_id_highlights_that_row_after_the_browse_lands():
    """AC#1: the list highlight sits on the opened item, not row 1, and stays
    there after the browse page lands. The ``▸`` cue and the ``loaded`` word
    paint on the same row, and that row is the opened item's."""
    host, _app, _service = _host()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        canonical = await _open_by_link(screen, pilot, "3")

        assert screen._media_state.selected_media_id == canonical
        highlighted = _highlighted_rows(screen)
        assert [str(row.media_id) for row in highlighted] == [canonical], [
            (str(row.media_id), row.label) for row in _rows(screen)
        ]
        row = highlighted[0]
        painted = _painted(host, row.region)
        assert TITLES["3"] in painted, painted
        assert "loaded" in painted, painted
        assert "▸" in painted, painted
        # Negative control: row 0 (the old target) carries neither cue.
        row_0 = _painted(host, _rows(screen)[0].region)
        assert TITLES["1"] in row_0 and "▸" not in row_0 and "loaded" not in row_0


async def test_read_later_and_edit_save_target_the_opened_item_not_row_1():
    """AC#2, AC#5 (verbatim), AC#4: open "3" by bare id; the Read later button
    and the ``l`` key save/remove id 3; Edit metadata › Save writes id 3 with
    the typed title; the Reader keeps showing item 3 throughout."""
    host, _app, service = _host()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        canonical = await _open_by_link(screen, pilot, "3")
        assert _reader_title(screen) == TITLES["3"]

        # -- the button the j2 user pressed.
        await pilot.click("#library-media-read-later")
        await _wait_for_condition(
            pilot,
            lambda: bool(service.read_it_later_calls),
            message="Read later never reached the service",
        )
        assert service.read_it_later_calls[-1] == {"action": "save", "media_id": 3}
        await _wait_for_condition(
            pilot,
            lambda: str(
                screen.query_one("#library-media-read-later", Button).label
            )
            == "Remove later",
            message="the Read later button never showed the saved state",
        )
        assert _reader_title(screen) == TITLES["3"]
        assert screen._media_state.reader_session.loaded_id == canonical

        # -- the ``l`` accelerator takes the same path.
        await pilot.press("l")
        await _wait_for_condition(
            pilot,
            lambda: len(service.read_it_later_calls) == 2,
            message="the l key never reached the service",
        )
        assert service.read_it_later_calls[-1] == {"action": "remove", "media_id": 3}

        # -- Edit metadata: the form is prefilled from the OPENED item, so a
        # Save that targets row 1 writes item 3's fields onto row 1's record.
        await _open_more_strip(screen, pilot)
        await pilot.click("#library-media-edit")
        title_input = await _wait_for_selector(screen, pilot, "#library-media-edit-title")
        assert isinstance(title_input, Input)
        assert title_input.value == TITLES["3"]
        await pilot.click("#library-media-edit-title")
        await pilot.press("end")
        for key in (
            "space", "left_parenthesis", "e", "d", "i", "t", "e", "d",
            "right_parenthesis",
        ):
            await pilot.press(key)
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-media-edit-title", Input).value
            == f"{TITLES['3']} (edited)",
            message=lambda: (
                f"typed title never landed: "
                f"{screen.query_one('#library-media-edit-title', Input).value!r}"
            ),
        )
        await pilot.click("#library-media-edit-save")
        await _wait_for_condition(
            pilot,
            lambda: bool(service.update_calls),
            message="Edit Save never reached the service",
        )
        assert service.update_calls[-1]["media_id"] == 3
        assert service.update_calls[-1]["title"] == f"{TITLES['3']} (edited)"
        # AC#4: the Reader stays on item 3 and paints the new title.
        await _wait_for_condition(
            pilot,
            lambda: not screen._media_state.editing
            and _reader_title(screen) == f"{TITLES['3']} (edited)",
            message=lambda: f"reader title after save: {_reader_title(screen)!r}",
        )
        assert screen._media_state.reader_session.loaded_id == canonical
        assert [str(row.media_id) for row in _highlighted_rows(screen)] == [canonical]
        assert TITLES["1"] not in _reader_title(screen)


async def test_trash_confirm_names_the_opened_item_and_trashes_it():
    """AC#2, AC#3: More ▸ Move to trash names the opened item in the confirm
    copy, and Delete trashes that item, not row 1."""
    host, _app, service = _host()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_by_link(screen, pilot, "3")

        await _open_more_strip(screen, pilot)
        await pilot.click("#library-media-delete")
        copy = await _wait_for_selector(screen, pilot, "#library-media-delete-confirm-copy")
        assert isinstance(copy, Static)
        await pilot.pause()
        expected = f'Move "{TITLES["3"]}" to Trash?'
        assert str(copy.renderable).startswith(expected), str(copy.renderable)
        assert "restore later from Trash" in str(copy.renderable)
        painted = _painted(host, copy.region)
        assert expected in painted, painted

        await pilot.click("#library-media-delete-confirm")
        await _wait_for_condition(
            pilot,
            lambda: bool(service.delete_calls),
            message="Delete never reached the service",
        )
        assert service.delete_calls[-1]["media_id"] == 3
        # The item on row 1 is untouched.
        assert any(item["id"] == "1" for item in service.media_items)
        assert not any(item["id"] == "3" for item in service.media_items)


async def test_use_in_console_payload_names_the_opened_item():
    """The AC#2 "Use in Console" half, pinned (green on the base): the
    hand-off payload resolves the Reader's loaded item, never the list row."""
    host, _app, _service = _host()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_by_link(screen, pilot, "3")

        payload = screen._selected_media_handoff_payload()

        assert payload is not None
        assert payload.item_type == "media"
        assert payload.title == TITLES["3"]
        assert payload.source_id == "3"
        assert payload.metadata["media_id"] == "3"
        assert f"Media: {TITLES['3']}" in payload.body
        assert TITLES["1"] not in payload.body


async def test_a_landing_recent_open_and_a_rag_open_land_on_the_same_identity():
    """AC#2 routes: the landing hub's recent "Open" passes the bare backing id
    and the Search/RAG "Open" passes ``local:media:N``; both must leave the
    selection AND the Reader on the same canonical identity."""
    host, _app, _service = _host()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        bare = await _open_by_link(screen, pilot, "4")
        assert screen._media_state.selected_media_id == bare == "local:media:4"
        assert screen._media_state.reader_session.loaded_id == "local:media:4"
        assert [str(row.media_id) for row in _highlighted_rows(screen)] == [bare]

        rag = await _open_by_link(screen, pilot, "local:media:4")
        assert screen._media_state.selected_media_id == rag == "local:media:4"
        assert screen._media_state.reader_session.loaded_id == "local:media:4"
        assert [str(row.media_id) for row in _highlighted_rows(screen)] == [rag]
        assert _reader_title(screen) == TITLES["4"]
