"""Critique-8 pins for the Library rail's Collections row (task-32057).

Three behaviours the critique-8 live review reported on the Collections
row, each pinned against the real mounted Library shell:

1. The row carried NO count until it was visited -- the capture count is
   fetched lazily on the first canvas visit, so a user reading the rail
   sees "Collections" beside "Media (11)"/"Notes (7)" and cannot tell
   whether it is empty or unloaded.
2. Selecting the row was reported to collapse the Create section and
   persist that collapse. This file pins the opposite: no rail section's
   disclosure state, and no ``[library.rail_state] sections`` write,
   follows from selecting a row.
3. The local Collections service refuses every write with
   ``LegacyCollectionsReadOnlyError``; the canvas never said so. The
   legacy-recovery disclosure now carries that reason and the recovery
   path the service itself names.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.Library.collections_capture_models import CaptureSaveRequest
from tldw_chatbook.Library.library_collections_service import (
    LegacyCollectionsReadOnlyError,
)
from Tests.UI.test_library_collections_capture_reader import _seed_legacy_records
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


pytestmark = pytest.mark.asyncio


def _collections_row_label(screen) -> str:
    """Return the rail's Collections row label as plain text."""
    button = screen.query_one("#library-row-browse-collections", Button)
    return str(button.label)


async def test_collections_row_carries_its_count_before_the_row_is_visited() -> None:
    """The rail's Collections count no longer waits for a canvas visit.

    Reproduces critique-8 register row 8 on the empty-captures profile the
    review used: at first paint the row read a bare "Collections" while
    every sibling row already carried "(N)", so the row was indistinguishable
    from a source whose count is off by design (Search / RAG).
    """
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # The Collections canvas has never been opened.
        assert screen._library_selected_row_id != "browse-collections"
        assert not screen.query("#library-collections-reader-shell")

        await _wait_for_condition(
            pilot,
            lambda: "(0)" in _collections_row_label(screen),
            message=(
                "The Collections rail row never showed a count before the "
                f"row was visited: {_collections_row_label(screen)!r}"
            ),
        )


async def test_selecting_collections_leaves_every_other_section_disclosure_alone() -> None:
    """Selecting a rail row is not a disclosure or a persistence event.

    Critique-8 reported that visiting Collections collapsed the Create
    section and that the collapse survived into the next launch. This pins
    both halves: the Create body stays displayed, the read-back preference
    stays open, and nothing writes ``[library.rail_state] sections``.
    """
    app = _build_test_app()
    host = LibraryHarness(app)
    writes: list[tuple[str, str]] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen._save_library_rail_preferences = lambda serialized: writes.append(
            ("library.rail_state", "sections")
        )

        create_body = screen.query_one("#library-rail-section-body-create")
        assert create_body.display is True
        assert screen._library_rail_preferences().create_open is True

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")

        assert screen.query_one("#library-rail-section-body-create").display is True
        assert screen._library_rail_preferences().create_open is True
        assert writes == []


async def test_legacy_collections_disclosure_names_the_read_only_recovery_path() -> None:
    """A profile holding legacy Collections records says they are read-only.

    ``LocalLibraryCollectionsService`` refuses create/rename/delete/restore/
    add-item with ``LegacyCollectionsReadOnlyError``; before this pin the
    canvas offered a "Legacy Collections data…" button with no statement
    that the records behind it can only be read and exported.
    """
    app = _build_test_app()
    _seed_legacy_records(app.local_library_collections_db, count=3)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")
        await _wait_for_selector(screen, pilot, "#library-collections-legacy-recovery")

        notice = await _wait_for_selector(
            screen, pilot, "#library-collections-legacy-read-only"
        )
        text = str(notice.renderable if isinstance(notice, Static) else notice)
        assert "read-only on this profile" in text
        # The next step is the service's own recovery sentence, verbatim.
        assert LegacyCollectionsReadOnlyError.recovery in text


async def test_rail_count_never_falls_back_to_the_unfiltered_total_after_a_visit() -> None:
    """Fix round 1, finding 1: the fallback is "never visited", not "no total".

    ``exact_total`` is ``None`` whenever the page is loading OR stale, not
    only before the first visit. So once the canvas owned a narrower scope,
    every page turn and every scope switch dropped the rail back onto the
    unfiltered page-1 read: a Favorites scope holding 2 captures flashed the
    whole-library total mid-load, and a stale page got a rail number the
    canvas itself withholds.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    for index in range(3):
        await scope.save_capture(
            CaptureSaveRequest(
                authority.key,
                f"https://example.test/capture-{index}",
                title=f"Capture {index}",
                text_content="Body.",
            )
        )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: "(3)" in _collections_row_label(screen),
            message="The unvisited rail count never arrived.",
        )

        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")
        controller = screen._library_collections_capture_controller
        assert controller is not None
        await _wait_for_condition(
            pilot,
            lambda: controller.state.page is not None,
            message="The Collections canvas never loaded its first page.",
        )

        # Narrow to a scope holding nothing, then force the loading/stale
        # window the fallback used to paint through.
        screen.query_one("#library-collections-scope-favorites", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.active_scope == "favorites",
            message="The Favorites scope was never applied.",
        )
        # The scoped total wins over the unfiltered prefetch, which is
        # still 3 and must not be substituted.
        assert controller.state.exact_total == 0
        assert screen._library_collections_prefetched_total == 3
        assert screen._build_library_shell_input().collections_count == 0

        # Mid-load and stale are the two windows the old gate leaked
        # through: ``exact_total`` is None in both, but the canvas has a
        # page, so the rail must withhold a number rather than borrow the
        # unfiltered one.
        for broken in (
            replace(controller.state, page_loading=True),
            replace(
                controller.state, page_stale=True, page_error="page_load_failed"
            ),
        ):
            controller.state = broken
            assert controller.state.exact_total is None
            count = screen._build_library_shell_input().collections_count
            assert count is None, (
                "A loading or stale Collections page must not borrow the "
                f"unfiltered prefetch total, got {count!r}"
            )


async def test_prefetched_total_is_dropped_when_the_capture_authority_goes_away() -> None:
    """Fix round 1, finding 2: a total must not outlive its authority.

    ``CollectionsCaptureScopeService.deactivate()`` nulls the active
    authority (an authority switch, or teardown). The count read used to
    return early on that path leaving the previous authority's number in
    place, so the rail kept painting it.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://example.test/only",
            title="Only capture",
            text_content="Body.",
        )
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_collections_prefetched_total == 1,
            message="The rail count prefetch never settled.",
        )

        scope.deactivate()
        await screen._read_library_collections_count()
        assert screen._library_collections_prefetched_total is None

        # A failing read clears it too, rather than leaving a stale number.
        screen._library_collections_prefetched_total = 99

        async def failing_list_page(_request):
            raise RuntimeError("controlled count read failure")

        scope.list_page = failing_list_page
        await screen._read_library_collections_count()
        assert screen._library_collections_prefetched_total is None
