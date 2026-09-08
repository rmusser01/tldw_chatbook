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

import pytest
from textual.widgets import Button, Static

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
