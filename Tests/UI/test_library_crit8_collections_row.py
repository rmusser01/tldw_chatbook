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

import asyncio
import threading
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.UI.Screens import library_screen
from tldw_chatbook.Library.library_shell_state import build_library_shell_state
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

        # A failing read clears it, rather than leaving a stale number.
        # Fix round 2, finding 1: this runs while the authority is still
        # ACTIVE -- ordered after the ``deactivate()`` case below, the read
        # returned on "no authority" and never reached ``list_page``, so the
        # exception branch went unexercised.
        async def failing_list_page():
            raise RuntimeError("controlled count read failure")

        scope.read_unfiltered_first_page = failing_list_page
        assert scope.active_authority is not None
        await screen._read_library_collections_count()
        assert screen._library_collections_prefetched_total is None

        # A read that never returns is treated the same way: the deadline is
        # what keeps the snapshot pass bounded when ``list_page`` is an HTTP
        # round trip with no timeout of its own.
        screen._library_collections_prefetched_total = 99

        async def hanging_list_page():
            await asyncio.sleep(60)

        scope.read_unfiltered_first_page = hanging_list_page
        with patch.object(
            library_screen, "LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS", 0.05
        ):
            await screen._read_library_collections_count()
        assert screen._library_collections_prefetched_total is None

        # Losing the authority clears it too.
        screen._library_collections_prefetched_total = 99
        scope.deactivate()
        await screen._read_library_collections_count()
        assert screen._library_collections_prefetched_total is None


async def test_the_collections_count_read_does_not_stack_its_deadline_on_the_gather() -> None:
    """Fix round 2, finding 3: the count read runs WITH the source gather.

    Awaited serially ahead of it, a stalled Collections read and a stalled
    source seam each burned the same 5 s deadline -- ~10 s for a pass whose
    own timeout copy says it waited 5 s. Here the count read parks until a
    gathered source call has started: under the old serial ordering the
    gather had not begun yet, so the whole snapshot would hang.
    """
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        gather_started = threading.Event()
        original_list_notes = app.notes_scope_service.list_notes

        def recording_list_notes(*args, **kwargs):
            gather_started.set()
            return original_list_notes(*args, **kwargs)

        app.notes_scope_service.list_notes = recording_list_notes

        async def blocked_count() -> None:
            while not gather_started.is_set():
                await asyncio.sleep(0.01)

        screen._read_library_collections_count = blocked_count
        # Generous: this deadline only bounds the FAILING (serial) shape,
        # which hangs outright -- it is not a timing assertion.
        await asyncio.wait_for(screen._list_local_source_snapshot(), timeout=20)


async def test_one_page_one_read_serves_both_the_count_and_the_evidence() -> None:
    """task-32103 AC#1: one page-1 read per snapshot pass, not two.

    ``get_library_user_content_evidence`` and the rail's count prefetch ask
    the authority the same question -- the unfiltered page-1 total -- and
    both run in the same pass. In server mode that was two HTTP round trips
    for one number.
    """
    from tldw_chatbook.Library.collections_capture_models import CapturePageRequest

    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    backend = scope._backend
    assert backend is not None
    calls: list[CapturePageRequest] = []
    original_list_page = backend.list_page

    async def counting_list_page(request):
        calls.append(request)
        await asyncio.sleep(0.02)
        return await original_list_page(request)

    backend.list_page = counting_list_page

    count_page, evidence = await asyncio.gather(
        scope.read_unfiltered_first_page(),
        scope.get_library_user_content_evidence(),
    )

    assert len(calls) == 1, f"expected one page-1 read, got {len(calls)}"
    assert count_page.total == 0
    assert evidence is not None

    # A settled read is never replayed: the next pass reads again.
    await scope.read_unfiltered_first_page()
    assert len(calls) == 2

    # The canvas's own page reads keep their unshared path.
    await scope.list_page(CapturePageRequest(authority.key, page=1))
    assert len(calls) == 3


async def test_re_entering_a_scoped_canvas_never_flashes_the_unfiltered_total() -> None:
    """task-32103 AC#2: the scope outlives the page, so the gate must see it.

    ``unmount()`` resets the capture controller's ``page`` while the
    screen-owned ``active_scope`` persists, so leaving and re-entering a
    scoped Collections canvas put the rail back on the "never loaded a
    page" branch and painted the UNFILTERED total for one load window --
    the exact flash the guide claims is gone.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    for index in range(3):
        await scope.save_capture(
            CaptureSaveRequest(
                authority.key,
                f"https://example.test/reentry-{index}",
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
            lambda: screen._library_collections_prefetched_total == 3,
            message="The rail count prefetch never settled.",
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
        await _wait_for_selector(
            screen, pilot, "#library-collections-scope-favorites"
        )
        screen.query_one("#library-collections-scope-favorites", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.active_scope == "favorites",
            message="The Favorites scope was never applied.",
        )

        # Leaving the canvas resets the page; the scope stays.
        controller.unmount()
        assert controller.state.page is None
        assert screen._collections_state.active_scope == "favorites"
        assert screen._library_collections_prefetched_total == 3

        assert screen._build_library_shell_input().collections_count is None


async def test_a_page_from_a_departed_authority_is_never_painted() -> None:
    """task-32103 AC#3: ``exact_total`` outlives its authority; the rail must not.

    ``deactivate()`` nulls the active authority without touching the
    canvas controller, whose retained page still answers ``exact_total`` --
    the PREVIOUS authority's number under the new one's name.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://example.test/departed",
            title="Departed",
            text_content="Body.",
        )
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")
        controller = screen._library_collections_capture_controller
        assert controller is not None
        await _wait_for_condition(
            pilot,
            lambda: controller.state.exact_total == 1,
            message="The Collections canvas never loaded its first page.",
        )
        assert screen._build_library_shell_input().collections_count == 1

        scope.deactivate()
        assert controller.state.exact_total == 1
        assert screen._build_library_shell_input().collections_count is None


async def test_a_count_that_times_out_says_so_on_the_row() -> None:
    """task-32103 AC#3: a deadline is visible, not a silently absent number.

    A timed-out count read cleared the stored total and returned, leaving a
    bare "Collections" -- identical to a row whose count is off by design.
    """
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_collections_prefetched_total == 0,
            message="The rail count prefetch never settled.",
        )

        async def hanging_list_page():
            await asyncio.sleep(60)

        scope = app.collections_capture_scope_service
        scope.read_unfiltered_first_page = hanging_list_page
        with patch.object(
            library_screen, "LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS", 0.05
        ):
            await screen._read_library_collections_count()

        shell_input = screen._build_library_shell_input()
        assert shell_input.collections_count is None
        assert shell_input.collections_count_unavailable is True

        shell = build_library_shell_state(
            shell_input, selected_row_id=screen._library_selected_row_id
        )
        row = next(
            row
            for section in shell.sections
            for row in section.rows
            if row.row_id == "browse-collections"
        )
        assert row.count_display == " (—)"
        # The Details sentence comes from the SAME gated value as the row's
        # "(—)" -- ``_build_library_shell_input`` computes it once -- so the
        # rail's two lines cannot contradict each other (fix round 1).
        deadline_line = next(
            line for line in shell_input.details_lines if "Collections count" in line
        )
        assert deadline_line == (
            "Collections count unavailable (waited "
            f"{library_screen.LIBRARY_SOURCE_SNAPSHOT_TIMEOUT_SECONDS:g} s) — "
            "open Collections to load it."
        )


async def test_details_stops_saying_unavailable_once_the_canvas_supplies_a_count() -> None:
    """task-32103 AC#3 (fix round 1): the row and Details share one gate.

    The row's "(—)" was gated on "no count AND the read failed" while the
    Details sentence read the raw failure flag, which only a later count
    read clears. So after a failed prefetch, opening Collections gave the
    row a real number while the line directly beneath it still said the
    count was unavailable and told the user to open Collections -- the
    thing they had just done.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://example.test/one",
            title="One",
            text_content="Body.",
        )
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # A failed prefetch: the row says so, and so does Details.
        screen._library_collections_prefetched_total = None
        screen._library_collections_count_authority = authority.key
        screen._library_collections_count_failure = "error"
        shell_input = screen._build_library_shell_input()
        assert shell_input.collections_count_unavailable is True
        assert any(
            "Collections count unavailable" in line
            for line in shell_input.details_lines
        )
        # ...and a non-deadline failure never claims a wait that did not
        # happen.
        assert not any("waited" in line for line in shell_input.details_lines)

        # Opening the canvas supplies a real count. The flag is still set
        # (only a later read clears it), so the two must agree via the gate.
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")
        controller = screen._library_collections_capture_controller
        assert controller is not None
        await _wait_for_condition(
            pilot,
            lambda: controller.state.exact_total == 1,
            message="The Collections canvas never loaded its first page.",
        )

        shell_input = screen._build_library_shell_input()
        assert shell_input.collections_count == 1
        assert shell_input.collections_count_unavailable is False
        assert not any(
            "Collections count unavailable" in line
            for line in shell_input.details_lines
        ), shell_input.details_lines


async def test_the_prefetched_total_is_fenced_to_the_authority_it_was_read_from() -> None:
    """task-32103 AC#3 (fix round 1): fence the PREFETCH path too.

    The controller-state branch already refused a page whose authority had
    gone, but the fallback returned the prefetch, which carried no authority
    tag. ``_activate_collections_capture_authority`` swaps the authority on
    every committed source switch and the Collections row is excluded from
    ``counts_loading``, so between the switch and the next read the rail
    painted the previous authority's total under the new authority's name.
    """
    from tldw_chatbook.Library.collections_capture_service import (
        build_local_capture_authority,
    )

    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    for index in range(2):
        await scope.save_capture(
            CaptureSaveRequest(
                authority.key,
                f"https://example.test/fenced-{index}",
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
            lambda: screen._library_collections_prefetched_total == 2,
            message="The rail count prefetch never settled.",
        )
        assert screen._build_library_shell_input().collections_count == 2

        # The authority is swapped without a new count read (the window a
        # committed source switch opens). Only the authority matters here --
        # nothing reads through this backend before the assertion.
        other = build_local_capture_authority("other-profile", "other-db")
        scope.activate(other, SimpleNamespace(authority=other))

        assert screen._library_collections_prefetched_total == 2
        assert screen._build_library_shell_input().collections_count is None

        # The FAILURE half is fenced the same way (Qodo #2): a failure from
        # the departed authority must not mark the new one unavailable.
        screen._library_collections_count_failure = "timeout"
        shell_input = screen._build_library_shell_input()
        assert shell_input.collections_count_unavailable is False
        assert not any(
            "Collections count unavailable" in line
            for line in shell_input.details_lines
        ), shell_input.details_lines


async def test_a_count_deadline_never_cancels_the_evidence_read() -> None:
    """task-32103 AC#1 (fix round 1): the shield is what makes sharing safe.

    Both consumers await the SAME task, and the rail count read wraps its
    await in a 5 s ``wait_for``. Without ``asyncio.shield`` that timeout
    would cancel the shared task out from under the evidence read, which
    would then degrade to UNKNOWN -- indistinguishable from an empty
    Library. Nothing pinned that property.
    """
    from tldw_chatbook.Library.library_content_evidence import LibraryContentEvidence

    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://example.test/shielded",
            title="Shielded",
            text_content="Body.",
        )
    )
    backend = scope._backend
    assert backend is not None
    original_list_page = backend.list_page

    async def slow_list_page(request):
        await asyncio.sleep(0.2)
        return await original_list_page(request)

    backend.list_page = slow_list_page

    evidence_task = asyncio.ensure_future(scope.get_library_user_content_evidence())
    await asyncio.sleep(0)  # let the evidence read claim the shared slot

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(scope.read_unfiltered_first_page(), timeout=0.05)

    assert await evidence_task is LibraryContentEvidence.HAS_USER_CONTENT


async def test_a_wedged_shared_read_is_not_inherited_by_the_next_pass() -> None:
    """task-32103 (fix round 1): the shared slot ages out.

    Every awaiter shields the in-flight read, so nothing cancels it. The
    consumer's budget is 5 s but the client's per-request budget is 300 s,
    so one wedged request owned the slot for up to five minutes and every
    later snapshot pass joined the same dead read instead of retrying.
    """
    from tldw_chatbook.Library import collections_capture_service

    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    backend = scope._backend
    assert backend is not None
    original_list_page = backend.list_page
    started = 0

    async def wedged_first_call(request):
        nonlocal started
        started += 1
        if started == 1:
            await asyncio.sleep(60)
        return await original_list_page(request)

    backend.list_page = wedged_first_call

    with patch.object(
        collections_capture_service, "FIRST_PAGE_READ_MAX_AGE_SECONDS", 0.05
    ):
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(scope.read_unfiltered_first_page(), timeout=0.05)
        await asyncio.sleep(0.06)
        # The next pass starts its OWN read rather than joining the wedge.
        page = await asyncio.wait_for(
            scope.read_unfiltered_first_page(), timeout=1.0
        )

    assert started == 2
    assert page.total == 0
