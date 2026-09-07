"""TASK-22209: a Prev/Next match click must not re-read the whole document.

Before this task each match-navigation click cost 3-4 O(document) passes:
``_advance_library_media_content_match`` rebuilt the viewer display state
(a full content copy), ran ``find_content_matches`` over the document, and
then ``sync_match_index -> sync_search -> build_raw_content_renderable``
ran a SECOND full ``find_content_matches`` and rebuilt the whole Rich
``Text`` (up to three appends per line of the document) purely to move one
line's highlight from ``reverse`` to ``reverse bold``.

The probes below count the document passes at their two shared seams --
``find_content_matches`` (imported by both the screen and the content
widget) and the content widget's renderable/plan builders -- and pin the
mounted ``Static``'s renderable OBJECT IDENTITY across clicks, which is
what "patched, not rebuilt" actually means. No wall-clock thresholds are
asserted (the 15457 probe rule); the multi-megabyte probe prints its
per-click timings for the task record and pins the counts.
"""

from __future__ import annotations

from contextlib import contextmanager
from statistics import median
from time import perf_counter

import pytest
from textual.selection import Selection
from textual.widgets import Button, Input, Static

import tldw_chatbook.UI.Screens.library_screen as library_screen_module
import tldw_chatbook.Widgets.Library.library_media_content as media_content_module
from Tests.UI.test_library_media_reader_flow import (
    _flow_app,
    _row_identity,
    _wait_for_detail_call,
)
from Tests.UI.test_library_media_side_by_side import (
    WIDE_SIZE,
    _open_media_list,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _wait_for_condition,
)
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
)
from tldw_chatbook.Widgets.Library.library_media_raw_view import (
    VirtualizedRawContent,
)

NEEDLE = "needle"
OTHER_NEEDLE = "beacon"


def _document(lines: int = 400) -> str:
    """Build a plain-text document with two needles at different densities.

    ``NEEDLE`` lands on every eighth line and ``OTHER_NEEDLE`` on every
    twentieth, so a 400-line document has 50 of the first and 20 of the
    second -- the two counts have to differ for the "Match N of M" status
    to prove WHICH query the screen's match list was built from.
    """
    body = []
    for index in range(lines):
        if index % 8 == 3:
            body.append(f"line {index}: the {NEEDLE} sits here")
        elif index % 20 == 6:
            body.append(f"line {index}: a {OTHER_NEEDLE} sits here")
        else:
            body.append(f"line {index}: ordinary reading material")
    return "\n".join(body)


@contextmanager
def _count_document_passes():
    """Count every O(document) pass the match-navigation path can take.

    Three seams, all module globals (never ``@on`` handlers, so class-free
    monkeypatching is dispatch-safe):

    * ``find_content_matches`` as the screen imported it,
    * ``find_content_matches`` as the content widget imported it,
    * the content widget's renderable builder and -- once TASK-22209 adds
      it -- its highlight-plan builder, each of which walks every line.

    The plan builder is looked up defensively so this probe still runs (and
    reds honestly) against the pre-task tree, where it does not exist.
    """
    counts = {"matches": 0, "renderable": 0, "plan": 0}
    originals: list[tuple[object, str, object]] = []

    def _wrap(module: object, name: str, key: str) -> None:
        original = getattr(module, name, None)
        if original is None:
            return

        def counting(*args, **kwargs):
            counts[key] += 1
            return original(*args, **kwargs)

        originals.append((module, name, original))
        setattr(module, name, counting)

    _wrap(library_screen_module, "find_content_matches", "matches")
    _wrap(media_content_module, "find_content_matches", "matches")
    # `build_raw_content_renderable` / `build_raw_content_highlight_plan`
    # were deleted by TASK-22500 (the virtualized view styles per row instead
    # of rebuilding a whole-document Text). `_wrap` returns early on a
    # missing attribute, so naming them here would be a silent no-op that
    # reads like coverage; the gate now rides on `find_content_matches`.
    try:
        yield counts
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def _total(counts: dict[str, int]) -> int:
    return counts["matches"] + counts["renderable"] + counts["plan"]


def _seed_row_document(screen, service, index: int, content: str):
    """Give row ``index``'s backing item ``content`` before it is fetched."""
    row = screen.query_one(f"#library-media-row-{index}", Button)
    canonical_id, backing_id, title = _row_identity(row)
    source = next(
        item for item in service.media_items if item["id"] == f"media-{backing_id}"
    )
    source["content"] = content
    return canonical_id, backing_id, title


async def _load_row_with_document(screen, pilot, service, index: int, content: str):
    """Seed row ``index`` with ``content`` and open it from the Items list."""
    canonical_id, backing_id, title = _seed_row_document(screen, service, index, content)
    screen.query_one(f"#library-media-row-{index}", Button).press()
    await _wait_for_detail_call(service, backing_id)
    service.release(backing_id)
    await _wait_for_condition(
        pilot,
        lambda: screen._library_media_reader_session.loaded_id == canonical_id,
        message=f"Row {index} never settled its detail.",
    )
    return canonical_id, backing_id, title


async def _traverse_to_row(screen, pilot, service, *, from_index: int, to_index: int):
    """Arrow-key from one Items row to the next and settle its detail.

    Traversal is the ONLY document swap that keeps a submitted query alive:
    a row *press* runs the viewer-entry reset (which blanks the query),
    while ``_select_library_media_reader_row`` -- the focus/arrow seam --
    deliberately does not.
    """
    canonical_id, backing_id, _ = _row_identity(
        screen.query_one(f"#library-media-row-{to_index}", Button)
    )
    screen.query_one(f"#library-media-row-{from_index}", Button).focus()
    await pilot.pause()
    await pilot.press("down")
    await _wait_for_detail_call(service, backing_id)
    service.release(backing_id)
    await _wait_for_condition(
        pilot,
        lambda: screen._library_media_reader_session.loaded_id == canonical_id,
        message=f"Row {to_index} never settled after traversal.",
    )
    await pilot.pause()
    return canonical_id, backing_id


async def _submit_query(screen, pilot, query: str) -> None:
    """Type ``query`` into the content search box and press Enter.

    task-31237: the Find bar is collapsed until the Find action opens it,
    so the helper presses Find first when the input is not yet mounted.
    """
    if not screen.query("#library-media-content-search"):
        screen.query_one("#library-media-reader-find", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-content-search")),
            message="Find never mounted the search input.",
        )
    search_input = screen.query_one("#library-media-content-search", Input)
    search_input.focus()
    await pilot.pause()
    search_input.value = query
    await pilot.press("enter")
    await pilot.pause()


def _raw_static(screen) -> VirtualizedRawContent:
    """Return the mounted Raw view.

    task-22500: this returned the plain ``Static`` the reader used to mount
    for its Raw view; that widget is now ``VirtualizedRawContent`` (a
    ``ScrollView`` with no ``.renderable``), so callers that used to read
    Rich ``Text`` spans off the return value now read its live paint output
    through ``render_line`` instead (see ``_highlighted_words_in_raw``).
    """
    body = screen.query_one("#library-media-viewer-content", LibraryMediaContentBody)
    return body.query_one(
        "#library-media-viewer-content-text", VirtualizedRawContent
    )


def _status_text(screen) -> str:
    status = screen.query_one("#library-media-content-search-status", Static)
    return str(status.renderable)


def _highlighted_words_in_raw(raw: VirtualizedRawContent) -> set[str]:
    """Collect every substring the mounted Raw view currently paints reversed.

    task-22500: the widget has no single ``Text`` renderable to inspect --
    each visible row is restyled at paint time by ``render_line`` -- so this
    walks every row through the SAME method Textual's compositor calls and
    collects the segments carrying the match style (``Style(reverse=True)``,
    which both the plain and active match styles set), the direct
    replacement for reading spans off a Rich ``Text``.
    """
    if raw.wrap_index is None:
        return set()
    words: set[str] = set()
    for row in range(raw.wrap_index.virtual_height):
        strip = raw.render_line(row)
        for segment in strip._segments:
            if segment.style is not None and segment.style.reverse:
                words.add(segment.text)
    return words


# ---------------------------------------------------------------------------
# AC#1: at most one O(document) scan per click (here: none -- both the match
# list and the highlight spans are cached for the open (document, query)).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_match_navigation_takes_no_document_pass_per_click():
    """Six Prev/Next clicks after a submitted query re-read nothing."""
    app, service = _flow_app(count=12)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_with_document(screen, pilot, service, 0, _document())
        await _submit_query(screen, pilot, NEEDLE)

        next_button = screen.query_one("#library-media-content-search-next", Button)
        prev_button = screen.query_one("#library-media-content-search-prev", Button)
        per_click: list[int] = []
        with _count_document_passes() as counts:
            for step, button in enumerate((next_button,) * 4 + (prev_button,) * 2):
                before = _total(counts)
                button.press()
                await pilot.pause()
                per_click.append(_total(counts) - before)

        assert per_click == [0] * 6, (
            "Match navigation re-read the document per click: "
            f"{per_click} pass(es) per click (counts={counts})."
        )


@pytest.mark.asyncio
async def test_match_navigation_patches_the_highlight_instead_of_rebuilding_it():
    """The mounted Raw view keeps its identity; only its search state moves.

    task-22500: the virtualized Raw view holds no whole-document Rich
    ``Text`` to patch anymore -- each visible row restyles itself from
    ``query``/``match_index`` when Textual paints it -- so "patched, not
    rebuilt" is now pinned at the WIDGET (it must never be replaced by a
    match-nav click) rather than at a ``Text`` object's spans. Which row
    paints ACTIVE-vs-plain styling is task 7's ``set_match_lines`` wiring,
    not yet reachable at this stage.
    """
    app, service = _flow_app(count=12)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_with_document(screen, pilot, service, 0, _document())
        await _submit_query(screen, pilot, NEEDLE)

        raw = _raw_static(screen)
        assert raw._query == NEEDLE
        assert raw._match_index == 0
        assert _highlighted_words_in_raw(raw) == {NEEDLE}

        screen.query_one("#library-media-content-search-next", Button).press()
        await pilot.pause()

        second = _raw_static(screen)
        assert second is raw, (
            "A match click rebuilt the Raw view instead of patching its "
            "search state."
        )
        assert second._match_index == 1
        assert _highlighted_words_in_raw(second) == {NEEDLE}


# ---------------------------------------------------------------------------
# The cache keys: query and document identity both have to be in them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_new_query_rehighlights_the_document():
    """A second query must not reuse the first query's cached match list."""
    app, service = _flow_app(count=12)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_with_document(screen, pilot, service, 0, _document())
        await _submit_query(screen, pilot, NEEDLE)
        screen.query_one("#library-media-content-search-next", Button).press()
        await pilot.pause()
        assert _status_text(screen) == "Match 2 of 50 matches"

        await _submit_query(screen, pilot, OTHER_NEEDLE)

        second = _raw_static(screen)
        assert _highlighted_words_in_raw(second) == {OTHER_NEEDLE}, (
            "The second query reused the first query's highlight spans."
        )
        assert _status_text(screen) == "Match 1 of 20 matches", (
            "The screen reused the first query's match list."
        )
        screen.query_one("#library-media-content-search-next", Button).press()
        await pilot.pause()
        assert _status_text(screen) == "Match 2 of 20 matches"


@pytest.mark.asyncio
async def test_a_new_document_rescans_for_the_same_query():
    """Traversing to another item with a live query must rescan the document."""
    app, service = _flow_app(count=12)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_with_document(screen, pilot, service, 0, _document(400))
        _seed_row_document(screen, service, 1, _document(80))
        await _submit_query(screen, pilot, NEEDLE)
        assert _status_text(screen) == "Match 1 of 50 matches"

        await _traverse_to_row(screen, pilot, service, from_index=0, to_index=1)
        assert screen._library_media_content_query == NEEDLE
        screen.query_one("#library-media-content-search-next", Button).press()
        await pilot.pause()

        assert _status_text(screen) == "Match 2 of 10 matches", (
            "Match navigation reused the previous document's match list."
        )
        highlighted = _highlighted_words_in_raw(_raw_static(screen))
        assert highlighted == {NEEDLE}


# ---------------------------------------------------------------------------
# Teardown / failure walk.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clearing_the_search_mid_navigation_drops_every_highlight():
    """Submitting an empty query mid-navigation restores the plain document."""
    app, service = _flow_app(count=12)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        content = _document()
        await _load_row_with_document(screen, pilot, service, 0, content)
        await _submit_query(screen, pilot, NEEDLE)
        screen.query_one("#library-media-content-search-next", Button).press()
        await pilot.pause()

        await _submit_query(screen, pilot, "")

        assert screen._library_media_content_query == ""
        # task-28002: the status child persists display-gated (tearing it
        # down recomposed away the focused Input), so "gone" means hidden.
        assert not screen.query_one("#library-media-content-search-status").display
        raw = _raw_static(screen)
        assert raw._query == ""
        assert _highlighted_words_in_raw(raw) == set()
        # Highlights gone is only half the claim -- the DOCUMENT must survive
        # the clear too. The retired assertion read `str(raw.renderable)`,
        # which the virtualized view no longer has; select-all is the
        # equivalent read of its full text.
        selected, _ = raw.get_selection(Selection(None, None))
        assert selected == content

        # A stray advance after the clear is a no-op, not a crash.
        screen._advance_library_media_content_match(1)
        await pilot.pause()
        assert _highlighted_words_in_raw(_raw_static(screen)) == set()


@pytest.mark.asyncio
async def test_a_cleared_detail_releases_the_cached_match_list():
    """The memo must not pin a document the reader has already let go of.

    The resets that leave the media reader (rail switch, delete) clear
    ``_library_media_detail``; the next match lookup has to drop the cached
    entry rather than hold the previous document's content alive behind it.
    """
    app, service = _flow_app(count=12)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_with_document(screen, pilot, service, 0, _document())
        await _submit_query(screen, pilot, NEEDLE)
        assert screen._library_media_content_match_memo is not None

        screen._library_media_detail = None

        assert screen._library_media_content_matches() == ()
        assert screen._library_media_content_match_memo is None
        # And a stray navigation against no open item is a no-op, not a crash.
        screen._advance_library_media_content_match(1)
        await pilot.pause()


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_multi_megabyte_document_match_navigation_timings():
    """Record per-click wall on a multi-MB document; pin the pass count."""
    app, service = _flow_app(count=12)
    big_document = "\n".join(
        (
            f"line {index}: the {NEEDLE} sits here"
            if index % 8 == 3
            else f"line {index}: ordinary reading material padded out to a "
            "realistic transcript width so the document is genuinely large"
        )
        for index in range(24_000)
    )
    assert len(big_document) > 2_000_000
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_with_document(screen, pilot, service, 0, big_document)
        await _submit_query(screen, pilot, NEEDLE)

        per_click_ms: list[float] = []
        with _count_document_passes() as counts:
            for _ in range(6):
                started = perf_counter()
                screen._advance_library_media_content_match(1)
                per_click_ms.append((perf_counter() - started) * 1000.0)
            passes = _total(counts)

        print(
            f"TASK-22209 {len(big_document) / 1_000_000:.1f}MB per-click: "
            f"median={median(per_click_ms):.3f}ms samples="
            + ",".join(f"{sample:.3f}" for sample in per_click_ms)
        )
        assert passes == 0, (
            f"Six clicks on a multi-MB document took {passes} document pass(es)."
        )
