"""Characterization pins for genuinely-unpressed Library Media handlers.

Wave-7 Task 1 (media series 1/3, state PR; recipe: ``backlog/docs/
library-decomposition-recipe.md``; the collections/ingest/prompts series'
own ``test_library_*_characterization.py`` files are the precedent this
mirrors). These pins exist BEFORE the Media extraction moves any state, so a
later move that silently breaks one of these ``@on`` dispatch paths goes red
rather than green-but-vacuous.

**Scope, and why it is only three tests.** Media is by far the widest
cluster this program has processed -- **79** ``@on``-decorated media-named
handlers over **76** distinct selectors, against ~20 dedicated
``Tests/UI/test_library_media_*.py`` files plus ``test_library_shell.py``,
``test_library_multiselect_media.py``, ``test_review_set_walker.py`` and the
canvas suites. The census ran every selector against ALL of ``Tests/`` with
exact id/class boundaries (a substring match scores ``#library-media-review``
as covered off ``#library-media-review-selected``), widened the
press-detection window to +/-4 lines, additionally derived the ``#<id>-N``
row spelling for the four CLASS-bound handlers (rows are pressed by id, never
by class), and separately grepped each handler's own METHOD NAME for the
unbound-``LibraryScreen.<name>(fake, event)`` call shape several media suites
favour. Every non-``COVERED`` verdict was then read rather than trusted.

Result: **73 of 79 already covered** -- 60 by a real selector press, 13 more
only by a direct unbound call. Of the remaining 6, three touch NO moved
``LibraryMediaState`` field at all (``handle_library_media_review_these``,
``handle_library_media_review_sets`` -- both are a bare ``event.stop()`` plus
``run_worker``; and ``handle_library_media_reader_mode``, which the census
first scored as a gap and a read overturned: it is class-bound
(``.library-media-reader-mode``) but pressed by id
(``#library-media-reader-select-analysis``) in
``test_library_reader_press_scope_t22228.py``, the same
bound-by-class/pressed-by-id shape the prompts series recorded).

The three genuine gaps pinned below, each of which reads state this PR
moves:

- ``handle_library_media_previous`` (``@on(Button.Pressed,
  "#library-media-previous")``) -- the pager's Previous action was queried
  eight times across two files for ``.disabled``, ``.tooltip`` and focus
  identity, and pressed zero times. Reads ``bulk_delete_in_flight``.
- ``handle_library_media_review_selected`` (``@on(Button.Pressed,
  "#library-media-review-selected")``) -- queried twice in
  ``test_library_media_toolbar_adapt.py`` for its label and its geometry, on
  a standalone canvas host (which cannot dispatch a SCREEN ``@on`` handler at
  all). Reads ``bulk_delete_in_flight`` (through a ``getattr`` literal) and
  ``row_selection``.
- ``handle_library_media_open_original`` (``@on(Button.Pressed,
  "#library-media-open-original")``) -- zero references anywhere in the repo
  outside its own decorator and the widget that yields the button. Reads
  ``detail``.

No live bugs were found writing these: all three are coverage gaps, not
behavior bugs. Each test asserts through a signal provably tied to the
handler's own logic (the pager's applied page, the exact ``id_allowlist``
the selection produces, the URL handed to ``webbrowser.open``) rather than a
bare DOM end-state, per recipe §3's warning that an end-state assertion can
be satisfied by an unrelated coincidence.

``webbrowser.open`` is patched on the ``webbrowser`` MODULE itself, never at
the ``library_screen``-scoped path -- deliberately, so the patch keeps
reaching the call after this series' controller PR moves that body to another
module (recipe §3's eighth bypass shape, avoided by construction rather than
discovered later).
"""

from __future__ import annotations

import webbrowser

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_media_state import MediaBrowseScope


def _paged_media_items(count: int = 45) -> list[dict[str, object]]:
    """Deterministic production-shaped rows spanning three exact pages."""
    return [
        {
            "id": f"media-{index}",
            "title": f"Media {index:02d}",
            "type": "document",
            "last_modified": f"2026-08-{index:02d}T00:00:00Z",
        }
        for index in range(1, count + 1)
    ]


def _sourced_media_items() -> list[dict[str, object]]:
    """Two rows whose stored ``url`` makes "Open original" composable."""
    return [
        {
            "id": "media-1",
            "title": "Interview Recording",
            "type": "audio",
            "last_modified": "2026-07-06T08:00:00Z",
            "author": "Jordan Lee",
            "keywords": ["interview", "audio"],
            "url": "https://example.invalid/interview-recording",
            "content": "Full transcript: the interview covers the roadmap.",
            "version": 1,
        },
        {
            "id": "media-2",
            "title": "Product Demo Video",
            "type": "video",
            "last_modified": "2026-07-06T10:00:00Z",
            "author": "Morgan Lee",
            "keywords": ["demo", "video"],
            "url": "https://example.invalid/product-demo",
            "content": "Full transcript: the demo walks through the dashboard.",
            "version": 2,
        },
    ]


async def _open_media_list(host, pilot):
    """Mount the Library shell, select Browse Media, await its first rows."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    if screen.query("#library-rail-explore-all"):
        screen.query_one("#library-rail-explore-all", Button).press()
    await _wait_for_selector(screen, pilot, "#library-row-browse-media")
    screen.query_one("#library-row-browse-media", Button).press()
    await _wait_for_selector(screen, pilot, "#library-media-row-0")
    return screen


def _page_status(screen) -> str:
    return str(screen.query_one("#library-media-page-status", Static).renderable)


@pytest.mark.asyncio
async def test_media_previous_page_returns_to_the_page_before_it() -> None:
    """"‹ Previous" walks the exact pager back, one page per press.

    The pager's Next action is pressed by several existing tests; Previous
    was only ever asserted disabled. This presses it for real and pins the
    applied scope AND the rendered page status, so a move that broke the
    handler's own ``applied.page - 1`` request would fail here instead of
    passing on an unrelated Next-driven assertion.
    """
    app = _build_test_app()
    _seed_conversations(app, [], media=_paged_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        controller = screen._library_media_browse_controller
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.applied_scope == MediaBrowseScope(page=1)
                and not screen.query_one("#library-media-next", Button).disabled
            ),
            message="Exact first Media page never applied with Next enabled.",
        )

        screen.query_one("#library-media-next", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.applied_scope == MediaBrowseScope(page=2)
                and _page_status(screen) == "21-40 of 45 · Page 2 of 3"
            ),
            message="Media page 2 never applied.",
        )

        previous = screen.query_one("#library-media-previous", Button)
        assert previous.disabled is False
        previous.press()
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.applied_scope == MediaBrowseScope(page=1)
                and _page_status(screen) == "1-20 of 45 · Page 1 of 3"
            ),
            message="Previous never returned the exact Media page to 1.",
        )
        assert len(screen.query(".library-media-row")) == 20


@pytest.mark.asyncio
async def test_media_review_selected_orders_exactly_the_selected_ids() -> None:
    """"Review" pins the CURRENT Select-mode selection, not the whole page.

    The observable is the ``id_allowlist`` the handler's own worker hands to
    the media scope service -- a signal that can only be produced by reading
    ``_library_media_row_selection`` after the dispatch reached the handler,
    unlike a DOM end-state assertion. Storage for the resulting review set is
    deliberately absent, so the run stops right after that call.
    """
    app = _build_test_app()
    _seed_conversations(app, [], media=_sourced_media_items())
    service = app.media_reading_scope_service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-select-toggle", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-select-all")

        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_row_selection.count == 1,
            message="A Select-mode row press never landed in the selection.",
        )
        selected = sorted(
            int(str(row_id).rsplit(":", 1)[-1])
            for row_id in screen._library_media_row_selection.ids
        )

        await _wait_for_selector(screen, pilot, "#library-media-review-selected")
        before = len(service.search_calls)
        screen.query_one("#library-media-review-selected", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: any(
                call.get("id_allowlist") == selected
                for call in service.search_calls[before:]
            ),
            message=(
                "Review selected never ordered the selection: "
                f"expected id_allowlist={selected!r}, saw "
                f"{[c.get('id_allowlist') for c in service.search_calls[before:]]!r}"
            ),
        )


@pytest.mark.asyncio
async def test_media_open_original_opens_the_stored_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Open original" hands the open item's stored URL to the browser.

    Patched on the ``webbrowser`` module itself (see this file's own
    docstring), so the pin survives the handler moving modules.
    """
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url))

    app = _build_test_app()
    _seed_conversations(app, [], media=_sourced_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-more")
        await _wait_for_condition(
            pilot,
            lambda: bool(screen._library_media_detail),
            message="The media viewer never received its detail.",
        )
        expected = str(screen._library_media_detail.get("url"))
        assert expected.startswith("https://")

        screen.query_one("#library-media-reader-more", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-open-original")
        screen.query_one("#library-media-open-original", Button).press()
        await pilot.pause()

        assert opened == [expected]
