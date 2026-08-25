"""TASK-22207: focus traversal must not rebuild the Reader document body.

The permanent Media Reader opens on FOCUS (PR #2064), so arrow-keying the
Items list drives ``_select_library_media_reader_row`` per keystroke. The
settle timer debounces only the DB fetch -- before this task, each
keystroke's ``_sync_library_media_viewer_or_recompose()`` fell through the
``unchanged`` comparison on the just-flipped loading flag and rebuilt the
FULL document body (a fresh ``LibraryMediaContentBody``; in rendered mode a
fresh ``Markdown`` parse of the document being LEFT) purely to paint the
"Loading..." banner, and the settle then rebuilt it a second time.

These probes count ``LibraryMediaContentBody`` constructions (and
``Markdown`` constructions) on the mounted harness: pass-through rows must
build ZERO bodies; only the settled row builds one, once. The loading
banner is asserted to paint IN PLACE (display-gated persistent ``Static``,
same body widget identity) rather than via recompose.

No wall-clock thresholds are asserted (the 15457 probe rule); the 1 MB
fixture test prints its per-keystroke timings for the task record and pins
the build count, which is what actually guarantees "no per-keystroke
parse".
"""

from __future__ import annotations

from contextlib import contextmanager
from statistics import median
from time import perf_counter
from unittest.mock import patch

import pytest
from textual.widgets import Button, Markdown, Static

from Tests.UI.test_library_media_reader_flow import (
    ControlledDetailMediaService,
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
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
)
from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentBody,
)


@contextmanager
def _count_body_builds():
    """Count document-body and Markdown constructions inside the block.

    Wraps the two ``__init__``s via ``patch.object`` -- neither is an
    ``@on``-decorated handler, so class-level patching is dispatch-safe
    (the lessons-textual trap applies to message handlers only).
    """
    counts = {"body": 0, "markdown": 0}
    body_init = LibraryMediaContentBody.__init__
    markdown_init = Markdown.__init__

    def counting_body(self, *args, **kwargs):
        counts["body"] += 1
        return body_init(self, *args, **kwargs)

    def counting_markdown(self, *args, **kwargs):
        counts["markdown"] += 1
        return markdown_init(self, *args, **kwargs)

    with (
        patch.object(LibraryMediaContentBody, "__init__", counting_body),
        patch.object(Markdown, "__init__", counting_markdown),
    ):
        yield counts


def _loading_banner_displayed(screen) -> bool:
    """True when the Reader's pending banner is mounted AND painted."""
    banners = screen.query("#library-media-viewer-loading")
    return bool(banners) and banners.first().display


async def _load_row(screen, pilot, service: ControlledDetailMediaService, index: int):
    """Press row ``index`` and settle its detail; return its identity."""
    row = screen.query_one(f"#library-media-row-{index}", Button)
    canonical_id, backing_id, title = _row_identity(row)
    row.press()
    await _wait_for_detail_call(service, backing_id)
    service.release(backing_id)
    await _wait_for_condition(
        pilot,
        lambda: screen._library_media_reader_session.loaded_id == canonical_id,
        message=f"Row {index} never settled its detail.",
    )
    return canonical_id, backing_id, title


def _release_everything(service: ControlledDetailMediaService) -> None:
    for media_id in tuple(service.detail_release):
        service.release(media_id)


@pytest.mark.asyncio
async def test_focus_traversal_builds_zero_bodies_for_pass_through_rows():
    """10-row focus traversal: 0 body builds pass-through, 1 for the settled row."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row(screen, pilot, service, 0)

        final_id, final_backing_id, _ = _row_identity(
            screen.query_one("#library-media-row-10", Button)
        )
        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        with _count_body_builds() as counts:
            for _ in range(10):
                await pilot.press("down")
            pass_through_builds = counts["body"]
            await _wait_for_detail_call(service, final_backing_id)
            _release_everything(service)
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_reader_session.loaded_id == final_id,
                message="The final traversal row never settled.",
            )
            await pilot.pause()

        assert pass_through_builds == 0, (
            "Pass-through focus rows rebuilt the document body "
            f"{pass_through_builds} time(s); traversal must build zero."
        )
        assert counts["body"] == 1, (
            f"Expected exactly one settled-row body build, saw {counts['body']}."
        )


@pytest.mark.asyncio
async def test_loading_banner_paints_in_place_without_body_rebuild():
    """The pending banner shows and hides without recomposing the body."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        _, _, title_a = await _load_row(screen, pilot, service, 0)
        body_before = screen.query_one(
            "#library-media-viewer-content", LibraryMediaContentBody
        )

        row_b = screen.query_one("#library-media-row-1", Button)
        canonical_b, backing_b, title_b = _row_identity(row_b)
        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        with _count_body_builds() as counts:
            await pilot.press("down")
            await _wait_for_condition(
                pilot,
                lambda: _loading_banner_displayed(screen),
                message="The pending banner never painted on focus.",
            )
            assert counts["body"] == 0, (
                "Painting the loading banner rebuilt the document body "
                f"({counts['body']} build(s)); it must patch in place."
            )
            assert (
                screen.query_one(
                    "#library-media-viewer-content", LibraryMediaContentBody
                )
                is body_before
            ), "The body widget was replaced just to show the loading banner."
            banner_copy = str(
                screen.query_one("#library-media-viewer-loading", Static).content
            )
            assert title_b in banner_copy
            assert title_a in banner_copy

            await _wait_for_detail_call(service, backing_b)
            service.release(backing_b)
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_reader_session.loaded_id
                == canonical_b,
                message="Row B never settled.",
            )
            await _wait_for_condition(
                pilot,
                lambda: not _loading_banner_displayed(screen),
                message="The banner stayed painted after the settle.",
            )
        assert counts["body"] == 1


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_one_megabyte_markdown_document_is_not_reparsed_per_keystroke():
    """Traversing past a loaded ~1 MB rendered document builds no bodies.

    The build-count assertion is the guarantee; the printed per-keystroke
    timings are task-record evidence only (no wall-clock threshold -- the
    15457 probe rule).
    """
    app, service = _flow_app(count=12)
    big_markdown = "# Big document\n\n" + (
        "A paragraph of steady reading material that pads the document "
        "toward one megabyte of markdown body text.\n\n" * 10000
    )
    assert len(big_markdown) > 1_000_000
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        # Rows sort by recency, so resolve which item row 0 actually shows
        # and hand THAT one the 1 MB markdown body before it is fetched.
        _, backing_id_0, _ = _row_identity(
            screen.query_one("#library-media-row-0", Button)
        )
        source = next(
            item
            for item in service.media_items
            if item["id"] == f"media-{backing_id_0}"
        )
        source["type"] = "markdown"
        source["content"] = big_markdown
        await _load_row(screen, pilot, service, 0)
        assert screen._library_media_content_mode == "rendered"
        assert screen.query_one("#library-media-viewer-content-markdown", Markdown)

        final_id, final_backing_id, _ = _row_identity(
            screen.query_one("#library-media-row-6", Button)
        )
        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        per_keystroke_ms: list[float] = []
        with _count_body_builds() as counts:
            for _ in range(6):
                started = perf_counter()
                await pilot.press("down")
                per_keystroke_ms.append((perf_counter() - started) * 1000.0)
            traversal_builds = counts["body"]
            traversal_parses = counts["markdown"]
            # Print BEFORE the settle wait so the red-first run still
            # records the per-keystroke numbers when the settle drowns in
            # the parse backlog it is red about.
            print(
                "TASK-22207 1MB traversal per-keystroke: "
                f"median={median(per_keystroke_ms):.3f}ms samples="
                + ",".join(f"{sample:.3f}" for sample in per_keystroke_ms)
            )
            await _wait_for_detail_call(service, final_backing_id)
            _release_everything(service)
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_reader_session.loaded_id == final_id,
                # Generous deadline: the PRE-fix tree queues a fresh 1 MB
                # Markdown parse per keystroke, and the settle has to drain
                # that backlog first -- the red run must still reach the
                # timing printout below to record the before numbers.
                timeout=180.0,
                message="The settled row never loaded past the 1 MB document.",
            )
            await pilot.pause()

        assert traversal_builds == 0, (
            f"Traversing past a 1 MB document rebuilt its body "
            f"{traversal_builds} time(s)."
        )
        assert traversal_parses == 0, (
            f"Traversing past a 1 MB document re-parsed Markdown "
            f"{traversal_parses} time(s)."
        )
        assert counts["body"] == 1


@pytest.mark.asyncio
async def test_settle_after_route_change_leaves_media_surface_alone():
    """A detail resolving after the route moved paints nothing anywhere."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row(screen, pilot, service, 0)
        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        await pilot.press("down")

        screen.query_one("#library-row-browse-conversations", Button).press()
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS

        with _count_body_builds() as counts:
            _release_everything(service)
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()
        assert counts["body"] == 0, (
            "A settle landing after the route changed rebuilt a media body."
        )
        assert screen.query("#library-conversations-canvas")
        assert not screen.query("#library-media-viewer")


@pytest.mark.asyncio
async def test_stale_failure_after_selection_moved_on_paints_no_error():
    """A stale fetch failing never paints an error or a body for a dead row."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row(screen, pilot, service, 0)
        canonical_b, backing_b, title_b = _row_identity(
            screen.query_one("#library-media-row-1", Button)
        )
        canonical_c, backing_c, title_c = _row_identity(
            screen.query_one("#library-media-row-2", Button)
        )
        screen._select_library_media_reader_row(canonical_b, title_b, immediate=True)
        await _wait_for_detail_call(service, backing_b)
        screen._select_library_media_reader_row(canonical_c, title_c, immediate=True)
        await _wait_for_detail_call(service, backing_c)

        with _count_body_builds() as counts:
            service.release(backing_b, RuntimeError("stale failure"))
            await pilot.pause()
            await pilot.pause()
            assert screen._library_media_reader_session.error is None
            assert not screen.query("#library-media-viewer-error")
            assert counts["body"] == 0

            service.release(backing_c)
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_reader_session.loaded_id
                == canonical_c,
                message="Row C never settled after the stale failure.",
            )
            await pilot.pause()
        assert screen._library_media_reader_session.error is None
        assert counts["body"] == 1


@pytest.mark.asyncio
async def test_fast_alternating_focus_settles_identical_content_without_rebuild():
    """A -> B -> A -> B -> A alternation re-settles A with zero body rebuilds."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        canonical_a, backing_a, _ = await _load_row(screen, pilot, service, 0)
        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()

        with _count_body_builds() as counts:
            for key in ("down", "up", "down", "up"):
                await pilot.press(key)
            _release_everything(service)
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_media_reader_session.loaded_id == canonical_a
                    and screen._library_media_reader_session.pending_request is None
                ),
                message="Alternating focus never re-settled row A.",
            )
            await pilot.pause()

        # The re-fetched detail is a NEW dict with identical values; the
        # display-state comparison must be structural, so the identical
        # document is NOT rebuilt and no stale body wins.
        assert counts["body"] == 0, (
            f"Re-settling identical content rebuilt the body {counts['body']} "
            "time(s); the unchanged comparison must be structural."
        )
        assert len(screen.query("#library-media-viewer-content")) == 1
        assert not _loading_banner_displayed(screen)
        assert screen._library_media_reader_session.error is None
