"""Real-DB reproduction for task-31220: the bulk-delete write, as seen by a fresh reader.

Critique #5 reported a ``✓ deleted · 1 item · in Trash`` receipt over a row a
direct DB query still showed with ``is_trash=0``. These tests answer three
factual questions against a REAL ``MediaDatabase`` under ``tmp_path`` and a
REAL mounted ``LibraryScreen`` (never the user's database):

1. is the write visible to a *fresh* connection right after the receipt?
2. can a long-lived ``MediaDatabase`` opened before the commit (the
   assessments' own read method) still report the OLD value?
3. under a contended write lock, does the receipt correctly stay empty and
   the ``_library_media_bulk_delete_in_flight`` interlock release?

The oracle is always a brand-new ``sqlite3.connect`` -- never the connection
the app wrote through.
"""

import sqlite3

import pytest
from loguru import logger

from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Media import LocalMediaReadingService, MediaReadingScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
)
from textual.widgets import Button, Static


def _seed(db: MediaDatabase, count: int) -> list[int]:
    """Insert ``count`` active media rows, returning their backing ids."""
    ids: list[int] = []
    for index in range(count):
        media_id, _, _ = db.add_media_with_keywords(
            title=f"BulkDelete seed {index}",
            content=f"bulk delete seed body {index} unique",
            media_type="article",
            keywords=["bulk"],
        )
        ids.append(int(media_id))
    return ids


def _real_media_host(tmp_path, *, items: int):
    """Mount a real ``LibraryScreen`` over a real file-backed ``MediaDatabase``.

    Returns ``(host, screen, db, db_path)``. The screen is built before mount
    so callers hold it without querying the screen stack; ``screen_media_ids``
    is only meaningful once ``_browse_media`` has applied a Media page.
    """
    db_path = str(tmp_path / "media.db")
    db = MediaDatabase(db_path=db_path, client_id="task-31220")
    _seed(db, items)

    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=[])
    app.media_reading_scope_service = MediaReadingScopeService(
        LocalMediaReadingService(db), None
    )
    # Record notifications instead of routing them through a non-running App.
    notified: list[tuple[str, dict]] = []
    app.notify = lambda message, **kwargs: notified.append((str(message), kwargs))
    app.library_media_notifications = notified

    screen = LibraryScreen(app)
    host = LibraryHarness(app, screen=screen)
    return host, screen, db, db_path


async def _browse_media(screen, pilot) -> None:
    """Open Browse ▸ Media and wait for a real page to apply."""
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-media", Button).press()
    controller = screen._library_media_browse_controller
    await _wait_for_condition(
        pilot,
        lambda: controller.applied_result is not None,
        message="Media page never applied from the real service.",
    )


def screen_media_ids(screen) -> tuple[str, ...]:
    """The canonical ``local:media:<id>`` ids the Media list is showing."""
    return tuple(
        str(item["id"])
        for item in screen._library_media_browse_controller.retained_items
    )


def _backing(media_id: str) -> int:
    return int(media_id.rsplit(":", 1)[1])


def _fresh_is_trash(db_path: str, media_id: int) -> int:
    """Read ``is_trash`` over a BRAND NEW connection -- the durable oracle."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT is_trash FROM Media WHERE id=?", (media_id,)
        ).fetchone()
    finally:
        conn.close()
    return int(row[0])


async def _run_bulk_delete(screen, media_ids: tuple[str, ...]) -> None:
    """Drive the worker body exactly as ``handle_library_media_bulk_delete_confirm``
    does: arm the interlock and open the mutation fence first."""
    screen._library_media_bulk_delete_in_flight = True
    screen._begin_library_media_mutation()
    await LibraryScreen._delete_library_media_selection(screen, media_ids)


@pytest.mark.asyncio
async def test_bulk_delete_write_is_visible_to_a_fresh_reader_and_receipt_matches(
    tmp_path,
):
    """Finding 1: single-instance, a ✓ receipt implies a durable is_trash=1."""
    host, screen, db, db_path = _real_media_host(tmp_path, items=3)
    try:
        async with host.run_test(size=(235, 52)) as pilot:
            await _browse_media(screen, pilot)
            ids = screen_media_ids(screen)
            assert len(ids) == 3
            target = ids[0]
            assert _fresh_is_trash(db_path, _backing(target)) == 0

            await _run_bulk_delete(screen, (target,))
            await pilot.pause()

            assert _fresh_is_trash(db_path, _backing(target)) == 1
            assert screen._library_media_delete_receipt_ids == (target,)
            assert screen._library_media_bulk_delete_in_flight is False
            assert host.app_instance.library_media_notifications == []
            # Every other seeded row is untouched.
            for other in ids[1:]:
                assert _fresh_is_trash(db_path, _backing(other)) == 0
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_long_lived_reader_connection_can_miss_the_write_the_receipt_reports(
    tmp_path,
):
    """Finding 2: reproduces the critique-#5 assessments' read method -- a
    ``MediaDatabase`` opened (and read through) BEFORE the app's commit. The
    fresh connection is the oracle; whatever the long-lived reader sees is
    recorded, not asserted."""
    host, screen, db, db_path = _real_media_host(tmp_path, items=2)
    reader = MediaDatabase(db_path=db_path, client_id="assessor")
    try:
        async with host.run_test(size=(235, 52)) as pilot:
            await _browse_media(screen, pilot)
            ids = screen_media_ids(screen)
            target = ids[0]
            backing = _backing(target)

            before = reader.get_media_by_id(backing, include_trash=True)
            assert int(before["is_trash"]) == 0  # the reader is open and warm

            await _run_bulk_delete(screen, (target,))
            await pilot.pause()

            assert screen._library_media_delete_receipt_ids == (target,)
            stale_view = int(
                reader.get_media_by_id(backing, include_trash=True)["is_trash"]
            )
            fresh_view = _fresh_is_trash(db_path, backing)
            print(
                f"[task-31220] long-lived reader saw is_trash={stale_view}; "
                f"fresh connection saw is_trash={fresh_view}"
            )
            assert fresh_view == 1
    finally:
        reader.close_connection()
        db.close_connection()


@pytest.mark.asyncio
async def test_contended_write_never_paints_a_success_receipt(tmp_path):
    """Finding 3: another instance holds the write lock -- no ✓, the per-item
    failure is logged and notified, and the interlock still releases.

    The contended write fails immediately -- measured 1.05s for the test
    body, so no busy timeout is waited out. ``mark_as_trash`` runs under the
    house DEFERRED ``BEGIN``, and SQLite cannot park a read transaction
    waiting to upgrade to a writer (it would deadlock), so the promotion
    returns ``SQLITE_BUSY`` at once rather than honouring ``busy_timeout``.
    """
    host, screen, db, db_path = _real_media_host(tmp_path, items=2)
    other = sqlite3.connect(db_path, isolation_level=None)
    warnings: list[dict] = []
    sink = logger.add(
        lambda message: warnings.append(dict(message.record)),
        level="WARNING",
        filter=lambda record: record["name"].endswith("library_screen"),
    )
    try:
        other.execute("BEGIN IMMEDIATE")
        async with host.run_test(size=(235, 52)) as pilot:
            await _browse_media(screen, pilot)
            ids = screen_media_ids(screen)
            target = ids[0]

            await _run_bulk_delete(screen, (target,))
            await pilot.pause()

            assert screen._library_media_delete_receipt_ids == ()
            assert screen._library_media_bulk_delete_in_flight is False
            assert _fresh_is_trash(db_path, _backing(target)) == 0
            notified = host.app_instance.library_media_notifications
            assert [message for message, _ in notified] == [
                "Could not delete 1 of 1 selected media item."
            ]
            assert all(kwargs.get("severity") == "warning" for _, kwargs in notified)
            # The per-item failure itself, not just its aggregate side
            # effects: the delete loop's own warning must have been raised.
            assert "Failed to delete Library media item in bulk delete." in [
                record["message"] for record in warnings
            ]
    finally:
        logger.remove(sink)
        try:
            other.execute("ROLLBACK")
        finally:
            other.close()
        db.close_connection()


@pytest.mark.asyncio
async def test_rows_still_open_while_the_page_sits_behind_the_stale_gate(tmp_path):
    """task-31220: recovery is never gated by what it recovers from.

    Critique #5 sat behind ``Media changed; retry to load a current page.``
    with every row disabled -- the gate is about a page whose ORDER and
    MEMBERSHIP may have moved, which says nothing about whether an item the
    list is still showing can be read. Pressing a row must load it.
    """
    host, screen, db, db_path = _real_media_host(tmp_path, items=3)
    try:
        async with host.run_test(size=(235, 52)) as pilot:
            await _browse_media(screen, pilot)
            controller = screen._library_media_browse_controller
            target = screen_media_ids(screen)[1]

            # Exactly the state a committed bulk delete leaves behind.
            screen._begin_library_media_mutation()
            controller.reconcile_committed_mutation(remove_ids=())
            screen._library_media_mutation_scope = None
            screen._library_media_mutation_authority = None
            screen._sync_library_media_browse_state(None)
            await pilot.pause()

            assert controller.freshness == "stale"
            assert controller.stale_copy == (
                "Media changed; retry to load a current page."
            )
            assert screen._library_media_bulk_delete_in_flight is False

            row = next(
                button
                for button in screen.query(".library-media-row")
                if str(getattr(button, "media_id", "")) == target
            )
            assert row.disabled is False
            row.press()
            await pilot.pause()

            assert screen._selected_media_id == target
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_retry_paints_a_new_reason_every_time_the_refresh_fails(tmp_path):
    """Step 5's substitute for the live outage: a failing request seam.

    The briefed live simulation (a scratch profile whose media DB path is a
    directory) cannot reach this state at all -- with no first page there is
    no stale gate, only the cold ``Couldn't load media`` error. So the outage
    is injected here instead, at the real mounted surface: a real page
    applies, the service then times out, and the pager's own status Static is
    read after each of two Retry presses.
    """
    host, screen, db, db_path = _real_media_host(tmp_path, items=3)
    try:
        async with host.run_test(size=(235, 52)) as pilot:
            await _browse_media(screen, pilot)
            controller = screen._library_media_browse_controller

            # Stale the page exactly as a committed mutation does.
            screen._begin_library_media_mutation()
            controller.reconcile_committed_mutation(remove_ids=())
            screen._library_media_mutation_scope = None
            screen._library_media_mutation_authority = None
            screen._sync_library_media_browse_state(None)
            await pilot.pause()
            status = screen.query_one("#library-media-status", Static)
            assert "Media changed" in str(status.renderable)

            service = host.app_instance.media_reading_scope_service

            async def _times_out(**_kwargs):
                raise TimeoutError

            service.search_media = _times_out

            painted: list[str] = []
            for _ in range(2):
                screen.query_one("#library-media-retry", Button).press()
                await _wait_for_condition(
                    pilot,
                    lambda: not controller.loading,
                    message="The failed retry never settled.",
                )
                await pilot.pause()
                painted.append(
                    str(screen.query_one("#library-media-status", Static).renderable)
                )

            assert painted == ["Couldn't retry · timed out"] * 2
            # Recovery is never gated by what it recovers from.
            rows = screen.query(".library-media-row")
            assert len(rows) == 3
            assert all(row.disabled is False for row in rows)
            assert screen.query_one("#library-media-retry", Button).disabled is False
    finally:
        db.close_connection()
