"""Production-shaped closeout walkthrough for local Media Trash paging."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import pwd
import types
from pathlib import Path
from typing import Any

import pytest
from loguru import logger
from textual.widgets import Button, Input, OptionList, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_media_state import (
    MediaBrowseScope,
    MediaTrashScope,
    build_media_trash_result,
)
from tldw_chatbook.Media import LocalMediaReadingService, MediaReadingScopeService
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.app import TldwCli


SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))
QUERY_SENTINEL = "quartzneedle18918"
TITLE_SENTINEL = "PRIVATE TITLE 18918 SHOULD NEVER REACH LOGS"
ID_SENTINEL = "local:media:45"
PATH_SENTINEL = "private-path-18918://never-log-this"
CREDENTIAL_SENTINEL = "credential-18918-never-log-this"
DELETE_TARGET_TITLE = "DELETE TARGET 18918 exact permanent record title"
RESTORE_TARGET_TITLE = "RESTORE TARGET 18918 retained-page proof"
PRIVACY_SENTINELS = (
    QUERY_SENTINEL,
    TITLE_SENTINEL,
    ID_SENTINEL,
    PATH_SENTINEL,
    CREDENTIAL_SENTINEL,
    DELETE_TARGET_TITLE,
)


def _real_profile_snapshot() -> dict[Path, tuple[bool, bytes]]:
    """Capture real config/Media bytes without trusting the test HOME."""
    home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    config = home / ".config" / "tldw_cli" / "config.toml"
    media = (
        home
        / ".local"
        / "share"
        / "tldw_cli"
        / "default_user"
        / "tldw_chatbook_media_v2.db"
    )
    paths = (config, media, Path(f"{media}-wal"), Path(f"{media}-shm"))
    return {
        path: (path.exists(), path.read_bytes() if path.exists() else b"")
        for path in paths
    }


def _trash_timestamp(media_id: int) -> str | None:
    if media_id == 1:
        return None
    minute = 10 if media_id in {10, 11} else media_id
    return f"2026-08-30T12:{minute:02d}:00+00:00"


def _trash_type(media_id: int) -> str:
    return {0: "pdf", 1: " pdf ", 2: "PDF", 3: "audio", 4: "video"}[
        media_id % 5
    ]


def _trash_title(media_id: int) -> str:
    if media_id == 45:
        return DELETE_TARGET_TITLE
    if media_id == 44:
        return RESTORE_TARGET_TITLE
    if media_id == 43:
        return TITLE_SENTINEL
    if media_id in {41, 42}:
        return "Duplicate visible title " + ("with bounded detail " * 4).rstrip()
    if media_id in {6, 16, 26, 36, 46}:
        return f"Trash {media_id:02d} {QUERY_SENTINEL}"
    return f"Trash {media_id:02d} " + ("long recovery detail " * 3).rstrip()


def _seed_real_media_database(path: Path, *, width: int) -> MediaDatabase:
    """Seed deterministic Trash pages plus active same-source decoys."""
    db = MediaDatabase(path, client_id=f"task-18918-live-{width}")
    for media_id in range(1, 48):
        url = PATH_SENTINEL if media_id == 43 else f"live://trash/{width}/{media_id}"
        inserted_id, _uuid, _message = db.add_media_with_keywords(
            url=url,
            title=_trash_title(media_id),
            media_type=_trash_type(media_id),
            content=f"unique trash content {width} {media_id}",
            keywords=["trash-live"],
        )
        assert inserted_id == media_id
        assert db.mark_as_trash(media_id)

    # Active decoys include the same query/type vocabulary and make normal
    # Media genuinely paged; neither their rows nor facets may leak into Trash.
    for active_index in range(1, 46):
        media_id = 47 + active_index
        title = (
            f"Active decoy {active_index:02d} {QUERY_SENTINEL}"
            if active_index <= 3
            else f"Active decoy {active_index:02d}"
        )
        inserted_id, _uuid, _message = db.add_media_with_keywords(
            url=f"live://active/{width}/{active_index}",
            title=title,
            media_type=" pdf " if active_index <= 3 else "video",
            content=f"unique active content {width} {active_index}",
            keywords=["active-live"],
        )
        assert inserted_id == media_id

    with db.transaction() as conn:
        for media_id in range(1, 48):
            timestamp = _trash_timestamp(media_id)
            last_modified = timestamp or "2026-08-01T00:00:00+00:00"
            conn.execute(
                "UPDATE Media SET trash_date = ?, last_modified = ?, "
                "version = version + 1 WHERE id = ?",
                (timestamp, last_modified, media_id),
            )
        for media_id in range(48, 93):
            conn.execute(
                "UPDATE Media SET last_modified = ?, version = version + 1 "
                "WHERE id = ?",
                (f"2026-08-31T12:{media_id - 47:02d}:00+00:00", media_id),
            )
    return db


def _restore_shrunk_rows(db: MediaDatabase) -> None:
    """Return the one-shot concurrent-shrink fixture to its original 47 rows."""
    for media_id in range(2, 17):
        assert db.mark_as_trash(media_id)
    with db.transaction() as conn:
        for media_id in range(2, 17):
            timestamp = _trash_timestamp(media_id)
            last_modified = timestamp or "2026-08-01T00:00:00+00:00"
            conn.execute(
                "UPDATE Media SET trash_date = ?, last_modified = ?, "
                "version = version + 1 WHERE id = ?",
                (timestamp, last_modified, media_id),
            )


class _LiveTrashProbe:
    """One-shot failure/shrink controls around the real scope service."""

    def __init__(
        self,
        scope_service: MediaReadingScopeService,
        local_service: LocalMediaReadingService,
    ) -> None:
        self.scope_service = scope_service
        self.local_service = local_service
        self.original = local_service.list_library_media_trash
        self.calls: list[dict[str, Any]] = []
        self.fail_scope: MediaTrashScope | None = None
        self.shrink_on_page_three = False

    def install(self) -> None:
        def list_with_controls(_service: object, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(dict(kwargs))
            scope = MediaTrashScope(
                query=str(kwargs.get("query") or ""),
                media_type=kwargs.get("media_type"),
                page=(int(kwargs["offset"]) // 20) + 1,
            )
            if self.fail_scope == scope:
                self.fail_scope = None
                raise RuntimeError(" | ".join(PRIVACY_SENTINELS))
            if self.shrink_on_page_three and scope == MediaTrashScope(page=3):
                self.shrink_on_page_three = False
                for media_id in range(2, 17):
                    assert self.local_service.restore_media_item(media_id)
            return self.original(**kwargs)

        self.local_service.list_library_media_trash = types.MethodType(
            list_with_controls, self.local_service
        )


def _stable_ids(result: Any) -> tuple[str, ...]:
    return tuple(str(item["id"]) for item in result.items)


def _painted_text(screen: LibraryScreen) -> str:
    """Read Textual's current compositor strips, never detached renderables."""
    return "\n".join(strip.text for strip in screen._compositor.render_strips())


async def _current_widget(screen, pilot, selector: str, widget_type=None):
    """Wait for and prove current mounted, displayed, laid-out DOM identity."""
    widget = await _wait_for_selector(screen, pilot, selector)
    await pilot.pause()
    current = (
        screen.query_one(selector, widget_type)
        if widget_type
        else screen.query_one(selector)
    )
    assert widget is current
    assert current.is_mounted
    assert current.screen is screen
    assert current.display is True
    assert current.region.area > 0
    return current


async def _wait_for_trash_page(
    screen: LibraryScreen,
    pilot,
    *,
    scope: MediaTrashScope,
    total: int,
    expected_ids: tuple[str, ...],
) -> Any:
    controller = screen._library_media_trash_browse_controller
    await _wait_for_condition(
        pilot,
        lambda: (
            controller.state.applied_result is not None
            and controller.state.applied_result.scope == scope
            and controller.state.applied_result.total == total
            and not controller.state.loading
        ),
        message=f"Trash scope {scope!r} never became authoritative.",
    )
    result = controller.state.applied_result
    assert result is not None
    assert _stable_ids(result) == expected_ids
    await _wait_for_condition(
        pilot,
        lambda: (
            len(screen.query(".library-media-trash-row")) == len(expected_ids)
            and (
                not expected_ids
                or getattr(
                    screen.query_one("#library-media-trash-row-0"), "media_id", None
                )
                == expected_ids[0]
            )
        ),
        message="Authoritative Trash rows never reached the current canvas.",
    )
    await pilot.pause()
    await pilot.pause()
    if expected_ids:
        await _current_widget(screen, pilot, "#library-media-trash-row-0", Button)
    return result


async def _ensure_items_open(screen: LibraryScreen, pilot) -> None:
    if screen._library_media_reader_layout.items_open:
        return
    grip = screen.query_one("#library-media-items-grip", Button)
    grip.focus()
    await pilot.press("enter")
    await _wait_for_condition(
        pilot,
        lambda: (
            screen._library_media_reader_layout.items_open
            and screen.query_one("#library-canvas").region.area > 0
        ),
        message="Items pane never opened.",
    )


async def _toggle_pane(
    screen: LibraryScreen, pilot, *, pane: str, expected_open: bool
) -> None:
    grip = screen.query_one(f"#library-media-{pane}-grip", Button)
    grip.focus()
    await pilot.press("enter")
    await _wait_for_condition(
        pilot,
        lambda: getattr(screen._library_media_reader_layout, f"{pane}_open")
        is expected_open,
        message=f"{pane} pane never reached open={expected_open}.",
    )


async def _submit_search(screen: LibraryScreen, pilot, value: str) -> None:
    search = await _current_widget(
        screen, pilot, "#library-media-trash-search", Input
    )
    search.value = value
    # A completed page request may still own one after-paint focus callback.
    # Yield it, then reacquire the current Input before the keyboard submit.
    await pilot.pause()
    search = await _current_widget(
        screen, pilot, "#library-media-trash-search", Input
    )
    search.focus()
    await pilot.pause()
    search = screen.query_one("#library-media-trash-search", Input)
    search.focus()
    assert screen.focused is search
    await pilot.press("enter")


async def _choose_type(screen: LibraryScreen, pilot, value: str | None) -> None:
    chooser_button = await _current_widget(
        screen, pilot, "#library-media-trash-type-filter", Button
    )
    chooser_button.focus()
    await pilot.press("enter")
    chooser = await _current_widget(
        screen, pilot, "#library-media-trash-type-choices", OptionList
    )
    target_prompt = "All types" if value is None else value
    target_index = next(
        index
        for index in range(chooser.option_count)
        if target_prompt in str(chooser.get_option_at_index(index).prompt)
    )
    chooser.highlighted = target_index
    chooser.focus()
    assert screen.focused is chooser
    await pilot.press("enter")


async def _assert_fixed_controls_painted(
    screen: LibraryScreen, pilot, *, width: int
) -> str:
    selectors = (
        "#library-media-trash-back",
        "#library-media-trash-search",
        "#library-media-trash-type-filter",
        "#library-media-trash-previous",
        "#library-media-trash-next",
        "#library-media-trash-restore",
        "#library-media-trash-delete",
    )
    items = screen.query_one("#library-canvas")
    for selector in selectors:
        control = await _current_widget(screen, pilot, selector)
        assert items.region.contains_region(control.region), (
            width,
            selector,
            control.region,
        )
    painted = _painted_text(screen)
    for copy in ("Local Trash", "Type: All", "Previous", "Next", "Restore"):
        assert copy in painted, (width, copy, painted)
    assert "Delete permanently" in painted
    return painted


def _row_for_media(screen: LibraryScreen, stable_id: str) -> Button:
    return next(
        row
        for row in screen.query(".library-media-trash-row")
        if getattr(row, "media_id", None) == stable_id
    )


def _normal_media_row(screen: LibraryScreen, stable_id: str) -> Button:
    return next(
        row
        for row in screen.query(".library-media-row")
        if getattr(row, "media_id", None) == stable_id
    )


async def _walk_size(
    *,
    db: MediaDatabase,
    size: tuple[int, int],
) -> dict[str, Any]:
    width, _height = size
    local_service = LocalMediaReadingService(db)
    scope_service = MediaReadingScopeService(local_service, None)

    # Real DB + both real service layers independently prove the exact
    # snapshot, stable tie-break, complete facets, padding, and local-only gate.
    first = await scope_service.list_library_media_trash(
        mode="local", query="", media_type=None, limit=20, offset=0
    )
    middle = await scope_service.list_library_media_trash(
        mode="local", query="", media_type=None, limit=20, offset=20
    )
    final = await scope_service.list_library_media_trash(
        mode="local", query="", media_type=None, limit=20, offset=40
    )
    assert tuple(item["id"] for item in first["items"]) == tuple(
        f"local:media:{media_id}" for media_id in range(47, 27, -1)
    )
    assert tuple(item["id"] for item in middle["items"]) == tuple(
        f"local:media:{media_id}" for media_id in range(27, 7, -1)
    )
    assert tuple(item["id"] for item in final["items"]) == tuple(
        f"local:media:{media_id}" for media_id in range(7, 0, -1)
    )
    assert {first["total"], middle["total"], final["total"]} == {47}
    assert build_media_trash_result(MediaTrashScope(), first).total == 47
    assert build_media_trash_result(MediaTrashScope(page=2), middle).total == 47
    assert build_media_trash_result(MediaTrashScope(page=3), final).total == 47
    assert first["types"] == ["PDF", "audio", "pdf", "video"]
    assert [
        item["id"]
        for item in middle["items"]
        if item["backing_media_id"] in {10, 11}
    ] == ["local:media:11", "local:media:10"]
    with pytest.raises(ValueError, match="requires local mode"):
        await scope_service.list_library_media_trash(
            mode="server", query="", media_type=None, limit=20, offset=0
        )

    app = _build_test_app(configured_default="library")
    assert type(app) is TldwCli
    app.library_new_profile_admission = False
    _seed_conversations(app, [], media=[])
    app.media_db = db
    app.media_reading_scope_service = scope_service
    probe = _LiveTrashProbe(scope_service, local_service)
    probe.install()

    try:
        async with app.run_test(size=size) as pilot:
            await _wait_for_condition(
                pilot,
                lambda: isinstance(app.screen, LibraryScreen),
                message="Production TldwCli did not mount LibraryScreen.",
            )
            screen = app.screen
            assert isinstance(screen, LibraryScreen)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-media", Button).focus()
            await pilot.press("enter")
            normal = screen._library_media_browse_controller
            await _wait_for_condition(
                pilot,
                lambda: normal.applied_scope == MediaBrowseScope(),
                message="Initial real normal-Media page never settled.",
            )
            await _ensure_items_open(screen, pilot)
            screen._request_library_media_page(2, focus_identity=None)
            await _wait_for_condition(
                pilot,
                lambda: normal.applied_scope == MediaBrowseScope(page=2),
                message="Normal Media page 2 never settled.",
            )
            normal_retained = normal.retained_items
            assert len(normal_retained) == 20
            selected_normal_id = str(normal_retained[7]["id"])
            await _wait_for_condition(
                pilot,
                lambda: (
                    len(screen.query(".library-media-row")) == 20
                    and int(
                        screen.query_one("#library-media-row-scroll").max_scroll_y
                    )
                    >= 5
                ),
                message="Normal Media page 2 never reached scrollable mounted rows.",
            )
            normal_scroll = screen.query_one("#library-media-row-scroll")
            normal_scroll.scroll_to(y=5, animate=False, force=True, immediate=True)
            await _wait_for_condition(
                pilot,
                lambda: int(normal_scroll.scroll_y) == 5,
                message="Normal Media scroll did not settle at row offset 5.",
            )
            saved_scroll = (int(normal_scroll.scroll_x), int(normal_scroll.scroll_y))
            assert saved_scroll[1] > 0
            screen._selected_media_id = selected_normal_id
            assert screen._selected_media_id == selected_normal_id

            # Prove the normal viewer's retained-return policy while the
            # page is fresh. Restore later marks this retained page stale,
            # and stale rows intentionally reject viewer actions.
            viewer_row = _normal_media_row(screen, selected_normal_id)
            viewer_row.press()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_view == "viewer",
                message="Normal Media row never opened the viewer.",
            )
            viewer_back = await _current_widget(
                screen, pilot, "#library-media-back", Button
            )
            viewer_back.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen._library_media_view == "list"
                    and getattr(screen.focused, "media_id", None)
                    == selected_normal_id
                ),
                message="Viewer Back did not finish on its semantic Media row.",
            )
            viewer_return_row = _normal_media_row(screen, selected_normal_id)
            assert screen.focused is viewer_return_row
            viewer_return_scroll = screen.query_one("#library-media-row-scroll")
            assert (
                int(viewer_return_scroll.scroll_x),
                int(viewer_return_scroll.scroll_y),
            ) == saved_scroll

            opener = await _current_widget(
                screen, pilot, "#library-media-trash-open", Button
            )
            opener.focus()
            await pilot.press("enter")
            controller = screen._library_media_trash_browse_controller
            first_ids = tuple(
                f"local:media:{media_id}" for media_id in range(47, 27, -1)
            )
            middle_ids = tuple(
                f"local:media:{media_id}" for media_id in range(27, 7, -1)
            )
            final_ids = tuple(
                f"local:media:{media_id}" for media_id in range(7, 0, -1)
            )
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(),
                total=47,
                expected_ids=first_ids,
            )
            await _assert_fixed_controls_painted(screen, pilot, width=width)

            # Every ordinary action is a real current focus target. Pager keys
            # are exercised on page 2 where both directions are enabled.
            for selector in (
                "#library-media-trash-search",
                "#library-media-trash-type-filter",
                "#library-media-trash-row-0",
                "#library-media-trash-restore",
                "#library-media-trash-delete",
                "#library-media-trash-back",
            ):
                control = await _current_widget(screen, pilot, selector)
                control.focus()
                await pilot.pause()
                assert screen.focused is control

            next_button = screen.query_one("#library-media-trash-next", Button)
            next_button.focus()
            await pilot.press("enter")
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(page=2),
                total=47,
                expected_ids=middle_ids,
            )
            for selector in (
                "#library-media-trash-previous",
                "#library-media-trash-next",
            ):
                control = await _current_widget(screen, pilot, selector, Button)
                assert control.disabled is False
                control.focus()
                await pilot.pause()
                assert screen.focused is control
            screen.query_one("#library-media-trash-next", Button).focus()
            await pilot.press("enter")
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(page=3),
                total=47,
                expected_ids=final_ids,
            )
            assert (
                screen.query_one("#library-media-trash-range", Static).renderable
                == "41-47 of 47"
            )
            assert (
                screen.query_one("#library-media-trash-page", Static).renderable
                == "Page 3 of 3"
            )
            screen.query_one("#library-media-trash-previous", Button).focus()
            await pilot.press("enter")
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(page=2),
                total=47,
                expected_ids=middle_ids,
            )
            screen.query_one("#library-media-trash-previous", Button).focus()
            await pilot.press("enter")
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(),
                total=47,
                expected_ids=first_ids,
            )

            # Failed query retains honest applied copy; current keyboard Retry
            # repeats exactly that failed target and applies five real DB rows.
            probe.fail_scope = MediaTrashScope(query=QUERY_SENTINEL)
            await _submit_search(screen, pilot, QUERY_SENTINEL)
            await _wait_for_condition(
                pilot,
                lambda: controller.state.failed_scope
                == MediaTrashScope(query=QUERY_SENTINEL),
                message="Failed real query never exposed its Retry target.",
            )
            assert controller.state.applied_result is not None
            assert controller.state.applied_result.scope == MediaTrashScope()
            assert screen.query_one(
                "#library-media-trash-status", Static
            ).renderable == "Filter not applied — showing All Trash · Retry"
            retry = await _current_widget(
                screen, pilot, "#library-media-trash-retry", Button
            )
            retry.focus()
            assert screen.focused is retry
            assert "Retry" in _painted_text(screen)
            await pilot.press("enter")
            query_ids = tuple(
                f"local:media:{media_id}" for media_id in (46, 36, 26, 16, 6)
            )
            query_result = await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(query=QUERY_SENTINEL),
                total=5,
                expected_ids=query_ids,
            )
            assert query_result.types == ("PDF", "audio", "pdf", "video")

            await _choose_type(screen, pilot, "pdf")
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(query=QUERY_SENTINEL, media_type="pdf"),
                total=5,
                expected_ids=query_ids,
            )
            await _submit_search(screen, pilot, "")
            pdf_ids = tuple(
                f"local:media:{media_id}"
                for media_id in range(46, 0, -1)
                if media_id % 5 in {0, 1}
            )
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(media_type="pdf"),
                total=19,
                expected_ids=pdf_ids,
            )
            await _choose_type(screen, pilot, "PDF")
            upper_ids = tuple(
                f"local:media:{media_id}"
                for media_id in range(47, 1, -1)
                if media_id % 5 == 2
            )
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(media_type="PDF"),
                total=10,
                expected_ids=upper_ids,
            )
            await _choose_type(screen, pilot, None)
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(),
                total=47,
                expected_ids=first_ids,
            )

            # A failed page request names both pages, retains page 1, then
            # keyboard Retry installs the exact real middle page.
            probe.fail_scope = MediaTrashScope(page=2)
            screen.query_one("#library-media-trash-next", Button).focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: controller.state.failed_scope == MediaTrashScope(page=2),
                message="Failed real page never exposed its Retry target.",
            )
            assert screen.query_one(
                "#library-media-trash-status", Static
                ).renderable == "Page 2 not loaded — showing page 1 · Retry"
            retry = await _current_widget(
                screen, pilot, "#library-media-trash-retry", Button
            )
            retry.focus()
            await pilot.press("enter")
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(page=2),
                total=47,
                expected_ids=middle_ids,
            )

            # The service shrinks underneath an attempted page 3. Production
            # may clamp exactly once, to the now-valid page 2 with 32 rows.
            calls_before_shrink = len(probe.calls)
            probe.shrink_on_page_three = True
            screen.query_one("#library-media-trash-next", Button).focus()
            await pilot.press("enter")
            shrunk_ids = tuple(
                f"local:media:{media_id}" for media_id in range(27, 16, -1)
            ) + ("local:media:1",)
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(page=2),
                total=32,
                expected_ids=shrunk_ids,
            )
            assert [
                call["offset"] for call in probe.calls[calls_before_shrink:]
            ] == [40, 20]
            await asyncio.to_thread(_restore_shrunk_rows, db)
            screen._request_library_media_trash_page(
                1, focus_identity="#library-media-trash-previous"
            )
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(),
                total=47,
                expected_ids=first_ids,
            )

            # Pane choices survive real refreshes. Compact posture makes the
            # two panes mutually exclusive; wide posture proves independent
            # collapse and re-expansion of both.
            initial_layout = screen._library_media_reader_layout
            wide_posture = bool(
                initial_layout.library_open and initial_layout.items_open
            )
            items_width_contract = "exclusive-full-width"
            if not wide_posture:
                assert screen._library_media_reader_layout.items_open is True
                await _toggle_pane(
                    screen, pilot, pane="library", expected_open=True
                )
                assert screen._library_media_reader_layout.items_open is False
                screen._request_library_media_trash_page(
                    2, focus_identity="#library-media-trash-next"
                )
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        controller.state.applied_result is not None
                        and controller.state.applied_result.scope
                        == MediaTrashScope(page=2)
                    ),
                    message="Hidden compact Items refresh never settled.",
                )
                assert screen._library_media_reader_layout.library_open is True
                assert screen._library_media_reader_layout.items_open is False
                await _toggle_pane(screen, pilot, pane="items", expected_open=True)
                assert screen._library_media_reader_layout.library_open is False
                await _submit_search(screen, pilot, QUERY_SENTINEL)
                await _wait_for_trash_page(
                    screen,
                    pilot,
                    scope=MediaTrashScope(query=QUERY_SENTINEL),
                    total=5,
                    expected_ids=query_ids,
                )
                assert screen._library_media_reader_layout.items_open is True
                assert screen._library_media_reader_layout.library_open is False
                compact_items = screen.query_one("#library-canvas")
                compact_row = screen.query_one("#library-media-trash-row-0", Button)
                assert compact_items.region.width > 0
                assert compact_row.region.width > 0
                await _submit_search(screen, pilot, "")
            else:
                assert screen._library_media_reader_layout.library_open is True
                assert screen._library_media_reader_layout.items_open is True
                split_items_width = screen.query_one("#library-canvas").region.width
                split_row_width = screen.query_one(
                    "#library-media-trash-row-0", Button
                ).region.width
                await _toggle_pane(
                    screen, pilot, pane="library", expected_open=False
                )
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        screen.query_one("#library-canvas").region.width
                        > split_items_width
                        and screen.query_one(
                            "#library-media-trash-row-0", Button
                        ).region.width
                        > split_row_width
                    ),
                    message="Collapsing Library did not widen Trash Items rows.",
                )
                items_width_contract = "expanded-title-and-detail"
                screen._request_library_media_trash_page(
                    2, focus_identity="#library-media-trash-next"
                )
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        controller.state.applied_result is not None
                        and controller.state.applied_result.scope
                        == MediaTrashScope(page=2)
                    ),
                    message="Wide Library-collapsed refresh never settled.",
                )
                assert screen._library_media_reader_layout.library_open is False
                await _toggle_pane(screen, pilot, pane="library", expected_open=True)
                await _toggle_pane(screen, pilot, pane="items", expected_open=False)
                screen._request_library_media_trash_page(
                    1, focus_identity="#library-media-trash-previous"
                )
                await _wait_for_condition(
                    pilot,
                    lambda: (
                        controller.state.applied_result is not None
                        and controller.state.applied_result.scope == MediaTrashScope()
                    ),
                    message="Wide Items-collapsed refresh never settled.",
                )
                assert screen._library_media_reader_layout.items_open is False
                await _toggle_pane(screen, pilot, pane="items", expected_open=True)

            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(),
                total=47,
                expected_ids=first_ids,
            )
            await _assert_fixed_controls_painted(screen, pilot, width=width)

            # Permanent deletion: the full captured title is visible, Cancel
            # owns initial focus, Enter is safe, and only a later explicit
            # Confirm commits. Its follow-up read fails truthfully as stale.
            delete_row = _row_for_media(screen, ID_SENTINEL)
            delete_row.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: controller.state.selected_id == ID_SENTINEL,
                message="Delete target selection never settled.",
            )
            delete = await _current_widget(
                screen, pilot, "#library-media-trash-delete", Button
            )
            delete.focus()
            await pilot.press("enter")
            await _current_widget(
                screen, pilot, "#library-media-trash-delete-cancel", Button
            )
            await _wait_for_condition(
                pilot,
                lambda: screen.focused
                is screen.query_one("#library-media-trash-delete-cancel", Button),
                message="Cancel did not own initial destructive focus.",
            )
            title = screen.query_one(
                "#library-media-trash-delete-confirm-title", Static
            )
            assert title.renderable == DELETE_TARGET_TITLE
            assert "permanent" in _painted_text(screen).lower()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: not screen.query(
                    "#library-media-trash-delete-confirmation"
                ),
                message="Keyboard Cancel did not close confirmation.",
            )
            assert db.get_media_by_id(45, include_trash=True) is not None

            delete = await _current_widget(
                screen, pilot, "#library-media-trash-delete", Button
            )
            delete.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: getattr(screen.focused, "id", None)
                == "library-media-trash-delete-cancel",
                message="Reopened confirmation did not focus Cancel.",
            )
            probe.fail_scope = MediaTrashScope()
            await pilot.press("tab")
            assert screen.focused is screen.query_one(
                "#library-media-trash-delete-confirm", Button
            )
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: (
                    controller.state.freshness == "stale"
                    and controller.state.failed_scope == MediaTrashScope()
                    and controller.state.committed_notice
                    == f"Deleted '{DELETE_TARGET_TITLE}' permanently."
                ),
                message="Committed delete refresh failure never became stale Retry.",
            )
            assert db.get_media_by_id(45, include_trash=True) is None
            stale_status = screen.query_one(
                "#library-media-trash-status", Static
            ).renderable
            assert stale_status == (
                f"Deleted '{DELETE_TARGET_TITLE}' permanently. "
                "List may be out of date · Retry"
            )
            retry = await _current_widget(
                screen, pilot, "#library-media-trash-retry", Button
            )
            retry.focus()
            await pilot.press("enter")
            after_delete_ids = tuple(
                f"local:media:{media_id}"
                for media_id in range(47, 26, -1)
                if media_id != 45
            )
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(),
                total=46,
                expected_ids=after_delete_ids,
            )

            # Restore mutates the real DB, stales but does not splice the
            # retained normal-Media page, and keeps Back's captured context.
            restore_id = "local:media:44"
            restore_row = _row_for_media(screen, restore_id)
            restore_row.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: controller.state.selected_id == restore_id,
                message="Restore target selection never settled.",
            )
            restore = await _current_widget(
                screen, pilot, "#library-media-trash-restore", Button
            )
            restore.focus()
            await pilot.press("enter")
            after_restore_ids = tuple(
                f"local:media:{media_id}"
                for media_id in range(47, 25, -1)
                if media_id not in {44, 45}
            )
            await _wait_for_trash_page(
                screen,
                pilot,
                scope=MediaTrashScope(),
                total=45,
                expected_ids=after_restore_ids,
            )
            assert db.get_media_by_id(44, include_trash=False) is not None
            assert normal.freshness == "stale"
            assert normal.retained_items is normal_retained
            assert all(
                str(item["id"]) != restore_id for item in normal.retained_items
            )
            trash_return = screen._library_media_trash_return
            assert trash_return is not None
            assert trash_return.stable_id == selected_normal_id
            assert trash_return.scroll_offset == saved_scroll
            assert trash_return.final_focus_identity == "library-media-trash-open"

            back = await _current_widget(
                screen, pilot, "#library-media-trash-back", Button
            )
            back.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: screen._library_media_view == "list",
                message="Back never restored normal Media.",
            )
            await _wait_for_selector(screen, pilot, "#library-media-canvas")
            await _wait_for_condition(
                pilot,
                lambda: getattr(screen.focused, "id", None)
                == "library-media-trash-open",
                message="Back never restored opener focus.",
            )
            assert normal.applied_scope == MediaBrowseScope(page=2)
            assert screen._selected_media_id == selected_normal_id
            await _wait_for_condition(
                pilot,
                lambda: (
                    int(
                        screen.query_one("#library-media-row-scroll").scroll_x
                    ),
                    int(
                        screen.query_one("#library-media-row-scroll").scroll_y
                    ),
                )
                == saved_scroll,
                message=lambda: (
                    "Back did not restore normal Media scroll: "
                    f"expected={saved_scroll!r}, "
                    "actual="
                    f"{(int(screen.query_one('#library-media-row-scroll').scroll_x), int(screen.query_one('#library-media-row-scroll').scroll_y))!r}, "
                    "max="
                    f"{(int(screen.query_one('#library-media-row-scroll').max_scroll_x), int(screen.query_one('#library-media-row-scroll').max_scroll_y))!r}"
                ),
            )
            restored_scroll = screen.query_one("#library-media-row-scroll")
            assert (
                int(restored_scroll.scroll_x),
                int(restored_scroll.scroll_y),
            ) == saved_scroll
            opener = screen.query_one("#library-media-trash-open", Button)
            assert screen.focused is opener
            assert opener.is_mounted and opener.region.area > 0

            unfinished_trash_workers = tuple(
                worker
                for worker in screen.workers
                if worker.group
                in {"library-media-trash-browse", "library_media_bulk_delete"}
            )
            if unfinished_trash_workers:
                await asyncio.wait_for(
                    screen.workers.wait_for_complete(unfinished_trash_workers),
                    timeout=10.0,
                )
            assert not tuple(
                worker
                for worker in screen.workers
                if worker.group
                in {"library-media-trash-browse", "library_media_bulk_delete"}
            )
        await asyncio.sleep(0)
        assert all(worker.is_finished for worker in app.workers)
    finally:
        # Close the main connection and any sequential default-executor
        # connection used by the real service calls before byte/isolation checks.
        await asyncio.gather(
            *(asyncio.to_thread(db.close_connection) for _ in range(8))
        )
        db.close_connection()

    return {
        "size": size,
        "initial_total": 47,
        "query_total": 5,
        "clamped_total": 32,
        "post_delete_total": 46,
        "post_restore_total": 45,
        "painted": True,
        "keyboard": True,
        "back": True,
        "viewer_row_return": True,
        "items_width_contract": items_width_contract,
    }


@pytest.mark.asyncio
async def test_live_real_database_media_trash_walkthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Walk real local Trash through every supported terminal posture."""
    profile_before = _real_profile_snapshot()
    child_pids_before = {child.pid for child in multiprocessing.active_children()}
    scratch_config = tmp_path / "profile" / "config.toml"
    scratch_config.parent.mkdir(parents=True)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(scratch_config))
    monkeypatch.setenv("TASK_18918_LIVE_CREDENTIAL", CREDENTIAL_SENTINEL)
    assert Path(os.environ["TLDW_CONFIG_PATH"]).resolve().is_relative_to(
        tmp_path.resolve()
    )
    package_path = Path(__import__("tldw_chatbook").__file__).resolve()
    assert Path(__file__).resolve().parents[2] in package_path.parents

    observations: list[dict[str, Any]] = []
    for size in SIZES:
        width, _height = size
        db = _seed_real_media_database(
            tmp_path / f"media-trash-live-{width}.db", width=width
        )
        assert Path(db.db_path).resolve().is_relative_to(tmp_path.resolve())
        log_path = tmp_path / "logs" / f"media-trash-live-{width}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        loguru_sink = logger.add(log_path, format="{message}", level="DEBUG")
        root_handler = logging.FileHandler(log_path, encoding="utf-8")
        root_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(root_handler)
        try:
            observation = await _walk_size(db=db, size=size)
        finally:
            logging.getLogger().removeHandler(root_handler)
            root_handler.flush()
            root_handler.close()
            logger.remove(loguru_sink)
        size_log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for sentinel in PRIVACY_SENTINELS:
            assert sentinel not in size_log_text
        assert _real_profile_snapshot() == profile_before
        observation["privacy"] = True
        observation["real_profile_unchanged"] = True
        observations.append(observation)
        print(
            "TASK-18918 LIVE "
            f"{width}x{size[1]}: pages=47/3 query=5 clamp=32 "
            "delete=46 restore=45 exact-trash-back=true "
            "viewer-row-return=true "
            f"items-width={observation['items_width_contract']} "
            "privacy=true real-profile-unchanged=true"
        )

    assert tuple(observation["size"] for observation in observations) == SIZES
    assert all(
        observation
        == {
            "size": observation["size"],
            "initial_total": 47,
            "query_total": 5,
            "clamped_total": 32,
            "post_delete_total": 46,
            "post_restore_total": 45,
            "painted": True,
            "keyboard": True,
            "back": True,
            "viewer_row_return": True,
            "items_width_contract": observation["items_width_contract"],
            "privacy": True,
            "real_profile_unchanged": True,
        }
        for observation in observations
    )

    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in tmp_path.rglob("*.log*")
        if path.is_file()
    )
    for sentinel in PRIVACY_SENTINELS:
        assert sentinel not in log_text

    await asyncio.sleep(0)
    assert {child.pid for child in multiprocessing.active_children()} == child_pids_before
    assert _real_profile_snapshot() == profile_before
