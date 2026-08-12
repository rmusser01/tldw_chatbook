"""TASK-15457 review round 1: defects a canvas-scoped sync must not introduce.

Companion to ``test_library_canvas_scoped_sync.py`` (which pins that the
conversions happen at all). This file pins the three things a canvas-scoped
sync silently STOPS doing, because they live in ``LibraryScreen.refresh`` --
the override a targeted sync deliberately bypasses:

* the resolved media selection is mirrored back into ``_selected_media_id``
  (otherwise the chooser highlights one row and "Open in viewer" opens
  another);
* portable Notes focus is restored, so DOM focus never escapes the canvas;
* the Notes list's scroll offset survives -- which is only observable BELOW
  ``LIBRARY_NOTES_COMPACT_BREAKPOINT``, where the list can actually scroll.

All six assertions were verified RED against dev's own implementation of this
task (``976dbafcb``) before the fixes were ported onto it, so this file is
evidence for dev's converted sites, not only for the ones this branch added.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_selection_updates import _spy_screen_recomposes
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _notes(count: int = 6):
    return [
        {
            "id": f"note-{i}",
            "title": f"Note number {i}",
            "content": f"Body of note {i}.",
            "last_modified": f"2026-06-{i + 1:02d}T10:00:00Z",
        }
        for i in range(count)
    ]


def _media(count: int = 4):
    return [
        {
            "id": i + 1,
            "media_id": str(i + 1),
            "title": f"Media item {i}",
            "type": "video" if i % 2 else "document",
            "content": f"Transcript {i}",
            "ingestion_date": f"2026-06-{i + 1:02d}T10:00:00Z",
        }
        for i in range(count)
    ]


async def _open_notes_canvas(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes").press()
    await _wait_for_selector(screen, pilot, "#library-notes-select-toggle")
    await pilot.pause()
    return screen


async def _open_media_canvas(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-type-filter")
    await pilot.pause()
    return screen


def _static_text(screen, selector: str) -> str:
    renderable = screen.query_one(selector, Static).renderable
    return getattr(renderable, "plain", str(renderable))


def _assert_notes_footer_is_current(screen) -> None:
    """The registered footer set must match the live Notes state.

    A whole-screen recompose re-derived this for free -- ``LibraryScreen.
    refresh`` calls ``_apply_library_notes_footer_context`` on every call.
    A canvas-scoped sync bypasses that override entirely, and the Notes
    footer tier genuinely branches on select mode and on the sort strip's
    visibility (``_library_notes_footer_shortcuts``), so every converted
    notes site has to keep it honest. Caught live: the first cut of the
    select-strip conversion left the footer advertising the browse keys
    while select mode was on.
    """
    assert screen._footer_shortcut_registration == (
        "library",
        screen._library_notes_footer_shortcuts(),
    )


# --------------------------------------------------------------------------
# Review round 1 — regressions the first cut of this task introduced.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_type_filter_keeps_selected_id_in_step_with_the_canvas():
    """CRITICAL: the media sync branch must mirror the resolved selection.

    ``compose_content`` and ``_replace_library_browse_canvas`` both write
    ``self._selected_media_id = media_state.selected_id`` after building the
    state, because ``build_library_media_canvas_state`` RESOLVES the
    selection: a requested id that the active type filter no longer renders
    falls back to the first row. The targeted sync skipped that mirror, so
    filtering the selected item out left the canvas highlighting row 0 while
    the screen still pointed at the filtered-out id -- and "Open in viewer"
    opened the invisible one.
    """
    app = _build_test_app()
    _seed_conversations(app, [], media=_media())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_media_canvas(host, pilot)

        # ``_media()`` alternates document/video, so id "4" is a video and
        # ids "1"/"3" are documents. Select the video, then filter to
        # documents so the selected row stops being rendered at all.
        video_row = next(
            button
            for button in screen.query(".library-media-row").results(Button)
            if str(getattr(button, "media_id", "")) == "4"
        )
        video_id = "4"
        video_row.press()
        await pilot.pause()
        assert screen._selected_media_id == video_id
        # Back to the list -- the selection survives the round trip, which is
        # how the browse canvas ends up pointing at a non-first row.
        screen.action_library_media_viewer_back()
        await _wait_for_selector(screen, pilot, "#library-media-type-filter")
        await pilot.pause()
        assert screen._selected_media_id == video_id

        screen.query_one("#library-media-type-filter", Button).focus()
        await pilot.pause()
        screen.query_one("#library-media-type-filter", Button).press()
        await _wait_for_selector(screen, pilot, ".library-media-type-choice")
        document_choice = next(
            button
            for button in screen.query(".library-media-type-choice").results(Button)
            if str(getattr(button, "choice_value", "")) == "document"
        )
        document_choice.press()
        await pilot.pause()
        await pilot.pause()

        canvas_state = screen._build_library_media_state()
        assert canvas_state.selected_id != video_id  # the filter dropped it
        # The screen's own pointer must agree with what the canvas renders.
        assert screen._selected_media_id == canvas_state.selected_id
        # ...and the primary action must therefore open the visible item.
        screen._open_library_media_viewer(screen._selected_media_id)
        await pilot.pause()
        assert screen._selected_media_id == canvas_state.selected_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trigger",
    ["select_toggle", "select_all", "sort_open"],
)
async def test_converted_notes_sites_keep_focus_inside_the_canvas(trigger):
    """CRITICAL: a converted site with no explicit ``then=`` must still
    restore focus.

    The screen path restores it through
    ``_rehydrate_library_notes_after_recompose`` ->
    ``_restore_library_notes_focus_identity``; a canvas-scoped sync bypasses
    ``LibraryScreen.refresh`` and therefore that machinery entirely, so
    every converted site WITHOUT its own focus follow-up let DOM focus
    escape the canvas when its focused child was recomposed away.
    """
    app = _build_test_app()
    _seed_conversations(app, [], notes=_notes(4))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_notes_canvas(host, pilot)
        if trigger == "select_all":
            screen.query_one("#library-notes-select-toggle", Button).press()
            await _wait_for_selector(screen, pilot, "#library-notes-select-all")
            await pilot.pause()

        selector = {
            "select_toggle": "#library-notes-select-toggle",
            "select_all": "#library-notes-select-all",
            "sort_open": "#library-notes-sort",
        }[trigger]
        screen.query_one(selector, Button).focus()
        await pilot.pause()
        assert screen.focused is not None and screen.focused.id == selector.lstrip("#")

        screen.query_one(selector, Button).press()
        await pilot.pause()
        await pilot.pause()

        # Focus must still be somewhere inside the notes canvas -- never
        # stranded on the rail/nav chrome outside it.
        focused = screen.focused
        assert focused is not None
        canvas = screen.query_one("#library-notes-canvas")
        assert canvas in focused.ancestors_with_self, (
            f"focus escaped the notes canvas to {focused.id!r} after {trigger}"
        )


@pytest.mark.asyncio
async def test_converted_notes_site_keeps_focus_on_a_real_key_press():
    """The same guarantee via the real keyboard, not a programmatic press."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=_notes(4))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_notes_canvas(host, pilot)
        screen.query_one("#library-notes-select-toggle", Button).focus()
        await pilot.pause()

        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert screen._library_notes_select_mode is True
        focused = screen.focused
        assert focused is not None
        canvas = screen.query_one("#library-notes-canvas")
        assert canvas in focused.ancestors_with_self, (
            f"focus escaped the notes canvas to {focused.id!r}"
        )


LIBRARY_COMPACT_TEST_SIZE = (100, 24)


@pytest.mark.asyncio
async def test_compact_notes_list_keeps_its_scroll_offset_across_a_sync():
    """IMPORTANT: the notes list's scroll offset must survive a converted site.

    Only reachable below ``LIBRARY_NOTES_COMPACT_BREAKPOINT`` (120 cols) with
    enough rows to scroll -- every other test in this file runs at 170x48,
    where the list never scrolls at all. That is the July compact-resize
    lesson: geometry that is never exercised at a second width is not
    measured. The screen path restores this via
    ``_restore_library_notes_scroll_offset``; the canvas-scoped sync bypassed
    it and dropped the user back to the top of the list.
    """
    app = _build_test_app()
    _seed_conversations(app, [], notes=_notes(40))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_COMPACT_TEST_SIZE) as pilot:
        screen = await _open_notes_canvas(host, pilot)
        assert screen._library_notes_compact is True

        notes_list = screen.query_one("#library-notes-list")
        notes_list.scroll_to(y=12, animate=False, force=True, immediate=True)
        await pilot.pause()
        offset_before = int(notes_list.scroll_offset.y)
        assert offset_before > 0, "the list did not scroll; the fixture is too short"

        screen.query_one("#library-notes-select-toggle", Button).focus()
        await pilot.pause()
        screen.query_one("#library-notes-select-toggle", Button).press()
        await pilot.pause()
        await pilot.pause()

        after = screen.query_one("#library-notes-list")
        assert int(after.scroll_offset.y) == offset_before, (
            f"notes list scroll fell {offset_before} -> {int(after.scroll_offset.y)}"
        )


@pytest.mark.asyncio
async def test_notes_footer_tier_follows_a_canvas_scoped_sync():
    """The Notes footer tier must track select mode across a targeted sync.

    ``LibraryScreen.refresh`` re-derives the footer on every whole-screen
    recompose; a canvas-scoped sync bypasses it. The Notes tier genuinely
    branches on select mode (``("enter", "select note") / ("esc", "done")``),
    so without an explicit refresh at the sync choke point the footer keeps
    advertising the browse keys.

    Note this assertion is only meaningful once focus restoration works: with
    focus escaping the canvas the region resolves to "" and the Notes tier is
    never selected at all, so the invariant passes vacuously.
    """
    app = _build_test_app()
    _seed_conversations(app, [], notes=_notes(4))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_notes_canvas(host, pilot)
        screen.query_one("#library-notes-select-toggle", Button).focus()
        await pilot.pause()

        screen.query_one("#library-notes-select-toggle", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_notes_select_mode is True
        assert screen._library_notes_focus_region() == "navigator"
        assert screen._footer_shortcut_registration == (
            "library",
            screen._library_notes_footer_shortcuts(),
        )
        assert ("esc", "done") in screen._library_notes_footer_shortcuts()
