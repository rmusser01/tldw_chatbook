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

from functools import partial

import pytest
from textual.widgets import Button, OptionList, Static

from Tests.UI.app_factory import _build_test_app as _build_tldw_test_app
from Tests.UI.test_library_selection_updates import _spy_screen_recomposes
from Tests.UI.test_library_shell import (
    _FakePromptScopeService,
    _FakeSkillsScopeService,
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_BROWSE_SKILLS,
)
from tldw_chatbook.UI.Screens.library_screen import _sync_library_canvas


def _build_test_app():
    """Build the legacy/full Library surface these canvas tests exercise."""
    app = _build_tldw_test_app()
    app.library_new_profile_admission = False
    return app


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
        await _wait_for_condition(
            pilot,
            lambda: len(screen.query(".library-media-row")) == 4,
            message="Exact Media page never settled before selection.",
        )

        # ``_media()`` alternates document/video, so backing id 4 is a video
        # and backing ids 1/3 are documents. Select the video, then filter to
        # documents so the selected row stops being rendered at all.
        video_row = next(
            button
            for button in screen.query(".library-media-row").results(Button)
            if str(getattr(button, "media_id", "")) == "local:media:4"
        )
        video_id = "local:media:4"
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
        chooser = await _wait_for_selector(
            screen, pilot, "#library-media-type-choices"
        )
        assert isinstance(chooser, OptionList)
        chooser.highlighted = next(
            index
            for index, option in enumerate(chooser.options)
            if getattr(option, "choice_value", None) == "document"
        )
        chooser.action_select()
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
async def test_converted_notes_sites_keep_focus_inside_the_canvas(monkeypatch, trigger):
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

        recompose_calls = _spy_screen_recomposes(monkeypatch)

        screen.query_one(selector, Button).press()
        await pilot.pause()
        await pilot.pause()

        # Discriminating: a whole-screen fallback would restore focus through
        # dev's own rehydration, so without this the assertion below would
        # pass for the wrong reason.
        assert recompose_calls == []
        # Focus must still be somewhere inside the notes canvas -- never
        # stranded on the rail/nav chrome outside it.
        focused = screen.focused
        assert focused is not None
        canvas = screen.query_one("#library-notes-canvas")
        assert canvas in focused.ancestors_with_self, (
            f"focus escaped the notes canvas to {focused.id!r} after {trigger}"
        )


@pytest.mark.asyncio
async def test_converted_notes_site_keeps_focus_on_a_real_key_press(monkeypatch):
    """The same guarantee via the real keyboard, not a programmatic press."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=_notes(4))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_notes_canvas(host, pilot)
        screen.query_one("#library-notes-select-toggle", Button).focus()
        await pilot.pause()

        recompose_calls = _spy_screen_recomposes(monkeypatch)

        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert recompose_calls == []  # see the sibling test: discriminating
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


@pytest.mark.asyncio
async def test_notes_row_press_to_editor_keeps_focus_inside_the_canvas(monkeypatch):
    """The list -> loading -> editor row press is a DOUBLE canvas sync.

    Dev's row press syncs the canvas to its loading surface, then again to the
    editor once the detail lands. Two syncs mean the first recompose can still
    be awaiting ``mount_all`` when the second one's state arrives -- so a
    follow-up fired at the end of the first recompose runs against children
    that are already stale, and focus lands nowhere useful.

    ``_apply_post_compose_state`` already gates on its own mounted children;
    this pins the same guarantee for the queued follow-up itself.
    """
    app = _build_test_app()
    _seed_conversations(app, [], notes=_notes(4))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_notes_canvas(host, pilot)
        row = screen.query_one("#library-notes-row-0", Button)
        row.focus()
        await pilot.pause()

        row.press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        await pilot.pause()
        await pilot.pause()

        focused = screen.focused
        assert focused is not None
        canvas = screen.query_one("#library-notes-canvas")
        assert canvas in focused.ancestors_with_self, (
            f"focus escaped the notes canvas to {focused.id!r} on row -> editor"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "row_id", "canvas_selector", "focus_selector"),
    [
        (
            "conversations",
            LIBRARY_ROW_BROWSE_CONVERSATIONS,
            "#library-conversations-canvas",
            "#library-conversations-filter",
        ),
        (
            "media",
            LIBRARY_ROW_BROWSE_MEDIA,
            "#library-media-canvas",
            "#library-media-type-filter",
        ),
        (
            "prompts",
            LIBRARY_ROW_BROWSE_PROMPTS,
            "#library-prompts-canvas",
            "#library-prompts-retry",
        ),
        (
            "skills",
            LIBRARY_ROW_BROWSE_SKILLS,
            "#library-skills-canvas",
            "#library-skills-filter",
        ),
    ],
    ids=("conversations", "media", "prompts", "skills"),
)
async def test_entry_canvas_sync_restores_portable_focus_and_scroll(
    kind,
    row_id,
    canvas_selector,
    focus_selector,
):
    """Removing the shared finish callback loses focus or rendered completion."""
    app = _build_test_app()
    conversations = [
        {
            "title": f"Conversation {index}",
            "conversation_id": f"chat-{index}",
            "message_count": index,
            "updated_at": f"2026-06-{(index % 28) + 1:02d}T10:00:00Z",
        }
        for index in range(24)
    ]
    media = _media(24)
    _seed_conversations(app, conversations, media=media)
    app.prompt_scope_service = _FakePromptScopeService(count=1)
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": f"skill-{index}"} for index in range(24)]
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_COMPACT_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(row_id)
        await _wait_for_selector(screen, pilot, focus_selector)
        await screen.workers.wait_for_complete()
        await pilot.pause()

        canvas = screen.query_one(canvas_selector)
        focused_before = screen.query_one(focus_selector)
        focused_before.focus()
        scroll_before: tuple[int, int] | None = None
        if kind == "skills":
            canvas.styles.height = 6
            canvas.refresh(layout=True)
        await pilot.pause()
        if kind == "skills":
            assert int(canvas.max_scroll_y) > 0, (
                "Skills scroll preservation needs an overflowing canvas."
            )
            canvas.scroll_to(
                y=canvas.max_scroll_y,
                animate=False,
                force=True,
                immediate=True,
            )
            await pilot.pause()
            scroll_before = (int(canvas.scroll_x), int(canvas.scroll_y))
            assert scroll_before != (0, 0), (
                "Skills scroll preservation needs a genuinely scrolled setup."
            )

        if kind == "prompts":
            screen._library_prompts_browse_error = "Changed prompt entry data"
            screen._library_snapshot_state_generation += 1
            screen._library_entry_reconcile_dirty = True
            generation = screen._library_snapshot_state_generation
            route_key = screen._library_entry_route_key()
            identity = screen._capture_library_entry_focus()
            finish = partial(
                screen._finish_library_entry_canvas_sync,
                identity,
                generation=generation,
                route_key=route_key,
            )
            assert _sync_library_canvas(
                screen,
                kind,
                then=finish,
                allow_screen_fallback=False,
            )
        else:
            records = dict(screen._local_source_records)
            counts = dict(screen._local_source_counts)
            if kind == "conversations":
                records["conversations"] = (
                    *records["conversations"],
                    {
                        "title": "Changed conversation",
                        "conversation_id": "chat-new",
                        "message_count": 1,
                        "updated_at": "2026-08-13T10:00:00Z",
                    },
                )
                counts["conversations"] += 1
            elif kind == "media":
                records["media"] = (
                    *records["media"],
                    {
                        "id": 99,
                        "media_id": "99",
                        "title": "Changed media",
                        "type": "document",
                        "content": "Changed",
                        "ingestion_date": "2026-08-13T10:00:00Z",
                    },
                )
                counts["media"] += 1
            else:
                skill_count, skill_context = records["skills"]
                changed_context = dict(skill_context)
                changed_context["available_skills"] = [
                    *skill_context["available_skills"],
                    {"name": "skill-new"},
                ]
                records["skills"] = (skill_count + 1, changed_context)
            screen._apply_local_source_snapshot(
                records,
                counts,
                dict(screen._local_source_total_known),
                screen._library_lookup_error,
                screen._library_lookup_recovery_state,
                dict(screen._library_study_counts),
            )

        await pilot.pause()
        await pilot.pause()

        assert screen.query_one(canvas_selector) is canvas
        assert screen.focused is not None
        assert screen.focused.id == focus_selector.lstrip("#")
        if scroll_before is not None:
            assert (int(canvas.scroll_x), int(canvas.scroll_y)) == scroll_before
        assert screen._library_snapshot_rendered_generation == (
            screen._library_snapshot_state_generation
        )


@pytest.mark.asyncio
async def test_entry_canvas_sync_does_not_focus_an_unrelated_replacement_row():
    """Falling back from a missing semantic row to its reused index is wrong."""
    app = _build_test_app()
    conversations = [
        {
            "title": "Outgoing",
            "conversation_id": "chat-outgoing",
            "message_count": 1,
            "updated_at": "2026-08-13T10:00:00Z",
        },
        {
            "title": "Survivor",
            "conversation_id": "chat-survivor",
            "message_count": 1,
            "updated_at": "2026-08-12T10:00:00Z",
        },
    ]
    _seed_conversations(app, conversations)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        screen._start_library_conversation_page_request(1, "")
        row = await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        row.focus()
        await pilot.pause()

        records = dict(screen._local_source_records)
        records["conversations"] = (
            {
                "title": "Replacement at row zero",
                "conversation_id": "chat-replacement",
                "message_count": 1,
                "updated_at": "2026-08-14T10:00:00Z",
            },
            conversations[1],
        )
        screen._apply_local_source_snapshot(
            records,
            dict(screen._local_source_counts),
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )
        await pilot.pause()
        await pilot.pause()

        assert getattr(screen.focused, "conversation_id", None) != "chat-replacement"


@pytest.mark.asyncio
async def test_overlapping_reconciles_retain_original_semantic_focus():
    """A replace-latest callback must not discard the first focus capture."""
    app = _build_test_app()
    _seed_conversations(app, [])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        focus_target = await _wait_for_selector(
            screen, pilot, "#library-conversations-filter"
        )
        focus_target.focus()
        await pilot.pause()
        route_key = screen._library_entry_route_key()

        first_generation = screen._library_snapshot_state_generation + 1
        screen._library_snapshot_state_generation = first_generation
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (first_generation, route_key)
        await screen._reconcile_library_entry_state(first_generation, route_key)

        second_generation = first_generation + 1
        screen._library_snapshot_state_generation = second_generation
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (second_generation, route_key)
        await screen._reconcile_library_entry_state(second_generation, route_key)
        await pilot.pause()
        await pilot.pause()

        assert screen.focused is not None
        assert screen.focused.id == "library-conversations-filter"
        assert screen._library_snapshot_rendered_generation == second_generation


@pytest.mark.asyncio
async def test_strict_failure_retry_retains_original_semantic_focus(monkeypatch):
    """Clearing a failed strict callback must not clear its focus capture."""
    app = _build_test_app()
    _seed_conversations(app, [])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_CONVERSATIONS)
        focus_target = await _wait_for_selector(
            screen, pilot, "#library-conversations-filter"
        )
        canvas = screen.query_one("#library-conversations-canvas")
        original_sync = canvas.sync_state
        attempts = 0

        def fail_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("forced strict-sync failure")
            return original_sync(*args, **kwargs)

        monkeypatch.setattr(canvas, "sync_state", fail_once)
        focus_target.focus()
        await pilot.pause()
        generation = screen._library_snapshot_state_generation + 1
        route_key = screen._library_entry_route_key()
        screen._library_snapshot_state_generation = generation
        screen._library_entry_reconcile_dirty = True
        screen._library_entry_reconcile_pending = (generation, route_key)

        await screen._reconcile_library_entry_state(generation, route_key)
        await pilot.pause()
        await pilot.pause()

        assert attempts == 2
        assert screen.focused is not None
        assert screen.focused.id == "library-conversations-filter"
        assert screen._library_snapshot_rendered_generation == generation


@pytest.mark.asyncio
async def test_automatic_notes_reconcile_restores_richer_control_focus():
    """Erasing focus before the Notes capture redirects it to a fallback."""
    notes = _notes(4)
    app = _build_test_app()
    _seed_conversations(app, [], notes=notes)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _open_notes_canvas(host, pilot)
        focus_target = screen.query_one("#library-notes-select-toggle", Button)
        focus_target.focus()
        await pilot.pause()
        records = dict(screen._local_source_records)
        records["notes"] = (
            *records["notes"],
            {
                "id": "note-new",
                "title": "New note",
                "content": "New body",
                "last_modified": "2026-08-13T10:00:00Z",
            },
        )
        counts = dict(screen._local_source_counts)
        counts["notes"] += 1

        screen._apply_local_source_snapshot(
            records,
            counts,
            dict(screen._local_source_total_known),
            screen._library_lookup_error,
            screen._library_lookup_recovery_state,
            dict(screen._library_study_counts),
        )
        await pilot.pause()
        await pilot.pause()

        assert screen.focused is not None
        assert screen.focused.id == "library-notes-select-toggle"
