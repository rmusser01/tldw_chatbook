"""Critique #10 media-row fixes: labelled age, applied-scope line, copy polish.

Covers tasks 32347 (the row age says it is an age), 32350 (a scope line
states the APPLIED filter, never the box's draft) and 32364 (a status word
never prefixes a title; one separator glyph; the import footer names what
Enter does at each step).
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_media_state import media_updated_age_copy
from tldw_chatbook.UI.Screens.library_screen import _sync_library_canvas
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
from Tests.UI.test_library_media_render_fixes import _apply_media_filter, _host
from Tests.UI.test_library_media_side_by_side import _open_media_list
from Tests.UI.test_library_shell import _wait_for_condition

#: ``_host()`` seeds "Interview Recording" (audio) and "Product Demo Video"
#: (video); "interview" is in exactly one title and no other row's content,
#: so a one-row applied scope is provable rather than incidental.
_ONE_HIT_QUERY = "interview"


def test_the_age_label_names_the_field_the_age_comes_from():
    """re-review finding A: the value is `last_modified`, so the word is
    "updated" -- the same word the preview pane uses for it. An "added"
    label would swap one ambiguity for another, since pressing Generate
    writes `last_modified` and would move a row's "added" age."""
    now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
    assert media_updated_age_copy("2026-09-11T11:50:00+00:00", now=now) == "updated 10m"
    assert media_updated_age_copy("2026-09-11T11:59:40+00:00", now=now) == "updated just now"
    assert media_updated_age_copy("", now=now) == ""


@pytest.mark.asyncio
async def test_the_media_header_states_the_applied_scope_not_the_typed_draft():
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, _ONE_HIT_QUERY)
        box = screen.query_one("#library-media-filter", Input)
        box.value = "a different draft"  # never submitted
        await pilot.pause()
        line = screen.query_one("#library-media-scope-line", Static)
        assert str(line.content).startswith("Media · 1 of "), str(line.content)
        assert f'filter “{_ONE_HIT_QUERY}”' in str(line.content), str(line.content)
        assert "a different draft" not in str(line.content), str(line.content)
        assert screen.query("#library-media-scope-clear")


def _ingest_enter_label(*, start_enabled: bool) -> str:
    """The Enter hint the Ingest footer registers for one Start-gate state."""
    from types import SimpleNamespace

    from tldw_chatbook.UI.Library_Modules.library_ingest_controller import (
        LibraryIngestController,
    )
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    stub = SimpleNamespace(
        LIBRARY_INGEST_SHORTCUTS=LibraryScreen.LIBRARY_INGEST_SHORTCUTS,
        _build_library_ingest_state=lambda: SimpleNamespace(
            start_enabled=start_enabled
        ),
        _library_ingest_start_consent=None,
        app_instance=None,
    )
    shortcuts = LibraryIngestController._library_ingest_shortcuts_for_current_state(
        stub
    )
    key, description = shortcuts[0]
    assert key == "enter", shortcuts
    return description


def test_the_import_footer_names_what_enter_does_at_each_step():
    """task-32364 AC#3: one label for two different actions taught the wrong
    thing at exactly the moment the user commits -- the first Enter on an
    unvalidated path validates it, only the second starts the import."""
    assert _ingest_enter_label(start_enabled=False) == "check this path"
    assert _ingest_enter_label(start_enabled=True) == "start import"


@pytest.mark.asyncio
async def test_the_import_footer_flips_when_the_gate_opens_without_a_recompose(
    tmp_path,
):
    """task-32364 AC#3, fix round 1: the label has to reach the FOOTER.

    Review finding 1 said nothing re-registers the Ingest footer when the
    Start gate opens. Tracing it under a real harness narrowed that: the
    COMMON path is saved by accident -- an empty ``type_groups`` filling in
    makes ``_update_library_ingest_dynamic_regions`` take its STRUCTURAL
    branch (``library_screen.py:19219-19228``), which recomposes and
    re-registers as a side effect. The hole is every gate transition that is
    NOT structural, and this drives one directly: the same pre-flight result
    applied twice, with only the blank path -- itself a Start gate -- filled
    in between. Without ``_resync_library_ingest_footer`` the footer keeps
    advertising "check this path" while Enter now starts the import.
    """
    from tldw_chatbook.Library.library_ingest_state import PreflightResult
    from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_ingest_keyboard import _enter_ingest_mode
    from Tests.UI.test_library_shell import (
        LIBRARY_TEST_SIZE,
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_conversations,
        _wait_for_library_shell,
    )

    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_ingest_mode(screen, pilot)

        # Queried fresh every time: a recompose replaces the footer widget,
        # and a handle bound once goes on reporting the detached copy's text.
        def footer_text() -> str:
            return screen.query_one(AppFooterStatus).shortcut_text

        assert "enter check this path" in footer_text(), footer_text()

        # The Start gate also wants a media DB seam, and this harness has
        # none; it is checked for presence only (library_screen.py:19406).
        host.app_instance.media_db = object()

        source = tmp_path / "a.pdf"
        source.write_text("dummy")
        form = screen._ingest_state.form
        # A blank path shuts the gate on its own, so the first application
        # lands the type groups with Start still closed.
        form.path = ""
        result = PreflightResult(
            type_groups={"pdf": [str(source)]},
            warnings=[],
            errors=[],
            total_size=1024,
            truncated=False,
            total_files=1,
        )
        screen._apply_library_ingest_preflight_result(
            result, screen._ingest_state.preflight_generation
        )
        await pilot.pause()
        assert screen._build_library_ingest_state().start_enabled is False
        assert "enter check this path" in footer_text(), footer_text()

        # Now the gate opens with `type_groups` unchanged, so nothing
        # recomposes and only the explicit resync can move the label.
        form.path = str(source)
        screen._apply_library_ingest_preflight_result(
            result, screen._ingest_state.preflight_generation
        )
        await pilot.pause()

        assert screen._build_library_ingest_state().start_enabled is True
        assert "enter start import" in footer_text(), footer_text()


@pytest.mark.asyncio
async def test_a_gate_change_on_a_suspended_screen_is_latched_for_resume():
    """Re-review finding B: skipping is fine, dropping is not.

    The resume pass re-registers the footer only when
    `_library_ingest_suspended_activity` is set, and nothing else
    re-registers for a reused, resumed screen. A bare return here left a
    gate that opened on another tab advertising the previous step's action
    once the user came back.
    """
    from Tests.UI.app_factory import _build_test_app
    from Tests.UI.test_library_ingest_keyboard import _enter_ingest_mode
    from Tests.UI.test_library_shell import (
        LIBRARY_TEST_SIZE,
        LibraryHarness,
        _active_library_screen,
        _seed_conversations,
        _two_conversations,
        _wait_for_library_shell,
    )

    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_ingest_mode(screen, pilot)

        controller = screen._ingest_controller
        screen._library_ingest_suspended_activity = False
        screen._library_screen_suspended = True
        try:
            controller._resync_library_ingest_footer()
        finally:
            screen._library_screen_suspended = False
        assert screen._library_ingest_suspended_activity is True


@pytest.mark.asyncio
async def test_a_prompt_row_does_not_repeat_the_canvas_it_is_already_on():
    """task-32364: "Prompt · " led every row on a canvas titled Prompts."""
    from tldw_chatbook.Library.library_prompts_state import (
        PromptListRow,
        PromptsListState,
    )
    from Tests.UI.test_library_prompts_canvas import _CanvasHost

    state = PromptsListState(
        rows=(
            PromptListRow(
                prompt_id=8,
                name="Outcome first",
                secondary="Reusable structure",
                artifact_type="prompt",
                type_label="Prompt",
                source_label="Local",
                lane_summary="has system and user text",
            ),
        ),
        count=1,
        sort="newest",
    )
    async with _CanvasHost(state).run_test() as pilot:
        label = str(pilot.app.query_one("#library-prompt-row-8", Button).label)
        assert "Local · has system and user text" in label, label
        assert "Prompt · " not in label, label


@pytest.mark.asyncio
async def test_the_scope_line_says_nothing_it_cannot_stand_behind():
    """Review finding 2: `of M` must respect `_local_source_total_known`.

    The screen keeps that flag beside the counts because the snapshot's
    count is sometimes a lower bound (a bounded preview page with no
    `total`), and every other consumer renders `5+` rather than `5`. A scope
    line claiming a flat `of 5` where the rail says `5+` is the one kind of
    statement this line exists to make impossible.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        assert "of " in str(
            screen.query_one("#library-media-scope-line", Static).content
        )

        screen._local_source_total_known["media"] = False
        _sync_library_canvas(screen, "media")
        await pilot.pause()

        line = str(screen.query_one("#library-media-scope-line", Static).content)
        assert " of " not in line, line
        assert line.startswith("Media · "), line


@pytest.mark.asyncio
async def test_the_scope_lines_clear_is_painted_inside_the_items_pane():
    """Review finding 3: the live-found off-screen Clear, pinned.

    `query_one(...).press()` is just as happy with a Button painted past the
    pane edge, which is exactly the state the CSS fix was made for -- an
    auto-width scope Static pushed its own Clear out of view once the Reader
    narrowed the list. A region assertion is the only thing that notices.
    """
    for size in ((235, 52), (100, 30)):
        host = _host()
        async with host.run_test(size=size) as pilot:
            screen = await _open_media_list(host, pilot)
            await _apply_media_filter(screen, pilot, _ONE_HIT_QUERY)
            await pilot.pause()

            clear = screen.query_one("#library-media-scope-clear", Button)
            canvas = screen.query_one(LibraryMediaCanvas)
            assert clear.region.width > 0, (size, clear.region)
            assert (
                clear.region.x + clear.region.width
                <= canvas.region.x + canvas.region.width
            ), (size, clear.region, canvas.region)


@pytest.mark.asyncio
async def test_a_clear_whose_request_failed_can_be_pressed_again():
    """Qodo review: the visible Clear must be able to retry.

    The scope line and its Clear are derived from the APPLIED result, so a
    browse that fails leaves the filtered page — and the Clear — on screen
    while `requested_scope` already holds the cleared target. Suppressing on
    the requested scope alone made every later press a no-op, with nothing
    the user could do to get back.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, _ONE_HIT_QUERY)

        # The state a failed clear leaves behind: requested is the target,
        # applied still carries the filter the rows and the Clear came from.
        controller = screen._library_media_browse_controller
        controller.requested_scope = dataclasses.replace(
            controller.requested_scope, query="", media_type=None, page=1
        )
        assert controller.applied_scope.query == _ONE_HIT_QUERY

        screen.query_one("#library-media-scope-clear", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_media_browse_controller.applied_scope.query,
            message="A second Clear press after a failed request did nothing.",
        )


@pytest.mark.asyncio
async def test_clearing_the_applied_filter_also_clears_the_box():
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, _ONE_HIT_QUERY)
        screen.query_one("#library-media-scope-clear", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query_one("#library-media-filter", Input).value,
            message="The scope line's Clear left a draft in the filter box.",
        )
        line = screen.query_one("#library-media-scope-line", Static)
        assert "filter" not in str(line.content), str(line.content)
        assert not screen.query("#library-media-scope-clear")
