"""TASK-21116: per-click Library paths stay canvas-scoped (no screen rebuild).

Mounted evidence for the second canvas-scoped conversion wave: media open
(row press and ``_open_library_item_by_id``), media viewer exits (Escape /
"‹ Back to list", including the edit/confirm sub-state Cancels), the
skills/prompts inline Import open/cancel, and the section "Export…" entry.
Each test pins BOTH that zero whole-screen recomposes happen and that the
shell (rail) keeps widget identity across the interaction.

When the harness boots into the legacy compact UNKNOWN-lifecycle rail,
media-path tests disclose the full rail via ``#library-rail-explore-all``;
the current lifecycle exposes those rows directly. The prompts/skills tests
reuse the real-service wiring that already resolves the full rail (the
``test_real_prompt_and_skill_rows_keep_their_canvas_identity`` recipe).
"""

from __future__ import annotations

from statistics import median
from time import perf_counter
from unittest.mock import patch

import pytest
from textual.widgets import Button

from Tests.Skills.test_skills_library_flow import (
    _real_skills_scope_service,
    _skill_content,
    _wire_empty_non_skill_services,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_canvas_scoped_sync import _screen_recompose_spy
from Tests.UI.test_library_prompts_canvas import (
    _real_prompt_scope_service,
    _wire_empty_non_prompt_services,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _disclose_media_viewer_more_actions,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_INGEST_EXPORT,
)
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.Widgets.Library import LibraryMediaCanvas
from tldw_chatbook.Widgets.Library.library_media_viewer import LibraryMediaViewer

# task-31249: every test in this module mounts the real Library app through
# LibraryHarness, whose service/config reads go through the config-participant
# admission. Under the per-test sandbox redirect that admission fails closed
# with ``RecoveryRequired("raw_source_selection_changed")`` (the TASK-32628
# class; whole-module enrollment follows the TASK-32873 precedent -- the
# suite is all real-app mounts, not a mostly-sandbox suite with a few).
pytestmark = pytest.mark.bootstrap_profile


async def _wait_for_selector_gone(screen, pilot, selector, *, attempts=80):
    """Wait until ``selector`` no longer matches anything on the screen."""
    for _ in range(attempts):
        if not screen.query(selector):
            return
        await pilot.pause(0.02)
    raise AssertionError(f"{selector} never left the DOM")


async def _boot_media_library(host, pilot):
    """Mount the Library shell, disclose legacy compact rail, enter Media."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    if screen.query("#library-rail-explore-all"):
        screen.query_one("#library-rail-explore-all", Button).press()
    await _wait_for_selector(screen, pilot, "#library-row-browse-media")
    screen.query_one("#library-row-browse-media", Button).press()
    await _wait_for_selector(screen, pilot, "#library-media-row-0")
    return screen


def _media_app_host():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=None, media=_two_media_items())
    return LibraryHarness(app)


@pytest.mark.asyncio
async def test_media_row_open_and_detail_arrival_are_canvas_scoped() -> None:
    """A media row press opens the viewer with zero whole-screen recomposes.

    Covers BOTH converted halves of the open: the click-time list->viewer
    canvas-child swap (``_open_library_media_viewer``) and the detail
    worker's arrival projection (``_refresh_library_media_detail`` ->
    ``_apply_library_media_active_surface``) -- waiting for the viewer's
    content search control means the fetched detail actually rendered.
    """
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        rail_before = screen.query_one("#library-rail")
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            screen.query_one("#library-media-row-0", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        assert calls == []
        assert screen.query_one("#library-rail") is rail_before
        assert screen.query_one("#library-media-viewer", LibraryMediaViewer)


@pytest.mark.asyncio
async def test_media_viewer_back_is_canvas_scoped_and_restores_list_focus() -> None:
    """ "‹ Back to list" swaps only the canvas child and re-arms row focus."""
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        rail_before = screen.query_one("#library-rail")
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            # task-31272: the "‹ Back" control itself is compact-only now
            # (the three-pane shell has no list mode of its own), so this
            # drives the exit seam the button and Escape share.
            screen._exit_library_media_viewer()
            await _wait_for_selector(screen, pilot, "#library-media-row-0")
            await pilot.pause()
            await pilot.pause()
        assert calls == []
        assert screen.query_one("#library-rail") is rail_before
        assert screen.query_one("#library-media-canvas", LibraryMediaCanvas)
        # task-2856 AC1 must survive the conversion: the entry-focus arm
        # now rides the swap continuation, and its immediate attempt runs
        # against the mounted list rows.
        focused_id = str(getattr(screen.focused, "id", "") or "")
        assert focused_id.startswith("library-media-row"), focused_id


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_media_viewer_substate_escape_is_viewer_scoped() -> None:
    """Escape out of metadata-edit rebuilds only the mounted viewer."""
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        screen.query_one("#library-media-row-0", Button).press()
        # task-31249: "Edit metadata" composes only inside the viewer's
        # collapsed "More" disclosure (d3c4b44a9b), so disclose it first.
        await _disclose_media_viewer_more_actions(screen, pilot)
        # Entering edit mode is out of this conversion's scope and may
        # rebuild the screen -- do it OUTSIDE the spy window.
        screen.query_one("#library-media-edit", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-cancel")
        viewer_before = screen.query_one("#library-media-viewer", LibraryMediaViewer)
        rail_before = screen.query_one("#library-rail")
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            screen.action_library_media_viewer_back()
            await _wait_for_selector(screen, pilot, "#library-media-edit")
        assert calls == []
        assert screen.query_one("#library-rail") is rail_before
        # Viewer-scoped recompose: the viewer NODE survives; only its
        # children were rebuilt back to the read-only form.
        assert (
            screen.query_one("#library-media-viewer", LibraryMediaViewer)
            is viewer_before
        )
        assert screen._media_state.editing is False
        assert screen._media_state.view == "viewer"


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_open_item_by_id_media_is_canvas_scoped() -> None:
    """A cross-canvas direct media open uses only the sanctioned seams.

    task-21116 pinned "never rebuilds the screen" for this open when every
    canvas lived under one ordinary canvas host, so the open could be a
    canvas-child swap. Since the adaptive reader shells (d99fb4a9ca) the
    content shells live IN the shell grid: a conversations -> media move
    swaps the shell-grid child and takes the same structural
    ``Screen.recompose()`` a rail-row press takes (measured at this base:
    a plain rail conversations -> media press also rebuilds the rail
    widget). What is still pinned here (task-31249):

    - zero ``refresh(recompose=True)`` calls. The retired whole-screen
      recompose STORM -- five racing refreshes at the 2026-09 filing
      (pre-mount browse/facet requests plus the media detail worker
      racing its own mount) -- must not come back;
    - the open lands on the media viewer with the rail selection moved
      and the rail mounted, without the mount race (the DuplicateIds
      shape canvas_sync.py records).
    """
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        if screen.query("#library-rail-explore-all"):
            screen.query_one("#library-rail-explore-all", Button).press()
        await _wait_for_selector(screen, pilot, "#library-row-browse-notes")
        # Start from a NON-media canvas so the open crosses canvas kinds
        # (the RAG "Open" shape).
        screen.query_one("#library-row-browse-conversations", Button).press()
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            await screen._open_library_item_by_id("media", "media-1")
            await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
            # The storm was worker-driven, not mount-driven: settle the
            # detail + browse/facet workers inside the spy window too.
            for _ in range(20):
                await pilot.pause(0.05)
        assert calls == []
        assert screen.query_one("#library-rail") is not None
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        assert screen._media_state.view == "viewer"


@pytest.mark.asyncio
async def test_open_item_by_id_notes_keeps_route_owned_source_strip() -> None:
    """A cross-kind notes open still mounts the Database/Files source strip.

    The strip is route-owned chrome composed OUTSIDE the canvas host, so
    the targeted open seam must detect the structural mismatch and take
    the sanctioned whole-screen path -- a canvas-child-only swap would
    land the notes editor without its source strip. This is the
    deliberately-not-converted case recorded in the task notes.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[
            {
                "id": "n-1",
                "title": "Probe note",
                "content": "Body",
                "last_modified": "2026-07-06T08:00:00Z",
                "version": 1,
            }
        ],
        media=_two_media_items(),
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        if screen.query("#library-rail-explore-all"):
            screen.query_one("#library-rail-explore-all", Button).press()
        await _wait_for_selector(screen, pilot, "#library-row-browse-conversations")
        screen.query_one("#library-row-browse-conversations", Button).press()
        await _wait_for_selector(screen, pilot, "#library-conversations-canvas")
        assert not screen.query("#library-notes-source-strip")
        await screen._open_library_item_by_id("notes", "n-1")
        await _wait_for_selector(screen, pilot, "#library-notes-canvas")
        await _wait_for_selector(screen, pilot, "#library-notes-source-strip")
        assert screen._notes_state.view == "editor"


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_export_open_from_media_is_canvas_scoped() -> None:
    """The media section "Export…" action opens Export without the retired seam.

    task-21116 pinned a canvas-child swap with rail widget identity. Since
    the adaptive reader shells (d99fb4a9ca) the media list lives inside an
    adaptive shell whose library pane owns the rail, and the export canvas
    is an ordinary route: the move crosses shells and takes the projection's
    structural ``Screen.recompose()``, which rebuilds the rail -- the same
    census-era "rail was recomposed" symptom, now by the sanctioned seam.
    What is still pinned here (task-31249):

    - zero ``refresh(recompose=True)`` calls (no retired whole-screen
      recompose storm);
    - the press DEADLOCKS no longer: the projection used to be awaited
      inline from this canvas's own press handler, and the structural
      recompose tearing down that very canvas hung the press forever
      (fixed in ``_open_library_export_canvas``);
    - Export opens pre-scoped from the media section with the rail row
      selection moved.
    """
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            screen.query_one("#library-media-export", Button).press()
            await _wait_for_selector(screen, pilot, "#library-export-canvas")
            for _ in range(10):
                await pilot.pause(0.05)
        assert calls == []
        assert screen.query_one("#library-rail") is not None
        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT


@pytest.mark.asyncio
async def test_prompts_import_open_and_cancel_are_canvas_scoped(tmp_path) -> None:
    """The prompts Import row opens/closes via its own canvas, caret managed."""
    prompt_path = tmp_path / "prompts"
    prompt_path.mkdir()
    prompt_db, prompt_service = _real_prompt_scope_service(prompt_path)
    prompt_db.add_prompt(
        name="Import probe prompt",
        author="Test",
        details="Recompose probe",
        system_prompt="Stay concise.",
        user_prompt="Summarize {text}",
        keywords=["test"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = prompt_service
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        await _wait_for_selector(screen, pilot, "#library-prompts-import")
        canvas_before = screen.query_one("#library-prompts-canvas")
        rail_before = screen.query_one("#library-rail")
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            screen.query_one("#library-prompts-import", Button).press()
            path_input = await _wait_for_selector(
                screen, pilot, "#library-prompts-import-path"
            )
            await pilot.pause()
            assert screen.focused is path_input
            screen.query_one("#library-prompts-import-cancel", Button).press()
            await _wait_for_selector_gone(screen, pilot, "#library-prompts-import-path")
            await pilot.pause()
        assert calls == []
        assert screen.query_one("#library-prompts-canvas") is canvas_before
        assert screen.query_one("#library-rail") is rail_before
        focused_id = str(getattr(screen.focused, "id", "") or "")
        assert focused_id == "library-prompts-import", focused_id


@pytest.mark.asyncio
async def test_skills_import_open_and_cancel_are_canvas_scoped(tmp_path) -> None:
    """The skills Import row opens/closes via its own canvas, caret managed."""
    skill_path = tmp_path / "skills"
    skill_path.mkdir()
    local_skills, skills_service = _real_skills_scope_service(skill_path)
    await local_skills.create_skill(
        name="probe-skill",
        content=_skill_content(title="Probe", description="Recompose probe"),
    )
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = skills_service
    app.local_skill_trust_service = None
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills").press()
        await _wait_for_selector(screen, pilot, "#library-skills-import")
        canvas_before = screen.query_one("#library-skills-canvas")
        rail_before = screen.query_one("#library-rail")
        calls, spy = _screen_recompose_spy()
        with patch.object(BaseAppScreen, "refresh", spy):
            screen.query_one("#library-skills-import", Button).press()
            path_input = await _wait_for_selector(
                screen, pilot, "#library-skills-import-path"
            )
            await pilot.pause()
            assert screen.focused is path_input
            screen.query_one("#library-skills-import-cancel", Button).press()
            await _wait_for_selector_gone(screen, pilot, "#library-skills-import-path")
            await pilot.pause()
        assert calls == []
        assert screen.query_one("#library-skills-canvas") is canvas_before
        assert screen.query_one("#library-rail") is rail_before
        focused_id = str(getattr(screen.focused, "id", "") or "")
        assert focused_id == "library-skills-import", focused_id


@pytest.mark.asyncio
async def test_media_row_open_latency_probe() -> None:
    """Measure click-to-settle for a media row open (viewer detail rendered).

    The task records the median externally; this test deliberately has no
    timing threshold because wall-clock assertions are not stable CI
    evidence (the 15457 probe's rule). Its behavioral assertion is that
    every measured click completes the row -> loaded-viewer -> back-to-list
    cycle.
    """
    host = _media_app_host()
    samples: list[float] = []
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        for _ in range(12):
            started = perf_counter()
            screen.query_one("#library-media-row-0", Button).press()
            await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
            samples.append((perf_counter() - started) * 1000.0)
            screen._exit_library_media_viewer()
            await _wait_for_selector(screen, pilot, "#library-media-row-0")
    assert len(samples) == 12
    assert all(sample > 0 for sample in samples)
    print(
        "TASK-21116 media-row open latency: "
        f"median={median(samples):.3f}ms samples="
        + ",".join(f"{sample:.3f}" for sample in samples)
    )
