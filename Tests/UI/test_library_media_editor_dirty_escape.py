"""TASK-34000.22: the Media reader's analysis and metadata forms keep unsaved
text on Escape and ask to save or discard first (ADR-055 Pattern D).

The review (nl-v-l-08, j4) typed an analysis by hand and pressed Escape out
of habit: the Reader showed "No analysis yet." and the paragraph was gone,
with the footer reading "esc close" the whole time. The same unguarded
discard sat behind the Edit metadata form, the always-visible "‹ Back",
a click on another row, a rail-row switch and the nav bar -- and, because
no draft existed, behind ANY viewer rebuild (a Reader tab switch, More,
Find), which re-prefilled the form from the saved value.

Cause (verified on 0cdd4953f2): ``action_library_media_viewer_back``
cleared ``editing`` / ``editing_analysis`` unconditionally (its docstring
said there was no dirty field to veto on); both Cancel handlers discarded;
``LibraryMediaState`` held no typed text, so ``_compose_edit_form`` and
``_compose_analysis_edit_form`` composed from ``viewer.edit_fields`` /
``viewer.analysis`` every time.

Real app stylesheet (``LibraryProductionCSSHarness``), the production-shaped
exact-page fake, real keys and clicks, painted text through the compositor,
the toast recorder from the quit-guard support. One boot per test; the
whole file stays inside the 35 s lane budget. The fix-round-1 arms (the
Items filter / 0-result / Clear pin, the row press / Up-Down / Tab / Back
boot, the tab reveal, the armed-delete Escape and the Media rail return)
are the slower siblings in
``test_library_media_editor_dirty_escape_extended.py``, also in the census.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.library_quit_guard_support import _record_toasts
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
)
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY,
    LIBRARY_MEDIA_EDIT_DIRTY_VETO_COPY,
    LIBRARY_PROMPT_DIRTY_ESCAPE_CHIP,
)
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

SIZE = (160, 45)
#: Row 1 of the two-item fixture (newest first): "Interview Recording",
#: backing id 1, no analysis -- the "No analysis yet." case the review hit.
ROW_1 = "local:media:1"


def _media_items() -> list[dict[str, object]]:
    """``_two_media_items`` with production-shaped ids (``"1"``, ``"2"``:
    what ``LocalMediaReadingService`` returns), so the by-link door of the
    last test resolves a bare id exactly as Task 1's lane file does."""
    items = _two_media_items()
    for item in items:
        item["id"] = str(item["id"]).removeprefix("media-")
    return items


def _host():
    app = _build_test_app()
    app.library_new_profile_admission = False
    _seed_conversations(app, _two_conversations(), media=_media_items())
    return LibraryProductionCSSHarness(app), app, app.media_reading_scope_service


def _painted(host, region) -> str:
    strips = list(host.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text.rstrip()
        for y in range(region.y, min(region.bottom, len(strips)))
    )


def _painted_footer(host, screen) -> str:
    return _painted(host, screen.query_one(AppFooterStatus).region)


def _label(screen, selector: str) -> str:
    return str(screen.query_one(selector, Button).label)


async def _open_row_1(screen, pilot) -> None:
    """Setup, not the subject: ``press()`` on the rail and the row (what
    ``test_library_shell`` does), so the compact 80x24 shell -- where the rail
    is collapsed -- opens the same item as the wide one."""
    screen.query_one("#library-row-browse-media", Button).press()
    row = await _wait_for_selector(screen, pilot, "#library-media-row-1")
    row.press()
    await _wait_for_condition(
        pilot,
        lambda: screen._media_state.reader_session.loaded_id == ROW_1
        and screen._media_state.reader_session.pending_request is None,
        message=lambda: f"row 1 never loaded: {screen._media_state.reader_session!r}",
    )
    await pilot.pause()


async def _open_analysis_form(screen, pilot) -> TextArea:
    """Analysis tab ▸ "Add analysis" ▸ click into the TextArea (the review's
    L-07: the press leaves focus on the Items row, so the user clicks in)."""
    screen.query_one("#library-media-reader-select-analysis", Button).press()
    edit = await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")
    edit.press()
    area = await _wait_for_selector(screen, pilot, "#library-media-analysis-edit-text")
    assert isinstance(area, TextArea)
    await pilot.click("#library-media-analysis-edit-text")
    await _wait_for_condition(
        pilot,
        lambda: screen.focused is area,
        message=lambda: f"the analysis TextArea never took focus: {screen.focused!r}",
    )
    return area


async def _type(pilot, text: str) -> None:
    for char in text:
        await pilot.press("space" if char == " " else char)


async def _wait_for_text(screen, pilot, text: str) -> None:
    await _wait_for_condition(
        pilot,
        lambda: screen.query_one("#library-media-analysis-edit-text", TextArea).text
        == text,
        message=lambda: (
            "typed analysis never landed: "
            f"{screen.query_one('#library-media-analysis-edit-text', TextArea).text!r}"
        ),
    )


async def _open_edit_form(screen, pilot) -> Input:
    """More ▸ Edit metadata (pressed: the primary toolbar re-lays out after
    the detail lands, and a coordinate click taken before that landed on the
    Info tab) ▸ click into the Title input. The strip stays open across Edit
    ▸ Discard, so More is pressed only when the strip is not showing --
    More toggles, and a second press would close it under the Edit press."""
    if not screen.query("#library-media-edit"):
        screen.query_one("#library-media-reader-more", Button).press()
    edit = await _wait_for_selector(screen, pilot, "#library-media-edit")
    edit.press()
    title = await _wait_for_selector(screen, pilot, "#library-media-edit-title")
    assert isinstance(title, Input)
    await pilot.click("#library-media-edit-title")
    await _wait_for_condition(
        pilot,
        lambda: screen.focused is title,
        message=lambda: f"the Title input never took focus: {screen.focused!r}",
    )
    return title


@pytest.mark.parametrize("size", [(160, 45), (80, 24)], ids=["160x45", "80x24"])
async def test_escape_keeps_a_dirty_analysis_and_names_the_blocker(monkeypatch, size):
    """AC#1, AC#3, AC#5, AC#6: Escape over a typed analysis keeps the form and
    the text, the toast names the unsaved analysis, the footer chip reads
    "esc save or discard first", Cancel reads "Discard changes" -- and that
    press is what drops the text ("No analysis yet.")."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        area = await _open_analysis_form(screen, pilot)
        await _type(pilot, "Key points")
        await _wait_for_text(screen, pilot, "Key points")
        await pilot.pause()
        assert ("esc", LIBRARY_PROMPT_DIRTY_ESCAPE_CHIP) in (
            screen._footer_shortcut_registration[1]
        ), screen._footer_shortcut_registration
        assert "esc save or discard first" in _painted_footer(host, screen)

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert screen._media_state.editing_analysis is True
        assert screen.query_one("#library-media-analysis-edit-text", TextArea) is area
        assert area.text == "Key points"
        assert "Key points" in _painted(host, area.region)
        assert toasts and toasts[-1] == LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY, toasts
        assert _label(screen, "#library-media-analysis-cancel") == "Discard changes"
        cancel = screen.query_one("#library-media-analysis-cancel", Button)
        assert "Discard changes" in _painted(host, cancel.region)

        await pilot.click("#library-media-analysis-cancel")
        await _wait_for_condition(
            pilot,
            lambda: not screen._media_state.editing_analysis
            and bool(screen.query("#library-media-viewer-analysis-text")),
            message="Discard changes never closed the form",
        )
        await pilot.pause()
        text = screen.query_one("#library-media-viewer-analysis-text", Static)
        assert "No analysis yet." in _painted(host, text.region)
        assert screen._media_state.analysis_draft is None


async def test_escape_keeps_dirty_metadata_and_a_clean_form_still_closes(monkeypatch):
    """AC#2, AC#3, AC#5: a changed Title keeps the form open on Escape with the
    edited value, Cancel reads "Discard changes" and drops it; an untouched
    form still closes on one Escape and its Cancel reads "Cancel"."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        title = await _open_edit_form(screen, pilot)
        saved_title = title.value
        assert saved_title == "Interview Recording"
        await pilot.press("end")
        await pilot.press("exclamation_mark")
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-media-edit-title", Input).value
            == saved_title + "!",
            message="the typed character never landed",
        )
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert screen._media_state.editing is True
        assert screen.query_one("#library-media-edit-title", Input).value == saved_title + "!"
        assert toasts and toasts[-1] == LIBRARY_MEDIA_EDIT_DIRTY_VETO_COPY, toasts
        assert _label(screen, "#library-media-edit-cancel") == "Discard changes"

        await pilot.click("#library-media-edit-cancel")
        await _wait_for_condition(
            pilot,
            lambda: not screen._media_state.editing
            and not screen.query("#library-media-edit-title"),
            message="Discard changes never closed the metadata form",
        )
        assert screen._media_state.edit_draft is None

        # A clean form: one Escape closes it, as it always did.
        await _open_edit_form(screen, pilot)
        assert _label(screen, "#library-media-edit-cancel") == "Cancel"
        assert ("esc", "close") in screen._footer_shortcut_registration[1]
        before = len(toasts)
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: not screen._media_state.editing
            and not screen.query("#library-media-edit-title"),
            message="a clean Escape no longer closes the form",
        )
        assert len(toasts) == before, toasts[before:]


async def test_switching_reader_tabs_keeps_the_typed_analysis():
    """The recompose class: a Read ▸ Analysis round trip rebuilds the form,
    which used to come back prefilled from the SAVED (empty) analysis."""
    host, _app, _service = _host()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        await _open_analysis_form(screen, pilot)
        await _type(pilot, "Key points")
        await _wait_for_text(screen, pilot, "Key points")

        await pilot.click("#library-media-reader-select-read")
        await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        assert not screen.query("#library-media-analysis-edit-text")
        await pilot.click("#library-media-reader-select-analysis")
        area = await _wait_for_selector(screen, pilot, "#library-media-analysis-edit-text")

        assert screen._media_state.editing_analysis is True
        assert area.text == "Key points"
        await pilot.pause()
        assert "Key points" in _painted(host, area.region)


async def test_rail_switch_and_nav_flush_refuse_to_drop_a_dirty_analysis(monkeypatch):
    """AC#4: a rail-row press and the nav bar's flush both refuse while the
    analysis is dirty, each saying so once; the quit flush refuses silently."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        area = await _open_analysis_form(screen, pilot)
        await _type(pilot, "Key points")
        await _wait_for_text(screen, pilot, "Key points")

        await screen._select_library_rail_row(LIBRARY_ROW_BROWSE_NOTES)
        await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        assert screen._media_state.view == "viewer"
        assert screen._media_state.editing_analysis is True
        assert screen.query_one("#library-media-analysis-edit-text", TextArea) is area
        assert area.text == "Key points"
        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY], toasts

        assert await screen.flush_pending_work(destination="Console") is False
        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY] * 2, toasts
        assert screen._media_state.editing_analysis is True
        assert area.text == "Key points"

        assert await screen.flush_pending_work(quitting=True) is False
        assert len(toasts) == 2, toasts


async def _open_item_2_by_bare_id(screen, pilot) -> None:
    """Task 1's by-link door, with the two-item page."""
    await screen._open_library_item_by_id("media", "2")
    await _wait_for_selector(screen, pilot, "#library-media-row-0")
    browse = screen._library_media_browse_controller

    def landed() -> bool:
        session = screen._media_state.reader_session
        return (
            session.loaded_id == "local:media:2"
            and session.pending_request is None
            and not browse.loading
            and browse.applied_scope is not None
            and browse.applied_scope == browse.requested_scope
        )

    await _wait_for_condition(pilot, landed, message="open-by-link of 2 never settled")
    await pilot.pause()
    await pilot.pause()


async def test_a_dirty_metadata_save_targets_the_opened_item(monkeypatch):
    """Task 1 x Task 2 seam: a dirty metadata form opened by bare id saves to
    the OPENED item (never row 0), the save clears the draft, and the next
    form opens clean."""
    host, app, service = _host()
    _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_item_2_by_bare_id(screen, pilot)
        title = await _open_edit_form(screen, pilot)
        assert title.value == "Product Demo Video"
        await pilot.press("end")
        await pilot.press("exclamation_mark")
        await _wait_for_condition(
            pilot,
            lambda: screen.query_one("#library-media-edit-title", Input).value
            == "Product Demo Video!",
            message="the typed character never landed",
        )
        assert _label(screen, "#library-media-edit-cancel") == "Discard changes"

        await pilot.click("#library-media-edit-save")
        await _wait_for_condition(
            pilot,
            lambda: bool(service.update_calls) and not screen._media_state.editing,
            message="Edit Save never completed",
        )

        assert service.update_calls[-1]["media_id"] == 2
        assert service.update_calls[-1]["title"] == "Product Demo Video!"
        assert screen._media_state.edit_draft is None
        assert screen._media_state.reader_session.loaded_id == "local:media:2"

        await _open_edit_form(screen, pilot)
        assert screen.query_one("#library-media-edit-title", Input).value == (
            "Product Demo Video!"
        )
        assert _label(screen, "#library-media-edit-cancel") == "Cancel"


async def test_export_beside_a_dirty_form_is_refused_with_one_toast(monkeypatch):
    """Fix round 1, C-1: the Items pane's Export… used to move the destination
    to Export while the dirty form stayed set behind it, and every way back
    was then refused for a form nobody could see (the only exit was Ctrl+Q).
    Now the opener refuses beside its note flush: the row stays Media, the
    form and its text stay on screen, and exactly one toast says why."""
    host, app, _service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        area = await _open_analysis_form(screen, pilot)
        await _type(pilot, "Key points")
        await _wait_for_text(screen, pilot, "Key points")

        screen.query_one("#library-media-export", Button).press()
        await _wait_for_condition(
            pilot, lambda: bool(toasts), message="Export… neither opened nor refused"
        )
        for _ in range(6):
            await pilot.pause()

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA
        assert toasts == [LIBRARY_MEDIA_ANALYSIS_DIRTY_VETO_COPY], toasts
        assert screen._media_state.editing_analysis is True
        assert screen._media_state.analysis_draft == "Key points"
        assert screen.query_one("#library-media-analysis-edit-text", TextArea) is area
        assert area.text == "Key points"
        assert not screen.query("#library-export-name")
