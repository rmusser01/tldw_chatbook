"""Ctrl+S saves the Media reader's analysis and metadata forms (TASK-34000.40
N-33; TASK-34000.22 AC#7, deferred to .40 by the P1-A owner-question-1 ruling).

Both forms already had a Save button; Ctrl+S did nothing in either, so a typed
analysis waited on a mouse click. The owner's rule (ADR-031 refinement
2026-10-03) makes Ctrl+S each form's visible Save -- only while focus is in
that form, so a form left open on another Reader tab is never written by it.

Production stylesheet harness over the production-shaped exact-page media fake
(the TASK-34000.22 precedent, ``test_library_media_editor_dirty_escape.py``):
the fake records every ``save_analysis_version`` / ``update_media_item`` call
the screen makes, and applies it to the stored item the Reader re-reads.
Real keys, real clicks, painted text through the compositor.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.widgets import Input

from Tests.UI.library_quit_guard_support import _record_toasts
from Tests.UI.test_library_ctrl_s_saves_in_every_editor import (
    _focused_id,
    _press_processed,
    _record_resolved_keys,
)
from Tests.UI.test_library_media_editor_dirty_escape import (
    _host,
    _open_analysis_form,
    _open_edit_form,
    _open_row_1,
    _painted,
    _painted_footer,
    _type,
    _wait_for_text,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
)
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_MEDIA_EDIT_DIRTY_VETO_COPY,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

SIZE = (160, 45)
#: Review R1 N-1: a held write waits at most this long -- far past every wait
#: in the test -- and reaching it FAILS the test; it never releases silently.
HOLD_BOUND = 120.0


def _title_value(screen) -> str | None:
    """The metadata form's Title, re-queried: never a widget held across a
    Reader rebuild."""
    matches = screen.query("#library-media-edit-title")
    return matches.first(Input).value if matches else None


async def _reader_settled(screen, pilot) -> None:
    """Wait until an analysis save (or Discard) has fully settled and the
    Reader is rebuilt without the form (review R1 N-1).

    ``editing_analysis`` flips BEFORE the save worker's detail refresh
    rebuilds the Reader, so a widget captured on that signal alone can be
    replaced under the test. Settled means: the save's guard is clear (it
    clears in the worker's ``finally``, after the refresh); the refresh's
    request has settled; the Reader has projected the fetched detail
    (``composed_detail is detail``, the projection's own "already rendered"
    test in ``_recompose_library_media_detail_if_unrendered``) with no
    viewer rebuild pending; and the form is gone from the DOM.
    """

    def settled() -> bool:
        state = screen._media_state
        viewer = screen._mounted_library_media_viewer()
        return (
            not state.editing_analysis
            and not state.saving_analysis
            and state.reader_session.pending_request is None
            and state.detail is not None
            and state.composed_detail is state.detail
            and viewer is not None
            and not viewer._recompose_required
            and not screen.query("#library-media-analysis-edit-form")
        )

    await _wait_for_condition(
        pilot,
        settled,
        message=lambda: (
            "the Reader never settled: "
            f"editing_analysis={screen._media_state.editing_analysis} "
            f"saving_analysis={screen._media_state.saving_analysis} "
            f"pending={screen._media_state.reader_session.pending_request!r} "
            f"rendered={screen._media_state.composed_detail is screen._media_state.detail}"
        ),
    )
    await pilot.pause()


async def test_ctrl_s_saves_the_media_analysis_and_metadata_forms(monkeypatch):
    """Analysis ▸ Add analysis ▸ type ▸ Ctrl+S writes one analysis version and
    closes the form; More ▸ Edit metadata ▸ change the Title ▸ Ctrl+S writes
    the title. Escape's dirty-form guard is untouched in between."""
    host, app, service = _host()
    toasts = _record_toasts(monkeypatch, app)
    resolved = _record_resolved_keys(monkeypatch, host)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)

        await _open_analysis_form(screen, pilot)
        await _type(pilot, "Key points")
        await _wait_for_text(screen, pilot, "Key points")
        await _wait_for_condition(
            pilot,
            lambda: _focused_id(screen) == "library-media-analysis-edit-text",
            message=lambda: f"typing moved focus: {screen.focused!r}",
        )

        await pilot.press("ctrl+s")
        await _wait_for_condition(
            pilot,
            lambda: [call["analysis_content"] for call in service.analysis_calls]
            == ["Key points"],
            message=lambda: f"Ctrl+S never saved the analysis: {service.analysis_calls}",
        )
        await _reader_settled(screen, pilot)

        await _open_edit_form(screen, pilot)
        await pilot.press("end")
        await pilot.press("exclamation_mark")
        await _wait_for_condition(
            pilot,
            lambda: _title_value(screen) == "Interview Recording!",
            message=lambda: f"the typed character never landed: {_title_value(screen)!r}",
        )
        # Escape's TASK-34000.22 guard still answers first, unchanged.
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: bool(toasts)
            and toasts[-1] == LIBRARY_MEDIA_EDIT_DIRTY_VETO_COPY
            and screen._media_state.editing is True
            and "ctrl+s save" in _painted_footer(host, screen),
            message=lambda: (
                f"Escape's veto did not keep the form: toasts={toasts} "
                f"editing={screen._media_state.editing} "
                f"footer={_painted_footer(host, screen)!r}"
            ),
        )

        # Review R1 N-2: while another Media change holds the shared write
        # interlock (setup: the flag its claim sets), the form refuses out
        # loud; when that change's interlock is released, the line goes.
        def _edit_status() -> str:
            line = screen.query_one("#library-media-edit-status")
            return _painted(host, line.region) if line.display else ""

        screen._media_state.bulk_delete_in_flight = True
        await _press_processed(pilot, resolved, "ctrl+s")
        await _wait_for_condition(
            pilot,
            lambda: "Media change in progress." in _edit_status(),
            message=lambda: f"no refusal under the form: {_edit_status()!r}",
        )
        assert service.update_calls == [], service.update_calls
        screen._complete_library_media_mutation()
        await _wait_for_condition(
            pilot,
            lambda: _edit_status() == "",
            message=lambda: f"the refusal outlived its interlock: {_edit_status()!r}",
        )

        await pilot.press("ctrl+s")
        await _wait_for_condition(
            pilot,
            lambda: [call.get("title") for call in service.update_calls]
            == ["Interview Recording!"],
            message=lambda: f"Ctrl+S never saved the metadata: {service.update_calls}",
        )
        await _wait_for_condition(
            pilot,
            lambda: not screen._media_state.editing
            and not screen.query("#library-media-edit-title"),
            message="the metadata form stayed open after its save",
        )
        # One analysis version, one metadata write: nothing saved twice.
        assert len(service.analysis_calls) == 1
        assert len(service.update_calls) == 1


def _within(widget, pane_id: str) -> bool:
    return widget is not None and any(
        node.id == pane_id for node in widget.ancestors_with_self
    )


async def _focus_out_of(screen, pilot, form_id: str) -> None:
    """Shift+F6 (a real key) to the previous pane, out of the dirty form."""
    await pilot.press("shift+f6")
    await _wait_for_condition(
        pilot,
        lambda: screen.focused is not None and not _within(screen.focused, form_id),
        message=lambda: f"Shift+F6 left focus in the form: {screen.focused!r}",
    )


async def _chip_gone(host, screen, pilot) -> None:
    """The chip shown while the dirty form held focus has gone: a
    transition, so this absence is a positive signal, not a pause."""
    await _wait_for_condition(
        pilot,
        lambda: "ctrl+s" not in _painted_footer(host, screen),
        message=lambda: f"the footer still offers ctrl+s: {_painted_footer(host, screen)!r}",
    )


async def test_ctrl_s_never_writes_a_media_form_that_lacks_focus(monkeypatch):
    """Review I-2: each form is written only while it holds focus. A dirty
    analysis, then a dirty metadata form, with focus moved out by a real key:
    Ctrl+S writes nothing, keeps the draft and the open form, and the footer
    offers no ctrl+s. Review R1 N-1: each absence is checked only after the
    key is fully processed (its binding check completed), and the save
    seams' synchronous claims (the analysis guard, the shared Media write
    interlock) prove no save even began."""
    host, app, service = _host()
    resolved = _record_resolved_keys(monkeypatch, host)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)

        await _open_analysis_form(screen, pilot)
        await _type(pilot, "Kept draft")
        await _wait_for_text(screen, pilot, "Kept draft")
        await _wait_for_condition(
            pilot,
            lambda: "ctrl+s save" in _painted_footer(host, screen),
            message="the focused, dirty analysis form never chipped Ctrl+S",
        )
        await _focus_out_of(screen, pilot, "library-media-analysis-edit-form")
        await _press_processed(pilot, resolved, "ctrl+s")
        assert screen._media_state.saving_analysis is False, service.analysis_calls
        assert service.analysis_calls == [], service.analysis_calls
        assert screen._media_state.editing_analysis is True
        assert screen._media_state.analysis_draft == "Kept draft"
        await _chip_gone(host, screen, pilot)

        # Discard drops the analysis (setup, not subject); the Reader settles.
        screen.query_one("#library-media-analysis-cancel").press()
        await _reader_settled(screen, pilot)

        await _open_edit_form(screen, pilot)
        await pilot.press("end")
        await pilot.press("exclamation_mark")
        await _wait_for_condition(
            pilot,
            lambda: _title_value(screen) == "Interview Recording!"
            and "ctrl+s save" in _painted_footer(host, screen),
            message=lambda: f"the typed character never landed: {_title_value(screen)!r}",
        )
        await _focus_out_of(screen, pilot, "library-media-edit-form")
        await _press_processed(pilot, resolved, "ctrl+s")
        assert screen._media_state.bulk_delete_in_flight is False, service.update_calls
        assert service.update_calls == [], service.update_calls
        assert screen._media_state.editing is True
        assert (screen._media_state.edit_draft or {}).get("title") == (
            "Interview Recording!"
        ), screen._media_state.edit_draft
        await _chip_gone(host, screen, pilot)


async def test_a_double_ctrl_s_writes_one_analysis_version(monkeypatch):
    """Review I-3: the analysis Save is single-flighted. With the service
    held mid-write, a second Ctrl+S says "Already saving." under the form and
    starts no second write; released, exactly one version is written. Review
    R1 N-1: the hold is bounded and opens in ``finally``; reaching the bound
    FAILS the test."""
    host, app, service = _host()
    entered: list[str] = []
    hold_timeouts: list[str] = []
    release = threading.Event()
    real_save = service.save_analysis_version

    async def held_save(**kwargs):
        entered.append(kwargs.get("analysis_content", ""))
        if not await asyncio.to_thread(release.wait, HOLD_BOUND):
            hold_timeouts.append(entered[-1])
            raise AssertionError(f"the held save outlived its {HOLD_BOUND} s bound")
        return await real_save(**kwargs)

    service.save_analysis_version = held_save
    resolved = _record_resolved_keys(monkeypatch, host)
    async with host.run_test(size=SIZE) as pilot:
        try:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_row_1(screen, pilot)
            await _open_analysis_form(screen, pilot)
            await _type(pilot, "Once")
            await _wait_for_text(screen, pilot, "Once")

            await pilot.press("ctrl+s")
            await _wait_for_condition(
                pilot,
                lambda: entered == ["Once"],
                message=lambda: f"no save: {entered}",
            )
            assert _focused_id(screen) == "library-media-analysis-edit-text"
            await _press_processed(pilot, resolved, "ctrl+s")
            await _wait_for_condition(
                pilot,
                lambda: "Already saving."
                in _painted(
                    host, screen.query_one("#library-media-analysis-status").region
                ),
                message=lambda: (
                    f"the second press said nothing under the form (writes: {entered})"
                ),
            )
            assert entered == ["Once"], f"a second write started: {entered}"
            await _wait_for_condition(
                pilot,
                lambda: "ctrl+s" not in _painted_footer(host, screen),
                message=lambda: f"a saving form chipped: {_painted_footer(host, screen)!r}",
            )
        finally:
            release.set()
        await _wait_for_condition(
            pilot,
            lambda: not screen._media_state.editing_analysis
            and not screen._media_state.saving_analysis,
            message="the held save never settled",
        )
        assert [call["analysis_content"] for call in service.analysis_calls] == [
            "Once"
        ]
    assert not hold_timeouts, f"a held save timed out: {hold_timeouts}"
