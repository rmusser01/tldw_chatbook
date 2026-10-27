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

import pytest
from Tests.UI.library_quit_guard_support import _record_toasts
from Tests.UI.test_library_media_editor_dirty_escape import (
    _host,
    _open_analysis_form,
    _open_edit_form,
    _open_row_1,
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


async def test_ctrl_s_saves_the_media_analysis_and_metadata_forms(monkeypatch):
    """Analysis ▸ Add analysis ▸ type ▸ Ctrl+S writes one analysis version and
    closes the form; More ▸ Edit metadata ▸ change the Title ▸ Ctrl+S writes
    the title. Escape's dirty-form guard is untouched in between."""
    host, app, service = _host()
    toasts = _record_toasts(monkeypatch, app)
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)

        area = await _open_analysis_form(screen, pilot)
        await _type(pilot, "Key points")
        await _wait_for_text(screen, pilot, "Key points")
        await pilot.pause()
        assert screen.focused is area

        await pilot.press("ctrl+s")
        await _wait_for_condition(
            pilot,
            lambda: [call["analysis_content"] for call in service.analysis_calls]
            == ["Key points"],
            message=lambda: f"Ctrl+S never saved the analysis: {service.analysis_calls}",
        )
        await _wait_for_condition(
            pilot,
            lambda: not screen._media_state.editing_analysis,
            message="the analysis form stayed open after its save",
        )

        title = await _open_edit_form(screen, pilot)
        await pilot.press("end")
        await pilot.press("exclamation_mark")
        await _wait_for_condition(
            pilot,
            lambda: title.value == "Interview Recording!",
            message="the typed character never landed",
        )
        # Escape's TASK-34000.22 guard still answers first, unchanged.
        await pilot.press("escape")
        await pilot.pause()
        assert screen._media_state.editing is True
        assert toasts and toasts[-1] == LIBRARY_MEDIA_EDIT_DIRTY_VETO_COPY, toasts
        assert "ctrl+s save" in _painted_footer(host, screen)

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
