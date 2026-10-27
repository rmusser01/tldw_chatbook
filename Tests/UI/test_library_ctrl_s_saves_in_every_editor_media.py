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

from Tests.UI.library_quit_guard_support import _record_toasts
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


def _within(widget, pane_id: str) -> bool:
    return widget is not None and any(
        node.id == pane_id for node in widget.ancestors_with_self
    )


async def _focus_out_of(screen, pilot, form_id: str) -> None:
    """Shift+F6 (a real key) to the previous pane, out of the form."""
    await pilot.press("shift+f6")
    await _wait_for_condition(
        pilot,
        lambda: screen.focused is not None and not _within(screen.focused, form_id),
        message=lambda: f"Shift+F6 left focus in the form: {screen.focused!r}",
    )
    await pilot.pause()


async def test_ctrl_s_never_writes_a_media_form_that_lacks_focus(monkeypatch):
    """Review I-2: each form is written only while it holds focus. A dirty
    analysis, then a dirty metadata form, with focus moved out by a real key:
    Ctrl+S writes nothing, keeps the draft and the open form, and the footer
    offers no ctrl+s."""
    host, app, service = _host()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)

        await _open_analysis_form(screen, pilot)
        await _type(pilot, "Kept draft")
        await _wait_for_text(screen, pilot, "Kept draft")
        await _focus_out_of(screen, pilot, "library-media-analysis-edit-form")
        await pilot.press("ctrl+s")
        await pilot.pause(0.4)
        assert service.analysis_calls == [], service.analysis_calls
        assert screen._media_state.editing_analysis is True
        assert screen._media_state.analysis_draft == "Kept draft"
        assert "ctrl+s" not in _painted_footer(host, screen)

        # Back into the form; Discard drops the analysis (setup, not subject).
        screen.query_one("#library-media-analysis-cancel").press()
        await _wait_for_condition(
            pilot,
            lambda: not screen._media_state.editing_analysis,
            message="Discard changes never closed the analysis form",
        )

        title = await _open_edit_form(screen, pilot)
        await pilot.press("end")
        await pilot.press("exclamation_mark")
        await _wait_for_condition(
            pilot,
            lambda: title.value == "Interview Recording!",
            message="the typed character never landed",
        )
        await _focus_out_of(screen, pilot, "library-media-edit-form")
        await pilot.press("ctrl+s")
        await pilot.pause(0.4)
        assert service.update_calls == [], service.update_calls
        assert screen._media_state.editing is True
        assert (screen._media_state.edit_draft or {}).get("title") == (
            "Interview Recording!"
        ), screen._media_state.edit_draft
        assert "ctrl+s" not in _painted_footer(host, screen)


async def test_a_double_ctrl_s_writes_one_analysis_version(monkeypatch):
    """Review I-3: the analysis Save is single-flighted. With the service
    held mid-write, a second Ctrl+S says "Already saving." under the form and
    starts no second write; released, exactly one version is written."""
    host, app, service = _host()
    entered: list[str] = []
    release = threading.Event()
    real_save = service.save_analysis_version

    async def held_save(**kwargs):
        entered.append(kwargs.get("analysis_content", ""))
        await asyncio.to_thread(release.wait, 15)
        return await real_save(**kwargs)

    service.save_analysis_version = held_save
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        area = await _open_analysis_form(screen, pilot)
        await _type(pilot, "Once")
        await _wait_for_text(screen, pilot, "Once")

        await pilot.press("ctrl+s")
        await _wait_for_condition(
            pilot, lambda: entered == ["Once"], message=lambda: f"no save: {entered}"
        )
        assert screen.focused is area
        await pilot.press("ctrl+s")
        await pilot.pause(0.5)
        assert entered == ["Once"], f"a second write started: {entered}"
        await _wait_for_condition(
            pilot,
            lambda: "Already saving."
            in _painted(host, screen.query_one("#library-media-analysis-status").region),
            message="the second press said nothing under the form",
        )
        assert "ctrl+s" not in _painted_footer(host, screen)

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

