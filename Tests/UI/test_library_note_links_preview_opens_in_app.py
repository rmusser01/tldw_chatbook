"""TASK-34000.11: Preview links stay in the app (the census sibling of
``test_library_note_links_picker_and_preview.py``, split out in fix round 1
so each lane file stays under its 35 s ceiling).

One boot at 160x45 over a real ChaChaNotes database: a click on a painted
note link opens that note; an http(s) link goes to ``webbrowser.open`` in a
thread worker exactly once with a notice; a link whose target was deleted
is refused with a notice; ``app.open_url`` is never called (AC#5: asserted
as ``== []`` after each click, with the in-app open as the positive signal).
The hostile-scheme pins are in the ``_extended.py`` sibling.
"""

from __future__ import annotations

import pytest

from Tests.UI.library_note_links_support import (
    DELETED_NOTICE,
    INDEX,
    LISBON,
    STALE,
    WEB,
    WEB_URL,
    _app,
    _armed_editor,
    _click_preview_link,
    _open_note,
    _preview,
    _record_browser,
    _record_open_url,
    _title_value,
    _until,
)
from Tests.UI.library_quit_guard_support import SIZE, _library, _record_toasts
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_clicking_a_note_link_in_preview_opens_that_note_and_never_reaches_open_url(
    tmp_path, monkeypatch
) -> None:
    """AC#3 / AC#5: a click on the painted link text opens Lisbon in the
    editor and ``app.open_url`` is never called. Negative control in the
    same boot: an http(s) link goes to ``webbrowser.open`` exactly once,
    with a notice, and still never to ``app.open_url``. Then a link whose
    target was deleted is refused with the notice (Review Focus 4)."""
    app, profile = _app(tmp_path)
    opened = _record_open_url(monkeypatch, app)
    browser = _record_browser(monkeypatch)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        toasts = _record_toasts(monkeypatch, app)
        await _open_note(screen, pilot, INDEX, profile.index_id)
        await _preview(screen, pilot, LISBON, app)

        await _click_preview_link(app, screen, pilot, LISBON)

        def _lisbon_armed() -> bool:
            snapshot = screen._library_note_session.snapshot
            return (
                snapshot is not None
                and snapshot.note_id == profile.lisbon_id
                and screen._notes_state.editor_armed
            )

        # Either the app opened Lisbon, or the OS handler was asked: the
        # base fails HERE, on the recorder, not on a bare timeout.
        await _until(pilot, lambda: bool(opened) or _lisbon_armed(), "the click to land")
        assert opened == [], f"note:// reached app.open_url: {opened}"
        await _armed_editor(screen, pilot, profile.lisbon_id)
        assert _title_value(screen) == LISBON
        assert browser == []

        await _open_note(screen, pilot, WEB, profile.web_id)
        await _preview(screen, pilot, "docs", app)
        await _click_preview_link(app, screen, pilot, "docs")
        await _until(pilot, lambda: browser == [WEB_URL], f"webbrowser.open({WEB_URL!r})")
        await _until(
            pilot,
            lambda: any(t == f"Opened in browser: {WEB_URL}" for t in toasts),
            "the browser notice",
        )
        await pilot.pause(0.3)
        assert browser == [WEB_URL]
        assert opened == [], opened
        assert screen._library_note_session.snapshot.note_id == profile.web_id

        # Arm C (Review Focus 4): a stale ``note://`` link is refused with
        # the notice; the editor does not move and the OS handler is never
        # asked. Same boot: one lane boot costs ~12 s here.
        await _open_note(screen, pilot, STALE, profile.stale_id)
        await _preview(screen, pilot, "Gone", app)
        await _click_preview_link(app, screen, pilot, "Gone")
        await _until(
            pilot,
            lambda: bool(opened) or DELETED_NOTICE in toasts,
            "the click to land",
        )
        assert opened == [], f"note:// reached app.open_url: {opened}"
        assert DELETED_NOTICE in toasts, toasts
        await pilot.pause(0.3)
        assert screen._library_note_session.snapshot.note_id == profile.stale_id
        assert screen.query_one("#library-note-preview-region").display
        assert browser == [WEB_URL]
