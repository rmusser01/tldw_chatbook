"""TASK-34000.11: the ``[[`` note-link picker, and Preview links that stay in
the app.

Review finding N-08 (notes-library-ux-review-2026-10-02): there was no link
control at all, a typed ``[[Title]]`` rendered as plain text and recorded
nothing, and activating a stored note link in Preview handed
``note://<id>`` to the operating system's URL handler.

One boot at 160x45 over a real ChaChaNotes database (see
``library_note_links_support``): the Escape-latch arm shares the insert
boot. The Preview arms (click, http, deleted target) are the sibling census
file ``test_library_note_links_preview_opens_in_app.py`` -- two boots in one
file sat at the lane's 35 s ceiling (fix round 1, I-4). The picker arms read what the compositor
paints inside ``#library-note-link-picker`` and what the body ``TextArea``
holds after Enter. The many-links Enter picker, the 80x24 size, the ``[b]``
title, the burst races and the hostile-scheme pins are in
``test_library_note_links_picker_and_preview_extended.py``.
"""

from __future__ import annotations

import pytest
from textual.widgets import OptionList, TextArea

from Tests.UI.library_note_links_support import (
    INDEX,
    LISBON,
    TWO,
    _app,
    _info_backlinks,
    _open_note,
    _painted,
    _picker_open,
    _record_open_url,
    _title_value,
    _until,
    _until_painted,
)
from Tests.UI.library_quit_guard_support import (
    SIZE,
    _library,
    _new_blank_note,
    _scaled_autosave,
    _type,
)
from Tests.UI.test_library_notes_files_sync_journey import (
    _isolate_actor_pack_recovery,  # noqa: F401 - autouse fixture, re-exported
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

VERIFIER = "Verifier link test"


async def _blank_note_with_body_focused(screen, pilot) -> TextArea:
    """New ▸ Blank, the title typed, the body clicked (the user's gesture)."""
    await _new_blank_note(screen, pilot)
    await _type(pilot, VERIFIER)
    await _until(pilot, lambda: _title_value(screen) == VERIFIER, "the title typed")
    body = screen.query_one("#library-note-body", TextArea)
    await pilot.click("#library-note-body")
    await _until(pilot, lambda: screen.focused is body, "the body to take focus")
    return body


async def test_typing_two_brackets_opens_a_keyboard_picker_that_inserts_a_real_link(
    tmp_path, monkeypatch
) -> None:
    """AC#1 / AC#5: Escape leaves the typed text and latches; ``[[`` lists
    recent titles, ``Lisb`` narrows to one, Enter inserts the stored link
    form, and once autosaved the target's Info counts the new note under
    Linked from (three, with the two stored ones)."""
    app, profile = _app(tmp_path)
    opened = _record_open_url(monkeypatch, app)
    # The quit-guard support's scaled autosave pair (1 s / 2 s): the arm
    # waits for the save, not for the production 2 s debounce.
    _scaled_autosave(monkeypatch)
    async with app.run_test(size=SIZE) as pilot:
        screen = await _library(app, pilot)
        body = await _blank_note_with_body_focused(screen, pilot)
        note_id = screen._library_note_session.snapshot.note_id

        # Arm A (AC#1, Escape): the picker is OPEN before Escape; Escape
        # leaves ``[[Tri`` exactly as typed and does NOT leave the editor;
        # typing on keeps it closed (the latch); a fresh ``[[`` reopens it.
        # Same boot as the insert below: one lane boot costs ~12 s here.
        await _type(pilot, "[[Tri")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        assert _picker_open(screen)
        await pilot.press("escape")
        await _until(pilot, lambda: not _picker_open(screen), "Escape to close the picker")
        assert body.text == "[[Tri"
        snapshot = screen._library_note_session.snapshot
        assert snapshot is not None and snapshot.note_id == note_id
        assert screen.focused is body, "Escape must not run the editor's Back"
        await _type(pilot, "p")
        await _until(pilot, lambda: body.text == "[[Trip", "the p to land")
        await pilot.pause(0.3)
        assert not _picker_open(screen), "the latch must hold while the context is unchanged"
        for _ in range(6):
            await pilot.press("backspace")
        await _until(pilot, lambda: body.text == "", "the text cleared")

        # Arm B (AC#1, insert): ``[[`` lists recent titles, ``Lisb`` narrows
        # to one, Enter inserts the stored form.
        await _type(pilot, "See [[")
        await _until_painted(app, screen, pilot, "#library-note-link-picker", LISBON)
        painted = _painted(app, screen, "#library-note-link-picker")
        assert INDEX in painted, painted
        assert _picker_open(screen)

        # "Lisb", not "Lis": the quit-guard profile's "Reading list" also
        # starts a word with "lis" (Task 3's last-word prefix), and the AC
        # wants exactly one row.
        await _type(pilot, "Lisb")
        options = screen.query_one("#library-note-link-picker-options", OptionList)

        def _one_lisbon_row() -> bool:
            painted = _painted(app, screen, "#library-note-link-picker")
            return (
                _picker_open(screen)
                and options.option_count == 1
                and LISBON in painted
                and INDEX not in painted
            )

        await _until(pilot, _one_lisbon_row, "the picker to narrow to the one Lisbon row")

        await pilot.press("enter")
        expected = f"See [[{LISBON}]](note://{profile.lisbon_id})"
        await _until(pilot, lambda: body.text == expected, f"the body to read {expected!r}")
        assert not _picker_open(screen)
        assert body.cursor_location == (0, len(expected))
        assert screen.focused is body

        def _saved() -> bool:
            row = profile.db.get_note_by_id(note_id)
            snapshot = screen._library_note_session.snapshot
            return (
                row is not None
                and row["content"] == expected
                and snapshot is not None
                and not snapshot.dirty
            )

        await _until(pilot, _saved, "autosave to store the linked body", timeout=25.0)

        # Index and "Two links" already link to Lisbon through the stored
        # form; the typed-and-picked link makes three.
        await _open_note(screen, pilot, LISBON, profile.lisbon_id)
        rows = await _info_backlinks(screen, pilot, "Linked from (3)")
        assert sorted(rows) == sorted([INDEX, TWO, VERIFIER]), rows
        assert opened == []
