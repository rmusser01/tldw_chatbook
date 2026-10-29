"""TASK-34000.11: the ``[[`` note-link picker, and Preview links that stay in
the app.

Review finding N-08 (notes-library-ux-review-2026-10-02): there was no link
control at all, a typed ``[[Title]]`` rendered as plain text and recorded
nothing, and activating a stored note link in Preview handed
``note://<id>`` to the operating system's URL handler.

One boot at 160x45 over a real ChaChaNotes database (see
``library_note_links_support``): the Escape-latch arm and the final fix
round's navigation-keys arm (C-1: Home/Right/Down/Enter/Tab over a stored
link and over a bare ``[[Title]]`` never open the picker and never change
the link) share the insert boot. The Preview arms (click, http, deleted
target) are the sibling census file
``test_library_note_links_preview_opens_in_app.py`` -- two boots in one
file sat at the lane's 35 s ceiling (fix round 1, I-4). The picker arms read what the compositor
paints inside ``#library-note-link-picker`` and what the body ``TextArea``
holds after Enter. The many-links Enter picker, the 80x24 size, the ``[b]``
title, the burst races and the hostile-scheme pins are in
``test_library_note_links_picker_and_preview_extended.py``; the pure link
policy table is ``Tests/Notes/test_note_link_policy.py`` (FAST_LANE).
"""

from __future__ import annotations

import pytest
from textual.widgets import OptionList, TextArea

from tldw_chatbook.Widgets.Library.library_note_link_picker import LibraryNoteLinkPicker

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
NAVIGATION = "Navigation probe"


def _link_edges(db, note_id: str) -> list[str]:
    """The ``note_links`` targets recorded for ``note_id``, sorted."""
    rows = db.get_connection().execute(
        "SELECT target_note_id FROM note_links WHERE source_note_id = ? ORDER BY 1",
        (note_id,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _count_picker_opens(monkeypatch) -> dict[str, int]:
    """Count every ``show_candidates`` -- the one way the picker ever opens
    (rows or a hint row), so "never opened" is a counted claim."""
    opens = {"n": 0}
    real = LibraryNoteLinkPicker.show_candidates

    def _counted(self, rows, **kwargs):
        opens["n"] += 1
        return real(self, rows, **kwargs)

    monkeypatch.setattr(LibraryNoteLinkPicker, "show_candidates", _counted)
    return opens


async def _navigation_keys_leave_the_line_alone(
    pilot, screen, body: TextArea, opens: dict[str, int], *, row: int, label: str
) -> None:
    """Probe 2's keys over the link on ``row``: Home, Right x10 (the caret
    inside the title), Down, Enter, then Tab. The picker never opens; Enter
    is the editor's own newline at the caret (undone with Backspace so the
    line is byte-identical again); Tab leaves the body."""
    before = body.text
    lines = before.split("\n")
    assert len(lines) == 2 and row in (0, 1)
    await pilot.click("#library-note-body")
    await _until(pilot, lambda: screen.focused is body, f"the body to take focus ({label})")
    # Ctrl+Home / Ctrl+End reach the row by document position: the stored
    # line is long enough to soft-wrap, so Down from row 0 would land in
    # its wrapped continuation, not on row 1.
    await pilot.press("ctrl+home" if row == 0 else "ctrl+end")
    await pilot.press("home")
    for _ in range(10):
        await pilot.press("right")
    await _until(pilot, lambda: body.cursor_location == (row, 10), f"the caret inside the title ({label})")
    # The base opened the picker from this caret move alone; give its load
    # the time it needs so the counter below is a real negative.
    await pilot.pause(0.3)
    assert opens["n"] == 0, f"the picker opened on a caret move ({label})"
    assert not _picker_open(screen), label
    assert body.text == before, label
    await pilot.press("down")
    await pilot.pause(0.3)
    assert opens["n"] == 0, f"the picker opened on Down ({label})"
    assert body.text == before, label
    caret_row, caret_column = body.cursor_location
    split = lines[caret_row]
    expected_lines = [*lines[:caret_row], split[:caret_column], split[caret_column:], *lines[caret_row + 1 :]]
    await pilot.press("enter")
    await _until(
        pilot,
        lambda: body.text == "\n".join(expected_lines),
        f"Enter to be the editor's own newline ({label}); body reads {body.text!r}",
    )
    assert body.text.count("note://") == before.count("note://"), label
    assert opens["n"] == 0, f"the picker opened on Enter ({label})"
    await pilot.press("backspace")
    await _until(pilot, lambda: body.text == before, f"Backspace to restore the line ({label})")
    await pilot.press("tab")
    await _until(pilot, lambda: screen.focused is not body, f"Tab to leave the body ({label})")
    assert opens["n"] == 0, f"the picker opened on Tab ({label})"
    assert not _picker_open(screen), label
    assert body.text == before, f"navigation keys changed the body ({label})"


async def _blank_note_with_body_focused(screen, pilot) -> TextArea:
    """New ▸ Blank, the title typed, the body clicked (the user's gesture)."""
    await _new_blank_note(screen, pilot)
    await pilot.click("#library-note-title")
    await _until(
        pilot,
        lambda: screen.focused is screen.query_one("#library-note-title"),
        "the title to take focus",
    )
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
    form; navigation keys over an existing link never open the picker and
    never change the link (final fix round, C-1); and once autosaved the
    target's Info counts the new notes under Linked from (four, with the
    two stored ones and the navigation probe)."""
    app, profile = _app(tmp_path)
    opened = _record_open_url(monkeypatch, app)
    # The quit-guard support's scaled autosave pair (1 s / 2 s): the arm
    # waits for the save, not for the production 2 s debounce.
    _scaled_autosave(monkeypatch)
    # Arm C's corpus (C-1): a stored link and a bare ``[[Title]]`` to Lisbon
    # on two lines, and a note titled "Trip" -- the reviewer's probe 2: on
    # the base, Home/Right x10 opened the picker on the rows [Lisbon, Trip],
    # Down chose "Trip" and Enter rewrote the Lisbon link to the WRONG note
    # (autosave then swapped the edge).
    navigation_body = (
        f"See [[{LISBON}]](note://{profile.lisbon_id}) today.\nSee [[{LISBON}]] today."
    )
    navigation_id = profile.db.add_note(NAVIGATION, navigation_body)
    assert navigation_id and profile.db.add_note("Trip", "the wrong note")
    opens = _count_picker_opens(monkeypatch)
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

        # Arm C (final fix round, C-1): probe 2's keys over the stored link
        # (line 0) and over the bare ``[[Title]]`` (line 1). The picker
        # never opens, the body is byte-identical, and the recorded edges
        # are the same after the save.
        body = await _open_note(screen, pilot, NAVIGATION, navigation_id)
        edges_before = _link_edges(profile.db, navigation_id)
        assert edges_before == [profile.lisbon_id], edges_before
        opens["n"] = 0
        await _navigation_keys_leave_the_line_alone(
            pilot, screen, body, opens, row=0, label="stored link"
        )
        await _navigation_keys_leave_the_line_alone(
            pilot, screen, body, opens, row=1, label="bare [[Title]]"
        )
        assert body.text == navigation_body

        def _navigation_saved() -> bool:
            snapshot = screen._library_note_session.snapshot
            return snapshot is not None and snapshot.note_id == navigation_id and not snapshot.dirty

        await _until(pilot, _navigation_saved, "the navigation probe to be saved", timeout=25.0)
        assert profile.db.get_note_by_id(navigation_id)["content"] == navigation_body
        assert _link_edges(profile.db, navigation_id) == edges_before
        assert opens["n"] == 0

        # Index and "Two links" already link to Lisbon through the stored
        # form; the typed-and-picked link and the navigation probe make four.
        await _open_note(screen, pilot, LISBON, profile.lisbon_id)
        rows = await _info_backlinks(screen, pilot, "Linked from (4)")
        assert sorted(rows) == sorted([INDEX, NAVIGATION, TWO, VERIFIER]), rows
        assert opened == []
