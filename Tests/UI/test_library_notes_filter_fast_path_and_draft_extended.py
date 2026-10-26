"""Two more arms of ``test_library_notes_filter_fast_path_and_draft.py``,
outside the PR lane's census so that file stays inside its 30 s budget; the
full ``Tests/UI`` run covers them.

- TASK-34000.28 (N-21) fix round 1, M6: following ``‹ <origin>`` back to an
  origin deleted meanwhile warns, returns to the list and writes nothing.
- TASK-34000.40 (N-31) ruling T5-b: a sibling list filter (Prompts) takes the
  same "no after esc: keys when Escape lands in another text field" rule
  from the shared footer transform.

Same harness as the census file: the real ``LibraryScreen`` with the
production stylesheet over a real ChaChaNotes database.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.library_quit_guard_support import SIZE, _armed_editor, _until
from Tests.UI.test_library_notes_filter_fast_path_and_draft import (
    INDEX,
    RECIPES,
    _focused_id,
    _footer_withholds_after_esc,
    _mounted,
    _notes_ready,
    _painted_in,
    _painted_of,
    _row_titled,
    _shown,
    _slash_to_filter,
    _wait,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


def _notes_rows(db) -> list[tuple]:
    return sorted(
        (row["id"], row["version"], row["deleted"])
        for row in db.execute_query("SELECT id, version, deleted FROM notes").fetchall()
    )


async def test_the_way_back_to_a_deleted_origin_warns_and_writes_nothing(
    tmp_path, monkeypatch
) -> None:
    """M6: Index -> its note link -> Recipes shows ``‹ Index``; Index is then
    deleted elsewhere. Pressing ``‹`` warns, returns to the list, writes no
    notes row, and leaves focus on a live control a real key moves on from."""
    host, seeded = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        _row_titled(screen, INDEX).press()
        await _armed_editor(screen, pilot, seeded.index_id)
        screen.query_one("#library-note-preview", Button).press()
        await _wait(
            pilot,
            lambda: _shown(screen, "#library-note-preview-region")
            and RECIPES in _painted_of(host, screen, "#library-note-preview-body"),
            "Preview to paint the link",
        )
        await pilot.click("#library-note-preview-region")
        await _wait(
            pilot,
            lambda: _focused_id(screen) == "library-note-preview-region",
            "the Preview region to hold focus",
        )
        await pilot.press("enter")
        await _armed_editor(screen, pilot, seeded.recipes_id)
        await _wait(
            pilot,
            lambda: bool(screen.query("#library-note-link-back"))
            and f"‹ {INDEX}" in _painted_in(host, screen.query("#library-note-link-back").first()),
            "'‹ Index — start here' on Recipes",
        )

        toasts: list[str] = []
        app = host.app_instance
        real_notify = app.notify

        def _recording_notify(message, *args, **kwargs):
            toasts.append(str(message))
            return real_notify(message, *args, **kwargs)

        monkeypatch.setattr(app, "notify", _recording_notify)
        db = seeded.profile.db
        index = db.get_note_by_id(seeded.index_id)
        assert db.soft_delete_note(seeded.index_id, index["version"]) is True
        before = _notes_rows(db)

        await pilot.click("#library-note-link-back")
        await _wait(
            pilot,
            lambda: "That note is no longer available." in toasts
            and screen._notes_state.view == "list",
            lambda: f"the missing-origin warning and the list (toasts {toasts!r})",
        )
        await pilot.pause(0.3)
        assert _notes_rows(db) == before, "following the way back wrote to the notes table"
        assert screen.focused is not None and screen.focused.is_attached

        # A real key still works: "/" reaches the filter.
        await _slash_to_filter(screen, pilot)


async def test_a_sibling_list_filter_withholds_after_esc_keys_too(tmp_path) -> None:
    """Ruling T5-b through the shared transform: Escape from the Prompts filter
    lands in the rail's search box as well, so its typing footer names no
    "after esc:" keys (it read ``after esc: / focus search`` before)."""
    host, _seeded = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        screen.query_one("#library-row-browse-prompts", Button).press()
        await _wait(
            pilot,
            lambda: _shown(screen, "#library-prompts-filter"),
            "the Prompts filter",
        )
        # The Prompts canvas recomposes once its (empty) browse answers; a
        # click landing on the box that recompose replaces is lost (the
        # TASK-34000.85 shape), so click again until the box holds focus.
        for _ in range(5):
            await pilot.click("#library-prompts-filter")
            try:
                await _until(
                    pilot,
                    lambda: _focused_id(screen) == "library-prompts-filter",
                    "",
                    timeout=1.0,
                )
                break
            except AssertionError:
                continue
        assert _focused_id(screen) == "library-prompts-filter", _focused_id(screen)
        await _footer_withholds_after_esc(host, screen, pilot, what="Prompts filter")
