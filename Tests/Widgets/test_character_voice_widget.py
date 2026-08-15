# test_character_voice_widget.py
# Description: Regression coverage for CharacterVoiceWidget's add/remove table refresh (task-15479)
"""
task-15479: ``_add_character_manually`` and ``_remove_selected_character`` used
``self.characters = self.characters`` to force the ``characters`` reactive
(``reactive([], recompose=True)``) to re-run its watcher and refresh the
character ``DataTable``. Assigning the same list object back to itself is a
no-op as far as Textual's reactive system is concerned -- it compares equal
to the previous value (same identity) and the reactive has no
``always_update=True``, so ``watch_characters`` never fires and the table
never reflects the add/remove that already happened on the underlying list.

These tests exercise the two button-handler code paths directly and assert
on the mounted ``DataTable``'s actual row content, so they are red against
the pre-fix code (table stays stale) and green once the refresh is forced
via ``mutate_reactive``.

Each test resets ``widget.characters`` to a fresh ``[]`` before mutating it.
This is *not* incidental: ``characters = reactive([])``'s default is a bare
list literal, which Textual's ``Reactive._initialize_reactive`` shares as
the *same object* across every ``CharacterVoiceWidget`` instance that never
explicitly reassigns it (the classic mutable-default-argument trap, applied
to a reactive default) -- a pre-existing bug independent of task-15479,
discovered here because it made these tests order-dependent within one
pytest session (an earlier test's ``.append()`` onto the shared default
leaked into a later test's row count). Assigning a fresh list breaks the
aliasing for that instance going forward. Out of scope for task-15479's
self-assignment-trigger fix; worth filing separately.

Update (task-15771): filed and fixed — ``characters`` (and every other
mutable reactive default in the package) is now a callable default
(``reactive(list)``), so each instance gets its own list and the defensive
``widget.characters = []`` lines below are redundant-but-harmless. They are
kept so these tests stay valid against trees that predate the fix; the
cross-instance aliasing itself is pinned by
``test_reactive_default_aliasing.py`` and the AST inventory guard in
``Tests/Architecture/test_reactive_mutable_default_inventory.py``.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

from tldw_chatbook.TTS.audiobook_generator import Character
from tldw_chatbook.Widgets.TTS.character_voice_widget import CharacterVoiceWidget


class _CharacterVoiceWidgetTestApp(App[None]):
    """Minimal host app mounting a single CharacterVoiceWidget."""

    def compose(self) -> ComposeResult:
        yield CharacterVoiceWidget(provider="openai", id="character-voice-widget")


def _table(pilot) -> DataTable:
    return pilot.app.query_one("#character-table", DataTable)


def _widget(pilot) -> CharacterVoiceWidget:
    return pilot.app.query_one("#character-voice-widget", CharacterVoiceWidget)


@pytest.mark.asyncio
async def test_add_character_manually_refreshes_table() -> None:
    """Adding a character via the button handler must show up in the table.

    Regression for the dead self-assignment at (pre-fix) line 455: before the
    fix, ``self.characters`` gains the new ``Character`` object but
    ``watch_characters`` never runs, so ``table.row_count`` stays 0.

    The table is re-queried fresh after each mutation rather than cached in
    a local variable: a stale reference can keep reporting whatever row
    count it had before being replaced/removed, which would let this test
    pass for the wrong reason instead of reflecting what is actually live
    in the DOM.
    """
    app = _CharacterVoiceWidgetTestApp()
    async with app.run_test() as pilot:
        widget = _widget(pilot)
        widget.characters = []  # isolate from the shared-default reactive (see module docstring)
        await pilot.pause()
        assert _table(pilot).row_count == 0

        widget._add_character_manually()
        await pilot.pause()

        table = _table(pilot)
        assert table.row_count == 1
        assert len(widget.characters) == 1
        added_name = widget.characters[0].name
        row = table.get_row_at(0)
        assert row[0] == added_name


@pytest.mark.asyncio
async def test_add_character_manually_twice_refreshes_table_each_time() -> None:
    """A second add must also be reflected -- guards against a fix that only
    happens to work once (e.g. relying on incidental identity changes)."""
    app = _CharacterVoiceWidgetTestApp()
    async with app.run_test() as pilot:
        widget = _widget(pilot)
        widget.characters = []  # isolate from the shared-default reactive (see module docstring)
        await pilot.pause()

        widget._add_character_manually()
        await pilot.pause()
        widget._add_character_manually()
        await pilot.pause()

        assert _table(pilot).row_count == 2
        assert len(widget.characters) == 2


@pytest.mark.asyncio
async def test_remove_selected_character_refreshes_table() -> None:
    """Removing the selected character via the button handler must clear its
    row from the table.

    Regression for the dead self-assignment at (pre-fix) line 465: before the
    fix, ``self.characters`` loses the popped ``Character`` object but
    ``watch_characters`` never runs, so the table keeps showing the stale row.
    """
    app = _CharacterVoiceWidgetTestApp()
    async with app.run_test() as pilot:
        widget = _widget(pilot)

        # Seed via a fresh-list assignment (a real reactive change, not the
        # buggy self-assignment) so this test isolates the remove path only.
        widget.characters = [Character(name="Seeded", voice="narrator")]
        await pilot.pause()
        assert _table(pilot).row_count == 1

        widget.selected_character_index = 0
        await pilot.pause()

        widget._remove_selected_character()
        await pilot.pause()

        assert _table(pilot).row_count == 0
        assert widget.characters == []
