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
from textual.widgets import DataTable, Select

from tldw_chatbook.TTS.audiobook_generator import Character
from tldw_chatbook.Widgets.TTS import (
    character_voice_widget as character_voice_widget_module,
)
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


# ---------------------------------------------------------------------------
# task-15772 review follow-up: `_update_voice_options`'s dynamic voice list
# and `#voice-style-select` were also composed backwards -- (id, label)
# instead of Textual's real (label, value). `character-voice-select` and
# `bulk-voice-select` start `compose()`d empty (`options=[]`), so a
# `grep -n "Select("` sweep alone misses them; the backwards shape lives in
# `_update_voice_options`'s `self.voice_options` list, fed straight into
# `voice_select.set_options(...)`/`bulk_select.set_options(...)`.
#
# Live-reproduced pre-fix (review15772.md): mounting the widget and setting
# `voice_select.value = "alloy"` raised
# `InvalidSelectValueError: Illegal select value 'alloy'.` -- the same
# crash-swallowed-into-`logger.debug` pattern `_initialize_audiobook_defaults`
# had before task-15772's first pass. `_update_assignment_ui` (fires via
# `watch_selected_character_index`, i.e. every character-row click) hits this
# exact path and swallows it into `logger.debug("Some UI elements not ready:
# ...")`, so the per-character voice dropdown silently never reflected the
# assigned voice.
# ---------------------------------------------------------------------------


def _options_as_pairs(select: Select) -> list[tuple[str, object]]:
    return [
        (str(label), value)
        for label, value in select._options
        if value is not Select.NULL
    ]


@pytest.mark.asyncio
async def test_character_voice_select_composes_label_value_order() -> None:
    app = _CharacterVoiceWidgetTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        select = pilot.app.query_one("#character-voice-select", Select)
        pairs = _options_as_pairs(select)
        assert pairs[0] == ("Use Narrator Voice", "narrator")
        assert ("Alloy", "alloy") in pairs
        assert ("Ash", "ash") in pairs

        # The real ids must be legal Select values -- fails with
        # `InvalidSelectValueError` against the pre-fix (id, label) shape.
        select.value = "alloy"
        assert select.value == "alloy"


@pytest.mark.asyncio
async def test_bulk_voice_select_shares_the_same_fixed_option_order() -> None:
    app = _CharacterVoiceWidgetTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        bulk = pilot.app.query_one("#bulk-voice-select", Select)
        assert _options_as_pairs(bulk)[0] == ("Use Narrator Voice", "narrator")

        bulk.value = "narrator"
        assert bulk.value == "narrator"


@pytest.mark.asyncio
async def test_selecting_a_character_row_lands_the_assigned_voice_without_swallowing_an_exception() -> (
    None
):
    """Regression for `_update_assignment_ui`'s crash-swallowed pattern.

    Before the fix, `voice_select.value = assigned_voice` (`assigned_voice`
    being a real id like "ash") raised `InvalidSelectValueError` -- caught
    by `except Exception as e: logger.debug("Some UI elements not ready:
    {e}")` -- so the dropdown silently kept whatever value it had instead of
    showing the character's actual assigned voice.
    """
    app = _CharacterVoiceWidgetTestApp()
    logs: list[str] = []
    sink_id = character_voice_widget_module.logger.add(lambda m: logs.append(str(m)))
    try:
        async with app.run_test() as pilot:
            widget = _widget(pilot)
            widget.characters = []  # isolate from the shared-default reactive
            widget.characters = [Character(name="Aria", voice="narrator")]
            await pilot.pause()
            widget.voice_assignments = {"Aria": "ash"}

            widget.selected_character_index = 0
            await pilot.pause()

            voice_select = pilot.app.query_one("#character-voice-select", Select)
            assert voice_select.value == "ash"
            assert "Some UI elements not ready" not in "".join(logs)
    finally:
        character_voice_widget_module.logger.remove(sink_id)


@pytest.mark.asyncio
async def test_get_voice_label_resolves_id_to_label() -> None:
    """`_get_voice_label` unpacks `self.voice_options` as `(label, id)` now
    -- must be updated in lockstep with the list's own order, not just the
    `Select.set_options()` call site."""
    app = _CharacterVoiceWidgetTestApp()
    async with app.run_test() as pilot:
        widget = _widget(pilot)
        await pilot.pause()
        assert widget._get_voice_label("alloy") == "Alloy"
        assert widget._get_voice_label("narrator") == "📖 Narrator"


@pytest.mark.asyncio
async def test_voice_style_select_composes_label_value_order() -> None:
    app = _CharacterVoiceWidgetTestApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        select = pilot.app.query_one("#voice-style-select", Select)
        assert _options_as_pairs(select) == [
            ("Neutral", "neutral"),
            ("Happy", "happy"),
            ("Sad", "sad"),
            ("Angry", "angry"),
            ("Excited", "excited"),
            ("Calm", "calm"),
        ]


@pytest.mark.asyncio
async def test_selecting_a_style_option_stores_the_id_not_the_label() -> None:
    """`_get_current_voice_settings`'s `settings["style"] = style_select.value`
    must store the id ("happy"), not the capitalized display label
    ("Happy") the backwards compose order used to leak into it."""
    app = _CharacterVoiceWidgetTestApp()
    async with app.run_test() as pilot:
        widget = _widget(pilot)
        await pilot.pause()
        style_select = pilot.app.query_one("#voice-style-select", Select)

        # Fails with `InvalidSelectValueError` against the pre-fix shape,
        # since the only legal values were the capitalized labels.
        style_select.value = "happy"

        settings = widget._get_current_voice_settings()
        assert settings["style"] == "happy"
