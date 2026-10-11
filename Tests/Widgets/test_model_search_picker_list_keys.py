"""The model picker's list under real key presses, and under late refreshes.

#3057 review R3 #1, #2, #5, #6 and #7. Each test mounts the shared picker in
the flat list and, where the list is built differently, in the grouped list
Settings ▸ Providers & Models shows; keys go through ``pilot.press``.
"""

import pytest
from textual.widgets import Input

from Tests.Widgets.test_model_search_picker import (
    PickerTestApp,
    _entries,
    _provenance_option,
    _result_prompts,
    _results,
    _status_text,
    _wait_for_catalog,
)
from tldw_chatbook.UI.Screens.provider_model_resolution import ConsoleModelProvenance
from tldw_chatbook.Widgets.model_search_picker import CURRENT_MARK, ModelSearchPicker

_CATALOG = ConsoleModelProvenance.CURRENT_CATALOG
#: Twenty-five ids that hold "gpt-4", then the id that is exactly "gpt-4":
#: a filter typed in full matches all 26, and its own id is past the cap.
_PAST_CAP = [f"gpt-4-{index:02d}" for index in range(25)] + ["gpt-4"]


async def _mounted(pilot, model_ids, *, grouped: bool):
    """Load ``model_ids`` and focus the field; grouped as Settings lists them."""
    await _wait_for_catalog(pilot)
    picker = pilot.app.query_one(ModelSearchPicker)
    if grouped:
        picker.set_provenance_options(
            "OpenRouter",
            tuple(_provenance_option(model_id, _CATALOG) for model_id in model_ids),
        )
    field = pilot.app.query_one("#model-search-picker-input", Input)
    field.focus()
    await pilot.pause()
    return picker, field, _results(pilot.app)


def _highlighted_row(results) -> str:
    return str(results.get_option_at_index(results.highlighted).prompt)


@pytest.mark.asyncio
@pytest.mark.parametrize("grouped", [False, True], ids=["flat", "grouped"])
async def test_down_moves_through_the_list_once_the_list_has_focus(grouped):
    """R3 #1: type a filter, Down twice, Enter picks the SECOND match.

    The picker handled every bubbling Down, put the highlight back on the
    first match and stopped the event, so the list's own cursor never moved.
    """
    model_ids = ["gpt-a", "gpt-b", "gpt-c", "other"]
    app = PickerTestApp({"OpenRouter": []}, _entries("OpenRouter", model_ids))
    async with app.run_test() as pilot:
        _picker, _field, results = await _mounted(pilot, model_ids, grouped=grouped)

        await pilot.press(*"gpt", "down")
        await pilot.pause()
        assert app.focused is results
        assert _highlighted_row(results) == "gpt-a"

        await pilot.press("down")
        await pilot.pause()
        assert _highlighted_row(results) == "gpt-b"
        await pilot.press("down", "up")
        await pilot.pause()
        assert _highlighted_row(results) == "gpt-b"

        await pilot.press("enter")
        await pilot.pause()
        assert app.selected_models == ["gpt-b"]


@pytest.mark.asyncio
async def test_down_from_the_committed_row_reaches_the_row_below_it():
    """R3 #1 with no filter typed: the first Down lands on the committed
    model, the second leaves it."""
    model_ids = ["a/one", "b/two", "c/three"]
    app = PickerTestApp(
        {"OpenRouter": []}, _entries("OpenRouter", model_ids), current_model="b/two"
    )
    async with app.run_test() as pilot:
        _picker, _field, results = await _mounted(pilot, model_ids, grouped=False)

        await pilot.press("down")
        await pilot.pause()
        assert _highlighted_row(results) == f"b/two  {CURRENT_MARK}"
        await pilot.press("down")
        await pilot.pause()
        assert _highlighted_row(results) == "c/three"


@pytest.mark.asyncio
@pytest.mark.parametrize("grouped", [False, True], ids=["flat", "grouped"])
async def test_enter_picks_an_id_typed_in_full_that_is_past_the_result_cap(grouped):
    """R3 #2: 26 ids hold "gpt-4" and the one that IS "gpt-4" is the 26th.

    Only the first 20 matches were searched for the exact id, so Enter picked
    the first row ("gpt-4-00"). The id typed in full keeps a row, is the one
    highlighted, and is the one Enter picks.
    """
    app = PickerTestApp({"OpenRouter": []}, _entries("OpenRouter", _PAST_CAP))
    async with app.run_test() as pilot:
        picker, _field, results = await _mounted(pilot, _PAST_CAP, grouped=grouped)

        await pilot.press(*"gpt-4")
        await pilot.pause()
        assert len(picker._matches) == ModelSearchPicker.MAX_RESULTS
        assert _highlighted_row(results) == "gpt-4"
        assert _status_text(app) == (
            "Showing 20 of 26 matching models. Type to narrow the list."
        )

        await pilot.press("enter")
        await pilot.pause()
        assert app.selected_models == ["gpt-4"]


@pytest.mark.asyncio
async def test_the_committed_model_and_a_typed_id_both_keep_a_row_past_the_cap():
    """R3 #2: the committed model's rescue (C7(b)) and the typed id's do not
    take each other's slot."""
    model_ids = _PAST_CAP + ["gpt-4-committed"]
    app = PickerTestApp(
        {"OpenRouter": []},
        _entries("OpenRouter", model_ids),
        current_model="gpt-4-committed",
    )
    async with app.run_test() as pilot:
        _picker, _field, results = await _mounted(pilot, model_ids, grouped=False)

        await pilot.press(*"gpt-4")
        await pilot.pause()
        rows = _result_prompts(results)
        assert len(rows) == ModelSearchPicker.MAX_RESULTS
        assert "gpt-4" in rows
        assert f"gpt-4-committed  {CURRENT_MARK}" in rows
        assert _highlighted_row(results) == "gpt-4"
