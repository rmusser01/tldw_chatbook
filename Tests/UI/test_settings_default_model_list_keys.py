"""Settings ▸ Providers & Models ▸ Model: the list under Down and Enter.

#3057 review R3 #1 and #2, on the mounted Settings screen with the real
stylesheet at 211x44 and real key presses: Down walks the open list, and an
id typed in full is the one Enter picks when over 20 ids match it.
"""

from __future__ import annotations

import re

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_default_model_picker import (
    _SIZE,
    PROVIDERS_MODELS,
    _app,
    _open_providers,
    _rows,
    _settle,
    _widgets,
)
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness


def _highlighted(results) -> str:
    return _rows(results)[results.highlighted]


def _source_word(screen) -> str:
    return str(screen.query_one("#settings-model-source", Static).renderable)


@pytest.mark.asyncio
@private_profile_test
async def test_down_twice_then_enter_stages_the_second_match(request):
    """R3 #1: type ``gpt``, Down, Down, Enter.

    The picker took every Down that bubbled out of the open list, put the
    highlight back on the first match and stopped the key, so the second
    Down went nowhere and Enter staged the first match.
    """
    app = _app("openai", "o3", catalog=("o3", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, results, adapter = _widgets(screen)

        field.focus()
        await pilot.pause()
        await pilot.press(*"gpt", "down")
        await pilot.pause()
        assert host.focused is results
        assert _highlighted(results) == "gpt-4o"

        await pilot.press("down")
        await pilot.pause()
        assert _highlighted(results) == "gpt-4.1"

        await pilot.press("enter")
        await pilot.pause()
        assert adapter.value == field.value == picker.value == "gpt-4.1"
        assert screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        assert _source_word(screen) == "edited *"


@pytest.mark.asyncio
@private_profile_test
async def test_an_id_typed_in_full_past_the_20_row_cap_is_the_one_enter_stages(
    request,
):
    """R3 #2: 26 catalog ids hold ``gpt-4`` and the 26th IS ``gpt-4``.

    The exact id was looked for in the 20 rows shown, so Enter staged the
    first row. The id typed keeps a row, is highlighted and painted, and is
    the model staged.
    """
    catalog = ("o3", *(f"gpt-4-{index:02d}" for index in range(25)), "gpt-4")
    app = _app("openai", "o3", catalog=catalog)
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, results, adapter = _widgets(screen)

        field.focus()
        await pilot.pause()
        await pilot.press(*"gpt-4")
        await pilot.pause()
        assert _highlighted(results) == "gpt-4"
        painted = "\n".join(_region_rows(screen, results))
        assert re.search(r"gpt-4(?![-\w])", painted), painted  # the row is on screen

        await pilot.press("enter")
        await pilot.pause()
        assert adapter.value == field.value == picker.value == "gpt-4"
        assert _source_word(screen) == "edited *"
