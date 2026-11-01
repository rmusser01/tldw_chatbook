"""Settings ▸ Providers & Models ▸ Model: keys typed faster than the screen
hears of them do not throw the field out of Custom ID mode (review of
#3057, R3 #3).

Each Custom ID keystroke reaches the screen as a message. When a second key
landed first (key repeat, an IME, a paste sent as keys), the screen compared
its adapter (one key behind) with the live field, took the difference for a
stale picker and pushed the older text back: the field left Custom ID mode,
was rewritten and selected.
"""

from __future__ import annotations

import pytest
from textual import events
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_default_model_picker import (
    _app,
    _open_providers,
    _settle,
    _widgets,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness

_SIZE = (211, 44)


@pytest.mark.asyncio
@private_profile_test
async def test_two_queued_keys_keep_the_field_in_custom_id_mode(request):
    host = _SettingsCssHarness(
        _app("openai", "gpt-4o", catalog=("gpt-4o",)), "settings"
    )

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, _results, adapter = _widgets(screen)
        field.focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert host.focused is picker.query_one("#model-search-picker-custom", Button)
        await pilot.press("enter")
        await pilot.pause()
        assert picker.custom_mode and host.focused is field
        await pilot.press(*"my-model-")
        await _settle(host, pilot)
        assert adapter.value == "my-model-"

        # Both keys are in the field's queue before the screen hears of "a".
        field.post_message(events.Key("a", "a"))
        field.post_message(events.Key("b", "b"))
        await _settle(host, pilot)
        await pilot.pause()

        assert picker.custom_mode
        assert field.value == "my-model-ab"
        assert field.selection.is_empty
        assert adapter.value == picker.value == "my-model-ab"
        assert screen._provider_setting_values_mapping()["model"] == "my-model-ab"
