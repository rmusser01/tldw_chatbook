"""Settings ▸ Providers & Models ▸ Model: "Model list" pressed over an invalid
Custom ID puts the previous model back (review of PR #3057, R1 #7).

Esc and leaving the field already rolled an invalid id back and said so. The
picker's own button did not: it left the model blank, staged, with no toast.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_default_model_picker import (
    _SIZE,
    _app,
    _open_providers,
    _settle,
    _widgets,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness


@pytest.mark.asyncio
@private_profile_test
async def test_model_list_over_an_invalid_custom_id_restores_the_model(request):
    model = "gpt-4.1"
    app = _app("openai", model, catalog=(model, "gpt-4o"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, _results, adapter = _widgets(screen)
        field.focus()
        await pilot.pause()
        await pilot.press("tab", "enter")  # Custom ID
        await pilot.pause()
        assert picker.custom_mode and host.focused is field
        field.value = "m" * 257  # over the 256-character limit
        await pilot.pause()
        assert adapter.value == ""

        await pilot.press("tab")
        await pilot.pause()
        button = picker.query_one("#model-search-picker-custom", Button)
        assert host.focused is button
        await pilot.press("enter")  # Model list
        await pilot.pause(0.2)

        assert not picker.custom_mode
        assert adapter.value == field.value == picker.value == model
        assert screen._provider_setting_values_mapping()["model"] == model
        assert host.focused is field
        kept = f"Invalid model ID not kept; the model is still {model}."
        assert kept in [str(n.message) for n in host._notifications]
