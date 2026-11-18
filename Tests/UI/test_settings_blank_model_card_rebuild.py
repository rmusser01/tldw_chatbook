"""Settings ▸ Providers & Models: an emptied Default model stays empty when
the card is rebuilt (review of PR #3057, R1 #2).

An emptied Custom ID stages no model (``None``). A card rebuild then printed
that ``None`` as text: the picker and the hidden adapter read "None", the
adapter's Changed staged it, and Save wrote ``chat_defaults.model = "None"``.
"""

from __future__ import annotations

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_category_sweep import _click_settings_category
from Tests.UI.test_settings_default_model_picker import (
    _SIZE,
    _app,
    _open_providers,
    _settle,
    _widgets,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Settings_Modules.settings_field_rows import MODEL_DEFAULTS_ID


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("rebuild", ["pane", "category-round-trip"])
async def test_an_emptied_model_is_not_rebuilt_as_the_text_none(request, rebuild):
    model = "gpt-4.1"
    app = _app("openai", model, catalog=(model,))
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
        await pilot.press("end", *["backspace"] * len(model))
        await pilot.pause()
        assert field.value == adapter.value == ""
        assert screen._provider_draft().values["model"] is None

        if rebuild == "pane":
            screen.mutate_reactive(SettingsScreen.active_category)
        else:
            await _click_settings_category(pilot, "appearance")
            await _click_settings_category(pilot, "providers-models")
        await _settle(host, pilot)

        picker, field, _results, adapter = _widgets(screen)
        assert (field.value, adapter.value, picker.value) == ("", "", None)
        assert screen._provider_draft().values["model"] is None
        title = str(screen.query_one(f"#{MODEL_DEFAULTS_ID}").title)
        assert title.endswith("· no model"), title
