"""Settings ▸ Model defaults: a field the provider does not accept is composed
blank (review of PR #3057, R1 #23).

Only Thinking budget went through the screen's ``_model_profile_input_value``;
every other hidden, disabled Input was composed holding the saved value,
which the screen's own sync blanks on the next provider or model change.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App
from textual.widgets import Input

from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Settings_Modules.settings_field_rows import (
    MODEL_DEFAULT_FIELDS,
    _model_default_control,
    draft_key,
)

# The Settings screen goes through config-participant admission, which the
# per-test sandbox refuses; keep the collection-time profile.
pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
async def test_an_input_the_provider_does_not_accept_is_composed_blank():
    screen = SettingsScreen.__new__(SettingsScreen)
    screen.app_instance = SimpleNamespace(app_config={})
    provider, model = "anthropic", "claude-sonnet-4-5"
    values = {draft_key(name): "" for name in MODEL_DEFAULT_FIELDS}
    values |= {draft_key("seed"): 7, draft_key("temperature"): 0.4}
    assert not screen._model_profile_field_supported(provider, draft_key("seed"), model)

    async with App().run_test():  # an Input with a value needs an active app
        hidden = _model_default_control(screen, provider, model, "seed", values, False)
        shown = _model_default_control(
            screen, provider, model, "temperature", values, True
        )

    assert isinstance(hidden, Input) and hidden.disabled
    assert hidden.value == ""
    assert shown.value == "0.4"
