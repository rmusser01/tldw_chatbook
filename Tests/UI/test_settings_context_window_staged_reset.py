"""Settings ▸ Providers & Models ▸ Context window: a staged Reset shows as an
edit even when the override equals the detected size (review of #3057, R2 F6).

Reset stages the detected value. With an override equal to it the shown
number does not change, so the row and the disclosure title went on saying
"saved in config" / "override set" while Save was about to remove it.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_advanced_disclosures import _app, _open, _title
from Tests.UI.test_settings_connect_rows import _SIZE, _text
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.UI.Screens.settings_context_memory import (
    model_context_window_state,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

_CONTEXT_ID = "settings-advanced-context-window"


@pytest.mark.asyncio
@private_profile_test
async def test_a_staged_reset_reads_edited_when_the_override_equals_the_detected_size(
    request,
):
    detected = model_context_window_state({}, "openai", "gpt-4.1").detected_tokens
    assert detected, "the test needs a model whose window is detected"
    app = _app()
    app.app_config["model_capabilities"] = {
        "models": {"gpt-4.1": {"context_window": detected}}
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        assert _text(screen, "#settings-model-context-window-source") == (
            "saved in config"
        )
        assert _title(screen, _CONTEXT_ID).endswith("override set")

        screen.query_one("#settings-model-context-window-reset", Button).press()
        await pilot.pause()
        await pilot.pause()

        assert screen._category_has_unsaved_changes(SettingsCategoryId.PROVIDERS_MODELS)
        assert _text(screen, "#settings-model-context-window-source") == "edited *"
        assert _title(screen, _CONTEXT_ID).endswith("edited *")
