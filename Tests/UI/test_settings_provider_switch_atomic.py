"""TASK-388: switching the provider updates the dependent fields atomically.

The review saw the form assert a stale provider/model/readiness combination for
~1-3s after selecting a new provider. In the current code the provider
Select.Changed handler updates every dependent field synchronously, so there is
no window where the form shows the previous provider. This locks that atomicity.
"""

import pytest
from textual.widgets import Input, Select, Static

from Tests.UI.test_settings_configuration_hub import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _open_settings_category,
)


@pytest.mark.asyncio
async def test_provider_switch_updates_dependent_fields_with_no_stale_window():
    """Selecting a new provider flips readiness/source/model in the same tick."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4o"}
    app.app_config["api_settings"] = {
        "openai": {"api_key": "fake-key-not-real"},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        await pilot.pause()

        provider_select = screen.query_one("#settings-provider-value", Select)
        # Drive the real Select.Changed handler, then read the dependent fields
        # WITHOUT pumping the event loop -- any staleness would show here.
        screen.handle_provider_value_changed(
            Select.Changed(provider_select, "llama.cpp")
        )

        readiness = str(
            screen.query_one("#settings-provider-readiness", Static).renderable
        )
        source = str(
            screen.query_one("#settings-provider-source", Static).renderable
        )
        model_value = screen.query_one("#settings-model-value", Input).value

        # The dependent fields reflect the NEW provider immediately...
        assert "llama.cpp" in readiness
        assert "draft" in source.lower()
        # ...and never assert the previous provider/model combination.
        assert "gpt-4o" not in readiness
        assert model_value != "gpt-4o"
