"""Settings ▸ Providers & Models: arriving on the Provider control does not
open its list (review of #3057, R2 F10).

The control is built holding the provider's name, and a mounted Input
posts Changed for that first value. When focus arrives before the message
does (the '/' search lands on the control), the handler took it for typing
and opened the list.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, OptionList

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import _FAKE_KEY, _SIZE, _open_providers
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness


@pytest.mark.asyncio
@private_profile_test
async def test_the_mount_echo_of_the_providers_name_does_not_open_the_list(request):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": _FAKE_KEY}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        control = screen.query_one("#settings-provider-search", Input)
        picker = screen.query_one("#settings-provider-picker", OptionList)
        control.focus()
        await pilot.pause()
        assert control.value == "OpenAI" and not picker.display

        # The first-value message, delivered after focus arrived.
        control.post_message(Input.Changed(control, control.value))
        await pilot.pause()
        await pilot.pause()  # the message reaches the screen a turn later
        assert not picker.display

        # Typing still opens it, and a name typed back in full keeps it open.
        await pilot.press(*"OpenA")
        await pilot.pause()
        assert picker.display
        await pilot.press("I")
        await pilot.pause()
        assert control.value == "OpenAI" and picker.display
