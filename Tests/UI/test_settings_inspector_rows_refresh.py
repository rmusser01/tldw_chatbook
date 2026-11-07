"""Settings ▸ Providers & Models: two Inspector refreshes that went wrong
(review of #3057, R2 F4 and F5 = R1 #16).

F4: the Key check row's verdict shared one try with the Inspector's Test
result row, so a missing Inspector row skipped it, and the same except
swallowed a real AttributeError raised while building the verdict.

F5: the Endpoint key row is composed folded at its dots and was refreshed
unfolded, so the first edit re-broke the key mid-word.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import (
    _FAKE_KEY,
    _SIZE,
    _open_providers,
    _text,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.UI.Settings_Modules import providers_models_card as card


def _app():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key": _FAKE_KEY}}
    return app


@pytest.mark.asyncio
@private_profile_test
async def test_the_key_check_row_updates_without_the_inspectors_test_result_row(
    request, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await screen.query_one("#settings-provider-test-result", Static).remove()
        screen._provider_test_result = "Readiness   Ready · verified\nModel   gpt-4.1"

        screen._update_provider_test_result()
        await pilot.pause()

        assert _text(screen, "#settings-provider-readiness") == "Ready · verified"

        def broken_verdict(_screen):
            raise AttributeError("a real defect in the verdict")

        monkeypatch.setattr(card, "key_check_verdict", broken_verdict)
        with pytest.raises(AttributeError, match="a real defect"):
            screen._update_provider_test_result()


@pytest.mark.asyncio
@private_profile_test
async def test_the_endpoint_key_row_stays_folded_when_it_is_refreshed(
    request, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        composed = _text(screen, "#settings-provider-endpoint-key")
        assert composed == "Endpoint key: api_settings.openai.\n  api_base_url"

        screen._update_provider_dynamic_widgets()
        await pilot.pause()

        assert _text(screen, "#settings-provider-endpoint-key") == composed
