"""Settings ▸ Providers & Models: the card at rest fits the detail pane (TASK-33007 AC#1).

At 211x44, with every Advanced disclosure closed, the whole card is visible
without scrolling. That used to be asserted for OpenAI · gpt-4.1 alone, yet
the card's height depends on the provider: Connect gains a row for some
(Anthropic's Sign in with, QwenCloud's API mode) and Model defaults shows
more rows for others (llama.cpp hides three fields, most cloud providers ten).
Here every provider the picker lists is chosen in turn on one mounted card.
"""

from __future__ import annotations

import pytest
from textual.widgets import Collapsible, Select

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_connect_rows import _SIZE, _held, _open_providers, _until
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness


@pytest.mark.asyncio
@private_profile_test
async def test_the_closed_card_fits_the_pane_for_every_listed_provider(request):
    """Parent AC#1: no provider's card needs scrolling at 211x44 while its
    Advanced disclosures are closed. A provider switch stages a draft, which
    only ever adds rows, so a card that fits here fits at rest too."""
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {}}
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        body = screen.query_one("#settings-detail-pane-body")
        card = screen.query_one("#settings-providers-models-card")
        # The hidden Select is the adapter the Provider control chooses through.
        adapter = screen.query_one("#settings-provider-value", Select)
        providers = [
            str(option.provider_id)
            for group in screen._provider_picker_groups()
            if group.group_id not in {"actions", "saved"}
            for option in group.options
        ]
        assert len(providers) >= 50, providers
        assert "anthropic" in providers and "qwencloud" in providers

        scrolls: dict[str, tuple[int, int]] = {}
        for provider in providers:
            adapter.value = screen._provider_select_value_for_provider(provider)
            await _until(pilot, lambda: _held(screen) == provider)
            assert _held(screen) == provider
            await pilot.pause()
            await pilot.pause()
            assert all(
                disclosure.collapsed
                for disclosure in card.query(".settings-advanced-disclosure").results(
                    Collapsible
                )
            )
            if body.max_scroll_y or card.region.bottom > body.region.bottom:
                scrolls[provider] = (body.max_scroll_y, card.region.height)

        assert not scrolls, scrolls
