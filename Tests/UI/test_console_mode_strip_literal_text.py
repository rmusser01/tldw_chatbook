"""The Console mode strip shows provider text literally (TASK-33002.19).

A hand-edited ``[chat_defaults] provider = "custom-ep:gone[/]"`` (a dangling
registry id whose text is Textual markup) raised ``MarkupError`` from the
workbench mode strip at startup, and a custom endpoint named with markup
rendered styled, not as typed. The strip's chips now render without markup
parsing, on the shipping Console screen over a scratch config the real
writer owns (``private_profile_test``).
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from Tests.UI.test_console_switch_model_keys import _Harness
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.config import save_settings_to_cli_config
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "endpoint_name", "shown"),
    [
        # AC#1: the dangling id is its own label, and `[/]` would raise.
        ("custom-ep:gone[/]", None, "Provider: custom-ep:gone[/]"),
        # AC#2: an endpoint's display name reaches the strip; `[b]` would
        # render bold and lose its brackets, without raising.
        ("custom-ep:gpu-box", "GPU [b]box[/b]", "Provider: GPU [b]box[/b]"),
    ],
)
async def test_the_mode_strip_shows_a_markup_like_provider_literally(
    request, provider: str, endpoint_name: str | None, shown: str
) -> None:
    sections: dict[str, dict[str, object]] = {
        "chat_defaults": {"provider": provider, "model": "model-a"},
    }
    if endpoint_name is not None:
        sections["custom_endpoints.gpu-box"] = {
            "display_name": endpoint_name,
            "family": "llama_cpp",
            "base_url": "http://127.0.0.1:9099",
            "models": ["model-a"],
        }
    assert save_settings_to_cli_config(sections)
    app = _build_test_app()
    attach_chachanotes_db(app)
    app.chat_api_provider_value = provider
    app.chat_api_model_value = "model-a"
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#workbench-mode-provider")
        chip = console.query_one("#workbench-mode-provider", Static)
        for _ in range(100):
            if str(chip.visual.plain) == shown:
                break
            await pilot.pause(0.05)
        assert str(chip.visual.plain) == shown
