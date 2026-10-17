"""The Console chrome shows provider text literally (TASK-33002.19).

A hand-edited ``[chat_defaults] provider = "custom-ep:gone[/]"`` (a dangling
registry id whose text is Textual markup) raised ``MarkupError`` from the
workbench mode strip at startup, and a custom endpoint named with markup
rendered styled, not as typed. The strip's chips now render without markup
parsing, on the shipping Console screen over a scratch config the real
writer owns (``private_profile_test``).

The control bar's hidden compatibility labels carry the same text, and the
Console pushes control state to them before the status chips and the mode
strip. Any later push (an MCP catalog count, a switch to a markup-named
endpoint) raised there, so the visible chips never got it; both later
pushes are driven here too.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app, attach_chachanotes_db
from Tests.UI.test_console_switch_model_keys import _Harness
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.config import save_settings_to_cli_config
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

_GPU_BOX = {
    "family": "llama_cpp",
    "base_url": "http://127.0.0.1:9099",
    "models": ["model-a"],
}


def _plain(console: ChatScreen, selector: str) -> str:
    return str(console.query_one(selector, Static).visual.plain)


async def _until_shown(pilot, console: ChatScreen, shown: str) -> None:
    """Wait for the mode strip's provider chip, then check the visible chip."""
    for _ in range(100):
        if _plain(console, "#workbench-mode-provider") == shown:
            break
        await pilot.pause(0.05)
    assert _plain(console, "#workbench-mode-provider") == shown
    assert _plain(console, "#console-provider-chip") == shown


async def _bump_mcp_tools(pilot, console: ChatScreen) -> None:
    """Push a changed control state, as a loaded MCP catalog does."""
    console.app_instance.console_mcp_tool_count = 3
    console._sync_console_control_bar()
    for _ in range(100):
        if "3" in _plain(console, "#workbench-mode-tools"):
            return
        await pilot.pause(0.05)
    raise AssertionError(_plain(console, "#workbench-mode-tools"))


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
            **_GPU_BOX,
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
        await _until_shown(pilot, console, shown)
        # A later push reaches the strip and leaves the provider literal.
        await _bump_mcp_tools(pilot, console)
        await _until_shown(pilot, console, shown)


@pytest.mark.asyncio
@private_profile_test
async def test_switching_to_a_markup_named_endpoint_shows_its_name_literally(
    request,
) -> None:
    """AC#2 on a runtime switch: opened on llama.cpp, the chat moves to an
    endpoint named ``GPU [/]box``, then a later push follows."""
    assert save_settings_to_cli_config(
        {
            "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
            "api_settings.llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "model-a",
            },
            "custom_endpoints.gpu-box": {"display_name": "GPU [/]box", **_GPU_BOX},
        }
    )
    app = _build_test_app()
    attach_chachanotes_db(app)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    harness = _Harness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(console, pilot, "#workbench-mode-provider")
        store = console._ensure_console_chat_store()
        store.replace_session_settings(
            store.active_session_id,
            ConsoleSessionSettings(provider="custom-ep:gpu-box", model="model-a"),
        )
        console._sync_console_control_bar()
        await _until_shown(pilot, console, "Provider: GPU [/]box")
        await _bump_mcp_tools(pilot, console)
        await _until_shown(pilot, console, "Provider: GPU [/]box")
