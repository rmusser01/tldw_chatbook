"""Settings ▸ Providers & Models ▸ Context window: no assumed size for a
server that reports its own (review of #3057, R3 #4, ruling C7).

For llama.cpp, Ollama, vLLM and the other servers the Console's live probe
reads, Settings said "unknown, 32,000 assumed · budgets use it until set".
That is false: the Console budgets with the size the server reports.
Settings cannot probe, so it says the size is unknown here and where it
comes from; other providers keep "unknown, N assumed".
"""

from __future__ import annotations

import pytest
from textual.widgets import Collapsible, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_advanced_disclosures import _open, _title
from Tests.UI.test_settings_connect_rows import _SIZE
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness
from tldw_chatbook.Chat.console_context_window import (
    SERVER_REPORTED_WINDOW_FAMILIES,
)
from tldw_chatbook.UI.Screens.settings_context_memory import (
    model_context_window_state,
)

_UNKNOWN_HERE = "unknown here · the Console reads it from the server"
_CONTEXT_ID = "settings-advanced-context-window"
_ENDPOINTS = {
    "custom_endpoints": {
        "gpu-box": {
            "display_name": "GPU box",
            "family": "llama_cpp",
            "base_url": "http://192.168.1.5:8080",
            "models": ["model-a"],
        },
        "proxy": {
            "display_name": "Proxy",
            "family": "openai_compatible",
            "base_url": "http://192.168.1.6:9000/v1",
            "models": ["model-a"],
        },
    }
}


@pytest.mark.bootstrap_profile
def test_the_probe_names_the_families_it_reads():
    assert {"llama_cpp", "ollama", "vllm", "custom"} <= SERVER_REPORTED_WINDOW_FAMILIES
    assert "openrouter" not in SERVER_REPORTED_WINDOW_FAMILIES
    assert "openai" not in SERVER_REPORTED_WINDOW_FAMILIES


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize(
    "provider",
    ["llama_cpp", "ollama", "vllm", "custom", "custom-ep:gpu-box", "custom-ep:proxy"],
)
def test_a_server_that_reports_its_window_gets_no_assumed_size(provider):
    state = model_context_window_state(_ENDPOINTS, provider, "made-up-model-x")

    assert state.effective_tokens is None and state.detected_tokens is None
    assert state.server_reported
    assert state.assumed_tokens is None


@pytest.mark.bootstrap_profile
def test_other_providers_keep_their_assumed_size_and_a_known_window_is_known():
    unknown = model_context_window_state({}, "openai", "made-up-model-x")
    assert (unknown.assumed_tokens, unknown.server_reported) == (32000, False)

    known = model_context_window_state({}, "openai", "gpt-4o")
    assert (known.effective_tokens, known.server_reported) == (128000, False)

    override = {"model_capabilities": {"models": {"local-x": {"context_window": 8192}}}}
    set_here = model_context_window_state(override, "llama_cpp", "local-x")
    assert (set_here.effective_tokens, set_here.server_reported) == (8192, False)


@pytest.mark.asyncio
@private_profile_test
async def test_settings_says_unknown_here_for_a_local_server(request):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "local-x"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099", "model": "local-x"}
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open(host, pilot)
        assert _title(screen, _CONTEXT_ID) == f"Context window · {_UNKNOWN_HERE}"

        screen.query_one(f"#{_CONTEXT_ID}", Collapsible).collapsed = False
        await pilot.pause()
        await pilot.pause()
        source = screen.query_one("#settings-model-context-window-source", Static)
        help_line = screen.query_one("#settings-model-context-window-status", Static)
        assert str(source.render()) == "not set"
        assert str(help_line.render()) == _UNKNOWN_HERE
        # Painted whole at the spec's size, not cut by the row.
        assert _UNKNOWN_HERE in "".join(_region_rows(screen, help_line))
