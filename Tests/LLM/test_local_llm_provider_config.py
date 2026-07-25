"""task-625: the local-llm provider must resolve its config from api_settings.

`chat_with_local_llm` read a TOP-LEVEL "local-llm" settings key, while the
provider's configuration lives at `api_settings.local-llm` — which is where the
app's own documented example puts it, and where every sibling local provider in
the same module looks. `load_settings()` does not preserve arbitrary top-level
sections either, so the provider was unusable from configuration entirely.
"""

import pytest

from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local_calls


@pytest.fixture
def captured_call(monkeypatch):
    """Intercept the outbound request so no network is touched."""
    seen = {}

    def fake_request(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    # `_chat_with_openai_compatible_local_server` is the shared transport every
    # local provider funnels through; stubbing it keeps this a pure config test.
    monkeypatch.setattr(
        local_calls,
        "_chat_with_openai_compatible_local_server",
        fake_request,
    )
    return seen


def _settings_with(api_settings_block):
    return {"api_settings": {"local-llm": api_settings_block}}


def test_resolves_api_url_from_api_settings(monkeypatch, captured_call):
    """The documented location must work."""
    monkeypatch.setattr(
        local_calls,
        "settings",
        _settings_with({"api_url": "http://127.0.0.1:9099/v1/chat/completions"}),
    )
    local_calls.chat_with_local_llm(
        input_data=[{"role": "user", "content": "hi"}],
    )
    assert captured_call["kwargs"].get("api_base_url") == (
        "http://127.0.0.1:9099/v1/chat/completions"
    )


def test_legacy_api_ip_key_still_accepted(monkeypatch, captured_call):
    """`api_ip` is what the code historically read; don't break anyone on it."""
    monkeypatch.setattr(
        local_calls,
        "settings",
        _settings_with({"api_ip": "http://127.0.0.1:9099/v1/chat/completions"}),
    )
    local_calls.chat_with_local_llm(
        input_data=[{"role": "user", "content": "hi"}],
    )
    assert captured_call["kwargs"].get("api_base_url") == (
        "http://127.0.0.1:9099/v1/chat/completions"
    )


def test_documented_key_wins_over_legacy(monkeypatch, captured_call):
    monkeypatch.setattr(
        local_calls,
        "settings",
        _settings_with(
            {
                "api_url": "http://documented:9099/v1/chat/completions",
                "api_ip": "http://legacy:9099/v1/chat/completions",
            }
        ),
    )
    local_calls.chat_with_local_llm(
        input_data=[{"role": "user", "content": "hi"}],
    )
    assert "documented" in captured_call["kwargs"].get("api_base_url", "")


def test_missing_url_still_raises_a_clear_configuration_error(monkeypatch):
    """A genuinely absent URL must stay a config error, not an opaque 502."""
    monkeypatch.setattr(local_calls, "settings", _settings_with({}))
    with pytest.raises(Exception) as excinfo:
        local_calls.chat_with_local_llm(
            input_data=[{"role": "user", "content": "hi"}],
            custom_prompt_arg=None,
        )
    assert "url" in str(excinfo.value).lower()
