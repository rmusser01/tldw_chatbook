from __future__ import annotations

import pytest

from tldw_chatbook.Chat import Chat_Functions
from tldw_chatbook.Chat.Chat_Deps import ChatConfigurationError
from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local

_MESSAGES = [{"role": "user", "content": "hello"}]


class _RuntimeConfigSnapshotStub:
    def __init__(self, values) -> None:
        self.values = values


@pytest.mark.parametrize("adapter_name", ["custom", "custom_2"])
def test_explicit_keyless_custom_adapter_does_not_reload_stale_credential(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
) -> None:
    stale_credential = "stale-credential-must-not-leak"
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "_chat_with_openai_compatible_local_server",
        lambda **kwargs: captured.update(kwargs) or {"choices": []},
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "get_runtime_config_snapshot",
        lambda: _RuntimeConfigSnapshotStub(
            {
                "api_settings": {
                    "custom": {
                        "api_url": "https://legacy.example.test/v1/chat/completions",
                        "api_key": stale_credential,
                        "model": "legacy-model",
                    }
                }
            }
        ),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "load_settings",
        lambda: {
            "custom_openai_api_2": {
                "api_ip": "https://legacy.example.test/v1/chat/completions",
                "api_key": stale_credential,
                "model": "legacy-model",
            }
        },
    )
    adapter = (
        LLM_API_Calls_Local.chat_with_custom_openai
        if adapter_name == "custom"
        else LLM_API_Calls_Local.chat_with_custom_openai_2
    )

    adapter(
        input_data=_MESSAGES,
        api_key=None,
        api_key_resolved=True,
        api_base_url="https://resolved.example.test/proxy/v1/chat/completions",
        model="resolved-model",
    )

    assert captured["api_key"] is None
    assert stale_credential not in repr(captured)


@pytest.mark.parametrize("adapter_name", ["custom", "custom_2"])
def test_legacy_custom_adapter_call_still_loads_configured_credential(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
) -> None:
    legacy_credential = "legacy-compatible-credential"
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "_chat_with_openai_compatible_local_server",
        lambda **kwargs: captured.update(kwargs) or {"choices": []},
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "get_runtime_config_snapshot",
        lambda: _RuntimeConfigSnapshotStub(
            {
                "api_settings": {
                    "custom": {
                        "api_url": "https://legacy.example.test/v1/chat/completions",
                        "api_key": legacy_credential,
                        "model": "legacy-model",
                    }
                }
            }
        ),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "load_settings",
        lambda: {
            "custom_openai_api_2": {
                "api_ip": "https://legacy.example.test/v1/chat/completions",
                "api_key": legacy_credential,
                "model": "legacy-model",
            }
        },
    )
    adapter = (
        LLM_API_Calls_Local.chat_with_custom_openai
        if adapter_name == "custom"
        else LLM_API_Calls_Local.chat_with_custom_openai_2
    )

    adapter(input_data=_MESSAGES)

    assert captured["api_key"] == legacy_credential


@pytest.mark.parametrize("adapter_name", ["custom", "custom_2"])
def test_explicit_keyless_custom_configuration_error_omits_stale_credential(
    monkeypatch: pytest.MonkeyPatch,
    adapter_name: str,
) -> None:
    stale_credential = "stale-error-credential-must-not-leak"
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "get_runtime_config_snapshot",
        lambda: _RuntimeConfigSnapshotStub(
            {
                "api_settings": {
                    "custom": {
                        "api_url": "https://legacy.example.test/v1/chat/completions",
                        "api_key": stale_credential,
                    }
                }
            }
        ),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "load_settings",
        lambda: {
            "custom_openai_api_2": {
                "api_ip": "https://legacy.example.test/v1/chat/completions",
                "api_key": stale_credential,
            }
        },
    )
    adapter = (
        LLM_API_Calls_Local.chat_with_custom_openai
        if adapter_name == "custom"
        else LLM_API_Calls_Local.chat_with_custom_openai_2
    )

    with pytest.raises(ChatConfigurationError) as captured:
        adapter(
            input_data=_MESSAGES,
            api_key=None,
            api_key_resolved=True,
            api_base_url="https://resolved.example.test/proxy/v1/chat/completions",
        )

    rendered_error = repr(captured.value)
    assert stale_credential not in rendered_error
    assert "object at 0x" not in rendered_error


@pytest.mark.parametrize("execution_key", ["custom-openai-api", "custom-openai-api-2"])
def test_dispatcher_forwards_explicit_credential_decision_without_value(
    monkeypatch: pytest.MonkeyPatch,
    execution_key: str,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setitem(
        Chat_Functions.API_CALL_HANDLERS,
        execution_key,
        lambda **kwargs: captured.update(kwargs) or {"choices": []},
    )

    Chat_Functions.chat_api_call(
        api_endpoint=execution_key,
        messages_payload=_MESSAGES,
        api_key=None,
        api_key_resolved=True,
        api_base_url="https://resolved.example.test/proxy/v1/chat/completions",
        model="resolved-model",
    )

    assert captured["api_key_resolved"] is True
    assert type(captured["api_key_resolved"]) is bool
    assert "api_key" not in captured
    assert "object at 0x" not in repr(captured)
