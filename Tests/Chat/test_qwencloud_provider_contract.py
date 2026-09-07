"""Cross-boundary registration contracts for the QwenCloud provider."""

from __future__ import annotations

import tomllib

import pytest

import tldw_chatbook.Chat.Chat_Functions as chat_functions

from tldw_chatbook.Chat.console_provider_support import (
    ConsoleProviderIdentity,
    resolve_console_provider_identity,
    supported_console_provider_catalog,
)
from tldw_chatbook.Chat.console_provider_endpoints import (
    effective_provider_endpoint,
    provider_uses_endpoint,
)
from tldw_chatbook.Chat.provider_catalog import provider_display_name
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.config import API_MODELS_BY_PROVIDER, CONFIG_TOML_CONTENT


def test_qwencloud_embedded_config_defaults() -> None:
    config = tomllib.loads(CONFIG_TOML_CONTENT)

    assert config["providers"]["QwenCloud"] == ["qwen3.8-max"]
    assert config["api_settings"]["qwencloud"] == {
        "api_mode": "responses",
        "api_key_env_var": "DASHSCOPE_API_KEY",
        "api_base_url": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        "model": "qwen3.8-max",
        "timeout": 120,
        "retries": 3,
        "retry_delay": 1,
        "streaming": True,
    }
    assert API_MODELS_BY_PROVIDER["QwenCloud"] == ["qwen3.8-max"]


def test_qwencloud_uses_one_supported_console_identity() -> None:
    handler_keys = {"qwencloud"}

    assert provider_display_name("qwencloud") == "QwenCloud"
    assert resolve_console_provider_identity(
        "QwenCloud",
        handler_keys=handler_keys,
    ) == ConsoleProviderIdentity(
        display_key="qwencloud",
        readiness_key="qwencloud",
        execution_key="qwencloud",
        is_supported=True,
    )

    catalog = supported_console_provider_catalog(handler_keys=handler_keys)
    assert len(catalog) == 1
    assert catalog[0].readiness_key == "qwencloud"
    assert catalog[0].execution_key == "qwencloud"
    assert catalog[0].display_name == "QwenCloud"


def test_qwencloud_readiness_uses_modern_config_before_its_env() -> None:
    modern = get_provider_readiness(
        "QwenCloud",
        {
            "api_settings": {
                "qwencloud": {
                    "api_key": "modern-qwencloud-key",
                    "api_key_env_var": "QWENCLOUD_OVERRIDE_KEY",
                }
            }
        },
        environ={
            "QWENCLOUD_OVERRIDE_KEY": "configured-env-key",
            "DASHSCOPE_API_KEY": "default-env-key",
        },
    )

    assert modern.requires_api_key is True
    assert modern.ready is True
    assert modern.api_key == "modern-qwencloud-key"
    assert modern.api_key_source == "config:api_settings.qwencloud.api_key"

    configured_env = get_provider_readiness(
        "QwenCloud",
        {"api_settings": {"qwencloud": {"api_key_env_var": "QWENCLOUD_OVERRIDE_KEY"}}},
        environ={
            "QWENCLOUD_OVERRIDE_KEY": "configured-env-key",
            "DASHSCOPE_API_KEY": "default-env-key",
        },
    )
    assert configured_env.api_key == "configured-env-key"
    assert configured_env.api_key_source == "env:QWENCLOUD_OVERRIDE_KEY"

    default_env = get_provider_readiness(
        "QwenCloud",
        {"api_settings": {"qwencloud": {"model": "qwen3.8-max"}}},
        environ={"DASHSCOPE_API_KEY": "default-env-key"},
    )
    assert default_env.api_key == "default-env-key"
    assert default_env.api_key_source == "env:DASHSCOPE_API_KEY"
    assert default_env.env_var == "DASHSCOPE_API_KEY"


def test_qwencloud_readiness_never_borrows_another_provider() -> None:
    readiness = get_provider_readiness(
        "QwenCloud",
        {
            "api_settings": {
                "qwencloud": {"model": "qwen3.8-max"},
                "openai": {"api_key": "openai-config-key"},
                "deepseek": {"api_key": "deepseek-config-key"},
                "custom": {"api_key": "custom-openai-config-key"},
            }
        },
        environ={
            "OPENAI_API_KEY": "openai-env-key",
            "DEEPSEEK_API_KEY": "deepseek-env-key",
            "CUSTOM_API_KEY": "custom-openai-env-key",
        },
    )

    assert readiness.requires_api_key is True
    assert readiness.ready is False
    assert readiness.api_key is None
    assert readiness.env_var == "DASHSCOPE_API_KEY"
    assert readiness.reason == "Missing API key"
    assert readiness.recovery == (
        "Set DASHSCOPE_API_KEY or add api_key under [api_settings.qwencloud]."
    )


def test_qwencloud_builtin_endpoint_is_international_compatible_base() -> None:
    assert effective_provider_endpoint("qwencloud", None, {}) == (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    assert provider_uses_endpoint("qwencloud", {}) is True


def test_chat_api_call_forwards_qwencloud_mode_base_and_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def qwencloud_handler(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    qwencloud_handler.__name__ = "qwencloud_handler"
    monkeypatch.setitem(
        chat_functions.API_CALL_HANDLERS, "qwencloud", qwencloud_handler
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    chat_functions.chat_api_call(
        api_endpoint="QwenCloud",
        messages_payload=[{"role": "user", "content": "hello"}],
        api_key="qwen-key",
        temp=0.2,
        system_message="Be concise.",
        streaming=False,
        model="qwen3.8-max",
        topk=20,
        topp=0.8,
        logprobs=True,
        top_logprobs=2,
        presence_penalty=0.1,
        frequency_penalty=0.4,
        tools=tools,
        tool_choice="auto",
        max_tokens=128,
        seed=7,
        stop=["END"],
        response_format={"type": "json_object"},
        n=1,
        reasoning_effort="medium",
        api_base_url="https://qwen.example/compatible-mode/v1",
        api_mode="chat_completions",
    )

    assert captured == {
        "input_data": [{"role": "user", "content": "hello"}],
        "model": "qwen3.8-max",
        "api_key": "qwen-key",
        "system_message": "Be concise.",
        "temp": 0.2,
        "streaming": False,
        "topp": 0.8,
        "topk": 20,
        "max_tokens": 128,
        "seed": 7,
        "stop": ["END"],
        "logprobs": True,
        "top_logprobs": 2,
        "presence_penalty": 0.1,
        "response_format": {"type": "json_object"},
        "n": 1,
        "tools": tools,
        "tool_choice": "auto",
        "reasoning_effort": "medium",
        "api_base_url": "https://qwen.example/compatible-mode/v1",
        "api_mode": "chat_completions",
    }
    assert "frequency_penalty" not in captured
    assert set(chat_functions.PROVIDER_PARAM_MAP["qwencloud"]) == {
        "messages_payload",
        "model",
        "api_key",
        "system_message",
        "temp",
        "streaming",
        "topp",
        "topk",
        "max_tokens",
        "seed",
        "stop",
        "logprobs",
        "top_logprobs",
        "presence_penalty",
        "response_format",
        "n",
        "tools",
        "tool_choice",
        "reasoning_effort",
        "api_base_url",
        "api_mode",
    }

    for endpoint in ("openai", "deepseek"):
        other: dict[str, object] = {}

        def representative_handler(**kwargs: object) -> dict[str, object]:
            other.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        representative_handler.__name__ = f"{endpoint}_handler"
        monkeypatch.setitem(
            chat_functions.API_CALL_HANDLERS, endpoint, representative_handler
        )
        chat_functions.chat_api_call(
            api_endpoint=endpoint,
            messages_payload=[{"role": "user", "content": "hello"}],
            model="representative-model",
            streaming=False,
            api_mode="responses",
        )
        assert "api_mode" not in other


def test_qwencloud_is_sensitive_auxiliary_audited() -> None:
    assert "qwencloud" in chat_functions.API_CALL_HANDLERS
    assert "qwencloud" in chat_functions.SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS
