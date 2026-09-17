"""Controller regressions for retaining an exact conversation settings draft."""

import pytest

from Tests.Chat.test_console_settings_apply import _field, _rebase, _state
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    QUICK_MODEL_DEFAULT_FIELDS,
    remember_model_draft,
)


@pytest.mark.parametrize(
    "exposed_fields", [QUICK_MODEL_DEFAULT_FIELDS, FULL_MODEL_DEFAULT_FIELDS]
)
def test_same_target_retains_conversation_values_and_profile_intent(exposed_fields):
    state = _state(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            temperature=0.23,
            top_p=0.42,
            base_url="http://127.0.0.1:9101",
        ),
        _field("temperature", 0.23, profile_override=0.23),
        _field("streaming", False, profile_override=False, dirty=True),
    )

    rebased = _rebase(
        state,
        provider="llama_cpp",
        model="model-a",
        app_config={"chat_defaults": {"temperature": 0.6, "top_p": 0.9}},
        exposed_fields=exposed_fields,
    )

    assert rebased.settings.temperature == 0.23
    assert rebased.settings.top_p == 0.42
    assert rebased.settings.streaming is False
    assert rebased.settings.base_url == "http://127.0.0.1:9101"
    assert rebased.endpoint_draft is None
    field = {field.name: field for field in rebased.field_drafts}["temperature"]
    assert field.profile_override == 0.23
    assert not field.dirty


@pytest.mark.parametrize(
    "exposed_fields", [QUICK_MODEL_DEFAULT_FIELDS, FULL_MODEL_DEFAULT_FIELDS]
)
def test_a_b_a_restores_untouched_conversation_values_and_endpoint(exposed_fields):
    state = _state(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="model-a",
            temperature=0.23,
            top_p=0.42,
            base_url="http://127.0.0.1:9101",
        ),
        _field("temperature", 0.23, profile_override=0.23),
        _field("streaming", True, profile_override=True),
    )
    config = {
        "chat_defaults": {"temperature": 0.6},
        "api_settings": {
            "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
            "vllm": {"api_url": "http://127.0.0.1:9098"},
        },
    }
    state_b = _rebase(
        remember_model_draft(state),
        provider="vllm",
        model="model-b",
        app_config=config,
        exposed_fields=exposed_fields,
    )
    assert state_b.settings.temperature == 0.6
    assert state_b.settings.base_url == "http://127.0.0.1:9098"

    restored = _rebase(
        remember_model_draft(state_b),
        provider="llama_cpp",
        model="model-a",
        app_config=config,
        exposed_fields=exposed_fields,
    )

    assert restored.settings.temperature == 0.23
    assert restored.settings.top_p == 0.42
    assert restored.settings.base_url == "http://127.0.0.1:9101"


def test_registry_target_and_remembered_draft_keep_literal_hyphenated_entry_id():
    config = {
        "custom_endpoints": {
            "gpu-node": {
                "display_name": "GPU node",
                "family": "llama_cpp",
                "base_url": "http://127.0.0.1:9101",
                "models": ["model-a"],
            }
        }
    }
    initial = _state(ConsoleSessionSettings(provider="llama_cpp", model="local"))
    target = _rebase(
        initial,
        provider="custom-ep:gpu-node",
        model="model-a",
        app_config=config,
    )
    assert target.settings.provider == "custom-ep:gpu-node"
    remembered = remember_model_draft(target)
    assert remembered.model_drafts[0].provider == "custom-ep:gpu-node"
    assert remembered.model_drafts[0].settings.provider == "custom-ep:gpu-node"
    reapplied = _rebase(
        remembered,
        provider="custom-ep:gpu-node",
        model="model-a",
        app_config=config,
    )
    assert reapplied.settings.provider == "custom-ep:gpu-node"
    assert reapplied.settings.base_url == "http://127.0.0.1:9101"
