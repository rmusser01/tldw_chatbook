"""Inference-cloud engine presets: together / fireworks / cerebras (ADR-179 Phase 2 Task 5).

Preset-cost: three registry records + three dispatch entries are the ENTIRE
implementation -- no per-provider ``LLM_Calls`` module may ship for them.

Current response allowances are qualified by primary-schema fixtures through
chat_api_call and the actual adapters in test_documented_provider_response_contracts.py
(TASK-34367.3-.5); no paid live-capture claim is made. Live reconciliation
remains Tests/LLM_Calls/test_live_capture_replay.py (TASK-33640).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.provider_registry import RECORDS_BY_KEY

PRESET_KEYS = ("together", "fireworks", "cerebras")
EXPECTED_DEFAULTS: dict[str, dict[str, Any]] = {
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_var": "TOGETHER_API_KEY",
        "config_key": "Together",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_var": "FIREWORKS_API_KEY",
        "config_key": "Fireworks",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_var": "CEREBRAS_API_KEY",
        "config_key": "Cerebras",
    },
}

# --- registration / record shape ---


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_preset_record_shape(key: str) -> None:
    record = RECORDS_BY_KEY[key]
    assert record.classification == "cloud"
    assert record.engine_driven is True
    assert record.auth_scheme == "bearer"
    assert record.tolerant_response_extras is False  # strict, unlike custom-ep
    assert record.auto_refresh is True
    assert record.native_tools is True
    assert record.base_url_suffix is None  # the default URL is already complete
    assert record.continuation_protocol == "chat_completions"
    assert record.discovery_route == "models"


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_default_urls_and_env_vars(key: str) -> None:
    expected = EXPECTED_DEFAULTS[key]
    record = RECORDS_BY_KEY[key]
    assert record.default_base_url == expected["base_url"]
    assert record.api_key_env_var == expected["env_var"]
    assert record.api_key_env_candidates == (expected["env_var"],)
    assert record.config_key == expected["config_key"]


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_settings_defaults_carry_no_model_key(key: str) -> None:
    """Phase 1 blank-model lesson: a present-but-blank shipped model fails
    closed at engine resolution; the UNSET key resolves to the payload-gated
    "". Only the unset key ships."""
    record = RECORDS_BY_KEY[key]
    assert "model" not in record.settings_defaults
    assert record.settings_defaults == {
        "api_key_env_var": EXPECTED_DEFAULTS[key]["env_var"],
        "streaming": True,
        "timeout": 90,
        "retries": 3,
        "retry_delay": 5.0,
    }


# --- tool-call extras: record-scoped live-capture contract ---


# Other preset response levels keep their documented contracts; this field is
# qualified by the actual Fireworks tool fixture (TASK-34364).
_CAPTURED_TOOL_CALL_ALLOWANCES = {
    "fireworks": frozenset({"index", "name"}),  # cloud_live/fireworks.json
}


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_tool_call_allowances_match_the_captured_contract(key: str) -> None:
    """Only the captured preset accepts these non-streamed tool-call extras."""
    assert RECORDS_BY_KEY[key].tool_call_allowances == (
        _CAPTURED_TOOL_CALL_ALLOWANCES.get(key, frozenset())
    )


# --- fireworks: proprietary reasoning ---


def test_fireworks_reasoning_is_proprietary() -> None:
    """Fireworks returns reasoning in ``reasoning_content`` and requires it
    replayed on interleaved tool turns, so the engine treats it as
    proprietary -- never surfaced, replayed only through continuations."""
    assert RECORDS_BY_KEY["fireworks"].reasoning_disposition == "proprietary"
    assert RECORDS_BY_KEY["together"].reasoning_disposition == "ignored"
    assert RECORDS_BY_KEY["cerebras"].reasoning_disposition == "ignored"


# --- preset-cost: no per-provider module ---


def test_no_per_provider_module_ships() -> None:
    """The whole preset is a registry record + dispatch entry: the engine
    closure replaces what a ``chat_with_together`` module used to be."""
    llm_calls_root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "LLM_Calls"
    for key in PRESET_KEYS:
        matches = sorted(llm_calls_root.glob(f"{key}*.py"))
        assert matches == [], f"unexpected per-provider module(s): {matches}"


# --- dispatch through the engine factory ---


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_dispatch_registered_through_the_engine_factory(key: str) -> None:
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    handler = API_CALL_HANDLERS.get(key)
    assert callable(handler), f"{key} missing from API_CALL_HANDLERS"


# --- config tables: empty [providers] seeds + [api_settings.*] ---


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_config_tables_match_the_record(key: str) -> None:
    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    record = RECORDS_BY_KEY[key]
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    assert parsed["providers"].get(record.config_key) == []
    table = parsed["api_settings"][key]
    assert table == dict(record.settings_defaults) | {
        "api_base_url": record.default_base_url
    }
    assert "model" not in table


# --- model discovery gate: each default URL must be discoverable ---


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_default_urls_pass_the_discovery_gate(key: str) -> None:
    from tldw_chatbook.LLM_Provider_Catalog.openai_compatible_model_discovery import (
        build_models_url,
        supports_openai_compatible_model_discovery,
    )

    base = EXPECTED_DEFAULTS[key]["base_url"]
    assert supports_openai_compatible_model_discovery(key, base) is True
    assert build_models_url(base, key) == f"{base}/models"
