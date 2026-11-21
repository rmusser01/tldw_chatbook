"""Inference-cloud engine presets: together / fireworks / cerebras (ADR-179 Phase 2 Task 5).

Preset-cost: three registry records + three dispatch entries are the ENTIRE
implementation -- no per-provider ``LLM_Calls`` module may ship for them.

Allowances: TASK-34367.3-.5 (3e618d7119) replaced the provisional empty sets
with ones taken from each provider's official schema (the reconciliation note
above ``TOGETHER`` in the registry names each source; offline contracts, live
qualification stays with TASK-33640). Two sets are proved by live capture
instead and keep their fixture citations: Together's choice ``logprobs`` and
Fireworks' tool call ``{index, name}`` (TASK-42132).

Live captures of these (and every other engine preset) replay under their
record in ``Tests/LLM_Calls/test_live_capture_replay.py`` (TASK-33640).
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


# --- allowances: the registry's doc-sourced contract plus captured sets ---


# What each provider's official schema documents (TASK-34367.3-.5; the
# registry's reconciliation note above TOGETHER names the source, below).
_DOCUMENTED_ALLOWANCES = {
    "together": {
        "response": frozenset({"prompt", "warnings"}),
        "choice": frozenset({"seed", "top_logprobs", "text"}),
        "message": frozenset({"reasoning"}),
    },
    "fireworks": {
        "response": frozenset({"perf_metrics", "prompt_token_ids"}),
        "choice": frozenset({"logprobs", "raw_output"}),
    },
    "cerebras": {
        "response": frozenset({"time_info", "service_tier", "service_tier_used"}),
        "choice": frozenset({"logprobs", "reasoning_logprobs"}),
        "message": frozenset({"reasoning"}),
    },
}
_ALLOWANCE_SOURCES = {
    "together": "docs.together.ai/reference/chat-completions",
    "fireworks": "docs.fireworks.ai/api-reference/post-chatcompletions",
    "cerebras": "inference-docs.cerebras.ai/api-reference/chat-completions",
}
# Allowances a live capture proved, with the fixture that proves them.
_CAPTURED_CHOICE_ALLOWANCES = {
    "together": frozenset({"logprobs"}),  # cloud_live/together.json (TASK-34362)
}
_CAPTURED_TOOL_CALL_ALLOWANCES = {
    "fireworks": frozenset({"index", "name"}),  # cloud_live/fireworks.json (TASK-34364)
}


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_allowances_are_the_documented_contract_plus_captured_sets(key: str) -> None:
    """TASK-42132: each family is the documented set plus any captured set,
    and every family neither names is empty. A captured set stays apart from
    the documented one, so its fixture citation is not lost."""
    record = RECORDS_BY_KEY[key]
    captured = {
        "choice": _CAPTURED_CHOICE_ALLOWANCES.get(key, frozenset()),
        "tool_call": _CAPTURED_TOOL_CALL_ALLOWANCES.get(key, frozenset()),
    }
    for family in ("response", "choice", "message", "tool_call"):
        documented = _DOCUMENTED_ALLOWANCES[key].get(family, frozenset())
        proved = captured.get(family, frozenset())
        assert not documented & proved, family
        assert getattr(record, f"{family}_allowances") == documented | proved, family


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_registry_reconciliation_note_names_each_presets_schema_source(key: str) -> None:
    """TASK-42132: the doc-sourced sets above must stay traceable from the
    data site, so the registry's TASK-34367.3-.5 note (which replaced the
    PROVISIONAL PENDING FIRST LIVE CAPTURE comment) names each preset's
    official schema."""
    registry_source = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "provider_registry.py"
    ).read_text(encoding="utf-8")
    # The note runs to the first blank comment line (later paragraphs cite
    # the same pages for other facts).
    note = registry_source.split("# TASK-34367.3-.5:", 1)[1].split("\n#\n", 1)[0]
    assert _ALLOWANCE_SOURCES[key] in note


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
    llm_calls_root = (
        Path(__file__).resolve().parents[2] / "tldw_chatbook" / "LLM_Calls"
    )
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
