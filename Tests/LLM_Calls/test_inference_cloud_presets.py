"""Inference-cloud engine presets: together / fireworks / cerebras (ADR-179 Phase 2 Task 5).

Preset-cost: three registry records + three dispatch entries are the ENTIRE
implementation -- no per-provider ``LLM_Calls`` module may ship for them.

Allowances reality (Task 2 outcome): this environment holds no provider
keys, so NO cloud fixtures exist. Every preset therefore ships EMPTY
allowance sets behind a PROVISIONAL PENDING FIRST LIVE CAPTURE comment;
Task 7's live probes capture real envelopes and reconcile the sets (amend,
never silent). Memory-not-evidence (fixture-unproven, recorded in the
registry comment only): Together a top-level ``prompt`` string plus
choice-level ``logprobs``; Cerebras a top-level ``time_info`` object.

The cloud fixture flip (Task 2's deferred cloud side) parametrizes over
fixture files that EXIST -- exactly like
``Tests/LLM_Calls/test_longtail_fixture_characterization.py`` -- so absent
fixtures skip cleanly and the first captured envelope replays under its
record's real (strict + empty) profile with no test change.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Iterator

import pytest

from tldw_chatbook.LLM_Calls import hosted_provider_engine
from tldw_chatbook.LLM_Calls.hosted_chat_streaming import SSERecord
from tldw_chatbook.LLM_Calls.hosted_provider_engine import (
    HostedProviderResolution,
    HostedProviderStream,
)
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


# --- allowances: empty + provisional (Task 2 fixture reality) ---


@pytest.mark.parametrize("key", PRESET_KEYS)
def test_allowances_ship_empty_pending_first_capture(key: str) -> None:
    record = RECORDS_BY_KEY[key]
    assert record.response_allowances == frozenset()
    assert record.choice_allowances == frozenset()
    assert record.message_allowances == frozenset()


def test_registry_carries_the_provisional_allowance_comment() -> None:
    """The empty sets must not be mistaken for fixture-proven cleanliness:
    the registry source itself marks them provisional so Task 7's live
    probes (amend, never silent) are discoverable from the data site."""
    registry_source = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "provider_registry.py"
    ).read_text(encoding="utf-8")
    assert "PROVISIONAL PENDING FIRST LIVE CAPTURE" in registry_source


# --- fireworks: proprietary reasoning ---


def test_fireworks_reasoning_is_proprietary() -> None:
    """Fireworks hides reasoning behind its own API surface (response
    format modes), so the engine treats any reasoning_content as
    proprietary -- never surfaced, never forwarded."""
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


# --- cloud fixture flip (Task 2's cloud side): parse under each record ---
# No cloud fixtures exist in this environment (no provider keys, Task 2
# outcome), so this parametrizes over fixture files that EXIST and skips
# cleanly today; the first captured envelope replays under its record's
# real profile -- strict parser + the record's (currently empty)
# allowances -- through the engine's own path.

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "longtail"


def load_cloud_fixtures() -> Iterator[Path]:
    """Every captured inference-cloud fixture that exists."""
    if not FIXTURE_DIR.is_dir():
        return
    for key in PRESET_KEYS:
        path = FIXTURE_DIR / f"{key}.json"
        if path.is_file():
            yield path


def _require_complete_stream(fixture: dict[str, Any]) -> None:
    """Loader-side degraded-capture guard (the longtail suite's rule)."""
    events = fixture.get("stream_events")
    if not isinstance(events, list) or not events:
        raise ValueError(
            f"fixture {fixture.get('server', '?')}: stream_events is empty"
            " -- degraded capture"
        )
    if events[-1] != "[DONE]":
        raise ValueError(
            f"fixture {fixture.get('server', '?')}: stream_events does not"
            " terminate in [DONE] -- degraded capture"
        )


def _fixture(path: Path) -> dict[str, Any]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    _require_complete_stream(fixture)
    return fixture


def _replay_under_record(
    monkeypatch: pytest.MonkeyPatch,
    record: Any,
    *,
    body: dict[str, Any] | None = None,
    stream_events: list[str] | None = None,
) -> Any:
    """Replay one captured envelope through the engine under the REAL
    record (strict parser + the record's allowances), the longtail-suite
    monkeypatch pattern: factory transport construction and both response
    wrappers, not a hand-built stream."""
    streaming = stream_events is not None
    resolution = HostedProviderResolution(
        provider=record.key,
        model="cloud-model",
        api_key="secret",
        base_url=record.default_base_url or "https://engine.invalid/v1",
        timeout=10.0,
        retries=0,
        retry_delay=0.0,
        streaming=streaming,
    )
    monkeypatch.setattr(
        hosted_provider_engine,
        "resolve_hosted_request",
        lambda _record, **_kwargs: resolution,
    )
    if streaming:
        sse_records = iter(
            [SSERecord(event=None, data=payload) for payload in stream_events]
        )
        monkeypatch.setattr(
            hosted_provider_engine, "owned_json_post", lambda **_kw: sse_records
        )
    else:
        monkeypatch.setattr(
            hosted_provider_engine, "owned_json_post", lambda **_kw: body
        )
    handler = hosted_provider_engine.build_hosted_chat_handler(record)
    return handler(
        input_data=[{"role": "user", "content": "Say ok."}],
        api_key="secret",
        streaming=streaming,
    )


@pytest.fixture(
    params=sorted(p.stem for p in load_cloud_fixtures()), ids=lambda s: s
)
def cloud_fixture(request: pytest.FixtureRequest) -> dict[str, Any]:
    return _fixture(FIXTURE_DIR / f"{request.param}.json")


def test_cloud_bodies_parse_under_their_record(
    cloud_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    record = RECORDS_BY_KEY[cloud_fixture["server"]]
    for field in ("chat_response", "tool_call_response"):
        body = cloud_fixture.get(field)
        if not isinstance(body, dict):
            continue
        result = _replay_under_record(monkeypatch, record, body=body)
        expected_reason = body["choices"][0]["finish_reason"]
        assert result["choices"][0]["finish_reason"] == expected_reason
        assert result.terminal_turn.finish_reason == expected_reason


def test_cloud_streams_parse_under_their_record(
    cloud_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    record = RECORDS_BY_KEY[cloud_fixture["server"]]
    stream = _replay_under_record(
        monkeypatch, record, stream_events=list(cloud_fixture["stream_events"])
    )
    assert isinstance(stream, HostedProviderStream)
    frames = list(stream)
    assert frames, "every captured event must replay as a visible frame"
    terminal = stream.terminal_turn
    assert terminal.finish_reason is not None
