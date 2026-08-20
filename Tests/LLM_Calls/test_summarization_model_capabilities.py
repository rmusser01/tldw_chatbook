"""Payload pins for TASK-18802: the summarization path must consult the
per-model request-capability predicates instead of unconditionally sending
sampling parameters and the classic ``max_tokens`` cap.

Provider facts pinned here were probe-verified against the real APIs:

* Anthropic (TASK-18414 probes, req_011CeB5qiXpMbYhtroLrCYVo et al.):
  ``temperature``/``top_p``/``top_k`` return HTTP 400 on the Fable 5,
  Mythos 5, Opus 5, Opus 4.8, Opus 4.7 and Sonnet 5 families; Opus 4.6,
  Sonnet 4.5 and Haiku 4.5 still accept them.
* OpenAI (TASK-18802 probes, 2026-08-20): ``max_tokens`` returns
  ``400 unsupported_parameter`` ("Use 'max_completion_tokens' instead") and
  ``temperature: 0.7`` returns ``400 unsupported_value`` on gpt-5, gpt-5.6,
  o3 and o4-mini; gpt-4o and gpt-4.1 accept the old shape unchanged.
* Anthropic combination rule (TASK-19020 probes, 2026-08-20): the families
  that still accept sampling parameters individually reject ``temperature``
  and ``top_p`` *together* -- ``claude-haiku-4-5``
  (req_011CeEDXPHNyF7apkaZepbTN), ``claude-sonnet-4-5``
  (req_011CeEDXa9V99yBoHN5vcjDG), ``claude-opus-4-6``
  (req_011CeEFGsbHd7VCjcjz4etar), ``claude-sonnet-4-6``
  (req_011CeEFGuRfeCzC6PiLyDtFb) and ``claude-opus-4-5``
  (req_011CeEFGvySC6z61NDRH5uN5) all return HTTP 400 for the trio, while
  ``temperature``+``top_k`` without ``top_p`` returns 200
  (req_011CeEDXVk4nXXCoBGdf9mFm, msg_011CeEFGzjeXQ6ftPf9KH45n). The
  function's former fallback default ``claude-3-haiku-20240307`` is RETIRED
  (404, req_011CeEDXZ8iS29MZCgyySwQa).
"""

from __future__ import annotations

import pytest

from tldw_chatbook.LLM_Calls import Summarization_General_Lib as sgl
from tldw_chatbook.model_capabilities import (
    ModelCapabilities,
    anthropic_model_rejects_temperature_top_p_combination,
    openai_model_rejects_sampling_params,
    openai_model_requires_max_completion_tokens,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _FakeOpenAIResponse:
    status_code = 200

    def raise_for_status(self) -> None:  # pragma: no cover - trivial
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "openai summary"}}]}

    def close(self) -> None:  # pragma: no cover - trivial
        return None


class _FakeAnthropicResponse:
    status_code = 200

    def json(self) -> dict:
        return {"content": [{"type": "text", "text": "anthropic summary"}]}

    def close(self) -> None:  # pragma: no cover - trivial
        return None


def _install_model_setting(monkeypatch: pytest.MonkeyPatch, section: str, model: str) -> None:
    """Route the module's config lookups: the model under test, defaults otherwise."""

    def fake_get_cli_setting(sec: str, key: str, default: object = None) -> object:
        if sec == section and key == "model":
            return model
        return default

    monkeypatch.setattr(sgl, "get_cli_setting", fake_get_cli_setting)


def _capture_openai_payload(monkeypatch: pytest.MonkeyPatch, model: str) -> dict:
    _install_model_setting(monkeypatch, "openai_api", model)
    captured: dict = {}

    def fake_session_post(self, url, headers=None, json=None, stream=False, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return _FakeOpenAIResponse()

    monkeypatch.setattr(sgl.requests.Session, "post", fake_session_post)
    result = sgl.summarize_with_openai("test-key", "some input text", "Summarize this.")
    assert result == "openai summary", result
    assert "json" in captured, "summarize_with_openai never posted a request"
    return captured["json"]


def _capture_anthropic_payload(monkeypatch: pytest.MonkeyPatch, model: str) -> dict:
    _install_model_setting(monkeypatch, "anthropic_api", model)
    captured: dict = {}

    def fake_post(url, headers=None, json=None, stream=False, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return _FakeAnthropicResponse()

    monkeypatch.setattr(sgl.requests, "post", fake_post)
    result = sgl.summarize_with_anthropic("test-key", "some input text", "Summarize this.")
    assert result == "anthropic summary", result
    assert "json" in captured, "summarize_with_anthropic never posted a request"
    return captured["json"]


# ---------------------------------------------------------------------------
# Anthropic payload pins (AC #1 / #3 / #4)
# ---------------------------------------------------------------------------

_ANTHROPIC_REJECTING_MODELS = [
    "claude-sonnet-5",  # the shipped [api_settings.anthropic] default
    "claude-opus-5",
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
]

# Served models that still accept sampling parameters individually but reject
# the ``temperature``+``top_p`` pair (TASK-19020 probes, module docstring).
_ANTHROPIC_SAMPLING_COMBO_MODELS = [
    "claude-haiku-4-5",  # the function's fallback default
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-5",
]

# Number-first 3.x-generation ids never parse into a family; the payload for
# them keeps its historical shape (the generation is retired server-side).
_ANTHROPIC_UNPARSED_LEGACY_MODELS = [
    "claude-3-haiku-20240307",  # the function's former (retired) fallback default
]


def _expected_anthropic_base_payload(model: str) -> dict:
    """The exact non-sampling payload TASK-18802 pinned, byte-for-byte."""
    return {
        "model": model,
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": "some input text \n\n\n\nSummarize this."}
        ],
        "stop_sequences": ["\n\nHuman:"],
        "metadata": {"user_id": "example_user_id"},
        "stream": False,
        "system": (
            "You are a helpful AI assistant who does whatever the user requests."
        ),
    }


@pytest.mark.parametrize("model", _ANTHROPIC_REJECTING_MODELS)
def test_anthropic_rejecting_model_omits_sampling_params(monkeypatch, model):
    payload = _capture_anthropic_payload(monkeypatch, model)
    offending = [k for k in ("temperature", "top_k", "top_p") if k in payload]
    assert offending == [], (
        f"{model} rejects sampling params but the summarization payload "
        f"still carries {offending}"
    )
    # AC #4: the whole request is byte-identical to the TASK-18802 shape.
    assert payload == _expected_anthropic_base_payload(model)


@pytest.mark.parametrize("model", _ANTHROPIC_SAMPLING_COMBO_MODELS)
def test_anthropic_sampling_model_sends_temperature_and_top_k_without_top_p(
    monkeypatch, model
):
    payload = _capture_anthropic_payload(monkeypatch, model)
    assert "top_p" not in payload, (
        f"{model} rejects temperature and top_p together (400 `temperature` "
        f"and `top_p` cannot both be specified) but the summarization payload "
        f"still carries top_p={payload.get('top_p')!r}"
    )
    # Temperature wins the pair, and top_k stays: probe-verified compatible
    # alongside temperature (req_011CeEDXVk4nXXCoBGdf9mFm on claude-haiku-4-5,
    # msg_011CeEFGzjeXQ6ftPf9KH45n on claude-opus-4-6).
    assert payload["temperature"] == pytest.approx(0.1)
    assert payload["top_k"] == 0
    assert payload == {
        **_expected_anthropic_base_payload(model),
        "temperature": 0.1,
        "top_k": 0,
    }


@pytest.mark.parametrize("model", _ANTHROPIC_UNPARSED_LEGACY_MODELS)
def test_anthropic_unparsed_legacy_id_payload_unchanged(monkeypatch, model):
    payload = _capture_anthropic_payload(monkeypatch, model)
    assert payload["temperature"] == pytest.approx(0.1)
    assert payload["top_k"] == 0
    assert payload["top_p"] == pytest.approx(1.0)


def test_anthropic_fallback_default_model_is_currently_served(monkeypatch):
    """AC #3: with no model configured, the function's own fallback default
    must resolve to a currently-served model, not the retired
    ``claude-3-haiku-20240307`` (404, req_011CeEDXZ8iS29MZCgyySwQa)."""

    def fake_get_cli_setting(sec: str, key: str, default: object = None) -> object:
        return default  # nothing configured anywhere

    monkeypatch.setattr(sgl, "get_cli_setting", fake_get_cli_setting)
    captured: dict = {}

    def fake_post(url, headers=None, json=None, stream=False, **kwargs):
        captured["json"] = json
        return _FakeAnthropicResponse()

    monkeypatch.setattr(sgl.requests, "post", fake_post)
    result = sgl.summarize_with_anthropic("test-key", "some input text", "Summarize this.")
    assert result == "anthropic summary", result
    payload = captured["json"]
    assert payload["model"] == "claude-haiku-4-5"
    # The fallback model rejects the temperature+top_p pair, so the payload it
    # gets must already respect the combination rule.
    assert "top_p" not in payload
    assert payload["temperature"] == pytest.approx(0.1)
    assert payload["top_k"] == 0


# ---------------------------------------------------------------------------
# Anthropic combination-rule predicate unit pins (TASK-19020)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        # Sampling-accepting families: probe-verified 400 on the pair.
        "claude-haiku-4-5",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-5-20251101",  # dated snapshot
        "Claude-Opus-4.6",  # dotted + case variant
        "anthropic/claude-haiku-4-5",  # provider-prefixed
        "us.anthropic.claude-sonnet-4-5",  # bedrock-prefixed
        # Families that reject sampling outright reject the pair a fortiori.
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-opus-5",
        "claude-fable-5",
        "claude-mythos-5",
    ],
)
def test_anthropic_combination_predicate_covers_claude_4_plus(model):
    assert anthropic_model_rejects_temperature_top_p_combination(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-2.1",
        "claude-instant-1.2",
        "",
        None,
        42,
    ],
)
def test_anthropic_combination_predicate_never_matches_unparsed_ids(model):
    assert anthropic_model_rejects_temperature_top_p_combination(model) is False


def test_anthropic_combination_predicate_survives_user_capability_table():
    """Request-validity fact: not reachable from the user-overridable
    capability tables (same design rule as TASK-18414/18802)."""
    ModelCapabilities(
        config={
            "models": {
                "claude-haiku-4-5": {
                    "vision": True,
                    "rejects_temperature_top_p_combination": False,
                }
            },
            "patterns": {},
        }
    )
    assert (
        anthropic_model_rejects_temperature_top_p_combination("claude-haiku-4-5")
        is True
    )


# ---------------------------------------------------------------------------
# OpenAI payload pins (AC #2 / #3 / #4)
# ---------------------------------------------------------------------------

_OPENAI_MODERN_MODELS = [
    "gpt-5",
    "gpt-5.6",
    "gpt-5.6-terra",  # the shipped chat-path default id shape
    "gpt-5-2025-08-07",
    "o3",
    "o4-mini",
]

_OPENAI_LEGACY_MODELS = [
    "gpt-4o",  # the function's own fallback default
    "gpt-4.1",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]


@pytest.mark.parametrize("model", _OPENAI_MODERN_MODELS)
def test_openai_modern_model_omits_temperature(monkeypatch, model):
    payload = _capture_openai_payload(monkeypatch, model)
    assert "temperature" not in payload, (
        f"{model} rejects non-default temperature but the summarization "
        f"payload still carries temperature={payload.get('temperature')!r}"
    )


@pytest.mark.parametrize("model", _OPENAI_MODERN_MODELS)
def test_openai_modern_model_uses_max_completion_tokens(monkeypatch, model):
    payload = _capture_openai_payload(monkeypatch, model)
    assert "max_tokens" not in payload, (
        f"{model} rejects 'max_tokens' (400 unsupported_parameter) but the "
        f"summarization payload still carries it"
    )
    assert payload.get("max_completion_tokens") == 4096, payload


@pytest.mark.parametrize("model", _OPENAI_LEGACY_MODELS)
def test_openai_legacy_model_payload_unchanged(monkeypatch, model):
    payload = _capture_openai_payload(monkeypatch, model)
    assert payload["temperature"] == pytest.approx(0.7)
    assert payload["max_tokens"] == 4096
    assert "max_completion_tokens" not in payload


# ---------------------------------------------------------------------------
# OpenAI predicate unit pins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "gpt-5.6",
        "gpt-5.1",
        "gpt-5.6-terra",
        "gpt-5-2025-08-07",
        "GPT-5",
        "openai/gpt-5",
        "openai/gpt-5.6-terra",
        "o1",
        "o1-mini",
        "o3",
        "o3-2025-04-16",
        "o4-mini",
        "o4-mini-2025-04-16",
    ],
)
def test_openai_predicates_cover_modern_family_variants(model):
    assert openai_model_rejects_sampling_params(model) is True
    assert openai_model_requires_max_completion_tokens(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "gpt-oss-120b",
        "o365-copilot",  # 'o' + digits, but not an o-series family
        "olmo-7b",
        "chatgpt-4o-latest",
        "",
        None,
        42,
    ],
)
def test_openai_predicates_never_match_legacy_or_lookalikes(model):
    assert openai_model_rejects_sampling_params(model) is False
    assert openai_model_requires_max_completion_tokens(model) is False


def test_openai_predicates_survive_a_user_configured_capability_table():
    """The predicates are request-validity facts and must not be reachable
    from the user-overridable capability tables (same design rule the
    Anthropic predicates pinned in TASK-18414)."""
    ModelCapabilities(
        config={
            "models": {"gpt-5": {"vision": True, "rejects_sampling_params": False}},
            "patterns": {},
        }
    )
    assert openai_model_rejects_sampling_params("gpt-5") is True
    assert openai_model_requires_max_completion_tokens("gpt-5") is True
