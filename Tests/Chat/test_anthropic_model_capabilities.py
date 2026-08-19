"""TASK-18414: per-model Anthropic request capabilities must be capability
predicates, not hand-maintained name checks in the request builder.

Every expectation below is pinned to a *live-observed* response from
api.anthropic.com (captured 2026-08-18 with a real key, recorded verbatim in
`Docs/superpowers/plans/2026-08-18-task-18414-report.md`):

  claude-opus-5   + temperature -> 400 "`temperature` is deprecated for this model."
  claude-opus-5   + top_p       -> 400 "`top_p` is deprecated for this model."
  claude-opus-5   + top_k       -> 400 "`top_k` is deprecated for this model."
  claude-opus-5   + budget      -> 400 '"thinking.type.enabled" is not supported
                                        for this model. Use "thinking.type.adaptive"
                                        and "output_config.effort" ...'
  claude-opus-4-8 + temperature -> 400 (same message)
  claude-opus-4-7 + temperature -> 400 (same message)
  claude-fable-5  + temperature -> 400 (same message)
  claude-sonnet-5 + temperature -> 400 (same message)
  claude-opus-4-6 + temperature -> 200   <- must stay unchanged (AC #6)
  claude-opus-4-6 + budget      -> 200   <- must stay unchanged (AC #6)
  claude-sonnet-4-5 + temperature -> 200 <- must stay unchanged (AC #6)
  claude-haiku-4-5  + temperature + top_k -> 200 <- must stay unchanged (AC #6)

The harness mirrors `Tests/Chat/test_anthropic_native_tools.py`: patch
`requests.Session.post`, drive the real dispatcher through `chat_api_call`, and
inspect the JSON body actually put on the wire.
"""

import pytest
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.model_capabilities import (
    anthropic_model_rejects_fixed_thinking_budget,
    anthropic_model_rejects_sampling_params,
)

SAMPLING_KEYS = ("temperature", "top_p", "top_k")

# Families the live API rejects sampling parameters / a fixed thinking budget on.
NO_SAMPLING_MODELS = [
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
]

# Families that still accept both; behaviour here must not change (AC #6).
LEGACY_SAMPLING_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229",
]


def _anthropic_text_response():
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _sent_payload(mock_post, model, **extra):
    """Drive the real dispatch path and return the JSON body actually posted."""
    mock_response = Mock()
    mock_response.json.return_value = _anthropic_text_response()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    chat_api_call(
        "anthropic",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model=model,
        streaming=False,
        temp=0.7,
        **extra,
    )
    return mock_post.call_args[1]["json"]


# ---------------------------------------------------------------------------
# (a) A no-sampling model with the thinking effort UNSET.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", NO_SAMPLING_MODELS)
@patch("requests.Session.post")
def test_no_sampling_model_omits_sampling_when_effort_unset(mock_post, model):
    """AC #3 / #5: the sampling parameters the live API rejects are never sent,
    including when no thinking effort is configured (the branch that reopened
    the temperature path for Opus 4.8/4.7)."""
    sent = _sent_payload(mock_post, model)
    present = [key for key in SAMPLING_KEYS if key in sent]
    assert present == [], f"{model} must not send {present}"


# ---------------------------------------------------------------------------
# (b) The same models with a thinking effort SET.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", NO_SAMPLING_MODELS)
@patch("requests.Session.post")
def test_no_sampling_model_omits_sampling_when_effort_set(mock_post, model):
    """AC #3: an effort must not reopen the sampling branch either."""
    sent = _sent_payload(mock_post, model, thinking_effort="high")
    present = [key for key in SAMPLING_KEYS if key in sent]
    assert present == [], f"{model} must not send {present}"


@pytest.mark.parametrize("model", NO_SAMPLING_MODELS)
@patch("requests.Session.post")
def test_no_sampling_model_never_sends_fixed_thinking_budget(mock_post, model):
    """AC #4: `thinking.type == "enabled"` with `budget_tokens` is rejected on
    every one of these families -- with an effort set, with an explicit budget
    set, and with neither."""
    for extra in (
        {"thinking_effort": "high"},
        {"thinking_budget_tokens": 4096},
        {"thinking_effort": "medium", "thinking_budget_tokens": 4096},
        {},
    ):
        sent = _sent_payload(mock_post, model, **extra)
        thinking = sent.get("thinking")
        if thinking is not None:
            assert thinking.get("type") != "enabled", (
                f"{model} with {extra} sent a legacy thinking config: {thinking}"
            )
            assert "budget_tokens" not in thinking, (
                f"{model} with {extra} sent budget_tokens: {thinking}"
            )


@patch("requests.Session.post")
def test_opus_5_with_effort_sends_adaptive_thinking_and_effort(mock_post):
    """AC #2 wire shape: live-verified as HTTP 200 (probe M in the report)."""
    sent = _sent_payload(mock_post, "claude-opus-5", thinking_effort="high")
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}


@patch("requests.Session.post")
def test_opus_5_with_effort_unset_sends_neither_thinking_nor_sampling(mock_post):
    """AC #1 wire shape: live-verified as HTTP 200 (probe L in the report).
    Opus 5 runs adaptive thinking by default when `thinking` is omitted."""
    sent = _sent_payload(mock_post, "claude-opus-5")
    assert "thinking" not in sent
    assert not any(key in sent for key in SAMPLING_KEYS)


# ---------------------------------------------------------------------------
# (c) Opus 4.8 / 4.7 specifically, with no effort configured.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-opus-4-7"])
@patch("requests.Session.post")
def test_opus_4_8_and_4_7_omit_sampling_with_no_effort(mock_post, model):
    """AC #5: these are in the adaptive-thinking set, but with no effort the
    mapper returns no thinking config -- which used to reopen the sampling
    branch and put `temperature` on a request the API rejects."""
    sent = _sent_payload(mock_post, model)
    assert "thinking" not in sent, "no effort configured -> no thinking config"
    present = [key for key in SAMPLING_KEYS if key in sent]
    assert present == [], f"{model} must not send {present}"


# ---------------------------------------------------------------------------
# (d) Legacy models that must still receive temperature (AC #6).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", LEGACY_SAMPLING_MODELS)
@patch("requests.Session.post")
def test_legacy_models_still_receive_temperature(mock_post, model):
    """AC #6 regression pin: Opus 4.6 and earlier, Sonnet 4.6/4.5 and Haiku
    still accept sampling parameters live, so they must still get them."""
    sent = _sent_payload(mock_post, model)
    assert sent.get("temperature") == 0.7, f"{model} lost its temperature"


@patch("requests.Session.post")
def test_legacy_model_still_receives_fixed_thinking_budget(mock_post):
    """AC #6 regression pin: `budget_tokens` is live-verified 200 on Opus 4.6."""
    sent = _sent_payload(
        mock_post, "claude-opus-4-6", thinking_effort="high", thinking_budget_tokens=4096
    )
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    # thinking enabled -> sampling suppressed, as before this change.
    assert not any(key in sent for key in SAMPLING_KEYS)


@patch("requests.Session.post")
def test_sonnet_4_6_still_uses_adaptive_without_a_capability_flag(mock_post):
    """Sonnet 4.6 *prefers* adaptive thinking but does not reject a fixed
    budget -- it stays on the preference marker list, not the capability set."""
    assert anthropic_model_rejects_fixed_thinking_budget("claude-sonnet-4-6") is False
    sent = _sent_payload(mock_post, "claude-sonnet-4-6", thinking_effort="high")
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}


# ---------------------------------------------------------------------------
# The predicates themselves: family matching must be robust without
# over-matching older families.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        # bare ids
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
        # dotted variants
        "claude-opus-4.8",
        "claude-opus-4.7",
        # dated / suffixed variants
        "claude-opus-5-20260101",
        "claude-opus-4-8-20260101",
        "claude-opus-4-8-fast",
        "claude-opus-5[1m]",
        # provider-prefixed forms the codebase passes through
        "anthropic/claude-opus-5",
        "anthropic.claude-opus-5",
        "us.anthropic.claude-opus-4-8",
        "claude-sonnet-5@20260101",
        # case insensitivity
        "Claude-Opus-5",
    ],
)
def test_predicates_match_the_whole_family(model):
    assert anthropic_model_rejects_sampling_params(model) is True, model
    assert anthropic_model_rejects_fixed_thinking_budget(model) is True, model


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-opus-4-5-20251101",
        "claude-opus-4-1",
        "claude-opus-4-0",
        "claude-opus-4-20250514",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "anthropic/claude-opus-4-6",
        "gpt-5.6",
        "",
        None,
    ],
)
def test_predicates_do_not_over_match(model):
    assert anthropic_model_rejects_sampling_params(model) is False, model
    assert anthropic_model_rejects_fixed_thinking_budget(model) is False, model


def test_predicates_survive_a_user_configured_capability_table():
    """The direct-mapping / pattern tables in `model_capabilities` are wholly
    replaceable from `config.toml`, and `claude-sonnet-5` already has a direct
    mapping that would shadow any Anthropic pattern. A provider *request
    validity* fact must therefore not be reachable from user config -- a user
    edit could only turn it off and reintroduce the 400."""
    from tldw_chatbook.model_capabilities import ModelCapabilities

    caps = ModelCapabilities(config={"models": {}, "patterns": {}})
    assert caps.get_model_capabilities("Anthropic", "claude-opus-5") is not None
    # Predicates are unaffected by an emptied capability table.
    assert anthropic_model_rejects_sampling_params("claude-opus-5") is True
    assert anthropic_model_rejects_fixed_thinking_budget("claude-opus-5") is True
