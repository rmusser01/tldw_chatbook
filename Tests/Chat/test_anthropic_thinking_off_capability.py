"""TASK-18800: Console thinking-effort `off` must actually disable thinking on
the Anthropic families that think BY DEFAULT when `thinking` is omitted -- and
must never send the explicit disabled config to the families that 400 on it.

Every expectation below is pinned to a *live-observed* response from
api.anthropic.com (captured 2026-08-20 with a real key, recorded verbatim in
`Docs/superpowers/plans/2026-08-21-task-18800-report.md`):

  claude-opus-5  + thinking={"type":"disabled"}, no effort -> 200
                   (thinking_tokens 0, no thinking block)
  claude-opus-5  , thinking omitted -> 200 WITH a thinking block and 13
                   billed thinking tokens (the silent pre-fix defect)
  claude-fable-5 + thinking={"type":"disabled"} -> 400
                   '"thinking.type.disabled" is not supported for this model.'
  claude-fable-5 , thinking omitted -> 200 (always-on: thinking block present)
  claude-sonnet-5 + disabled -> 200          <- must stay unchanged
  claude-sonnet-4-6 / claude-haiku-4-5, omitted -> 200 with NO thinking block
                   (omission genuinely means off on the legacy families, AC #4)
  claude-mythos-5 -> 404 on this key (Project Glasswing-only); handled on the
                   documented grounds that it shares Fable 5's surface exactly.

The harness mirrors `Tests/Chat/test_anthropic_model_capabilities.py`: patch
`requests.Session.post`, drive the real dispatcher through `chat_api_call`, and
inspect the JSON body actually put on the wire.
"""

import pytest
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.model_capabilities import (
    anthropic_model_rejects_disabled_thinking,
    anthropic_model_thinks_by_default,
)

SAMPLING_KEYS = ("temperature", "top_p", "top_k")

# Families where thinking runs by default AND the explicit disabled config is
# accepted: OFF must be expressed as thinking={"type": "disabled"}.
EXPLICIT_DISABLED_MODELS = [
    "claude-opus-5",
    "claude-sonnet-5",
]

# Always-on families: the disabled config is a live-observed 400, omission is
# the only valid move (and thinking still runs -- surfaced via the Console
# settings warning, see test_console_session_settings.py).
ALWAYS_ON_MODELS = [
    "claude-fable-5",
    "claude-mythos-5",
]

# Families where omitting `thinking` already means no thinking (AC #4).
OMISSION_IS_OFF_MODELS = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022",
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
# (a) OFF on the thinks-by-default families that accept the disabled config.
#     RED before the fix for claude-opus-5: the old branch sent no thinking
#     key at all, which on Opus 5 silently leaves adaptive thinking running.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", EXPLICIT_DISABLED_MODELS)
@patch("requests.Session.post")
def test_off_sends_explicit_disabled_thinking(mock_post, model):
    """AC #1: `off` must put thinking={"type": "disabled"} on the wire --
    live-verified 200 with zero thinking tokens (probes 1 and 7)."""
    sent = _sent_payload(mock_post, model, thinking_effort="off")
    assert sent.get("thinking") == {"type": "disabled"}, (
        f"{model} with thinking_effort='off' sent thinking={sent.get('thinking')!r}"
    )
    # The Opus 5 effort cap on disabled thinking binds only at xhigh/max
    # (req_011CeFGfT1wJxmsd2rRUszbc): OFF must never carry an effort.
    assert "output_config" not in sent
    assert not any(key in sent for key in SAMPLING_KEYS)


@patch("requests.Session.post")
def test_opus_5_off_ignores_a_stray_thinking_budget(mock_post):
    """A leftover budget must not turn OFF into a legacy enabled config."""
    sent = _sent_payload(
        mock_post, "claude-opus-5", thinking_effort="off", thinking_budget_tokens=4096
    )
    assert sent.get("thinking") == {"type": "disabled"}


# ---------------------------------------------------------------------------
# (b) OFF on the always-on families. The disabled config is a live-observed
#     400 here (req_011CeFGfU3CpiKFwRigU2jRa), so the only valid payload is
#     omission. This PASSES pre-fix (the old branch already omitted) -- it is
#     a CONTROL against naively widening the disabled branch, not a red pin;
#     the mutation runs below turn it red.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ALWAYS_ON_MODELS
    + [
        "claude-fable-5-20260101",
        "anthropic/claude-fable-5",
        "claude-mythos-5[1m]",
    ],
)
@patch("requests.Session.post")
def test_always_on_model_off_sends_no_thinking_key(mock_post, model):
    """AC #2: no thinking key, no effort, no sampling -- the exact payload
    live-verified 200 on claude-fable-5 (probe 5)."""
    sent = _sent_payload(mock_post, model, thinking_effort="off")
    assert "thinking" not in sent, (
        f"{model} with thinking_effort='off' sent thinking={sent.get('thinking')!r} "
        "-- an explicit config is a 400 on this family"
    )
    assert "output_config" not in sent
    assert not any(key in sent for key in SAMPLING_KEYS)


# ---------------------------------------------------------------------------
# (c) OFF on the families where omission already means no thinking (AC #4).
#     These pass pre-fix and must keep passing: regression pins.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", OMISSION_IS_OFF_MODELS)
@patch("requests.Session.post")
def test_omission_is_off_family_off_sends_no_thinking_key(mock_post, model):
    """AC #4: Opus 4.8 and earlier, Sonnet 4.6 and earlier, and Haiku are
    unchanged -- `off` still means simply no thinking config."""
    sent = _sent_payload(mock_post, model, thinking_effort="off")
    assert "thinking" not in sent, f"{model} must not gain a thinking config"
    assert "output_config" not in sent


@pytest.mark.parametrize(
    "model", ["claude-opus-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"]
)
@patch("requests.Session.post")
def test_legacy_sampling_model_off_still_receives_temperature(mock_post, model):
    """AC #4: on the families that still accept sampling parameters, `off`
    leaves the sampling branch exactly as it was."""
    sent = _sent_payload(mock_post, model, thinking_effort="off")
    assert sent.get("temperature") == 0.7, f"{model} lost its temperature"


# ---------------------------------------------------------------------------
# The predicates themselves (AC #3): capability facts, not name checks in the
# request builder; family matching must be robust without over-matching.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
        # dated / suffixed / prefixed variants
        "claude-opus-5-20260101",
        "claude-sonnet-5@20260101",
        "claude-fable-5[1m]",
        "anthropic/claude-opus-5",
        "us.anthropic.claude-fable-5",
        "Claude-Fable-5",
    ],
)
def test_thinks_by_default_matches_the_whole_family(model):
    assert anthropic_model_thinks_by_default(model) is True, model


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-3-5-sonnet-20241022",
        "gpt-5.6",
        "",
        None,
    ],
)
def test_thinks_by_default_does_not_over_match(model):
    assert anthropic_model_thinks_by_default(model) is False, model


@pytest.mark.parametrize(
    "model",
    [
        "claude-fable-5",
        "claude-mythos-5",
        "claude-fable-5-20260101",
        "claude-mythos-5[1m]",
        "anthropic/claude-fable-5",
        "Claude-Mythos-5",
    ],
)
def test_rejects_disabled_thinking_matches_the_always_on_family(model):
    assert anthropic_model_rejects_disabled_thinking(model) is True, model


@pytest.mark.parametrize(
    "model",
    [
        # Opus 5 ACCEPTS the disabled config (probe 1) -- widening the
        # always-on set to include it would wrongly drop the disabled config
        # OFF depends on.
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "gpt-5.6",
        "",
        None,
    ],
)
def test_rejects_disabled_thinking_does_not_over_match(model):
    assert anthropic_model_rejects_disabled_thinking(model) is False, model


def test_every_always_on_model_also_thinks_by_default():
    """Consistency of the current table: rejecting `disabled` only makes sense
    on a model whose omission runs thinking. A future model may break this --
    the predicates are deliberately independent -- but today's table must not
    break it by accident."""
    for model in ALWAYS_ON_MODELS:
        assert anthropic_model_thinks_by_default(model) is True, model


def test_predicates_survive_a_user_configured_capability_table():
    """Same guarantee as TASK-18414's predicates: the config-driven capability
    tables are wholly replaceable from `config.toml`, and a user edit to a
    request-validity fact could only reintroduce the 400 / the silent billing."""
    from tldw_chatbook.model_capabilities import ModelCapabilities

    caps = ModelCapabilities(config={"models": {}, "patterns": {}})
    assert caps.get_model_capabilities("Anthropic", "claude-fable-5") is not None
    assert anthropic_model_thinks_by_default("claude-opus-5") is True
    assert anthropic_model_rejects_disabled_thinking("claude-fable-5") is True
