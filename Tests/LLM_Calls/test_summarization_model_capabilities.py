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

from tldw_chatbook.model_capabilities import (
    ModelCapabilities,
    anthropic_model_rejects_temperature_top_p_combination,
    openai_model_rejects_sampling_params,
    openai_model_requires_max_completion_tokens,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Anthropic payload pins (AC #1 / #3 / #4)
# ---------------------------------------------------------------------------

# Served models that still accept sampling parameters individually but reject
# the ``temperature``+``top_p`` pair (TASK-19020 probes, module docstring).
# Number-first 3.x-generation ids never parse into a family; the payload for
# them keeps its historical shape (the generation is retired server-side).
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
