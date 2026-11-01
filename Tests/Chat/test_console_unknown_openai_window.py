"""TASK-33940.5: a fresh profile must be able to send with the shipped default.

The shipped default OpenAI model (`[api_settings.openai] model`) has no
catalog context window, and a profile that declines the online model list never
learns one. ``resolve_context_window`` then fell back to a GPT-3.5-era
``PROVIDER_CONTEXT_WINDOWS["openai"] = 4096``; with the shipped 4,096-token
response reservation and the 512-token safety margin that left no input
capacity, so every send was refused ("This request cannot fit the selected
model"). ADR-052 (TASK-32709 amendment) sets the final fallback to exactly
32,000 tokens and keeps it visibly unverified.
"""

from __future__ import annotations

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextCapacity,
    resolve_context_policy,
)
from tldw_chatbook.Chat.console_prepared_request import MINIMUM_SAFETY_MARGIN_TOKENS
from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
from tldw_chatbook.Utils.token_counter import (
    SYSTEM_CONTEXT_WINDOW,
    resolve_context_window,
)


def _shipped_openai_default() -> tuple[str, int]:
    openai = DEFAULT_CONFIG_FROM_TOML["api_settings"]["openai"]
    return str(openai["model"]), int(openai["max_tokens"])


@private_profile_test
def test_shipped_default_openai_model_has_input_capacity_in_a_fresh_profile(
    request: pytest.FixtureRequest,
) -> None:
    model, reservation = _shipped_openai_default()
    window = resolve_context_window("openai", model)
    margin = max(MINIMUM_SAFETY_MARGIN_TOKENS, window.tokens // 50)

    resolved = resolve_context_policy(
        capacity=ConsoleContextCapacity(
            model_context_window_tokens=window.tokens,
            response_reservation_tokens=reservation,
            safety_margin_tokens=margin,
            model_window_verified=window.verified,
        )
    )

    assert resolved.validation_errors == ()
    assert window.tokens > reservation + margin


@pytest.mark.parametrize("model", ["gpt-9-unlisted", "chatgpt-unlisted-preview"])
def test_an_unlisted_openai_model_gets_the_adr_052_fallback_unverified(
    model: str,
) -> None:
    window = resolve_context_window("openai", model)

    assert window.tokens == SYSTEM_CONTEXT_WINDOW
    assert window.verified is False


def test_openrouter_openai_models_inherit_the_same_fallback() -> None:
    window = resolve_context_window("openrouter", "openai/gpt-9-unlisted")

    assert window.tokens == SYSTEM_CONTEXT_WINDOW
    assert window.verified is False
