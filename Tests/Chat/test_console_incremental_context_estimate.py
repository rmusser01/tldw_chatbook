"""Settled-history and draft token accounting must equal the full request."""

import pytest

from tldw_chatbook.Chat.console_session_settings import build_console_context_estimate


@pytest.mark.parametrize("model", ["gpt-4o", "custom-model"])
@pytest.mark.parametrize("history", [[], [{"role": "user", "content": "older text"}]])
@pytest.mark.parametrize("draft", ["", "new draft"])
def test_split_context_estimate_equals_full_projection(model, history, draft):
    kwargs = dict(
        provider="openai",
        model=model,
        system_prompt="system text",
        staged_text="staged evidence",
        token_limit_resolver=lambda *_: 10000,
    )
    base = build_console_context_estimate(history, **kwargs)
    full = build_console_context_estimate(
        history + ([{"role": "user", "content": draft}] if draft else []), **kwargs
    )
    incremental = build_console_context_estimate(
        ([{"role": "user", "content": draft}] if draft else []),
        provider="openai",
        model=model,
        history_used_tokens=base.used_tokens,
        token_limit_resolver=lambda *_: 10000,
    )
    assert incremental.used_tokens == full.used_tokens


def test_split_context_estimate_handles_an_empty_settled_prefix():
    full = build_console_context_estimate(
        [{"role": "user", "content": "draft"}],
        "anthropic",
        "custom-model",
        token_limit_resolver=lambda *_: 10000,
    )
    incremental = build_console_context_estimate(
        [{"role": "user", "content": "draft"}],
        "anthropic",
        "custom-model",
        history_used_tokens=0,
        token_limit_resolver=lambda *_: 10000,
    )
    assert incremental.used_tokens == full.used_tokens
