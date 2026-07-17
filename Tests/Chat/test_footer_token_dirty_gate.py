"""Footer token-count dirty gate (task-261).

The footer's 10 s interval timer used to re-run the full tokenizer over the
entire visible chat history every tick even when nothing had changed. These
tests prove `_estimate_tokens_cached` skips re-tokenizing for unchanged
inputs, recomputes for any changed input, and always returns exactly what the
real `estimate_remaining_tokens` returns (behavior unchanged). The tokenizer
spy delegates to the REAL implementation — no fakes.
"""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Event_Handlers.Chat_Events import chat_token_events
from tldw_chatbook.Utils.token_counter import estimate_remaining_tokens

SETTINGS = dict(
    model="gpt-3.5-turbo",
    provider="openai",
    max_tokens_response=2048,
    system_prompt="Be concise.",
)


@pytest.fixture
def tokenizer_spy(monkeypatch):
    """Count calls into the real tokenizer without changing its behavior."""
    calls: list = []
    real = chat_token_events.estimate_remaining_tokens

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(chat_token_events, "estimate_remaining_tokens", spy)
    return calls


def _history():
    return [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi, how can I help?"},
    ]


def test_unchanged_history_skips_retokenizing(tokenizer_spy):
    app = SimpleNamespace()
    history = _history()

    first = chat_token_events._estimate_tokens_cached(app, history, **SETTINGS)
    second = chat_token_events._estimate_tokens_cached(app, history, **SETTINGS)

    assert len(tokenizer_spy) == 1, "unchanged inputs must not re-tokenize"
    assert second == first


def test_cached_result_matches_direct_tokenizer_output(tokenizer_spy):
    app = SimpleNamespace()
    history = _history()

    cached = chat_token_events._estimate_tokens_cached(app, history, **SETTINGS)
    direct = estimate_remaining_tokens(history, **SETTINGS)

    assert cached == direct


def test_history_growth_retokenizes(tokenizer_spy):
    app = SimpleNamespace()
    history = _history()

    chat_token_events._estimate_tokens_cached(app, history, **SETTINGS)
    history.append({"role": "user", "content": "one more question"})
    grown = chat_token_events._estimate_tokens_cached(app, history, **SETTINGS)

    assert len(tokenizer_spy) == 2, "a new message must re-tokenize"
    assert grown == estimate_remaining_tokens(history, **SETTINGS)


def test_last_message_edit_retokenizes(tokenizer_spy):
    app = SimpleNamespace()
    history = _history()

    chat_token_events._estimate_tokens_cached(app, history, **SETTINGS)
    history[-1] = {"role": "assistant", "content": "a completely different, much longer reply"}
    edited = chat_token_events._estimate_tokens_cached(app, history, **SETTINGS)

    assert len(tokenizer_spy) == 2, "an edited last message must re-tokenize"
    assert edited == estimate_remaining_tokens(history, **SETTINGS)


def test_settings_change_retokenizes(tokenizer_spy):
    app = SimpleNamespace()
    history = _history()

    chat_token_events._estimate_tokens_cached(app, history, **SETTINGS)
    changed = dict(SETTINGS, system_prompt="Answer in French.")
    chat_token_events._estimate_tokens_cached(app, history, **changed)

    assert len(tokenizer_spy) == 2, "a settings change must re-tokenize"


def test_cache_is_per_app_instance(tokenizer_spy):
    history = _history()

    chat_token_events._estimate_tokens_cached(SimpleNamespace(), history, **SETTINGS)
    chat_token_events._estimate_tokens_cached(SimpleNamespace(), history, **SETTINGS)

    assert len(tokenizer_spy) == 2, "each app instance keeps its own cache"


def test_attribute_rejecting_app_still_returns_counts(tokenizer_spy):
    """A frozen/slotted app double must degrade to compute-every-time."""

    class Frozen:
        __slots__ = ()

    app = Frozen()
    history = _history()

    result = chat_token_events._estimate_tokens_cached(app, history, **SETTINGS)

    assert result == estimate_remaining_tokens(history, **SETTINGS)
