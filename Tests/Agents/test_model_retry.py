"""Transient provider failures must not discard a whole agent run.

TASK-25901. `run_agent_loop` turned any `call_model` exception into a
STEP_MODEL_ERROR trace and re-raised, which `agent_service` reported as
RUN_ERROR -- so a single 429 or dropped connection threw away the run along with
every tool result already in it. A named grep for `fallback_model`,
`retry_model`, `max_retries` and `backoff` across `Agents/` returned zero.

Classification is by exception type, not by string matching: retrying an auth
failure or a content-policy refusal wastes the user's money and their time, and
"terminal" has to stay terminal.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_chatbook.Agents.model_retry import (
    RetryPolicy,
    is_transient_model_error,
    retry_delay_seconds,
)


TRANSIENT = [
    pytest.param(ChatRateLimitError("slow down"), id="429-rate-limit"),
    pytest.param(ChatProviderError("bad gateway", status_code=502), id="502"),
    pytest.param(ChatProviderError("unavailable", status_code=503), id="503"),
    pytest.param(ChatProviderError("gateway timeout", status_code=504), id="504"),
    pytest.param(ConnectionResetError("reset by peer"), id="connection-reset"),
    pytest.param(TimeoutError("read timed out"), id="read-timeout"),
    pytest.param(ConnectionError("dropped"), id="connection-error"),
]

TERMINAL = [
    pytest.param(ChatAuthenticationError("bad key"), id="401-auth"),
    pytest.param(ChatBadRequestError("malformed"), id="400-bad-request"),
    pytest.param(ChatConfigurationError("no model configured"), id="config"),
    pytest.param(ChatProviderError("refused", status_code=403), id="403"),
    pytest.param(ChatProviderError("not found", status_code=404), id="404"),
    pytest.param(ValueError("a bug in our own code"), id="programming-error"),
    pytest.param(KeyError("missing"), id="key-error"),
]


@pytest.mark.parametrize("exc", TRANSIENT)
def test_transient_errors_are_retryable(exc):
    assert is_transient_model_error(exc) is True


@pytest.mark.parametrize("exc", TERMINAL)
def test_terminal_errors_are_not_retryable(exc):
    """Retrying these burns money and time and cannot succeed."""
    assert is_transient_model_error(exc) is False


def test_an_unknown_exception_is_treated_as_terminal():
    """Fail closed: an error we cannot classify must not be retried blindly."""

    class _Weird(Exception):
        pass

    assert is_transient_model_error(_Weird("?")) is False


def test_backoff_grows_with_each_attempt():
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=60.0)
    delays = [
        retry_delay_seconds(attempt, None, policy, jitter=lambda: 1.0)
        for attempt in range(1, 5)
    ]

    assert delays == sorted(delays)
    assert delays[0] < delays[-1]


def test_backoff_is_capped():
    policy = RetryPolicy(max_attempts=20, base_delay=1.0, max_delay=5.0)

    for attempt in range(1, 20):
        assert (
            retry_delay_seconds(attempt, None, policy, jitter=lambda: 1.0) <= 5.0
        )


def test_jitter_is_applied():
    """Without jitter, every client retries in lockstep and re-stampedes."""
    policy = RetryPolicy(max_attempts=5, base_delay=10.0, max_delay=60.0)

    low = retry_delay_seconds(2, None, policy, jitter=lambda: 0.0)
    high = retry_delay_seconds(2, None, policy, jitter=lambda: 1.0)

    assert low < high


def test_retry_after_header_is_honoured_over_backoff():
    """AC#2: the server told us when to come back."""
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=60.0)
    exc = ChatRateLimitError("slow down")
    exc.retry_after = 12.0

    assert retry_delay_seconds(1, exc, policy, jitter=lambda: 1.0) == 12.0


def test_retry_after_is_still_capped():
    """A hostile or broken server must not park the run for an hour."""
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=30.0)
    exc = ChatRateLimitError("slow down")
    exc.retry_after = 3600.0

    assert retry_delay_seconds(1, exc, policy, jitter=lambda: 1.0) == 30.0


@pytest.mark.parametrize("bad", ["soon", -5, None, float("nan")])
def test_unusable_retry_after_falls_back_to_backoff(bad):
    policy = RetryPolicy(max_attempts=5, base_delay=2.0, max_delay=60.0)
    exc = ChatRateLimitError("slow down")
    exc.retry_after = bad

    delay = retry_delay_seconds(1, exc, policy, jitter=lambda: 1.0)

    assert delay > 0
