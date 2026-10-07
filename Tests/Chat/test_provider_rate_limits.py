"""Remaining provider rate limit in the Console cost tooltip (TASK-28229).

The hook on ``create_default_session`` keeps a provider's rate-limit headers
only while the Console gateway marks a provider call; the cost tooltip shows
them as its last info line. A provider that sends none changes nothing.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pytest
import requests

from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderSelection,
)
from tldw_chatbook.Chat.provider_rate_limits import (
    format_rate_limit_line,
    parse_rate_limit_headers,
)
from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
    console_rate_limit_line,
)
from tldw_chatbook.Utils import egress
from tldw_chatbook.Utils.egress import (
    capture_rate_limits_for,
    create_default_session,
    latest_rate_limit_headers,
)

# create_default_session reads timeout and TLS settings through the guarded
# config loader (same admission signature as test_hosted_chat.py in
# Tests/conftest.py), so keep the bootstrap profile.
pytestmark = pytest.mark.bootstrap_profile

T0 = 1_800_000_000.0
OPENAI_HEADERS = {
    "x-ratelimit-limit-requests": "5000",
    "x-ratelimit-remaining-requests": "4999",
    "x-ratelimit-reset-requests": "48s",
    "x-ratelimit-limit-tokens": "40000",
    "x-ratelimit-remaining-tokens": "39200",
    "x-ratelimit-reset-tokens": "6m0s",
}


@pytest.fixture(autouse=True)
def _empty_store() -> Any:
    """Start and end every test with no recorded rate limits."""
    egress._LATEST_RATE_LIMITS.clear()
    yield
    egress._LATEST_RATE_LIMITS.clear()


class _HeaderAdapter(requests.adapters.BaseAdapter):
    """Answers every request with fixed headers; never touches the network."""

    def __init__(self, headers: dict[str, str]) -> None:
        super().__init__()
        self.headers = headers

    def send(self, request: Any, **_kwargs: Any) -> requests.Response:
        response = requests.Response()
        response.status_code = 200
        response.headers.update(self.headers)
        response._content = b'{"ok": true}'
        response.request = request
        response.url = request.url
        return response

    def close(self) -> None:
        """Nothing to release."""


def _post_with(headers: dict[str, str]) -> None:
    session = create_default_session()
    session.mount("https://", _HeaderAdapter(headers))
    session.post("https://api.example.invalid/v1/chat/completions", json={})


def _clock(at: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(at))


# -- parsing -----------------------------------------------------------------


def test_openai_style_headers_give_one_window_per_metric() -> None:
    """Remaining, limit and a Go-style duration reset, requests first."""
    requests_window, tokens_window = parse_rate_limit_headers(OPENAI_HEADERS, T0)
    assert (requests_window.metric, requests_window.remaining, requests_window.limit) == (
        "requests", 4999, 5000,
    )
    assert requests_window.reset_at == T0 + 48
    assert (tokens_window.metric, tokens_window.remaining, tokens_window.limit) == (
        "tokens", 39200, 40000,
    )
    assert tokens_window.reset_at == T0 + 360


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("20ms", 0.02), ("2m59.56s", 179.56), ("1h2m", 3720), ("1s", 1), ("7", 7)],
)
def test_reset_durations(value: str, seconds: float) -> None:
    """Durations and plain seconds count from the response time.

    Args:
        value: A reset header value.
        seconds: The seconds it means.
    """
    (window,) = parse_rate_limit_headers(
        {"x-ratelimit-remaining-requests": "1", "x-ratelimit-reset-requests": value}, T0
    )
    assert window.reset_at == pytest.approx(T0 + seconds)


def test_anthropic_headers_and_rfc3339_resets() -> None:
    """Anthropic names its metrics differently and resets at a timestamp."""
    at = datetime(2027, 1, 15, 8, 0, tzinfo=timezone.utc).timestamp()
    windows = parse_rate_limit_headers(
        {
            "anthropic-ratelimit-requests-limit": "50",
            "anthropic-ratelimit-requests-remaining": "49",
            "anthropic-ratelimit-requests-reset": "2027-01-15T08:00:30Z",
            "anthropic-ratelimit-input-tokens-remaining": "29000",
        },
        at,
    )
    assert [(w.metric, w.remaining) for w in windows] == [
        ("requests", 49), ("input-tokens", 29000),
    ]
    assert windows[0].reset_at == at + 30


def test_epoch_resets_and_windowed_headers() -> None:
    """Bare epoch seconds/ms resets; a per-day header keeps its period."""
    (bare,) = parse_rate_limit_headers(
        {"X-RateLimit-Remaining": "9", "X-RateLimit-Reset": str(int((T0 + 60) * 1000))}, T0
    )
    assert (bare.metric, bare.remaining, bare.reset_at) == ("requests", 9, T0 + 60)
    (daily,) = parse_rate_limit_headers(
        {
            "x-ratelimit-remaining-requests-day": "14000",
            "x-ratelimit-limit-requests-day": "14400",
            "x-ratelimit-reset-requests-day": str(T0 + 3600),
        },
        T0,
    )
    assert (daily.window, daily.remaining, daily.limit, daily.reset_at) == (
        "day", 14000, 14400, T0 + 3600,
    )


def test_unusable_values_are_dropped_not_guessed() -> None:
    """Text, negatives, far-off resets and limit-only metrics show nothing."""
    assert parse_rate_limit_headers(
        {
            "x-ratelimit-remaining-requests": "<b>lots</b>",
            "x-ratelimit-remaining-tokens": "-1",
            "x-ratelimit-limit-requests-day": "100",  # no remaining
            "x-ratelimit-remaining-foo": "5",  # unknown period
        },
        T0,
    ) == ()
    (window,) = parse_rate_limit_headers(
        {"x-ratelimit-remaining-requests": "3", "x-ratelimit-reset-requests": "99999999"},
        T0,
    )
    assert window.reset_at is None


# -- the tooltip line -------------------------------------------------------------


def test_line_names_both_budgets_with_clock_times() -> None:
    """Absolute clock times: the tooltip is rebuilt on refresh, not on hover."""
    assert format_rate_limit_line(OPENAI_HEADERS, T0) == (
        f"Rate limit at {_clock(T0)}: 4,999/5,000 requests left (resets {_clock(T0 + 48)})"
        f" · 39,200/40,000 tokens left (resets {_clock(T0 + 360)})"
    )


def test_no_remaining_value_means_no_line() -> None:
    """No fake zeros: a provider that reports no remaining budget shows nothing."""
    assert format_rate_limit_line({"x-ratelimit-limit-requests": "5000"}, T0) is None


# -- capture ------------------------------------------------------------------


def test_a_marked_call_records_only_rate_limit_headers() -> None:
    """Inside the gateway's scope the hook keeps rate-limit headers, nothing else."""
    with capture_rate_limits_for("openai"):
        _post_with({**OPENAI_HEADERS, "set-cookie": "session=SECRET", "x-request-id": "r1"})
    captured_at, headers = latest_rate_limit_headers("openai")
    assert headers == OPENAI_HEADERS
    assert captured_at == pytest.approx(time.time(), abs=60)


def test_requests_outside_a_provider_call_are_not_recorded() -> None:
    """Every other caller of create_default_session behaves as before."""
    _post_with(OPENAI_HEADERS)
    assert latest_rate_limit_headers("openai") is None
    assert console_rate_limit_line("openai") is None


def test_a_reply_without_headers_keeps_the_last_reading() -> None:
    """A provider that stops sending them keeps the last time-stamped values."""
    with capture_rate_limits_for("openai"):
        _post_with(OPENAI_HEADERS)
        _post_with({"content-type": "application/json"})
    assert latest_rate_limit_headers("openai")[1] == OPENAI_HEADERS


@pytest.mark.asyncio
@pytest.mark.parametrize("lazy", [False, True])
async def test_the_console_stream_path_records_the_provider_headers(lazy: bool) -> None:
    """Through ``stream_chat``, including a stream that requests lazily.

    Args:
        lazy: Whether the adapter returns a generator that sends on iteration.
    """

    def adapter(**_kwargs: Any) -> Any:
        if not lazy:
            _post_with(OPENAI_HEADERS)
            return "done"

        def stream() -> Any:
            _post_with(OPENAI_HEADERS)
            yield "done"

        return stream()

    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": {"openai": {"api_key": "sk-test"}}},
        chat_api_call_fn=adapter,
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1", streaming=lazy)
    )
    _ = [chunk async for chunk in gateway.stream_chat(resolution, [{"role": "user", "content": "hi"}])]
    line = console_rate_limit_line("openai")
    assert line is not None and "4,999/5,000 requests left" in line


@pytest.mark.asyncio
async def test_the_auxiliary_completion_path_records_the_provider_headers() -> None:
    """Side calls (titles, rewrites) spend the same budget, so they count too."""
    from Tests.Chat.test_console_provider_gateway import _auxiliary_request

    def adapter(**_kwargs: Any) -> Any:
        _post_with(OPENAI_HEADERS)
        return {"choices": [{"message": {"content": "ok"}}]}

    gateway = ConsoleProviderGateway(chat_api_call_fn=adapter)
    assert (await gateway.complete_auxiliary(_auxiliary_request())).text == "ok"
    assert latest_rate_limit_headers("openai") is not None


# -- placement in the tooltip ---------------------------------------------------


def _spend_state(rate_limit_line: str | None) -> Any:
    from tldw_chatbook.Chat.console_cost_tracker import ConsoleCacheState, ConsoleCostSnapshot
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSessionSettings,
        ConsoleSettingsContextEstimate,
    )
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        build_console_spend_cost_state,
    )
    from tldw_chatbook.Widgets.Console.console_context_controls import (
        build_console_context_control_state,
    )

    context = build_console_context_control_state(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-4.1", max_tokens=1_000),
        estimate=ConsoleSettingsContextEstimate(used_tokens=1_000, token_limit=10_000, label="context"),
    )
    return build_console_spend_cost_state(
        ConsoleCostSnapshot(0.01, 1_000, True, False, 1),
        ConsoleCacheState.NONE, None, None, None, None, True, context,
        False, False, 3.0, "draft", rate_limit_line,
    )


def test_the_line_is_the_last_info_line_above_the_inspector_hint() -> None:
    """Owner placement: bottom of the info stack, after the next-send estimate."""
    line = "Rate limit at 14:32:05: 49/50 requests left"
    lines = _spend_state(line).tooltip.splitlines()
    assert lines[-1].startswith("Open Conversation Inspector")
    assert lines[-2] == line
    assert any(text.startswith("On next send") for text in lines[:-2])


def test_without_a_reading_the_tooltip_is_unchanged() -> None:
    """A provider that sends no rate-limit headers behaves exactly as before."""
    line = "Rate limit at 14:32:05: 49/50 requests left"
    without = _spend_state(None).tooltip
    assert "Rate limit" not in without
    assert _spend_state(line).tooltip.replace(f"\n{line}", "") == without


# -- Qodo #2981 -------------------------------------------------------------------


def test_values_are_bounded_by_the_validation_types() -> None:
    """Out-of-range counts and non-finite resets are dropped (Qodo #2981, 1)."""
    (window,) = parse_rate_limit_headers(
        {
            "x-ratelimit-remaining-requests": "7",
            "x-ratelimit-limit-requests": str(10**13),
            "x-ratelimit-reset-requests": "nan",
        },
        T0,
    )
    assert (window.remaining, window.limit, window.reset_at) == (7, None, None)


def test_a_reset_a_week_or_more_away_shows_its_date() -> None:
    """A weekday alone is ambiguous for weekly/monthly windows (Qodo #2981, 3)."""
    reset = T0 + 20 * 24 * 3600
    line = format_rate_limit_line(
        {"x-ratelimit-remaining-tokens-month": "5", "x-ratelimit-reset-tokens-month": str(reset)},
        T0,
    )
    assert line.endswith(
        f"5 tokens per month left (resets {time.strftime('%b %d %H:%M', time.localtime(reset))})"
    )


def test_hugging_face_streams_go_through_the_capturing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its streaming POST used module-level requests.post (Qodo #2981, 2).

    Args:
        monkeypatch: Mounts a fake adapter on the handler's default session.
    """
    import io

    from tldw_chatbook.LLM_Calls import LLM_API_Calls

    body = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    class _StreamAdapter(_HeaderAdapter):
        def send(self, request: Any, **kwargs: Any) -> requests.Response:
            response = super().send(request, **kwargs)
            response._content = False
            response._content_consumed = False
            response.raw = io.BytesIO(body)
            return response

    real_factory = LLM_API_Calls.create_default_session

    def factory(**kwargs: Any) -> Any:
        session = real_factory(**kwargs)
        session.mount("https://", _StreamAdapter(OPENAI_HEADERS))
        return session

    monkeypatch.setattr(LLM_API_Calls, "create_default_session", factory)
    with capture_rate_limits_for("huggingface"):
        stream = LLM_API_Calls.chat_with_huggingface(
            input_data=[{"role": "user", "content": "hi"}],
            model="org/model",
            api_key="test-key",
            streaming=True,
        )
        _ = list(stream)
    assert latest_rate_limit_headers("huggingface") is not None


@pytest.mark.asyncio
async def test_side_calls_record_under_the_session_provider_not_the_handler() -> None:
    """A custom endpoint's side calls run on a shared handler key (Qodo #2981, 4)."""
    from Tests.Chat.test_console_provider_gateway import (
        _auxiliary_request,
        _auxiliary_resolution,
    )
    from tldw_chatbook.Chat.provider_readiness import provider_config_key

    def adapter(**_kwargs: Any) -> Any:
        _post_with(OPENAI_HEADERS)
        return {"choices": [{"message": {"content": "ok"}}]}

    resolution = _auxiliary_resolution(
        provider="custom-ep:acme", execution_key="custom-hosted", readiness_key="custom-hosted"
    )
    gateway = ConsoleProviderGateway(chat_api_call_fn=adapter)
    assert (await gateway.complete_auxiliary(_auxiliary_request(resolution=resolution))).text == "ok"
    assert latest_rate_limit_headers(provider_config_key("custom-ep:acme")) is not None
    assert latest_rate_limit_headers(provider_config_key("custom-hosted")) is None
