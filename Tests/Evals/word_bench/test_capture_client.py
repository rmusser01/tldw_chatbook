"""Capture client. Network is faked; the real-server contract is pinned by
the normalizer's fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tldw_chatbook.Evals.word_bench.capture_client import (
    CANARY_EXPECT,
    CANARY_PROMPT,
    NEUTRAL_SAMPLER,
    WordBenchCaptureClient,
)
from tldw_chatbook.Evals.word_bench.models import CellCapture, CellError, Target

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "word_bench"
RAW = json.loads((FIXTURES / "llamacpp_raw_completions.json").read_text())


def _client(handler) -> WordBenchCaptureClient:
    transport = httpx.MockTransport(handler)
    return WordBenchCaptureClient(base_url="http://127.0.0.1:9099", transport=transport)


def test_neutral_sampler_does_not_collapse_the_distribution():
    """temperature must be 1.0, not 0 -- zero collapses what we measure."""
    assert NEUTRAL_SAMPLER["temperature"] == 1.0
    assert NEUTRAL_SAMPLER["top_p"] == 1.0
    assert NEUTRAL_SAMPLER["top_k"] == 0


@pytest.mark.asyncio
async def test_raw_mode_posts_to_completions_with_neutral_sampler():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).capture("The protestors were met with", target, "raw", 5)

    assert seen["url"].endswith("/v1/completions")
    assert seen["body"]["max_tokens"] == 1
    assert seen["body"]["temperature"] == 1.0
    assert seen["body"]["top_p"] == 1.0
    assert seen["body"]["top_k"] == 0
    assert isinstance(result, CellCapture)
    assert result.content_offset == 0
    assert result.canary == "unchecked", (
        "the real client always returns 'unchecked' -- turning that into the "
        "real verdict is the runner's _stamp_canary job, not capture()'s"
    )


@pytest.mark.asyncio
async def test_raw_mode_prepends_target_prefix_to_the_snippet():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m", prefix="Note: ")
    await _client(handler).capture("the snippet", target, "raw", 5)
    assert seen["body"]["prompt"] == "Note: the snippet"


@pytest.mark.asyncio
async def test_chat_mode_sends_system_prompt_as_a_message():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m",
                    system_prompt="Be careful.")
    await _client(handler).capture("the snippet", target, "chat", 5)

    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be careful."}
    assert seen["body"]["messages"][-1] == {"role": "user", "content": "the snippet"}


@pytest.mark.asyncio
async def test_chat_mode_also_carries_the_neutral_sampler():
    """The two branches share one **NEUTRAL_SAMPLER splat today, but nothing
    stops a chat-specific override from landing in the later payload.update()
    call and shipping undetected. Pin it on chat's wire body too, not just
    raw's."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    await _client(handler).capture("the snippet", target, "chat", 5)

    assert seen["body"]["temperature"] == 1.0
    assert seen["body"]["top_p"] == 1.0
    assert seen["body"]["top_k"] == 0


@pytest.mark.asyncio
async def test_chat_mode_requests_a_window_not_a_single_token():
    """It must be able to skip leading control tokens."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    await _client(handler).capture("s", target, "chat", 5)
    assert seen["body"]["max_tokens"] > 1


@pytest.mark.asyncio
async def test_transport_failure_becomes_a_cell_error_not_an_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).capture("s", target, "raw", 5)
    assert isinstance(result, CellError)
    assert result.reason == "unreachable"


@pytest.mark.asyncio
async def test_http_error_status_becomes_a_cell_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).capture("s", target, "raw", 5)
    assert isinstance(result, CellError)
    assert result.reason == "http_error"


@pytest.mark.asyncio
async def test_preflight_reports_ok_and_actual_k():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 20)
    assert result.state == "ok"
    assert result.k_returned == 5, "must report K actually returned, not requested"
    assert result.status_label == "Ready"


@pytest.mark.asyncio
async def test_preflight_marks_a_degenerate_canary_without_blocking():
    """A model that continues the canary with nonsense is still runnable --
    it may be exactly what the user wants to study -- but the whole column
    must carry the warning."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=RAW)  # " a", not " Paris"

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.canary == "degenerate"
    assert result.state == "ok"
    assert result.is_warned is True


@pytest.mark.asyncio
async def test_preflight_passes_canary_when_expected_token_is_present():
    payload = {
        "choices": [{"logprobs": {"content": [{
            "id": 1, "token": " Paris", "bytes": [], "logprob": -0.2,
            "top_logprobs": [{"id": 1, "token": " Paris", "bytes": [], "logprob": -0.2}],
        }]}}]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.canary == "pass"
    assert result.is_warned is False


@pytest.mark.asyncio
async def test_preflight_passes_canary_when_expected_token_is_present_at_rank_2():
    """Rank order has been observed to flip between identical requests, so
    the canary must scan the whole top-K, not only rank 1 -- a rank-1-only
    check would be flaky in exactly this situation. Here the top-1 token is
    NOT the expected one; the expected token is present but ranked second."""
    payload = {
        "choices": [{"logprobs": {"content": [{
            "id": 2, "token": " a", "bytes": [], "logprob": -0.1,
            "top_logprobs": [
                {"id": 2, "token": " a", "bytes": [], "logprob": -0.1},
                {"id": 1, "token": " Paris", "bytes": [], "logprob": -0.5},
            ],
        }]}}]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.canary == "pass"
    assert result.is_warned is False


@pytest.mark.asyncio
async def test_preflight_reports_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.state == "unreachable"
    assert result.status_label == "Unavailable"


@pytest.mark.asyncio
async def test_preflight_reports_no_logprobs_as_blocked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.state == "no_logprobs"
    assert result.status_label == "Blocked"


@pytest.mark.asyncio
async def test_preflight_reports_no_logprobs_as_blocked_when_top_logprobs_is_empty():
    """The critical case this fix closes: an endpoint that honours the
    request shape (choices[0].logprobs.content is present and carries a
    token) but returns an empty top_logprobs at the measured position. Before
    the fix this silently produced a zero-token CellCapture that preflight
    read as state='ok' / 'Ready', letting a run proceed and produce an
    all-zero-divergence grid."""
    payload = {
        "choices": [{"logprobs": {"content": [{
            "id": 1, "token": " a", "bytes": [32, 97], "logprob": -0.5,
            "top_logprobs": [],
        }]}}]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)
    assert result.state == "no_logprobs"
    assert result.status_label == "Blocked"


@pytest.mark.asyncio
async def test_html_200_response_becomes_a_cell_error_not_an_exception():
    """A proxy in front of the endpoint can return an HTML error page with a
    200 status. response.json() raises json.JSONDecodeError (a ValueError,
    not an httpx.HTTPError) in that case -- it must not abort an entire
    multi-hundred-cell run."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>Bad Gateway</body></html>")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).capture("s", target, "raw", 5)
    assert isinstance(result, CellError)
    assert result.reason == "bad_response"


@pytest.mark.asyncio
async def test_malformed_top_logprobs_entry_becomes_a_cell_error_not_an_exception():
    """A top_logprobs entry missing "token" or "logprob" raises KeyError
    inside _to_token_probs -- that must not escape capture() either."""
    payload = {
        "choices": [{"logprobs": {"content": [{
            "id": 1, "token": " a", "bytes": [32, 97], "logprob": -0.5,
            "top_logprobs": [{"id": 1, "bytes": [32, 97]}],  # missing "token"/"logprob"
        }]}}]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).capture("s", target, "raw", 5)
    assert isinstance(result, CellError)
    assert result.reason == "bad_response"


def test_canary_expectation_is_a_widely_agreed_continuation():
    assert "capital of France" in CANARY_PROMPT
    assert any("Paris" in tok for tok in CANARY_EXPECT)
