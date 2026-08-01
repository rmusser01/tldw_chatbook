"""Capture client. Network is faked; the real-server contract is pinned by
the normalizer's fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import httpx
import pytest

from tldw_chatbook.Evals.word_bench.capture_client import (
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


@pytest.mark.asyncio
async def test_raw_capture_pins_the_unchecked_canary_contract_for_all_call_shapes():
    """TASK-709: two runner tests (test_canary_pass_verdict_is_also_stamped_
    onto_every_cell, test_degenerate_canary_propagates_onto_every_cell) rely
    on capture() NEVER computing a real canary verdict itself -- only
    _stamp_canary does, using preflight's separately-resolved value. That
    contract was previously only a code comment; a FakeClient already
    drifted from it once (see storage/runner history). Exercised against a
    payload that WOULD trip the degenerate canary check if capture() ever
    computed one (RAW's top token is " a", not " Paris"), so a regression
    that started computing a real verdict here would flip this to
    "degenerate" or "pass", not just silently keep "unchecked" by luck."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=RAW)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).capture(CANARY_PROMPT, target, "raw", 5)

    assert isinstance(result, CellCapture)
    assert result.canary == "unchecked"


@pytest.mark.asyncio
async def test_preflight_reports_a_4xx_as_blocked_not_unavailable():
    """A 4xx means the server was reachable and rejected the request (e.g.
    "logprobs not supported") -- that is Blocked, not Unavailable."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": "logprobs not supported"})

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)

    assert result.state == "no_logprobs"
    assert result.status_label == "Blocked"


@pytest.mark.asyncio
async def test_preflight_reports_a_404_specifically_as_mode_unsupported():
    """_build_request always posts to a fixed, mode-selected path, so a 404
    reliably means that path does not exist on this server -- the design
    spec's "raw mode unsupported by endpoint" row."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)

    assert result.state == "mode_unsupported"
    assert result.status_label == "Blocked"


@pytest.mark.asyncio
async def test_preflight_reports_a_5xx_as_unavailable_not_blocked():
    """A 5xx means the server itself failed -- still Unavailable, the same
    as a transport-level failure, not Blocked."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)

    assert result.state == "unreachable"
    assert result.status_label == "Unavailable"


@pytest.mark.asyncio
async def test_capture_client_reuses_one_pooled_async_client_across_calls():
    """No keep-alive across a 100+ cell grid was the defect: a fresh
    httpx.AsyncClient opened (and closed) per request. Pin that repeated
    capture() calls through the SAME WordBenchCaptureClient instance reuse
    ONE underlying httpx.AsyncClient rather than constructing a new one
    every time."""
    created: list[object] = []
    real_init = httpx.AsyncClient.__init__

    def counting_init(self, *args, **kwargs):
        created.append(self)
        return real_init(self, *args, **kwargs)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=RAW)

    transport = httpx.MockTransport(handler)
    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")

    with mock.patch.object(httpx.AsyncClient, "__init__", counting_init):
        client = WordBenchCaptureClient(
            base_url="http://127.0.0.1:9099", transport=transport
        )
        await client.capture("a", target, "raw", 5)
        await client.capture("b", target, "raw", 5)
        await client.capture("c", target, "raw", 5)
        await client.aclose()

    assert len(created) == 1, (
        "capture() must reuse one pooled AsyncClient across calls, not "
        "open a new one per request"
    )


@pytest.mark.asyncio
async def test_aclose_releases_the_pooled_client_and_is_safe_to_call_twice():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=RAW)

    client = _client(handler)
    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    await client.capture("a", target, "raw", 5)
    assert client._client is not None

    await client.aclose()
    assert client._client is None

    # Nothing open -- must not raise.
    await client.aclose()


##############################################################################
# task-1691 -- preflight captures a short continuation per target
##############################################################################


@pytest.mark.asyncio
async def test_preflight_captures_a_continuation_in_raw_mode():
    """Raw mode's canary is deliberately max_tokens: 1 (see module docstring
    and preflight's own docstring) -- a real continuation needs a genuinely
    separate request, never a lengthened canary. That second request must
    actually happen and its text must land on the result."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json=RAW)  # the canary capture
        return httpx.Response(200, json={"choices": [{"text": " blue skies ahead", "index": 0}]})

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)

    assert len(calls) == 2, "one canary request, one separate continuation request"
    assert calls[0]["max_tokens"] == 1, "the canary request itself must stay untouched"
    assert calls[1]["max_tokens"] > 1, "the continuation request must ask for more than one token"
    assert result.continuation == " blue skies ahead"


@pytest.mark.asyncio
async def test_preflight_captures_a_continuation_in_chat_mode_without_a_second_request():
    """Chat mode's canary already requests CHAT_TOKEN_WINDOW tokens and
    discards the generated text today -- that discard is a genuine salvage
    opportunity (unlike raw mode's single token), so capturing it must not
    cost a second request."""
    calls = []
    payload = {
        "choices": [{
            "message": {"role": "assistant", "content": "<|channel>thought The sky is blue"},
            "logprobs": {"content": [
                {"id": 1, "token": "<|channel>", "bytes": [], "logprob": -0.01,
                 "top_logprobs": [{"id": 1, "token": "<|channel>", "bytes": [], "logprob": -0.01}]},
                {"id": 2, "token": " Paris", "bytes": [], "logprob": -0.2,
                 "top_logprobs": [{"id": 2, "token": " Paris", "bytes": [], "logprob": -0.2}]},
            ]},
        }]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "chat", 5)

    assert len(calls) == 1, (
        "chat mode's continuation must be salvaged from the canary response, "
        "never a second request"
    )
    assert result.state == "ok"
    assert result.canary == "pass"
    assert result.continuation == "<|channel>thought The sky is blue"


@pytest.mark.asyncio
async def test_preflight_continuation_request_in_raw_mode_carries_the_target_prefix():
    """Readiness (and its continuation) is a property of what the run will
    actually send -- the same steering contract preflight's own docstring
    already states for the canary applies equally here."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json=RAW)
        return httpx.Response(200, json={"choices": [{"text": " continues", "index": 0}]})

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m", prefix="Note: ")
    result = await _client(handler).preflight(target, "raw", 5)

    assert seen[1]["prompt"] == f"Note: {CANARY_PROMPT}"
    assert result.continuation == " continues"


@pytest.mark.asyncio
async def test_preflight_continuation_in_chat_mode_reflects_system_prompt_steering():
    seen = []
    payload = {
        "choices": [{
            "message": {"role": "assistant", "content": "steered output"},
            "logprobs": {"content": [
                {"id": 2, "token": " Paris", "bytes": [], "logprob": -0.2,
                 "top_logprobs": [{"id": 2, "token": " Paris", "bytes": [], "logprob": -0.2}]},
            ]},
        }]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m",
                     system_prompt="Be careful.")
    result = await _client(handler).preflight(target, "chat", 5)

    assert seen[0]["messages"][0] == {"role": "system", "content": "Be careful."}
    assert result.continuation == "steered output"


@pytest.mark.asyncio
async def test_preflight_continuation_degrades_to_empty_string_when_the_canary_itself_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)

    assert result.state == "unreachable"
    assert result.continuation == ""


@pytest.mark.asyncio
async def test_preflight_continuation_degrades_to_empty_string_when_the_continuation_request_fails():
    """The canary succeeds (state stays 'ok') but the SEPARATE continuation
    request fails -- must degrade to "" without raising and without touching
    the already-resolved state/canary verdict."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json=RAW)
        raise httpx.ConnectError("continuation endpoint down")

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)

    assert result.state == "ok"
    assert result.continuation == ""


@pytest.mark.asyncio
async def test_preflight_continuation_is_empty_string_when_chat_response_has_no_message_content():
    payload = {
        "choices": [{
            "logprobs": {"content": [
                {"id": 2, "token": " Paris", "bytes": [], "logprob": -0.2,
                 "top_logprobs": [{"id": 2, "token": " Paris", "bytes": [], "logprob": -0.2}]},
            ]},
            # no "message" key at all
        }]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "chat", 5)

    assert result.state == "ok"
    assert result.continuation == ""


@pytest.mark.asyncio
async def test_preflight_continuation_is_capped_to_a_bounded_length():
    calls = []
    long_text = "x" * 5000

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json=RAW)
        return httpx.Response(200, json={"choices": [{"text": long_text}]})

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)

    assert 0 < len(result.continuation) < len(long_text), (
        "the stored continuation must be capped to a bounded window, "
        "not the full generated text"
    )
    assert long_text.startswith(result.continuation)


@pytest.mark.asyncio
async def test_preflight_canary_verdict_is_unaffected_by_the_continuation_capture():
    """Approach (a) was chosen: the continuation is a separate request/
    response that is never passed through normalize_logprobs, so the canary
    verdict must be identical to what it was before continuation capture
    existed -- regardless of what the continuation request returns, even
    when it errors outright."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(200, json=RAW)  # canary: top token " much", not Paris
        return httpx.Response(500, text="boom")  # continuation request fails outright

    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")
    result = await _client(handler).preflight(target, "raw", 5)

    assert result.canary == "degenerate"
    assert result.state == "ok"
    assert result.k_returned == 5
    assert result.continuation == ""


@pytest.mark.asyncio
async def test_used_as_an_async_context_manager_closes_on_exit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=RAW)

    transport = httpx.MockTransport(handler)
    target = Target(id="t", name="n", provider="llama_cpp", model_id="m")

    async with WordBenchCaptureClient(
        base_url="http://127.0.0.1:9099", transport=transport
    ) as client:
        result = await client.capture("a", target, "raw", 5)
        assert isinstance(result, CellCapture)
        assert client._client is not None

    assert client._client is None
