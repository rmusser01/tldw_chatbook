"""Exercise the native provider serializers over a private fixture connection."""

import json
import threading
from argparse import Namespace
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests
from loguru import logger

_PATH = "/v1/chat/completions"
_KEY_CANARY = "SOURCE_READER_SECRET_CANARY"
_SOURCE_CANARY = "practice garden opens"


def _fixture_usage(raw, *, provider, model):
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    def count(value):
        return value if type(value) is int and value >= 0 else None

    if not isinstance(raw, Mapping):
        return None
    prompt = count(raw.get("prompt_tokens"))
    if provider == "deepseek":
        cached = count(raw.get("prompt_cache_hit_tokens"))
        uncached = count(raw.get("prompt_cache_miss_tokens"))
    else:
        details = raw.get("prompt_tokens_details")
        cached = (
            count(details.get("cached_tokens"))
            if isinstance(details, Mapping)
            else None
        )
        uncached = (
            prompt - cached if prompt is not None and cached is not None else None
        )
    if (
        prompt is None
        or cached is None
        or uncached is None
        or uncached < 0
        or uncached + cached != prompt
    ):
        return None
    output = count(raw.get("completion_tokens"))
    return ProviderUsage(
        uncached_input=uncached,
        cache_read=cached,
        output=output if output is not None else 0,
        provider=provider,
        model=model,
        partial=output is None or count(raw.get("total_tokens")) != prompt + output,
    )


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.loopback_network
@pytest.mark.parametrize(
    ("provider", "response_mode"),
    [
        ("openai", "ok"),
        ("openai", "error"),
        ("openai", "missing_output_usage"),
        ("deepseek", "ok"),
        ("deepseek", "error"),
        ("deepseek", "missing_output_usage"),
        ("deepseek", "redirect"),
    ],
)
async def test_runner_real_transport_is_isolated_and_does_not_retry(
    tmp_path, monkeypatch, response_mode, provider
):
    from tldw_chatbook.Chat.Chat_Functions import chat_api_call
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Evals.source_reader.experiment import prepare, run_experiment
    from tldw_chatbook.LLM_Calls import LLM_API_Calls, deepseek
    from tldw_chatbook.LLM_Calls.hosted_chat import normalize_hosted_chat_response
    from tldw_chatbook.Provider_Contract import ChatProviderError

    received, logs, responses, observed_usage = [], [], [], []
    opened_on, live_sessions = [], {}
    call_usage = ContextVar("source_reader_fixture_usage")
    auth_header = f"Bearer {_KEY_CANARY}"
    is_deepseek = provider == "deepseek"
    main_model, worker_model = (
        ("deepseek-v4-pro", "deepseek-flash")
        if is_deepseek
        else ("gpt-4o", "gpt-4o-mini")
    )

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def do_POST(self):
            if self.path != _PATH:
                received.append((self.path, None, False))
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(
                (self.path, payload, self.headers.get("Authorization") == auth_header)
            )
            data = json.loads(payload["messages"][-1]["content"])
            response_text = "Answer from fixture evidence."
            if "packets" in data:
                packet = data["packets"][0]
                response_text = json.dumps(
                    {
                        "findings": [
                            {
                                "statement": "Candidate fixture fact",
                                "evidence": [
                                    {
                                        "packet_id": packet["packet_id"],
                                        "quote": packet["text"].split(".")[0] + ".",
                                    }
                                ],
                            }
                        ]
                    }
                )
            usage = {"prompt_tokens": 100}
            if is_deepseek:
                usage.update(prompt_cache_hit_tokens=40, prompt_cache_miss_tokens=60)
            else:
                usage["prompt_tokens_details"] = {"cached_tokens": 0}
            if response_mode != "missing_output_usage":
                usage.update(completion_tokens=20, total_tokens=120)
            status = {"error": 503, "redirect": 307}.get(response_mode, 200)
            if status == 200:
                envelope = {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            else:
                envelope = {"error": {"message": f"{_KEY_CANARY} {_SOURCE_CANARY}"}}
            body = json.dumps(envelope).encode()
            self.send_response(status)
            if response_mode == "redirect":
                self.send_header("Location", "/redirect-target")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    prepared, output = tmp_path / "prepared", tmp_path / "run"
    prepare(prepared)
    monkeypatch.setenv("SOURCE_READER_TEST_KEY", _KEY_CANARY)
    with (
        HTTPServer(("127.0.0.1", 0), Handler) as server,
        monkeypatch.context() as patch,
    ):
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        endpoint = f"http://127.0.0.1:{server.server_port}{_PATH}"

        class FixtureSession(requests.Session):
            def __init__(self):
                super().__init__()
                self.trust_env = False
                self.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
                owner = threading.get_ident()
                live_sessions[self] = owner
                opened_on.append(owner)

            def request(self, method, url, **kwargs):
                if (
                    method != "POST"
                    or url != endpoint
                    or kwargs.get("headers", {}).get("Authorization") != auth_header
                    or kwargs.get("proxies")
                ):
                    raise ChatProviderError(
                        "Unpinned fixture request.", provider=provider
                    )
                kwargs.update(allow_redirects=False, timeout=3)
                response = super().request(method, url, **kwargs)
                responses.append(response)
                if 300 <= response.status_code < 400:
                    response.close()
                    raise ChatProviderError(
                        "Fixture redirect refused.",
                        provider=provider,
                        status_code=response.status_code,
                    )
                return response

            def close(self):
                owner = live_sessions.get(self)
                assert owner is None or owner == threading.get_ident()
                super().close()
                live_sessions.pop(self, None)

        def deepseek_fixture_request(
            *, config, payload, streaming, finish_policy, allowed_choice_keys
        ):
            if (
                config.provider != "deepseek"
                or config.base_url != base_url
                or config.api_key != _KEY_CANARY
                or config.auth_scheme != "bearer"
                or config.extra_headers
                or config.retries != 0
                or streaming
            ):
                raise ChatProviderError("Unpinned fixture request.", provider=provider)
            with (
                FixtureSession() as client,
                client.post(
                    endpoint, json=payload, headers={"Authorization": auth_header}
                ) as response,
            ):
                if response.status_code >= 400:
                    raise ChatProviderError(
                        "Fixture service failure.",
                        provider=provider,
                        status_code=response.status_code,
                    )
                return normalize_hosted_chat_response(
                    response.json(),
                    finish_policy=finish_policy,
                    allowed_choice_keys=allowed_choice_keys,
                )

        if is_deepseek:
            patch.setattr(deepseek, "hosted_chat_request", deepseek_fixture_request)
        else:
            patch.setattr(LLM_API_Calls, "create_owned_default_session", FixtureSession)

        def adapter(**kwargs):
            if (
                kwargs.get("api_endpoint") != provider
                or kwargs.get("api_base_url") != base_url
                or kwargs.get("api_key") != _KEY_CANARY
                or kwargs.get("model") not in {main_model, worker_model}
                or kwargs.get("streaming") is not False
            ):
                raise ChatProviderError("Unpinned fixture request.", provider=provider)
            response = chat_api_call(**kwargs)
            call_usage.get().append(
                response.get("usage") if isinstance(response, Mapping) else None
            )
            return response

        class FixtureGateway:
            async def complete_auxiliary(self, request):
                observations = []
                token = call_usage.set(observations)
                try:
                    result = await gateway.complete_auxiliary(request)
                finally:
                    call_usage.reset(token)
                (raw,) = observations
                observed_usage.append(raw)
                return replace(
                    result,
                    usage=_fixture_usage(
                        raw, provider=result.provider, model=result.model
                    ),
                )

        thread = threading.Thread(target=server.serve_forever)
        sink = logger.add(lambda message: logs.append(str(message)), level="DEBUG")
        try:
            gateway = ConsoleProviderGateway(chat_api_call_fn=adapter)
            try:
                thread.start()
                attempts = await run_experiment(
                    Namespace(
                        provider=provider,
                        prepared=prepared,
                        output=output,
                        main_model=main_model,
                        worker_model=worker_model,
                        base_url=base_url,
                        api_key_env="SOURCE_READER_TEST_KEY",
                        request_cap=100,
                        token_cap=3000000,
                        spend_cap_usd=100.0,
                        context_limit=32768,
                        input_limit=24000,
                        output_limit=512,
                        split="development",
                    ),
                    gateway=FixtureGateway(),
                )
            finally:
                await gateway.aclose()
        finally:
            try:
                if thread.ident is not None:
                    server.shutdown()
                    server.server_close()
                    thread.join(3)
                    assert not thread.is_alive()
            finally:
                logger.remove(sink)

    assert (
        len(received)
        == len(opened_on)
        == len(responses)
        == (24 if response_mode == "ok" else 1)
    )
    assert not live_sessions
    assert all(owner != threading.get_ident() for owner in opened_on)
    assert all(response.raw.closed for response in responses)
    for path, payload, matched_auth in received:
        assert path == _PATH and matched_auth
        assert [row["role"] for row in payload["messages"]] == ["system", "user"]
        assert not {"tools", "tool_choice", "stop", "thinking"}.intersection(payload)
        assert payload["stream"] is False
        assert payload["max_tokens"] == 512
        worker = "packets" in json.loads(payload["messages"][-1]["content"])
        assert payload["model"] == (worker_model if worker else main_model)
        if is_deepseek:
            assert payload["reasoning_effort"] == "none"

    calls = [call for row in attempts for call in row["calls"]]
    if response_mode in {"error", "redirect"}:
        assert responses[0].status_code == (307 if response_mode == "redirect" else 503)
        assert any(row["status"] == "provider_error" for row in attempts)
        assert len(calls) == 1 and not observed_usage
        assert calls[0]["status"] == "provider_error"
        assert calls[0]["usage"] is None and calls[0]["cost_usd"] is None
    elif response_mode == "missing_output_usage":
        assert any(row["status"] == "unknown_usage" for row in attempts)
        (call,) = calls
        (raw,) = observed_usage
        assert "completion_tokens" not in raw and "total_tokens" not in raw
        assert call["usage"]["partial"] and "response" not in call
        assert call["cost_usd"] is None and call["cost_upper_bound_usd"] is None
        assert call["known_cost_usd"] or call["observed_cost_upper_bound_usd"]
    else:
        assert all(
            row["status"] == "ok" for row in attempts if row["arm"] != "retrieval"
        )
        assert len(calls) == len(observed_usage) == 24
        assert all(
            call["cost_usd"] is not None or call["cost_upper_bound_usd"] is not None
            for call in calls
        )
        assert all(
            call["usage"]["output"] == 20 and not call["usage"]["partial"]
            for call in calls
        )
    if response_mode != "ok":
        assert not any("answer" in row for row in attempts)
    if observed_usage:
        assert all(call["usage"]["provider"] == provider for call in calls)
        assert all(
            call["usage"]["cache_read"] == (40 if is_deepseek else 0) for call in calls
        )
        assert all(
            call["usage"]["uncached_input"] == (60 if is_deepseek else 100)
            for call in calls
        )
    captured = "".join(logs)
    assert _KEY_CANARY not in captured and _SOURCE_CANARY not in captured
    assert all(
        _KEY_CANARY not in path.read_text(encoding="utf-8")
        for path in output.glob("*.json")
    )
