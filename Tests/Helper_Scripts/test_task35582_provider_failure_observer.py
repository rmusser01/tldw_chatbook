"""Keep qualification error evidence before Console sanitizes the failure."""

import json
from contextlib import ExitStack

import pytest

from Helper_Scripts.task35582_provider_failure_observer import (
    install_provider_failure_observer,
)
from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor
from tldw_chatbook.Chat.Chat_Deps import ChatProviderError
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.LLM_Calls import hosted_provider_engine, provider_sessions
from tldw_chatbook.LLM_Calls.hosted_provider_engine import HostedProviderResolution

pytestmark = pytest.mark.bootstrap_profile
CANARY = "SYNTHETIC-PRIVATE-BODY-NOT-A-LIVE-RESPONSE"
MODEL = "offline-diagnostic-model"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect",
            "description": "Inspect the offline fixture.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _sse(value):
    return ("data: " + json.dumps(value) + "\n\n").encode()


class FakeHTTPResponse:
    def __init__(self, mode):
        self.headers = {}
        self.mode = mode
        self.status_code = 502 if mode == "http_502" else 200
        self.closed = False

    def iter_content(self, chunk_size):
        yield _sse(
            {
                "choices": [
                    {"index": 0, "delta": {"content": "ok"}, "finish_reason": None}
                ]
            }
        )
        if self.mode == "malformed_json":
            yield b"data: {" + CANARY.encode() + b"}\n\n"
            return
        if self.mode == "read_failure":
            raise OSError(CANARY)
        yield _sse(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
            }
        )
        if self.mode != "missing_done":
            yield b"data: [DONE]\n\n"

    def json(self):
        return {"error": {"message": CANARY}}

    def close(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["valid", "malformed_json", "read_failure", "missing_done", "http_502"]
)
async def test_real_handler_gateway_projection_does_not_identify_response_cause(
    monkeypatch, mode, tmp_path, request
):
    response = FakeHTTPResponse(mode)
    sends = []
    lower_errors = []

    class FakeSession:
        def post(self, url, **kwargs):
            assert url == "http://offline.invalid/v1/chat/completions"
            assert "Authorization" not in kwargs["headers"]
            assert kwargs["json"]["stream"] is True
            assert kwargs["json"]["stream_options"] == {"include_usage": True}
            assert kwargs["json"]["tools"] == TOOLS
            sends.append(response.status_code)
            return response

    monkeypatch.setattr(provider_sessions, "get_session", lambda *_args: FakeSession())
    monkeypatch.setattr(
        hosted_provider_engine,
        "resolve_hosted_request",
        lambda record, **_kwargs: HostedProviderResolution(
            provider="custom-hosted",
            model=MODEL,
            api_key="",
            base_url="http://offline.invalid/v1",
            timeout=1.0,
            retries=0,
            retry_delay=0.0,
            streaming=True,
        ),
    )
    original_next = hosted_provider_engine.HostedProviderStream.__next__

    def observe_next(stream):
        try:
            return original_next(stream)
        except StopIteration:
            raise
        except Exception as exc:
            lower_errors.append((type(exc).__name__, str(exc)))
            raise

    monkeypatch.setattr(
        hosted_provider_engine.HostedProviderStream, "__next__", observe_next
    )

    class Evidence:
        def write(self, name, value):
            (tmp_path / name).write_text(json.dumps(value))

    observer_stack = ExitStack()
    install_provider_failure_observer(observer_stack, Evidence())
    request.addfinalizer(observer_stack.close)
    gateway = ConsoleProviderGateway(config_provider=dict)
    resolution = ConsoleProviderResolution(
        provider="custom-ep:offline",
        selected_provider="custom-ep:offline",
        execution_key="custom-hosted",
        base_url="http://offline.invalid/v1",
        model=MODEL,
        ready=True,
        api_key="",
        streaming=True,
        max_tokens=1024,
    )
    chunks = []
    error = None
    with use_run_actor(CurrentRunActor("primary", "offline-primary", None)):
        try:
            async for item in gateway.stream_chat(
                resolution,
                [{"role": "user", "content": "offline control"}],
                tools=TOOLS,
            ):
                chunks.append(item)
        except ChatProviderError as exc:
            error = exc
    assert sends == [response.status_code]
    assert response.closed
    if mode == "valid":
        assert error is None
        assert "".join(item for item in chunks if isinstance(item, str)) == "ok"
        assert lower_errors == []
    elif mode == "http_502":
        assert error.status_code == 502
        assert "provider unavailable" in str(error)
        assert "Status: 502" in str(error)
        assert lower_errors == []
    else:
        assert error.status_code == 502
        assert (
            str(error)
            == "Provider error from custom-ep:offline: unexpected provider error."
        )
        assert lower_errors == [
            (
                "HostedChatProtocolError",
                {
                    "malformed_json": "Hosted Chat stream JSON is malformed.",
                    "read_failure": "Hosted Chat stream read failed.",
                    "missing_done": "Hosted Chat stream ended before a clean terminal record.",
                }[mode],
            )
        ]
    assert CANARY not in str(error)
    assert all(CANARY not in str(item) for item in chunks)

    packets = sorted(tmp_path.glob("provider-error-*.json"))
    if mode == "valid":
        assert packets == []
    else:
        assert len(packets) == 1, (
            "the original failure was lost before error projection"
        )
        record = json.loads(packets[0].read_text())
        assert record["run_id"] == "offline-primary" and record["actor"] == "primary"
        expected_reason = {
            "malformed_json": "stream_json_malformed",
            "read_failure": "stream_read_failed",
            "missing_done": "stream_terminal_missing",
            "http_502": "unclassified",
        }[mode]
        assert record["reason_code"] == expected_reason
        assert record["exception_class"] == (
            "ChatProviderError" if mode == "http_502" else "HostedChatProtocolError"
        )
        assert record["original_status"] == (502 if mode == "http_502" else None)
        assert record["frames"] and len(record["frames"]) <= 16
        assert CANARY not in packets[0].read_text()
        assert all(
            set(frame) == {"file", "function", "line"} for frame in record["frames"]
        )


@pytest.mark.parametrize("sink_fails", [False, True])
def test_observer_preserves_default_copy_credentials_and_restores_patch(sink_fails):
    from types import SimpleNamespace

    from tldw_chatbook.Chat import console_provider_gateway as gateway

    packets = []

    class Evidence:
        def write(self, name, value):
            if sink_fails:
                raise OSError(CANARY)
            packets.append(value)

    original = gateway.safe_provider_error_copy
    resolution = SimpleNamespace(provider="custom-ep:offline", api_key=CANARY)
    try:
        raise RuntimeError(CANARY)
    except RuntimeError as exc:
        expected = gateway._error_copy_for(original, resolution, exc)
        with ExitStack() as stack:
            install_provider_failure_observer(stack, Evidence())
            actual = gateway._error_copy_for(
                gateway.safe_provider_error_copy, resolution, exc
            )
        assert actual == expected and CANARY not in actual
    assert gateway.safe_provider_error_copy is original
    assert CANARY not in json.dumps(packets)
    assert len(packets) == (0 if sink_fails else 1)
    if packets:
        assert packets[0]["reason_code"] == "unclassified"
        assert packets[0]["run_id"] is None
