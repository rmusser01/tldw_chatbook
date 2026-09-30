"""The live-capture tool builds each preset's request like the engine and never leaks the key.

Runs ``Tests/fixtures/cloud_live/capture.py`` against a loopback stdlib HTTP
server standing in for providers (TASK-33640): auth header style, request
body shape per record, fixture contents, and the uncovered-key report.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from Tests.fixtures.cloud_live import capture as capture_tool

pytestmark = pytest.mark.loopback_network  # owns a numeric loopback listener

_KEY = "sk-live-canary-9f8e7d6c5b4a"
_BODY = {
    "id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    "surprise_extra": 1,
}


class _Provider(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:
        """Keep the test output quiet."""

    def _send(self, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve a model listing."""
        type(self).requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": None})
        self._send(json.dumps({"data": [{"id": "text-embed-1"}, {"id": "chat-small-7b"}]}).encode())

    def do_POST(self) -> None:
        """Serve a chat completion, streamed when asked."""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        if body.get("stream"):
            chunk = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                     "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}
            self._send(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode(), "text/event-stream")
        else:
            self._send(json.dumps(_BODY).encode())


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A loopback provider URL, with fixtures redirected to ``tmp_path``."""
    _Provider.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(capture_tool, "FIXTURE_DIR", tmp_path)
    for record in capture_tool.engine_presets():
        for name in record.api_key_env_candidates:
            monkeypatch.delenv(name, raising=False)
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def _run(tmp_path: Path, provider_key: str, extra: dict[str, str]) -> dict[str, Any]:
    keys = tmp_path / "keys.env"
    env_name = next(r for r in capture_tool.engine_presets() if r.key == provider_key).api_key_env_var
    keys.write_text(f"# test keys\nexport {env_name}=\"{_KEY}\"\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        for name, value in extra.items():
            patch.setenv(name, value)
        assert capture_tool.main([provider_key, "--keys-file", str(keys)]) == 0
    return json.loads((tmp_path / f"{provider_key}.json").read_text(encoding="utf-8"))


def test_azure_capture_uses_the_api_key_header_and_max_completion_tokens(
    provider: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Azure: ``api-key`` header, ``max_completion_tokens``, deployment model, redacted URL."""
    fixture = _run(tmp_path, "azure", {"TLDW_LIVE_AZURE_BASE_URL": provider, "TLDW_LIVE_AZURE_MODEL": "my-deploy"})
    posts = [r for r in _Provider.requests if r["body"] is not None]
    assert {r["headers"].get("api-key") for r in posts} == {_KEY}
    assert not any("authorization" in r["headers"] for r in posts)
    assert all("max_completion_tokens" in r["body"] and "max_tokens" not in r["body"] for r in posts)
    assert all(r["body"]["model"] == "my-deploy" for r in posts)
    assert not any(r["body"] is None for r in _Provider.requests)  # no discovery for deployments
    assert fixture["base_url"] == capture_tool.PER_ACCOUNT
    assert fixture["statuses"] == {"plain": 200, "tool": 200, "stream": 200}
    assert fixture["stream_events"][-1] == "[DONE]"
    out = capsys.readouterr().out
    assert _KEY not in out and _KEY not in json.dumps(fixture)
    assert "surprise_extra" in out  # the uncovered-key report names it


def test_listing_picks_a_chat_model_and_ollama_cloud_sends_no_tool_choice(
    provider: str, tmp_path: Path
) -> None:
    """Discovery skips a non-chat model; a record without the tool_choice flag omits it."""
    fixture = _run(tmp_path, "ollama_cloud", {"TLDW_LIVE_OLLAMA_CLOUD_BASE_URL": provider})
    assert fixture["model"] == "chat-small-7b"
    tool_request = next(r for r in _Provider.requests if r["body"] and "tools" in r["body"])
    assert "tool_choice" not in tool_request["body"]
    assert tool_request["headers"]["authorization"] == f"Bearer {_KEY}"
    streamed = next(r for r in _Provider.requests if r["body"] and r["body"].get("stream"))
    assert streamed["body"]["stream_options"] == {"include_usage": True}


def test_tools_off_preset_skips_the_tool_round(provider: str, tmp_path: Path) -> None:
    """Nous ships native tools off, so no tool request is sent."""
    fixture = _run(tmp_path, "nous", {"TLDW_LIVE_NOUS_BASE_URL": provider})
    assert fixture["tool_call_response"] is None
    assert not any(r["body"] and "tools" in r["body"] for r in _Provider.requests)


def test_missing_key_or_account_url_is_a_clean_skip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without a key (or Azure's URL and deployment), nothing is sent or written."""
    keys = tmp_path / "keys.env"
    keys.write_text(f"AZURE_OPENAI_API_KEY={_KEY}\n", encoding="utf-8")
    assert capture_tool.main(["azure", "together", "--keys-file", str(keys)]) == 0
    out = capsys.readouterr().out
    assert "[azure] skip: per-account URL" in out
    assert "[together] skip: no key" in out
    assert _KEY not in out
