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
    # "ok", "echo" (bodies repeat the Authorization header), "refuse" (every
    # round 401) or "truncate" (the stream ends without [DONE]).
    mode = "ok"

    def log_message(self, *args: Any) -> None:
        """Keep the test output quiet."""

    def _send(self, body: bytes, content_type: str = "application/json", status: int = 200) -> None:
        self.send_response(status)
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
        if type(self).mode == "refuse":
            self._send(json.dumps({"error": {"message": "Invalid API key"}}).encode(), status=401)
            return
        if body.get("stream"):
            chunk = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                     "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}
            tail = "" if type(self).mode == "truncate" else "data: [DONE]\n\n"
            self._send(f"data: {json.dumps(chunk)}\n\n{tail}".encode(), "text/event-stream")
        elif type(self).mode == "echo":
            self._send(json.dumps({**_BODY, "debug": {"authorization": self.headers.get("Authorization")}}).encode())
        else:
            self._send(json.dumps(_BODY).encode())


@pytest.fixture
def provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A loopback provider URL, with fixtures redirected to ``tmp_path``."""
    _Provider.requests = []
    _Provider.mode = "ok"
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
    # The app's User-Agent, not urllib's: Cloudflare blocks "Python-urllib" with 1010.
    assert {r["headers"]["user-agent"] for r in _Provider.requests} == {capture_tool.USER_AGENT}
    assert capture_tool.USER_AGENT.startswith("python-requests")
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


def test_no_auth_probe_sends_only_the_fake_key(
    provider: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe records listing entries and both chat answers, and never sends a real key."""
    monkeypatch.setattr(capture_tool, "NOAUTH_DIR", tmp_path / "noauth")
    monkeypatch.setenv("TLDW_LIVE_META_BASE_URL", provider)
    monkeypatch.setenv("META_API_KEY", _KEY)
    assert capture_tool.main(["meta", "--no-auth"]) == 0
    fixture = json.loads((tmp_path / "noauth" / "meta.json").read_text(encoding="utf-8"))
    assert fixture["listing"]["entries"] == [{"id": "text-embed-1"}, {"id": "chat-small-7b"}]
    assert fixture["probe_model"] == "chat-small-7b"
    assert fixture["base_url"] == capture_tool.PER_ACCOUNT
    auth = [r["headers"].get("authorization") for r in _Provider.requests if r["body"] is not None]
    assert auth == [None, f"Bearer {capture_tool.PROBE_KEY}"]
    assert _KEY not in json.dumps(fixture)


def test_an_echoed_credential_is_redacted_before_the_fixture_is_written(
    provider: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A provider repeating the Authorization header never puts the key on disk."""
    _Provider.mode = "echo"
    fixture = _run(tmp_path, "meta", {"TLDW_LIVE_META_BASE_URL": provider})
    text = (tmp_path / "meta.json").read_text(encoding="utf-8")
    assert _KEY not in text and capture_tool.REDACTED in text
    assert fixture["chat_response"]["debug"]["authorization"] == f"Bearer {capture_tool.REDACTED}"
    assert "echoed the credential" in capsys.readouterr().out


def test_a_capture_with_no_successful_round_writes_nothing(
    provider: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every round refused: no fixture, not counted as captured."""
    _Provider.mode = "refuse"
    keys = tmp_path / "keys.env"
    keys.write_text(f"META_API_KEY={_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_LIVE_META_BASE_URL", provider)
    monkeypatch.setenv("TLDW_LIVE_META_MODEL", "m")
    assert capture_tool.main(["meta", "--keys-file", str(keys)]) == 0
    out = capsys.readouterr().out
    assert not (tmp_path / "meta.json").exists()
    assert "FAILED: no round succeeded" in out and "captured 0 of 1" in out


def test_a_truncated_stream_is_flagged(provider: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A stream cut off before [DONE] is kept as evidence but called out."""
    _Provider.mode = "truncate"
    fixture = _run(tmp_path, "meta", {"TLDW_LIVE_META_BASE_URL": provider})
    assert fixture["stream_events"] and fixture["stream_events"][-1] != "[DONE]"
    assert "cut off before [DONE]" in capsys.readouterr().out


def test_a_configured_alternate_key_variable_is_read_first() -> None:
    """Like the engine, a configured api_key_env_var (StepFun's STEP_API_KEY) wins."""
    record = next(r for r in capture_tool.engine_presets() if r.key == "stepfun")
    env = {"STEP_API_KEY": _KEY, "TLDW_LIVE_STEPFUN_API_KEY_ENV_VAR": "STEP_API_KEY"}
    target = capture_tool.Target(record, env)
    assert target.skip_reason() is None
    assert target.key_names[0] == "STEP_API_KEY"
    assert capture_tool.Target(record, {"STEP_API_KEY": _KEY}).skip_reason().startswith("no key")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TLDW_LIVE_META_BASE_URL", "ftp://example.invalid/v1"),
        ("TLDW_LIVE_META_BASE_URL", "https://user:pw@example.invalid/v1"),
        ("TLDW_LIVE_META_MODEL", "m\ninjected"),
        ("TLDW_LIVE_META_API_KEY_ENV_VAR", "NOT A NAME"),
    ],
)
def test_an_invalid_override_skips_the_provider_without_echoing_it(name: str, value: str) -> None:
    """Overrides are validated before any request; the bad value is never printed.

    Args:
        name: The override variable.
        value: An unusable value for it.
    """
    record = next(r for r in capture_tool.engine_presets() if r.key == "meta")
    reason = capture_tool.Target(record, {"META_API_KEY": _KEY, name: value}).skip_reason()
    assert reason is not None and reason.startswith("invalid override")
    assert value not in reason


def test_a_traversal_keys_file_path_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    """The keys-file path goes through the shared path validator."""
    with pytest.raises(SystemExit):
        capture_tool.main(["--list", "--keys-file", "../../etc/passwd"])
    assert "--keys-file is not a usable path" in capsys.readouterr().err


# --- uncovered_keys, on its own ---

_RECORD = next(r for r in capture_tool.engine_presets() if r.key == "zenmux")  # allows service_tier, refusal


def test_uncovered_keys_reports_each_level_and_subtracts_allowances() -> None:
    """Body top/choice/message/tool-call and stream event/choice/delta keys, minus allowances."""
    fixture = {
        "chat_response": {
            "id": "c", "choices": [{"index": 0, "finish_reason": "stop", "extra_choice": 1,
                                    "message": {"role": "assistant", "content": "ok", "refusal": None, "extra_msg": 1,
                                                "tool_calls": [{"id": "t", "type": "function", "index": 0,
                                                                "function": {"name": "f", "arguments": "{}"}}]}}],
            "service_tier": "default", "extra_top": 1,
        },
        "tool_call_response": None,
        "stream_events": [
            json.dumps({"id": "c", "event_extra": 1, "choices": [
                {"index": 0, "delta": {"content": "o", "delta_extra": 1}, "stream_choice_extra": 1}]}),
            "[DONE]",
        ],
    }
    assert capture_tool.uncovered_keys(_RECORD, fixture) == {
        "top": ["event_extra", "extra_top"],
        "choice": ["extra_choice", "stream_choice_extra"],
        "message": ["delta_extra", "extra_msg"],
        "tool_call": ["index"],
    }


def test_uncovered_keys_is_empty_for_a_strict_shape_and_ignores_error_bodies() -> None:
    """A clean reply reports nothing; a refused round's error body is not a response shape."""
    fixture = {
        "chat_response": {"id": "c", "choices": [{"index": 0, "finish_reason": "stop",
                                                  "message": {"role": "assistant", "content": "ok"}}]},
        "tool_call_response": {"error": {"message": "bad request"}},
        "stream_events": ["not json", "[DONE]"],
    }
    assert capture_tool.uncovered_keys(_RECORD, fixture) == {
        "top": [], "choice": [], "message": [], "tool_call": [],
    }

