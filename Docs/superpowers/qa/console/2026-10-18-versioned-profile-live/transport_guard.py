"""Disposable QA transport accounting; never imports or modifies app policy."""

import collections
import fcntl
import hashlib
import json
import threading
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import requests


class ResponseObservation:
    def __init__(self, emit, attempt, content_type):
        self.emit = emit
        self.attempt = attempt
        self.sse = "text/event-stream" in content_type
        self.buffer = b""
        self.bytes = 0
        self.reasoning_bytes = 0
        self.content_bytes = 0
        self.done = False
        self.invalid = 0
        self.tool_names = set()
        self.finish_reasons = set()
        self.closed = False
        self.truncated = False

    def payload(self, raw):
        if raw.strip() == b"[DONE]":
            self.done = True
            return
        try:
            item = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self.invalid += 1
            return
        if not isinstance(item, dict) or not isinstance(item.get("choices", []), list):
            self.invalid += 1
            return
        for choice in item.get("choices", []):
            if not isinstance(choice, dict):
                self.invalid += 1
                continue
            if choice.get("finish_reason") is not None:
                self.finish_reasons.add(str(choice["finish_reason"]))
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                self.invalid += 1
                continue
            for key in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(key), str):
                    self.reasoning_bytes += len(delta[key].encode())
            if isinstance(delta.get("content"), str):
                self.content_bytes += len(delta["content"].encode())
            for call in delta.get("tool_calls", []):
                name = (call.get("function") or {}).get("name")
                if isinstance(name, str):
                    self.tool_names.add(name)

    def feed(self, chunk):
        self.bytes += len(chunk)
        if self.truncated:
            return
        self.buffer += chunk
        if len(self.buffer) > 1024 * 1024:
            self.truncated = True
            self.buffer = b""
            return
        if self.sse:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if line.startswith(b"data:"):
                    self.payload(line[5:].strip())

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.buffer and not self.truncated:
            if self.sse and self.buffer.startswith(b"data:"):
                self.payload(self.buffer[5:].strip())
            elif not self.sse:
                self.payload(self.buffer)
        self.buffer = b""
        self.emit(
            {
                "event": "response_observation",
                "attempt": self.attempt,
                "body_bytes": self.bytes,
                "reasoning_bytes": self.reasoning_bytes,
                "content_bytes": self.content_bytes,
                "sse_done": self.done,
                "invalid_json": self.invalid,
                "observation_truncated": self.truncated,
                "tool_names": sorted(self.tool_names),
                "finish_reasons": sorted(self.finish_reasons),
            }
        )


def install(root, *, consumed, cap):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    state = root / "counter.json"
    if not state.exists():
        with state.open("x") as f:
            json.dump({"consumed": consumed, "authorized_total": cap}, f)
    lock = threading.Lock()

    def emit(row):
        with lock, (root / "transport-metadata.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def admit(method, url, body, transport, retries):
        u = urlsplit(str(url))
        local = (
            u.scheme == "http"
            and u.hostname in ("127.0.0.1", "localhost")
            and u.port == 8010
        )
        inference = method.upper() == "POST" and u.path in (
            "/v1/chat/completions",
            "/chat/completions",
        )
        readonly = method.upper() == "GET" and u.path in (
            "/health",
            "/v1/models",
            "/models",
            "/props",
        )
        if not local or not (inference or readonly) or retries != 0:
            emit(
                {
                    "event": "denied_route",
                    "transport": transport,
                    "method": method,
                    "local_endpoint": local,
                    "inference": inference,
                    "retry_total": retries,
                }
            )
            raise RuntimeError("QA transport route/retry boundary refused")
        if readonly:
            emit(
                {"event": "readonly_transport", "transport": transport, "path": u.path}
            )
            return None
        raw = body.encode() if isinstance(body, str) else body
        payload = json.loads(raw)
        with state.open("r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            counter = json.load(f)
            if counter["authorized_total"] != cap or counter["consumed"] < consumed:
                raise RuntimeError("QA counter contract changed")
            if counter["consumed"] >= cap:
                emit(
                    {
                        "event": "denied_cap",
                        "transport": transport,
                        "consumed": counter["consumed"],
                    }
                )
                raise RuntimeError("whole-probe model-attempt cap reached")
            counter["consumed"] += 1
            attempt = counter["consumed"]
            f.seek(0)
            json.dump(counter, f)
            f.truncate()
            f.flush()
            import os

            os.fsync(f.fileno())
        schemas = payload.get("tools") or []
        emit(
            {
                "event": "request",
                "attempt": attempt,
                "transport": transport,
                "stream": payload.get("stream"),
                "max_tokens": payload.get("max_tokens"),
                "request_sha256": hashlib.sha256(raw).hexdigest(),
                "role_counts": dict(
                    collections.Counter(
                        m.get("role") for m in payload.get("messages", [])
                    )
                ),
                "tool_names": [t.get("function", {}).get("name") for t in schemas],
                "retry_total": retries,
            }
        )
        return attempt

    async_original = httpx.AsyncHTTPTransport.handle_async_request
    sync_original = httpx.HTTPTransport.handle_request
    requests_original = requests.adapters.HTTPAdapter.send

    class ObservedAsync(httpx.AsyncByteStream):
        def __init__(self, stream, obs):
            self.stream, self.obs = stream, obs

        async def __aiter__(self):
            try:
                async for chunk in self.stream:
                    self.obs.feed(chunk)
                    yield chunk
            finally:
                self.obs.close()

        async def aclose(self):
            try:
                await self.stream.aclose()
            finally:
                self.obs.close()

    class ObservedSync(httpx.SyncByteStream):
        def __init__(self, stream, obs):
            self.stream, self.obs = stream, obs

        def __iter__(self):
            try:
                for chunk in self.stream:
                    self.obs.feed(chunk)
                    yield chunk
            finally:
                self.obs.close()

        def close(self):
            try:
                self.stream.close()
            finally:
                self.obs.close()

    def response_observation(response, attempt):
        emit(
            {
                "event": "http",
                "attempt": attempt,
                "status": response.status_code,
                "content_type": response.headers.get("content-type", "").split(";")[0],
            }
        )
        return ResponseObservation(
            emit, attempt, response.headers.get("content-type", "")
        )

    async def observed_async(transport, request):
        attempt = admit(
            request.method,
            request.url,
            request.content,
            "httpx_async",
            transport._pool._retries,
        )
        try:
            response = await async_original(transport, request)
        except Exception as error:
            emit(
                {
                    "event": "transport_exception",
                    "attempt": attempt,
                    "exception_type": type(error).__name__,
                }
            )
            raise
        if attempt is not None:
            response.stream = ObservedAsync(
                response.stream, response_observation(response, attempt)
            )
        return response

    def observed_sync(transport, request):
        attempt = admit(
            request.method,
            request.url,
            request.content,
            "httpx_sync",
            transport._pool._retries,
        )
        try:
            response = sync_original(transport, request)
        except Exception as error:
            emit(
                {
                    "event": "transport_exception",
                    "attempt": attempt,
                    "exception_type": type(error).__name__,
                }
            )
            raise
        if attempt is not None:
            response.stream = ObservedSync(
                response.stream, response_observation(response, attempt)
            )
        return response

    def observed_requests(adapter, request, **kwargs):
        attempt = admit(
            request.method,
            request.url,
            request.body,
            "requests",
            adapter.max_retries.total,
        )
        try:
            response = requests_original(adapter, request, **kwargs)
        except Exception as error:
            emit(
                {
                    "event": "transport_exception",
                    "attempt": attempt,
                    "exception_type": type(error).__name__,
                }
            )
            raise
        if attempt is not None:
            obs = response_observation(response, attempt)
            raw_stream = response.raw.stream

            def stream(*args, **kw):
                try:
                    for chunk in raw_stream(*args, **kw):
                        obs.feed(chunk)
                        yield chunk
                finally:
                    obs.close()

            response.raw.stream = stream
        return response

    httpx.AsyncHTTPTransport.handle_async_request = observed_async
    httpx.HTTPTransport.handle_request = observed_sync
    requests.adapters.HTTPAdapter.send = observed_requests
    return {"state": state, "events": root / "transport-metadata.jsonl"}
