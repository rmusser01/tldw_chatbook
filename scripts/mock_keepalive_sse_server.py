"""Mock OpenAI-compatible SSE server that injects keep-alive comment frames.

Manual UAT helper for the "keep-alive spam" bug (task-37237): proxies and
gateways in front of OpenAI-compatible endpoints legally inject SSE comment
frames (``: keep-alive``) between records. This server reproduces that wire
behavior deterministically so the fix can be verified live in the Console:

    python scripts/mock_keepalive_sse_server.py [port]   # default 8799

Then point any OpenAI-compatible provider at it, e.g. custom-openai-api with
API URL ``http://127.0.0.1:8799/v1/chat/completions``, model
``keepalive-mock-1``, streaming ON. Expected with the fix: the reply streams
clean and no "keep-alive" text appears in the transcript. Before the fix this
exact server spams ": keep-alive" between every token chunk.

Endpoints: POST */chat/completions (streaming + non-streaming),
GET */models. Stdlib only; no dependencies.
"""

from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "keepalive-mock-1"
REPLY = (
    "This reply is served between keep-alive heartbeat frames. "
    "If you can read this sentence without any 'keep-alive' noise, "
    "the SSE comment filter is working."
)

#: Transport noise injected between every data record, mirroring what
#: LiteLLM-style gateways emit on slow upstreams.
HEARTBEAT_FRAMES = (": keep-alive", ": keep-alive", "event: ping")


def _chunk(delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": "chatcmpl-keepalive-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, **({"finish_reason": finish} if finish else {})}],
    }
    return f"data: {json.dumps(payload)}\n\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # quiet, keep-alives spam logs
        pass

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            body = json.dumps(
                {"object": "list", "data": [{"id": MODEL, "object": "model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        request = {}
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            pass

        if not request.get("stream"):
            body = json.dumps(
                {
                    "id": "chatcmpl-keepalive-mock",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": REPLY},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 29,
                        "total_tokens": 40,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(text: str) -> None:
            # Chunked encoding framing around each SSE burst.
            data = text.encode()
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def heartbeat() -> None:
            for frame in HEARTBEAT_FRAMES:
                send(f"{frame}\n\n")

        send(_chunk({"role": "assistant"}))
        for word in REPLY.split(" "):
            heartbeat()
            send(_chunk({"content": word + " "}))
            time.sleep(0.15)
        heartbeat()
        send(_chunk({}, finish="stop"))
        send(
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-keepalive-mock",
                    "object": "chat.completion.chunk",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 29,
                        "total_tokens": 40,
                    },
                }
            )
            + "\n\n"
        )
        send("data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")  # end of chunked body
        self.wfile.flush()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(
        f"keep-alive mock SSE server on http://127.0.0.1:{port}/v1 "
        f"(model: {MODEL}) — Ctrl+C to stop"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
