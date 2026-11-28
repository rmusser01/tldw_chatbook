"""One real local HTTP SSE endpoint for the frozen DeepSeek adapter."""

from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from socket import socket
from threading import Thread
from time import monotonic_ns

PORT = 9519
PROMPT = "Remember the marker CEDAR. Reply exactly CEDAR READY."
REPLY_CHUNKS = (
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "CEDAR READY"}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}},
)
WIRE = (''.join(f"data: {json.dumps(chunk)}\n\n" for chunk in REPLY_CHUNKS) + "data: [DONE]\n\n").encode()


def require_closed_port(port):
    with socket() as connection:
        assert connection.connect_ex(("127.0.0.1", port)) != 0, port


@contextmanager
def serve(receipt, expected_requests):
    """Keep the fixture in the benchmark owner, with no extra process."""
    observations = []

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            self.request.settimeout(5)
            super().setup()

        def log_message(self, *_):
            pass  # Never echo request data, URLs, or headers.

        def do_POST(self):
            arrival = monotonic_ns()
            status = 400
            schemas = []
            try:
                length = int(self.headers["Content-Length"])
                if not 0 < length < 2**20:
                    raise ValueError("Fixture body is out of bounds")
                message = json.loads(self.rfile.read(length))
                users = [row["content"] for row in message["messages"] if row["role"] == "user"]
                if (
                    len(observations) >= expected_requests
                    or self.path != "/chat/completions"
                    or self.headers.get("Authorization") != "Bearer ci-loopback-fixture"
                    or message["model"] != "deepseek-flash"
                    or message["stream"] is not True
                    or message["stream_options"] != {"include_usage": True}
                    or users != [PROMPT]
                    or not 0 < len(message["tools"]) <= 128
                ):
                    raise ValueError("Unexpected provider request")
                tools = message["tools"]
                names = [row["function"]["name"] for row in tools]
                if len(set(names)) != len(names) or any(row["type"] != "function" for row in tools):
                    raise ValueError("Unexpected tool shape or duplicated name")
                schemas = [
                    {"ordinal": i, "sha256": hashlib.sha256(
                        json.dumps(value, allow_nan=False, ensure_ascii=False,
                                   separators=(",", ":"), sort_keys=True).encode()
                    ).hexdigest()}
                    for i, value in enumerate(tools)
                ]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Content-Length", str(len(WIRE)))
                self.end_headers()
                self.wfile.write(WIRE)
                self.wfile.flush()
                status = 200
            except (ValueError, TypeError, KeyError):
                self.send_error(400, "Fixture request did not qualify")
            except OSError:
                status = 499
            finally:
                observations.append({"method": "POST", "received_monotonic_ns": arrival, "status": status, "tool_schemas": schemas})

    require_closed_port(PORT)
    with HTTPServer(("127.0.0.1", PORT), Handler) as server:
        worker = Thread(target=server.serve_forever)
        worker.start()
        try:
            yield observations
        finally:
            server.shutdown()
            worker.join()
            with receipt.open("x") as output:
                json.dump(observations, output, indent=2)
    require_closed_port(PORT)
