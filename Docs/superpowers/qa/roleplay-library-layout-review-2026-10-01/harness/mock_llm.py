#!/usr/bin/env python3
"""Tiny stdlib OpenAI-compatible mock LLM for rp-review captures.

GET  /v1/models                -> one model list
POST /v1/chat/completions      -> canned in-character reply echoing the last user message;
                                  SSE streaming when "stream": true (word-by-word, ~30 ms apart)

Usage: python mock_llm.py [port] [--bodies]        (default port 18765 = mock_llm.toml)
Logs go to $HARNESS_STATE/mock/ (never next to this script):
  mock_llm.<port>.requests.log   one line per request (path, stream, model, message/tool counts,
                                 whether an Authorization header was present - never its value)
  mock_llm.<port>.bodies.jsonl   with --bodies: every full request body (what the app sent)
Binds 127.0.0.1 only.
"""
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.dont_write_bytecode = True  # keep the harness directory free of __pycache__
import harness_guard  # noqa: E402 - same directory as this script

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
PORT = int(ARGS[0]) if ARGS else 18765
BODIES = "--bodies" in sys.argv
STATE = harness_guard.check_state_dir(os.environ.get("HARNESS_STATE") or harness_guard.default_state())
LOG_DIR = os.path.join(STATE, "mock")
os.makedirs(LOG_DIR, exist_ok=True)
LOG = os.path.join(LOG_DIR, f"mock_llm.{PORT}.requests.log")
BODY_LOG = os.path.join(LOG_DIR, f"mock_llm.{PORT}.bodies.jsonl")


def reply_for(body):
    msgs = body.get("messages") or []
    last = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            last = c if isinstance(c, str) else json.dumps(c)[:200]
            break
    sysmsg = next((m.get("content") for m in msgs if m.get("role") == "system"), "") or ""
    persona = "the character"
    if isinstance(sysmsg, str) and sysmsg:
        persona = sysmsg.split("\n", 1)[0][:60]
    return (f"*(mock reply)* You said: \"{last[:120]}\". Staying in character as {persona}. "
            "The wind shifts, a lantern flickers, and the story moves forward one beat.")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        with open(LOG, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} GET {self.path} auth={'yes' if self.headers.get('Authorization') else 'no'}\n")
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list", "data": [{"id": "gpt-4.1-mini", "object": "model", "owned_by": "mock"}]})
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            body = {}
        with open(LOG, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} POST {self.path} stream={body.get('stream')} model={body.get('model')} "
                     f"n_msgs={len(body.get('messages') or [])} tools={len(body.get('tools') or [])} "
                     f"auth={'yes' if self.headers.get('Authorization') else 'no'}\n")
        if BODIES:
            with open(BODY_LOG, "a") as fb:
                fb.write(json.dumps({"t": time.strftime('%H:%M:%S'), "path": self.path, "body": body}) + "\n")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": {"message": "not found"}})
        text = reply_for(body)
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]
        model = body.get("model") or "gpt-4.1-mini"
        usage = {"prompt_tokens": 50, "completion_tokens": len(text.split()), "total_tokens": 50 + len(text.split())}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(obj):
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()

            base = {"id": cid, "object": "chat.completion.chunk", "model": model}
            emit({**base, "created": int(time.time()),
                  "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
            for w in text.split(" "):
                emit({**base, "created": int(time.time()),
                      "choices": [{"index": 0, "delta": {"content": w + " "}, "finish_reason": None}]})
                time.sleep(0.03)
            emit({**base, "created": int(time.time()),
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        return self._json(200, {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                             "finish_reason": "stop"}],
                                "usage": usage})


if __name__ == "__main__":
    print(f"mock LLM on http://127.0.0.1:{PORT}/v1  log={LOG}" + (f"  bodies={BODY_LOG}" if BODIES else ""), flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
