"""Small HTTP model double and independent oracle for TASK-37270 UAT.

The model receives only outbound request payloads, never the expected fixture code
or a filesystem path. These helpers import no product modules.
"""

from __future__ import annotations

import copy
import json
import re
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator, SchemaError, ValidationError

VARIANTS = ("fenced", "native", "compact-recovery", "wrong-family")
MODEL = "uat-project-model"


def assert_project_evidence(
    *, code, alias, steps, provider_request, final_text, persisted_text, painted_text
):
    """Correlate private DB metadata with the actual provider's README exchange."""
    calls = [
        s
        for s in steps
        if s.get("kind") == "tool_call" and s.get("tool_name") == "fs_read"
    ]
    results = [
        s
        for s in steps
        if s.get("kind") == "tool_result" and s.get("tool_name") == "fs_read"
    ]
    observed_calls = []
    for message in provider_request["messages"]:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] == "fs_read":
                observed_calls.append(
                    (json.loads(call["function"]["arguments"]), call["id"])
                )
        fence = re.fullmatch(
            r"\s*```tool_call\s+(.*?)\s*```\s*", message.get("content") or "", re.DOTALL
        )
        if fence:
            call = json.loads(fence[1])
            if call.get("name") == "fs_read":
                observed_calls.append((call.get("arguments", {}), None))
    observed_results = [
        text for name, text in tool_results(provider_request) if name == "fs_read"
    ]
    # AgentRunsDB deliberately omits file arguments/bodies. Require its real
    # outcome/correlation, and get the body from the HTTP exchange we observed.
    read = (
        len(calls) == len(results) == len(observed_calls) == len(observed_results) == 1
    )
    if read:
        args, native_id = observed_calls[0]
        result = observed_results[0]
        read = (
            bool(calls[0].get("call_id"))
            and results[0].get("call_id") == calls[0]["call_id"]
            and (native_id is None or native_id == calls[0]["call_id"])
            and results[0].get("tool_outcome") == "success"
            and args.get("root_alias") == alias
            and args.get("path") == "README.md"
            and code in result
            and not result.startswith("ERROR:")
        )
    assert read, "no successful read of the bound README with the fixture code"
    for label, text in (
        ("final_text", final_text),
        ("persisted_text", persisted_text),
        ("painted_text", painted_text),
    ):
        assert code in text, f"{label} has no README verification code"
    assert (
        persisted_text == final_text
    ), "persisted answer differs from completed answer"


def assert_project_ownership(run, saved, conversation_id, assistant_message_id):
    """The inspected persisted answer must be the owning run's actual anchor."""
    assert (
        run.get("conversation_id") == conversation_id
    ), "run conversation owner mismatch"
    assert (
        run.get("assistant_message_id") == assistant_message_id
    ), "run answer owner mismatch"
    assert saved.get("id") == assistant_message_id, "saved answer owner mismatch"
    assert (
        saved.get("conversation_id") == conversation_id
    ), "saved conversation owner mismatch"


def disclosed_schemas(payload):
    """Read schemas from the actual native array or fenced protocol system text."""
    native = payload.get("tools") or []
    if native:
        return {item["function"]["name"]: item["function"] for item in native}
    decoder = json.JSONDecoder()
    schemas = {}
    for message in payload["messages"]:
        if message.get("role") not in ("system", "developer"):
            continue
        text = message.get("content") or ""
        if not isinstance(text, str):
            continue
        for match in re.finditer(r'\{\s*"name"\s*:', text):
            try:
                obj, _ = decoder.raw_decode(text[match.start() :])
            except ValueError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("parameters"), dict):
                schemas[obj["name"]] = obj
    return schemas


def tool_results(payload):
    """Extract only actual protocol tool results, not a marker in assistant prose."""
    names = {}
    results = []
    for message in payload["messages"]:
        for call in message.get("tool_calls") or []:
            names[call["id"]] = call["function"]["name"]
        content = message.get("content") or ""
        if message.get("role") == "tool":
            results.append((names.get(message.get("tool_call_id"), ""), content))
        elif message.get("role") == "user" and isinstance(content, str):
            if content.startswith("Tool result for "):
                name, _, result = content.removeprefix("Tool result for ").partition(
                    ": "
                )
                results.append((name, result))
    return results


class ProjectChatModel:
    """A strict script that learns the answer only through returned file evidence."""

    def __init__(self, variant):
        assert variant in VARIANTS
        self.variant = variant
        self.received = []
        self.requests = []
        self.calls = []
        self.final = ""
        self.alias = ""
        self.lock = threading.Lock()
        self.sequence = (
            ["guess", "load", "list", "read", "final"]
            if variant == "compact-recovery"
            else ["find", "load"]
            + (["wrong"] if variant == "wrong-family" else [])
            + ["list", "read", "final"]
        )

    def reply(self, payload):
        with self.lock:
            self.received.append(copy.deepcopy(payload))
            return self._reply(payload)

    def _reply(self, payload):
        assert payload.get("model") == MODEL, "unexpected model identity"
        index = len(self.requests)
        assert index < len(self.sequence), "unexpected generation after final answer"
        schemas = disclosed_schemas(payload)
        if self.variant == "native":
            assert payload.get("tools"), "native schemas absent from actual request"
        else:
            assert not payload.get(
                "tools"
            ), "fenced variant unexpectedly used native schemas"
        results = tool_results(payload)
        if self.calls:
            expected = self.calls[-1]["name"]
            assert len(results) == len(
                self.calls
            ), f"expected one new {expected} result"
            assert results[-1][0] == expected, f"expected {expected} result"
            previous_stage = self.sequence[index - 1]
            last = results[-1][1]
            if previous_stage == "guess":
                assert "Tool schema not loaded" in last and "load_tools" in last
            elif previous_stage == "wrong":
                assert "ERROR:" in last and "does not accept root_alias or path" in last
            else:
                assert not last.startswith("ERROR:"), f"{expected} failed: {last[:350]}"
            if previous_stage == "list":
                assert "README.md" in last, "bound listing did not contain README.md"
            if previous_stage == "load":
                assert (
                    "fs_list" in schemas and "fs_read" in schemas
                ), "loaded schemas not disclosed"
        raw = json.dumps(payload["messages"], ensure_ascii=False)
        aliases = set(re.findall(r"folder-[a-f0-9]{12}", raw))
        if not aliases:
            for schema in schemas.values():
                aliases.update(
                    alias
                    for alias in schema["parameters"]
                    .get("properties", {})
                    .get("root_alias", {})
                    .get("enum", [])
                    if alias.startswith("folder-")
                )
        assert (
            len(aliases) == 1
        ), f"expected one advertised project binding, got {sorted(aliases)}"
        self.alias = next(iter(aliases))
        stage = self.sequence[index]
        if stage == "final":
            read = results[-1][1]
            match = re.search(r"Verification code:\s*([A-Za-z0-9-]+)", read)
            assert match, "README read did not supply a verification code"
            self.final = "Project README evidence:\n" + read
            self.requests.append(copy.deepcopy(payload))
            return {"role": "assistant", "content": self.final}
        names = {
            "find": "find_tools",
            "load": "load_tools",
            "guess": "fs_list",
            "wrong": "list_directory",
            "list": "fs_list",
            "read": "fs_read",
        }
        name = names[stage]
        if stage == "guess":
            assert (
                name not in schemas
            ), "recovery precondition: reader was already disclosed"
            assert (
                "load_tools" in schemas
            ), "recovery instruction would name an unavailable loader"
        else:
            assert name in schemas, f"{name} schema absent from actual request"
        if stage == "find":
            args = {"query": "fs_list fs_read list_directory"}
        elif stage == "load":
            args = {
                "ids": ["fs_list", "fs_read"]
                + (["list_directory"] if self.variant == "wrong-family" else [])
            }
        else:
            args = {
                "root_alias": self.alias,
                "path": "README.md" if stage == "read" else ".",
            }
        if stage not in ("guess", "wrong"):
            parameters = schemas[name]["parameters"]
            assert set(args) <= set(
                parameters.get("properties", {})
            ), f"{name}: argument schema missing"
            try:
                Draft202012Validator.check_schema(parameters)
                Draft202012Validator(parameters).validate(args)
            except (ValidationError, SchemaError) as error:
                raise AssertionError(
                    f"{name}: argument schema rejected the call: {error.message}"
                ) from error
        call = {"name": name, "arguments": args}
        self.calls.append(call)
        self.requests.append(copy.deepcopy(payload))
        if self.variant == "native":
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"uat-call-{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": "```tool_call\n" + json.dumps(call) + "\n```",
        }


@contextmanager
def project_chat_server(variant):
    """Own a bounded loopback OpenAI-compatible server; surface handler failures."""
    model = ProjectChatModel(variant)
    errors = []
    probes = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            probes.append(self.path)
            route = urlsplit(self.path).path
            if route == "/health":
                self.send_json({"status": "ok"})
            elif route in ("/models", "/v1/models"):
                entry = {"id": MODEL, "object": "model"}
                if variant != "compact-recovery":
                    entry["meta"] = {"n_ctx": 32768}
                self.send_json({"object": "list", "data": [entry]})
            elif route in ("/props", "/v1/props"):
                result = {
                    "chat_template_caps": {
                        "supports_tool_calls": variant == "native",
                        "supports_tools": variant == "native",
                    }
                }
                if variant != "compact-recovery":
                    result["default_generation_settings"] = {"n_ctx": 32768}
                self.send_json(result)
            else:
                errors.append(f"unexpected GET {self.path}")
                self.send_json({"error": errors[-1]}, 404)

        def do_POST(self):
            try:
                assert (
                    self.path == "/v1/chat/completions"
                ), f"unexpected POST {self.path}"
                length = int(self.headers.get("Content-Length", "0"))
                assert 0 < length <= 1024 * 1024, "invalid request body length"
                payload = json.loads(self.rfile.read(length))
                message = model.reply(payload)
                finish = "tool_calls" if message.get("tool_calls") else "stop"
                if payload.get("stream"):
                    delta = dict(message)
                    if delta.get("tool_calls"):
                        delta["tool_calls"] = [
                            {"index": i, **call}
                            for i, call in enumerate(delta["tool_calls"])
                        ]
                    chunks = [
                        {
                            "id": "uat-response",
                            "object": "chat.completion.chunk",
                            "model": MODEL,
                            "choices": [
                                {"index": 0, "delta": delta, "finish_reason": None}
                            ],
                        },
                        {
                            "id": "uat-response",
                            "object": "chat.completion.chunk",
                            "model": MODEL,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": finish}
                            ],
                        },
                    ]
                    body = (
                        "".join(
                            "data: " + json.dumps(chunk) + "\n\n" for chunk in chunks
                        )
                        + "data: [DONE]\n\n"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_json(
                        {
                            "id": "uat-response",
                            "object": "chat.completion",
                            "model": MODEL,
                            "choices": [
                                {
                                    "index": 0,
                                    "message": message,
                                    "finish_reason": finish,
                                }
                            ],
                        }
                    )
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                self.send_json({"error": errors[-1]}, 400)

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    http.daemon_threads = False
    thread = threading.Thread(
        target=lambda: http.serve_forever(poll_interval=0.05), name="uat-model-server"
    )
    thread.start()
    port = http.server_address[1]
    try:
        yield SimpleNamespace(
            model=model,
            errors=errors,
            probes=probes,
            port=port,
            url=f"http://127.0.0.1:{port}",
            thread=thread,
        )
    finally:
        http.shutdown()
        http.server_close()
        thread.join(5)
        assert not thread.is_alive(), "UAT model server did not retire"


@contextmanager
def only_owned_server(port):
    """Narrow the suite's loopback allowance without weakening local socketpair IPC."""
    attempts = []
    local = threading.local()
    connect, connect_ex, socketpair = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.socketpair,
    )

    def check(sock, address):
        if sock.family not in (socket.AF_INET, socket.AF_INET6) or getattr(
            local, "ipc", False
        ):
            return
        if tuple(address[:2]) != ("127.0.0.1", port):
            attempts.append(tuple(address[:2]))
            raise OSError(
                "UAT destination refused: only the owned model server is allowed"
            )

    def guarded_connect(sock, address):
        check(sock, address)
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        check(sock, address)
        return connect_ex(sock, address)

    def local_pair(*args, **kwargs):
        previous = getattr(local, "ipc", False)
        local.ipc = True
        try:
            return socketpair(*args, **kwargs)
        finally:
            local.ipc = previous

    with patch.object(socket.socket, "connect", guarded_connect), patch.object(
        socket.socket, "connect_ex", guarded_connect_ex
    ), patch.object(socket, "socketpair", local_pair):
        yield attempts
