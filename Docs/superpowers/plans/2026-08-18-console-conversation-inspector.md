# Console Conversation Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture every Console LLM exchange (request + response, per tool-loop call) at the gateway seam, persist it locally per turn, and replace the cost/context modals with one unified Conversation Inspector opened from the token/cost chip.

**Architecture:** A pure capture module builds allowlisted, binary-stubbed exchange records; `ConsoleProviderStreamSignals` carries them per run (mirroring its usage API); the gateway records per call; the store attaches/flushes them on usage's exact schedule into a new local-only ChaChaNotes `message_exchanges` table (zlib blobs); a three-tab modal (Costs / Exchange / Next Send) replaces `ConsoleCostModal` + `ConsoleContextModal`.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite, zlib, pytest.

**Spec:** `Docs/superpowers/specs/2026-08-18-console-conversation-inspector-design.md` — read it first; every task argues from it.

## Global Constraints

- Work happens in a git worktree off `origin/dev` under `<repo>/.worktrees/` (NEVER `/tmp`; NEVER the main checkout). Create a backlog task (`backlog task create` — sweep ALL remotes + worktrees for ID collisions first, assign against origin/dev) before the first commit.
- Run tests with the worktree venv: `VIRTUAL_ENV=.venv uv pip install -e ".[dev]"` if pytest is missing; a "no tests ran" result is a FAILED gate.
- Capture must never break a send: every capture-path exception is caught, logged via safe summaries (never raw payloads — loguru sinks leak), and degrades to "no capture".
- Redaction is allowlist-shaped by construction; a kwarg key not on the allowlist NEVER persists (its name goes to `omitted_keys`).
- No DB writes on the event-loop thread between send and first token; flush on usage's schedule only.
- **Re-anchored 2026-08-18 against origin/dev @ `1bdbcac61`** (spec/plan were first drafted against a stale branch): schema is now **v40 → v41**; the gateway uses a per-call `ConsoleProviderCallSignals` publishing into the aggregate `ConsoleProviderStreamSignals` (created via `new_usage_call()`), and adapter kwargs build via `_chat_api_kwargs_from_prepared(resolution, request)` from a `PreparedProviderRequest`. Task texts reflect this. Still **re-verify `_CURRENT_SCHEMA_VERSION` in `tldw_chatbook/DB/ChaChaNotes_DB.py` (line ~247) at Task 5 start**; renumber if it moved again, and update the sibling tests that hard-assert the constant (they are OUR regression when red).
- Worktree/branch/backlog task already exist: `.worktrees/console-conversation-inspector`, branch `feat/task-18300-console-conversation-inspector`, backlog `task-18300`. Do not create new ones.
- The default send path is the AGENT path (`_run_agent_reply`); every gateway-level test must cover a multi-call run through ONE signals object, not just a single direct call.
- Commit after every task; never push from a block that has not `cd`'d into the worktree and echoed `pwd` in the SAME block.

---

### Task 1: Pure capture module (`console_exchange_capture.py`)

**Files:**
- Create: `tldw_chatbook/Chat/console_exchange_capture.py`
- Test: `Tests/Chat/test_console_exchange_capture.py`

**Interfaces:**
- Consumes: `ProviderUsage` (`tldw_chatbook/Chat/provider_usage.py` — has `.to_json()`, `.from_json()`).
- Produces (later tasks rely on these exact names):
  - `ExchangeCapture` frozen dataclass (fields below)
  - `build_request_capture(kwargs: Mapping[str, Any]) -> tuple[dict, tuple[str, ...]]` — (allowlisted+stubbed request dict, omitted key names)
  - `stub_binary_strings(obj: Any) -> Any` — recursive stubber
  - `capture_to_blob(capture: ExchangeCapture) -> bytes` / `capture_from_blob(blob: bytes) -> ExchangeCapture`
  - `CAPTURE_REQUEST_ALLOWLIST: frozenset[str]`
  - `EXCHANGE_BLOB_MAX_BYTES = 16 * 1024 * 1024`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_exchange_capture.py
"""Capture-core tests: allowlist by construction, stubbing, blob round-trip."""
import json
import zlib

from hypothesis import given, strategies as st

from tldw_chatbook.Chat.console_exchange_capture import (
    CAPTURE_REQUEST_ALLOWLIST,
    EXCHANGE_BLOB_MAX_BYTES,
    ExchangeCapture,
    build_request_capture,
    capture_from_blob,
    capture_to_blob,
    stub_binary_strings,
)


def _kwargs():
    return {
        "api_endpoint": "anthropic",
        "system_message": "You are helpful.",
        "messages_payload": [{"role": "user", "content": "hi"}],
        "api_key": "sk-SECRET",
        "model": "claude-sonnet-5",
        "streaming": True,
        "temp": 0.7,
        "tools": [{"type": "function", "function": {"name": "get_time"}}],
    }


def test_api_key_never_in_capture_and_named_in_omitted():
    request, omitted = build_request_capture(_kwargs())
    assert "sk-SECRET" not in json.dumps(request)
    assert "api_key" not in request
    assert "api_key" in omitted


@given(st.text(min_size=1, max_size=30).filter(lambda k: k not in CAPTURE_REQUEST_ALLOWLIST))
def test_unknown_kwarg_never_persists(key):
    request, omitted = build_request_capture({**_kwargs(), key: "future-secret"})
    assert key not in request
    assert "future-secret" not in json.dumps(request)
    assert key in omitted


def test_allowlisted_content_survives_verbatim():
    request, _ = build_request_capture(_kwargs())
    assert request["system_message"] == "You are helpful."
    assert request["messages_payload"] == [{"role": "user", "content": "hi"}]
    assert request["tools"][0]["function"]["name"] == "get_time"


def test_base64_data_uri_is_stubbed_deterministically():
    blob = "data:image/png;base64," + ("QUJD" * 2000)
    row = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": blob}}]}
    first = stub_binary_strings(row)
    second = stub_binary_strings(row)
    text = json.dumps(first)
    assert "QUJDQUJD" not in text
    assert "image/png" in text and "sha256:" in text
    assert first == second


def test_anthropic_source_b64_is_stubbed():
    row = {"role": "user", "content": [{"type": "image", "source": {
        "type": "base64", "media_type": "image/jpeg", "data": "QUJE" * 2000}}]}
    text = json.dumps(stub_binary_strings(row))
    assert "QUJEQUJE" not in text and "image/jpeg" in text


def test_short_strings_untouched():
    row = {"role": "user", "content": "data:image/png;base64,QUJD"}
    assert stub_binary_strings(row) == row


def test_blob_round_trip():
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="2026-08-18T00:00:00Z",
        provider="anthropic", model="claude-sonnet-5", endpoint=None,
        request={"model": "claude-sonnet-5"}, response={"content": "hello"},
        status="complete", usage_json=None, omitted_keys=("api_key",),
    )
    assert capture_from_blob(capture_to_blob(cap)) == cap


def test_blob_is_compressed_json():
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="t", provider="p", model="m",
        endpoint=None, request={"system_message": "x" * 5000},
        response={}, status="complete", usage_json=None, omitted_keys=(),
    )
    blob = capture_to_blob(cap)
    assert len(blob) < 5000
    assert json.loads(zlib.decompress(blob))["request"]["system_message"] == "x" * 5000


def test_oversize_blob_truncates_with_marker():
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="t", provider="p", model="m",
        endpoint=None,
        request={"messages_payload": [{"role": "user", "content": __import__("os").urandom(9 * 1024 * 1024).hex()}]},
        response={}, status="complete", usage_json=None, omitted_keys=(),
    )
    blob = capture_to_blob(cap)
    assert len(blob) <= EXCHANGE_BLOB_MAX_BYTES
    restored = capture_from_blob(blob)
    assert restored.status == "truncated"
    assert "truncated" in json.dumps(restored.request)


def test_unserializable_value_degrades_not_raises():
    request, _ = build_request_capture({**_kwargs(), "tools": [object()]})
    json.dumps(request)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_exchange_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Chat.console_exchange_capture`

- [ ] **Step 3: Implement the module**

```python
# tldw_chatbook/Chat/console_exchange_capture.py
"""Pure exchange-capture records for the Console Conversation Inspector.

No I/O here: builders take the gateway's ``chat_api_call`` kwargs and
produce allowlisted, binary-stubbed, blob-serializable records. The
allowlist is the contract: a kwarg key not named below NEVER persists
(spec: Docs/superpowers/specs/2026-08-18-console-conversation-inspector-design.md).
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import zlib
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

CAPTURE_REQUEST_ALLOWLIST: frozenset[str] = frozenset({
    "api_endpoint", "api_base_url", "system_message", "messages_payload",
    "tools", "model", "streaming", "temp", "topp", "maxp", "topk", "minp",
    "max_tokens", "seed", "presence_penalty", "frequency_penalty",
    "reasoning_effort", "reasoning_summary", "verbosity", "thinking_effort",
    "thinking_budget_tokens", "prompt_caching", "response_format",
    "api_mode", "request_timeout", "request_retries", "request_retry_delay",
    "provider_continuations",
})
# Deliberately OFF the allowlist: "api_key" (credential) and
# "api_key_resolved" (credential-adjacent marker) — they surface in
# omitted_keys instead.

#: Strings at/above this length are candidates for base64 stubbing.
_STUB_MIN_CHARS = 4096
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")
_DATA_URI_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+);base64,(?P<data>.+)$", re.DOTALL)

EXCHANGE_BLOB_MAX_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ExchangeCapture:
    """One provider call's captured request/response pair."""

    run_tag: str
    seq: int
    created_at: str
    provider: str
    model: str
    endpoint: str | None
    request: dict
    response: dict
    status: str  # "complete" | "stopped" | "error" | "truncated"
    usage_json: str | None  # THIS call's normalized ProviderUsage.to_json()
    omitted_keys: tuple[str, ...]


def _stub_for(data: str, mime: str) -> str:
    digest = hashlib.sha256(data.encode("utf-8", errors="replace")).hexdigest()[:16]
    approx = (len(data) * 3) // 4
    if approx >= 1024 * 1024:
        size = f"{approx / (1024 * 1024):.1f} MB"
    else:
        size = f"{approx / 1024:.1f} KB"
    return f"[{mime}, {size}, sha256:{digest}]"


def _maybe_stub_string(value: str, mime_hint: str | None = None) -> str:
    if len(value) < _STUB_MIN_CHARS:
        return value
    match = _DATA_URI_RE.match(value)
    if match:
        return _stub_for(match.group("data"), match.group("mime"))
    if _BASE64_RE.match(value):
        try:
            base64.b64decode(value[:4096], validate=True)
        except Exception:
            return value
        return _stub_for(value, mime_hint or "application/octet-stream")
    return value


def stub_binary_strings(obj: Any) -> Any:
    """Recursively replace base64/data-URI payloads with honest stubs.

    Deterministic: identical input bytes always produce the identical stub
    (size + sha256 prefix), so a viewer can verify attachment identity
    across calls without the bytes themselves.
    """
    if isinstance(obj, str):
        return _maybe_stub_string(obj)
    if isinstance(obj, Mapping):
        mime_hint = obj.get("media_type") or obj.get("mime_type")
        out = {}
        for key, value in obj.items():
            if key in {"data", "b64_json"} and isinstance(value, str):
                out[key] = _maybe_stub_string(value, mime_hint if isinstance(mime_hint, str) else None)
            else:
                out[key] = stub_binary_strings(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [stub_binary_strings(item) for item in obj]
    return obj


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return json.loads(json.dumps(obj, default=str))


def build_request_capture(kwargs: Mapping[str, Any]) -> tuple[dict, tuple[str, ...]]:
    """Return (allowlisted+stubbed request dict, names of dropped keys)."""
    request: dict = {}
    omitted: list[str] = []
    for key, value in kwargs.items():
        if key in CAPTURE_REQUEST_ALLOWLIST:
            request[key] = stub_binary_strings(_jsonable(value))
        else:
            omitted.append(str(key))
    return request, tuple(sorted(omitted))


def capture_to_blob(capture: ExchangeCapture) -> bytes:
    """zlib-compressed JSON; oversize captures truncate, never fail."""
    blob = zlib.compress(json.dumps(asdict(capture), default=str).encode("utf-8"))
    if len(blob) <= EXCHANGE_BLOB_MAX_BYTES:
        return blob
    truncated = replace(
        capture,
        status="truncated",
        request={"truncated": f"capture exceeded {EXCHANGE_BLOB_MAX_BYTES} bytes compressed"},
        response={"truncated": True},
    )
    return zlib.compress(json.dumps(asdict(truncated), default=str).encode("utf-8"))


def capture_from_blob(blob: bytes) -> ExchangeCapture:
    data = json.loads(zlib.decompress(blob))
    data["omitted_keys"] = tuple(data.get("omitted_keys") or ())
    return ExchangeCapture(**data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_exchange_capture.py -v`
Expected: all PASS (Hypothesis is a dev dependency; if the import fails, install `.[dev]` per Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_exchange_capture.py Tests/Chat/test_console_exchange_capture.py
git commit -m "feat(console): exchange-capture core — allowlist, stubbing, blobs"
```

---

### Task 2: Exchange API on the stream-signals pair

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py:124-315` — BOTH classes: the aggregate `ConsoleProviderStreamSignals` and the per-call `ConsoleProviderCallSignals` (created by `aggregate.new_usage_call()`; a call publishes into its aggregate via a private `_token`). Exchange capture mirrors the scoped-usage design exactly.
- Test: `Tests/Chat/test_console_provider_gateway.py` (append new test class)

**Interfaces:**
- Consumes: `ExchangeCapture` from Task 1; `ProviderUsage.from_provider_payload` (`tldw_chatbook/Chat/provider_usage.py`).
- Produces (Tasks 3, 4, 7 rely on these exact names):
  - aggregate field `run_tag: str` (uuid4 hex, minted per signals object = per dispatch)
  - aggregate field `exchange_capture_enabled: bool = True` (controller sets from config, Task 7)
  - aggregate `exchange_captures() -> list[ExchangeCapture]` — completed calls + in-flight tails (tails render as `"stopped"`), mirroring `usage_payloads()`
  - on `ConsoleProviderCallSignals`:
    - `begin_exchange(*, provider: str, model: str, endpoint: str | None, request: dict, omitted_keys: tuple[str, ...]) -> None` — no-ops when the aggregate's `exchange_capture_enabled` is False
    - `record_exchange_content(text: str) -> None`
    - `record_exchange_tool_calls(calls: Sequence[Mapping]) -> None`
    - `close_exchange(status: str = "complete") -> None` — computes this call's `usage_json` itself from `self.usage_snapshot()` + the flight's provider/model via `ProviderUsage.from_provider_payload`; idempotent (move semantics, second close is a no-op)

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_provider_gateway.py`:

```python
class TestSignalsExchangeCapture:
    @staticmethod
    def _begin(call, label="hi"):
        call.begin_exchange(
            provider="anthropic", model="m", endpoint=None,
            request={"messages_payload": [{"role": "user", "content": label}]},
            omitted_keys=("api_key",),
        )

    def test_per_call_boundaries_never_merge(self):
        aggregate = ConsoleProviderStreamSignals()
        call0 = aggregate.new_usage_call()
        self._begin(call0, "call0")
        call0.record_exchange_content("hel")
        call0.record_exchange_content("lo")
        call0.close_exchange()
        call1 = aggregate.new_usage_call()
        self._begin(call1, "call1")
        call1.record_exchange_content("again")
        call1.close_exchange()
        captures = aggregate.exchange_captures()
        assert [c.seq for c in captures] == [0, 1]
        assert captures[0].response["content"] == "hello"
        assert captures[1].response["content"] == "again"
        assert captures[0].run_tag == captures[1].run_tag == aggregate.run_tag

    def test_in_flight_tail_reports_stopped(self):
        aggregate = ConsoleProviderStreamSignals()
        call = aggregate.new_usage_call()
        self._begin(call)
        call.record_exchange_content("part")
        captures = aggregate.exchange_captures()
        assert len(captures) == 1
        assert captures[0].status == "stopped"
        assert captures[0].response["content"] == "part"

    def test_close_moves_never_copies(self):
        aggregate = ConsoleProviderStreamSignals()
        call = aggregate.new_usage_call()
        self._begin(call)
        call.close_exchange()
        call.close_exchange()  # second close is a no-op
        assert len(aggregate.exchange_captures()) == 1

    def test_tool_calls_recorded(self):
        aggregate = ConsoleProviderStreamSignals()
        call = aggregate.new_usage_call()
        self._begin(call)
        call.record_exchange_tool_calls([{"id": "t1", "function": {"name": "get_time"}}])
        call.close_exchange()
        assert aggregate.exchange_captures()[0].response["tool_calls"][0]["id"] == "t1"

    def test_close_attaches_this_calls_normalized_usage(self):
        aggregate = ConsoleProviderStreamSignals()
        call = aggregate.new_usage_call()
        self._begin(call)
        call.record_usage_payload({"prompt_tokens": 10, "completion_tokens": 5})
        call.close_exchange()
        cap = aggregate.exchange_captures()[0]
        assert cap.usage_json is not None
        from tldw_chatbook.Chat.provider_usage import ProviderUsage
        usage = ProviderUsage.from_json(cap.usage_json)
        assert usage is not None and usage.total_tokens == 15

    def test_disabled_records_nothing(self):
        aggregate = ConsoleProviderStreamSignals(exchange_capture_enabled=False)
        call = aggregate.new_usage_call()
        self._begin(call)
        call.record_exchange_content("x")
        call.close_exchange()
        assert aggregate.exchange_captures() == []

    def test_run_tags_differ_across_signals_objects(self):
        assert ConsoleProviderStreamSignals().run_tag != ConsoleProviderStreamSignals().run_tag
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -k ExchangeCapture -v`
Expected: FAIL — `begin_exchange` not defined (and `exchange_capture_enabled` unexpected kwarg).

- [ ] **Step 3: Implement on both signals classes**

Both classes are `@dataclass(slots=True)` — new fields MUST go at the end of each class's field list and all carry defaults. Imports to add at the top of `console_provider_gateway.py`: `import uuid`, `from datetime import datetime, timezone`, `from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture, build_request_capture, stub_binary_strings` (the latter two are used by Tasks 3-4), `from tldw_chatbook.Chat.provider_usage import ProviderUsage`.

On the AGGREGATE (`ConsoleProviderStreamSignals`), new fields + methods (mirror the `_record_scoped_usage_call`/`_complete_scoped_usage_call` pattern directly above them):

```python
    run_tag: str = field(default_factory=lambda: uuid.uuid4().hex)
    exchange_capture_enabled: bool = True
    completed_exchanges: list["ExchangeCapture"] = field(default_factory=list, repr=False)
    _active_exchanges: dict[object, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False)
    _exchange_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False)

    def _begin_scoped_exchange(self, token: object, flight: dict[str, Any]) -> None:
        with self._exchange_lock:
            self._active_exchanges[token] = flight

    def _mutate_scoped_exchange(self, token: object, key: str, items: list) -> None:
        with self._exchange_lock:
            flight = self._active_exchanges.get(token)
            if flight is not None:
                flight[key].extend(items)

    def _complete_scoped_exchange(
        self, token: object, status: str,
        usage_payload: dict[str, Any] | None,
    ) -> None:
        with self._exchange_lock:
            flight = self._active_exchanges.pop(token, None)
            if flight is None:
                return
            self.completed_exchanges.append(_flight_capture(
                self.run_tag, len(self.completed_exchanges), flight,
                status, usage_payload))

    def exchange_captures(self) -> list["ExchangeCapture"]:
        """Completed calls + in-flight tails (as "stopped") — tails cover
        aborted streams whose generator never reached its own close-out,
        mirroring usage_payloads()."""
        with self._exchange_lock:
            captures = list(self.completed_exchanges)
            for flight in self._active_exchanges.values():
                captures.append(_flight_capture(
                    self.run_tag, len(captures), flight, "stopped", None))
            return captures
```

Module-level helper (near `safe_provider_error_copy`):

```python
def _flight_capture(run_tag: str, seq: int, flight: dict[str, Any],
                    status: str, usage_payload: dict[str, Any] | None) -> ExchangeCapture:
    """Build the immutable capture for one call's in-flight record.

    Normalizes THIS call's usage payload on its own (never a cross-call
    merge — the same disjoint-buckets rule the aggregate documents).
    """
    usage_json = None
    if usage_payload:
        try:
            usage = ProviderUsage.from_provider_payload(
                usage_payload, provider=flight["provider"], model=flight["model"])
            usage_json = usage.to_json() if usage is not None else None
        except Exception:
            usage_json = None
    return ExchangeCapture(
        run_tag=run_tag, seq=seq, created_at=flight["created_at"],
        provider=flight["provider"], model=flight["model"],
        endpoint=flight["endpoint"], request=flight["request"],
        response={"content": "".join(flight["content"]),
                  "tool_calls": list(flight["tool_calls"])},
        status=status, usage_json=usage_json,
        omitted_keys=flight["omitted_keys"],
    )
```

On `ConsoleProviderCallSignals` (new methods at the end; it publishes through its existing `_aggregate`/`_token`):

```python
    def begin_exchange(self, *, provider: str, model: str, endpoint: str | None,
                       request: dict, omitted_keys: tuple[str, ...]) -> None:
        """Open this call's capture. ONE stream_chat invocation == one
        exchange; close_exchange in stream_chat's finally is the close site."""
        if not self._aggregate.exchange_capture_enabled:
            return
        self._aggregate._begin_scoped_exchange(self._token, {
            "provider": provider, "model": model, "endpoint": endpoint,
            "request": request, "omitted_keys": omitted_keys,
            "content": [], "tool_calls": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    def record_exchange_content(self, text: str) -> None:
        if text:
            self._aggregate._mutate_scoped_exchange(self._token, "content", [text])

    def record_exchange_tool_calls(self, calls: "Sequence[Mapping[str, Any]]") -> None:
        self._aggregate._mutate_scoped_exchange(
            self._token, "tool_calls", [dict(c) for c in calls])

    def close_exchange(self, status: str = "complete") -> None:
        """Publish this call's capture exactly once (token pop = move
        semantics; a second close finds nothing)."""
        self._aggregate._complete_scoped_exchange(
            self._token, status, self.usage_snapshot())
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -k ExchangeCapture -v` then the whole file: `pytest Tests/Chat/test_console_provider_gateway.py -q`
Expected: new tests PASS; zero pre-existing failures introduced.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "feat(console): exchange-capture API on stream signals"
```

---

### Task 3: Gateway hooks — generic `chat_api_call` path

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` — `stream_chat` (~line 1499, the `finally`) and `_stream_generic_chat.worker()` (~line 1586)
- Test: `Tests/Chat/test_console_provider_gateway.py`

**Interfaces:**
- Consumes: Task 2's signals API; `build_request_capture` (Task 1); existing `_chat_api_kwargs`.
- Produces: after any `stream_chat` call with signals, `signals.exchange_captures()` holds one capture per call with the real request kwargs (minus allowlist drops) and accumulated response.

- [ ] **Step 1: Write the failing tests**

Append (reuse the file's existing fake `chat_api_call_fn` fixtures/idioms — read the file's existing `ConsoleProviderGateway(chat_api_call_fn=...)` tests first and copy their construction shape):

```python
class TestGatewayExchangeCapture:
    @staticmethod
    def _resolution():
        # Copy the file's existing helper for building a ready
        # ConsoleProviderResolution (execution_key="openai", model set,
        # ready=True, streaming=False); do not invent a new shape.
        ...

    @staticmethod
    async def _drain(gen):
        return [chunk async for chunk in gen]

    async def test_one_capture_per_call_with_request_and_response(self):
        calls = []
        def fake_chat_api_call(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "pong"}}]}
        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals()
        resolution = self._resolution()
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "ping"}]
        await self._drain(gateway.stream_chat(resolution, messages, signals=signals))
        await self._drain(gateway.stream_chat(resolution, messages, signals=signals))
        captures = signals.exchange_captures()
        assert len(captures) == 2
        assert captures[0].status == "complete"
        assert captures[0].request["system_message"] == "sys"
        assert captures[0].request["messages_payload"] == [{"role": "user", "content": "ping"}]
        assert "api_key" not in captures[0].request
        assert "api_key" in captures[0].omitted_keys
        assert captures[0].response["content"] == "pong"

    async def test_transcript_output_byte_identical_with_capture(self):
        def fake_chat_api_call(**kwargs):
            return {"choices": [{"message": {"content": "exact bytes"}}]}
        resolution = self._resolution()
        messages = [{"role": "user", "content": "q"}]
        with_signals = await self._drain(
            ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
            .stream_chat(resolution, messages, signals=ConsoleProviderStreamSignals()))
        without = await self._drain(
            ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
            .stream_chat(resolution, messages, signals=None))
        assert with_signals == without

    async def test_provider_error_closes_capture_as_error(self):
        def fake_chat_api_call(**kwargs):
            raise RuntimeError("boom")
        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals()
        with pytest.raises(Exception):
            await self._drain(gateway.stream_chat(self._resolution(),
                              [{"role": "user", "content": "q"}], signals=signals))
        captures = signals.exchange_captures()
        assert len(captures) == 1 and captures[0].status == "error"

    async def test_disabled_signals_capture_nothing(self):
        def fake_chat_api_call(**kwargs):
            return {"choices": [{"message": {"content": "pong"}}]}
        gateway = ConsoleProviderGateway(chat_api_call_fn=fake_chat_api_call)
        signals = ConsoleProviderStreamSignals(exchange_capture_enabled=False)
        await self._drain(gateway.stream_chat(self._resolution(),
                          [{"role": "user", "content": "q"}], signals=signals))
        assert signals.exchange_captures() == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -k GatewayExchangeCapture -v`
Expected: FAIL — captures list empty (hooks not wired).

- [ ] **Step 3: Wire the hooks**

Context: on dev, `stream_chat` (line ~2201) wraps any aggregate `signals` into a per-call `call_signals = signals.new_usage_call()` and passes it to `_stream_generic_chat(effective_resolution, prepared, signals=call_signals)` — so inside `_stream_generic_chat` and its `worker()`, `signals` IS the call-scoped `ConsoleProviderCallSignals`. Tests that pass an aggregate to `stream_chat` read captures back from the aggregate.

In `_stream_generic_chat.worker()` (line ~2362), immediately after `kwargs = self._chat_api_kwargs_from_prepared(resolution, request)`:

```python
                if signals is not None:
                    try:
                        capture_request, omitted = build_request_capture(kwargs)
                        signals.begin_exchange(
                            provider=str(resolution.provider or ""),
                            model=str(resolution.model or ""),
                            endpoint=getattr(resolution, "base_url", None),
                            request=capture_request, omitted_keys=omitted,
                        )
                    except Exception:
                        logger.opt(exception=True).warning("exchange_capture_begin_failed")
```

(No enabled-flag check at the hook — `begin_exchange` itself no-ops when the aggregate's `exchange_capture_enabled` is False, and every later record against a never-begun exchange is a no-op by construction.)

In the content loop (a `while not stop_event.is_set():` / `text = next(normalized_response)` shape on dev), right after `if text:` sets `emitted_content = True`:

```python
                    if signals is not None and text:
                        signals.record_exchange_content(text)
```

After `calls = accumulator.calls()`, when `calls` is truthy (leave the `metadata` handling untouched):

```python
                    if signals is not None and calls:
                        signals.record_exchange_tool_calls(calls)
```

In the worker's `except BaseException` handler, before enqueueing the error item:

```python
                if signals is not None:
                    signals.close_exchange(status="error")
```

In `stream_chat`'s `finally` (line ~2311), close the exchange BEFORE the usage call so `usage_snapshot()` is read for this call while its identity is unambiguous:

```python
        finally:
            if call_signals is not None:
                call_signals.close_exchange()
                call_signals.close_usage_call()
```

(`close_exchange()` after the worker's error-close is a no-op — Task 2's token-pop move semantics guarantee it. `close_exchange` computes the per-call `usage_json` internally; no helper needed here.)

- [ ] **Step 4: Run to verify pass, plus the whole gateway file**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -q`
Expected: all PASS, including the pre-existing cache-stability and signals tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "feat(console): capture exchanges at the chat_api_call gateway seam"
```

---

### Task 4: Gateway hooks — llama.cpp branch (wire-literal)

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` — the `llama_cpp`/`local_llamacpp` branch of `stream_chat` (lines ~1531-1557)
- Test: `Tests/Chat/test_console_provider_gateway.py`

**Interfaces:**
- Consumes: `build_llamacpp_chat_payload` (same file, line ~582), Task 2 signals API.
- Produces: llama.cpp calls produce captures whose `request` is the literal wire payload (this branch builds its own HTTP body — the one place capture IS wire-literal; spec Non-goals).

- [ ] **Step 1: Write the failing test**

```python
class TestLlamaCppExchangeCapture:
    async def test_llamacpp_capture_is_wire_literal_and_keyless(self, monkeypatch):
        import json as _json
        gateway = ConsoleProviderGateway()
        streamed = ["hel", "lo"]
        async def fake_stream(self, **kwargs):
            for chunk in streamed:
                yield chunk
        monkeypatch.setattr(ConsoleProviderGateway, "stream_llamacpp_chat", fake_stream)
        aggregate = ConsoleProviderStreamSignals()
        resolution = ...  # file's existing llama_cpp resolution helper: provider="llama_cpp", ready, streaming=True, api_key="local-secret"
        out = [c async for c in gateway.stream_chat(
            resolution, [{"role": "user", "content": "q"}], signals=aggregate)]
        assert out == streamed
        captures = aggregate.exchange_captures()
        assert len(captures) == 1
        wire = captures[0].request["wire_payload"]
        assert wire["messages"][-1]["content"] == "q"
        assert captures[0].response["content"] == "hello"
        # resolution.api_key rides stream_llamacpp_chat's kwargs (headers),
        # never the wire body — the capture must contain no trace of it.
        assert "local-secret" not in _json.dumps(captures[0].request)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -k LlamaCpp -k Exchange -v`
Expected: FAIL — no captures.

- [ ] **Step 3: Implement**

In `stream_chat`'s llama branch (line ~2272), the wire messages are already built: `wire_messages = [thaw_json(item) for item in prepared.messages]`. Before the streaming loop (and analogously before `complete_llamacpp_chat` on the non-streaming side), hook `call_signals`:

```python
                if call_signals is not None:
                    try:
                        wire = build_llamacpp_chat_payload(
                            model=resolution.model, messages=wire_messages,
                            temperature=resolution.temperature,
                            top_p=resolution.top_p, min_p=resolution.min_p,
                            top_k=resolution.top_k,
                            max_tokens=effective_resolution.max_tokens,
                            stream=resolution.streaming,
                        )
                        capture_request, omitted = build_request_capture(
                            {"model": resolution.model})
                        capture_request["wire_payload"] = stub_binary_strings(wire)
                        call_signals.begin_exchange(
                            provider=str(resolution.provider or ""),
                            model=str(resolution.model or ""),
                            endpoint=normalize_llamacpp_base_url(resolution.base_url),
                            request=capture_request, omitted_keys=omitted,
                        )
                    except Exception:
                        logger.opt(exception=True).warning("exchange_capture_begin_failed")
```

**First match `build_llamacpp_chat_payload`'s real signature at line ~847 and pass exactly its parameters** — the call above is indicative, not gospel (dev's `stream_llamacpp_chat` now also takes `reasoning_effort`/`thinking_budget_tokens`/`api_key`; if the payload builder gained those, forward them — but `api_key` must NEVER enter the captured payload, which the Step 1 test pins). Then record each yielded chunk (`call_signals.record_exchange_content(chunk)`) in both the streaming loop and after the non-streaming completion. The existing `stream_chat` `finally` from Task 3 already closes the exchange.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_provider_gateway.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "feat(console): wire-literal exchange capture on the llama.cpp branch"
```

---

### Task 5: ChaChaNotes `message_exchanges` table (v41)

**Files:**
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` — `_CURRENT_SCHEMA_VERSION` (~line 247), new `_migrate_from_v40_to_v41` (after the last existing `_migrate_from_v*` method), the `migration_steps` dict (grep `migration_steps = {` / `39: self._migrate_from_v39_to_v40`), new public methods near `update_message_usage`
- Test: `Tests/DB/test_chachanotes_message_exchanges.py` (create)

**Interfaces:**
- Produces (Task 6 relies on):
  - `CharactersRAGDB.append_message_exchanges_local(message_id: str, rows: Sequence[Mapping[str, Any]]) -> int` — upsert; each row: `{"run_tag": str, "seq": int, "status": str, "abandoned": bool, "capture_blob": bytes, "created_at": str}`; returns rows written
  - `CharactersRAGDB.get_message_exchanges(message_id: str) -> list[dict]` — ordered by (run_tag, seq); each dict carries the row columns with `capture_blob` as bytes

- [ ] **Step 0: Re-verify the version constant** (Global Constraints). Run `grep -n "_CURRENT_SCHEMA_VERSION" tldw_chatbook/DB/ChaChaNotes_DB.py` — it was 40 at re-anchor time; if it moved again, renumber every "41" in this task accordingly.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/DB/test_chachanotes_message_exchanges.py
"""v33 message_exchanges: local-only, idempotent upsert, cascade delete."""
import pytest

# Copy the in-memory CharactersRAGDB fixture idiom from the existing
# Tests/DB test modules (real SQLite in-memory, no mocks) — e.g. the
# fixture used by the usage_json/v30 tests.


def _seed_message(db):
    # Use the DB's real public API to create a character/conversation/message;
    # copy the seeding helper from the v30 usage tests. Returns message_id.
    ...


def test_append_and_read_round_trip(db):
    mid = _seed_message(db)
    rows = [
        {"run_tag": "r1", "seq": 0, "status": "complete", "abandoned": False,
         "capture_blob": b"blob0", "created_at": "2026-08-18T00:00:00Z"},
        {"run_tag": "r1", "seq": 1, "status": "stopped", "abandoned": False,
         "capture_blob": b"blob1", "created_at": "2026-08-18T00:00:01Z"},
    ]
    assert db.append_message_exchanges_local(mid, rows) == 2
    stored = db.get_message_exchanges(mid)
    assert [(r["run_tag"], r["seq"], r["capture_blob"]) for r in stored] == [
        ("r1", 0, b"blob0"), ("r1", 1, b"blob1")]


def test_upsert_idempotent_and_updates_in_place(db):
    mid = _seed_message(db)
    row = {"run_tag": "r1", "seq": 0, "status": "complete", "abandoned": False,
           "capture_blob": b"v1", "created_at": "t"}
    db.append_message_exchanges_local(mid, [row])
    db.append_message_exchanges_local(mid, [{**row, "capture_blob": b"v2", "abandoned": True}])
    stored = db.get_message_exchanges(mid)
    assert len(stored) == 1
    assert stored[0]["capture_blob"] == b"v2" and stored[0]["abandoned"]


def test_no_sync_log_rows_written(db):
    mid = _seed_message(db)
    with db.transaction() as cursor:
        before = cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
    db.append_message_exchanges_local(mid, [
        {"run_tag": "r1", "seq": 0, "status": "complete", "abandoned": False,
         "capture_blob": b"b", "created_at": "t"}])
    with db.transaction() as cursor:
        after = cursor.execute("SELECT COUNT(*) FROM sync_log").fetchone()[0]
    assert after == before


def test_hard_delete_cascades(db):
    mid = _seed_message(db)
    db.append_message_exchanges_local(mid, [
        {"run_tag": "r1", "seq": 0, "status": "complete", "abandoned": False,
         "capture_blob": b"b", "created_at": "t"}])
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM messages WHERE id = ?", (mid,))
        count = cursor.execute(
            "SELECT COUNT(*) FROM message_exchanges").fetchone()[0]
    assert count == 0


def test_schema_version_is_41(db):
    # Mirrors the house sibling-version test pattern; update alongside the
    # existing version-pin tests, which MUST also be bumped in this task.
    assert db._get_db_version_public_or_equivalent() == 41  # use the accessor the sibling tests use
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/DB/test_chachanotes_message_exchanges.py -v`
Expected: FAIL — no `append_message_exchanges_local`, version still 40.

- [ ] **Step 3: Implement migration + methods**

Follow the `_migrate_from_v29_to_v30` shape EXACTLY (version pre-check raising `SchemaError`, idempotence guard — for a CREATE TABLE the guard is `CREATE TABLE IF NOT EXISTS` itself — guarded version UPDATE with rowcount check, final version verify, `sqlite3.Error` logging; find it by name, it is the `usage_json` local-only precedent this feature extends). DDL:

```sql
CREATE TABLE IF NOT EXISTS message_exchanges(
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id   TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  run_tag      TEXT NOT NULL,
  seq          INTEGER NOT NULL,
  status       TEXT NOT NULL,
  abandoned    BOOLEAN NOT NULL DEFAULT 0,
  capture_blob BLOB NOT NULL,
  created_at   TEXT NOT NULL,
  UNIQUE(message_id, run_tag, seq)
);
CREATE INDEX IF NOT EXISTS idx_message_exchanges_message
  ON message_exchanges(message_id);
```

Add the same DDL to the base schema script (near the messages table's CREATE block) so fresh databases get it without migrating; NO sync triggers, NO FTS. Bump `_CURRENT_SCHEMA_VERSION` to 41 and register `40: self._migrate_from_v40_to_v41` in the `migration_steps` dict.

Public methods (place beside `update_message_usage`, matching its docstring/local-only framing):

```python
    def append_message_exchanges_local(self, message_id, rows):
        """Upsert exchange captures. Local-only by design: never touches
        sync_log, never bumps the parent message's version (usage_json
        precedent, v30)."""
        written = 0
        with self.transaction() as cursor:
            for row in rows:
                cursor.execute(
                    """
                    INSERT INTO message_exchanges
                        (message_id, run_tag, seq, status, abandoned,
                         capture_blob, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(message_id, run_tag, seq) DO UPDATE SET
                        status = excluded.status,
                        abandoned = excluded.abandoned,
                        capture_blob = excluded.capture_blob
                    """,
                    (message_id, row["run_tag"], int(row["seq"]),
                     row["status"], 1 if row.get("abandoned") else 0,
                     row["capture_blob"], row["created_at"]),
                )
                written += 1
        return written

    def get_message_exchanges(self, message_id):
        """Ordered captures for one message (run_tag, then seq)."""
        with self.transaction() as cursor:
            cursor.execute(
                """
                SELECT run_tag, seq, status, abandoned, capture_blob, created_at
                  FROM message_exchanges
                 WHERE message_id = ?
                 ORDER BY run_tag, seq
                """,
                (message_id,),
            )
            return [
                {"run_tag": r[0], "seq": r[1], "status": r[2],
                 "abandoned": bool(r[3]), "capture_blob": r[4],
                 "created_at": r[5]}
                for r in cursor.fetchall()
            ]
```

Update every sibling test that hard-asserts the version constant (grep `-rn "== 40\|_CURRENT_SCHEMA_VERSION" Tests/` and fix each; those failures are OURS, not pre-existing).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/DB/test_chachanotes_message_exchanges.py -v` then `pytest Tests/DB/ -q` (whole DB suite — migration-adjacent tests must stay green).
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py Tests/DB/
git commit -m "feat(db): v33 message_exchanges local-only table + upsert/read"
```

---

### Task 6: Persistence service + store attach/flush lifecycle

**Files:**
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py` (beside `update_message_usage`, line ~578)
- Modify: `tldw_chatbook/Chat/console_chat_store.py` — `ConsoleChatPersistence` protocol (~line 163), `ConsoleChatMessage` consumers, `set_message_usage` region (~line 2146), `_persist_usage_only` region (~line 3255), `_persist_existing_message` (~line 3344)
- Modify: `tldw_chatbook/Chat/console_chat_models.py` — `ConsoleChatMessage` (~line 402): add field
- Test: `Tests/Chat/test_console_chat_store_exchanges.py` (create)

**Interfaces:**
- Consumes: Task 1 `ExchangeCapture`/`capture_to_blob`; Task 5 `append_message_exchanges_local`.
- Produces (Task 7 relies on):
  - `ConsoleChatMessage.exchanges: tuple[ExchangeCapture, ...] = ()`
  - `ConsoleChatStore.attach_message_exchanges(message_id: str, captures: Sequence[ExchangeCapture]) -> None`
  - `ChatPersistenceService.append_message_exchanges(*, message_id: str, rows: Sequence[Mapping[str, Any]]) -> bool`
  - protocol method with the same signature on `ConsoleChatPersistence`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_chat_store_exchanges.py
"""Exchange attach lifecycle: dedup, stop-path flush, regen keep-marked,
ephemeral no-persist. Copy the store fixture + fake-persistence idioms from
Tests/Chat/test_console_chat_store.py (they model terminal marks and the
variant-restore path already)."""
from tldw_chatbook.Chat.console_exchange_capture import ExchangeCapture


def _cap(run_tag="r1", seq=0, status="complete"):
    return ExchangeCapture(
        run_tag=run_tag, seq=seq, created_at="t", provider="p", model="m",
        endpoint=None, request={"messages_payload": []},
        response={"content": "x"}, status=status, usage_json=None,
        omitted_keys=())


def test_attach_dedups_by_run_tag_and_seq(store_with_streaming_assistant):
    store, mid = store_with_streaming_assistant
    store.attach_message_exchanges(mid, [_cap(seq=0)])
    store.attach_message_exchanges(mid, [_cap(seq=0), _cap(seq=1)])
    message = store.get_message(mid)  # use the store's real snapshot accessor
    assert [c.seq for c in message.exchanges] == [0, 1]


def test_terminal_mark_flushes_exchanges(store_with_fake_persistence):
    store, mid, persistence = store_with_fake_persistence
    store.attach_message_exchanges(mid, [_cap()])
    # drive the message terminal via the store's real terminal-mark API
    ...
    assert persistence.appended_exchange_rows  # fake recorded the flush


def test_attach_after_terminal_flushes_immediately(store_with_fake_persistence):
    """Stop-path inversion: stop finalizes first, capture attaches late."""
    store, mid, persistence = store_with_fake_persistence
    # drive terminal FIRST, then attach
    ...
    store.attach_message_exchanges(mid, [_cap(status="stopped")])
    assert persistence.appended_exchange_rows


def test_variant_restored_message_keeps_captures_marked_abandoned(
        store_after_variant_restore):
    """CONTRAST with usage (which drops): spec owner decision 6."""
    store, mid = store_after_variant_restore  # mid in _variant_restored_message_ids
    store.attach_message_exchanges(mid, [_cap(run_tag="r2")])
    message = store.get_message(mid)
    assert any(c.run_tag == "r2" for c in message.exchanges)
    # the flush row carries abandoned=True
    ...


def test_ephemeral_session_never_persists(ephemeral_store):
    store, mid = ephemeral_store
    store.attach_message_exchanges(mid, [_cap()])
    # drive terminal; assert the persistence fake saw NO exchange append
    ...
```

The `...` bodies are filled by copying the exact fixture/driver calls the sibling store tests use for the same lifecycle transitions (terminal mark, variant restore, ephemeral) — those tests already exist for `set_message_usage`; mirror them mechanically, do not invent new drivers.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_chat_store_exchanges.py -v`
Expected: FAIL — `attach_message_exchanges` not defined.

- [ ] **Step 3: Implement**

`console_chat_models.py`, on `ConsoleChatMessage` (line ~402), after `generation_metadata`:

```python
    #: Captured provider exchanges for this turn (Conversation Inspector).
    #: Tuple for snapshot-safety; the store replaces, never mutates.
    exchanges: tuple["ExchangeCapture", ...] = ()
```

(Import `ExchangeCapture` under `TYPE_CHECKING` to avoid cycles if needed.)

`console_chat_store.py` — new method beside `set_message_usage`, following its docstring style:

```python
    def attach_message_exchanges(
        self, message_id: str, captures: Sequence["ExchangeCapture"]
    ) -> None:
        """Attach captured exchanges; flush now if the message is terminal.

        Mirrors set_message_usage's stop-path contract (terminal mark first,
        late attach flushes itself) with ONE deliberate divergence: a
        variant-restored message KEEPS incoming captures, marked abandoned
        (spec owner decision 6) — the traffic really happened; usage drops
        here because it would misprice the restored answer, but captures
        carry their own run_tag and cannot misattribute.
        """
        message = self._message_or_raise(message_id)
        abandoned = message.id in self._variant_restored_message_ids
        merged = {(c.run_tag, c.seq): c for c in message.exchanges}
        for capture in captures:
            merged.setdefault((capture.run_tag, capture.seq), capture)
        message.exchanges = tuple(
            sorted(merged.values(), key=lambda c: (c.run_tag, c.seq)))
        if abandoned:
            self._abandoned_exchange_run_tags.setdefault(message.id, set()).update(
                c.run_tag for c in captures)
        if message.status not in {"pending", "streaming"}:
            self._persist_exchanges_only(message)
```

Add `self._abandoned_exchange_run_tags: dict[str, set[str]] = {}` in `__init__` beside `_variant_restored_message_ids` (locate it with grep).

`_persist_exchanges_only`, modeled line-for-line on `_persist_usage_only` (line 3255): guard `self.persistence is None` / `message.persisted_message_id is None` / `not message.exchanges` (bail silently — unlike usage there is no content-carrying fallback; captures only ever ride the dedicated path), probe `getattr(self.persistence, "append_message_exchanges", None)` for callability, then:

```python
        abandoned_tags = self._abandoned_exchange_run_tags.get(message.id, set())
        rows = [
            {"run_tag": c.run_tag, "seq": c.seq,
             "status": c.status,
             "abandoned": c.run_tag in abandoned_tags,
             "capture_blob": capture_to_blob(c),
             "created_at": c.created_at}
            for c in message.exchanges
        ]
        try:
            writer(message_id=message.persisted_message_id, rows=rows)
        except Exception as exc:
            logger.bind(message_id=message.id, error=repr(exc)).warning(
                "exchange_flush_failed")
```

Hook the normal path: at the end of `_persist_existing_message` (line ~3344), after its successful durable write, add `self._persist_exchanges_only(message)` guarded by `if message.exchanges:` — the DB upsert makes repeat flushes harmless.

`ConsoleChatPersistence` protocol (~line 163 region): add

```python
    def append_message_exchanges(
        self, *, message_id: str, rows: Sequence[Mapping[str, Any]]
    ) -> bool: ...
```

`chat_persistence_service.py` beside `update_message_usage` (578):

```python
    def append_message_exchanges(
        self, *, message_id: str, rows: Sequence[Mapping[str, Any]]
    ) -> bool:
        """Local-only exchange-capture flush (Conversation Inspector).

        Same contract as update_message_usage: version-neutral, never
        enqueues sync rows, never raises past a log line.
        """
        try:
            self._db.append_message_exchanges_local(message_id, rows)
            return True
        except Exception as exc:
            logger.bind(message_id=message_id, error=repr(exc)).warning(
                "exchange_append_failed")
            return False
```

(Match the service's real DB-handle attribute name — read `update_message_usage`'s body and copy it.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_chat_store_exchanges.py -v` then `pytest Tests/Chat/test_console_chat_store.py -q`
Expected: all PASS; no store regressions.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/chat_persistence_service.py Tests/Chat/test_console_chat_store_exchanges.py
git commit -m "feat(console): store/persistence lifecycle for exchange captures"
```

---

### Task 7: Controller wiring + config kill-switch

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` — BOTH signals-creation sites (`grep -n "ConsoleProviderStreamSignals()" tldw_chatbook/Chat/console_chat_controller.py` — the dispatch site ~line 9627 AND the defensive belt ~line 10149 inside the stream method) and the usage-attach method (locate: `grep -n "usage_attach_failed"`, ~line 10054)
- Modify: `tldw_chatbook/config.py` — `[console]` defaults block (~line 2584): add `exchange_capture = true` with a comment
- Test: `Tests/Chat/test_console_chat_controller_exchanges.py` (create)

**Interfaces:**
- Consumes: Task 2 `exchange_captures()`/`exchange_capture_enabled`; Task 6 `attach_message_exchanges`; `get_cli_setting` (`config.py:5128`).
- Produces: every run (agent AND direct — both flow through the one dispatch site) attaches its captures to the assistant message at the same moments usage attaches; `[console] exchange_capture = false` disables capture end-to-end.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_chat_controller_exchanges.py
"""Controller attaches exchange captures alongside usage; config gates it.
Copy the controller-test fixture idioms from the existing usage-attach tests
(grep Tests/Chat for "usage_attach" / set_message_usage fakes)."""


def test_signals_created_with_capture_enabled_by_default(controller):
    signals = controller._new_run_stream_signals()  # the seam under test; see Step 3
    assert signals.exchange_capture_enabled is True


def test_kill_switch_disables_capture(controller, monkeypatch):
    # Patch get_cli_setting AT THE CONTROLLER'S NAMESPACE (a from-import
    # binds at import time — patch the consumer, prove it with a counter).
    calls = []
    def fake_get_cli_setting(section, key, default=None):
        calls.append((section, key))
        if (section, key) == ("console", "exchange_capture"):
            return False
        return default
    monkeypatch.setattr(
        "tldw_chatbook.Chat.console_chat_controller.get_cli_setting",
        fake_get_cli_setting)
    signals = controller._new_run_stream_signals()
    assert signals.exchange_capture_enabled is False
    assert ("console", "exchange_capture") in calls


def test_attach_site_forwards_captures_to_store(controller_with_fake_store):
    """The usage-attach method also attaches signals.exchange_captures()."""
    controller, store = controller_with_fake_store
    signals = ConsoleProviderStreamSignals()
    signals.begin_exchange(provider="p", model="m", endpoint=None,
                           request={}, omitted_keys=())
    signals.close_exchange()
    drive_usage_attach(controller, signals, assistant_message_id="a1")  # the existing test driver
    assert store.attached_exchanges  # fake store recorded attach_message_exchanges("a1", [...])


def test_attach_never_fails_the_send(controller_with_raising_store):
    """Store raising from attach_message_exchanges is swallowed+logged —
    same never-fail contract as usage_attach_failed."""
    ...
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Chat/test_console_chat_controller_exchanges.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add a small controller helper (near the other private helpers) and use it at BOTH `ConsoleProviderStreamSignals()` creation sites — the dispatch site (~9627) and the defensive belt (~10149, `if stream_signals is None:`):

```python
    def _new_run_stream_signals(self) -> ConsoleProviderStreamSignals:
        """One run's signals object, with exchange capture gated by config.

        get_cli_setting reads the RESOLVED settings layer — never raw TOML
        top-level, which nests under COMPREHENSIVE_CONFIG_RAW and silently
        never fires (cost-ticker PR2 Qodo F4 was exactly that bug).
        """
        return ConsoleProviderStreamSignals(
            exchange_capture_enabled=bool(
                get_cli_setting("console", "exchange_capture", True)
            )
        )
```

Both sites become `stream_signals = self._new_run_stream_signals()`. If the controller does not already import `get_cli_setting`, add it to the config imports.

In the usage-attach method (the one logging `usage_attach_failed`), after the usage attach block and INSIDE the same never-fail posture:

```python
        try:
            captures = list(stream_signals.exchange_captures())
            if captures:
                self.store.attach_message_exchanges(assistant_message_id, captures)
        except Exception as exc:
            logger.bind(
                message_id=assistant_message_id, error=repr(exc)
            ).warning("exchange_attach_failed")
```

Note: this method already runs on BOTH completion and stop/cancel paths (that is why usage lives here) — captures inherit both for free. `exchange_captures()`'s in-flight tail covers the aborted-stream case exactly as `usage_payloads()` does. `_new_run_stream_signals` (Step 3 above) is the test seam.

`config.py` `[console]` block (~2584): add

```toml
# Conversation Inspector: capture each provider exchange (request/response)
# locally per turn. Local-only; never synced. Set false to disable.
exchange_capture = true
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Chat/test_console_chat_controller_exchanges.py -v` plus the controller's existing usage tests: `pytest Tests/Chat -k "usage" -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/config.py Tests/Chat/test_console_chat_controller_exchanges.py
git commit -m "feat(console): wire exchange capture through the controller + kill-switch"
```

---

### Task 8: Conversation Inspector modal — scaffold + Costs tab + entry rewiring

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — `_console_cost_chip_activated` (~line 9784), `_open_console_cost_breakdown` (~line 9792), `action_view_chat_context` (~line 3325), imports (beside the existing `console_context_modal`/`console_cost_modal` imports)
- Test: `Tests/UI/test_console_conversation_inspector.py` (create)

**Interfaces:**
- Consumes: `build_cost_rows`/`build_cost_rows_totals`/`ConsoleCostRow` (`console_cost_tracker.py`); `ExchangeCapture`, `capture_from_blob` (Task 1); an injected `exchanges_loader: Callable[[str], Awaitable[list[ExchangeCapture]]]` the screen builds over store + `get_message_exchanges`.
- Produces (Tasks 9-10 build on): `ConsoleConversationInspector(ModalScreen[None])` with `TabbedContent` panes `#inspector-costs`, `#inspector-exchange`, `#inspector-next-send`; constructor:

```python
    def __init__(self, *, rows, totals, turns, exchanges_loader,
                 snapshot_factory, token_estimate=None, estimate_factory=None,
                 in_progress=False, ephemeral=False, initial_tab="inspector-costs"):
```

where `turns` is `list[InspectorTurn]` — `@dataclass(frozen=True) class InspectorTurn: message_id: str; native_message_id: str; index: int; role: str; preview: str` (define it in the new module).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_console_conversation_inspector.py
"""Inspector scaffold: three tabs, costs rows render, per-turn drill-in
lazy-loads captures. Copy the modal-test harness idiom from
Tests/UI/test_console_cost_modal.py (run_test + pilot)."""
import pytest
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRow, ConsoleCostRowTotals
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector, InspectorTurn,
)


def _row(index=0):
    return ConsoleCostRow(index=index, role="assistant", model="m",
                          uncached_input=10, cache_read=0, cache_write=0,
                          output=5, cost_usd=0.001, estimated=False)


def _totals():
    return ConsoleCostRowTotals(total_tokens=15, total_cost_usd=0.001,
                                has_estimated_entries=False, row_count=1)


async def _noop_snapshot():
    from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
    return ConsoleContextSnapshot(current_messages=[], next_send_payload={})


@pytest.mark.asyncio
async def test_three_tabs_render(...):
    modal = ConsoleConversationInspector(
        rows=[_row()], totals=_totals(),
        turns=[InspectorTurn(message_id="p1", native_message_id="n1",
                             index=0, role="assistant", preview="hi")],
        exchanges_loader=lambda mid: _async_list([]),
        snapshot_factory=_noop_snapshot)
    # mount via the harness; assert the three TabPane ids exist


@pytest.mark.asyncio
async def test_costs_rows_render_and_totals(...):
    # assert the row text contains "in:10" and the totals line "15 tokens"
    # (reuse ConsoleCostModal._format_row's exact format — Step 3 moves it here)


@pytest.mark.asyncio
async def test_loader_called_lazily_only_on_expand(...):
    # loader is a spy; assert zero calls after mount, one call after
    # expanding turn "p1"'s row


@pytest.mark.asyncio
async def test_no_capture_recorded_row(...):
    # loader returns []; expanded row shows "No capture recorded for this turn"
```

Also add to `Tests/UI/test_console_cost_chip_screen.py` (screen-level): clicking the cost chip pushes `ConsoleConversationInspector` (not `ConsoleCostModal`) — copy that file's existing chip-press driver.

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/UI/test_console_conversation_inspector.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement scaffold + Costs tab**

New module `console_conversation_inspector.py`: `ModalScreen[None]` with `DEFAULT_CSS` (width 110, max-width 95%, height 42, max-height 90% — wider than the old modals; keep the `align: center middle` + `border: tall gray` idiom), `BINDINGS = [("escape", "dismiss", "Close"), ("r", "refresh", "Refresh")]`, and:

- `compose()`: header Static, `TabbedContent(id="console-inspector-tabs")` with three `TabPane`s (`Costs`/`Exchange`/`Next Send`, ids above), actions row with Close.
- Costs pane: move `ConsoleCostModal._format_row` and `_format_totals` here VERBATIM (they are pure staticmethods; Task 10 deletes the old file). Render each row as a `Collapsible(title=self._format_row(row), collapsed=True)` whose body is a lazy container.
- Lazy drill-in: on `Collapsible.Toggled`, if expanding and not yet loaded, `self.run_worker` the `exchanges_loader(turn.message_id)` call, then mount per-call Statics: `f"call {c.seq} [{c.status}] {c.model} — {self._call_cost_line(c)}"` where `_call_cost_line` prices `ProviderUsage.from_json(c.usage_json)` through the same catalog call `build_cost_rows` uses (read `build_cost_rows`'s body at `console_cost_tracker.py:688` and reuse its pricing helper; show "unpriced" when `usage_json` is None). Empty list mounts `Static("No capture recorded for this turn (recorded before capture existed, capture disabled, or capture failed).")`.
- Map cost rows to turns by `row.index` into `turns` (both are transcript-ordered; `InspectorTurn.index` is the same index `build_cost_rows` reports).

Screen rewiring in `chat_screen.py`:

- `_open_console_cost_breakdown` (~7913): keep the rows/totals computation, then build `turns` from the SAME `messages` list it already fetches (`InspectorTurn(message_id=msg.persisted_message_id or "", native_message_id=msg.id, index=i, role=..., preview=first 60 chars of content)`), build `exchanges_loader`:

```python
        async def _exchanges_loader(persisted_id: str) -> list[ExchangeCapture]:
            if not persisted_id:
                return []
            def _read() -> list[ExchangeCapture]:
                db = self._console_chachanotes_db()  # the screen's existing DB accessor — find how ConsoleChatStore's persistence reaches CharactersRAGDB and reuse that handle
                out = []
                for row in db.get_message_exchanges(persisted_id):
                    try:
                        out.append(capture_from_blob(row["capture_blob"]))
                    except Exception:
                        logger.opt(exception=True).warning("exchange_blob_decode_failed")
                return out
            return await asyncio.to_thread(_read)
```

  For an EPHEMERAL session there is no DB row — the loader must first check the in-memory store (`message.exchanges` on the native message) and only fall back to the DB read; native captures win when present (they are fresher).
- `_console_cost_chip_activated` pushes `ConsoleConversationInspector(..., initial_tab="inspector-costs")`.
- `action_view_chat_context` (~2240) keeps its existing `_factory`/`_estimate_factory` bodies but pushes the SAME inspector with `initial_tab="inspector-next-send"` (the Next Send pane is a placeholder Static until Task 10 — the command-palette entry at `UI/console_command_provider.py:97` follows automatically since it calls this action).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/UI/test_console_conversation_inspector.py Tests/UI/test_console_cost_chip_screen.py -v`
Expected: PASS. (`Tests/UI/test_console_cost_modal.py` still passes — the old modal is untouched until Task 10.)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_conversation_inspector.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/
git commit -m "feat(console): Conversation Inspector scaffold + Costs tab + entry rewiring"
```

---

### Task 9: Exchange tab

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`
- Test: `Tests/UI/test_console_conversation_inspector.py`

**Interfaces:**
- Consumes: Task 8's `turns`/`exchanges_loader`; `estimate_tokens` (`tldw_chatbook/Utils/token_counter.py:137`) — THE estimator seam, same function+call-shape the composer/budget path uses (parity requirement; grep the composer's estimate call and match its arguments).
- Produces: the `#inspector-exchange` pane — per-turn, per-call detail.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_exchange_call_sections_render(...):
    cap = ExchangeCapture(
        run_tag="r1", seq=0, created_at="t", provider="anthropic",
        model="m", endpoint=None,
        request={"system_message": "SYS PROMPT", "messages_payload": [
                     {"role": "user", "content": "hello"}],
                 "tools": [{"function": {"name": "get_time"}}],
                 "temp": 0.7},
        response={"content": "world", "tool_calls": []},
        status="complete", usage_json=None, omitted_keys=("api_key",))
    # mount inspector with a loader returning [cap]; open Exchange tab,
    # expand the turn, expand the call. Assert visible sections contain:
    #  "System prompt" + "SYS PROMPT", "Tools" + "get_time",
    #  "Response" + "world", "Sampling" + "temp",
    #  "Omitted by capture policy: api_key"


@pytest.mark.asyncio
async def test_estimates_labeled_and_reported_authoritative(...):
    # capture WITH usage_json → per-piece lines carry "~" + "est." labels
    # while the call header shows the reported buckets unprefixed


@pytest.mark.asyncio
async def test_status_badges(...):
    # statuses "stopped"/"error" and abandoned=True (loader returns the
    # abandoned flag — extend the loader to yield (capture, abandoned)
    # pairs; see Step 3) render "[stopped]", "[error]",
    # "[abandoned regeneration]" in the call title


@pytest.mark.asyncio
async def test_collapsible_bodies_mount_lazily(...):
    # after expanding the call, section TextAreas for a section that is
    # still collapsed do not exist in the DOM yet
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/UI/test_console_conversation_inspector.py -k exchange -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Change the loader contract (Task 8's screen-side builder updates too): it returns `list[tuple[ExchangeCapture, bool]]` — (capture, abandoned) — reading `abandoned` from the DB row; in-memory store rows compute it from the store's abandoned-tag set exposed on the message snapshot (add a tiny accessor if absent — check what Task 6 stored).

Exchange pane structure, all lazily mounted:

```
Collapsible "[3] assistant — 2 calls"            (one per turn, collapsed)
  Collapsible "call 0 [complete] m — $0.0012"    (one per call, collapsed)
    Static  "Omitted by capture policy: api_key"
    Collapsible "System prompt (~12 tokens est.)" → TextArea(read_only)
    Collapsible "Messages (2)" → per-message Collapsible → TextArea(json)
    Collapsible "Tools (1)" → TextArea(json)
    Collapsible "Response (~5 tokens est. / reported out:5)" → TextArea
    Collapsible "Tool calls (0)" — omitted when empty
    Collapsible "Sampling & routing" → TextArea(json of the scalar kwargs)
```

Per-piece estimates: `f"~{estimate_tokens(text, '', '')} tokens est."` — the `('', '')` call shape matches `chat_screen._estimate_tokens` (line ~2300); reported buckets come from `ProviderUsage.from_json(capture.usage_json)` and render WITHOUT the `~`/`est.` marker. Lazy mounting: each `Collapsible` starts with no body child; an `@on(Collapsible.Toggled)` handler mounts the body on first expand (track mounted ids in a set). JSON rendering uses `json.dumps(obj, indent=2, default=str)` (the context modal's `_json_block` idiom). All Statics `markup=False` (Button.label markup-eats bracketed text — known trap; badges like `[stopped]` are exactly that shape).

Per-call Copy/Save buttons: reuse the pyperclip copy and `~/Downloads` save idioms from `ConsoleContextModal._copy_json`/`_save_json` verbatim, writing `json.dumps(asdict(capture), ...)`; the Save button honors the same `blocked_reason("save-context", ephemeral=...)` gate (constructor already receives `ephemeral`).

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/UI/test_console_conversation_inspector.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_conversation_inspector.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_conversation_inspector.py
git commit -m "feat(console): Exchange tab — per-call payload review"
```

---

### Task 10: Next Send tab + retire the old modals

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_conversation_inspector.py`
- Delete: `tldw_chatbook/Widgets/Console/console_cost_modal.py`, `tldw_chatbook/Widgets/Console/console_context_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (drop old imports, lines ~437-438)
- Modify: `Tests/UI/test_console_context_modal.py`, `Tests/UI/test_chat_screen_context_modal.py`, `Tests/UI/test_console_cost_modal.py` — migrate to the inspector
- Test: migrated files above

**Interfaces:**
- Consumes: Task 8 constructor params `snapshot_factory`/`token_estimate`/`estimate_factory`/`in_progress`/`ephemeral` (already plumbed).
- Produces: feature-complete inspector; zero references to the old modal classes anywhere (`grep -rn "ConsoleCostModal\|ConsoleContextModal" tldw_chatbook Tests` returns nothing).

- [ ] **Step 1: Port the Next Send pane**

Move `ConsoleContextModal`'s behavior into the pane wholesale — this is a mechanical port, keep the code and its comments (LY-13 compaction note, save/copy caveats) intact: the snapshot worker (`_load_snapshot` + `on_worker_state_changed` error notify), `watch_` reactives, Raw JSON checkbox, Refresh (and the `r` binding routing to it when this tab is active), Copy JSON, Save to File with `blocked_reason` gating, in-progress warning line, 1 MiB size threshold, empty-state compaction (`context-empty` class — scope the CSS to the pane, not the modal frame). Collapsible builders (`_build_current_context_widgets`/`_build_next_send_widgets`) port as-is under the pane's two sub-tabs Current/Next Send (keep the inner `TabbedContent` exactly as the old modal had it).

- [ ] **Step 2: Migrate the tests**

For each of the three test files: keep every behavioral assertion (they are pins — empty-state copy prefix "No conversation context", save-blocked tooltip, raw-JSON toggle, size threshold, `_format_row` strings), re-target construction/queries to `ConsoleConversationInspector` with the right `initial_tab`. Delete only assertions about the old modal FRAME (ids `console-cost-modal`, `console-context-modal`) — replace with the pane ids. Rename `test_console_cost_modal.py` → merge its cases into `test_console_conversation_inspector.py`; keep the other two filenames (they test screen-level wiring).

- [ ] **Step 3: Delete the old modules and imports; run the sweep**

Run: `grep -rn "ConsoleCostModal\|ConsoleContextModal" tldw_chatbook Tests` — must return zero hits. Then delete the two widget files.

- [ ] **Step 4: Run the full affected suites**

Run: `pytest Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_context_modal.py Tests/UI/test_console_context_modal.py Tests/UI/test_console_cost_chip_screen.py -v` then `pytest Tests/UI -q -k "console"` and `pytest --collect-only -q` (import-error sweep for the deleted modules).
Expected: all PASS; collect-only completes with no errors.

- [ ] **Step 5: Commit**

```bash
git add -A tldw_chatbook/Widgets/Console tldw_chatbook/UI/Screens/chat_screen.py Tests/UI
git commit -m "feat(console): Next Send tab; retire ConsoleCostModal + ConsoleContextModal"
```

---

### Task 11: Docs, backlog close-out, live verification

**Files:**
- Modify: the `Docs/User_Guide/` page covering the Console screen's cost chip / context viewer (find it: `grep -rln "cost chip\|Chat Context\|ctrl+shift+p" Docs/User_Guide/ -i`) — describe the inspector's three tabs, the capture kill-switch, the adapter-boundary caveat and the llama.cpp exception, and refresh the page's "Verified against" stamp
- Modify: the backlog task file (Implementation Notes + AC ticks) per repo Definition of Done

**Steps:**

- [ ] **Step 1: Update the User Guide page** (content per the spec's UI section; state the two honesty caveats verbatim from the spec's "Risks" list).

- [ ] **Step 2: Targeted full gate.** Run every test file this plan created or touched in one invocation, confirm a READ nonzero passed-count (a "no tests ran" result is a FAILED gate):

```bash
pytest Tests/Chat/test_console_exchange_capture.py Tests/Chat/test_console_provider_gateway.py Tests/DB/test_chachanotes_message_exchanges.py Tests/Chat/test_console_chat_store_exchanges.py Tests/Chat/test_console_chat_controller_exchanges.py Tests/UI/test_console_conversation_inspector.py Tests/UI/test_chat_screen_context_modal.py Tests/UI/test_console_context_modal.py Tests/UI/test_console_cost_chip_screen.py Tests/Chat/test_console_chat_store.py Tests/DB/ -q
```

- [ ] **Step 3: Live verification** (spec Testing section — use the `verify` skill to drive the TUI; API keys for agent use are in the repo root). One real-provider session: (a) an agent turn that invokes a tool → inspector shows ≥2 calls with tool schemas in call 0's request and tool results in call 1's messages; (b) Stop mid-stream → the turn's tail call shows `[stopped]` with partial content; (c) regenerate then Stop it → original answer's captures intact, abandoned run visible marked; (d) confirm the captured system prompt matches the session's configured one byte-for-byte; (e) flip `exchange_capture = false`, send, confirm the new turn shows "No capture recorded". Write the evidence (commands, painted-frame observations) into the backlog task's Implementation Notes. Cost expectation: under $0.10.

- [ ] **Step 4: Commit docs + notes; prepare the PR** per `superpowers:finishing-a-development-branch` (branch off dev, CI is intentionally cancelled — the local gate above is the merge signal).

```bash
git add Docs/User_Guide backlog/tasks
git commit -m "docs(console): Conversation Inspector user guide + task close-out"
```

---

## Plan Self-Review (performed at write time)

- **Spec coverage:** capture module/allowlist/stubs → T1; signals per-call API → T2; generic + llama.cpp seams → T3/T4; schema+local-only persistence → T5; store lifecycle incl. stop/regen/ephemeral → T6; controller + kill-switch → T7; UI Costs/Exchange/Next Send + retirement → T8/T9/T10; docs + live verify → T11. Spec's "no capture" degradation rows → T8/T9; per-call pricing → T8; estimator parity → T9; oversize truncation → T1.
- **Known gap, deliberate:** the spec's "reason when known" on no-capture rows is delivered as a static multi-reason string (T8) — per-turn reason attribution would need capture-absence bookkeeping that YAGNI fails.
- **Type consistency check:** `ExchangeCapture.usage_json: str | None` used consistently (T1 def, T2 writer via `_flight_capture`/`close_exchange`, T8/T9 readers via `ProviderUsage.from_json`); loader returns `list[tuple[ExchangeCapture, bool]]` after T9's amendment — T8 builds it, T9 documents the change, T10 inherits.
- **Re-anchor pass (2026-08-18, origin/dev @ `1bdbcac61`):** Tasks 1-5, 7, 8 updated for the dev refactor (scoped call signals, PreparedProviderRequest kwargs, v41, two controller signals sites, moved chat_screen anchors). Line numbers are indicative; names are the contract.
