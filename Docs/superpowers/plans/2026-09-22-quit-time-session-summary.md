# Quit-time Session Usage Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a confirmed quit, optionally show a brief session summary (total tokens + elapsed time) that auto-dismisses, fed by a new process-wide session usage ledger.

**Architecture:** A thread-safe `SessionUsageLedger` in `Chat/session_usage.py` accumulates exact provider usage (falling back to char estimates) recorded once per response at each provider-parse boundary in `LLM_API_Calls.py`, the Console gateway, agent service, Library RAG, and realtime. A `SessionSummaryDialog` modal reads a snapshot at quit time; the quit worker shows it after approved-quit cleanup, hard-capped so `App.exit()` always proceeds. Config: `[session_summary]` section (default off), settings-screen instant-apply group.

**Tech Stack:** Python 3.12, Textual 8.x, `threading.Lock`, existing `ProviderUsage`/`estimate_tokens`, pytest + Textual pilot, `unittest.mock.patch("requests.Session.post")` mocked-HTTP idiom.

**Spec:** `Docs/superpowers/specs/2026-09-22-quit-time-session-summary-design.md` (read it first — this plan argues from it)

## Global Constraints

- Work on a branch off `main`, not any in-flight chore branch.
- Targeted test runs only; never a full-suite sweep unless the user asks (AGENTS.md).
- Existing usage histograms in `LLM_API_Calls.py` stay **byte-identical** — only additive record lines.
- Every ledger/record path never raises (accounting must not break provider calls).
- Record each provider response **exactly once** — downstream re-reads (Console cost tracker, transcript attachment) are not taps.
- No `$ds-*` tokens in Python-side `DEFAULT_CSS` (they only resolve in the bundled tcss); use legacy semantic variables (`$panel`, `$secondary`, `$accent`, `$text-muted`). No hex literals.
- Do not bind `ctrl+q`/terminal-convention keys on the new screen (ADR-031).
- `time.perf_counter()` for elapsed time — `app._startup_start_time` is a `perf_counter` stamp; never mix with `time.monotonic()`.
- Textual timers: never `set_timer(0.0)`; keep a strong reference and `stop()` on every close path; `[DONE]`-sentinel/`finally`-yield GeneratorExit rules apply.
- ADR check: none required (additive feature, ephemeral data — reasoning recorded in the spec's ADR Check section).
- Backlog: task A "Session usage ledger and provider tap points" = Tasks 1–3; task B "Quit-time session usage summary dialog and quit integration" = Tasks 4–7.

---

### Task 1: SessionUsageLedger

**Files:**
- Create: `tldw_chatbook/Chat/session_usage.py`
- Test: `Tests/Chat/test_session_usage.py`

**Interfaces:**
- Consumes: `ProviderUsage` (`tldw_chatbook/Chat/provider_usage.py`, dataclass with `.total_tokens`), `estimate_tokens` (`tldw_chatbook/Chat/usage_recorder.py`, ~4 chars/token, empty→1).
- Produces: `session_usage() -> SessionUsageLedger`; `SessionUsageLedger.record_exact(usage: ProviderUsage | None)`, `.record_estimate(prompt_text: str | None, completion_text: str | None)`, `.record_provider_payload(usage_payload, *, provider: str = "", model: str = "", fallback_texts: tuple[str | None, str | None] = (None, None))`, `.snapshot() -> SessionUsageSnapshot`; `SessionUsageSnapshot(exact_tokens: int, estimated_tokens: int, calls: int, total_tokens: int property)`; `reset_for_tests()`.

- [ ] **Step 1: Write the failing tests** — create `Tests/Chat/test_session_usage.py`:

```python
"""SessionUsageLedger unit tests (issue #365).

The ledger is the quit-time summary's single data source; these tests pin
its accumulation rules: exact wins over estimate, malformed payloads never
raise, and recording is thread-safe.
"""

import threading

import pytest

from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Chat.session_usage import (
    SessionUsageSnapshot,
    reset_for_tests,
    session_usage,
)
from tldw_chatbook.Chat.usage_recorder import estimate_tokens


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_for_tests()
    yield
    reset_for_tests()


def _openai_usage(prompt: int, completion: int) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def test_exact_accumulates():
    ledger = session_usage()
    ledger.record_provider_payload(_openai_usage(10, 5), provider="openai", model="m")
    ledger.record_provider_payload(_openai_usage(1, 2), provider="openai", model="m")
    snap = ledger.snapshot()
    assert snap.exact_tokens == 18
    assert snap.estimated_tokens == 0
    assert snap.calls == 2
    assert snap.total_tokens == 18


def test_estimate_fallback_when_payload_lacks_usage():
    ledger = session_usage()
    ledger.record_provider_payload(
        None, fallback_texts=("hello world, quite long", "ok")
    )
    snap = ledger.snapshot()
    assert snap.exact_tokens == 0
    assert snap.estimated_tokens == (
        estimate_tokens("hello world, quite long") + estimate_tokens("ok")
    )
    assert snap.calls == 1


def test_exact_wins_no_estimate_double_count():
    ledger = session_usage()
    ledger.record_provider_payload(
        _openai_usage(10, 5),
        provider="openai",
        model="m",
        fallback_texts=("a very long prompt text", "a very long reply"),
    )
    snap = ledger.snapshot()
    assert snap.exact_tokens == 15
    assert snap.estimated_tokens == 0
    assert snap.calls == 1


def test_malformed_payloads_never_raise():
    ledger = session_usage()
    ledger.record_provider_payload(object())
    ledger.record_provider_payload({"prompt_tokens": "not-a-number"})
    ledger.record_provider_payload("usage")
    assert ledger.snapshot() == SessionUsageSnapshot(0, 0, 0)


def test_none_payload_without_texts_is_noop():
    session_usage().record_provider_payload(None)
    assert session_usage().snapshot().calls == 0


def test_record_exact_accepts_none():
    session_usage().record_exact(None)
    assert session_usage().snapshot().calls == 0


def test_record_exact_sums_provider_usage_totals():
    usage = ProviderUsage.from_provider_payload(
        {"input_tokens": 7, "output_tokens": 3}, provider="anthropic", model="m"
    )
    assert usage is not None
    session_usage().record_exact(usage)
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 10
    assert snap.calls == 1


def test_concurrent_records_are_thread_safe():
    ledger = session_usage()

    def worker() -> None:
        for _ in range(1000):
            ledger.record_provider_payload(
                _openai_usage(1, 1), provider="openai", model="m"
            )

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = ledger.snapshot()
    assert snap.exact_tokens == 8 * 1000 * 2
    assert snap.calls == 8 * 1000


def test_reset_for_tests_gives_fresh_singleton():
    old = session_usage()
    old.record_provider_payload(_openai_usage(1, 1), provider="openai", model="m")
    reset_for_tests()
    assert session_usage() is not old
    assert session_usage().snapshot() == SessionUsageSnapshot(0, 0, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_session_usage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.session_usage'`

- [ ] **Step 3: Implement the ledger** — create `tldw_chatbook/Chat/session_usage.py`:

```python
"""Process-wide session usage accumulation (issue #365).

Complements ``provider_usage`` (shape normalization) and ``usage_recorder``
(scoped research estimates): this module accumulates token usage for the
whole app session so the quit-time summary
(Docs/superpowers/specs/2026-09-22-quit-time-session-summary-design.md)
can display a total without recomputing anything during shutdown.

Rules (spec, "Tap Points and the Boundary Rule"):
- record exactly once, where a provider response's usage is parsed;
- never raise — accounting must not be able to break a provider call;
- exact provider usage always wins over a char-based estimate.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Optional

from .provider_usage import ProviderUsage
from .usage_recorder import estimate_tokens

__all__ = [
    "SessionUsageSnapshot",
    "SessionUsageLedger",
    "reset_for_tests",
    "session_usage",
]


@dataclass(frozen=True)
class SessionUsageSnapshot:
    """Immutable point-in-time view of the session accumulator."""

    exact_tokens: int = 0
    estimated_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.exact_tokens + self.estimated_tokens


class SessionUsageLedger:
    """Thread-safe, O(1)-memory session accumulator. Never raises."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._exact = 0
        self._estimated = 0
        self._calls = 0

    def record_exact(self, usage: Optional[ProviderUsage]) -> None:
        """Add a provider-reported (exact) usage record."""
        try:
            if usage is None:
                return
            total = usage.total_tokens
            if total <= 0:
                return
            with self._lock:
                self._exact += total
                self._calls += 1
        except Exception:  # noqa: BLE001 - accounting must never break a call
            pass

    def record_estimate(
        self, prompt_text: Optional[str], completion_text: Optional[str]
    ) -> None:
        """Add a char-based (~4 chars/token) estimate record."""
        try:
            total = estimate_tokens(prompt_text or "") + estimate_tokens(
                completion_text or ""
            )
            with self._lock:
                self._estimated += total
                self._calls += 1
        except Exception:  # noqa: BLE001
            pass

    def record_provider_payload(
        self,
        usage_payload: Any,
        *,
        provider: str = "",
        model: str = "",
        fallback_texts: tuple[Optional[str], Optional[str]] = (None, None),
    ) -> None:
        """Record a raw provider usage payload; estimate when it carries none.

        ``ProviderUsage.from_provider_payload`` never raises and returns
        ``None`` for unrecognized/malformed shapes (fabricating no zeros).
        """
        try:
            usage = ProviderUsage.from_provider_payload(
                usage_payload, provider=provider or "unknown", model=model
            )
            if usage is not None and usage.total_tokens > 0:
                self.record_exact(usage)
                return
            prompt_text, completion_text = fallback_texts
            if prompt_text is None and completion_text is None:
                return
            self.record_estimate(prompt_text, completion_text)
        except Exception:  # noqa: BLE001
            pass

    def snapshot(self) -> SessionUsageSnapshot:
        with self._lock:
            return SessionUsageSnapshot(
                exact_tokens=self._exact,
                estimated_tokens=self._estimated,
                calls=self._calls,
            )


_LEDGER = SessionUsageLedger()


def session_usage() -> SessionUsageLedger:
    """The process-wide ledger singleton."""
    return _LEDGER


def reset_for_tests() -> None:
    """Swap in a fresh singleton; call from test fixtures only."""
    global _LEDGER
    _LEDGER = SessionUsageLedger()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Chat/test_session_usage.py -v`
Expected: PASS (all 9)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/session_usage.py Tests/Chat/test_session_usage.py
git commit -m "feat(chat): add SessionUsageLedger for session token totals (issue #365)"
```

---

### Task 2: Non-streaming provider taps (the nine sites)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (nine sites + one module helper + one import)
- Test: `Tests/Chat/test_session_usage_taps.py` (create)

**Interfaces:**
- Consumes: `session_usage().record_provider_payload(...)` from Task 1.
- Produces: `_completion_text_from_response(response_data) -> str` (module-private, never raises) used again by Task 3.

Per-site facts (verified): the request parameter is `input_data: List[Dict[str, Any]]` at **all** nine sites; usage/model variable names and completion-text expressions:

| Function (def line) | Insert after | Usage var | Model var | provider= | Completion text for `fallback_texts` |
| --- | --- | --- | --- | --- | --- |
| `chat_with_openai` (:559) | the `if usage:` histogram block, before `logger.debug("OpenAI: Non-streaming request successful.")` | `usage` | `final_model` | `"openai"` | `_completion_text_from_response(response_data)` |
| `chat_with_anthropic` (:1274) | its `if usage:` block | `usage` | `current_model` | `"anthropic"` | `full_assistant_content` |
| `chat_with_cohere` (:2422) | its `if usage_data:` block | `usage_data` | `final_model` | `"cohere"` | `text` |
| `chat_with_deepseek` (:3133) | its `if usage:` block | `usage` | `current_model` | `"deepseek"` | `_completion_text_from_response(result)` |
| `chat_with_google` (:3465) | the `normalized_response["usage"] = {...}` assignment (:4013–4017) | (see Step 3) | `current_model` | `"google"` | `assistant_content` |
| `chat_with_groq` (:4194) | its `if usage:` block | `usage` | `current_model` | `"groq"` | `_completion_text_from_response(result)` |
| `chat_with_huggingface` (:4453) | its `if usage:` block | `usage` | `final_model_for_payload` | `"huggingface"` | `_completion_text_from_response(result)` |
| `chat_with_mistral` (:4974) | its `if usage:` block | `usage` | `current_model` | `"mistral"` | `_completion_text_from_response(result)` |
| `chat_with_openrouter` (:5215) | its `if usage:` block | `usage` | `current_model` | `"openrouter"` | `_completion_text_from_response(result)` |

- [ ] **Step 1: Write the failing tests** — create `Tests/Chat/test_session_usage_taps.py`, mirroring the imports and config sandboxing preamble of `Tests/Chat/test_openai_streaming_usage.py` (which drives the real dispatcher `chat_api_call` under `@patch("requests.Session.post")`):

```python
"""Session-usage tap tests: provider functions record into the ledger.

Mirrors the mocking pattern from Tests/Chat/test_chat_mocked_apis.py /
test_openai_streaming_usage.py: patch ``requests.Session.post``, drive the
real dispatcher via ``chat_api_call``, and inspect the session ledger.
"""

import json
from unittest.mock import Mock, patch

import pytest

from tldw_chatbook.Chat.session_usage import reset_for_tests, session_usage


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_for_tests()
    yield
    reset_for_tests()


def _mock_post(payload_dict):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = payload_dict
    return response


def test_openai_nonstreaming_records_exact_usage():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "hello there"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "openai",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="gpt-4o",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 15
    assert snap.estimated_tokens == 0
    assert snap.calls == 1


def test_openai_nonstreaming_without_usage_records_estimate():
    body = {
        "choices": [{"message": {"role": "assistant", "content": "hello there"}}],
        # no "usage" key at all
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "openai",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="gpt-4o",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 0
    assert snap.estimated_tokens > 0
    assert snap.calls == 1


def test_anthropic_nonstreaming_records_exact_usage():
    body = {
        "content": [{"type": "text", "text": "hi back"}],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    with patch("requests.Session.post", return_value=_mock_post(body)):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="claude-3-5-sonnet",
            streaming=False,
        )
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 10
    assert snap.calls == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_session_usage_taps.py -v`
Expected: FAIL — `snap.exact_tokens == 0` (nothing records yet)

- [ ] **Step 3: Implement the taps**

3a. Add the import near the other `tldw_chatbook.Chat.*` imports (~:42–51):

```python
from tldw_chatbook.Chat.session_usage import session_usage
```

3b. Add the defensive completion-text extractor at module level (below the other private helpers, e.g. after `_contains_extended_ttl`):

```python
def _completion_text_from_response(response_data: Any) -> str:
    """Best-effort completion text for token estimates. Never raises."""
    try:
        choices = response_data.get("choices")
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
        content_blocks = response_data.get("content")
        if isinstance(content_blocks, list):
            return "".join(
                part.get("text", "")
                for part in content_blocks
                if isinstance(part, dict)
            )
    except Exception:
        pass
    return ""
```

3c. Worked example — `chat_with_openai`, insert between the `if usage:` histogram block and `logger.debug("OpenAI: Non-streaming request successful.")`:

```python
            session_usage().record_provider_payload(
                usage,
                provider="openai",
                model=final_model,
                fallback_texts=(
                    json.dumps(input_data),
                    _completion_text_from_response(response_data),
                ),
            )
```

3d. Apply the identical pattern at the other eight sites per the table above, adapting only the variable names listed. Google is the one structural change (name the normalized dict so it can be recorded; histograms keep reading `usage_meta` untouched):

```python
            usage_meta = response_data.get("usageMetadata")
            if usage_meta and all(
                k in usage_meta
                for k in ["promptTokenCount", "candidatesTokenCount", "totalTokenCount"]
            ):
                normalized_usage = {
                    "prompt_tokens": usage_meta.get("promptTokenCount"),
                    "completion_tokens": usage_meta.get("candidatesTokenCount"),
                    "total_tokens": usage_meta.get("totalTokenCount"),
                }
                normalized_response["usage"] = normalized_usage
                session_usage().record_provider_payload(
                    normalized_usage,
                    provider="google",
                    model=current_model,
                    fallback_texts=(json.dumps(input_data), assistant_content),
                )
```

- [ ] **Step 4: Run tap tests + existing usage-histogram tests**

Run: `python -m pytest Tests/Chat/test_session_usage_taps.py Tests/Chat/test_cache_usage_metrics.py Tests/Chat/test_chat_mocked_apis.py -v`
Expected: PASS (new tests pass; existing histogram tests unchanged — the blocks were untouched)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_session_usage_taps.py
git commit -m "feat(llm): record session usage at the nine non-streaming provider sites"
```

---

### Task 3: Streaming taps + service taps

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (anthropic accumulator ≈:1993, responses conversion :331, openai pass-through loop ≈:877)
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` (`_maybe_record_usage` ≈:6930)
- Modify: `tldw_chatbook/Agents/agent_service.py` (≈:1594)
- Modify: `tldw_chatbook/Library/library_rag_answer_service.py` (≈:654)
- Modify: `tldw_chatbook/LLM_Calls/realtime/openai_session.py` (at its `from_provider_payload` call)
- Test: `Tests/Chat/test_session_usage_taps.py` (extend)

**Interfaces:**
- Consumes: `session_usage().record_provider_payload` / `.record_exact` from Task 1; `_completion_text_from_response` not needed here (exact only).
- Produces: none consumed later.

- [ ] **Step 1: Write the failing tests** — append to `Tests/Chat/test_session_usage_taps.py`:

```python
def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}".encode("utf-8")


ANTHROPIC_STREAM_LINES = [
    _sse({"type": "message_start", "message": {"usage": {"input_tokens": 12}}}),
    _sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "he"}}),
    _sse({"type": "message_delta", "delta": {"usage": {"output_tokens": 4}}}),
    _sse({"type": "message_stop"}),
]


def test_anthropic_streaming_records_exact_once():
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(ANTHROPIC_STREAM_LINES)
    response.close = Mock()
    with patch("requests.Session.post", return_value=response):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        generator = chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="claude-3-5-sonnet",
            streaming=True,
        )
        list(generator)
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 16  # 12 input + 4 output
    assert snap.calls == 1  # exactly once per response


OPENAI_STREAM_LINES = [
    'data: {"id": "1", "choices": [{"index": 0, "delta": {"content": "he"}, "finish_reason": null}]}',
    'data: {"id": "1", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}',
    "data: [DONE]",
]


def test_openai_chat_completions_stream_records_exact_once():
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.iter_lines.return_value = iter(OPENAI_STREAM_LINES)
    response.close = Mock()
    with patch("requests.Session.post", return_value=response):
        from tldw_chatbook.Chat.Chat_Functions import chat_api_call

        generator = chat_api_call(
            "openai",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            model="gpt-4o",
            streaming=True,
        )
        list(generator)
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 10
    assert snap.calls == 1


def test_gateway_sse_usage_records_exactly_once():
    from tldw_chatbook.Chat.console_provider_gateway import _content_from_sse_data

    class _Signals:
        def __init__(self):
            self.payloads = []

        def record_usage_payload(self, usage):
            self.payloads.append(usage)

    signals = _Signals()
    line = (
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }
        )
        + "\n\n"
    )
    _content_from_sse_data(line, signals=signals)
    snap = session_usage().snapshot()
    assert snap.exact_tokens == 6
    assert snap.calls == 1  # ledger and console signal each saw it once
    assert len(signals.payloads) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_session_usage_taps.py -v`
Expected: the three new tests FAIL (`exact_tokens == 0`), Task 2's still pass.

- [ ] **Step 3: Implement the streaming taps**

3a. Anthropic accumulator — at the `if output_captured:` seam (≈:1993), record **before** the yield (never in `finally` — GeneratorExit hazard documented at the `[DONE]` sentinel):

```python
                    if output_captured:
                        if usage_accumulator:
                            session_usage().record_provider_payload(
                                usage_accumulator,
                                provider="anthropic",
                                model=current_model,
                            )
                        yield _usage_sse_chunk()
```

3b. OpenAI Responses conversion — inside `_responses_stream_to_chat_sse` where `completed_usage` is extracted (≈:331):

```python
                completed_usage = (event.get("response") or {}).get("usage")
                if isinstance(completed_usage, dict):
                    chunk["usage"] = completed_usage
                    session_usage().record_provider_payload(
                        completed_usage, provider="openai", model=model
                    )
```

3c. OpenAI chat-completions pass-through loop — add the substring guard (lines arrive as `str` because `iter_lines(decode_unicode=True)`):

```python
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.strip():
                            if '"usage"' in line:
                                _record_openai_stream_usage_line(
                                    line, model=final_model
                                )
                            # Pass through OpenAI's SSE lines directly.
                            # Ensure they end with \n\n if not already.
                            # OpenAI's SSE usually includes double newlines.
                            yield line if line.endswith("\n") else line + "\n"
```

with the module-level helper (next to `_completion_text_from_response`):

```python
def _record_openai_stream_usage_line(line: str, *, model: str) -> None:
    """Record exact usage from an OpenAI SSE line, best-effort.

    ``stream_options.include_usage`` is requested for chat-completions
    streams, so the final chunk carries usage; the substring guard avoids
    parsing every delta chunk.
    """
    try:
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            return
        payload = json.loads(data)
        usage = payload.get("usage")
        if isinstance(usage, dict) and usage:
            session_usage().record_provider_payload(
                usage, provider="openai", model=model
            )
    except Exception:
        pass
```

3d. Service taps — one `record_exact`/`record_provider_payload` line at each boundary:

- `console_provider_gateway.py` `_maybe_record_usage` (≈:6930), with `from .session_usage import session_usage` added to the module imports:

```python
def _maybe_record_usage(
    payload: Mapping[str, Any],
    signals: "_ProviderStreamSignals | None",
) -> None:
    if signals is None:
        return
    usage = payload.get("usage")
    if isinstance(usage, Mapping) and usage:
        signals.record_usage_payload(usage)
        session_usage().record_provider_payload(usage)
```

- `Agents/agent_service.py` (≈:1594, inside the existing never-break-the-run `try`), adding `from tldw_chatbook.Chat.session_usage import session_usage` to imports:

```python
    usage = ProviderUsage.from_provider_payload(
        resp.get("usage") if isinstance(resp, dict) else None,
        provider=provider,
        model=model,
    )
    if usage is not None:
        session_usage().record_exact(usage)
```

- `Library/library_rag_answer_service.py` (≈:654), same import, immediately after the existing `usage = ProviderUsage.from_provider_payload(...)`:

```python
    if usage is not None:
        session_usage().record_exact(usage)
```

- `LLM_Calls/realtime/openai_session.py`: locate its `from_provider_payload(...)` call (grep `from_provider_payload` in the file), and add the same two lines after it, with the import.

3e. **Audit step (spec requirement):** run `grep -rn "from_provider_payload" tldw_chatbook/ --include="*.py" | grep -v session_usage | grep -v provider_usage.py` and classify each site as *boundary parse* (needs a record line as above) or *downstream re-read* (must NOT record — e.g. `console_chat_controller._attach_stream_usage` reading gateway signals, the Console cost tracker reading transcript rows, and `LLM_API_Calls.py`'s own normalization helpers whose callers were already tapped). Sites already covered by Task 2/3 code above: the gateway's `:1428`/`:4972` normalizations are upstream of `_maybe_record_usage` only if they feed SSE lines through it — verify by reading each; if any parses a response that never passes through `_maybe_record_usage` (e.g. a non-SSE HTTP body), add `session_usage().record_exact(usage)` there and extend the exactly-once test to cover it.

- [ ] **Step 4: Run tap tests + streaming regression tests**

Run: `python -m pytest Tests/Chat/test_session_usage_taps.py Tests/Chat/test_openai_streaming_usage.py Tests/Chat/test_anthropic_streaming_usage.py -v`
Expected: PASS — including the existing streaming usage tests (request payloads untouched; `stream_options` was already sent)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Library/library_rag_answer_service.py tldw_chatbook/LLM_Calls/realtime/openai_session.py Tests/Chat/test_session_usage_taps.py
git commit -m "feat(llm): record streaming and service session usage at parse boundaries"
```

---

### Task 4: Config section + duration loader

**Files:**
- Modify: `tldw_chatbook/config.py` (TOML template, between `[splash_screen.effects]` and `[logging]`, ≈:3899)
- Modify: `tldw_chatbook/app.py` (one method on `TldwCli`)
- Test: `Tests/test_config_session_summary_defaults.py` (create)

**Interfaces:**
- Consumes: `CONFIG_TOML_CONTENT`, `get_cli_setting` (both `config.py`; already imported in `app.py`).
- Produces: `TldwCli._session_summary_duration_seconds(self) -> int` (clamped 1–30, default 3) consumed by Task 6; TOML keys `session_summary.enabled` (bool, default `false`) and `session_summary.duration_seconds` (int, default `3`).

- [ ] **Step 1: Write the failing tests** — create `Tests/test_config_session_summary_defaults.py`:

```python
"""Config defaults for the quit-time session summary (issue #365)."""

import tomllib

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.config import CONFIG_TOML_CONTENT


def test_session_summary_defaults_exist():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    section = parsed["session_summary"]
    assert section["enabled"] is False
    assert section["duration_seconds"] == 3


def test_duration_loader_clamps_and_defaults(monkeypatch):
    from tldw_chatbook import app as app_module

    class _Harness:
        _session_summary_duration_seconds = (
            app_module.TldwCli._session_summary_duration_seconds
        )

    harness = _Harness()

    def _make_setting(duration):
        def setting(section, key, default=None):
            if (section, key) == ("session_summary", "duration_seconds"):
                return duration
            return default

        return setting

    for duration, expected in [(0, 1), (1, 1), (3, 3), (30, 30), (99, 30), (-5, 1)]:
        monkeypatch.setattr(app_module, "get_cli_setting", _make_setting(duration))
        assert harness._session_summary_duration_seconds() == expected

    for bad in ["abc", None, ""]:
        monkeypatch.setattr(app_module, "get_cli_setting", _make_setting(bad))
        assert harness._session_summary_duration_seconds() == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/test_config_session_summary_defaults.py -v`
Expected: FAIL — `KeyError: 'session_summary'` and `AttributeError: ... _session_summary_duration_seconds`

- [ ] **Step 3: Implement**

3a. In `CONFIG_TOML_CONTENT` (`config.py`), insert between the `[splash_screen.effects]` block (ends with the `custom_image_path = ""` line, ≈:3900) and `[logging]`:

```toml
[session_summary]
# Optional quit-time session usage summary (issue #365).
# After a confirmed quit (Ctrl+Q), briefly show total session tokens and
# elapsed session time before the app exits. Any key skips it.
enabled = false  # Show the summary on quit (default off)
duration_seconds = 3  # Auto-dismiss delay in seconds (clamped to 1..30)
```

3b. In `app.py`, add this method to `TldwCli` (place it near `_run_approved_quit_cleanup`, ≈:19214, so it reads with its only consumer):

```python
    def _session_summary_duration_seconds(self) -> int:
        """Configured quit-summary duration, clamped to 1..30 (default 3)."""
        raw = get_cli_setting("session_summary", "duration_seconds", 3)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 3
        return max(1, min(30, value))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/test_config_session_summary_defaults.py Tests/test_config_model_catalog_defaults.py -v`
Expected: PASS (new tests; model_catalog defaults unaffected by the template addition)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/config.py tldw_chatbook/app.py Tests/test_config_session_summary_defaults.py
git commit -m "feat(config): add [session_summary] defaults and duration clamp"
```

---

### Task 5: SessionSummaryDialog

**Files:**
- Create: `tldw_chatbook/Widgets/session_summary_dialog.py`
- Test: `Tests/UI/test_session_summary_dialog.py` (create)

**Interfaces:**
- Consumes: `SessionUsageSnapshot` from Task 1.
- Produces: `SessionSummaryDialog(snapshot: SessionUsageSnapshot, *, started_at: float, duration_seconds: float)` — `ModalScreen[None]`; `started_at` is a `time.perf_counter()` stamp (pass `app._startup_start_time`); `duration_seconds` is pre-clamped by the caller (constructor tolerates small test values, floored at 0.05).

- [ ] **Step 1: Write the failing tests** — create `Tests/UI/test_session_summary_dialog.py`:

```python
"""SessionSummaryDialog pilot tests (issue #365).

Render evidence: export_screenshot SVG text assertions (lessons-testing-
evidence). Timer evidence: the close callback must actually run.
"""

import asyncio
import time

from tldw_chatbook.Chat.session_usage import SessionUsageSnapshot
from tldw_chatbook.Widgets.session_summary_dialog import SessionSummaryDialog

from textual.app import App, ComposeResult
from textual.widgets import Static


class _DialogApp(App[None]):
    def __init__(self, dialog: SessionSummaryDialog) -> None:
        super().__init__()
        self._dialog = dialog

    def compose(self) -> ComposeResult:
        yield Static("behind")

    def on_mount(self) -> None:
        self.push_screen(self._dialog)


def _dialog(duration: float, snapshot: SessionUsageSnapshot | None = None) -> SessionSummaryDialog:
    return SessionSummaryDialog(
        snapshot or SessionUsageSnapshot(exact_tokens=42318, estimated_tokens=0, calls=3),
        started_at=time.perf_counter() - 4320.0,  # 1h 12m ago
        duration_seconds=duration,
    )


async def test_dialog_renders_summary_copy():
    app = _DialogApp(_dialog(duration=30))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        svg = app.export_screenshot()
        assert "Session summary" in svg
        assert "42,318 tokens" in svg
        assert "session" in svg
        assert "press any key to exit" in svg


async def test_dialog_renders_no_usage_and_estimate_marker():
    empty = _DialogApp(_dialog(duration=30, snapshot=SessionUsageSnapshot()))
    async with empty.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        svg = empty.export_screenshot()
        assert "No usage recorded this session" in svg

    mixed = _DialogApp(
        _dialog(
            duration=30,
            snapshot=SessionUsageSnapshot(exact_tokens=100, estimated_tokens=40, calls=2),
        )
    )
    async with mixed.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        svg = mixed.export_screenshot()
        assert "140 tokens" in svg
        assert "includes estimates" in svg


async def test_any_key_skips_dialog():
    dialog = _dialog(duration=30)
    app = _DialogApp(dialog)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is not dialog


async def test_dialog_auto_dismisses_after_duration():
    dialog = _dialog(duration=0.05)
    app = _DialogApp(dialog)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        dismissed = False
        for _ in range(60):  # poll, never a fixed pause (lessons)
            if app.screen is not dialog:
                dismissed = True
                break
            await asyncio.sleep(0.05)
        assert dismissed, "auto-close timer never fired"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/UI/test_session_summary_dialog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Widgets.session_summary_dialog'`

- [ ] **Step 3: Implement the dialog** — create `tldw_chatbook/Widgets/session_summary_dialog.py`:

```python
"""Quit-time session summary dialog (issue #365).

Shown once by the quit worker after approved-quit cleanup completes and
before App.exit(). Reads only the snapshot passed in; never recomputes.
Spec: Docs/superpowers/specs/2026-09-22-quit-time-session-summary-design.md
"""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Container
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Static

from ..Chat.session_usage import SessionUsageSnapshot

__all__ = ["SessionSummaryDialog"]


def _format_elapsed(seconds: float) -> str:
    total_minutes = int(seconds // 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m session"
    return f"{minutes}m session"


class SessionSummaryDialog(ModalScreen[None]):
    """Brief session usage summary; auto-closes, any key skips.

    Follows the SplashScreen timed-close idiom (strong timer reference,
    idempotent close that stops the timer) and the ConfirmationDialog
    small-modal structure. No BINDINGS: any key skips (ADR-031 — no
    terminal-convention keys bound; ctrl+q stays app-global).
    """

    DEFAULT_CSS = """
    SessionSummaryDialog {
        align: center middle;
    }
    #session-summary-dialog {
        background: $panel;
        border: round $secondary;
        padding: 1 2;
        width: auto;
        height: auto;
    }
    .session-summary-title {
        text-style: bold;
        color: $accent;
    }
    .session-summary-line {
        color: $text;
    }
    .session-summary-hint {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        snapshot: SessionUsageSnapshot,
        *,
        started_at: float,
        duration_seconds: float,
    ) -> None:
        super().__init__()
        self.snapshot = snapshot
        self._started_at = started_at  # time.perf_counter() stamp
        self._duration = max(0.05, float(duration_seconds))
        self._closed = False
        self._auto_close_timer = None

    def compose(self) -> ComposeResult:
        with Container(id="session-summary-dialog"):
            yield Static("Session summary", classes="session-summary-title")
            if self.snapshot.calls:
                line = f"{self.snapshot.total_tokens:,} tokens"
                if self.snapshot.estimated_tokens > 0:
                    line += " · includes estimates"
                yield Static(line, classes="session-summary-line")
            else:
                yield Static(
                    "No usage recorded this session",
                    classes="session-summary-line",
                )
            elapsed = max(0.0, time.perf_counter() - self._started_at)
            yield Static(_format_elapsed(elapsed), classes="session-summary-line")
            yield Static("press any key to exit", classes="session-summary-hint")

    def on_mount(self) -> None:
        self._auto_close_timer = self.set_timer(self._duration, self._close)

    def on_key(self, event: Key) -> None:
        """Any key skips: the key's whole job here is to dismiss (splash
        precedent — consume it)."""
        event.stop()
        event.prevent_default()
        self._close()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._auto_close_timer is not None:
            self._auto_close_timer.stop()
        self.dismiss(None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/UI/test_session_summary_dialog.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/session_summary_dialog.py Tests/UI/test_session_summary_dialog.py
git commit -m "feat(ui): add SessionSummaryDialog with auto-close and any-key skip"
```

---

### Task 6: Quit-flow integration with hard cap

**Files:**
- Modify: `tldw_chatbook/app.py` (`_run_approved_quit_cleanup` ≈:19214–19232 + new method + imports)
- Test: `Tests/UI/test_session_summary_quit.py` (create)

**Interfaces:**
- Consumes: `SessionSummaryDialog` (Task 5), `session_usage()` (Task 1), `TldwCli._session_summary_duration_seconds` (Task 4), `get_cli_setting`.
- Produces: `TldwCli._show_session_summary_before_exit(self) -> None` (async).

- [ ] **Step 1: Write the failing tests** — create `Tests/UI/test_session_summary_quit.py`, using the bind-real-methods harness idiom of `Tests/UI/test_app_quit_guard.py`:

```python
"""Quit-flow session summary integration tests (issue #365).

Harness idiom from Tests/UI/test_app_quit_guard.py: bind the REAL unbound
methods, stub only the collaborators they touch. CancelledError semantics
on the quit path must be preserved (lessons-textual).
"""

import asyncio
import time

import pytest

from tldw_chatbook import app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.session_usage import reset_for_tests, session_usage
from tldw_chatbook.Chat.usage_recorder import estimate_tokens


class _CleanupHarness:
    _run_approved_quit_cleanup = TldwCli._run_approved_quit_cleanup
    _show_session_summary_before_exit = TldwCli._show_session_summary_before_exit
    _session_summary_duration_seconds = TldwCli._session_summary_duration_seconds

    def __init__(self, *, never_dismiss: bool = False) -> None:
        self._startup_start_time = time.perf_counter()
        self.never_dismiss = never_dismiss
        self.exited = False
        self.pushed = []

    async def _cleanup_audio_for_quit(self) -> None:
        return None

    def _run_blocking_quit_persistence(self) -> None:
        return None

    async def push_screen_wait(self, screen) -> None:
        self.pushed.append(screen)
        if self.never_dismiss:
            await asyncio.sleep(60.0)

    def exit(self) -> None:
        self.exited = True


def _settings(enabled: bool, duration: int = 1):
    def setting(section, key, default=None):
        if section == "session_summary":
            if key == "enabled":
                return enabled
            if key == "duration_seconds":
                return duration
        return default

    return setting


async def test_disabled_summary_never_pushes(monkeypatch):
    monkeypatch.setattr(app_module, "get_cli_setting", _settings(enabled=False))
    harness = _CleanupHarness()
    await harness._run_approved_quit_cleanup()
    assert harness.pushed == []
    assert harness.exited is True


async def test_enabled_summary_pushes_then_exits(monkeypatch):
    monkeypatch.setattr(app_module, "get_cli_setting", _settings(enabled=True))
    reset_for_tests()
    session_usage().record_estimate("some prompt", "some reply")
    harness = _CleanupHarness()
    await harness._run_approved_quit_cleanup()
    assert len(harness.pushed) == 1
    assert harness.exited is True


async def test_stuck_dialog_hard_cap_still_exits(monkeypatch):
    monkeypatch.setattr(
        app_module, "get_cli_setting", _settings(enabled=True, duration=1)
    )
    harness = _CleanupHarness(never_dismiss=True)
    started = time.perf_counter()
    await harness._run_approved_quit_cleanup()
    elapsed = time.perf_counter() - started
    assert harness.exited is True
    assert elapsed < 10.0  # capped at duration + 2s, not the 60s stub sleep
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/UI/test_session_summary_quit.py -v`
Expected: FAIL — `AttributeError: ... no attribute '_show_session_summary_before_exit'`

- [ ] **Step 3: Implement the integration**

3a. Imports in `app.py` (match the file's existing import style; absolute form shown):

```python
from tldw_chatbook.Chat.session_usage import session_usage
from tldw_chatbook.Widgets.session_summary_dialog import SessionSummaryDialog
```

3b. New method next to `_run_approved_quit_cleanup`:

```python
    async def _show_session_summary_before_exit(self) -> None:
        """Show the optional quit-time usage summary, hard-capped so exit
        always proceeds (issue #365; spec "Quit-Flow Integration")."""
        duration = self._session_summary_duration_seconds()
        dialog = SessionSummaryDialog(
            session_usage().snapshot(),
            started_at=self._startup_start_time,
            duration_seconds=duration,
        )
        try:
            await asyncio.wait_for(
                self.push_screen_wait(dialog), timeout=duration + 2.0
            )
        except asyncio.TimeoutError:
            loguru_logger.warning(
                "Session summary dialog did not dismiss in time; exiting anyway."
            )
        except Exception:
            # Deliberately narrow scope beyond TimeoutError for robustness,
            # but CancelledError is BaseException on 3.12 — it propagates
            # and must never be swallowed on the quit path (lessons-textual).
            loguru_logger.warning("Session summary display failed; exiting anyway.")
```

3c. In `_run_approved_quit_cleanup`, add the gated call in the `try:` body, after the persistence step (keep `self.exit()` in the `finally:` — a persistence failure degrades to exit-without-summary):

```python
            try:
                await asyncio.to_thread(self._run_blocking_quit_persistence)
            except Exception:
                loguru_logger.warning("Blocking quit persistence failed")
            if get_cli_setting("session_summary", "enabled", False):
                await self._show_session_summary_before_exit()
        finally:
            self.exit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/UI/test_session_summary_quit.py Tests/UI/test_app_quit_guard.py -v`
Expected: PASS — new tests (the hard-cap test runs ~3s: duration 1 + cap 2) and existing quit-guard tests unaffected

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/app.py Tests/UI/test_session_summary_quit.py
git commit -m "feat(app): show session summary between quit cleanup and exit, hard-capped"
```

---

### Task 7: Settings screen group + user documentation

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (group + handlers + persist + pure helper)
- Modify: `Docs/User_Guide/settings/` (relevant category page; fall back to `Docs/User_Guide/settings.md`)
- Test: `Tests/UI/test_settings_session_summary.py` (create)

**Interfaces:**
- Consumes: `get_cli_setting`, `save_settings_to_cli_config` (both already imported/used in `settings_screen.py` for `[model_catalog]` — verify, add import if missing).
- Produces: `_session_summary_section_values(enabled: bool, duration_text: str) -> dict` (module-level pure function) consumed by the persist path and tests.

- [ ] **Step 1: Write the failing test** — create `Tests/UI/test_settings_session_summary.py`:

```python
"""Settings session-summary pure-helper tests (issue #365)."""

from tldw_chatbook.UI.Screens.settings_screen import _session_summary_section_values


def test_valid_values_pass_through():
    assert _session_summary_section_values(True, "3") == {
        "session_summary": {"enabled": True, "duration_seconds": 3}
    }


def test_duration_clamped():
    assert _session_summary_section_values(True, "0")["session_summary"]["duration_seconds"] == 1
    assert _session_summary_section_values(True, "99")["session_summary"]["duration_seconds"] == 30


def test_invalid_duration_falls_back_to_default():
    result = _session_summary_section_values(False, "abc")["session_summary"]
    assert result["duration_seconds"] == 3
    assert result["enabled"] is False


def test_empty_duration_falls_back():
    assert _session_summary_section_values(True, "")["session_summary"]["duration_seconds"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest Tests/UI/test_settings_session_summary.py -v`
Expected: FAIL — `ImportError: cannot import name '_session_summary_section_values'`

- [ ] **Step 3: Implement**

3a. Module-level pure helper in `settings_screen.py` (near the model_catalog constants, ≈:992–1031):

```python
def _session_summary_section_values(
    enabled: bool, duration_text: str
) -> dict[str, dict[str, object]]:
    """Normalize the [session_summary] settings group for persistence."""
    try:
        duration = int(float(str(duration_text).strip()))
    except (TypeError, ValueError):
        duration = 3
    duration = max(1, min(30, duration))
    return {
        "session_summary": {"enabled": bool(enabled), "duration_seconds": duration}
    }
```

3b. Compose the group immediately before `yield from self._render_custom_endpoints_section()` (≈:16064), after the model_catalog per-provider rows:

```python
            with Vertical(
                id="settings-session-summary-group",
                classes="settings-instant-apply-group",
            ):
                yield Static("Session summary on quit", classes="destination-section")
                yield Static(
                    INSTANT_APPLY_BEHAVIOR_COPY,
                    id="settings-session-summary-instant-hint",
                    classes="settings-instant-apply-hint",
                )
                yield Checkbox(
                    "Show session usage summary when quitting",
                    value=bool(
                        get_cli_setting("session_summary", "enabled", False)
                    ),
                    id="settings-session-summary-enabled",
                )
                with Horizontal(classes="settings-input-row"):
                    yield Static(
                        "Summary duration (seconds):",
                        classes="settings-status-row",
                    )
                    yield Input(
                        str(get_cli_setting("session_summary", "duration_seconds", 3)),
                        id="settings-session-summary-duration",
                        type="integer",
                        tooltip="How long the quit summary shows before auto-exit (1-30).",
                    )
```

3c. Handlers (near the model_catalog handlers, ≈:27077):

```python
    @on(Checkbox.Changed, "#settings-session-summary-enabled")
    def handle_session_summary_toggle_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        self._persist_session_summary_settings()

    @on(Input.Changed, "#settings-session-summary-duration")
    def handle_session_summary_duration_changed(self, event: Input.Changed) -> None:
        event.stop()
        self._persist_session_summary_settings()
```

3d. Persist path (near `_persist_model_catalog_section_values`, ≈:14017) with a no-op guard against per-keystroke disk writes:

```python
    def _persist_session_summary_settings(self) -> None:
        checkbox = self.query_one("#settings-session-summary-enabled", Checkbox)
        duration_input = self.query_one("#settings-session-summary-duration", Input)
        section_values = _session_summary_section_values(
            checkbox.value, duration_input.value
        )
        current = _session_summary_section_values(
            bool(get_cli_setting("session_summary", "enabled", False)),
            str(get_cli_setting("session_summary", "duration_seconds", 3)),
        )
        if section_values == current:
            return
        self._persist_session_summary_section_values(section_values)

    @work(thread=True)
    def _persist_session_summary_section_values(
        self, section_values: dict[str, dict[str, object]]
    ) -> None:
        try:
            save_settings_to_cli_config(section_values)
        except Exception:
            logger.warning("Failed to persist session_summary settings.")
```

3e. User Guide — add this section to the settings category page covering the group (the providers/models page that contains "Automatic refresh"; if categories are split across `Docs/User_Guide/settings/` files, use the one covering that page, else `Docs/User_Guide/settings.md`):

```markdown
### Session summary on quit

An optional farewell screen (default **off**). When enabled, confirming a
quit (Ctrl+Q) first shows a short summary — total session tokens and
elapsed session time — for a few seconds before the app exits. Any key
skips it immediately; the screen also auto-dismisses after the configured
duration and never delays shutdown cleanup.

- Toggle: **Settings → "Session summary on quit" → "Show session usage
  summary when quitting"** (instant apply).
- **Summary duration (seconds)**: 1–30, default 3.
- Token totals are exact where the provider reports usage; char-based
  estimates fold in (marked "includes estimates") when it doesn't.
  Embeddings are not counted.
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest Tests/UI/test_settings_session_summary.py Tests/UI/test_settings_network_defaults.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_session_summary.py Docs/User_Guide/
git commit -m "feat(settings): expose session summary toggle and duration with docs"
```

---

## Final verification (after Task 7)

- [ ] Run the full new-test set: `python -m pytest Tests/Chat/test_session_usage.py Tests/Chat/test_session_usage_taps.py Tests/test_config_session_summary_defaults.py Tests/UI/test_session_summary_dialog.py Tests/UI/test_session_summary_quit.py Tests/UI/test_settings_session_summary.py -v`
- [ ] Run neighboring regression suites: `Tests/Chat/test_openai_streaming_usage.py Tests/Chat/test_anthropic_streaming_usage.py Tests/Chat/test_cache_usage_metrics.py Tests/Chat/test_chat_mocked_apis.py Tests/UI/test_app_quit_guard.py`
- [ ] Live verification per `backlog/docs/lessons-live-verification.md`: run the app with the feature enabled (`[session_summary] enabled = true`), exchange one Console message, quit with Ctrl+Q, and confirm the summary shows the exchanged tokens and reasonable elapsed time in a tmux capture — evidence for the backlog task notes.
- [ ] Update both backlog tasks' Implementation Notes and mark ACs checked (`backlog task edit <id> -s Done`).
