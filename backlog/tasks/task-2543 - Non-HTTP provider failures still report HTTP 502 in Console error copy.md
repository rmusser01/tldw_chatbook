---
id: TASK-2543
title: Non-HTTP provider failures still report "HTTP 502" in Console error copy
status: Done
assignee: []
created_date: '2026-08-06 09:48'
labels:
  - chat
  - honesty
dependencies: []
priority: medium
---

## Description

PR-T3 Task 7 fixed a real 400 rendering as a contradictory `"provider returned HTTP
502 (… Status: 400.)"` — the real status now carries through instead of
`ChatProviderError`'s 502 upstream-error default. That fix's scope was "stop
reporting two contradictory numbers", not "redesign the no-status fallback", and the
fallback itself was deliberately left alone.

A separate, pre-existing problem in the same fallback survives, byte-identical to
base: when a provider call fails with something that has no HTTP status at all — a
timeout, a connection reset, a bare adapter exception — the Console's error copy
still reads `provider returned HTTP 502`. The stream consumer's re-raise
(`console_provider_gateway.py`, the `while True: item = await queue.get()` loop) and
`describe_stream_failure` (`Chat/provider_failures.py`) both classify by inspecting
the RE-RAISED `ChatProviderError` wrapper, never the original exception — and that
wrapper's `status_code` defaults to `502` whenever the original had no `status_code`
attribute at all (true for `TimeoutError`/`ConnectionError`/a bare adapter
exception). `describe_stream_failure` does have dedicated timeout/connection-refused
branches, but they only fire when passed the *original* exception's type — by the
time it runs here, that type information is already gone, replaced by
`ChatProviderError`. A user debugging a network problem is told the provider
returned a gateway error it never returned.

For a status-less exception, `safe_provider_error_copy` (the prose baked into
`item.text`) omits the status clause from its text entirely — so today's output
carries exactly ONE number, and it is simply the wrong one, not a contradictory pair
the way Task 7's bug was.

**Scope note:** this is pre-existing and unrelated to PR-T3 Task 7's actual change —
Task 7 deliberately did not touch this fallback. The same fallback default is shared
with the untouched `complete_auxiliary` call site (`console_provider_gateway.py`,
its own `except Exception as exc: status_code = getattr(exc, "status_code", 502)`
block), so any fix must cover both, not just the streaming path.

## Acceptance Criteria

- [x] A provider stream failure with no real HTTP status (timeout, connection reset, a
      bare adapter exception with no `status_code` attribute) surfaces user-facing
      copy naming the actual failure mode (e.g. "request timed out…",
      "could not connect to the provider…") and does not assert `HTTP 502` or any
      other status the provider never returned.
- [x] The fix covers both call sites that share the `502` fallback default: the stream
      consumer's re-raise in `console_provider_gateway.py` and `complete_auxiliary`'s
      parallel `except Exception` default.
- [x] `Tests/Chat/test_console_chat_controller.py`'s existing pin for a genuine 502
      HTTP response keeps passing unchanged (that case is a real gateway error, not
      this fallback).
- [x] Additive regression test: a raw `TimeoutError` (or `ConnectionError`) with no
      `status_code` attribute, propagating through the stream, produces copy that
      never contains "HTTP 502" or any fabricated status.

## Closeout audit 2026-10-11

Verdict: DONE (verified against origin/dev @ a3c47c6c72).

Evidence:
- `tldw_chatbook/Chat/provider_failures.py:117-147` now names the actual failure mode for status-less errors: "request timed out waiting for the provider", "connection refused - is the provider server running?", "could not connect to the provider", task-32342's "the app could not build the request or read the reply", and plain "unexpected provider error" as the terminal fallback. `provider returned HTTP {status_code}` fires only when a real status exists.
- The gateway's stream-consumer/auxiliary re-raise maps exception classes before wrapping (`console_provider_gateway.py:5569-5582`: TimeoutError→408, ConnectionError/OSError→503, HTTPStatusError→real status) and re-raises status-less ChatConfigurationError untouched.
- Regression pins: `Tests/Chat/test_console_chat_controller.py:2528-2531` (TimeoutError → "timed out", ConnectionError → "could not connect") alongside the unchanged genuine-502 pin (:2536-2538). Landed via `7dbbc401bb` (TASK-2154 honest error semantics) and refined by `f679524b06` (TASK-34100.5).
- All four ACs ticked.
