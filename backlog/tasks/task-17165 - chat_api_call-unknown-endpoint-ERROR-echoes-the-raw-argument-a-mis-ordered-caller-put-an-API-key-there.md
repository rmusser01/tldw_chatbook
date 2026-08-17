---
id: taREDACTED-17165
title: >-
  chat_api_call unknown-endpoint ERROR echoes the raw argument — a mis-ordered
  caller put an API key there
status: Done
assignee: []
created_date: '2026-08-16'
labels: [security, llm-calls]
dependencies: []
priority: high
---

## Description (the why)

Found during TASK-3502's fix wave (2026-08-16, reproduction in that
arc's TASK-17065 filing). `BaseReranker._call_llm_impl` passes its
arguments to `chat_api_call` POSITIONALLY in the wrong order
(`RAG_Search/reranker.py:228-237`), so the API key arrives as
`api_endpoint` — and `chat_api_call`'s unknown-endpoint failure path
echoes that raw argument into an ERROR log line
("Routing to endpoint: <the key>"). Reproduced with an isolated config: a
deepseek-keyed profile with reranking enabled logs the key at ERROR
level on every search.

The mis-ordered CALLER is TASK-17065's to fix (the dispatch repair that
deliberately converts a no-op into spend). THIS task is the sink-side
defence, per the loguru-diagnose lesson (fix at the SINK): an
error path that interpolates an unrecognised caller-supplied string into
a log line must not echo credential-shaped values, because some caller
someday will hand it one again.

## Acceptance Criteria (the what)

- [x] chat_api_call's unknown/unsupported-endpoint failure path redacts
      credential-shaped values (at minimum: strings matching common key
      prefixes or containing no spaces and exceeding a length bound are
      elided to a short prefix + "…redacted") before logging or raising
      into any user-visible error string
- [x] A test feeds a key-shaped string as the endpoint and asserts the
      raw value appears in NO log record and NO exception text
- [x] The redaction is at the sink (chat_api_call's own failure path),
      not in the reranker caller — TASK-17065 owns the caller

## Implementation Notes

**The leak was wider than filed: FIVE sites, and two fire on EVERY call.**
The task described the unsupported-endpoint failure path; reading the sink
found `logger.info("Routing to endpoint: …")` and
`log_counter(labels={"api_endpoint": …})` execute BEFORE the handler
lookup — so a mis-ordered caller's credential was logged at INFO on every
single call and stamped into a metrics label (also an unbounded-cardinality
hazard, and metrics are exported). The other three: the ERROR log, the
`ValueError`, and `ChatConfigurationError`'s provider field + message.

**The defence is an ALLOWLIST, not a key-shape blocklist.** A registered
endpoint is safe by definition and prints verbatim, so genuine diagnostics
stay readable; anything else is unknown-provenance text and is elided to
`<unrecognised endpoint, N chars, redacted>`. A blocklist would miss any
credential that does not look like one, and this sink cannot know what it
was handed. The `ValueError` now lists the valid endpoints instead of
echoing the bad one — strictly more useful for a real typo.

**A vacuous test of my own, caught before it shipped.** The first draft of
the metrics-label test PASSED while the label still carried the whole
value: the sink lowercases the endpoint, so a case-sensitive substring
check could not see it. All leak assertions are now case-insensitive
through one `_leaks()` helper, and the label test went red before the fix.

**One existing contract deliberately changed**: `test_unsupported_endpoint_
raises_error` asserted the endpoint was echoed back. It now asserts the
redaction marker plus the valid-endpoint list, with the reason in the test.

Caller repair (the mis-ordered positional call) remains TASK-17065's.
Files: `tldw_chatbook/Chat/Chat_Functions.py` (`safe_endpoint_for_display`
+ five call sites), `Tests/Chat/test_chat_api_call_endpoint_redaction.py`
(new, 4 tests), `Tests/Chat/test_chat_functions.py` (contract update).
