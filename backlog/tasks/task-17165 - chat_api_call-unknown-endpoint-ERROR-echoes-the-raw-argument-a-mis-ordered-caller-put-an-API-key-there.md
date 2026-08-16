---
id: task-17165
title: >-
  chat_api_call unknown-endpoint ERROR echoes the raw argument — a mis-ordered
  caller put an API key there
status: To Do
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
("Routing to endpoint: sk-…"). Reproduced with an isolated config: a
deepseek-keyed profile with reranking enabled logs the key at ERROR
level on every search.

The mis-ordered CALLER is TASK-17065's to fix (the dispatch repair that
deliberately converts a no-op into spend). THIS task is the sink-side
defence, per the loguru-diagnose lesson (fix at the SINK): an
error path that interpolates an unrecognised caller-supplied string into
a log line must not echo credential-shaped values, because some caller
someday will hand it one again.

## Acceptance Criteria (the what)

- [ ] chat_api_call's unknown/unsupported-endpoint failure path redacts
      credential-shaped values (at minimum: strings matching common key
      prefixes or containing no spaces and exceeding a length bound are
      elided to a short prefix + "…redacted") before logging or raising
      into any user-visible error string
- [ ] A test feeds a key-shaped string as the endpoint and asserts the
      raw value appears in NO log record and NO exception text
- [ ] The redaction is at the sink (chat_api_call's own failure path),
      not in the reranker caller — TASK-17065 owns the caller
