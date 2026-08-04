---
id: TASK-2119
title: >-
  Provider exception handlers leak API keys into logs via loguru diagnose
status: To Do
assignee: []
created_date: '2026-08-03 18:50'
labels:
  - security
  - llm-calls
  - observability
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**A real API key was disclosed while verifying this. The Moonshot key in the repo root
was leaked in plaintext by an ordinary HTTP 429 during multi-provider verification on
2026-08-03 and must be rotated.**

`tldw_chatbook/LLM_Calls/LLM_API_Calls.py` has ~30 call sites where a provider request's
`RequestException` / generic exception handler calls
`logger.opt(exception=True).error(...)`. Loguru's `diagnose` option defaults to True and
dumps **every stack frame's local variables** alongside the traceback. In these handlers
the locals in scope include the raw request `headers` (carrying
`Authorization: Bearer <key>` / `x-api-key`) and `final_api_key`. So any transient
provider error during normal chat writes the user's key to the log sink in cleartext.

Confirmed affected by grep: OpenAI, Cohere, Moonshot, Z.AI. Live-confirmed for Moonshot
via a genuine 429. Google's and OpenRouter's specific error branches happen not to use
`opt(exception=True)` — coincidence, not design.

**Pre-existing, not introduced by the cost-ticker program:** `git blame` traces the
sampled sites to PR #707 (2026-07-19) and PR #1235 (2026-08-02).

Note this is a *different* surface from the one PR #1295 closed. That work made the
**debug payload logs** allowlist-shaped; these are **exception handlers**, where the
secret arrives via frame locals rather than via a logged payload dict — so no amount of
payload redaction touches it.

**Preferred fix is the class-killer, not 30 patches.** Frame locals should never reach a
persistent sink: set `diagnose=False` on the app's logger sink configuration, which
eliminates the entire category in one place regardless of which handler runs or what a
future contributor adds. Per-call-site `opt(exception=True, diagnose=False)` is the
fallback if a sink-level change is judged too broad, but it leaves the next new handler
exposed by default.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An exception raised inside a provider call with an API key in scope does NOT write the key value to any log sink (file or console), on both the sensitive and non-sensitive request paths
- [ ] #2 The fix is applied at the sink/configuration level so a newly added exception handler is safe by default, not dependent on the author remembering a flag
- [ ] #3 A regression test plants a sentinel key value in scope, forces a provider exception, captures log output, and asserts the sentinel appears nowhere
- [ ] #4 Traceback/diagnostic value is preserved to the extent possible (the exception type, message, and stack are still logged — only frame-local dumping is suppressed)
- [ ] #5 `tldw_chatbook/` swept for other `opt(exception=True)` / `exc_info=True` sites where credentials or full payloads are in scope; each fixed or justified
- [ ] #6 The exposed Moonshot key is rotated by the owner (tracked here for closure; not an agent action)
<!-- AC:END -->
