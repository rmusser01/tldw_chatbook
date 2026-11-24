---
id: TASK-42529
title: "A streamed OpenAI 429 reads 'unexpected provider error' instead of a rate limit"
status: To Do
assignee: []
created_date: 2026-11-24 09:14
updated_date: 2026-11-24 09:14
labels:
- follow-up
- console
- providers
- rate-limit
priority: medium
references:
- tldw_chatbook/LLM_Calls/LLM_API_Calls.py
- tldw_chatbook/Chat/console_provider_gateway.py
- Tests/Chat/test_rate_limit_reads_as_rate_limit.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-42472's round-24 check (2026-11-24) on fu/r24-diag (dev af4b13ea75 plus round 24's stacked branches). Paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** The Console streams by default (TASK-33620.10). TASK-34100.2 AC#18 says a provider rate limit reads in Console as a rate limit, with the provider's own reason where the handler reads the error body. A non-streamed OpenAI 429 does: "Provider error from OpenAI: rate limit exceeded. Status: 429. OpenAI says: “…”". A streamed one reads "Provider error from OpenAI: unexpected provider error.", with capture on or off. A streamed 503 reads the same, so its status is lost too.

**Cause.** `LLM_Calls/LLM_API_Calls.py::chat_with_openai`'s stream generator calls `response.raise_for_status()` inside a `try` whose `except requests.exceptions.RequestException` catches the `HTTPError`. It then yields an SSE line, `{"error": {"message": "Stream connection error: 429 Client Error ...", "type": "openai_stream_error"}}`, so the status and OpenAI's body never reach the error classifier. This has been on dev since ee46503f9c (2026-06-08). Anthropic's (`anthropic_stream_error`) and Gemini's (`gemini_stream_error`) stream handlers yield the same kind of line; they were not probed.

**Evidence.** Probe `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/3f117384-6f37-40d1-bdd4-c465d56c3f56/scratchpad/fu/r24fx-openai-stream-429-probe.py`. Copy it into `Tests/Chat/` of a worktree and run it through `$FU/run.sh`; the scratch path may have been cleaned. It drives the real controller, gateway and trace factory (`Tests/Chat/test_console_project_instruction_traces.py::_project_console(hosted=True)`) and OpenAI's real handler, with only `requests.Session.post` patched to return a real `requests.Response` (429 or 503, a JSON error body).
- Complete mode: "rate limit exceeded. Status: 429" with OpenAI's reason, and "provider unavailable. Status: 503".
- Stream mode: "Provider error from OpenAI: unexpected provider error." in all 4 cases (429 and 503, capture on and off).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A streamed OpenAI request answered with 429 reads in Console as a rate limit with OpenAI's reason, as a non-streamed one does; a test drives the real handler over a real `requests.Response` and is red before the fix
- [ ] #2 A streamed OpenAI 5xx keeps its status in the Console copy
- [ ] #3 Anthropic's and Gemini's streaming handlers are checked the same way, and each is fixed or its result is recorded
- [ ] #4 The agent's in-place retry still classifies a streamed 429 or 5xx as transient
<!-- AC:END -->
