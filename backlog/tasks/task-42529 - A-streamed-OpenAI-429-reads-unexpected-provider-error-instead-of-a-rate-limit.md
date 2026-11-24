---
id: TASK-42529
title: "A streamed OpenAI 429 reads 'unexpected provider error' instead of a rate limit"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 09:14
updated_date: 2026-11-24 22:47
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
- [x] #1 A streamed OpenAI request answered with 429 reads in Console as a rate limit with OpenAI's reason, as a non-streamed one does; a test drives the real handler over a real `requests.Response` and is red before the fix
- [x] #2 A streamed OpenAI 5xx keeps its status in the Console copy
- [x] #3 Anthropic's and Gemini's streaming handlers are checked the same way, and each is fixed or its result is recorded
- [x] #4 The agent's in-place retry still classifies a streamed 429 or 5xx as transient
<!-- AC:END -->

## Implementation Plan

1. Survey every requests-based `chat_with_*` stream path for where it checks the HTTP status: before returning the generator (the status error reaches `chat_api_call`'s classifier) or inside it (swallowed into an SSE error line). Check the hosted engine too.
2. `chat_with_openai`'s generator: let the status check's `HTTPError` leave the generator, its body read (bounded) before `finally` closes the response. Other stream failures keep their SSE error line.
3. `chat_api_call`: move the `HTTPError` classification into one function, and run it also for an `HTTPError` raised while a returned generator is read, so a lazy status check reads as the eager one does.
4. Tests in `Tests/Chat/test_rate_limit_reads_as_rate_limit.py` (already laned): a Console test over the real controller, gateway and handler for a streamed 429 (Capture On and Off) and 503, which also counts the agent's in-place retries; loopback stub-server tests for streamed OpenAI 429/5xx and Anthropic/Gemini 429; a negative case for a stream that fails after it opened. Update `test_openai_streaming_usage.py`'s streamed-400 pin, and lane it.

## Implementation Notes

**Cause.** `chat_with_openai`'s stream generator runs its status check (`raise_for_status`) on the first read, after `chat_api_call` has returned the stream. Its `except requests.exceptions.RequestException` turned that `HTTPError` into an SSE error line, which the gateway's `_raise_for_sse_error` raised as a status-less `RuntimeError`: "Provider error from OpenAI: unexpected provider error." for a streamed 429, 5xx or 400, Capture On or Off. The Responses-API path runs in the same generator, so it shared the cause.

**Fix.** `chat_api_call`'s `HTTPError` branch is now one function, `Chat_Functions.raise_for_http_error(endpoint, error, api_key)`, unchanged in behaviour. OpenAI's generator catches the status check's `HTTPError` ahead of the `RequestException` clause and raises through that same function. The except body reads the body before `finally` closes the response, so the error keeps OpenAI's reason. A streamed 429 is now `ChatRateLimitError` (with Retry-After and OpenAI's reason), a 5xx is `ChatProviderError` with its status, and a 400 is `ChatBadRequestError`. Every other stream failure keeps its SSE error line.

**Deviations from the plan.**
- Plan step 3 (wrap every generator `chat_api_call` returns) was dropped. OpenAI's handler returns `recovery_review._OpenAIStream` through `@openai_call`, not a generator, so the wrapper never ran: every OpenAI test stayed red. Wrapping that stream would also break its cross-thread `close()` on Console Stop.
- Plan step 2's bounded pre-read was dropped. A negative control showed it was redundant over a real socket, because the except body reads the body before `finally` closes the response.
- The covering run showed that the User Guide's stale "OpenAI does not do this yet" note in `console/chat-basics.md` is now wrong. A streamed OpenAI 400 is no longer sent three times as a 502: it is sent once and reads "bad request. Status: 400. OpenAI says: “…” If this server cannot stream replies, turn Streaming off…", which is TASK-37180's cloud-refusal copy. The page is updated, and a Console case pins that copy.

**Providers checked (AC#3).** Each of these checks the status before it returns its stream, so a 429 reaches `chat_api_call`'s classifier, or the handler's own one, with its status and reason:
- Anthropic and Gemini;
- Cohere and Hugging Face;
- the local OpenAI-compatible family (`_chat_with_openai_compatible_local_server`: llama.cpp, vLLM, Ollama, Custom, …);
- the hosted engine (`owned_json_post` raises before it returns `OwnedSSEStream`): Groq, Mistral, Moonshot, OpenRouter, DeepSeek, Z.ai and the presets;
- QwenCloud.

None needed a fix. A streamed 429 is pinned over a loopback stub for Anthropic, Cohere, Hugging Face and Gemini (`ChatRateLimitError`, the provider's reason, one POST). A streamed Hugging Face 503 keeps its status after one POST; its stream session mounts no transport retry, unlike its non-streaming one. That was already so, and nothing pins it.

**Retry contract.** OpenAI's owned session mounts no transport retry. The loopback tests count one POST for a streamed 429 and for a 503 (AC#18). The agent's in-place retry still re-asks a streamed 429 or 503 `max_model_retries` times (AC#4). A streamed 400 is terminal and sent once.

**Evidence (red at base 75bac413f5, green after).**
- `Tests/Chat/test_rate_limit_reads_as_rate_limit.py`:
  - `test_a_streamed_openai_failure_keeps_its_status_and_reason[429 Capture On / 429 Capture Off / 503 / 400]`: the real controller, gateway, trace factory and OpenAI handler, with `requests.Session.post` answering real `requests.Response`s. Red: all four read "unexpected provider error".
  - The loopback stub-server tests with `streaming=True`, for OpenAI's 429 (two tests) and its 503 (which also asserts the 503 copy with OpenAI's reason). Red: `DID NOT RAISE`.
  - The negative case `test_a_stream_that_drops_after_it_opened_reads_as_before` (200, then the connection drops) is green at base and after.
- `Tests/Chat/test_openai_streaming_usage.py::test_400_not_naming_stream_options_is_passed_through_unretried` now expects `ChatBadRequestError(400)` after one POST. Red at base with `DID NOT RAISE`.
- Base run: 8 failed, 41 passed. Final run: 49 passed. Soak: 20 runs, 49 passed each, at `-n 4` and load 8-15.

**Lane.** The changed `test_openai_streaming_usage.py` node reads config, so its local run hit RecoveryRequired. It is marked `bootstrap_profile`, as its neighbour is, and laned in "Run admission-sensitive suites 3" after the rate-limit file (`ADMISSION_3_TARGETS` too). The recorder counted 0 RecoveryRequired over the rate-limit file plus the node. New Console cases, serial with the lane's flags at load 11-13: 3.8-5.7 s each, about 19 s for the four. The rate-limit file plus the node took 42 s serially.

**Runs.**
- Lane step "Run admission-sensitive suites 3" (serial, lane flags): 458 passed in 1000 s at load 9-15.
- `Tests/CI` at `-n 4`: 1946 passed, 2 skipped.
- `Tests/Architecture` serially: 1507 passed, 5 skipped.
- `./scripts/preflight.sh`, `ruff check tldw_chatbook --select F` and `Tests/lint/ruff_gate.py`: all pass. The gate reports B904 down 5, all from the moved raises; its baseline is left as is.

**Other.**
- `LLM_API_Calls.py` size row: 5,232 -> 5,239 (+7, dated).
- Diagnostic inventory: the moved `logger.error` in `Chat_Functions.py` is only re-indented (`--statements`: moved 1, added 0, removed 0), so the inventory is regenerated.
- 14 covering-file nodes are red on the RecoveryRequired trap with the same command at base: `test_chat_functions.py` ×7, `test_openai_streaming_usage.py` ×4, `test_console_provider_gateway.py` ×3. The two OpenAI stop tests also fail identically at base and after under a temporary `bootstrap_profile` mark ("close called 2 times").

**Review fixes (three Minors).**
1. An unreadable error body lost the status again. A streamed reply's body is first read in `raise_for_http_error`. A connection that drops after the headers made `.json()` or `.text` raise `ChunkedEncodingError`, which left with no status. Both reads now treat a `RequestException` as no body, and the status classification still applies.
   - Pinned by `test_a_streamed_openai_status_whose_body_never_arrives_keeps_its_status[429, 503]`: a real `requests.Response` whose raw stream raises urllib3's `ProtocolError`, through `chat_api_call(streaming=True)`. Expected: `ChatRateLimitError`, or `ChatProviderError` with status 503, and the status-only copy.
   - Red with the first commit's classifier (`ChunkedEncodingError` escapes); green after.
2. The User Guide now says the provider's own reason "where it sent one".
3. The admission-sensitive 3 step's budget comment records the measured cost. All timings are serial, at load 9-10:
   - five mounted Console cases: 19 s;
   - seven streamed loopback parameters: 3.7 s (the review counted six; the collection has seven);
   - two unit cases: 0.04 s;
   - the rate-limit file plus the node: 51 tests, 42 s.

**Runs after the review fixes.**
- Soak: 5 runs, 51 passed each, at `-n 4`, load 10-12.
- `Tests/CI`: 1946 passed, 2 skipped.
- Preflight, `ruff check tldw_chatbook --select F` and `ruff_gate`: all pass.
- No new Ruff findings in the touched files.

**Files.**
- `tldw_chatbook/Chat/Chat_Functions.py`
- `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`
- `Tests/Chat/test_rate_limit_reads_as_rate_limit.py`
- `Tests/Chat/test_openai_streaming_usage.py`
- `Tests/CI/test_ci_queue_pressure_contract.py`
- `Tests/Architecture/test_module_size_ratchet.py`
- `.github/workflows/derived-artifacts.yml`
- `Docs/User_Guide/console/chat-basics.md`
- `Docs/security/production-diagnostic-inventory.json`
- `backlog/docs/lessons-console-wiring.md`
