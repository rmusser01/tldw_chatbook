---
id: TASK-2116
title: Anthropic debug log missing its f prefix logs template text instead of values
status: Done
assignee:
  - '@claude'
created_date: '2026-08-03 14:20'
updated_date: '2026-08-03 22:01'
labels:
  - llm-calls
  - anthropic
  - observability
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A debug log statement in `chat_with_anthropic` is missing its `f` prefix, so it emits
the literal template text (`{...}` braces and all) instead of interpolating the payload
values it was written to show.

This is trivial to fix and easy to dismiss, but it has already cost real time: during
the cost-ticker real-provider verification (2026-08-03) it was the log a verifier
reached for while diagnosing why early turns were not caching. It showed template text,
which sent the investigation sideways before the actual cause was found (the turns were
below Anthropic's per-model cacheable minimum — see the cost-ticker spec's amended
Reference facts). A diagnostic that lies costs more than one that is absent.

Pre-existing; not introduced by the cost-ticker PRs.

While fixing, sweep the module for the same class of bug — a missing-`f` logging call is
invisible until someone reads that exact line's output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The `chat_with_anthropic` debug log interpolates its values instead of emitting literal template text
- [x] #2 `tldw_chatbook/LLM_Calls/` swept for other logging calls containing `{}` placeholders without an `f` prefix; each hit fixed or explicitly justified
- [x] #3 No log statement introduced or modified by this task emits secret material (api keys, tokens, full payload bodies)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fix the missing f-prefix on chat_with_anthropic's request-payload debug log so it interpolates real values.
2. Sweep tldw_chatbook/LLM_Calls/ (AST-based, catching logger.<method>(...) and logger.opt(...).<method>(...) calls whose first arg is a plain string literal containing {...}) for the same class of bug.
3. Fix every hit found, preserving each log's existing field-exclusion scope (never widen what gets logged).
4. Verify none of the fixed/touched log lines can emit API keys, tokens, or full message/conversation content -- headers (which carry credentials) are built separately from the payload dicts these logs print, and each log already excludes the messages/contents key.
5. Add a permanent regression test encoding the sweep (so the same copy-paste bug can't return unnoticed) plus a runtime test proving chat_with_anthropic's specific log now shows real values.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: fixed the reported chat_with_anthropic line first (added the missing f-prefix), then wrote an AST-based sweep (catches logger.<method>(...) and logger.opt(...).<method>(...) calls whose first positional argument is a plain ast.Constant string -- i.e. genuinely NOT an f-string, which parses as ast.JoinedStr -- containing a '{' and '}') across every module under tldw_chatbook/LLM_Calls/. Validated the sweep against the pre-fix source via `git show HEAD:...` before trusting it: it caught exactly 9 real hits and 0 false positives, and 0/0 after fixing.

Findings: the exact same copy-pasted bug existed in EVERY other cloud provider's "Request Payload (excluding messages/contents)" debug log in LLM_API_Calls.py -- OpenAI, DeepSeek, Google Gemini (excludes "contents", the Gemini-native key name), Groq, Mistral, OpenRouter, Moonshot, and Z.AI, all missing their f-prefix identically to Anthropic. No other logging call in the LLM_Calls package (huggingface_api.py, LLM_API_Calls_Local.py, Local_Summarization_Lib.py, pricing_catalog.py, Summarization_General_Lib.py) had this shape.

CORRECTION caught by the mandatory full-suite gate, not by design: my first pass just added the f-prefix to all 9 lines and declared AC#3 satisfied because each log already excluded its provider's `messages`/`contents` key. That reasoning was wrong in practice -- these logs also include `system`/`systemInstruction`-shaped fields, and because the bug meant they had NEVER actually fired with real content, that exposure had never been exercised. Running the full `Tests/Chat/` sweep (the task's own gate 3) surfaced two real failures in `Tests/Chat/test_sensitive_llm_logging.py` (`test_sensitive_anthropic_error_body_and_exception_are_not_exposed`, `test_sensitive_google_request_content_and_error_body_are_not_logged`): a canary planted in `system_prompt` now leaked into the debug log precisely because the fix made it fire for the first time. Fixing the f-prefix bug had turned a silent no-op into a real secret-material leak under the app's own `sensitive_llm_request()` policy (`tldw_chatbook/Utils/sensitive_llm_logging.py`) -- exactly the class of regression AC#3 exists to catch.

Fix: found `chat_with_huggingface`'s own "Final Payload" debug log already had the correct pattern (`is_sensitive_llm_request()`-gated, logging only safe metadata when sensitive, the real payload otherwise) -- it was never one of the 9 broken hits because it was already both f-prefixed and gated. Applied the same, minimal gate to all 9 fixed logs: `if not is_sensitive_llm_request(): logger.debug(...)`, so the real payload only logs OUTSIDE a sensitive/auxiliary request. `sensitive_llm_request()` is entered ONLY on the auxiliary/one-shot completion path (`console_provider_gateway.py`'s `_run_auxiliary_completion`/`_complete_sensitive_sync`) -- the primary Console send path this task's motivating incident actually used never sets it, so the diagnostic value that prompted this task is fully preserved there; the log is simply skipped (not replaced with a redacted variant) for the auxiliary path, the smallest change consistent with "no secret material," rather than inventing a new safe-metadata branch for 8 more providers, which would have gone beyond this task's scope.

Verified with the FULL required gate sequence in the foreground: `Tests/Chat/test_anthropic_native_tools.py`+3 siblings (49 passed), the cost-chip trio (53 passed), and the full `Tests/Chat/ Tests/LLM_Calls/` sweep -- 3252 passed, 0 failed, 66 skipped (env-gated only) -- including `test_sensitive_llm_logging.py` now fully green (53/53).

Tests added (new file `Tests/LLM_Calls/test_debug_log_fstring_hygiene.py`): a permanent AST-based regression sweep (`test_no_logging_call_has_an_unevaluated_brace_placeholder`) that fails if the same missing-f-prefix pattern is ever reintroduced anywhere under `tldw_chatbook/LLM_Calls/`, and a runtime test (`test_anthropic_debug_log_interpolates_payload_values`) that drives a real (non-sensitive) `chat_with_anthropic` call and asserts the emitted debug line shows real values, never the old template text, and never a planted "secret" user-message string. The pre-existing `Tests/Chat/test_sensitive_llm_logging.py` suite is the test that actually caught and pins the sensitive-context regression and its fix -- no new test duplicates it.

Files touched: tldw_chatbook/LLM_Calls/LLM_API_Calls.py (9 log-statement fixes, each now `is_sensitive_llm_request()`-gated); Tests/LLM_Calls/test_debug_log_fstring_hygiene.py (new); backlog/tasks/task-2116.
<!-- SECTION:NOTES:END -->
