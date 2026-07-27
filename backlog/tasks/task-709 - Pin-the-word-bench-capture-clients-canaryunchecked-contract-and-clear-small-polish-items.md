---
id: TASK-709
title: >-
  Pin the word bench capture client's canary='unchecked' contract and clear
  small polish items
status: Done
assignee:
  - '@claude'
created_date: '2026-07-26 14:30'
updated_date: '2026-07-27 05:54'
labels:
  - evals
  - word-bench
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of PR 2 of the Evals rebuild (the word bench engine). Not a defect introduced by that PR unless stated; each is a seam the engine leaves for the screen that consumes it.

**The contract that two tests depend on is unpinned.** `WordBenchCaptureClient.capture()` always returns `canary="unchecked"`; the runner's `_stamp_canary` is what converts it to the target's real verdict. Two runner tests depend on that, and PR 2 fixed a `FakeClient` that had drifted from it — but the contract itself is asserted only in a code comment. If `capture()` ever computed a canary, the fake would drift back and both stamp tests would go green for the wrong reason. One assertion in the existing raw-capture test closes it permanently.

Polish items deferred from the same review:

- `capture_client` maps every `http_error` to `unreachable` / **Unavailable**. A 4xx (e.g. "logprobs not supported") is reachable-but-rejected and should read **Blocked**. Split on 4xx vs 5xx.
- `mode_unsupported` exists in `_STATUS_LABELS` and is produced nowhere; the spec's "raw mode unsupported by endpoint" row has no implementation and currently arrives as a 4xx.
- `capture_client` builds a fresh `httpx.AsyncClient` per request — no keep-alive across a 100+ cell grid. Hold one on the instance.
- Preflight sends the canary through the target's steering prefix, so a legitimate prefix warns `degenerate` on that column. Defensible, but needs a docstring sentence so PR 3's callout does not over-claim.
- `test_canary_expectation_is_a_widely_agreed_continuation` restates a constant and cannot fail meaningfully.
- A storage test still uses a literal `dataset_id="d1"`. TASK-705 resolved by making the dataset immutable on edit, so this stays inert — but it still violates the "no literal ids" rule the fixtures established.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test asserts `WordBenchCaptureClient.capture()` returns `canary='unchecked'`
- [x] #2 4xx responses preflight as Blocked rather than Unavailable
- [x] #3 The capture client reuses a single `httpx.AsyncClient`
- [x] #4 The remaining polish items are either fixed or explicitly closed as won't-do
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the canary=unchecked contract with an explicit assertion in the existing raw-capture test (found a pre-existing assertion already covering this from an earlier commit; verify and extend if needed).
2. Split capture_client.preflight's http_error handling on status code: 4xx -> Blocked (no_logprobs, or mode_unsupported for a 404 specifically), 5xx -> Unavailable (unreachable).
3. Decide mode_unsupported: wire a real producer (404 on the fixed, mode-selected request path) rather than removing it, since the UI inspector already has bespoke copy for it.
4. Hold one httpx.AsyncClient per WordBenchCaptureClient instance (lazy init, aclose()/async-context-manager support) instead of one per request; wire cleanup into WordBenchRunner.run via a duck-typed aclose() call in a finally block so it interacts safely with TASK-707's concurrency.
5. Add the docstring sentence on preflight() about the canary going through the steering prefix.
6. Delete the trivial constant-restating test; fix the literal dataset_id="d1" in test_storage.py.
7. Revert-check the 4xx/5xx split against the pre-fix mapping.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Canary contract**: found the raw-capture test already asserted `result.canary == "unchecked"` (landed in an earlier commit on this branch, "fix(evals): word bench canary logging and test coverage gaps"), so AC #1 was already satisfied by the existing suite. Added a second, independent test (`test_raw_capture_pins_the_unchecked_canary_contract_for_all_call_shapes`) that calls `capture()` with the CANARY_PROMPT itself and a RAW payload whose top token (" a") would trip the degenerate-canary check if `capture()` ever computed a verdict, so a future regression that starts computing one is caught by a distinct assertion, not just the pre-existing one.

**4xx vs 5xx**: `capture_client._preflight_state_for_error` now branches on the parsed HTTP status code carried in `CellError.detail`: a 404 maps to `"mode_unsupported"`, any other 4xx to `"no_logprobs"` (both render "Blocked"), and 5xx / non-HTTP failures stay `"unreachable"` ("Unavailable"). `CellError.reason` itself is untouched ("http_error" for all statuses) so the one pre-existing test asserting that string still holds.

**mode_unsupported decision -- WIRED, not removed**: `capture_client._build_request` always posts to a FIXED, mode-selected path (`/v1/completions` or `/v1/chat/completions`), so a 404 reliably means that route does not exist on this server -- unlike guessing an unobserved JSON response *shape* (which `normalizer.py`'s "shapes are never inferred" rule forbids), interpreting a standard HTTP status by its own defined meaning is not a guess. Also found `tldw_chatbook/UI/Evals/inspector.py` already has bespoke recovery copy for this exact state ("Switch the bench to the other prompt mode..."), confirming it is worth producing rather than deleting.

**Shared AsyncClient**: `WordBenchCaptureClient` now holds one `httpx.AsyncClient` per instance, built lazily on first use (safe under concurrent coroutines -- no `await` between the `None` check and the assignment). Added `aclose()` (idempotent) and `__aenter__`/`__aexit__`. `WordBenchRunner.run` now wraps its whole body in `try/finally`, closing every client it created via a duck-typed `aclose()` call (fakes without one are left alone) -- this runs strictly after the entire run (including any concurrent in-flight row) has finished, so it can never race a live request, and fires on every exit path: success, cooperative cancel, or a hard `asyncio.CancelledError`.

**Docstring**: added a paragraph to `WordBenchCaptureClient.preflight` explaining the canary is sent through the target's own steering (prefix/system_prompt), so a legitimate prefix can itself warn `degenerate` -- a real, honest signal about this bench's configuration, not a target defect.

**Polish**: deleted `test_canary_expectation_is_a_widely_agreed_continuation` (asserted only that two module constants contain expected substrings of themselves; the real behavior is already exercised by the preflight pass/degenerate tests). Fixed the literal `dataset_id="d1"` in `test_storage.py` to reuse the fixture's real dataset id, with a comment explaining why its value is inert either way (save_bench's edit path never passes dataset_id through, per its own docstring) but the project's "no literal ids" fixture convention still applies.

**Revert-check performed** (4xx/5xx split, not formally required by the task but done for parity with 707/708): reverted `_preflight_state_for_error` to the original `"http_error" -> "unreachable"` unconditional mapping; `test_preflight_reports_a_4xx_as_blocked_not_unavailable` and `test_preflight_reports_a_404_specifically_as_mode_unsupported` failed as expected (both asserted `"unreachable"` instead of the intended state); the other 22 tests in that file were unaffected. Restored and confirmed all 24 pass.

**Files**: `tldw_chatbook/Evals/word_bench/capture_client.py`, `tldw_chatbook/Evals/word_bench/models.py`, `tldw_chatbook/Evals/word_bench/runner.py`, `Tests/Evals/word_bench/test_capture_client.py`, `Tests/Evals/word_bench/test_runner.py`, `Tests/Evals/word_bench/test_storage.py`.
<!-- SECTION:NOTES:END -->
