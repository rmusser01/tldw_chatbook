---
id: TASK-709
title: >-
  Pin the word bench capture client's canary='unchecked' contract and clear small polish items
status: To Do
assignee: []
created_date: '2026-07-26 14:30'
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
- `_cell_from_payload` has no docstring.
- A storage test still uses a literal `dataset_id="d1"`; inert today, but a live FK failure if TASK-705 makes the edit path persist it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A test asserts `WordBenchCaptureClient.capture()` returns `canary='unchecked'`
- [ ] 4xx responses preflight as Blocked rather than Unavailable
- [ ] The capture client reuses a single `httpx.AsyncClient`
- [ ] The remaining polish items are either fixed or explicitly closed as won't-do
<!-- AC:END -->
