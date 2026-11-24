---
id: TASK-42515
title: "The provider 400 consistent-status test is red on dev: it looks for the gateway sentence in the content-free diagnostic"
status: Done
assignee:
- '@claude'
created_date: 2026-11-24 16:54
updated_date: 2026-11-24 17:19
labels:
- follow-up
- tests
- dev-red
- ci-lanes
- console
priority: medium
references:
- Tests/Chat/test_console_provider_failure_copy.py
- tldw_chatbook/Chat/console_provider_gateway.py
- tldw_chatbook/Chat/provider_failures.py
- .github/workflows/derived-artifacts.yml
- Tests/CI/test_ci_queue_pressure_contract.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by round 24 of the follow-up burn-down (2026-11-24) and reproduced on dev 9fa2bf1b1d. `Tests/Chat/test_console_provider_failure_copy.py::test_stream_chat_provider_400_reports_one_consistent_status` fails with `AssertionError: assert 'Status: 400.' in 'provider returned HTTP 400 (Invalid request sent to the chat provider)'`. Its other checks pass: the re-raised error carries the real 400, the copy states it once, and there is no 502.

TASK-19642.8.7 (2026-08-21) and TASK-19642.8 already noted this node as a pre-existing red and left it for its owner; neither tracks it, and no PR lane runs the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is recorded, with the commits that made the test red
- [x] #2 The test passes and still checks that the real status (400) survives, is stated once, is never 502, and that the user sees the gateway's "Status: 400." sentence
- [x] #3 The file runs in a PR lane step, timed with that step's flags, and none of its nodes runs in a second step
<!-- AC:END -->

## Implementation Plan

1. Reproduce through the scratch-profile wrapper; trace which error `stream_chat` raises for an adapter `ChatBadRequestError` and what `describe_stream_failure` makes of it.
2. Find the commits that changed the raise and the assertion, and decide product drift or test drift from their stated intent and the file's other pins.
3. If test drift: keep the status assertions on the diagnostic copy and check the gateway sentence where Console shows it (`console_failure_copy`, which the stream failure path uses); show the new check red under a temporary source edit.
4. Lane the file in the Dreams tracking riders step (bootstrap_profile through `Tests/conftest.py`, so after `Tests/Tool_Packs`) and in `DREAMS_TRACKING_RIDERS`; time it serially with the step's flags.
5. Soak the file 20 times at `-n 4` under load; run `Tests/CI`, `Tests/Architecture` serially, preflight and the Ruff gates; run the changed step.

## Implementation Notes

- **Cause: test drift, from two changes that landed a day apart.**
  - c55fc726dff25a600d769816c2846201064b582d (TASK-34100.5 review, on dev via #3025, f99ad86ebd, 2026-10-05) added `"Status: 400." in describe_stream_failure(err)`. It held then: `stream_chat` re-raised `ChatProviderError(item.text)` carrying the gateway's sentence ("Provider error from OpenAI: bad request. Status: 400."), and `describe_stream_failure` returned a sentence containing `Status: N.` whole (dev 6feb84c1d2^1, `provider_failures.py:159`).
  - 818c7c8ea0ba511e2aead89fe541d2247555b217 ("preserve typed provider recovery copy at Console display", PR #2918, rebased onto that dev and merged as 6feb84c1d2203bc3c6d0eecd2823f4e2409b4b4e on 2026-10-06) made `stream_chat` re-raise the typed projection (`project_provider_error`, from 7212692c8b), whose message is the class default, and put the gateway's sentence on `console_copy` ("Sanitized Console presentation is separate from diagnostic str/message"; durable agent summaries keep the diagnostic helper). It changed this test's expected type to `ChatBadRequestError` but left the "Status: 400." check on the diagnostic.
  - That behaviour is intended: the same commit pins the diagnostic as exactly `provider returned HTTP {status} (Invalid request sent to the chat provider)` with no "Provider error from …" in `test_typed_console_presentation_never_enters_durable_agent_error`. No PR lane ran the file.
  - TASK-19642.8.7 recorded this node red on 2026-08-21, under an older assertion (`"HTTP 400" in copy`); that earlier red's cause was not established here.
- **Fix (test).** The diagnostic keeps its checks: status 400, stated once, no 502. "Status: 400." and the one-status and no-502 checks now apply to `console_failure_copy(err, "Provider stream failed")`, the controller's stream-failure seam (`console_chat_controller.py`), which must equal `err.console_copy`.
- **Red/green.** Dev 9fa2bf1b1d: `AssertionError: assert 'Status: 400.' in 'provider returned HTTP 400 (Invalid request sent to the chat provider)'`. After: 18 passed. Negative control: with the gateway's `item.payload.console_copy = item.text` temporarily set to `None`, the test fails at `visible == err.console_copy` (`'Provider stream failed: provider returned HTTP 400 (...)' == None`); reverted.
- **Lane.** Dreams tracking riders step, after `Tests/Tool_Packs` (the whole file is on the bootstrap profile through `Tests/conftest.py`'s `keep_bootstrap_profile` list), and in `DREAMS_TRACKING_RIDERS`. No other PR-lane step runs it (the contract's one-owner check). Serial time with the step's flags: 18 tests, 8.8 s, 1-min load 11.8 → 11.6. Also green in a scratch venv from the lane's install line (pytest 9.2.0).
- **Checks.** As TASK-42514: soak 20/20 at `-n 4` (load 6.9-36.3), `Tests/CI` 1948 passed, `Tests/Architecture` serially 1507 passed, preflight and the Ruff gates passed. The step: `bash $FU/r19-step.sh fu-r25-devreds "Run the Dreams tracking tests and unlaned riders" r25-step -p no:xdist --timeout=180`: 28 targets, 1386 passed, 6 skipped, 0 failed, 15 min 15 s wall (pytest 906.98 s), 1-min load 19.0 at start, 16.1 at end.
- **Files.** `Tests/Chat/test_console_provider_failure_copy.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
