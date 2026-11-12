---
id: TASK-40334
title: "test_console_refreezes_after_painting_a_conversation_and_after_a_run_settles times out before the provider is called on CI"
status: To Do
created_date: 2026-11-12 21:05
labels:
- follow-up
- tests
- console
- flaky
- ci
priority: medium
references:
- Tests/UI/test_console_idle_heap_refreeze.py
- Tests/UI/test_console_send_acknowledgement.py
- https://github.com/rmusser01/tldw_chatbook/pull/3491
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found on 2026-11-12 while #3491 waited in the merge queue. Checked with read-only `gh` and the test at dev 4b90eb860b. Line numbers are that tree's.

**Why.** The test runs in UI Fast Lane shard 2, which feeds the required check. The merge queue retries a failed required check once and evicts the PR on the second failure, so this flake can evict an unrelated PR. Shard 2 finished in 18 PR runs between 2026-11-12 05:15Z and 20:12Z, and this test failed in 4 of them, on 3 branches, each time the shard's only failure (`1 failed`, with 108 passing, or 133 in the 06:01Z run):
- run 43165198830 (codex/send-feedback-dev-regressions, 06:01Z, job 131012505672);
- run 43242842744 (fu/hotfix-restyle, 14:58Z, job 131272987686);
- run 43245073678 (fu/census-nonliteral-lookups, 15:11Z, job 131294344060);
- run 43284372195 attempt 2 (fu/census-nonliteral-lookups, 20:00Z, job 131429596853).

It passed 3 of 3 runs locally on dev e0186fa10a under an isolated profile.

**Where it stops.** Every failure is the same wait, `test_console_idle_heap_refreeze.py:123`: `await until(gateway.validation_started.is_set)`, which times out after 15 s. The test has released its `HeldAdmission` hold (line 121), but the send never reaches the provider gateway's validation. The earlier waits in the same test pass: the admission hold is entered, and the Console marks the send as busy. The stall therefore sits between the released admission hold and the provider call.

It is not yet known whether this is a harness race (for example, the hold's `release.wait` thread starved on a loaded runner) or a real intermittent stall in the send path after admission. A real stall would be user-visible: Send accepted, but no provider call.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the stall between the released admission hold and provider validation is identified and recorded, with evidence (a CI log or a local reproduction under load).
- [ ] #2 If the cause is in the send path, a send whose admission completes reaches the provider without waiting on unrelated work, and a test pins that.
- [ ] #3 The test passes 30 consecutive runs under load comparable to CI's UI Fast Lane shard.
- [ ] #4 The test still fails if the Console re-freezes its heap during a send: its busy assertions are not weakened, skipped or retried away.
<!-- AC:END -->
