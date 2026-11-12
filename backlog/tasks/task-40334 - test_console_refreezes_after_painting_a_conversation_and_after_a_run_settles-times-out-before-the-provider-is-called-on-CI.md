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
Found on 2026-11-12 while #3491 waited in the merge queue. Checked with read-only `gh` and the code at dev 4b90eb860b. Line numbers are that tree's.

**Why.** The test runs in UI Fast Lane shard 2, which feeds the required check. The merge queue retries a failed required check once and evicts the PR on the second failure (`scripts/merge_queue.py:224-226`), so this flake can evict an unrelated PR. Of the 21 shard-2 PR jobs that finished between 2026-11-12 05:15Z and 20:12Z (re-runs counted once), this test failed in 6, on 5 branches. Each time it was the shard's only failure. The times below are when each job finished:
- run 43165198830 (codex/send-feedback-dev-regressions, job 131012505672, 08:26Z);
- run 43172047439 (codex/workspace-wipe-guard, job 131039345585, 08:56Z);
- run 43148114307 (codex/task-39147-native-phase-rebuild, job 131085054099, 10:50Z);
- run 43242842744 (fu/hotfix-restyle, job 131272987686, 15:39Z);
- run 43245073678 attempt 2 (fu/census-nonliteral-lookups, job 131294344060, 17:04Z);
- run 43284372195 attempt 2 (fu/census-nonliteral-lookups, job 131429596853, 20:12Z).

It passed 3 of 3 runs locally on dev e0186fa10a under an isolated profile.

**Where it stops.** Every failure is the same wait, `test_console_idle_heap_refreeze.py:123`: `await until(gateway.validation_started.is_set)`, which times out after 15 s. Before it, the test has seen the Console register the send's flight (line 112) and enter the `HeldAdmission` hold on the hook-permission snapshot (line 115), and it has then released the hold (line 121).

The jobs' captured `console_send_stage` lines narrow it further.
- In 5 of the 6 failures, `controller_submit entered` is logged at 2.2–5.7 s and nothing follows until `provider_resolution entered` at about 15.4–15.7 s. In job 131429596853 that is 3.52 s, then 15.69 s. Job 131272987686 logs `controller_submit failed` at 15.64 s instead.
- So in those five the send stalls inside the controller's submit: after `controller_submit` is recorded (`console_chat_controller.py:3560`) and before `provider_resolution` (`:9285`).
- Job 131085054099 stops earlier: its `ui_submit entered` only comes at 15.17 s.

It is not yet known what the controller waits on. The test harness's hold thread is ruled out for those five. If the wait is in the product, it is user-visible: Send accepted, then no provider call for many seconds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 What the controller's submit waits on between `controller_submit` and `provider_resolution` is identified and recorded with evidence (a CI log or a local reproduction under load), and so is the earlier stall before `ui_submit` in job 131085054099.
- [ ] #2 If the wait is in the send path, a send released from the hook-permission snapshot reaches provider resolution without waiting on unrelated work, and a test pins that.
- [ ] #3 The test passes 30 consecutive runs of its shard-2 file set under xdist with a concurrent CPU stressor, and the next 10 CI shard-2 runs after the fix are green on it.
- [ ] #4 The test still fails if the Console re-freezes its heap during a send: its busy assertions are not weakened, skipped or retried away.
<!-- AC:END -->
