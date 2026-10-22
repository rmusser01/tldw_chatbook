---
id: TASK-35713
title: Console P0 changed-source replay test keys full syncs by a reused frame address
status: To Do
created_date: 2026-10-20 09:30
labels:
- follow-up
- tests
- flaky
- console
- ci
priority: medium
references:
- Tests/UI/test_console_preparing_current_sources.py
- Tests/UI/test_console_poll_reconciliation.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while checking PR #3169's CI (2026-10-20). `Tests/UI/test_console_preparing_current_sources.py::test_preparing_changed_source_after_original_tab_await_replays_full` runs in the required check's Console P0 regression gate, in the admission-sensitive step. It fails intermittently on plain dev at its second wait (line 243 at dev 0fef489ece): 5 of 25 local single-node runs on dev failed (4 at line 243, 1 at line 258), 8 of 25 at PR #3169's head 509f389f94 (all at 243), and 2 of 10 in a Linux python:3.12.15 container at that head.

The cause of the line-243 failures is in the test. It records each returning full sync as `(id(frame), narrowed, value)` and later requires one entry whose identity differs from the held sync's `id(full)`. CPython reuses a freed frame's address, so the later successful full sync can carry the held one's identity, and the condition never holds. Seen: `[(A, True, False), (B, False, False), (A, False, False), (A, False, True), (A, False, True), ...]`.

Separately, in CI on #3169's head 509f389f94 (run 39382891881, attempts 1 and 2), the test failed at its first wait (line 231, `assert await _until(lambda: bool(held), 5)`): the preparing poll's full sync never yielded on the held surface lock within 5 s. That mode did not reproduce locally (macOS or Linux, single node, whole step, full job sequence). It was investigated on #3169, with the result below.

CI evidence (2026-10-20). Two throwaway branches each ran only the P0 gate job, then this node 10 more times; both branches were deleted afterwards. Dev 0fef489ece (run 39405011350): the admission step passed; the repeats failed 3 of 10, 2 at line 243 and 1 at line 231. So line 231's mode is dev's own, not #3169's. #3169's head 509f389f94 (run 39405008429): the admission step passed; the repeats failed 1 of 10, at line 243. #3169's PR run 39382891881 passed on attempt 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test identifies each full sync by something that outlives its frame (not `id(frame)`), and 25 consecutive single-node runs on dev pass
- [ ] #2 The line-231 mode (the held sync never yields within 5 s; seen on dev in CI, 1 of 10) is root-caused and fixed
<!-- AC:END -->


## Implementation Notes

Partial repair in PR #3189 (2026-10-21): run 39566575303, job 119110873891
reproduced the original line-243 frame-address collision on head 8952057f6d.
The test now identifies live FULL invocations by fresh PY_START tokens retained
past PY_RETURN; no original frame is retained and the replay/ownership/custody
assertions and deadlines remain. An exact-observer probe using real
sys.monitoring/asyncio primitives reproduced one raw frame address for 26
returns: the base predicate rejects the valid replay, while the repair's 26
tokens distinguish it. These are observer completions, not 25 full dev tests.

Both acceptance criteria remain unchecked. The original five-case Windows
file failed earlier at the preparation reader-entry wait (246.53s total),
and the untouched replay source failed the same earlier wait (57.40s).
The separately filed original line-231 mode is not fixed or claimed here;
25 consecutive complete single-node dev runs have not been established.

Follow-up in PR #3202 / TASK-32755.2 (2026-10-22 UTC): queuing an unchanged
original FULL immediately after the sampled idle predicate reproduced the
first held-wait signature. That FULL acquired the real surface lock with no
Preparing record; the natural timer then declined under the existing
sync-in-progress guard. Setup now witnesses a successful original narrowed
timer completion before taking the lock. It shares the original five-second
setup budget and retains the source-change, replay and physical-custody
assertions. A retained queued-FULL variant and the natural variant passed
together (70.75s). The older hosted logs did not retain those scheduling
rows, so this establishes the setup hazard rather than every historical
first-wait cause. The 25-run qualification remains outstanding; the task's
acceptance checklist is unchanged.

The hosted Console P0 gate on ea48677119 subsequently passed both variants
in run 39704168922, job 119580155764 (120 + 238 cases across its two steps).
This remains separate from the outstanding 25-run qualification.
